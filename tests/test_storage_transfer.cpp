// Tests for src/storage.h and src/transfer.h, ending with the units wired
// together and driven by hand through Tpu::tick().

#include "test_support.h"
SECTION("ub") {
    Config cfg = small_cfg();
    cfg.ub_bytes = 256;
    cfg.ub_banks = 4;

    UnifiedBuffer ub(cfg);
    REQUIRE(ub.bytes() == 256);
    REQUIRE(ub.banks() == 4);

    // Consecutive bytes land in consecutive banks, spreading a row over ports.
    REQUIRE(ub.bank_of(0) == 0);
    REQUIRE(ub.bank_of(1) == 1);
    REQUIRE(ub.bank_of(3) == 3);
    REQUIRE(ub.bank_of(4) == 0);
    REQUIRE(ub.bank_of(37) == 1);

    // A tile round-trips byte-identically, strides and all.
    I8Tensor src(3, 5);
    Lcg rng(7);
    for (uint32_t r = 0; r < 3; ++r) {
        for (uint32_t c = 0; c < 5; ++c) src.view().at(r, c) = rng.byte();
    }
    REQUIRE(ub.write_tile(16, 8, src.view()));      // pitch 8, wider than the tile

    I8Tensor back(3, 5);
    REQUIRE(ub.read_tile(16, 8, back.view()));
    REQUIRE(same_values(src.view(), back.view()));

    // The gap between rows is untouched, which is what a stride means.
    REQUIRE(ub.at(16 + 5) == 0);
    REQUIRE(ub.at(16 + 7) == 0);

    // A strided view aliases the buffer instead of copying it.
    ub.view(16, 3, 5, 8).at(1, 1) = 99;
    I8Tensor after(3, 5);
    REQUIRE(ub.read_tile(16, 8, after.view()));
    REQUIRE(after.view().at(1, 1) == 99);

    // A tile ending flush against the top of the buffer fits.
    REQUIRE(ub.tile_fits(0, 4, 4, 4));
    REQUIRE(ub.tile_fits(252, 1, 4, 4));
    // 2 rows at pitch 8 span 8 + 4 = 12 bytes, not 16, so this ends at 256.
    REQUIRE(ub.tile_fits(244, 2, 4, 8));
    REQUIRE(!ub.tile_fits(245, 2, 4, 8));
    REQUIRE(!ub.tile_fits(248, 2, 4, 8));
    REQUIRE(!ub.tile_fits(0, 1, 5, 4));             // cols wider than the stride
    REQUIRE(!ub.tile_fits(250, 4, 4, 4));

    // An out-of-range tile is refused rather than trapping or writing wild.
    I8Tensor big(9, 9);
    big.view().fill(1);
    REQUIRE(!ub.write_tile(250, 9, big.view()));
    REQUIRE(!ub.read_tile(250, 9, big.view()));

    // Two reads to one bank collide; two reads to different banks do not.
    using Port   = UnifiedBuffer::Port;
    using Access = UnifiedBuffer::Access;

    REQUIRE(ub.port_conflict(std::vector<UbAddr>{0, 4}));         // both bank 0
    REQUIRE(!ub.port_conflict(std::vector<UbAddr>{0, 1, 2, 3}));  // one each
    REQUIRE(!ub.port_conflict(std::vector<UbAddr>{}));
    REQUIRE(!ub.port_conflict(std::vector<UbAddr>{7}));
    REQUIRE(ub.port_conflict(std::vector<UbAddr>{0, 1, 2, 3, 8}));

    // A read and a write to one bank proceed together, one port of each, so the
    // buffer can feed the array and take activation output in the same cycle.
    REQUIRE(!ub.port_conflict(std::vector<Access>{{0, Port::READ}, {4, Port::WRITE}}));
    REQUIRE(ub.port_conflict(std::vector<Access>{{0, Port::WRITE}, {4, Port::WRITE}}));
    REQUIRE(!ub.port_conflict(std::vector<Access>{{0, Port::READ}, {1, Port::WRITE}}));

    // A row read is conflict-free exactly when it spans no more banks than exist.
    {
        std::vector<UbAddr> row;
        for (uint32_t i = 0; i < 4; ++i) row.push_back(i);
        REQUIRE(!ub.port_conflict(row));
        row.push_back(4);                        // a fifth byte wraps to bank 0
        REQUIRE(ub.port_conflict(row));
    }

    // Conflicts are counted by whoever stalls, so the query itself stays pure.
    REQUIRE(ub.stats().bank_conflicts == 0);
    ub.note_bank_conflict();
    REQUIRE(ub.stats().bank_conflicts == 1);

    REQUIRE(ub.stats().tile_writes == 1);
    REQUIRE(ub.stats().bytes_written == 15);
}

SECTION("accumulators") {
    Config cfg = small_cfg(4, 3);

    Accumulators acc(cfg);
    REQUIRE(acc.banks() == 3);
    REQUIRE(acc.dim() == 4);
    REQUIRE(acc.valid(2));
    REQUIRE(!acc.valid(3));

    // Banks start zeroed and are independent.
    I32Tensor got(4, 4);
    REQUIRE(acc.read(0, got.view()));
    bool all_zero = true;
    for (uint32_t r = 0; r < 4; ++r) {
        for (uint32_t c = 0; c < 4; ++c) {
            if (got.view().at(r, c) != 0) all_zero = false;
        }
    }
    REQUIRE(all_zero);

    I32Tensor ones(4, 4);
    ones.view().fill(1);
    REQUIRE(acc.write(0, ones.view()));
    REQUIRE(acc.read(0, got.view()));
    REQUIRE(got.view().at(3, 3) == 1);
    REQUIRE(acc.bank(1).at(3, 3) == 0);         // bank 1 unaffected

    // Accumulate adds in place; write replaces.
    REQUIRE(acc.accumulate(0, ones.view()));
    REQUIRE(acc.accumulate(0, ones.view()));
    REQUIRE(acc.read(0, got.view()));
    REQUIRE(got.view().at(0, 0) == 3);
    REQUIRE(acc.write(0, ones.view()));
    REQUIRE(acc.read(0, got.view()));
    REQUIRE(got.view().at(0, 0) == 1);

    acc.zero(0);
    REQUIRE(acc.bank(0).at(0, 0) == 0);

    // A partial tile touches only its corner.
    I32Tensor small(2, 2);
    small.view().fill(5);
    acc.zero(0);
    REQUIRE(acc.write(0, small.view()));
    REQUIRE(acc.bank(0).at(1, 1) == 5);
    REQUIRE(acc.bank(0).at(2, 2) == 0);

    // A locked bank refuses access and counts each refusal as a hazard.
    const uint64_t before = acc.stats().hazards;
    acc.lock(1);
    REQUIRE(acc.locked(1));
    REQUIRE(!acc.read(1, got.view()));
    REQUIRE(!acc.accumulate(1, ones.view()));
    REQUIRE(!acc.write(1, ones.view()));
    REQUIRE(acc.stats().hazards == before + 3);

    acc.unlock(1);
    REQUIRE(!acc.locked(1));
    REQUIRE(acc.read(1, got.view()));

    // Locking one bank leaves the others alone.
    acc.lock(0);
    REQUIRE(!acc.read(0, got.view()));
    REQUIRE(acc.read(2, got.view()));
    acc.unlock(0);

    // An invalid bank is a malformed instruction, not a hazard.
    const uint64_t h = acc.stats().hazards;
    REQUIRE(!acc.read(9, got.view()));
    REQUIRE(acc.stats().hazards == h);

    // K four times the array width, split into four tiles accumulating into one
    // bank, against a direct triple loop over the whole of K.
    {
        const uint32_t dim = 4;
        const uint32_t tiles = 4;
        const uint32_t K = dim * tiles;
        const uint32_t len = 3;

        Config c = small_cfg(dim, 2);
        c.double_buffer = false;

        Lcg rng2(4242);
        I8Tensor a(len, K), w(K, dim);
        for (uint32_t r = 0; r < len; ++r) {
            for (uint32_t k = 0; k < K; ++k) a.view().at(r, k) = rng2.byte();
        }
        for (uint32_t k = 0; k < K; ++k) {
            for (uint32_t col = 0; col < dim; ++col) w.view().at(k, col) = rng2.byte();
        }

        // The independent oracle: one loop nest over the full K.
        std::vector<int> want;
        for (uint32_t r = 0; r < len; ++r) {
            for (uint32_t col = 0; col < dim; ++col) {
                int64_t sum = 0;
                for (uint32_t k = 0; k < K; ++k) {
                    sum += static_cast<int64_t>(a.view().at(r, k)) * w.view().at(k, col);
                }
                want.push_back(static_cast<int>(sum));
            }
        }

        // Four partial matmuls accumulating into one bank.
        Accumulators banks(c);
        Mxu mxu(c);
        I32Tensor partial(len, dim);

        for (uint32_t t = 0; t < tiles; ++t) {
            mxu.load_weights(w.view().tile(t * dim, 0, dim, dim));
            mxu.matmul(a.view().tile(0, t * dim, len, dim), partial.view(), false);
            if (t == 0) {
                REQUIRE(banks.write(0, partial.view()));
            } else {
                REQUIRE(banks.accumulate(0, partial.view()));
            }
        }

        std::vector<int> got_k;
        for (uint32_t r = 0; r < len; ++r) {
            for (uint32_t col = 0; col < dim; ++col) got_k.push_back(banks.bank(0).at(r, col));
        }
        REQUIRE_MSG(got_k == want, diff_vec("K-tiled", got_k, want));

        // And the tiling is not vacuous: one tile alone is not the answer.
        REQUIRE(banks.stats().accumulates == tiles - 1);
    }
}

SECTION("weight_fifo") {
    Config cfg = small_cfg();
    cfg.weight_fifo_depth = 4;
    cfg.ddr_tile_latency  = 8;

    auto make_tile = [](uint32_t ddr, i8 fill) {
        WeightTile t;
        t.ddr_addr = ddr;
        t.rows = t.cols = 4;
        t.data.assign(16, fill);
        return t;
    };

    WeightFifo fifo(cfg);
    REQUIRE(fifo.depth() == 4);
    REQUIRE(fifo.latency() == 8);
    REQUIRE(fifo.empty());
    REQUIRE(fifo.occupancy() == 0);

    // Popping an empty FIFO is the weight_fifo_empty stall.
    WeightTile out;
    REQUIRE(!fifo.pop(out));
    REQUIRE(fifo.stats().empty_stalls == 1);

    // A refill takes its slot at once but is poppable only after the latency.
    REQUIRE(fifo.push_refill(make_tile(0, 1), 0));
    REQUIRE(fifo.occupancy() == 1);
    REQUIRE(fifo.empty());
    REQUIRE(fifo.next_ready_cycle() == 8);

    fifo.tick(7);
    REQUIRE(fifo.empty());
    REQUIRE(!fifo.pop(out));

    fifo.tick(8);
    REQUIRE(!fifo.empty());
    REQUIRE(fifo.ready() == 1);
    REQUIRE(fifo.pop(out));
    REQUIRE(out.data[0] == 1);
    REQUIRE(out.rows == 4);
    REQUIRE(fifo.occupancy() == 0);

    // Order is preserved, and staging ahead means later tiles are already there.
    {
        WeightFifo f(cfg);
        REQUIRE(f.push_refill(make_tile(0, 10), 0));
        REQUIRE(f.push_refill(make_tile(16, 20), 1));
        REQUIRE(f.push_refill(make_tile(32, 30), 2));
        REQUIRE(f.occupancy() == 3);

        f.tick(8);
        REQUIRE(f.ready() == 1);            // only the first has landed
        f.tick(10);
        REQUIRE(f.ready() == 3);

        WeightTile a, b, c;
        REQUIRE(f.pop(a));
        REQUIRE(f.pop(b));
        REQUIRE(f.pop(c));
        REQUIRE(a.data[0] == 10);
        REQUIRE(b.data[0] == 20);
        REQUIRE(c.data[0] == 30);
        REQUIRE(f.empty());
    }

    // A full FIFO refuses a refill; in-flight tiles count against the depth.
    {
        Config c2 = cfg;
        c2.weight_fifo_depth = 2;
        WeightFifo f(c2);
        REQUIRE(f.push_refill(make_tile(0, 1), 0));
        REQUIRE(f.push_refill(make_tile(16, 2), 0));
        REQUIRE(f.full());
        REQUIRE(!f.push_refill(make_tile(32, 3), 0));
        REQUIRE(f.stats().full_rejects == 1);
    }

    // With depth 1 the second refill waits for the first pop, so two loads cost
    // two full latencies instead of overlapping into one.
    {
        Config deep = cfg;
        deep.weight_fifo_depth = 4;
        Config shallow = cfg;
        shallow.weight_fifo_depth = 1;

        auto time_two_loads = [&](const Config& c) -> uint64_t {
            WeightFifo f(c);
            uint64_t now = 0;
            WeightTile t;

            // Stage as much as the FIFO allows, then consume both tiles.
            REQUIRE(f.push_refill(make_tile(0, 1), now));
            const bool staged_both = f.push_refill(make_tile(16, 2), now);

            f.tick(now = f.next_ready_cycle());
            REQUIRE(f.pop(t));

            if (!staged_both) REQUIRE(f.push_refill(make_tile(16, 2), now));
            f.tick(now = f.next_ready_cycle());
            REQUIRE(f.pop(t));
            return now;
        };

        const uint64_t deep_time    = time_two_loads(deep);
        const uint64_t shallow_time = time_two_loads(shallow);

        REQUIRE(deep_time == 8);                 // both refills overlapped
        REQUIRE(shallow_time == 16);             // strictly serialized
        REQUIRE(shallow_time == 2 * deep_time);
    }
}

SECTION("dma") {
    Config cfg = small_cfg();
    cfg.ub_bytes = 1024;
    cfg.dma_bytes_per_cycle = 16;

    Dma dma(cfg);
    REQUIRE(dma.bytes_per_cycle() == 16);

    // ceil(B / bandwidth), and a nonzero transfer always costs at least a cycle.
    REQUIRE(dma.transfer_cycles(0) == 0);
    REQUIRE(dma.transfer_cycles(1) == 1);
    REQUIRE(dma.transfer_cycles(16) == 1);
    REQUIRE(dma.transfer_cycles(17) == 2);
    REQUIRE(dma.transfer_cycles(160) == 10);
    REQUIRE(dma.transfer_cycles(161) == 11);

    // Host to buffer and back again, byte-identically.
    UnifiedBuffer ub(cfg);
    std::vector<uint8_t> host(512, 0);
    for (uint32_t i = 0; i < 64; ++i) host[100 + i] = static_cast<uint8_t>(i * 3 + 1);

    DmaRequest in;
    in.dir = DmaDir::HOST_TO_UB;
    in.host = 100;
    in.ub = 32;
    in.bytes = 64;
    REQUIRE(dma.start(in, 0, host, ub));
    REQUIRE(dma.finish_cycle() == 4);          // 64 bytes at 16/cycle

    // Busy for exactly the duration, free the cycle it finishes.
    REQUIRE(dma.busy());
    dma.tick(3);
    REQUIRE(dma.busy());
    dma.tick(4);
    REQUIRE(!dma.busy());

    bool bytes_ok = true;
    for (uint32_t i = 0; i < 64; ++i) {
        if (ub.at(32 + i) != static_cast<i8>(host[100 + i])) bytes_ok = false;
    }
    REQUIRE(bytes_ok);

    DmaRequest out;
    out.dir = DmaDir::UB_TO_HOST;
    out.ub = 32;
    out.host = 300;
    out.bytes = 64;
    REQUIRE(dma.start(out, 4, host, ub));
    bool round_trip = true;
    for (uint32_t i = 0; i < 64; ++i) {
        if (host[300 + i] != host[100 + i]) round_trip = false;
    }
    REQUIRE(round_trip);
    REQUIRE(dma.stats().transfers == 2);
    REQUIRE(dma.stats().bytes_moved == 128);

    // One transfer at a time; a second start while busy is refused.
    {
        Dma d(cfg);
        UnifiedBuffer u(cfg);
        std::vector<uint8_t> h(512, 7);
        REQUIRE(d.start(in, 0, h, u));
        REQUIRE(d.busy());
        REQUIRE(!d.start(in, 0, h, u));
        REQUIRE(d.stats().rejects == 1);
        d.tick(d.finish_cycle());
        REQUIRE(d.start(in, d.finish_cycle(), h, u));
    }

    // Out-of-range endpoints are refused on either side.
    {
        Dma d(cfg);
        UnifiedBuffer u(cfg);
        std::vector<uint8_t> h(512, 0);
        DmaRequest bad = in;
        bad.host = 500;
        bad.bytes = 64;                          // runs off host memory
        REQUIRE(!d.start(bad, 0, h, u));
        bad = in;
        bad.ub = 1000;
        bad.bytes = 64;                          // runs off the buffer
        REQUIRE(!d.start(bad, 0, h, u));
        REQUIRE(d.stats().rejects == 2);
    }

    // The 256-byte load that hides under a matmul at full bandwidth dominates
    // it at one byte per cycle, and Config::dma_bound agrees.
    {
        const uint32_t dim = 8, len = 8, bytes = 256;
        const uint64_t compute = len + 2ull * dim - 1ull;      // 23 cycles

        Config fast = small_cfg(dim, 2);
        fast.dma_bytes_per_cycle = 64;
        Config slow = small_cfg(dim, 2);
        slow.dma_bytes_per_cycle = 1;

        REQUIRE(Dma(fast).transfer_cycles(bytes) == 4);
        REQUIRE(Dma(slow).transfer_cycles(bytes) == 256);
        REQUIRE(Dma(fast).transfer_cycles(bytes) < compute);
        REQUIRE(Dma(slow).transfer_cycles(bytes) > compute);

        // The same verdict from arithmetic intensity: len*dim*dim MACs per byte.
        const std::size_t macs = static_cast<std::size_t>(len) * dim * dim;
        REQUIRE(!fast.dma_bound(macs, bytes));
        REQUIRE(slow.dma_bound(macs, bytes));
    }
}

SECTION("tpu_units") {
    // The units wired together and stepped through tick(): DMA in, stage and
    // load a weight tile, matmul, read the bank, all against the oracle.
    const uint32_t dim = 4;
    const uint32_t len = 4;

    Config cfg = small_cfg(dim, 2);
    cfg.ub_bytes = 1024;
    cfg.dma_bytes_per_cycle = 4;
    cfg.ddr_tile_latency = 6;
    cfg.double_buffer = false;

    Lcg rng(31337);
    std::vector<i8> weights(static_cast<std::size_t>(dim) * dim);
    std::vector<i8> acts(static_cast<std::size_t>(len) * dim);
    for (i8& v : weights) v = rng.byte();
    for (i8& v : acts) v = rng.byte();

    Tpu tpu(cfg);
    REQUIRE(tpu.cycle() == 0);
    REQUIRE(tpu.config().dim == dim);

    // Host memory holds the activations at 0x100; weight memory holds the tile.
    for (std::size_t i = 0; i < acts.size(); ++i) {
        tpu.host()[0x100 + i] = static_cast<uint8_t>(acts[i]);
    }
    for (std::size_t i = 0; i < weights.size(); ++i) tpu.weight_mem()[i] = weights[i];

    // DMA in, and wait for it by ticking rather than by assuming a duration.
    REQUIRE(tpu.dma_to_ub(0x100, 0, static_cast<uint32_t>(acts.size())));
    REQUIRE(tpu.dma().busy());
    REQUIRE(tpu.run_until([&] { return !tpu.dma().busy(); }));
    REQUIRE(tpu.cycle() == 4);                  // 16 bytes at 4/cycle

    // The buffer now holds what the host had.
    bool dma_ok = true;
    for (std::size_t i = 0; i < acts.size(); ++i) {
        if (tpu.ub().at(static_cast<UbAddr>(i)) != acts[i]) dma_ok = false;
    }
    REQUIRE(dma_ok);

    // Not poppable until the DDR latency elapses, so an immediate load fails.
    REQUIRE(tpu.stage_weights(0, dim, dim));
    REQUIRE(tpu.weight_fifo().empty());
    REQUIRE(!tpu.load_weights_from_fifo());
    REQUIRE(tpu.weight_fifo().stats().empty_stalls == 1);

    REQUIRE(tpu.run_until([&] { return !tpu.weight_fifo().empty(); }));
    REQUIRE(tpu.cycle() == 4 + 6);
    REQUIRE(tpu.load_weights_from_fifo());

    // Without double buffering the load costs the array dim cycles.
    REQUIRE(tpu.cycle() == 4 + 6 + dim);
    REQUIRE(tpu.mxu().cycle() == tpu.cycle());

    const uint64_t before_matmul = tpu.cycle();
    REQUIRE(tpu.matmul(0, len, 0, false));
    REQUIRE(tpu.cycle() == before_matmul + len + 2ull * dim - 1ull);

    // The bank is unlocked once the matmul is done, so it reads out.
    REQUIRE(!tpu.acc().locked(0));
    I32Tensor got(len, dim);
    REQUIRE(tpu.acc().read(0, got.view()));

    std::vector<int> flat;
    for (uint32_t r = 0; r < len; ++r) {
        for (uint32_t c = 0; c < dim; ++c) flat.push_back(got.view().at(r, c));
    }
    const std::vector<int> want = ref_matmul(cfg, weights, acts, len);
    REQUIRE_MSG(flat == want, diff_vec("hand-driven tile", flat, want));

    // The return path: the bank's low byte per element, then out to the host.
    for (uint32_t i = 0; i < len * dim; ++i) {
        tpu.ub().at(static_cast<UbAddr>(64 + i)) = static_cast<i8>(flat[i] & 0xFF);
    }
    REQUIRE(tpu.dma_to_host(64, 0x200, len * dim));
    REQUIRE(tpu.run_until([&] { return !tpu.dma().busy(); }));
    bool out_ok = true;
    for (uint32_t i = 0; i < len * dim; ++i) {
        if (tpu.host()[0x200 + i] != static_cast<uint8_t>(flat[i] & 0xFF)) out_ok = false;
    }
    REQUIRE(out_ok);

    // The array's clock never lags the machine's, which matters because the
    // sequencer advances the clock while the array waits on a hazard.
    {
        Tpu t(cfg);
        REQUIRE(t.mxu().cycle() == t.cycle());
        for (int i = 0; i < 5; ++i) t.tick();
        REQUIRE(t.mxu().cycle() == t.cycle());

        for (std::size_t i = 0; i < weights.size(); ++i) t.weight_mem()[i] = weights[i];
        for (std::size_t i = 0; i < acts.size(); ++i) {
            t.ub().at(static_cast<UbAddr>(i)) = acts[i];
            t.host()[0x100 + i] = static_cast<uint8_t>(acts[i]);
        }

        REQUIRE(t.dma_to_ub(0x100, 0, static_cast<uint32_t>(acts.size())));
        REQUIRE(t.run_until([&] { return !t.dma().busy(); }));
        REQUIRE(t.mxu().cycle() == t.cycle());

        REQUIRE(t.stage_weights(0, dim, dim));
        REQUIRE(t.run_until([&] { return !t.weight_fifo().empty(); }));
        REQUIRE(t.mxu().cycle() == t.cycle());

        REQUIRE(t.load_weights_from_fifo());
        REQUIRE(t.mxu().cycle() == t.cycle());

        REQUIRE(t.matmul(0, len, 0, false));
        REQUIRE(t.mxu().cycle() == t.cycle());

        // Idling is counted, so waiting on DMA shows up as lost utilization.
        REQUIRE(t.mxu().stats().cycles == t.cycle());
        REQUIRE(t.mxu().utilization() < 1.0);
        REQUIRE(t.mxu().stats().useful_macs == static_cast<uint64_t>(len) * dim * dim);
    }

    // rejections, so a malformed operation stalls rather than corrupts
    {
        Tpu t2(cfg);
        REQUIRE(!t2.matmul(0, dim + 1, 0, false));      // len exceeds bank rows
        REQUIRE(!t2.matmul(0, len, cfg.acc_banks, false));
        REQUIRE(!t2.stage_weights(0, dim + 1, dim));    // tile bigger than array
        REQUIRE(!t2.dma_to_ub(0xFFFF0000, 0, 64));      // off host memory
        REQUIRE(!t2.load_weights_from_fifo());          // nothing staged

        // A locked bank refuses a matmul: the caller must wait.
        t2.acc().lock(0);
        REQUIRE(!t2.matmul(0, len, 0, false));
        t2.acc().unlock(0);
    }

    // Two accumulating matmuls into one bank, driven by hand, against the oracle.
    {
        Config c = cfg;
        Tpu t3(c);

        Lcg r2(88);
        std::vector<i8> w0(16), w1(16), a0(16), a1(16);
        for (i8& v : w0) v = r2.byte();
        for (i8& v : w1) v = r2.byte();
        for (i8& v : a0) v = r2.byte();
        for (i8& v : a1) v = r2.byte();

        for (std::size_t i = 0; i < 16; ++i) {
            t3.weight_mem()[i]      = w0[i];
            t3.weight_mem()[16 + i] = w1[i];
            t3.ub().at(static_cast<UbAddr>(i))      = a0[i];
            t3.ub().at(static_cast<UbAddr>(32 + i)) = a1[i];
        }

        REQUIRE(t3.stage_weights(0, dim, dim));
        REQUIRE(t3.run_until([&] { return !t3.weight_fifo().empty(); }));
        REQUIRE(t3.load_weights_from_fifo());
        REQUIRE(t3.matmul(0, len, 1, false));

        REQUIRE(t3.stage_weights(16, dim, dim));
        REQUIRE(t3.run_until([&] { return !t3.weight_fifo().empty(); }));
        REQUIRE(t3.load_weights_from_fifo());
        REQUIRE(t3.matmul(32, len, 1, true));

        std::vector<int> mine;
        for (uint32_t r = 0; r < len; ++r) {
            for (uint32_t c = 0; c < dim; ++c) mine.push_back(t3.acc().bank(1).at(r, c));
        }

        ref::Machine m(c);
        for (std::size_t i = 0; i < 16; ++i) {
            m.weight_mem[i]      = w0[i];
            m.weight_mem[16 + i] = w1[i];
            m.ub[i]      = a0[i];
            m.ub[32 + i] = a1[i];
        }
        tpuasm::Program p;
        p.read_weights(0).matmul(0, len, 1)
         .read_weights(16).matmul(32, len, 1, /*accumulate=*/true)
         .halt();
        const ref::Result rr = ref::run(m, p.code());
        REQUIRE(rr.halted);
        const std::vector<int> want_k = acc_slice(m, 1, len, dim);
        REQUIRE_MSG(mine == want_k, diff_vec("machine K-tiling", mine, want_k));
    }
}

