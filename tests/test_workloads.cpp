// Workload tests: whole layers lowered by tests/workloads.h and run end to
// end on the machine. Quantized activation and pooling programs, dense and
// tiled matmuls, K larger than the physical array, convolution lowering via
// im2col, and multi-layer MLPs -- each held against plain nested-loop golden
// implementations that never see a tile or an instruction, across
// configuration and scheduling variations.

#include "test_support.h"
// ---------------------------------------------------- @section("activate") ---
SECTION("activate") {
    // A matmul-then-activate layer, byte-for-byte against the oracle, across every
    // activation function and a spread of requantization scales. The arithmetic was
    // pinned exhaustively in the quant section; under test here is that the machine
    // feeds it the right accumulator and puts the result in the right place.
    const uint32_t dim = 4, len = 4;

    Config cfg = small_cfg(dim, 2);
    cfg.ub_bytes = 1024;
    cfg.dma_bytes_per_cycle = 4;
    cfg.ddr_tile_latency = 3;

    const Inputs in = random_inputs(6001, dim, len);

    struct Scale {
        i32      bias;
        i32      multiplier;
        uint32_t shift;
    };
    // Shift 0 saturates hard, large shifts crush everything toward zero, and the
    // middle ones land on ties. A negative bias pushes values through the ReLU
    // boundary rather than leaving every element on one side of it.
    const std::vector<Scale> scales = {
        {0, 1, 0}, {0, 1, 4}, {0, 1, 8}, {0, 3, 7}, {0, 127, 14},
        {100, 1, 6}, {-100, 1, 6}, {0, 1, 31}, {2000000, 1000000, 20},
    };
    const std::vector<ActFn> fns = {ActFn::IDENTITY, ActFn::RELU, ActFn::RELU6};

    int checked = 0;
    for (const ActFn fn : fns) {
        for (const Scale& s : scales) {
            tpuasm::ActArgs a;
            a.acc        = 0;
            a.dst        = 512;
            a.len        = len;
            a.fn         = fn;
            a.bias       = s.bias;
            a.multiplier = s.multiplier;
            a.shift      = s.shift;

            tpuasm::Program p;
            p.read_host(in.host_at, 0, len * dim)
             .read_weights(0)
             .matmul(0, len, 0)
             .activate(a)
             .write_host(512, 0x400, len * dim)
             .halt();

            const std::string diff = diff_tpu(p.code(), cfg, ref_setup(in), tpu_setup(in));
            REQUIRE_MSG(diff.empty(), diff);
            ++checked;
        }
    }
    REQUIRE(checked == static_cast<int>(fns.size() * scales.size()));

    // The three functions do not all produce the same bytes, so the sweep above is
    // actually discriminating between them.
    {
        auto run_with = [&](ActFn fn) {
            Tpu t(cfg);
            tpu_setup(in)(t);
            tpuasm::Program p;
            p.read_host(in.host_at, 0, len * dim)
             .read_weights(0)
             .matmul(0, len, 0)
             .activate(0, 512, len, fn, 1, 6)
             .halt();
            REQUIRE(t.run(p.code()).halted);
            std::vector<int> out;
            for (uint32_t i = 0; i < len * dim; ++i) out.push_back(t.ub().at(512 + i));
            return out;
        };
        const std::vector<int> ident = run_with(ActFn::IDENTITY);
        const std::vector<int> relu  = run_with(ActFn::RELU);
        const std::vector<int> relu6 = run_with(ActFn::RELU6);
        REQUIRE(ident != relu);
        REQUIRE(relu != relu6);

        // ReLU never emits a negative, ReLU6 never exceeds 6, and identity did
        // produce something out of both ranges or the comparison above was luck.
        bool relu_nonneg = true, relu6_bounded = true, ident_negative = false;
        for (std::size_t i = 0; i < ident.size(); ++i) {
            if (relu[i] < 0) relu_nonneg = false;
            if (relu6[i] < 0 || relu6[i] > 6) relu6_bounded = false;
            if (ident[i] < 0) ident_negative = true;
        }
        REQUIRE(relu_nonneg);
        REQUIRE(relu6_bounded);
        REQUIRE(ident_negative);
    }

    // ---- the clamp boundaries, one requantized value at a time --------------
    // Random matmul products land on small integers too rarely to pin down where
    // each function turns over: a ReLU clamping everything below 2 rather than below
    // 0 passed the sweep above, because nothing in it ever requantized to exactly 1.
    // Planting the accumulator directly and walking it across the interesting range,
    // with multiplier 1 and shift 0, requantizes each planted value to itself and
    // hits every boundary exactly.
    {
        Inputs planted;
        planted.acc_at = 0;
        planted.acc_vals.resize(static_cast<std::size_t>(dim) * dim);

        // Enough windows to carry -8..8 plus the int8 limits past every clamp.
        const std::vector<i32> corners = {
            -129, -128, -127, -8, -7, -6, -3, -2, -1, 0, 1, 2, 3,
            5, 6, 7, 8, 126, 127, 128, 1000, -1000,
        };
        for (std::size_t base = 0; base < corners.size(); base += dim * dim) {
            for (std::size_t i = 0; i < planted.acc_vals.size(); ++i) {
                planted.acc_vals[i] = corners[(base + i) % corners.size()];
            }
            for (const ActFn fn : fns) {
                tpuasm::Program p;
                p.activate(0, 512, len, fn, /*multiplier=*/1, /*shift=*/0)
                 .write_host(512, 0x400, len * dim)
                 .halt();
                const std::string diff =
                    diff_tpu(p.code(), cfg, ref_setup(planted), tpu_setup(planted));
                REQUIRE_MSG(diff.empty(), diff);
            }
        }

        // Spot-check the turnover points directly, so the sweep is anchored to
        // stated values and not only to the oracle agreeing with itself.
        for (std::size_t i = 0; i < planted.acc_vals.size(); ++i) {
            planted.acc_vals[i] = static_cast<i32>(i) - 4;   // -4 .. dim*dim-5
        }
        auto emit = [&](ActFn fn) {
            Tpu t(cfg);
            tpu_setup(planted)(t);
            tpuasm::Program p;
            p.activate(0, 512, len, fn, 1, 0).halt();
            REQUIRE(t.run(p.code()).halted);
            std::vector<int> out;
            for (uint32_t i = 0; i < len * dim; ++i) out.push_back(t.ub().at(512 + i));
            return out;
        };
        const std::vector<int> id = emit(ActFn::IDENTITY);
        const std::vector<int> rl = emit(ActFn::RELU);
        const std::vector<int> r6 = emit(ActFn::RELU6);
        for (std::size_t i = 0; i < id.size(); ++i) {
            const int v = static_cast<int>(i) - 4;
            REQUIRE(id[i] == v);
            REQUIRE(rl[i] == (v < 0 ? 0 : v));
            REQUIRE(r6[i] == (v < 0 ? 0 : (v > 6 ? 6 : v)));
        }
        // The three differ at 1 and at 7, which is exactly what the sweep missed.
        REQUIRE(id[3] == -1 && rl[3] == 0);
        REQUIRE(id[5] == 1 && rl[5] == 1 && r6[5] == 1);
        REQUIRE(id[11] == 7 && r6[11] == 6);
    }

    // Activating a bank a K-tiled sequence built up, so the pipeline reads a sum
    // rather than a single matmul's output.
    {
        const Inputs two = random_inputs(6002, dim, len, 2);
        tpuasm::Program p;
        p.read_host(two.host_at, 0, len * dim)
         .read_weights(0).matmul(0, len, 0)
         .read_weights(dim * dim).matmul(0, len, 0, /*accumulate=*/true)
         .activate(0, 512, len, ActFn::RELU, 1, 5)
         .write_host(512, 0x400, len * dim)
         .halt();
        const std::string diff = diff_tpu(p.code(), cfg, ref_setup(two), tpu_setup(two));
        REQUIRE_MSG(diff.empty(), diff);
    }

    // Fewer rows than the array: the pipeline must read `len` rows and leave the
    // rest of the bank alone.
    for (uint32_t rows = 1; rows <= dim; ++rows) {
        tpuasm::Program p;
        p.read_host(in.host_at, 0, len * dim)
         .read_weights(0)
         .matmul(0, len, 0)
         .activate(0, 512, rows, ActFn::RELU, 1, 5)
         .halt();
        const std::string diff = diff_tpu(p.code(), cfg, ref_setup(in), tpu_setup(in));
        REQUIRE_MSG(diff.empty(), diff);
    }
}

// -------------------------------------------------------- @section("pool") ---
SECTION("pool") {
    // Pooling on the requantized stream, against the oracle, over every window and
    // stride that fits -- including the ones that do not divide the input evenly and
    // so drop a ragged edge.
    const uint32_t dim = 6;

    Config cfg = small_cfg(dim, 2);
    cfg.ub_bytes = 1024;
    cfg.dma_bytes_per_cycle = 8;
    cfg.ddr_tile_latency = 3;

    const Inputs in = random_inputs(6100, dim, dim);

    int shapes = 0, uneven = 0;
    for (const Pool mode : {Pool::MAX, Pool::AVG}) {
        for (uint32_t len = 1; len <= dim; ++len) {
            for (uint32_t w = 1; w <= len; ++w) {
                for (uint32_t s = 1; s <= 3; ++s) {
                    if (w > dim) continue;

                    tpuasm::ActArgs a;
                    a.acc         = 0;
                    a.dst         = 512;
                    a.len         = len;
                    a.fn          = ActFn::RELU;
                    a.multiplier  = 1;
                    a.shift       = 5;
                    a.pool        = mode;
                    a.pool_window = w;
                    a.pool_stride = s;

                    tpuasm::Program p;
                    p.read_host(in.host_at, 0, dim * dim)
                     .read_weights(0)
                     .matmul(0, dim, 0)
                     .activate(a)
                     .halt();

                    const std::string diff =
                        diff_tpu(p.code(), cfg, ref_setup(in), tpu_setup(in));
                    REQUIRE_MSG(diff.empty(),
                                "    mode " + std::string(mode == Pool::MAX ? "max" : "avg") +
                                    " len " + std::to_string(len) + " window " +
                                    std::to_string(w) + " stride " + std::to_string(s) + "\n" +
                                    diff);
                    ++shapes;
                    if ((len - w) % s != 0 || (dim - w) % s != 0) ++uneven;
                }
            }
        }
    }
    REQUIRE(shapes > 50);
    REQUIRE(uneven > 0);        // the ragged-edge cases really were exercised

    // Pooling changes the output, so the sweep is not comparing two identical
    // things, and max is not the same as average.
    {
        auto run_pool = [&](Pool mode, uint32_t w, uint32_t s) {
            Tpu t(cfg);
            tpu_setup(in)(t);
            tpuasm::ActArgs a;
            a.acc = 0; a.dst = 512; a.len = dim;
            a.fn = ActFn::IDENTITY; a.multiplier = 1; a.shift = 5;
            a.pool = mode; a.pool_window = w; a.pool_stride = s;
            tpuasm::Program p;
            p.read_host(in.host_at, 0, dim * dim)
             .read_weights(0).matmul(0, dim, 0).activate(a).halt();
            REQUIRE(t.run(p.code()).halted);
            const uint32_t rows = mode == Pool::NONE ? dim : (dim - w) / s + 1;
            std::vector<int> out;
            for (uint32_t i = 0; i < rows * rows; ++i) out.push_back(t.ub().at(512 + i));
            return out;
        };
        const std::vector<int> mx = run_pool(Pool::MAX, 2, 2);
        const std::vector<int> av = run_pool(Pool::AVG, 2, 2);
        REQUIRE(mx.size() == 9);              // a 6x6 tile pooled 2x2 stride 2
        REQUIRE(av.size() == 9);
        REQUIRE(mx != av);
        // A maximum is never below the average of the same window.
        bool max_ge_avg = true;
        for (std::size_t i = 0; i < mx.size(); ++i) {
            if (mx[i] < av[i]) max_ge_avg = false;
        }
        REQUIRE(max_ge_avg);

        // Overlapping windows produce more outputs than disjoint ones.
        REQUIRE(run_pool(Pool::MAX, 2, 1).size() == 25);
    }

    // A window of 1 is a no-op reduction: the result is the unpooled tile.
    {
        Tpu pooled(cfg), plain(cfg);
        tpu_setup(in)(pooled);
        tpu_setup(in)(plain);

        tpuasm::ActArgs a;
        a.acc = 0; a.dst = 512; a.len = dim;
        a.fn = ActFn::RELU; a.multiplier = 1; a.shift = 5;
        a.pool = Pool::MAX; a.pool_window = 1; a.pool_stride = 1;

        tpuasm::Program pp;
        pp.read_host(in.host_at, 0, dim * dim).read_weights(0).matmul(0, dim, 0)
          .activate(a).halt();
        tpuasm::Program qq;
        qq.read_host(in.host_at, 0, dim * dim).read_weights(0).matmul(0, dim, 0)
          .activate(0, 512, dim, ActFn::RELU, 1, 5).halt();

        REQUIRE(pooled.run(pp.code()).halted);
        REQUIRE(plain.run(qq.code()).halted);
        bool same = true;
        for (uint32_t i = 0; i < dim * dim; ++i) {
            if (pooled.ub().at(512 + i) != plain.ub().at(512 + i)) same = false;
        }
        REQUIRE(same);
    }

    // Malformed pooling traps, with the oracle's wording.
    {
        for (const auto& bad : std::vector<std::pair<uint32_t, uint32_t>>{{dim + 1, 1}, {0, 1}, {2, 0}}) {
            tpuasm::ActArgs a;
            a.acc = 0; a.dst = 512; a.len = dim;
            a.fn = ActFn::RELU; a.multiplier = 1; a.shift = 0;
            a.pool = Pool::MAX;
            a.pool_window = bad.first;
            a.pool_stride = bad.second;

            tpuasm::Program p;
            p.activate(a).halt();

            ref::Machine m(cfg);
            Tpu t(cfg);
            const ref::Result rr = ref::run(m, p.code());
            const TpuResult   tr = t.run(p.code());
            REQUIRE(rr.trapped);
            REQUIRE(tr.trapped);
            REQUIRE(tr.trap_reason == rr.trap_reason);
        }
    }
}

// ------------------------------------------------------ tiling helpers -------

namespace {

// Everything one lowered layer produced: what the program computed, what plain
// nested loops say it should be, and whether the machine and the oracle agreed
// along the way.
//
// The two checks answer different questions. The oracle diff cannot catch a
// mis-lowered layer, because the oracle runs the same program: if the tiler emits
// the wrong instructions, both models faithfully execute the wrong thing and agree.
// Only the golden comparison, which never sees a tile or an instruction, can tell
// that the program does not compute the layer it claims to.
struct LayerRun {
    std::vector<i8> got;
    std::vector<i8> want;
    std::string     diff;
    wl::Lowering    low;
    uint64_t        cycles  = 0;
    bool            halted  = false;
    StallStats      stalls;
};

// Lower a layer, run it on the machine, and hold the result against both the
// oracle and the golden loops.
LayerRun run_layer(const wl::Layer& l, const Config& cfg, uint32_t seed,
                   const wl::Placement& at = {}, const wl::Sched& sched = {}) {
    Lcg rng(seed);
    std::vector<i8> a(static_cast<std::size_t>(l.M) * l.K);
    std::vector<i8> b(static_cast<std::size_t>(l.K) * l.N);
    for (i8& v : a) v = rng.byte();
    for (i8& v : b) v = rng.byte();

    LayerRun out;
    out.low  = wl::lower_layer(l, cfg, at, sched);
    out.want = wl::golden_layer(l, a, b);

    Inputs in;
    in.host_at     = out.low.a_host;
    in.host_bytes  = wl::pack(a, l.M, l.K, cfg.dim);
    in.ddr_at      = out.low.b_ddr;
    in.weights     = wl::pack(b, l.K, l.N, cfg.dim);

    Tpu t(cfg);
    tpu_setup(in)(t);
    const TpuResult r = t.run(out.low.code, TpuOptions{});
    out.halted = r.halted;
    out.cycles = r.cycles;
    out.stalls = t.stalls();

    std::vector<i8> packed_c(out.low.c_bytes, 0);
    for (std::size_t i = 0; i < packed_c.size(); ++i) {
        packed_c[i] = static_cast<i8>(t.host()[out.low.c_host + i]);
    }
    out.got = wl::unpack(packed_c, l.M, l.N, cfg.dim);

    out.diff = diff_tpu(out.low.code, cfg, ref_setup(in), tpu_setup(in));
    return out;
}

}  // namespace

// ------------------------------------------------------ @section("tiling") ---
SECTION("tiling") {
    // ---- packing round-trips, including ragged shapes ----------------------
    // The tiler leans on pack/unpack being inverses, so pin that before anything
    // depends on it.
    {
        for (const uint32_t dim : {1u, 2u, 4u, 8u}) {
            for (uint32_t rows = 1; rows <= 9; ++rows) {
                for (uint32_t cols = 1; cols <= 9; ++cols) {
                    Lcg rng(7000 + rows * 16 + cols);
                    std::vector<i8> src(static_cast<std::size_t>(rows) * cols);
                    for (i8& v : src) v = rng.byte();

                    const std::vector<i8> packed = wl::pack(src, rows, cols, dim);
                    REQUIRE(packed.size() == wl::packed_bytes(rows, cols, dim));
                    REQUIRE(wl::unpack(packed, rows, cols, dim) == src);
                }
            }
        }

        // Padding is zero, which is what makes a ragged tile harmless: a padded
        // activation column meets a padded weight row, so the product is zero and
        // the sum is unchanged.
        {
            const std::vector<i8> src(3 * 3, 5);
            const std::vector<i8> packed = wl::pack(src, 3, 3, 4);
            REQUIRE(packed.size() == 16);
            std::size_t fives = 0, zeros = 0;
            for (const i8 v : packed) {
                if (v == 5) ++fives;
                if (v == 0) ++zeros;
            }
            REQUIRE(fives == 9);
            REQUIRE(zeros == 7);
        }

        // A tile-aligned shape needs no padding at all.
        REQUIRE(wl::packed_bytes(8, 8, 4) == 64);
        REQUIRE(wl::packed_bytes(9, 8, 4) == 96);   // rows round up to 12, cols stay 8
        REQUIRE(wl::tiles_of(0, 4) == 0);
        REQUIRE(wl::tiles_of(1, 4) == 1);
        REQUIRE(wl::tiles_of(4, 4) == 1);
        REQUIRE(wl::tiles_of(5, 4) == 2);
    }

    // ---- the headline case: 128x128 x 128x128 on a 32x32 array -------------
    {
        Config cfg;                       // the default configuration
        cfg.dim = 32;

        wl::Layer l;
        l.M = 128; l.K = 128; l.N = 128;
        l.multiplier = 1; l.shift = 10; l.fn = ActFn::RELU;

        const LayerRun r = run_layer(l, cfg, 7100);
        REQUIRE(r.halted);
        REQUIRE_MSG(r.diff.empty(), r.diff);
        REQUIRE_MSG(r.got == r.want, tensor_mismatch(r.got, r.want, l.M, l.N));

        // Four tiles in each of M, N and K.
        REQUIRE(r.low.tiles == 4 * 4 * 4);
        REQUIRE(r.low.macs == 128ull * 128 * 128);

        // The result is not trivially all one value, or the comparison proves
        // nothing. ReLU should leave roughly half of it clamped to zero.
        std::size_t zeros = 0, distinct_hi = 0;
        for (const i8 v : r.got) {
            if (v == 0) ++zeros;
            if (v > 8) ++distinct_hi;
        }
        REQUIRE(zeros > r.got.size() / 8);
        REQUIRE(distinct_hi > r.got.size() / 8);
    }

    // ---- K tiling accumulates across four tiles ----------------------------
    // Isolated from M and N tiling: one output tile, four K tiles piling into one
    // bank. If accumulate-in-place were wrong, only the last K tile would survive and
    // the answer would be off by three quarters of the sum.
    {
        Config cfg;
        cfg.dim = 8;
        cfg.ub_bytes = 4096;

        wl::Layer l;
        l.M = 8; l.K = 32; l.N = 8;
        l.multiplier = 1; l.shift = 8; l.fn = ActFn::IDENTITY;

        const LayerRun r = run_layer(l, cfg, 7200);
        REQUIRE(r.halted);
        REQUIRE_MSG(r.diff.empty(), r.diff);
        REQUIRE_MSG(r.got == r.want, tensor_mismatch(r.got, r.want, l.M, l.N));
        REQUIRE(r.low.tiles == 4);          // one output tile, four K tiles

        // The same shape with K truncated to a single tile must give a different
        // answer, so the extra three tiles demonstrably contributed.
        wl::Layer one = l;
        one.K = 8;
        const LayerRun r1 = run_layer(one, cfg, 7200);
        REQUIRE(r1.halted);
        REQUIRE(r1.low.tiles == 1);
        REQUIRE(r.got != r1.got);
    }

    // ---- shapes that do not divide the array ------------------------------
    // Every combination of ragged M, N and K, on a small array so the sweep is cheap.
    // Zero padding and the short-`len` last row block have to agree with plain loops.
    {
        Config cfg;
        cfg.dim = 4;
        cfg.ub_bytes = 4096;
        cfg.acc_banks = 2;

        int cases = 0, ragged = 0;
        for (const uint32_t M : {1u, 3u, 4u, 5u, 8u, 11u}) {
            for (const uint32_t K : {1u, 3u, 4u, 7u, 8u}) {
                for (const uint32_t N : {1u, 4u, 6u, 8u}) {
                    wl::Layer l;
                    l.M = M; l.K = K; l.N = N;
                    l.multiplier = 1; l.shift = 6; l.fn = ActFn::RELU;

                    const LayerRun r = run_layer(l, cfg, 7300 + cases);
                    REQUIRE(r.halted);
                    REQUIRE_MSG(r.diff.empty(),
                                "    " + std::to_string(M) + "x" + std::to_string(K) + " * " +
                                    std::to_string(K) + "x" + std::to_string(N) + "\n" + r.diff);
                    REQUIRE_MSG(r.got == r.want,
                                "    " + std::to_string(M) + "x" + std::to_string(K) + " * " +
                                    std::to_string(K) + "x" + std::to_string(N) + "\n" +
                                    tensor_mismatch(r.got, r.want, M, N));
                    ++cases;
                    if (M % 4 || K % 4 || N % 4) ++ragged;
                }
            }
        }
        REQUIRE(cases == 6 * 5 * 4);
        REQUIRE(ragged > 100);
    }

    // ---- the last row block is short, not padded out ----------------------
    // A short final block is expressed with a smaller `len`, not by processing padding
    // rows and discarding them afterwards. Padding rows would give the same answer,
    // since unpack drops them, so only the instruction stream and the cycle count can
    // tell the difference; both are checked here.
    {
        Config cfg;
        cfg.dim = 8;
        cfg.ub_bytes = 4096;

        wl::Layer l;
        l.M = 19; l.K = 8; l.N = 8;      // 19 rows: blocks of 8, 8, 3
        l.multiplier = 1; l.shift = 6; l.fn = ActFn::RELU;

        const wl::Lowering low = wl::lower_layer(l, cfg);

        std::vector<uint32_t> mm_len, act_len, dma_bytes;
        for (const RawInst& inst : low.code) {
            const Decoded d = decode(inst);
            if (d.op == Op::MATMUL)     mm_len.push_back(d.len);
            if (d.op == Op::ACTIVATE)   act_len.push_back(d.len);
            if (d.op == Op::WRITE_HOST) dma_bytes.push_back(d.bytes);
        }
        const std::vector<uint32_t> want_len{8, 8, 3};
        REQUIRE(mm_len == want_len);
        REQUIRE(act_len == want_len);
        REQUIRE(dma_bytes == std::vector<uint32_t>({64, 64, 24}));

        // And the short block costs less, which is the point of not padding it.
        wl::Layer full = l;
        full.M = 24;                      // three whole blocks
        const LayerRun ragged = run_layer(l, cfg, 7350);
        const LayerRun padded = run_layer(full, cfg, 7350);
        REQUIRE(ragged.halted);
        REQUIRE(padded.halted);
        REQUIRE(ragged.cycles < padded.cycles);
    }

    // ---- the requantization operands survive lowering ---------------------
    // A layer spec carries bias, multiplier, shift and activation function; the
    // tiler has to put each on every Activate it emits.
    {
        Config cfg;
        cfg.dim = 4;
        cfg.ub_bytes = 4096;

        for (const ActFn fn : {ActFn::IDENTITY, ActFn::RELU, ActFn::RELU6}) {
            for (const i32 bias : {0, 300, -300}) {
                for (const uint32_t shift : {4u, 7u, 11u}) {
                    wl::Layer l;
                    l.M = 6; l.K = 6; l.N = 6;
                    l.bias = bias; l.multiplier = 3; l.shift = shift; l.fn = fn;

                    const LayerRun r = run_layer(l, cfg, 7400);
                    REQUIRE(r.halted);
                    REQUIRE_MSG(r.diff.empty(), r.diff);
                    REQUIRE_MSG(r.got == r.want, tensor_mismatch(r.got, r.want, l.M, l.N));
                }
            }
        }
    }

    // ---- the lowering is deterministic ------------------------------------
    // Bundled examples are only reproducible if the same spec lowers to the same
    // bytes every time.
    {
        Config cfg;
        cfg.dim = 8;
        wl::Layer l;
        l.M = 20; l.K = 12; l.N = 16;

        const wl::Lowering x = wl::lower_layer(l, cfg);
        const wl::Lowering y = wl::lower_layer(l, cfg);
        REQUIRE(x.code.size() == y.code.size());
        bool identical = true;
        for (std::size_t i = 0; i < x.code.size(); ++i) {
            for (uint32_t w = 0; w < ISA_WORDS; ++w) {
                if (x.code[i].word[w] != y.code[i].word[w]) identical = false;
            }
        }
        REQUIRE(identical);
    }

    // ---- tiling actually overlaps ----------------------------------------
    // Alternating accumulator banks and double buffering the output staging tile let
    // the drain of one output tile hide under the next tile's arithmetic. With a
    // single bank there is nowhere to hide.
    {
        Config cfg;
        cfg.dim = 8;
        cfg.ub_bytes = 8192;

        wl::Layer l;
        l.M = 16; l.K = 16; l.N = 32;
        l.multiplier = 1; l.shift = 8; l.fn = ActFn::RELU;

        Config one_bank = cfg;
        one_bank.acc_banks = 1;
        Config four_bank = cfg;
        four_bank.acc_banks = 4;

        const LayerRun a = run_layer(l, one_bank, 7500);
        const LayerRun b = run_layer(l, four_bank, 7500);
        REQUIRE(a.halted);
        REQUIRE(b.halted);
        REQUIRE_MSG(a.got == a.want, tensor_mismatch(a.got, a.want, l.M, l.N));
        REQUIRE_MSG(b.got == b.want, tensor_mismatch(b.got, b.want, l.M, l.N));

        // Same answer either way; only the schedule differs.
        REQUIRE(a.got == b.got);

        auto brk = [](const LayerRun& r) {
            const StallStats& s = r.stalls;
            return "cyc " + std::to_string(r.cycles) + " raw " + std::to_string(s.ub_raw) +
                   " war " + std::to_string(s.ub_war) + " waw " + std::to_string(s.ub_waw) +
                   " acc " + std::to_string(s.accum_hazard) + " wt " +
                   std::to_string(s.weight_stall) + " busy " + std::to_string(s.unit_busy) +
                   " fifoF " + std::to_string(s.weight_fifo_full) + " fifoE " +
                   std::to_string(s.weight_fifo_empty) + " drain " + std::to_string(s.drain);
        };
        REQUIRE_MSG(b.cycles < a.cycles,
                    "    1 bank:  " + brk(a) + "\n    4 banks: " + brk(b) + "\n");
        REQUIRE(a.stalls.accum_hazard > b.stalls.accum_hazard);

        // ---- the output staging tile is double buffered ---------------------
        // Consecutive Activates have to write different buffers. Sharing one stays
        // correct, since the scoreboard would stall the Activate until the drain it
        // collides with finished, so the only trace of the choice is in the addresses
        // the tiler emits.
        {
            const wl::Lowering low = wl::lower_layer(l, four_bank);
            std::vector<UbAddr> dsts;
            for (const RawInst& inst : low.code) {
                const Decoded d = decode(inst);
                if (d.op == Op::ACTIVATE) dsts.push_back(d.ub_addr);
            }
            REQUIRE(dsts.size() > 2);

            std::vector<UbAddr> distinct = dsts;
            std::sort(distinct.begin(), distinct.end());
            distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
            REQUIRE(distinct.size() == 2);

            // Strictly alternating, so no Activate ever waits on the drain of the
            // one immediately before it.
            bool alternates = true;
            for (std::size_t i = 1; i < dsts.size(); ++i) {
                if (dsts[i] == dsts[i - 1]) alternates = false;
            }
            REQUIRE(alternates);
        }

        // ---- and the deferred drain is what makes the banks reachable -------
        // Emitting each Write_Host right after its Activate costs nothing in
        // correctness and a great deal in cycles: issue is in order, so the drain
        // stalls on its own activation and the next tile's MatMuls queue behind it,
        // unable to reach the idle banks waiting for them.
        wl::Sched naive;
        naive.defer_drain = false;

        const LayerRun c = run_layer(l, four_bank, 7500, wl::Placement{}, naive);
        REQUIRE(c.halted);
        REQUIRE(c.got == b.got);
        REQUIRE_MSG(b.cycles < c.cycles,
                    "    deferred: " + brk(b) + "\n    naive:    " + brk(c) + "\n");

        // With the naive order the extra banks buy nothing at all, because the
        // schedule never lets two tiles be in flight to use them.
        const LayerRun d = run_layer(l, one_bank, 7500, wl::Placement{}, naive);
        REQUIRE(d.got == b.got);
        REQUIRE_MSG(d.cycles == c.cycles,
                    "    naive 1 bank:  " + brk(d) + "\n    naive 4 banks: " + brk(c) + "\n");
    }
}

// -------------------------------------------------------- @section("conv") ---
SECTION("conv") {
    // Run a convolution by lowering it to im2col plus the dense tiler, and hold the
    // result against a direct convolution that knows nothing about either.
    auto run_conv = [](const wl::Conv& c, const Config& cfg, uint32_t seed) {
        REQUIRE(c.valid());
        const wl::Layer l = c.as_layer();

        Lcg rng(seed);
        std::vector<i8> in(static_cast<std::size_t>(c.H) * c.W * c.Cin);
        std::vector<i8> w(static_cast<std::size_t>(l.K) * l.N);
        for (i8& v : in) v = rng.byte();
        for (i8& v : w) v = rng.byte();

        LayerRun out;
        out.low  = wl::lower_layer(l, cfg);
        out.want = wl::golden_conv(c, in, w);

        // im2col is the only conv-specific step; after it this is a dense layer.
        const std::vector<i8> cols = wl::im2col(c, in);

        Inputs tensors;
        tensors.host_at    = out.low.a_host;
        tensors.host_bytes = wl::pack(cols, l.M, l.K, cfg.dim);
        tensors.ddr_at     = out.low.b_ddr;
        tensors.weights    = wl::pack(w, l.K, l.N, cfg.dim);

        Tpu t(cfg);
        tpu_setup(tensors)(t);
        const TpuResult r = t.run(out.low.code);
        out.halted = r.halted;
        out.cycles = r.cycles;

        std::vector<i8> packed(out.low.c_bytes, 0);
        for (std::size_t i = 0; i < packed.size(); ++i) {
            packed[i] = static_cast<i8>(t.host()[out.low.c_host + i]);
        }
        out.got  = wl::unpack(packed, l.M, l.N, cfg.dim);
        out.diff = diff_tpu(out.low.code, cfg, ref_setup(tensors), tpu_setup(tensors));

        // im2col has to reproduce the direct convolution's accumulators exactly,
        // checked before requantization so a padding or stride error cannot hide
        // inside a rounding step.
        const std::vector<i32> via_cols = wl::golden_matmul(cols, w, l.M, l.K, l.N);
        REQUIRE(via_cols == wl::golden_conv_acc(c, in, w));
        return out;
    };

    Config cfg;
    cfg.dim = 8;
    cfg.ub_bytes = 32 * 1024;

    // ---- the shapes that make padding and stride matter --------------------
    int cases = 0;
    for (const uint32_t pad : {0u, 1u}) {
        for (const uint32_t stride : {1u, 2u}) {
            for (const uint32_t k : {1u, 3u}) {
                wl::Conv c;
                c.H = 7; c.W = 7; c.Cin = 3;
                c.R = k; c.S = k; c.Cout = 5;
                c.stride = stride;
                c.pad    = pad;
                c.multiplier = 1; c.shift = 7; c.fn = ActFn::RELU;
                if (!c.valid()) continue;

                const std::string what = "    " + std::to_string(k) + "x" + std::to_string(k) +
                                         " stride " + std::to_string(stride) + " pad " +
                                         std::to_string(pad) + "\n";

                const LayerRun r = run_conv(c, cfg, 7600 + cases);
                REQUIRE_MSG(r.halted, what);
                REQUIRE_MSG(r.diff.empty(), what + r.diff);
                REQUIRE_MSG(r.got == r.want,
                            what + tensor_mismatch(r.got, r.want, c.out_h() * c.out_w(), c.Cout));
                ++cases;
            }
        }
    }
    REQUIRE(cases == 8);

    // ---- the output shape follows from padding and stride ------------------
    {
        wl::Conv c;
        c.H = 7; c.W = 7; c.Cin = 1; c.R = 3; c.S = 3; c.Cout = 1;

        c.stride = 1; c.pad = 0;
        REQUIRE(c.out_h() == 5 && c.out_w() == 5);
        c.pad = 1;                              // "same" padding at stride 1
        REQUIRE(c.out_h() == 7 && c.out_w() == 7);
        c.stride = 2; c.pad = 0;
        REQUIRE(c.out_h() == 3 && c.out_w() == 3);
        c.stride = 2; c.pad = 1;
        REQUIRE(c.out_h() == 4 && c.out_w() == 4);

        // A kernel bigger than the padded input is not a convolution.
        c.R = 9; c.S = 9; c.pad = 0;
        REQUIRE(!c.valid());
    }

    // ---- padding really is zero -------------------------------------------
    // With an all-ones input and an all-ones 3x3 kernel, each output counts the input
    // pixels the window covered: four at the corners of a padded convolution, six at
    // the edges, nine in the interior. Padding contributing anything other than zero
    // would show up as a wrong border.
    {
        wl::Conv c;
        c.H = 5; c.W = 5; c.Cin = 1;
        c.R = 3; c.S = 3; c.Cout = 1;
        c.stride = 1; c.pad = 1;
        c.multiplier = 1; c.shift = 0; c.fn = ActFn::IDENTITY;

        const std::vector<i8> in(25, 1);
        const std::vector<i8> w(9, 1);
        const std::vector<i32> acc = wl::golden_conv_acc(c, in, w);
        REQUIRE(acc.size() == 25);
        REQUIRE(acc[0] == 4);                  // top-left corner
        REQUIRE(acc[4] == 4);                  // top-right corner
        REQUIRE(acc[2] == 6);                  // top edge
        REQUIRE(acc[12] == 9);                 // interior
        REQUIRE(acc[24] == 4);                 // bottom-right corner

        // And im2col agrees, which is the property the lowering depends on.
        REQUIRE(wl::golden_matmul(wl::im2col(c, in), w, 25, 9, 1) == acc);
    }

    // ---- a layer whose K needs several tiles -------------------------------
    // A 3x3x16 kernel flattens to K = 144, eighteen tiles on an 8-wide array, so the
    // convolution exercises K accumulation as well as im2col.
    {
        wl::Conv c;
        c.H = 6; c.W = 6; c.Cin = 16;
        c.R = 3; c.S = 3; c.Cout = 12;
        c.stride = 1; c.pad = 1;
        c.multiplier = 1; c.shift = 11; c.fn = ActFn::RELU6;

        const wl::Layer l = c.as_layer();
        REQUIRE(l.M == 36);
        REQUIRE(l.K == 144);
        REQUIRE(l.N == 12);

        const LayerRun r = run_conv(c, cfg, 7700);
        REQUIRE(r.halted);
        REQUIRE_MSG(r.diff.empty(), r.diff);
        REQUIRE_MSG(r.got == r.want, tensor_mismatch(r.got, r.want, l.M, l.N));

        // ReLU6 means every output sits in [0, 6]; without that the comparison
        // above could be passing on a saturated constant.
        bool bounded = true, varied = false;
        for (const i8 v : r.got) {
            if (v < 0 || v > 6) bounded = false;
            if (v != r.got[0]) varied = true;
        }
        REQUIRE(bounded);
        REQUIRE(varied);
    }
}

// --------------------------------------------------------- @section("mlp") ---
SECTION("mlp") {
    // Run a shipped workload and hold it against the golden answer and the oracle.
    // This is the only place the two are checked on a program nobody wrote by hand.
    auto run_workload = [](const wl::Workload& w) {
        Inputs tensors;
        tensors.host_at    = w.a_host;
        tensors.host_bytes = w.a;
        tensors.ddr_at     = w.b_ddr;
        tensors.weights    = w.b;

        LayerRun out;
        out.want = w.expect;

        Tpu t(w.cfg);
        tpu_setup(tensors)(t);
        const TpuResult r = t.run(w.code);
        out.halted = r.halted;
        out.cycles = r.cycles;
        out.stalls = t.stalls();

        std::vector<i8> packed(w.y_bytes, 0);
        for (std::size_t i = 0; i < packed.size(); ++i) {
            packed[i] = static_cast<i8>(t.host()[w.y_host + i]);
        }
        out.got  = wl::unpack(packed, w.out_rows, w.out_cols, w.cfg.dim);
        out.diff = diff_tpu(w.code, w.cfg, ref_setup(tensors), tpu_setup(tensors));
        return out;
    };

    // ---- a three-layer network chained with no repacking -------------------
    {
        Config cfg;
        cfg.dim = 8;
        cfg.ub_bytes = 16 * 1024;

        auto make = [](uint32_t M, uint32_t K, uint32_t N, ActFn fn) {
            wl::Layer l;
            l.M = M; l.K = K; l.N = N;
            l.multiplier = 1; l.shift = 8; l.fn = fn;
            return l;
        };

        wl::Mlp net;
        net.layers.push_back(make(16, 24, 16, ActFn::RELU));
        net.layers.push_back(make(16, 16, 24, ActFn::RELU));
        net.layers.push_back(make(16, 24, 8, ActFn::IDENTITY));
        REQUIRE(net.chains());

        const wl::Workload w = wl::mlp_workload("t", "", net, cfg, 7800);
        const LayerRun r = run_workload(w);
        REQUIRE(r.halted);
        REQUIRE_MSG(r.diff.empty(), r.diff);
        REQUIRE_MSG(r.got == r.want,
                    tensor_mismatch(r.got, r.want, w.out_rows, w.out_cols));

        // Three stages really ran: the program is the concatenation of three
        // lowerings plus one Halt, and exactly one Halt.
        std::size_t halts = 0, matmuls = 0;
        for (const RawInst& inst : w.code) {
            const Decoded d = decode(inst);
            if (d.op == Op::HALT)   ++halts;
            if (d.op == Op::MATMUL) ++matmuls;
        }
        REQUIRE(halts == 1);
        // 2x2x3 + 2x3x2 + 2x1x3 tiles of (m, n, k).
        REQUIRE(matmuls == 12 + 12 + 6);

        // Dropping the last layer changes the answer, so all three contributed.
        wl::Mlp two = net;
        two.layers.pop_back();
        const wl::Workload w2 = wl::mlp_workload("t2", "", two, cfg, 7800);
        const LayerRun r2 = run_workload(w2);
        REQUIRE(r2.halted);
        REQUIRE_MSG(r2.got == r2.want, tensor_mismatch(r2.got, r2.want, w2.out_rows, w2.out_cols));
        REQUIRE(r2.got.size() != r.got.size());
    }

    // ---- chaining survives widths that do not divide the array -------------
    // A layer's packed output has padding columns holding whatever the activation made
    // of an all-zero accumulator, and the next layer reads those bytes as activations.
    // They stay harmless only because they meet the zero rows of the next layer's
    // packed weights. A non-zero bias makes that padding non-zero, so this is the
    // shape that would catch it.
    {
        Config cfg;
        cfg.dim = 8;
        cfg.ub_bytes = 16 * 1024;

        auto make = [](uint32_t M, uint32_t K, uint32_t N, i32 bias) {
            wl::Layer l;
            l.M = M; l.K = K; l.N = N;
            l.bias = bias; l.multiplier = 1; l.shift = 6; l.fn = ActFn::RELU;
            return l;
        };

        wl::Mlp net;
        net.layers.push_back(make(11, 13, 5, 400));    // padded output columns
        net.layers.push_back(make(11, 5, 7, 400));     // reads them as activations
        net.layers.push_back(make(11, 7, 3, 400));
        REQUIRE(net.chains());

        const wl::Workload w = wl::mlp_workload("ragged", "", net, cfg, 7900);
        const LayerRun r = run_workload(w);
        REQUIRE(r.halted);
        REQUIRE_MSG(r.diff.empty(), r.diff);
        REQUIRE_MSG(r.got == r.want,
                    tensor_mismatch(r.got, r.want, w.out_rows, w.out_cols));

        // The bias really did make the padding non-zero, so the case above is not
        // passing by accident. An all-zero accumulator plus this bias requantizes to
        // something positive, and every padding column holds it.
        REQUIRE(quant::requantize_biased(0, 400, 1, 6) > 0);
    }

    // ---- a mismatched network is rejected ---------------------------------
    {
        auto make = [](uint32_t M, uint32_t K, uint32_t N) {
            wl::Layer l;
            l.M = M; l.K = K; l.N = N;
            return l;
        };
        wl::Mlp bad_width;
        bad_width.layers.push_back(make(8, 8, 8));
        bad_width.layers.push_back(make(8, 4, 8));      // K != previous N
        REQUIRE(!bad_width.chains());

        wl::Mlp bad_batch;
        bad_batch.layers.push_back(make(8, 8, 8));
        bad_batch.layers.push_back(make(4, 8, 8));      // different batch
        REQUIRE(!bad_batch.chains());

        REQUIRE(!wl::Mlp{}.chains());
    }

    // ---- every shipped workload ------------------------------------------
    // The bundled examples are exactly these, so a green run here is what makes a
    // file in examples/ trustworthy.
    {
        const std::vector<wl::Workload> all = wl::corpus();
        REQUIRE(all.size() == 4);

        std::vector<std::string> names;
        for (const wl::Workload& w : all) {
            REQUIRE(!w.code.empty());
            REQUIRE(w.expect.size() ==
                    static_cast<std::size_t>(w.out_rows) * w.out_cols);
            REQUIRE(w.macs > 0);

            const LayerRun r = run_workload(w);
            REQUIRE_MSG(r.halted, "    " + w.name + " did not halt\n");
            REQUIRE_MSG(r.diff.empty(), "    " + w.name + "\n" + r.diff);
            REQUIRE_MSG(r.got == r.want,
                        "    " + w.name + "\n" +
                            tensor_mismatch(r.got, r.want, w.out_rows, w.out_cols));

            // A workload whose output is one repeated value would satisfy the
            // comparison without exercising anything.
            bool varied = false;
            for (const i8 v : r.got) {
                if (v != r.got.front()) varied = true;
            }
            REQUIRE_MSG(varied, "    " + w.name + " produced a constant output\n");

            names.push_back(w.name);
        }

        // Names are unique, since they become filenames.
        std::sort(names.begin(), names.end());
        REQUIRE(std::unique(names.begin(), names.end()) == names.end());

        // A shipped workload survives the round trip through the formats
        // gen_examples writes and the CLI reads. Done in memory rather than against
        // examples/ so the check does not depend on the generator having been run.
        {
            const wl::Workload& w = all.back();

            std::ostringstream prog_text;
            save_program_hex(prog_text, w.code);
            std::istringstream prog_in(prog_text.str());
            const std::vector<RawInst> reloaded = load_program_hex(prog_in);
            REQUIRE(reloaded.size() == w.code.size());
            bool code_ok = true;
            for (std::size_t i = 0; i < reloaded.size(); ++i) {
                for (uint32_t word = 0; word < ISA_WORDS; ++word) {
                    if (reloaded[i].word[word] != w.code[i].word[word]) code_ok = false;
                }
            }
            REQUIRE(code_ok);

            TensorBlob blob;
            blob.rows = w.out_rows;
            blob.cols = w.out_cols;
            blob.i8v  = w.expect;
            std::ostringstream tensor_bytes(std::ios::binary);
            save_tensor_mtpu(tensor_bytes, blob);
            std::istringstream tensor_in(tensor_bytes.str(), std::ios::binary);
            const TensorBlob back = load_tensor_mtpu(tensor_in);
            REQUIRE(back.rows == w.out_rows);
            REQUIRE(back.cols == w.out_cols);
            REQUIRE(!back.wide);
            REQUIRE(back.i8v == w.expect);

            // And the reloaded program still computes the workload.
            Inputs tensors;
            tensors.host_at    = w.a_host;
            tensors.host_bytes = w.a;
            tensors.ddr_at     = w.b_ddr;
            tensors.weights    = w.b;

            Tpu t(w.cfg);
            tpu_setup(tensors)(t);
            REQUIRE(t.run(reloaded).halted);
            std::vector<i8> packed(w.y_bytes, 0);
            for (std::size_t i = 0; i < packed.size(); ++i) {
                packed[i] = static_cast<i8>(t.host()[w.y_host + i]);
            }
            REQUIRE(wl::unpack(packed, w.out_rows, w.out_cols, w.cfg.dim) == w.expect);
        }

        // And the corpus is reproducible: building it twice gives the same bytes.
        const std::vector<wl::Workload> again = wl::corpus();
        REQUIRE(again.size() == all.size());
        bool same = true;
        for (std::size_t i = 0; i < all.size(); ++i) {
            if (all[i].name != again[i].name) same = false;
            if (all[i].a != again[i].a || all[i].b != again[i].b) same = false;
            if (all[i].expect != again[i].expect) same = false;
            if (all[i].code.size() != again[i].code.size()) { same = false; continue; }
            for (std::size_t k = 0; k < all[i].code.size(); ++k) {
                for (uint32_t word = 0; word < ISA_WORDS; ++word) {
                    if (all[i].code[k].word[word] != again[i].code[k].word[word]) same = false;
                }
            }
        }
        REQUIRE(same);
    }
}
