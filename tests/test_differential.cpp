// Differential validation: the eager reference model itself, the comparison
// scaffold that diffs the timed machine against it on output bytes,
// accumulator snapshots at every Sync and the retired count, the derived
// statistics and their exact stall partition, machine-level structural and
// roofline properties, and the workload x configuration sweep that runs
// every shipped workload on every reference configuration.

#include "test_support.h"
// --------------------------------------------------------- @section("ref") ---
SECTION("ref") {
    const Config cfg = small_cfg();      // 4x4 array, 2 accumulator banks

    // ---- a dense matmul, hand-checked ------------------------------------
    // W[k][c] = 4k + c + 1. A row of A picks out rows of W:
    //   [1 0 0 0] -> W row 0            = [ 1  2  3  4]
    //   [0 1 1 0] -> W row 1 + W row 2  = [14 16 18 20]
    {
        ref::Machine m(cfg);
        put_weights(m, 0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        tpuasm::UbAlloc ua(0);
        const tpuasm::Region acts = ua.tile(2, 4);
        put_bytes(m, acts.addr, {1, 0, 0, 0, 0, 1, 1, 0});

        tpuasm::Program p;
        p.read_weights(0).matmul(acts, 2, 0).sync().halt(0);

        const ref::Result r = ref::run(m, p.code());
        REQUIRE(r.halted);
        REQUIRE(!r.trapped);
        REQUIRE(r.retired == 4);
        REQUIRE(r.exit_code == 0u);
        REQUIRE(r.syncs.size() == 1);
        const std::vector<int> want{1, 2, 3, 4, 14, 16, 18, 20};
        REQUIRE_MSG(acc_slice(m, 0, 2, 4) == want,
                    diff_vec("acc", acc_slice(m, 0, 2, 4), want));
    }

    // ---- DMA in and back out again ---------------------------------------
    {
        ref::Machine m(cfg);
        for (uint32_t i = 0; i < 8; ++i) m.host[0x100 + i] = static_cast<uint8_t>(0xF0 + i);

        tpuasm::Program p;
        p.read_host(0x100, 0x0, 8).write_host(0x0, 0x200, 8).halt();

        const ref::Result r = ref::run(m, p.code());
        REQUIRE(r.halted);
        REQUIRE(host_slice(m, 0x200, 8) == host_slice(m, 0x100, 8));
        REQUIRE(ub_slice(m, 0, 8) == host_slice(m, 0x100, 8));
    }

    // ---- K-tiling: two matmuls accumulating into one bank ----------------
    // tile 0 is the identity, so the first matmul deposits A0 = [1 2 3 4].
    // tile 1 is all 2s, so the second adds sum(A1) * 2 = 8 to every column.
    {
        ref::Machine m(cfg);
        put_weights(m, 0, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
        put_weights(m, 16, {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
        put_bytes(m, 0, {1, 2, 3, 4});
        put_bytes(m, 4, {1, 1, 1, 1});

        tpuasm::Program p;
        p.read_weights(0).matmul(0, 1, 0)
         .read_weights(16).matmul(4, 1, 0, /*accumulate=*/true)
         .sync().halt();

        const ref::Result r = ref::run(m, p.code());
        REQUIRE(r.halted);
        const std::vector<int> want{9, 10, 11, 12};
        REQUIRE_MSG(acc_slice(m, 0, 1, 4) == want,
                    diff_vec("k-tiled acc", acc_slice(m, 0, 1, 4), want));

        // Without the accumulate flag the second matmul would have overwritten.
        ref::Machine m2(cfg);
        put_weights(m2, 0, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
        put_weights(m2, 16, {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
        put_bytes(m2, 0, {1, 2, 3, 4});
        put_bytes(m2, 4, {1, 1, 1, 1});
        tpuasm::Program p2;
        p2.read_weights(0).matmul(0, 1, 0)
          .read_weights(16).matmul(4, 1, 0, /*accumulate=*/false)
          .halt();
        ref::run(m2, p2.code());
        const std::vector<int> overwritten{8, 8, 8, 8};
        REQUIRE_MSG(acc_slice(m2, 0, 1, 4) == overwritten,
                    diff_vec("overwritten acc", acc_slice(m2, 0, 1, 4), overwritten));
    }

    // ---- activation: ReLU over a saturating requantize -------------------
    // -5 stays -5 through requantize then clamps to 0 at ReLU; 200 saturates
    // to 127 before ReLU ever sees it.
    {
        ref::Machine m(cfg);
        put_acc(m, 0, {-5, 0, 3, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

        tpuasm::Program p;
        p.activate(0, 0x100, 1, ActFn::RELU, 1, 0).halt();
        ref::run(m, p.code());

        const std::vector<int> want{0, 0, 3, 127};
        REQUIRE_MSG(ub_slice(m, 0x100, 4) == want,
                    diff_vec("relu out", ub_slice(m, 0x100, 4), want));
    }

    // ---- activation: bias and shift, hand-checked ------------------------
    // biased = [5 10 13 210], then /2 rounding halves away from zero.
    {
        ref::Machine m(cfg);
        put_acc(m, 0, {-5, 0, 3, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

        tpuasm::ActArgs a;
        a.acc        = 0;
        a.dst        = 0x100;
        a.len        = 1;
        a.fn         = ActFn::IDENTITY;
        a.bias       = 10;
        a.multiplier = 1;
        a.shift      = 1;

        tpuasm::Program p;
        p.activate(a).halt();
        ref::run(m, p.code());

        const std::vector<int> want{3, 5, 7, 105};
        REQUIRE_MSG(ub_slice(m, 0x100, 4) == want,
                    diff_vec("biased out", ub_slice(m, 0x100, 4), want));
    }

    // ---- activation: ReLU6 clamps in the output's own units --------------
    {
        ref::Machine m(cfg);
        put_acc(m, 0, {-2, 3, 6, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
        tpuasm::Program p;
        p.activate(0, 0x100, 1, ActFn::RELU6, 1, 0).halt();
        ref::run(m, p.code());
        const std::vector<int> want{0, 3, 6, 6};
        REQUIRE_MSG(ub_slice(m, 0x100, 4) == want,
                    diff_vec("relu6 out", ub_slice(m, 0x100, 4), want));
    }

    // ---- pooling: 2x2 windows, stride 2, over a 4x4 tile ----------------
    // Every average here lands on a .5 tie, so this also pins the rounding.
    {
        const std::initializer_list<int> ramp{1, 2, 3, 4, 5, 6, 7, 8,
                                             9, 10, 11, 12, 13, 14, 15, 16};
        {
            ref::Machine m(cfg);
            put_acc(m, 0, ramp);
            tpuasm::ActArgs a;
            a.acc = 0; a.dst = 0x100; a.len = 4;
            a.pool = Pool::MAX; a.pool_window = 2; a.pool_stride = 2;
            tpuasm::Program p;
            p.activate(a).halt();
            ref::run(m, p.code());
            const std::vector<int> want{6, 8, 14, 16};
            REQUIRE_MSG(ub_slice(m, 0x100, 4) == want,
                        diff_vec("max pool", ub_slice(m, 0x100, 4), want));
        }
        {
            ref::Machine m(cfg);
            put_acc(m, 0, ramp);
            tpuasm::ActArgs a;
            a.acc = 0; a.dst = 0x100; a.len = 4;
            a.pool = Pool::AVG; a.pool_window = 2; a.pool_stride = 2;
            tpuasm::Program p;
            p.activate(a).halt();
            ref::run(m, p.code());
            const std::vector<int> want{4, 6, 12, 14};   // 3.5 5.5 11.5 13.5
            REQUIRE_MSG(ub_slice(m, 0x100, 4) == want,
                        diff_vec("avg pool", ub_slice(m, 0x100, 4), want));
        }
        // A window that does not divide the input evenly drops the ragged edge:
        // 3x3 stride 2 over 4x4 leaves exactly one window.
        {
            ref::Machine m(cfg);
            put_acc(m, 0, ramp);
            tpuasm::ActArgs a;
            a.acc = 0; a.dst = 0x100; a.len = 4;
            a.pool = Pool::MAX; a.pool_window = 3; a.pool_stride = 2;
            tpuasm::Program p;
            p.activate(a).halt();
            ref::run(m, p.code());
            REQUIRE(ub_slice(m, 0x100, 1) == std::vector<int>{11});
        }
    }

    // ---- a partial tile is correct, not merely plausible ----------------
    // A real 2x3 by 3x2 matmul padded into the 4x4 array: the padding
    // contributes zero, so the live 2x2 corner is the true product.
    {
        ref::Machine m(cfg);
        put_weights(m, 0, {1, 2, 0, 0,
                           3, 4, 0, 0,
                           5, 6, 0, 0,
                           0, 0, 0, 0});
        put_bytes(m, 0, {1, 0, 1, 0,
                         2, 1, 0, 0});

        tpuasm::Program p;
        p.read_weights(0).matmul(0, 2, 0).halt();
        ref::run(m, p.code());

        const std::vector<int> want{6, 8, 0, 0, 5, 8, 0, 0};
        REQUIRE_MSG(acc_slice(m, 0, 2, 4) == want,
                    diff_vec("padded matmul", acc_slice(m, 0, 2, 4), want));
    }

    // ---- Sync snapshots capture state as of the barrier ------------------
    {
        ref::Machine m(cfg);
        put_weights(m, 0, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
        put_bytes(m, 0, {1, 2, 3, 4});

        tpuasm::Program p;
        p.read_weights(0)
         .matmul(0, 1, 0).sync()
         .matmul(0, 1, 0, /*accumulate=*/true).sync()
         .halt();

        const ref::Result r = ref::run(m, p.code());
        REQUIRE(r.syncs.size() == 2);
        // The first barrier sees one pass, the second sees it doubled.
        REQUIRE(r.syncs[0].acc[0] == 1);
        REQUIRE(r.syncs[0].acc[3] == 4);
        REQUIRE(r.syncs[1].acc[0] == 2);
        REQUIRE(r.syncs[1].acc[3] == 8);
        // The snapshot is a copy, not a view of live state.
        REQUIRE(r.syncs[0].acc.size() == m.acc.size());
    }

    // ---- traps: every out-of-range operand is caught ---------------------
    {
        ref::Machine m(cfg);
        tpuasm::Program p;
        p.read_host(static_cast<HostAddr>(m.host.size() - 4), 0, 64).halt();
        const ref::Result r = ref::run(m, p.code());
        REQUIRE(r.trapped);
        REQUIRE(!r.halted);
        REQUIRE(r.trap_reason == "Read_Host_Memory: host range");
    }
    {
        ref::Machine m(cfg);
        tpuasm::Program p;
        p.matmul(0, cfg.dim + 1, 0).halt();
        const ref::Result r = ref::run(m, p.code());
        REQUIRE(r.trapped);
        REQUIRE(r.trap_reason == "MatMul: len exceeds accumulator rows");
    }
    {
        ref::Machine m(cfg);
        tpuasm::Program p;
        p.matmul(0, 1, cfg.acc_banks).halt();
        const ref::Result r = ref::run(m, p.code());
        REQUIRE(r.trapped);
        REQUIRE(r.trap_reason == "MatMul: bank out of range");
    }
    {
        ref::Machine m(cfg);
        tpuasm::Program p;
        p.activate(cfg.acc_banks, 0, 1, ActFn::RELU).halt();
        const ref::Result r = ref::run(m, p.code());
        REQUIRE(r.trapped);
        REQUIRE(r.trap_reason == "Activate: bank out of range");
    }
    {
        // A pool window larger than the input has no valid position.
        ref::Machine m(cfg);
        tpuasm::ActArgs a;
        a.acc = 0; a.dst = 0; a.len = 2;
        a.pool = Pool::MAX; a.pool_window = 3; a.pool_stride = 1;
        tpuasm::Program p;
        p.activate(a).halt();
        const ref::Result r = ref::run(m, p.code());
        REQUIRE(r.trapped);
        REQUIRE(r.trap_reason == "Activate: pool window larger than input");
    }
    {
        // Falling off the end is a program bug, so it is reported as one.
        ref::Machine m(cfg);
        tpuasm::Program p;
        p.nop();
        const ref::Result r = ref::run(m, p.code());
        REQUIRE(r.trapped);
        REQUIRE(r.trap_reason == "ran past the end of the program");
    }
    {
        ref::Machine m(cfg);
        RawInst bad;
        bad.word[0] = 0xFFu;
        tpuasm::Program p;
        p.raw(bad);
        const ref::Result r = ref::run(m, p.code());
        REQUIRE(r.trapped);
        REQUIRE(r.trap_reason == "illegal opcode");
    }

    // ---- the runaway backstop --------------------------------------------
    {
        ref::Machine m(cfg);
        tpuasm::Program p;
        for (int i = 0; i < 50; ++i) p.nop();
        p.halt();
        ref::Options opts;
        opts.max_insts = 10;
        const ref::Result r = ref::run(m, p.code(), opts);
        REQUIRE(r.budget);
        REQUIRE(!r.halted);
        REQUIRE(!r.trapped);
        REQUIRE(r.retired == 10u);
    }

    // ---- exit code comes back from Halt ---------------------------------
    {
        ref::Machine m(cfg);
        tpuasm::Program p;
        p.halt(37);
        const ref::Result r = ref::run(m, p.code());
        REQUIRE(r.halted);
        REQUIRE(r.exit_code == 37u);
        REQUIRE(r.retired == 1u);
    }
}

// ------------------------------------------------ @section("diff_scaffold") ---
SECTION("diff_scaffold") {
    const Config cfg = small_cfg();

    tpuasm::UbAlloc ua(0);
    const tpuasm::Region acts = ua.tile(4, 4);
    const tpuasm::Region out  = ua.tile(4, 4);

    tpuasm::Program p;
    p.read_host(0x100, acts, 16)
     .read_weights(0)
     .matmul(acts, 4, 0)
     .sync()
     .activate(0, out, 4, ActFn::RELU, 1, 2)
     .write_host(out, 0x200, 16)
     .sync()
     .halt(3);

    const Setup setup = [](ref::Machine& m) {
        for (uint32_t i = 0; i < 16; ++i) m.host[0x100 + i] = static_cast<uint8_t>(i + 1);
        for (uint32_t i = 0; i < 16; ++i) {
            m.weight_mem[i] = static_cast<i8>(static_cast<int>(i % 5) - 2);
        }
    };

    // Two runs of the same program over the same inputs agree on every comparand,
    // establishing that the comparison works before there is a second
    // implementation to point it at.
    const std::string same = diff_run(p.code(), cfg, setup);
    REQUIRE_MSG(same.empty(), same);

    // The program is doing real work, so agreement is not vacuous.
    {
        ref::Machine m(cfg);
        setup(m);
        const ref::Result r = ref::run(m, p.code());
        REQUIRE(r.halted);
        REQUIRE(r.exit_code == 3u);
        REQUIRE(r.syncs.size() == 2);
        bool any_nonzero = false;
        for (const i32 v : m.acc) {
            if (v != 0) any_nonzero = true;
        }
        REQUIRE(any_nonzero);
    }

    // ---- and now the part that makes the check above mean something -------
    // A comparator that always answered "equal" would pass every assertion so
    // far. Each divergence below must be caught, one comparand at a time.
    auto two_runs = [&](const std::vector<RawInst>& pa, const Setup& sa,
                        const std::vector<RawInst>& pb, const Setup& sb) {
        ref::Machine ma(cfg), mb(cfg);
        sa(ma);
        sb(mb);
        const ref::Result ra = ref::run(ma, pa);
        const ref::Result rb = ref::run(mb, pb);
        return compare_runs("a", view_of(ma, ra), "b", view_of(mb, rb));
    };

    // exit code
    {
        tpuasm::Program other;
        other.read_host(0x100, acts, 16).read_weights(0).matmul(acts, 4, 0).sync()
             .activate(0, out, 4, ActFn::RELU, 1, 2).write_host(out, 0x200, 16).sync()
             .halt(4);
        REQUIRE(!two_runs(p.code(), setup, other.code(), setup).empty());
    }
    // retired count
    {
        tpuasm::Program longer;
        longer.read_host(0x100, acts, 16).read_weights(0).matmul(acts, 4, 0).sync()
              .activate(0, out, 4, ActFn::RELU, 1, 2).write_host(out, 0x200, 16).sync()
              .nop().halt(3);
        REQUIRE(!two_runs(p.code(), setup, longer.code(), setup).empty());
    }
    // sync count
    {
        tpuasm::Program fewer;
        fewer.read_host(0x100, acts, 16).read_weights(0).matmul(acts, 4, 0)
             .activate(0, out, 4, ActFn::RELU, 1, 2).write_host(out, 0x200, 16)
             .halt(3);
        REQUIRE(!two_runs(p.code(), setup, fewer.code(), setup).empty());
    }
    // accumulators, via different weights
    {
        const Setup other_weights = [](ref::Machine& m) {
            for (uint32_t i = 0; i < 16; ++i) m.host[0x100 + i] = static_cast<uint8_t>(i + 1);
            for (uint32_t i = 0; i < 16; ++i) m.weight_mem[i] = 1;
        };
        REQUIRE(!two_runs(p.code(), setup, p.code(), other_weights).empty());
    }
    // host and unified-buffer bytes, via different activations
    {
        const Setup other_acts = [](ref::Machine& m) {
            for (uint32_t i = 0; i < 16; ++i) m.host[0x100 + i] = static_cast<uint8_t>(i + 2);
            for (uint32_t i = 0; i < 16; ++i) {
                m.weight_mem[i] = static_cast<i8>(static_cast<int>(i % 5) - 2);
            }
        };
        REQUIRE(!two_runs(p.code(), setup, p.code(), other_acts).empty());
    }
    // a trap on one side only
    {
        tpuasm::Program bad;
        bad.matmul(0, cfg.dim + 1, 0).halt(3);
        REQUIRE(!two_runs(p.code(), setup, bad.code(), setup).empty());
    }
}

// ------------------------------------------------ characterization helpers ---

// Run a shipped workload on a given configuration and gather its statistics.
struct Measured {
    stats::Stats stats;
    std::vector<i8> got;
    bool halted = false;
    uint64_t cycles = 0;

    // Resource-leak checks: after a run every unit must be quiet and every
    // accumulator-bank lock released, or a hazard would deadlock the next run.
    bool     quiet        = false;
    uint32_t locked_banks = 0;
};

// How much host and weight memory a workload's addresses reach, so a big array
// does not silently run out of room.
std::size_t host_span(const wl::Workload& w) {
    const std::size_t a_end = w.a_host + w.a.size();
    const std::size_t y_end = w.y_host + w.y_bytes;
    return std::max<std::size_t>(1u << 16, std::max(a_end, y_end) + 64);
}

std::size_t weight_span(const wl::Workload& w) {
    return std::max<std::size_t>(1u << 16, w.b_ddr + w.b.size() + 64);
}

Inputs tensors_of(const wl::Workload& w) {
    Inputs in;
    in.host_at    = w.a_host;
    in.host_bytes = w.a;
    in.ddr_at     = w.b_ddr;
    in.weights    = w.b;
    return in;
}

Measured measure(const wl::Workload& w, const Config& cfg) {
    const Inputs tensors = tensors_of(w);

    Measured out;
    Tpu t(cfg, host_span(w), weight_span(w));
    tpu_setup(tensors)(t);

    const TpuResult r = t.run(w.code);
    out.halted = r.halted;
    out.cycles = r.cycles;
    out.stats  = stats::gather(t, r, w.macs, /*useful_known=*/true);

    out.quiet = t.quiet();
    for (BankId b = 0; b < t.acc().banks(); ++b) {
        if (t.acc().locked(b)) ++out.locked_banks;
    }

    std::vector<i8> packed(w.y_bytes, 0);
    for (std::size_t i = 0; i < packed.size(); ++i) {
        packed[i] = static_cast<i8>(t.host()[w.y_host + i]);
    }
    out.got = wl::unpack(packed, w.out_rows, w.out_cols, cfg.dim);
    return out;
}

// ------------------------------------------------------- @section("stats") ---
SECTION("stats") {
    // ---- the accounting balances ------------------------------------------
    // Everything else in this section reads a number off the breakdown, so first
    // establish that the breakdown is a partition: every cycle is either one the array
    // was busy or one charged to exactly one cause. Without it, a "dominant cause"
    // would be whichever bucket got double counted the most.
    {
        const std::vector<wl::Workload> all = wl::corpus();
        for (const wl::Workload& w : all) {
            const Measured m = measure(w, w.cfg);
            REQUIRE_MSG(m.halted, "    " + w.name + "\n");

            const stats::Stats& s = m.stats;
            REQUIRE_MSG(s.balances(),
                        "    " + w.name + ": busy " + std::to_string(s.array_busy) +
                            " + idle " + std::to_string(s.lost.idle()) + " != cycles " +
                            std::to_string(s.cycles) + "; stream " +
                            std::to_string(s.stream_cycles) + " + fill/drain " +
                            std::to_string(s.lost.array_fill_drain) + " != busy\n");

            // And the run did real work, so the balance is not the trivial one.
            REQUIRE(s.cycles > 0);
            REQUIRE(s.array_busy > 0);
            REQUIRE(s.stream_cycles > 0);
            REQUIRE(s.macs_useful == w.macs);
            REQUIRE(s.dma_bytes > 0);
            REQUIRE(s.retired == w.code.size());
        }
    }

    // ---- per-instruction cycle counts ------------------------------------
    {
        Config cfg;
        cfg.dim = 8;
        cfg.ub_bytes = 4096;
        cfg.act_pipeline_depth = 4;
        cfg.dma_bytes_per_cycle = 8;

        tpuasm::Program p;
        p.read_host(0, 0, 64)          // 64 bytes at 8 B/cyc = 8 cycles
         .read_weights(0)
         .matmul(0, 8, 0)              // 8 + 2*8 - 1 = 23 cycles
         .activate(0, 512, 8, ActFn::RELU, 1, 4)   // 8*8 + 4 = 68 cycles
         .halt();

        Tpu t(cfg);
        const TpuResult r = t.run(p.code());
        REQUIRE(r.halted);
        const stats::Stats s = stats::gather(t, r);
        REQUIRE(s.balances());

        const auto idx = [](Op op) { return static_cast<std::size_t>(op); };
        REQUIRE(s.op_count[idx(Op::READ_HOST)] == 1);
        REQUIRE(s.op_cycles[idx(Op::READ_HOST)] == 8);
        REQUIRE(s.op_count[idx(Op::MATMUL)] == 1);
        REQUIRE(s.op_cycles[idx(Op::MATMUL)] == 8 + 2 * 8 - 1);
        REQUIRE(s.op_count[idx(Op::ACTIVATE)] == 1);
        REQUIRE(s.op_cycles[idx(Op::ACTIVATE)] == 8 * 8 + 4);
        REQUIRE(s.dma_bytes == 64);

        // With no workload MAC count supplied, "useful" falls back to what the
        // array performed, so there is no phantom padding waste.
        REQUIRE(!s.useful_known);
        REQUIRE(s.macs_useful == s.macs_performed);
        REQUIRE(s.lost.partial_tile_waste == 0);
    }

    // ---- utilization and TOPS move the way they should --------------------
    {
        wl::Layer l;
        l.M = 64; l.K = 64; l.N = 64;
        l.multiplier = 1; l.shift = 10; l.fn = ActFn::RELU;

        Config small = wl::example_config();
        small.dim = 8;
        Config big = wl::example_config();
        big.dim = 64;

        const wl::Workload ws = wl::dense_workload("s", "", l, small, 8000);
        const wl::Workload wb = wl::dense_workload("b", "", l, big, 8000);

        const Measured ms = measure(ws, small);
        const Measured mb = measure(wb, big);
        REQUIRE(ms.halted);
        REQUIRE(mb.halted);

        // Same useful work either way.
        REQUIRE(ms.stats.macs_useful == mb.stats.macs_useful);
        REQUIRE(ms.stats.macs_useful == 64ull * 64 * 64);

        // The big array finishes sooner in cycles but wastes a larger share of its
        // width doing it, so utilization falls as the array grows past the problem.
        REQUIRE(mb.cycles < ms.cycles);
        REQUIRE(mb.stats.utilization() < ms.stats.utilization());

        // Utilization is a fraction, and TOPS scales with useful work over time.
        REQUIRE(ms.stats.utilization() > 0.0 && ms.stats.utilization() <= 1.0);
        REQUIRE(mb.stats.utilization() > 0.0 && mb.stats.utilization() <= 1.0);
        REQUIRE(ms.stats.tops() > 0.0);
        REQUIRE(mb.stats.tops(1.4) > mb.stats.tops(0.7));   // twice the clock
    }

    // ---- padding shows up as partial_tile_waste ---------------------------
    {
        Config cfg;
        cfg.dim = 16;
        cfg.ub_bytes = 8192;

        // K = 1 on a 16-wide array: fifteen sixteenths of every MAC is padding.
        wl::Layer thin;
        thin.M = 16; thin.K = 1; thin.N = 16;
        thin.multiplier = 1; thin.shift = 4; thin.fn = ActFn::RELU;

        wl::Layer full = thin;
        full.K = 16;

        const Measured a = measure(wl::dense_workload("thin", "", thin, cfg, 8100), cfg);
        const Measured b = measure(wl::dense_workload("full", "", full, cfg, 8100), cfg);
        REQUIRE(a.halted);
        REQUIRE(b.halted);

        // Both perform the same number of MACs; only the useful fraction differs.
        REQUIRE(a.stats.macs_performed == b.stats.macs_performed);
        REQUIRE(a.stats.macs_useful * 16 == b.stats.macs_useful);
        REQUIRE(a.stats.lost.partial_tile_waste > b.stats.lost.partial_tile_waste);
        REQUIRE(b.stats.lost.partial_tile_waste == 0);   // nothing padded
        REQUIRE(a.stats.utilization() < b.stats.utilization());
    }

    // ---- the breakdown diagnoses a starved configuration ------------------
    // Starve one resource of a workload and the bucket named after that resource must
    // account for the slowdown.
    //
    // Checked as a delta rather than as a dominant cause, because on these workloads
    // the largest bucket is almost always array_fill_drain: a matmul streams dim rows
    // but pays 2*dim-1 cycles of pipeline latency, so most of the array's time goes
    // into filling and draining whatever the configuration. That is a property of a
    // small systolic array rather than a provisioning problem. What makes the
    // breakdown a diagnosis is that removing a resource puts the extra cycles in that
    // resource's bucket and nowhere else.
    {
        struct Starved {
            const char*  name;
            Config       cfg;
            stats::Cause expect;
        };

        for (const wl::Spec& spec : wl::specs()) {
            Config base = spec.cfg;
            base.dim = 16;              // one array size, so only the knob varies
            base.ub_bytes = 64 * 1024;

            // A 1-deep FIFO alone is not a stress: the prefetcher starts the next
            // fetch while the array works, and the default 8-cycle DDR latency hides
            // completely behind a single matmul. Depth only costs anything once DDR is
            // slower than the compute available to cover it, so the weight-starved
            // configuration slows DDR down too.
            Config fifo = base;
            fifo.weight_fifo_depth = 1;
            fifo.ddr_tile_latency  = 400;

            Config dma = base;
            dma.dma_bytes_per_cycle = 1;

            const Starved cases[] = {
                {"1-deep FIFO, slow DDR", fifo, stats::Cause::WEIGHT_FIFO_EMPTY},
                {"1 byte/cycle DMA",      dma,  stats::Cause::DMA_BOUND},
            };

            const Measured ref = measure(wl::build(spec, base), base);
            REQUIRE(ref.halted);

            for (const Starved& s : cases) {
                const Measured m = measure(wl::build(spec, s.cfg), s.cfg);
                const std::string where = "    " + spec.name + " / " + s.name + ": ";
                REQUIRE_MSG(m.halted, where + "did not halt\n");

                // Starving the resource cost time,
                REQUIRE_MSG(m.cycles > ref.cycles,
                            where + std::to_string(m.cycles) + " cycles vs baseline " +
                                std::to_string(ref.cycles) + "\n");

                // the named bucket grew,
                const uint64_t before = ref.stats.weight_of(s.expect);
                const uint64_t after  = m.stats.weight_of(s.expect);
                REQUIRE_MSG(after > before,
                            where + std::string(stats::cause_name(s.expect)) + " " +
                                std::to_string(before) + " -> " + std::to_string(after) +
                                "\n");

                // and it grew by enough to explain most of the slowdown. Not all of
                // it: a stall that delays issue also shifts everything behind it, so
                // a few cycles leak into neighbouring buckets.
                const uint64_t slower = m.cycles - ref.cycles;
                const uint64_t grew   = after - before;
                REQUIRE_MSG(2 * grew >= slower,
                            where + std::to_string(slower) + " cycles slower but " +
                                stats::cause_name(s.expect) + " only grew by " +
                                std::to_string(grew) + "\n");

                // and no other bucket grew more than it did, so the breakdown points
                // at the resource that was actually taken away.
                const stats::Cause others[] = {
                    stats::Cause::WEIGHT_FIFO_EMPTY, stats::Cause::UB_BANK_CONFLICT,
                    stats::Cause::ACCUM_HAZARD,      stats::Cause::DMA_BOUND,
                    stats::Cause::ACTIVATION,        stats::Cause::OTHER,
                };
                for (const stats::Cause c : others) {
                    if (c == s.expect) continue;
                    const uint64_t b = ref.stats.weight_of(c);
                    const uint64_t a = m.stats.weight_of(c);
                    const uint64_t g = a > b ? a - b : 0;
                    REQUIRE_MSG(g <= grew,
                                where + std::string(cause_name(c)) + " grew by " +
                                    std::to_string(g) + ", more than " +
                                    stats::cause_name(s.expect) + "'s " +
                                    std::to_string(grew) + "\n");
                }
            }
        }
    }

    // ---- Unified Buffer port contention -----------------------------------
    // A single-bank UB is a weaker stress than it looks, and the reason is worth
    // stating precisely.
    //
    // A bank carries one read and one write per cycle, so the bank count limits how
    // many same-direction transfers can be in flight. But at most one instruction per
    // unit is ever in flight, so the only same-direction pairs that can exist are a
    // matmul and a Write_Host reading, or a Read_Host and an Activate writing: two
    // streams per direction, whatever the workload. One bank therefore serializes at
    // most one pair at a time, and above one bank no contention remains to find at
    // any depth.
    //
    // The mechanism is real and worth measuring, but on the shipped workloads it is
    // worth tens of cycles rather than thousands and is never the dominant cause.
    // Making it one would take more concurrency than this sequencer has.
    {
        const uint32_t dim = 8, len = 8;
        const uint32_t a_ub = 0, c_ub = 256;
        const uint32_t drain_bytes = 1024;   // long enough to outlast the activation

        // A long drain of one region overlapping matmuls that read another. Nothing
        // connects them but the buffer's read ports.
        tpuasm::Program p;
        p.read_weights(0)
         .matmul(a_ub, len, 0)
         .activate(0, c_ub, len, ActFn::IDENTITY, 1, 0)
         .write_host(c_ub, 0, drain_bytes)
         .matmul(a_ub, len, 1)
         .matmul(a_ub, len, 2)
         .halt();

        auto run_banks = [&](uint32_t nbanks) {
            Config cfg = small_cfg(dim, 3);
            cfg.ub_bytes = 4096;
            cfg.ub_banks = nbanks;
            cfg.dma_bytes_per_cycle = 8;    // 1024 bytes = 128 cycles, longer than a matmul

            Inputs in = random_inputs(7788, dim, len, 3);
            in.ub_at = a_ub;
            in.ub_bytes.assign(in.host_bytes.begin(), in.host_bytes.end());

            Tpu t(cfg);
            tpu_setup(in)(t);
            const TpuResult r = t.run(p.code());
            REQUIRE(r.halted);
            return stats::gather(t, r);
        };

        const stats::Stats many = run_banks(8);
        const stats::Stats one  = run_banks(1);

        // Eight banks: the drain and the matmuls behind it overlap freely.
        REQUIRE(many.lost.ub_bank_conflict == 0);

        // One bank: the matmuls are readers too, so they wait for the drain to give
        // the read port back, and the wait is charged to the right bucket.
        REQUIRE_MSG(one.lost.ub_bank_conflict > 0,
                    "    one bank produced no conflicts at all\n");
        REQUIRE_MSG(one.cycles > many.cycles,
                    "    1 bank: " + std::to_string(one.cycles) + " cycles, 8 banks: " +
                        std::to_string(many.cycles) + "\n");

        const uint64_t slower = one.cycles - many.cycles;
        REQUIRE_MSG(one.lost.ub_bank_conflict >= slower,
                    "    " + std::to_string(slower) + " cycles slower, " +
                        std::to_string(one.lost.ub_bank_conflict) + " charged to banks\n");

        // Here it *is* the dominant resource stall, because the program was built to
        // make it one.
        REQUIRE_MSG(one.dominant_stall_cause() == stats::Cause::UB_BANK_CONFLICT,
                    "    largest stall is " + std::string(one.dominant_stall_name()) + "\n");

        // Above one bank the knob is inert: nothing in this machine can ask for a
        // third stream in either direction, so every shipped workload runs
        // conflict-free at any bank count above one.
        for (uint32_t nbanks : {2u, 4u, 8u}) {
            REQUIRE(run_banks(nbanks).lost.ub_bank_conflict == 0);
        }
        for (const wl::Spec& spec : wl::specs()) {
            Config cfg = spec.cfg;
            cfg.dim = 16;
            cfg.ub_bytes = 64 * 1024;
            cfg.ub_banks = 2;
            const Measured m = measure(wl::build(spec, cfg), cfg);
            REQUIRE(m.halted);
            REQUIRE_MSG(m.stats.lost.ub_bank_conflict == 0,
                        "    " + spec.name + " conflicted with two banks\n");
        }
    }

    // ---- a DMA-bound run is dominated by its transfers ---------------------
    // Starve the DMA far enough and data movement outweighs everything else,
    // fill and drain included.
    {
        for (const wl::Spec& spec : wl::specs()) {
            Config cfg = spec.cfg;
            cfg.dim = 16;
            cfg.ub_bytes = 64 * 1024;
            cfg.dma_bytes_per_cycle = 1;

            const Measured m = measure(wl::build(spec, cfg), cfg);
            REQUIRE(m.halted);
            REQUIRE_MSG(m.stats.dominant_stall_cause() == stats::Cause::DMA_BOUND,
                        "    " + spec.name + ": largest stall is " +
                            m.stats.dominant_stall_name() + "\n");
            REQUIRE_MSG(m.stats.below_ridge(),
                        "    " + spec.name + ": intensity " +
                            std::to_string(m.stats.arithmetic_intensity()) + " vs ridge " +
                            std::to_string(m.stats.ridge_point()) + "\n");
        }
    }

    // ---- the report renders ----------------------------------------------
    {
        const wl::Workload w = wl::corpus().front();
        const Measured m = measure(w, w.cfg);
        const std::string text = m.stats.report(w.name);
        REQUIRE(text.find(w.name) != std::string::npos);
        REQUIRE(text.find("dominant cause") != std::string::npos);
        REQUIRE(text.find("MatMul") != std::string::npos);
        REQUIRE(text.find("utilization") != std::string::npos);
        REQUIRE(text.size() > 200);
    }
}

// -------------------------------------------------- @section("properties") ---

// Requantization computed from the specification rather than from quant.h: a
// multiply, a divide by 2^shift rounding halves away from zero, and a saturating
// narrow to int8. The timed model and the oracle deliberately share quant.h, so a
// bit-exactness test against that same code would be vacuous; the third opinion has
// to come from here.
i8 spec_requantize(i32 acc, i32 bias, i32 multiplier, uint32_t shift) {
    const long long biased  = static_cast<long long>(acc) + bias;
    const long long product = biased * multiplier;

    long long scaled = product;
    if (shift != 0) {
        const long long half = 1LL << (shift - 1);
        scaled = product >= 0 ? (product + half) >> shift
                              : -((-product + half) >> shift);
    }

    if (scaled >  127) return static_cast<i8>(127);
    if (scaled < -128) return static_cast<i8>(-128);
    return static_cast<i8>(scaled);
}

SECTION("properties") {
    // Invariants that hold without reference to the oracle. Everything else in the
    // suite says "the two models agree"; these say what the machine actually is, so
    // a change that moves both models together still has to answer to them.

    // ---- a matmul takes exactly len + 2*dim - 1 cycles ---------------------
    // The systolic fill and drain, stated as a closed form and checked across every
    // stream length the ISA permits, on three array sizes.
    for (const uint32_t dim : {4u, 8u, 16u}) {
        for (uint32_t len = 1; len <= dim; ++len) {
            Config cfg = small_cfg(dim, 2);
            cfg.ub_bytes = 16 * dim * dim;
            cfg.ddr_tile_latency = 0;      // measure the array, not memory

            const Inputs in = random_inputs(3300 + dim, dim, len, 2);

            // One matmul, timed against the same program without it.
            tpuasm::Program with, without;
            with.read_weights(0).matmul(0, len, 0).halt();
            without.read_weights(0).halt();

            auto cycles_of = [&](const std::vector<RawInst>& code) {
                Tpu t(cfg);
                tpu_setup(in)(t);
                for (uint32_t i = 0; i < len * dim; ++i) t.ub().at(i) = in.host_bytes[i];
                const TpuResult r = t.run(code);
                REQUIRE(r.halted);
                return r.cycles;
            };

            const uint64_t delta = cycles_of(with.code()) - cycles_of(without.code());
            REQUIRE_MSG(delta == len + 2ull * dim - 1ull,
                        "    dim " + std::to_string(dim) + " len " + std::to_string(len) +
                            ": matmul cost " + std::to_string(delta) + " cycles, want " +
                            std::to_string(len + 2 * dim - 1) + "\n");
        }
    }

    // ---- and that fixes a ceiling on utilization --------------------------
    // A full tile cannot reach >90% array utilization, and the reason is structural.
    //
    // Utilization is amortized over the stream: len cycles of work for len + 2*dim - 1
    // cycles of occupancy. Long streams would amortize the fill away, but `len` cannot
    // exceed dim, because a matmul's results land in one accumulator bank and a bank
    // is dim rows deep. The best a single matmul can do is therefore dim / (3*dim - 1),
    // which falls towards one third as the array grows.
    //
    // The array is draining, not stalling: deeper accumulator banks rather than more
    // bandwidth are what would move this number.
    for (const uint32_t dim : {8u, 32u}) {
        Config cfg = small_cfg(dim, 2);
        cfg.ub_bytes = 16 * dim * dim;
        cfg.ddr_tile_latency = 0;

        const Inputs in = random_inputs(4400 + dim, dim, dim, 2);

        tpuasm::Program p;
        p.read_weights(0).matmul(0, dim, 0).halt();

        Tpu t(cfg);
        tpu_setup(in)(t);
        for (uint32_t i = 0; i < dim * dim; ++i) t.ub().at(i) = in.host_bytes[i];
        const TpuResult r = t.run(p.code());
        REQUIRE(r.halted);

        const stats::Stats s = stats::gather(t, r);
        const double best = static_cast<double>(dim) / static_cast<double>(3 * dim - 1);

        REQUIRE(s.stream_cycles == dim);
        REQUIRE(s.array_busy == dim + 2ull * dim - 1ull);
        REQUIRE_MSG(std::abs(s.busy_utilization() - best) < 1e-9,
                    "    dim " + std::to_string(dim) + ": busy utilization " +
                        std::to_string(s.busy_utilization()) + ", ceiling " +
                        std::to_string(best) + "\n");
        REQUIRE(s.busy_utilization() < 0.35);
        REQUIRE(s.busy_utilization() > 1.0 / 3.0);   // and it does not fall below a third
    }

    // ---- the activation pipeline is scalar, and that is what binds --------
    // The throughput-1 pipeline emits one requantized element per cycle, so an
    // Activate over a full bank costs dim*dim + depth cycles while the matmul that
    // filled that bank cost only 3*dim - 1. The ratio grows linearly with dim, so past
    // a small array the machine spends most of its time requantizing rather than
    // multiplying, which is what the breakdown reports for the 32x32 and 256x256
    // configurations.
    //
    // There is no crossover: dim*dim + depth exceeds 3*dim - 1 at every array size, so
    // the pipeline is always the more expensive half and the gap only widens. Pinning
    // it here means a future dim-wide activation pipeline has to change this test on
    // purpose.
    double prev_ratio = 0.0;
    for (const uint32_t dim : {4u, 8u, 32u}) {
        Config cfg = small_cfg(dim, 2);
        cfg.ub_bytes = 16 * dim * dim;
        cfg.act_pipeline_depth = 4;

        const Inputs in = random_inputs(6600 + dim, dim, dim, 2);

        tpuasm::Program p;
        p.read_weights(0).matmul(0, dim, 0).activate(0, 0, dim, ActFn::RELU, 1, 8).halt();

        Tpu t(cfg);
        tpu_setup(in)(t);
        for (uint32_t i = 0; i < dim * dim; ++i) t.ub().at(i) = in.host_bytes[i];
        const TpuResult r = t.run(p.code());
        REQUIRE(r.halted);

        const stats::Stats s = stats::gather(t, r);
        const uint64_t act_cost = s.op_cycles[static_cast<std::size_t>(Op::ACTIVATE)];
        const uint64_t mm_cost  = s.op_cycles[static_cast<std::size_t>(Op::MATMUL)];

        REQUIRE_MSG(act_cost == static_cast<uint64_t>(dim) * dim + 4,
                    "    dim " + std::to_string(dim) + ": activate cost " +
                        std::to_string(act_cost) + "\n");
        REQUIRE(mm_cost == 3ull * dim - 1ull);

        // Always the expensive half, and increasingly so.
        const double ratio = static_cast<double>(act_cost) / static_cast<double>(mm_cost);
        REQUIRE_MSG(ratio > 1.0,
                    "    dim " + std::to_string(dim) + ": activate/matmul " +
                        std::to_string(ratio) + "\n");
        REQUIRE(ratio > prev_ratio);
        prev_ratio = ratio;
    }
    // By dim 32 the pipeline costs an order of magnitude more than the matmul that
    // fed it, which is why `activation` is the dominant stall at that size.
    REQUIRE(prev_ratio > 10.0);

    // ---- the weight-load bubble is zero, or exactly dim -------------------
    {
        const uint32_t dim = 8, len = 8;
        const Inputs   in  = random_inputs(5500, dim, len, 2);

        tpuasm::Program p;
        p.read_weights(0)
         .matmul(0, len, 0)
         .read_weights(dim * dim)
         .matmul(0, len, 1)
         .halt();

        auto bubble_of = [&](bool double_buffer) {
            Config cfg = small_cfg(dim, 2);
            cfg.ub_bytes = 16 * dim * dim;
            cfg.double_buffer = double_buffer;

            Tpu t(cfg);
            tpu_setup(in)(t);
            for (uint32_t i = 0; i < len * dim; ++i) t.ub().at(i) = in.host_bytes[i];
            REQUIRE(t.run(p.code()).halted);
            // The sequencer's tally, not the array's: through the sequencer the
            // shift is untimed at the array and charged at issue instead.
            return t.stalls().weight_load_bubble;
        };

        REQUIRE(bubble_of(true) == 0);
        REQUIRE(bubble_of(false) == 2ull * dim);   // two loads, dim cycles each
    }

    // ---- requantization is bit-exact against the specification ------------
    // Driven through the machine rather than through quant.h: accumulator values are
    // planted in a bank and an Activate requantizes them, so the comparison is the
    // hardware path's output against an independent computation.
    {
        const uint32_t dim = 8;
        const struct { i32 mult; uint32_t shift; i32 bias; } scales[] = {
            {1, 0, 0},        // identity, no rounding at all
            {1, 1, 0},        // every other value is a tie
            {1, 8, 0},
            {3, 9, 0},
            {7, 4, 0},        // large products, saturating both ends
            {1, 7, 300},      // bias big enough to move the clamp
            {5, 3, -250},
            {129, 15, 11},
        };

        for (const auto& sc : scales) {
            Config cfg = small_cfg(dim, 1);
            cfg.ub_bytes = 16 * dim * dim;

            // A dense sweep around zero, where the ties and both clamps live, plus
            // the extremes of the accumulator range.
            std::vector<i32> vals(static_cast<std::size_t>(dim) * dim);
            for (std::size_t i = 0; i < vals.size(); ++i) {
                const int k = static_cast<int>(i) - static_cast<int>(vals.size() / 2);
                vals[i] = k;
            }
            vals[0] = std::numeric_limits<i32>::min() / 256;
            vals[1] = std::numeric_limits<i32>::max() / 256;
            vals[2] = 0;

            Inputs in;
            in.acc_at   = 0;
            in.acc_vals = vals;

            tpuasm::ActArgs act;
            act.acc = 0;
            act.dst = 0;
            act.len = dim;
            act.fn   = ActFn::IDENTITY;
            act.bias = sc.bias;
            act.multiplier = sc.mult;
            act.shift = sc.shift;

            tpuasm::Program p;
            p.activate(act).halt();

            Tpu t(cfg);
            tpu_setup(in)(t);
            REQUIRE(t.run(p.code()).halted);

            for (std::size_t i = 0; i < vals.size(); ++i) {
                const i8 want = spec_requantize(vals[i], sc.bias, sc.mult, sc.shift);
                const i8 got  = t.ub().at(static_cast<UbAddr>(i));
                REQUIRE_MSG(got == want,
                            "    mult " + std::to_string(sc.mult) + " shift " +
                                std::to_string(sc.shift) + " bias " +
                                std::to_string(sc.bias) + " acc " +
                                std::to_string(vals[i]) + ": got " +
                                std::to_string(static_cast<int>(got)) + " want " +
                                std::to_string(static_cast<int>(want)) + "\n");
            }
        }

        // Ties in both signs, explicitly, and both clamps -- the classic
        // requantization bug magnets.
        REQUIRE(spec_requantize(5, 0, 1, 1) == 3);     //  2.5 -> 3
        REQUIRE(spec_requantize(-5, 0, 1, 1) == -3);   // -2.5 -> -3
        REQUIRE(spec_requantize(3, 0, 1, 1) == 2);     //  1.5 -> 2
        REQUIRE(spec_requantize(-3, 0, 1, 1) == -2);
        REQUIRE(spec_requantize(1 << 20, 0, 1, 0) == 127);
        REQUIRE(spec_requantize(-(1 << 20), 0, 1, 0) == -128);
    }

    // ---- below the ridge, time scales with bytes and not with MACs --------
    // Halving the DMA bandwidth of a memory-bound run should very nearly double it,
    // because the array is already waiting; the same change to a compute-bound run
    // should barely register.
    {
        const wl::Spec spec = wl::specs().front();   // matmul_128

        auto at_bandwidth = [&](uint32_t bw) {
            Config cfg = spec.cfg;
            cfg.dim = 16;
            cfg.ub_bytes = 64 * 1024;
            cfg.dma_bytes_per_cycle = bw;
            const Measured m = measure(wl::build(spec, cfg), cfg);
            REQUIRE(m.halted);
            return m.stats;
        };

        const stats::Stats slow   = at_bandwidth(1);
        const stats::Stats slower = at_bandwidth(1);   // same, as a determinism check
        REQUIRE(slow.cycles == slower.cycles);

        const stats::Stats half = at_bandwidth(2);
        REQUIRE(slow.below_ridge());
        REQUIRE(slow.dominant_stall_cause() == stats::Cause::DMA_BOUND);

        // The same bytes at half the rate: the DMA-bound part of the run doubles.
        REQUIRE(slow.dma_bytes == half.dma_bytes);
        REQUIRE_MSG(slow.cycles > half.cycles,
                    "    1 B/cyc: " + std::to_string(slow.cycles) + ", 2 B/cyc: " +
                        std::to_string(half.cycles) + "\n");
        const uint64_t moved_slow = slow.dma_bytes;          // 1 cycle per byte
        REQUIRE_MSG(slow.lost.dma_bound * 2 >= moved_slow,
                    "    only " + std::to_string(slow.lost.dma_bound) +
                        " cycles charged to DMA for " + std::to_string(moved_slow) +
                        " bytes\n");

        // And a well-fed configuration of the same workload is compute-bound, so the
        // ridge is telling us something about the configuration and not just about
        // the shape.
        const stats::Stats fast = at_bandwidth(64);
        REQUIRE(!fast.below_ridge());
        REQUIRE(fast.dominant_stall_cause() != stats::Cause::DMA_BOUND);
        REQUIRE(fast.cycles < slow.cycles);

        // Bandwidth does not change the arithmetic, only the schedule.
        REQUIRE(slow.macs_useful == fast.macs_useful);
    }

    // ---- more banks cannot make port contention worse ---------------------
    // The monotonicity the bank model has to have: adding ports never adds conflicts.
    {
        uint64_t prev = 0;
        bool     first = true;
        for (const uint32_t nbanks : {1u, 2u, 4u, 8u}) {
            Config cfg = wl::specs().front().cfg;
            cfg.dim = 16;
            cfg.ub_bytes = 64 * 1024;
            cfg.ub_banks = nbanks;

            const Measured m = measure(wl::build(wl::specs().front(), cfg), cfg);
            REQUIRE(m.halted);
            const uint64_t conflicts = m.stats.lost.ub_bank_conflict;
            if (!first) REQUIRE(conflicts <= prev);
            prev  = conflicts;
            first = false;
        }
        REQUIRE(prev == 0);   // eight banks: none left
    }
}

// ------------------------------------------------ @section("config_sweep") ---
SECTION("config_sweep") {
    // Every workload on all six reference configurations.
    //
    // Dataflow, skew, interlock and padding bugs are configuration-dependent in a way
    // arithmetic bugs are not: a skew off by one produces the right answer whenever
    // dim divides the shape and the wrong one whenever it does not, and a lowering
    // that assumes a full tile is only wrong on the ragged edge. One array size proves
    // little; 8x8 next to 256x256 with the same expected output proves much more.
    struct Named {
        const char* name;
        Config (*make)(Config);
    };

    const Named configs[] = {
        {"default",     [](Config c) { return c; }},
        {"8x8",         [](Config c) { c.dim = 8;   return c; }},
        {"256x256",     [](Config c) { c.dim = 256; return c; }},
        {"1-bank UB",   [](Config c) { c.ub_banks = 1; return c; }},
        {"1-deep FIFO", [](Config c) {
             c.weight_fifo_depth = 1;
             c.ddr_tile_latency  = 200;   // depth is inert unless DDR is slow
             return c;
         }},
        {"slow DMA",    [](Config c) { c.dma_bytes_per_cycle = 1; return c; }},
    };

    for (const wl::Spec& spec : wl::specs()) {
        for (const Named& n : configs) {
            Config cfg = n.make(spec.cfg);
            // Room for the largest tile any of these array sizes needs.
            cfg.ub_bytes = std::max<uint32_t>(cfg.ub_bytes, 16 * cfg.dim * cfg.dim);

            const wl::Workload w = wl::build(spec, cfg);
            const std::string  at = "    " + spec.name + " / " + n.name + ": ";

            const Measured m = measure(w, cfg);
            REQUIRE_MSG(m.halted, at + "did not halt\n");

            // The answer is the answer, on any array size.
            REQUIRE_MSG(m.got == w.expect,
                        at + tensor_mismatch(m.got, w.expect, w.out_rows, w.out_cols));

            // Every instruction accounted for, none dropped or issued twice.
            REQUIRE_MSG(m.stats.retired == w.code.size(),
                        at + "retired " + std::to_string(m.stats.retired) + " of " +
                            std::to_string(w.code.size()) + "\n");

            // And the machine agrees with the oracle on the whole visible state,
            // not just the output tile: accumulators at each Sync, buffer, host
            // memory, trap reason, retired count.
            const std::string diff =
                diff_tpu(w.code, cfg, ref_setup(tensors_of(w)), tpu_setup(tensors_of(w)),
                         host_span(w), weight_span(w));
            REQUIRE_MSG(diff.empty(), at + "\n" + diff);

            // The accounting holds on every configuration, not only the default one.
            REQUIRE_MSG(m.stats.balances(), at + "statistics do not balance\n");

            // Nothing leaked: the machine went quiet and every accumulator-bank
            // lock was released, so a following program could run immediately.
            REQUIRE_MSG(m.quiet, at + "left work in flight after Halt\n");
            REQUIRE_MSG(m.locked_banks == 0,
                        at + std::to_string(m.locked_banks) +
                            " accumulator bank(s) left locked\n");
        }
    }
}

