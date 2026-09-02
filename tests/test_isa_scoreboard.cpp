// ISA and sequencer tests: instruction encoding, decoding and program file
// formats (src/isa.h/.cpp, src/loader.h), the CLI surface of src/main.cpp,
// and the in-order scoreboarded sequencer of src/tpu.cpp -- RAW, WAR and WAW
// interlocks on the Unified Buffer, accumulator and weight-tile hazards,
// Sync, overlap across independent units, halt and trap behavior, and
// stall-cause accounting.

#include "test_support.h"

// Include the driver TU so parse_args / print_help are unit-testable without
// shelling out. Its own int main() is guarded off.
#define MINI_TPU_NO_ENTRY
#include "main.cpp"

namespace {

// argv plumbing for the CLI section.
int run_parse(std::vector<std::string> args, CliOpts& out) {
    args.insert(args.begin(), "minitpu");
    std::vector<std::vector<char>> store;
    store.reserve(args.size());
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (const std::string& s : args) {
        store.emplace_back(s.begin(), s.end());
        store.back().push_back('\0');
        argv.push_back(store.back().data());
    }
    return parse_args(static_cast<int>(argv.size()), argv.data(), out);
}

}  // namespace

// ------------------------------------------------------ @section("decode") ---
SECTION("decode") {
    // Hand-checked word 0: ACTIVATE(3) | relu(1)<<9 | avg(2)<<11 | shift 7<<13
    //                      | window 3<<21 | stride 2<<27
    Decoded a;
    a.op          = Op::ACTIVATE;
    a.act         = ActFn::RELU;
    a.pool        = Pool::AVG;
    a.shift       = 7;
    a.pool_window = 3;
    a.pool_stride = 2;
    a.acc_bank    = 2;
    a.ub_addr     = 0x1000;
    a.len         = 64;
    a.bias        = -5;
    a.multiplier  = 1234;

    const RawInst ra = encode(a);
    REQUIRE(ra.word[0] == 0x1060F203u);

    const Decoded da = decode(ra);
    REQUIRE(da.op == Op::ACTIVATE);
    REQUIRE(!da.trap);
    REQUIRE(da.act == ActFn::RELU);
    REQUIRE(da.pool == Pool::AVG);
    REQUIRE(da.shift == 7);
    REQUIRE(da.pool_window == 3);
    REQUIRE(da.pool_stride == 2);
    REQUIRE(da.acc_bank == 2);
    REQUIRE(da.ub_addr == 0x1000u);
    REQUIRE(da.len == 64u);
    REQUIRE(da.bias == -5);              // a negative immediate survives the trip
    REQUIRE(da.multiplier == 1234);

    // Every opcode round-trips its own operands.
    Decoded rh; rh.op = Op::READ_HOST;    rh.host_addr = 0xDEAD0000; rh.ub_addr = 0x40; rh.bytes = 256;
    Decoded wh; wh.op = Op::WRITE_HOST;   wh.ub_addr = 0x80; wh.host_addr = 0xBEEF0000; wh.bytes = 128;
    Decoded rw; rw.op = Op::READ_WEIGHTS; rw.ddr_addr = 0x2000; rw.tile = 5;
    Decoded mm; mm.op = Op::MATMUL;       mm.ub_addr = 0x100; mm.len = 32; mm.acc_bank = 1; mm.accumulate = true;
    Decoded ht; ht.op = Op::HALT;         ht.code = 42;

    const Decoded d_rh = decode(encode(rh));
    REQUIRE(d_rh.op == Op::READ_HOST);
    REQUIRE(d_rh.host_addr == 0xDEAD0000u);
    REQUIRE(d_rh.ub_addr == 0x40u);
    REQUIRE(d_rh.bytes == 256u);

    const Decoded d_wh = decode(encode(wh));
    REQUIRE(d_wh.op == Op::WRITE_HOST);
    REQUIRE(d_wh.ub_addr == 0x80u);
    REQUIRE(d_wh.host_addr == 0xBEEF0000u);
    REQUIRE(d_wh.bytes == 128u);

    const Decoded d_rw = decode(encode(rw));
    REQUIRE(d_rw.op == Op::READ_WEIGHTS);
    REQUIRE(d_rw.ddr_addr == 0x2000u);
    REQUIRE(d_rw.tile == 5u);

    const Decoded d_mm = decode(encode(mm));
    REQUIRE(d_mm.op == Op::MATMUL);
    REQUIRE(d_mm.ub_addr == 0x100u);
    REQUIRE(d_mm.len == 32u);
    REQUIRE(d_mm.acc_bank == 1u);
    REQUIRE(d_mm.accumulate);

    REQUIRE(decode(encode(Decoded{})).op == Op::NOP);
    REQUIRE(!decode(encode(Decoded{})).accumulate);

    Decoded sy; sy.op = Op::SYNC;
    REQUIRE(decode(encode(sy)).op == Op::SYNC);

    const Decoded d_ht = decode(encode(ht));
    REQUIRE(d_ht.op == Op::HALT);
    REQUIRE(d_ht.code == 42u);
    REQUIRE(!d_ht.trap);

    // An unknown opcode traps to Halt rather than doing something undefined.
    RawInst bad;
    bad.word[0] = 0xFFu;
    const Decoded d_bad = decode(bad);
    REQUIRE(d_bad.op == Op::HALT);
    REQUIRE(d_bad.trap);

    RawInst just_past;
    just_past.word[0] = isa::MAX_OPCODE + 1;
    REQUIRE(decode(just_past).trap);

    RawInst last_legal;
    last_legal.word[0] = isa::MAX_OPCODE;
    REQUIRE(!decode(last_legal).trap);

    // Opcode names are all distinct and none is the fallback.
    for (uint32_t o = 0; o <= isa::MAX_OPCODE; ++o) {
        REQUIRE(std::string(op_name(static_cast<Op>(o))) != "?");
    }
}

// ------------------------------------------------------ @section("loader") ---
SECTION("loader") {
    tpuasm::Program p;
    p.read_host(0x10, 0x20, 64).matmul(0x20, 8, 0).halt(7);
    const std::vector<RawInst> prog = p.code();

    // hex and raw are two spellings of the same words.
    std::ostringstream hex_out, raw_out;
    save_program_hex(hex_out, prog);
    save_program_raw(raw_out, prog);

    std::istringstream hex_in(hex_out.str()), raw_in(raw_out.str());
    const std::vector<RawInst> from_hex = load_program_hex(hex_in);
    const std::vector<RawInst> from_raw = load_program_raw(raw_in);
    REQUIRE(from_hex.size() == prog.size());
    REQUIRE(from_raw.size() == prog.size());
    bool words_match = true;
    for (std::size_t i = 0; i < prog.size(); ++i) {
        for (uint32_t w = 0; w < ISA_WORDS; ++w) {
            if (from_hex[i].word[w] != prog[i].word[w]) words_match = false;
            if (from_raw[i].word[w] != prog[i].word[w]) words_match = false;
        }
    }
    REQUIRE(words_match);

    // Comments and 0x prefixes are accepted.
    std::istringstream commented(
        "# a Sync, spelled awkwardly\n"
        "0x00000005  // opcode\n"
        "\n"
        "0\n0\n0\n0\n0\n");
    const std::vector<RawInst> one = load_program_hex(commented);
    REQUIRE(one.size() == 1);
    REQUIRE(decode(one[0]).op == Op::SYNC);

    // A partial instruction is rejected, not silently rounded down.
    std::istringstream stub("00000001\n00000002\n");
    bool threw = false;
    try { load_program_hex(stub); } catch (const std::exception&) { threw = true; }
    REQUIRE(threw);

    threw = false;
    std::istringstream odd_size("abc");
    try { load_program_raw(odd_size); } catch (const std::exception&) { threw = true; }
    REQUIRE(threw);

    threw = false;
    std::istringstream not_hex("xyz\n");
    try { load_program_hex(not_hex); } catch (const std::exception&) { threw = true; }
    REQUIRE(threw);

    // Tensors: the container, raw and hex must all land the same bytes.
    TensorBlob t;
    t.rows = 2;
    t.cols = 3;
    t.wide = false;
    t.i8v  = {1, -2, 3, -4, 127, -128};

    std::ostringstream mtpu;
    save_tensor_mtpu(mtpu, t);
    std::istringstream mtpu_in(mtpu.str());
    const TensorBlob back = load_tensor_mtpu(mtpu_in);
    REQUIRE(back.rows == 2);
    REQUIRE(back.cols == 3);
    REQUIRE(!back.wide);
    REQUIRE(back.i8v == t.i8v);

    const std::string raw_bytes(reinterpret_cast<const char*>(t.i8v.data()), t.i8v.size());
    std::istringstream raw_t(raw_bytes);
    REQUIRE(load_tensor_raw(raw_t, 2, 3, false).i8v == t.i8v);

    std::istringstream hex_t("01\nFE\n03\nFC\n7F\n80\n");
    REQUIRE(load_tensor_hex(hex_t, 2, 3, false).i8v == t.i8v);

    // int32 payloads round-trip, extremes included.
    TensorBlob w;
    w.rows = 1;
    w.cols = 3;
    w.wide = true;
    w.i32v = {-1, 2147483647, -2147483647 - 1};
    std::ostringstream w_out;
    save_tensor_mtpu(w_out, w);
    std::istringstream w_in(w_out.str());
    const TensorBlob w_back = load_tensor_mtpu(w_in);
    REQUIRE(w_back.wide);
    REQUIRE(w_back.i32v == w.i32v);

    // A blob is addressed row-major through a view.
    REQUIRE(back.i8_view().at(0, 1) == -2);
    REQUIRE(back.i8_view().at(1, 2) == -128);

    // Bad magic, wrong version and short payloads are all errors.
    threw = false;
    std::istringstream junk("NOPE................");
    try { load_tensor_mtpu(junk); } catch (const std::exception&) { threw = true; }
    REQUIRE(threw);

    threw = false;
    std::string truncated = mtpu.str();
    truncated.resize(truncated.size() - 2);
    std::istringstream short_in(truncated);
    try { load_tensor_mtpu(short_in); } catch (const std::exception&) { threw = true; }
    REQUIRE(threw);
}

// --------------------------------------------------------- @section("cli") ---
SECTION("cli") {
    // The parser reports errors on stderr, so a few "minitpu: ..." lines below
    // are expected output, not failures.
    {
        CliOpts o;
        REQUIRE(run_parse({"--prog", "p.hex", "--dump"}, o) == 0);
        REQUIRE(o.prog_hex_path == "p.hex");
        REQUIRE(o.dump);
        REQUIRE(!o.trace);
        REQUIRE(!o.show_help);
    }
    {
        CliOpts o;
        REQUIRE(run_parse({"--help"}, o) == 0);
        REQUIRE(o.show_help);
    }
    {
        CliOpts o;
        REQUIRE(run_parse({"-h"}, o) == 0);
        REQUIRE(o.show_help);
    }

    // Every Config knob is reachable by flag, in both spellings.
    REQUIRE(NUM_KNOBS == 8);
    for (const auto& k : KNOBS) {
        CliOpts spaced, joined;
        REQUIRE(run_parse({"--prog", "p.hex", k.flag, "123"}, spaced) == 0);
        REQUIRE(spaced.cfg.*(k.member) == 123u);
        REQUIRE(run_parse({"--prog", "p.hex", std::string(k.flag) + "=456"}, joined) == 0);
        REQUIRE(joined.cfg.*(k.member) == 456u);
    }

    // 0x values are accepted, since addresses and sizes are natural in hex.
    {
        CliOpts o;
        REQUIRE(run_parse({"--prog", "p.hex", "--ub", "0x1000"}, o) == 0);
        REQUIRE(o.cfg.ub_bytes == 0x1000u);
    }

    // The one boolean knob has both spellings, and the default is on.
    {
        CliOpts o;
        REQUIRE(run_parse({"--prog", "p.hex"}, o) == 0);
        REQUIRE(o.cfg.double_buffer);
        CliOpts off;
        REQUIRE(run_parse({"--prog", "p.hex", "--no-double-buffer"}, off) == 0);
        REQUIRE(!off.cfg.double_buffer);
        CliOpts on;
        REQUIRE(run_parse({"--prog", "p.hex", "--no-double-buffer", "--double-buffer"}, on) == 0);
        REQUIRE(on.cfg.double_buffer);
    }

    // Rejections.
    {
        CliOpts o;
        REQUIRE(run_parse({"--nope"}, o) != 0);
    }
    {
        CliOpts o;
        REQUIRE(run_parse({"--dim"}, o) != 0);           // flag with no value
    }
    {
        CliOpts o;
        REQUIRE(run_parse({"--dim", "twelve"}, o) != 0);  // value that is not a number
    }
    {
        CliOpts o;
        REQUIRE(run_parse({"stray.hex"}, o) != 0);        // no positional arguments
    }
}

// ------------------------------------------------------ @section("tpuasm") ---
SECTION("tpuasm") {
    // The builder is the only thing the later tests use to make programs, so it
    // has to encode exactly what it claims to.
    tpuasm::Program p;
    p.read_host(0x100, 0x200, 64)
     .read_weights(0x300, 2)
     .matmul(0x200, 8, 1, /*accumulate=*/true)
     .write_host(0x400, 0x500, 32)
     .sync()
     .nop()
     .halt(9);

    const std::vector<RawInst> code = p.code();
    REQUIRE(code.size() == 7);
    REQUIRE(p.size() == 7);

    const Decoded d0 = decode(code[0]);
    REQUIRE(d0.op == Op::READ_HOST);
    REQUIRE(d0.host_addr == 0x100u);
    REQUIRE(d0.ub_addr == 0x200u);
    REQUIRE(d0.bytes == 64u);

    const Decoded d1 = decode(code[1]);
    REQUIRE(d1.op == Op::READ_WEIGHTS);
    REQUIRE(d1.ddr_addr == 0x300u);
    REQUIRE(d1.tile == 2u);

    const Decoded d2 = decode(code[2]);
    REQUIRE(d2.op == Op::MATMUL);
    REQUIRE(d2.ub_addr == 0x200u);
    REQUIRE(d2.len == 8u);
    REQUIRE(d2.acc_bank == 1u);
    REQUIRE(d2.accumulate);

    const Decoded d3 = decode(code[3]);
    REQUIRE(d3.op == Op::WRITE_HOST);
    REQUIRE(d3.ub_addr == 0x400u);
    REQUIRE(d3.host_addr == 0x500u);
    REQUIRE(d3.bytes == 32u);

    REQUIRE(decode(code[4]).op == Op::SYNC);
    REQUIRE(decode(code[5]).op == Op::NOP);
    REQUIRE(decode(code[6]).op == Op::HALT);
    REQUIRE(decode(code[6]).code == 9u);

    // The full Activate form carries every operand through.
    tpuasm::ActArgs a;
    a.acc         = 1;
    a.dst         = 0x800;
    a.len         = 16;
    a.fn          = ActFn::RELU6;
    a.bias        = -3;
    a.multiplier  = 2000;
    a.shift       = 11;
    a.pool        = Pool::MAX;
    a.pool_window = 2;
    a.pool_stride = 2;

    tpuasm::Program q;
    q.activate(a);
    const Decoded da = decode(q.code()[0]);
    REQUIRE(da.op == Op::ACTIVATE);
    REQUIRE(da.acc_bank == 1u);
    REQUIRE(da.ub_addr == 0x800u);
    REQUIRE(da.len == 16u);
    REQUIRE(da.act == ActFn::RELU6);
    REQUIRE(da.bias == -3);
    REQUIRE(da.multiplier == 2000);
    REQUIRE(da.shift == 11u);
    REQUIRE(da.pool == Pool::MAX);
    REQUIRE(da.pool_window == 2u);
    REQUIRE(da.pool_stride == 2u);

    // The short form leaves bias and pooling alone.
    tpuasm::Program s;
    s.activate(0, 0x40, 4, ActFn::RELU, 3, 5);
    const Decoded ds = decode(s.code()[0]);
    REQUIRE(ds.act == ActFn::RELU);
    REQUIRE(ds.bias == 0);
    REQUIRE(ds.multiplier == 3);
    REQUIRE(ds.shift == 5u);
    REQUIRE(ds.pool == Pool::NONE);

    // Regions are handed out in order, aligned, and never overlap.
    tpuasm::UbAlloc ua(0x40, 4);
    const tpuasm::Region r1 = ua.tile(2, 3);      // 6 bytes, rounded to 8
    const tpuasm::Region r2 = ua.tile(4, 4);      // 16 bytes
    REQUIRE(r1.addr == 0x40u);
    REQUIRE(r1.bytes == 6u);
    REQUIRE(r2.addr == 0x48u);
    REQUIRE(r2.bytes == 16u);
    REQUIRE(ua.next() == 0x58u);
    REQUIRE(r1.addr + r1.bytes <= r2.addr);

    // A Region stands in for its own address.
    tpuasm::Program u;
    u.matmul(r2, 4, 0);
    REQUIRE(decode(u.code()[0]).ub_addr == r2.addr);

    // The raw escape hatch is what the illegal-encoding tests need.
    RawInst bad;
    bad.word[0] = 0xFFu;
    tpuasm::Program b;
    b.raw(bad);
    REQUIRE(decode(b.code()[0]).trap);
}

// --------------------------------------------------- @section("sequencer") ---
SECTION("sequencer") {
    const uint32_t dim = 4, len = 4;
    Config cfg = small_cfg(dim, 2);
    cfg.ub_bytes = 1024;
    cfg.dma_bytes_per_cycle = 8;
    cfg.ddr_tile_latency = 5;

    const Inputs in = random_inputs(555, dim, len);

    tpuasm::UbAlloc ua(0);
    const tpuasm::Region acts = ua.tile(dim, dim);
    const tpuasm::Region out  = ua.tile(dim, dim);

    tpuasm::Program p;
    p.read_host(in.host_at, acts, len * dim)
     .read_weights(0)
     .matmul(acts, len, 0)
     .activate(0, out, len, ActFn::RELU, 1, 3)
     .write_host(out, 0x400, len * dim)
     .halt(5);

    // A straight-line program runs to Halt and agrees with the oracle on every
    // comparand. This is the first time the timed machine is checked against it.
    const std::string diff = diff_tpu(p.code(), cfg, ref_setup(in), tpu_setup(in));
    REQUIRE_MSG(diff.empty(), diff);

    // The retired count equals the instruction count, and the exit code comes
    // back from Halt.
    {
        Tpu t(cfg);
        tpu_setup(in)(t);
        const TpuResult r = t.run(p.code());
        REQUIRE(r.halted);
        REQUIRE(!r.trapped);
        REQUIRE(!r.budget);
        REQUIRE(r.retired == p.code().size());
        REQUIRE(r.exit_code == 5);
        REQUIRE(r.pc == p.code().size());
        REQUIRE(t.quiet());
        REQUIRE(r.cycles > 0);
    }

    // Nothing but Halt.
    {
        Tpu t(cfg);
        tpuasm::Program h;
        h.halt(9);
        const TpuResult r = t.run(h.code());
        REQUIRE(r.halted);
        REQUIRE(r.exit_code == 9);
        REQUIRE(r.retired == 1);
    }

    // One issue per cycle: a run of NOPs costs a cycle each.
    {
        Tpu t(cfg);
        tpuasm::Program n;
        for (int i = 0; i < 10; ++i) n.nop();
        n.halt();
        const TpuResult r = t.run(n.code());
        REQUIRE(r.halted);
        REQUIRE(r.retired == 11);
        REQUIRE(r.cycles == 11);
    }

    // ---- traps agree with the oracle, reason included ---------------------
    {
        struct Case {
            const char* name;
            std::vector<RawInst> code;
        };
        std::vector<Case> cases;
        {
            tpuasm::Program q;
            q.matmul(0, dim + 1, 0).halt();
            cases.push_back({"len exceeds bank", q.code()});
        }
        {
            tpuasm::Program q;
            q.matmul(0, len, cfg.acc_banks).halt();
            cases.push_back({"bank out of range", q.code()});
        }
        {
            tpuasm::Program q;
            q.activate(cfg.acc_banks, 0, len, ActFn::RELU).halt();
            cases.push_back({"activate bank", q.code()});
        }
        {
            tpuasm::Program q;
            q.read_host(0xFFFF0000, 0, 64).halt();
            cases.push_back({"host range", q.code()});
        }
        {
            tpuasm::Program q;
            q.nop();
            cases.push_back({"ran off the end", q.code()});
        }
        {
            RawInst bad;
            bad.word[0] = 0xFFu;
            tpuasm::Program q;
            q.raw(bad);
            cases.push_back({"illegal opcode", q.code()});
        }

        for (const Case& c : cases) {
            ref::Machine m(cfg);
            Tpu t(cfg);
            const ref::Result rr = ref::run(m, c.code);
            const TpuResult   tr = t.run(c.code);
            REQUIRE(rr.trapped);
            REQUIRE_MSG(tr.trapped, std::string("    ") + c.name + " did not trap\n");
            REQUIRE_MSG(tr.trap_reason == rr.trap_reason,
                        std::string("    ") + c.name + ": tpu=\"" + tr.trap_reason +
                            "\" ref=\"" + rr.trap_reason + "\"\n");
            REQUIRE(tr.retired == rr.retired);
        }
    }

    // A runaway program hits the cycle budget instead of spinning forever.
    {
        Tpu t(cfg);
        tpuasm::Program q;
        for (int i = 0; i < 100; ++i) q.nop();
        q.halt();
        TpuOptions opts;
        opts.max_cycles = 10;
        const TpuResult r = t.run(q.code(), opts);
        REQUIRE(r.budget);
        REQUIRE(!r.halted);
        REQUIRE(!r.trapped);
    }
}

// -------------------------------------------------- @section("scoreboard") ---
SECTION("scoreboard") {
    const uint32_t dim = 4, len = 4;
    Config cfg = small_cfg(dim, 2);
    cfg.ub_bytes = 1024;
    cfg.dma_bytes_per_cycle = 1;      // slow, so the DMA is unmistakably long
    cfg.ddr_tile_latency = 4;

    const Inputs in = random_inputs(777, dim, len);
    const uint32_t bytes = len * dim;

    // ---- a MatMul reading a region a DMA is still filling must wait --------
    {
        tpuasm::Program p;
        p.read_host(in.host_at, 0, bytes)     // 16 bytes at 1/cycle = 16 cycles
         .read_weights(0)
         .matmul(0, len, 0)                   // reads exactly what the DMA writes
         .halt();

        Tpu t(cfg);
        tpu_setup(in)(t);
        const TpuResult r = t.run(p.code());
        REQUIRE(r.halted);

        // The MatMul could not have issued before the DMA retired, so the RAW
        // counter has to have fired.
        REQUIRE(t.stalls().ub_raw > 0);

        // And the answer is right, which is the point of the interlock.
        const std::string diff = diff_tpu(p.code(), cfg, ref_setup(in), tpu_setup(in));
        REQUIRE_MSG(diff.empty(), diff);
    }

    // A MatMul reading a region the DMA is *not* filling does not wait on it.
    {
        tpuasm::Program p;
        p.read_host(in.host_at, 512, bytes)   // lands somewhere else entirely
         .read_weights(0)
         .matmul(0, len, 0)
         .halt();

        Tpu t(cfg);
        tpu_setup(in)(t);
        const TpuResult r = t.run(p.code());
        REQUIRE(r.halted);
        REQUIRE(t.stalls().ub_raw == 0);
    }

    // ---- an Activate reading an accumulator waits for its MatMul -----------
    {
        tpuasm::Program p;
        p.read_weights(0)
         .matmul(0, len, 0)
         .activate(0, 512, len, ActFn::RELU, 1, 0)
         .halt();

        Tpu t(cfg);
        tpu_setup(in)(t);
        const TpuResult r = t.run(p.code());
        REQUIRE(r.halted);
        REQUIRE(t.stalls().accum_hazard > 0);

        const std::string diff = diff_tpu(p.code(), cfg, ref_setup(in), tpu_setup(in));
        REQUIRE_MSG(diff.empty(), diff);
    }

    // An Activate on a different bank has nothing to wait for.
    {
        tpuasm::Program p;
        p.read_weights(0)
         .matmul(0, len, 0)
         .activate(1, 512, len, ActFn::RELU, 1, 0)
         .halt();

        Tpu t(cfg);
        tpu_setup(in)(t);
        REQUIRE(t.run(p.code()).halted);
        REQUIRE(t.stalls().accum_hazard == 0);
    }

    // ---- write-after-read: a DMA must not overwrite what a MatMul is still
    // streaming out of the buffer.
    //
    // This can only ever be a timing property. An instruction reads its inputs at
    // issue, so by the time a later writer could commit, the reader already has what
    // it needed and no answer can change. A real array streams activations over many
    // cycles and would be corrupted, so the stall belongs in the model, but the
    // stall counter is the only thing that can witness it.
    {
        tpuasm::Program p;
        p.read_weights(0)
         .matmul(0, len, 0)
         .read_host(in.host_at, 0, bytes)     // same region the MatMul reads
         .halt();

        Tpu t(cfg);
        tpu_setup(in)(t);
        REQUIRE(t.run(p.code()).halted);
        REQUIRE(t.stalls().ub_war > 0);
    }

    // Two DMAs to one region cannot overlap, though the single engine would have
    // serialized them regardless of the region.
    {
        tpuasm::Program p;
        p.read_host(in.host_at, 0, bytes)
         .read_host(in.host_at, 0, bytes)
         .halt();

        Tpu t(cfg);
        tpu_setup(in)(t);
        REQUIRE(t.run(p.code()).halted);
        REQUIRE(t.stalls().unit_busy > 0 || t.stalls().ub_waw > 0);
    }

    // ---- write-after-write across two different units -----------------------
    // The case needing the WAW check on its own merits: an Activate and a DMA live
    // in different units, so nothing structural stops them overlapping. The Activate
    // is the slower of the two, so overlapping would let the DMA commit first and
    // the Activate overwrite it, leaving the region holding whichever finished last
    // rather than what the program said.
    {
        tpuasm::Program p;
        p.read_weights(0)
         .matmul(0, len, 0)
         .activate(0, 512, len, ActFn::RELU, 1, 0)   // writes [512, 528)
         .read_host(in.host_at, 512, bytes)          // same region, and later
         .halt();

        Tpu t(cfg);
        tpu_setup(in)(t);
        REQUIRE(t.run(p.code()).halted);
        REQUIRE(t.stalls().ub_waw > 0);

        const std::string diff = diff_tpu(p.code(), cfg, ref_setup(in), tpu_setup(in));
        REQUIRE_MSG(diff.empty(), diff);
    }

    // Accumulating into a bank waits for the matmul already producing into it,
    // and the K-tiled result still matches the oracle.
    {
        const Inputs two = random_inputs(31, dim, len, 2);
        tpuasm::Program p;
        p.read_weights(0).matmul(0, len, 0)
         .read_weights(static_cast<uint32_t>(dim) * dim).matmul(0, len, 0, /*accumulate=*/true)
         .halt();

        const std::string diff = diff_tpu(p.code(), cfg, ref_setup(two), tpu_setup(two));
        REQUIRE_MSG(diff.empty(), diff);
    }
}

// ----------------------------------------------------- @section("overlap") ---
SECTION("overlap") {
    // Independent work in different units runs concurrently; a dependent chain
    // does not. Same instructions in both programs, so the only difference is
    // whether the operands collide.
    const uint32_t dim = 4, len = 4;
    Config cfg = small_cfg(dim, 4);
    cfg.ub_bytes = 2048;
    cfg.dma_bytes_per_cycle = 1;      // 16-byte transfer = 16 cycles
    cfg.ddr_tile_latency = 4;
    cfg.act_pipeline_depth = 4;
    cfg.double_buffer = true;

    const uint32_t bytes   = len * dim;
    const uint64_t dma_dur = bytes;                       // 16
    const uint64_t mm_dur  = len + 2ull * dim - 1ull;     // 11
    const uint64_t act_dur = static_cast<uint64_t>(len) * dim + cfg.act_pipeline_depth;  // 20

    const Inputs in = random_inputs(999, dim, len);

    // Independent: the DMA writes a region nobody reads, the MatMul reads a
    // different region into bank 0, the Activate reads bank 1.
    tpuasm::Program indep;
    indep.read_weights(0)
         .matmul(0, len, 0)             // reads UB [0, 16)
         .read_host(in.host_at, 1024, bytes)
         .activate(1, 512, len, ActFn::RELU, 1, 0)
         .halt();

    // Dependent: the DMA fills what the MatMul reads, and the Activate reads the
    // bank the MatMul writes.
    tpuasm::Program dep;
    dep.read_weights(0)
       .read_host(in.host_at, 0, bytes)
       .matmul(0, len, 0)
       .activate(0, 512, len, ActFn::RELU, 1, 0)
       .halt();

    Tpu ti(cfg), td(cfg);
    tpu_setup(in)(ti);
    tpu_setup(in)(td);
    const TpuResult ri = ti.run(indep.code());
    const TpuResult rd = td.run(dep.code());
    REQUIRE(ri.halted);
    REQUIRE(rd.halted);

    // The dependent chain pays the sum of its stages; the independent one pays
    // roughly the longest.
    const uint64_t sum = dma_dur + mm_dur + act_dur;
    REQUIRE(ri.cycles < sum);
    REQUIRE(rd.cycles >= sum);
    REQUIRE(ri.cycles < rd.cycles);

    // More precisely: the independent version finishes within a few issue slots
    // of its longest stage, since the three run side by side.
    const uint64_t longest = std::max(dma_dur, std::max(mm_dur, act_dur));
    REQUIRE(ri.cycles <= longest + cfg.ddr_tile_latency + indep.code().size());

    // The stall counters say which one was which.
    REQUIRE(ti.stalls().ub_raw == 0);
    REQUIRE(ti.stalls().accum_hazard == 0);
    REQUIRE(td.stalls().ub_raw > 0);
    REQUIRE(td.stalls().accum_hazard > 0);

    // Overlapping must not change the answer: both programs match the oracle.
    {
        const std::string di = diff_tpu(indep.code(), cfg, ref_setup(in), tpu_setup(in));
        REQUIRE_MSG(di.empty(), di);
        const std::string dd = diff_tpu(dep.code(), cfg, ref_setup(in), tpu_setup(in));
        REQUIRE_MSG(dd.empty(), dd);
    }

    // Two DMAs cannot overlap however independent their regions, because there is
    // one engine: a structural limit, not a data one.
    {
        tpuasm::Program p;
        p.read_host(in.host_at, 0, bytes)
         .read_host(in.host_at, 1024, bytes)
         .halt();
        Tpu t(cfg);
        tpu_setup(in)(t);
        const TpuResult r = t.run(p.code());
        REQUIRE(r.halted);
        REQUIRE(t.stalls().unit_busy > 0);
        REQUIRE(r.cycles >= 2 * dma_dur);
    }
}

// ---------------------------------------------- @section("weight_overlap") ---
SECTION("weight_overlap") {
    // Through the full sequencer: the second tile's weight load hides under the
    // first tile's matmul when there is a shadow plane, and is fully exposed
    // when there is not.
    const uint32_t dim = 4, len = 4;
    const uint32_t latency = 6;

    const Inputs in = random_inputs(2024, dim, len, 2);
    const uint32_t tile_bytes = dim * dim;

    tpuasm::Program p;
    p.read_weights(0)
     .matmul(0, len, 0)
     .read_weights(tile_bytes)
     .matmul(0, len, 1)
     .halt();

    auto measure = [&](bool double_buffer) {
        Config cfg = small_cfg(dim, 2);
        cfg.ub_bytes = 1024;
        cfg.ddr_tile_latency = latency;
        cfg.double_buffer = double_buffer;

        Tpu t(cfg);
        tpu_setup(in)(t);
        // Activations come from the buffer directly, so the timing is only about
        // weights and compute.
        for (uint32_t i = 0; i < len * dim; ++i) {
            t.ub().at(i) = in.host_bytes[i];
        }
        const TpuResult r = t.run(p.code());
        REQUIRE(r.halted);
        return std::make_pair(r.cycles, t.stalls());
    };

    const auto db   = measure(true);
    const auto nodb = measure(false);

    const uint64_t mm  = len + 2ull * dim - 1ull;   // 11

    // With a shadow plane no cycle is spent shifting weights into the array.
    REQUIRE(db.second.weight_load_bubble == 0);
    // Without one, each of the two loads costs dim exposed cycles.
    REQUIRE(nodb.second.weight_load_bubble == 2ull * dim);

    // Only one DDR latency is ever exposed. The prefetcher requested both tiles
    // before the program issued anything, so the second was already on its way while
    // the first was being waited for; what remains is the first tile, which has
    // nothing to hide behind.
    REQUIRE_MSG(db.first == latency + 2 * mm + 2,
                "    double-buffered: " + std::to_string(db.first) + " cycles, want " +
                    std::to_string(latency + 2 * mm + 2) + "\n");

    // Without a shadow plane each load also costs dim exposed cycles shifting into
    // the array, and the second cannot start until the first matmul has finished
    // with the weight plane it is about to overwrite.
    REQUIRE_MSG(nodb.first == latency + 2 * dim + 2 * mm + 1,
                "    single-plane:    " + std::to_string(nodb.first) + " cycles, want " +
                    std::to_string(latency + 2 * dim + 2 * mm + 1) + "\n");
    REQUIRE(nodb.first > db.first);

    // The DDR latency shows up as the array waiting on the FIFO, once.
    REQUIRE(db.second.weight_fifo_empty == latency);
    REQUIRE(nodb.second.weight_fifo_empty == latency);

    // The array itself confirms it: with a shadow plane it switches planes on
    // every matmul, and it never stood idle for a load.
    {
        Config cfg = small_cfg(dim, 2);
        cfg.ub_bytes = 1024;
        cfg.ddr_tile_latency = latency;
        cfg.double_buffer = true;
        Tpu t(cfg);
        tpu_setup(in)(t);
        for (uint32_t i = 0; i < len * dim; ++i) t.ub().at(i) = in.host_bytes[i];
        REQUIRE(t.run(p.code()).halted);
        REQUIRE(t.mxu().stats().plane_switches == 2);
        REQUIRE(t.mxu().stats().weight_load_bubble == 0);
        // The wait is on DDR, not on the weight plane: with a shadow plane to load into,
        // nothing ever blocks on the resident tile itself.
        REQUIRE(t.stalls().weight_fifo_empty > 0);
        REQUIRE(t.stalls().weight_stall == 0);
    }

    // ---- FIFO depth is what decides how much latency is hidden --------------
    // Four tiles, so a deep FIFO can have them all in flight at once and pay the DDR
    // latency once, while a 1-deep FIFO cannot start a fetch until the previous tile
    // has been consumed and pays it four times over.
    //
    // The latency is deliberately longer than a matmul. A shallow FIFO still starts
    // its next fetch while the array works, so with a short latency the wait hides
    // behind the compute and depth costs nothing measurable; depth only buys
    // something once DDR is slower than the work available to cover it.
    {
        const uint32_t slow_ddr = 40;      // longer than mm
        REQUIRE(slow_ddr > mm);

        tpuasm::Program four;
        four.read_weights(0).matmul(0, len, 0)
            .read_weights(tile_bytes).matmul(0, len, 1)
            .read_weights(2 * tile_bytes).matmul(0, len, 0)
            .read_weights(3 * tile_bytes).matmul(0, len, 1)
            .halt();

        auto run_depth = [&](uint32_t depth) {
            Config cfg = small_cfg(dim, 2);
            cfg.ub_bytes = 1024;
            cfg.ddr_tile_latency = slow_ddr;
            cfg.double_buffer = true;
            cfg.weight_fifo_depth = depth;

            Inputs many = in;
            many.weights.assign(4 * tile_bytes, 0);
            Lcg rng(4242);
            for (i8& v : many.weights) v = rng.byte();

            Tpu t(cfg);
            tpu_setup(many)(t);
            for (uint32_t i = 0; i < len * dim; ++i) t.ub().at(i) = in.host_bytes[i];
            const TpuResult r = t.run(four.code());
            REQUIRE(r.halted);
            return std::make_pair(r.cycles, t.stalls().weight_fifo_empty);
        };

        const auto deep    = run_depth(4);
        const auto shallow = run_depth(1);

        // Deep: one exposed latency for the first tile, the rest hidden.
        REQUIRE_MSG(deep.second == slow_ddr,
                    "    depth 4 waited " + std::to_string(deep.second) +
                        " cycles on DDR, want " + std::to_string(slow_ddr) + "\n");
        // Shallow: the FIFO holds one tile, so every load waits on DDR again, and
        // the waits are too long to hide behind the matmuls.
        REQUIRE(shallow.second > deep.second);
        REQUIRE_MSG(shallow.first > deep.first,
                    "    depth 1: " + std::to_string(shallow.first) + " cycles, depth 4: " +
                        std::to_string(deep.first) + "\n");

        // And the depth changes only the schedule, not the answer.
        Config cfg = small_cfg(dim, 2);
        cfg.ub_bytes = 1024;
        cfg.ddr_tile_latency = slow_ddr;
        cfg.weight_fifo_depth = 1;

        Inputs many = in;
        many.weights.assign(4 * tile_bytes, 0);
        Lcg rng(4242);
        for (i8& v : many.weights) v = rng.byte();
        many.ub_at = 0;
        many.ub_bytes.assign(in.host_bytes.begin(), in.host_bytes.end());

        const std::string diff =
            diff_tpu(four.code(), cfg, ref_setup(many), tpu_setup(many));
        REQUIRE_MSG(diff.empty(), diff);
    }

    // Faster must still mean identical: both configurations agree with the oracle.
    for (const bool db_on : {true, false}) {
        Config cfg = small_cfg(dim, 2);
        cfg.ub_bytes = 1024;
        cfg.ddr_tile_latency = latency;
        cfg.double_buffer = db_on;

        Inputs seeded = in;
        seeded.ub_at = 0;
        seeded.ub_bytes.assign(in.host_bytes.begin(), in.host_bytes.end());

        const std::string diff =
            diff_tpu(p.code(), cfg, ref_setup(seeded), tpu_setup(seeded));
        REQUIRE_MSG(diff.empty(), diff);
    }
}

// -------------------------------------------------------- @section("sync") ---
SECTION("sync") {
    const uint32_t dim = 4, len = 4;
    Config cfg = small_cfg(dim, 2);
    cfg.ub_bytes = 1024;
    cfg.dma_bytes_per_cycle = 2;
    cfg.ddr_tile_latency = 5;

    const Inputs in = random_inputs(4242, dim, len, 2);
    const uint32_t tile_bytes = dim * dim;

    // A barrier between two K-tiles: the snapshot at each Sync must match the
    // oracle, which is what makes a timing bug bisectable to the interval between
    // two barriers rather than merely visible at Halt.
    tpuasm::Program p;
    p.read_host(in.host_at, 0, len * dim)
     .read_weights(0)
     .matmul(0, len, 0)
     .sync()
     .read_weights(tile_bytes)
     .matmul(0, len, 0, /*accumulate=*/true)
     .sync()
     .activate(0, 512, len, ActFn::RELU, 1, 2)
     .write_host(512, 0x400, len * dim)
     .sync()
     .halt(2);

    const std::string diff = diff_tpu(p.code(), cfg, ref_setup(in), tpu_setup(in));
    REQUIRE_MSG(diff.empty(), diff);

    {
        Tpu t(cfg);
        tpu_setup(in)(t);
        const TpuResult r = t.run(p.code());
        REQUIRE(r.halted);
        REQUIRE(r.syncs.size() == 3);
        REQUIRE(r.retired == p.code().size());

        // A barrier only issues once the machine is quiet, so it has to have
        // waited for the work in front of it.
        REQUIRE(t.stalls().drain > 0);

        // The snapshots are distinct: the second K-tile really did add to the
        // first, so the barriers are observing progress rather than a static
        // bank.
        REQUIRE(r.syncs[0].acc != r.syncs[1].acc);

        // Activate does not touch the accumulators, so the last two agree.
        REQUIRE(r.syncs[1].acc == r.syncs[2].acc);
    }

    // Every Phase-2 workload shape, with and without barriers, and under
    // configurations that change the schedule but must not change the result.
    {
        tpuasm::Program bare;
        bare.read_host(in.host_at, 0, len * dim)
            .read_weights(0)
            .matmul(0, len, 0)
            .read_weights(tile_bytes)
            .matmul(0, len, 0, true)
            .activate(0, 512, len, ActFn::RELU, 1, 2)
            .write_host(512, 0x400, len * dim)
            .halt(2);

        for (const uint32_t bw : {1u, 4u, 64u}) {
            for (const bool db : {true, false}) {
                for (const uint32_t lat : {0u, 3u, 9u}) {
                    Config c = cfg;
                    c.dma_bytes_per_cycle = bw;
                    c.double_buffer = db;
                    c.ddr_tile_latency = lat;

                    const std::string d1 = diff_tpu(p.code(), c, ref_setup(in), tpu_setup(in));
                    REQUIRE_MSG(d1.empty(), d1);
                    const std::string d2 =
                        diff_tpu(bare.code(), c, ref_setup(in), tpu_setup(in));
                    REQUIRE_MSG(d2.empty(), d2);
                }
            }
        }
    }

    // A Sync with nothing in flight still snapshots, and back-to-back barriers
    // agree with each other.
    {
        Tpu t(cfg);
        tpuasm::Program q;
        q.sync().sync().halt();
        const TpuResult r = t.run(q.code());
        REQUIRE(r.halted);
        REQUIRE(r.syncs.size() == 2);
        REQUIRE(r.syncs[0].acc == r.syncs[1].acc);
        REQUIRE(r.retired == 3);
    }
}

// --------------------------------------------- @section("activate_timing") ---
SECTION("activate_timing") {
    const uint32_t dim = 4;

    // ---- M + act_pipeline_depth, over several shapes and depths ------------
    for (const uint32_t depth : {0u, 1u, 4u, 16u}) {
        Config cfg = small_cfg(dim, 4);
        cfg.ub_bytes = 1024;
        cfg.act_pipeline_depth = depth;

        for (uint32_t len = 1; len <= dim; ++len) {
            Tpu t(cfg);
            tpuasm::Program p;
            p.activate(0, 512, len, ActFn::RELU, 1, 0).halt();
            const TpuResult r = t.run(p.code());
            REQUIRE(r.halted);

            // The Activate issues at cycle 0 and occupies its unit for
            // len*dim + depth cycles; Halt waits for it and spends one more.
            const uint64_t expect = static_cast<uint64_t>(len) * dim + depth + 1;
            REQUIRE_MSG(r.cycles == expect,
                        "    depth " + std::to_string(depth) + " len " + std::to_string(len) +
                            ": got " + std::to_string(r.cycles) + " want " +
                            std::to_string(expect) + "\n");
        }
    }

    // Pooling does not make an Activate cheaper: the cost is one cycle per
    // accumulator element read, and every element is still read before the window
    // reduction sees it.
    {
        Config cfg = small_cfg(dim, 4);
        cfg.ub_bytes = 1024;
        cfg.act_pipeline_depth = 4;

        auto time_it = [&](Pool mode) {
            Tpu t(cfg);
            tpuasm::ActArgs a;
            a.acc = 0; a.dst = 512; a.len = dim;
            a.fn = ActFn::IDENTITY; a.multiplier = 1; a.shift = 0;
            a.pool = mode; a.pool_window = 2; a.pool_stride = 2;
            tpuasm::Program p;
            p.activate(a).halt();
            const TpuResult r = t.run(p.code());
            REQUIRE(r.halted);
            return r.cycles;
        };
        REQUIRE(time_it(Pool::NONE) == time_it(Pool::MAX));
        REQUIRE(time_it(Pool::NONE) == dim * dim + 4 + 1);
    }

    // ---- an Activate overlaps a following MatMul on an independent bank -----
    {
        Config cfg = small_cfg(dim, 4);
        cfg.ub_bytes = 2048;
        cfg.dma_bytes_per_cycle = 16;
        cfg.ddr_tile_latency = 2;
        cfg.act_pipeline_depth = 8;
        cfg.double_buffer = true;

        const uint32_t len     = dim;
        const uint64_t act_dur = static_cast<uint64_t>(len) * dim + cfg.act_pipeline_depth;
        const uint64_t mm_dur  = len + 2ull * dim - 1ull;

        // The premise of the comparison: the activation is the longer of the two, so
        // an overlapping matmul has room to finish inside it.
        REQUIRE(act_dur > mm_dur);

        const Inputs in = random_inputs(6200, dim, len);

        // Independent: the Activate drains bank 0 while the next matmul fills
        // bank 1 from a different buffer region.
        tpuasm::Program indep;
        indep.read_weights(0)
             .matmul(0, len, 0)
             .activate(0, 512, len, ActFn::RELU, 1, 4)   // reads bank 0, writes [512,528)
             .matmul(256, len, 1)                        // reads [256,272), writes bank 1
             .halt();

        // Serializing: the second matmul accumulates into the very bank the
        // Activate is draining.
        tpuasm::Program serial;
        serial.read_weights(0)
              .matmul(0, len, 0)
              .activate(0, 512, len, ActFn::RELU, 1, 4)
              .matmul(256, len, 0, /*accumulate=*/true)
              .halt();

        // The same program with the second matmul removed, as the baseline the
        // overlapped run is measured against.
        tpuasm::Program alone;
        alone.read_weights(0)
             .matmul(0, len, 0)
             .activate(0, 512, len, ActFn::RELU, 1, 4)
             .halt();

        // Both matmuls read a buffer region, so seed the UB directly rather than
        // spending DMA cycles that would blur the comparison.
        // Both matmuls read a region, at 0 and at 256.
        Inputs seeded = in;
        seeded.ub_at = 0;
        seeded.ub_bytes.assign(256 + len * dim, 0);
        for (uint32_t i = 0; i < len * dim; ++i) {
            seeded.ub_bytes[i]       = static_cast<i8>(in.host_bytes[i]);
            seeded.ub_bytes[256 + i] = static_cast<i8>(in.host_bytes[i]);
        }

        auto run_prog = [&](const std::vector<RawInst>& code, Tpu& t) {
            tpu_setup(seeded)(t);
            const TpuResult r = t.run(code);
            REQUIRE(r.halted);
            return r;
        };

        Tpu ti(cfg), ts(cfg), ta(cfg);
        const TpuResult ri = run_prog(indep.code(), ti);
        const TpuResult rs = run_prog(serial.code(), ts);
        const TpuResult ra = run_prog(alone.code(), ta);

        // Overlapped, the extra matmul is free: it hides entirely inside the
        // activation, and even the cycle it spends issuing was one the machine would
        // have spent waiting at the Halt anyway.
        REQUIRE_MSG(ri.cycles == ra.cycles,
                    "    alone " + std::to_string(ra.cycles) + " indep " +
                        std::to_string(ri.cycles) + " serial " + std::to_string(rs.cycles) +
                        " mm_dur " + std::to_string(mm_dur) + "\n");

        // Serialized, it cannot start until the activation retires, so it costs its
        // whole duration on top.
        REQUIRE(rs.cycles == ra.cycles + mm_dur);
        REQUIRE(rs.cycles > ri.cycles);

        // Both programs stall on the accumulator, because the Activate waits for the
        // matmul feeding it either way -- but only the serializing one stalls twice.
        REQUIRE(ti.stalls().accum_hazard > 0);
        REQUIRE(ts.stalls().accum_hazard > ti.stalls().accum_hazard);

        // And overlapping did not change either answer.
        const std::string d1 = diff_tpu(indep.code(), cfg, ref_setup(seeded), tpu_setup(seeded));
        REQUIRE_MSG(d1.empty(), d1);
        const std::string d2 = diff_tpu(serial.code(), cfg, ref_setup(seeded), tpu_setup(seeded));
        REQUIRE_MSG(d2.empty(), d2);
    }

    // An Activate whose output a following MatMul reads must serialize, since the
    // buffer region is the dependence.
    {
        Config cfg = small_cfg(dim, 4);
        cfg.ub_bytes = 2048;
        cfg.act_pipeline_depth = 4;

        tpuasm::Program p;
        p.read_weights(0)
         .matmul(0, dim, 0)
         .activate(0, 256, dim, ActFn::RELU, 1, 4)   // writes [256, 272)
         .matmul(256, dim, 1)                        // reads exactly that
         .halt();

        Tpu t(cfg);
        const Inputs in = random_inputs(6201, dim, dim);
        tpu_setup(in)(t);
        REQUIRE(t.run(p.code()).halted);
        REQUIRE(t.stalls().ub_raw > 0);
    }
}
