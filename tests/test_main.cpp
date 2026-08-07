// Test driver: one SECTION() block per group of assertions, all in one
// translation unit so `make test` builds a single binary.

#include <cstdint>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "types.h"
#include "config.h"
#include "tensor.h"
#include "quant.h"
#include "isa.h"
#include "decoder.h"
#include "loader.h"
#include "pe.h"
#include "mxu.h"
#include "tpuasm.h"
#include "ref.h"

// Include the driver TU so parse_args / print_help are unit-testable without
// shelling out. Its own int main() is guarded off.
#define MINI_TPU_NO_ENTRY
#include "main.cpp"

// ---------------------------------------------------------- harness plumbing ---
namespace test {

using Fn = std::function<void()>;

inline std::vector<std::pair<std::string, Fn>>& registry() {
    static std::vector<std::pair<std::string, Fn>> r;
    return r;
}

struct Register {
    Register(const char* name, Fn fn) { registry().emplace_back(name, std::move(fn)); }
};

inline int assertion_failures = 0;

inline void report_fail(const char* expr, const char* file, int line) {
    std::fprintf(stderr, "  FAIL: %s   at %s:%d\n", expr, file, line);
    ++assertion_failures;
}

// For failures where the expression alone says nothing useful, such as a
// tensor compare that needs to print what actually came out.
inline void report_fail_msg(const char* expr, const std::string& detail,
                            const char* file, int line) {
    std::fprintf(stderr, "  FAIL: %s   at %s:%d\n%s", expr, file, line, detail.c_str());
    if (!detail.empty() && detail.back() != '\n') std::fputc('\n', stderr);
    ++assertion_failures;
}

}  // namespace test

#define MT_CAT_INNER(a, b) a##b
#define MT_CAT(a, b) MT_CAT_INNER(a, b)
#define SECTION(name)                                                          \
    static void MT_CAT(section_fn_, __LINE__)();                               \
    static const ::test::Register MT_CAT(section_reg_, __LINE__)(              \
        name, &MT_CAT(section_fn_, __LINE__));                                 \
    static void MT_CAT(section_fn_, __LINE__)()

#define REQUIRE(expr)                                                          \
    do {                                                                       \
        if (!(expr)) ::test::report_fail(#expr, __FILE__, __LINE__);            \
    } while (0)

#define REQUIRE_MSG(expr, detail)                                              \
    do {                                                                       \
        if (!(expr)) ::test::report_fail_msg(#expr, (detail), __FILE__, __LINE__); \
    } while (0)

// ------------------------------------------------------------------ helpers ---
namespace {

// A small machine: hand-checked expectations are only possible when the array
// is small enough to multiply on paper.
Config small_cfg(uint32_t dim = 4, uint32_t banks = 2) {
    Config c;
    c.dim       = dim;
    c.acc_banks = banks;
    c.ub_bytes  = 4096;
    return c;
}

std::string show(const std::vector<int>& v) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i != 0) os << ", ";
        os << v[i];
    }
    os << "]";
    return os.str();
}

// Detail block for a failed tensor compare.
std::string diff_vec(const char* what, const std::vector<int>& got,
                     const std::vector<int>& want) {
    if (got == want) return "";
    std::ostringstream os;
    os << "    " << what << " got  " << show(got) << "\n"
       << "    " << what << " want " << show(want) << "\n";
    return os.str();
}

// ---- reference-machine pokes and peeks ------------------------------------
// The reference model's state is public on purpose; a test sets up exactly the
// state it means to exercise instead of arranging a program to produce it.

void put_bytes(ref::Machine& m, UbAddr at, std::initializer_list<int> vals) {
    std::size_t i = 0;
    for (const int v : vals) m.ub[at + i++] = static_cast<i8>(v);
}

void put_weights(ref::Machine& m, uint32_t at, std::initializer_list<int> vals) {
    std::size_t i = 0;
    for (const int v : vals) m.weight_mem[at + i++] = static_cast<i8>(v);
}

void put_acc(ref::Machine& m, BankId b, std::initializer_list<int> vals) {
    const I32View bank = m.acc_bank(b);
    std::size_t i = 0;
    for (const int v : vals) {
        bank.at(static_cast<uint32_t>(i / m.dim()), static_cast<uint32_t>(i % m.dim())) = v;
        ++i;
    }
}

std::vector<int> ub_slice(const ref::Machine& m, UbAddr at, std::size_t n) {
    std::vector<int> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) out.push_back(m.ub[at + i]);
    return out;
}

std::vector<int> host_slice(const ref::Machine& m, HostAddr at, std::size_t n) {
    std::vector<int> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) out.push_back(static_cast<int8_t>(m.host[at + i]));
    return out;
}

std::vector<int> acc_slice(const ref::Machine& m, BankId b, uint32_t rows, uint32_t cols) {
    const ConstI32View bank = m.acc_bank(b);
    std::vector<int> out;
    out.reserve(static_cast<std::size_t>(rows) * cols);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) out.push_back(bank.at(r, c));
    }
    return out;
}

// ---- differential comparison ---------------------------------------------

template <typename T>
std::size_t first_diff(const std::vector<T>& a, const std::vector<T>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) return i;
    }
    return n;   // a common prefix; the sizes are what differ
}

// Every comparand from DESIGN 8.1: output bytes, accumulators at each Sync, and
// the retired count. Returns "" when the two runs agree.
std::string compare_runs(const char* a_name, const ref::Machine& ma, const ref::Result& ra,
                         const char* b_name, const ref::Machine& mb, const ref::Result& rb) {
    std::ostringstream os;
    auto note = [&](const char* what, const std::string& av, const std::string& bv) {
        os << "    " << what << ": " << a_name << "=" << av << "  " << b_name << "=" << bv << "\n";
    };
    auto num = [](uint64_t v) { return std::to_string(v); };

    if (ra.halted != rb.halted) {
        note("halted", ra.halted ? "1" : "0", rb.halted ? "1" : "0");
    }
    if (ra.trapped != rb.trapped) {
        note("trapped", ra.trapped ? ra.trap_reason : "no", rb.trapped ? rb.trap_reason : "no");
    }
    if (ra.exit_code != rb.exit_code) note("exit code", num(ra.exit_code), num(rb.exit_code));
    if (ra.retired != rb.retired)     note("retired", num(ra.retired), num(rb.retired));

    if (ra.syncs.size() != rb.syncs.size()) {
        note("sync count", num(ra.syncs.size()), num(rb.syncs.size()));
    } else {
        for (std::size_t s = 0; s < ra.syncs.size(); ++s) {
            const auto& av = ra.syncs[s].acc;
            const auto& bv = rb.syncs[s].acc;
            if (av == bv) continue;
            const std::size_t i = first_diff(av, bv);
            os << "    sync " << s << " accumulator[" << i << "]: " << a_name << "="
               << (i < av.size() ? num(static_cast<uint64_t>(av[i])) : "-") << "  " << b_name
               << "=" << (i < bv.size() ? num(static_cast<uint64_t>(bv[i])) : "-") << "\n";
        }
    }

    if (ma.acc != mb.acc) {
        const std::size_t i = first_diff(ma.acc, mb.acc);
        note(("final accumulator[" + num(i) + "]").c_str(),
             num(static_cast<uint64_t>(ma.acc[i])), num(static_cast<uint64_t>(mb.acc[i])));
    }
    if (ma.ub != mb.ub) {
        const std::size_t i = first_diff(ma.ub, mb.ub);
        note(("unified buffer[" + num(i) + "]").c_str(),
             num(static_cast<uint64_t>(ma.ub[i])), num(static_cast<uint64_t>(mb.ub[i])));
    }
    if (ma.host != mb.host) {
        const std::size_t i = first_diff(ma.host, mb.host);
        note(("host memory[" + num(i) + "]").c_str(),
             num(ma.host[i]), num(mb.host[i]));
    }
    return os.str();
}

using Setup = std::function<void(ref::Machine&)>;

// Run a program down both paths and compare. Both sides are the reference model
// until `Tpu` arrives in Phase 3, at which point one side is swapped for it and
// every existing caller becomes a real differential test.
std::string diff_run(const std::vector<RawInst>& prog, const Config& cfg,
                     const Setup& setup = {}) {
    ref::Machine ma(cfg), mb(cfg);
    if (setup) { setup(ma); setup(mb); }
    const ref::Result ra = ref::run(ma, prog);
    const ref::Result rb = ref::run(mb, prog);
    return compare_runs("ref", ma, ra, "ref-again", mb, rb);
}

// ---- systolic array helpers ----------------------------------------------

// A deterministic byte source. Taking the top bits of an LCG gives a spread over
// the whole int8 range, -128 included, which matters because -128 has no
// positive counterpart and is where a sloppy negation would show up.
struct Lcg {
    uint32_t s;
    explicit Lcg(uint32_t seed) : s(seed) {}
    uint32_t next() {
        s = s * 1664525u + 1013904223u;
        return s;
    }
    i8 byte() { return static_cast<i8>(next() >> 24); }
};

// The oracle's answer, obtained by actually running Read_Weights + MatMul
// through ref.h rather than by reimplementing the matmul here.
std::vector<int> ref_matmul(const Config& cfg, const std::vector<i8>& weights,
                            const std::vector<i8>& acts, uint32_t len) {
    ref::Machine m(cfg);
    for (std::size_t i = 0; i < weights.size(); ++i) m.weight_mem[i] = weights[i];
    for (std::size_t i = 0; i < acts.size(); ++i) m.ub[i] = acts[i];

    tpuasm::Program p;
    p.read_weights(0).matmul(0, len, 0).halt();
    const ref::Result r = ref::run(m, p.code());
    if (!r.halted) return {};        // a trap shows up as a shape mismatch
    return acc_slice(m, 0, len, cfg.dim);
}

// The same matmul on the cycle-accurate array.
std::vector<int> mxu_matmul(const Config& cfg, const std::vector<i8>& weights,
                            const std::vector<i8>& acts, uint32_t len) {
    Mxu mxu(cfg);
    mxu.load_weights(ConstI8View(weights.data(), cfg.dim, cfg.dim));

    I32Tensor out(len == 0 ? 1 : len, cfg.dim);
    if (len != 0) {
        mxu.matmul(ConstI8View(acts.data(), len, cfg.dim), out.view(), false);
    }

    std::vector<int> flat;
    flat.reserve(static_cast<std::size_t>(len) * cfg.dim);
    for (uint32_t r = 0; r < len; ++r) {
        for (uint32_t c = 0; c < cfg.dim; ++c) flat.push_back(out.view().at(r, c));
    }
    return flat;
}

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

// ------------------------------------------------------- @section("types") ---
SECTION("types") {
    static_assert(std::is_same_v<i8,  int8_t>);
    static_assert(std::is_same_v<i32, int32_t>);
    static_assert(std::is_same_v<UbAddr,   uint32_t>);
    static_assert(std::is_same_v<HostAddr, uint32_t>);
    static_assert(std::is_same_v<BankId,   uint32_t>);
    static_assert(std::is_same_v<TileId,   uint32_t>);

    // Sentinels sit outside any plausible in-band value.
    REQUIRE(INVALID_UBADDR   > (1u << 24));
    REQUIRE(INVALID_HOSTADDR > (1u << 24));
    REQUIRE(INVALID_BANK     > (1u << 16));
    REQUIRE(INVALID_TILE     > (1u << 16));

    // Round-trip through the std::optional idiom the scoreboard will use.
    auto to_opt = [](BankId b) -> std::optional<BankId> {
        return b == INVALID_BANK ? std::nullopt : std::optional<BankId>(b);
    };
    auto from_opt = [](std::optional<BankId> o) -> BankId { return o.value_or(INVALID_BANK); };

    REQUIRE(!to_opt(INVALID_BANK).has_value());
    REQUIRE(from_opt(to_opt(INVALID_BANK)) == INVALID_BANK);
    for (const BankId b : std::initializer_list<BankId>{0u, 1u, 7u}) {
        REQUIRE(to_opt(b).has_value());
        REQUIRE(from_opt(to_opt(b)) == b);
    }

    // Enumerators are distinct.
    static_assert(static_cast<int>(Op::READ_HOST) != static_cast<int>(Op::HALT));
    static_assert(static_cast<int>(Op::MATMUL)    != static_cast<int>(Op::ACTIVATE));
    static_assert(static_cast<int>(ActFn::RELU)   != static_cast<int>(ActFn::RELU6));
    static_assert(static_cast<int>(Pool::MAX)     != static_cast<int>(Pool::AVG));

    // HALT is the last opcode; the decoder's illegal-encoding check relies on
    // it, so a new opcode appended after it would silently break that test.
    static_assert(static_cast<uint32_t>(Op::HALT) == isa::MAX_OPCODE);
}

// ------------------------------------------------------ @section("config") ---
SECTION("config") {
    Config c;

    // Documented defaults (DESIGN 9.1).
    REQUIRE(c.dim                 == 32);
    REQUIRE(c.ub_bytes            == 256u * 1024u);
    REQUIRE(c.ub_banks            == 8);
    REQUIRE(c.acc_banks           == 4);
    REQUIRE(c.weight_fifo_depth   == 4);
    REQUIRE(c.double_buffer       == true);
    REQUIRE(c.dma_bytes_per_cycle == 16);

    // A 32x32 array is 1024 MACs per cycle, so the ridge point is 1024/16 = 64
    // MACs per byte moved.
    REQUIRE(c.peak_macs_per_cycle() == 1024u);
    REQUIRE(c.dma_bound(1000, 100));        // 10 MACs/byte: memory bound
    REQUIRE(!c.dma_bound(12800, 100));      // 128 MACs/byte: compute bound
    REQUIRE(!c.dma_bound(6400, 100));       // exactly at the ridge is not below it
    REQUIRE(c.dma_bound(6399, 100));        // one MAC under, and it tips over

    // A bigger array moves the ridge point up: the same workload that saturated
    // a small array becomes memory bound on a large one.
    Config big = c;
    big.dim = 256;
    REQUIRE(big.peak_macs_per_cycle() == 65536u);
    REQUIRE(big.dma_bound(12800, 100));

    static_assert(std::is_trivially_copyable_v<Config>);
}

// ------------------------------------------------------ @section("tensor") ---
SECTION("tensor") {
    // A strided window must alias the parent, not copy it.
    I8Tensor t(6, 6);
    const I8View v = t.view();
    for (uint32_t r = 0; r < 6; ++r) {
        for (uint32_t c = 0; c < 6; ++c) v.at(r, c) = static_cast<i8>(r * 10 + c);
    }

    const I8View sub = v.tile(1, 2, 2, 3);
    REQUIRE(sub.rows() == 2);
    REQUIRE(sub.cols() == 3);
    REQUIRE(sub.stride() == 6);          // the pitch is inherited from the parent
    for (uint32_t r = 0; r < 2; ++r) {
        for (uint32_t c = 0; c < 3; ++c) {
            REQUIRE(sub.at(r, c) == t.data()[(1 + r) * 6 + (2 + c)]);
        }
    }

    sub.at(1, 2) = 99;                   // written through the view
    REQUIRE(t.data()[2 * 6 + 4] == 99);

    // A window of a window composes.
    const I8View sub2 = sub.tile(0, 1, 1, 2);
    REQUIRE(sub2.at(0, 0) == v.at(1, 3));

    // Partial tile: a 2x3 source into a 4x4 tile zero-pads the remainder, which
    // is what makes an undersized matmul produce the right answer.
    I8Tensor src(2, 3);
    src.view().fill(7);
    I8Tensor dst(4, 4);
    dst.view().fill(-1);
    dst.view().zero_pad_from(src.view());
    std::vector<int> got, want;
    for (uint32_t r = 0; r < 4; ++r) {
        for (uint32_t c = 0; c < 4; ++c) {
            got.push_back(dst.view().at(r, c));
            want.push_back((r < 2 && c < 3) ? 7 : 0);
        }
    }
    REQUIRE_MSG(got == want, diff_vec("padded tile", got, want));

    // copy_from crosses strides.
    I8Tensor wide(4, 8);
    wide.view().fill(0);
    wide.view().tile(0, 4, 2, 3).copy_from(src.view());
    REQUIRE(wide.view().at(0, 4) == 7);
    REQUIRE(wide.view().at(1, 6) == 7);
    REQUIRE(wide.view().at(0, 7) == 0);

    // same_values reports a shape mismatch rather than asserting.
    REQUIRE(same_values(src.view(), src.view()));
    REQUIRE(!same_values(src.view(), dst.view()));

    // A non-const view converts to a const one.
    const ConstI8View cv = v;
    REQUIRE(cv.at(2, 4) == 99);
}

// ------------------------------------------------------- @section("quant") ---
SECTION("quant") {
    using namespace quant;

    // Ties round away from zero, in both signs.
    REQUIRE(round_shift(5, 1) == 3);
    REQUIRE(round_shift(-5, 1) == -3);
    REQUIRE(round_shift(3, 1) == 2);
    REQUIRE(round_shift(-3, 1) == -2);
    REQUIRE(round_shift(1, 1) == 1);
    REQUIRE(round_shift(-1, 1) == -1);
    REQUIRE(round_shift(7, 0) == 7);

    // A bare arithmetic shift floors, which would give -1 and -2 here.
    REQUIRE(round_shift(-6, 2) == -2);
    REQUIRE(round_shift(-10, 2) == -3);

    // Average pooling's divisor is a window area, not a power of two, and it
    // follows the same rule through the same helper.
    REQUIRE(round_div(5, 2) == 3);
    REQUIRE(round_div(-5, 2) == -3);
    REQUIRE(round_div(1, 3) == 0);
    REQUIRE(round_div(2, 3) == 1);
    REQUIRE(round_div(-2, 3) == -1);
    REQUIRE(round_div(14, 4) == 4);      // 3.5 -> 4
    REQUIRE(round_div(-14, 4) == -4);

    // Saturation clamps at both ends, not just the positive one.
    REQUIRE(requantize(127, 1, 0) == 127);
    REQUIRE(requantize(128, 1, 0) == 127);
    REQUIRE(requantize(100000, 1, 0) == 127);
    REQUIRE(requantize(-128, 1, 0) == -128);
    REQUIRE(requantize(-129, 1, 0) == -128);
    REQUIRE(requantize(-100000, 1, 0) == -128);

    // Bias folds in, and folding in a zero bias changes nothing.
    REQUIRE(requantize_biased(100, 0, 1, 0) == requantize(100, 1, 0));
    REQUIRE(requantize_biased(-5, 10, 1, 1) == 3);      // (-5+10)/2 = 2.5 -> 3
    REQUIRE(requantize_biased(200, 10, 1, 1) == 105);
    // A bias large enough to overflow int32 if it were added there.
    REQUIRE(requantize_biased(2147483647, 2147483647, 1, 8) == 127);

    // Sweep against an independently written rounding rule: this one compares
    // twice the remainder against the divisor instead of adding half first.
    auto expect_rq = [](i32 acc, i32 mult, uint32_t shift) -> int {
        const int64_t num = static_cast<int64_t>(acc) * mult;
        const int64_t den = int64_t{1} << shift;
        const int64_t a   = num < 0 ? -num : num;
        int64_t q = a / den;
        if (2 * (a % den) >= den) ++q;
        if (num < 0) q = -q;
        if (q > 127)  q = 127;
        if (q < -128) q = -128;
        return static_cast<int>(q);
    };

    int mismatches = 0;
    for (const i32 mult : std::initializer_list<i32>{1, 3, 127, 1000000}) {
        for (const uint32_t shift : std::initializer_list<uint32_t>{0u, 1u, 7u, 15u, 31u}) {
            for (i32 acc = -2000; acc <= 2000; ++acc) {
                if (requantize(acc, mult, shift) != expect_rq(acc, mult, shift)) ++mismatches;
            }
        }
    }
    REQUIRE(mismatches == 0);
}

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

// ---------------------------------------------------------- @section("pe") ---
SECTION("pe") {
    Pe p;
    p.set_weight(3, 0);
    p.set_weight(-5, 1);
    REQUIRE(p.weight(0) == 3);
    REQUIRE(p.weight(1) == -5);

    // Nothing is visible until commit: this is the property that keeps the order
    // PEs are visited in out of the results.
    p.tick(7, 100, 0);
    REQUIRE(p.act_out() == 0);
    REQUIRE(p.psum_out() == 0);

    p.commit();
    REQUIRE(p.psum_out() == 100 + 7 * 3);
    REQUIRE(p.act_out() == 7);       // the activation passed through as well

    // The plane argument selects which weight multiplies.
    p.tick(7, 0, 1);
    p.commit();
    REQUIRE(p.psum_out() == 7 * -5);

    // A held value is not sticky: feeding zero clears the products back out.
    p.tick(0, 0, 0);
    p.commit();
    REQUIRE(p.psum_out() == 0);
    REQUIRE(p.act_out() == 0);

    // The extremes of int8, including the asymmetric -128.
    Pe q;
    q.set_weight(-128, 0);
    q.tick(-128, 0, 0);
    q.commit();
    REQUIRE(q.psum_out() == 16384);

    q.set_weight(127, 0);
    q.tick(-128, 0, 0);
    q.commit();
    REQUIRE(q.psum_out() == -16256);

    // clear_pipeline drops data in flight but keeps the weights, because weights
    // are loaded by their own instruction and outlive any one matmul.
    q.tick(5, 999, 0);
    q.commit();
    REQUIRE(q.psum_out() != 0);
    q.clear_pipeline();
    REQUIRE(q.psum_out() == 0);
    REQUIRE(q.act_out() == 0);
    REQUIRE(q.weight(0) == 127);

    // A chain of PEs delays by exactly one cycle per hop. Driving three in a row
    // for three cycles, the value entering the first should be leaving the third
    // only on the third commit.
    Pe chain[3];
    for (Pe& c : chain) c.set_weight(1, 0);
    for (int cycle = 0; cycle < 3; ++cycle) {
        const i8 in = (cycle == 0) ? i8{42} : i8{0};
        chain[0].tick(in, 0, 0);
        chain[1].tick(chain[0].act_out(), 0, 0);
        chain[2].tick(chain[1].act_out(), 0, 0);
        for (Pe& c : chain) c.commit();
        REQUIRE(chain[2].act_out() == (cycle == 2 ? 42 : 0));
    }
}

// ------------------------------------------------- @section("mxu_weights") ---
SECTION("mxu_weights") {
    Config cfg = small_cfg();        // dim 4
    cfg.double_buffer = false;       // load straight into the active plane

    Mxu mxu(cfg);
    REQUIRE(mxu.dim() == 4);
    REQUIRE(mxu.active_plane() == 0);
    REQUIRE(mxu.load_plane() == 0);

    // A full tile: PE[k][c] must hold W[k][c] and not its transpose, which is the
    // single easiest thing to get backwards in the whole array.
    I8Tensor w(4, 4);
    for (uint32_t k = 0; k < 4; ++k) {
        for (uint32_t c = 0; c < 4; ++c) w.view().at(k, c) = static_cast<i8>(k * 10 + c);
    }
    mxu.load_weights(w.view());

    bool placed = true;
    for (uint32_t k = 0; k < 4; ++k) {
        for (uint32_t c = 0; c < 4; ++c) {
            if (mxu.weight_at(k, c, 0) != static_cast<i8>(k * 10 + c)) placed = false;
        }
    }
    REQUIRE(placed);
    REQUIRE(mxu.weight_at(0, 3, 0) == 3);     // and not 30
    REQUIRE(mxu.weight_at(3, 0, 0) == 30);
    REQUIRE(mxu.plane_rows(0) == 4);
    REQUIRE(mxu.plane_cols(0) == 4);
    REQUIRE(mxu.stats().weight_loads == 1);

    // A partial tile zero-pads the rest of the array.
    Mxu part(cfg);
    I8Tensor small(2, 3);
    small.view().fill(9);
    part.load_weights(small.view());

    std::vector<int> got, want;
    for (uint32_t k = 0; k < 4; ++k) {
        for (uint32_t c = 0; c < 4; ++c) {
            got.push_back(part.weight_at(k, c, 0));
            want.push_back((k < 2 && c < 3) ? 9 : 0);
        }
    }
    REQUIRE_MSG(got == want, diff_vec("padded weights", got, want));
    REQUIRE(part.plane_rows(0) == 2);
    REQUIRE(part.plane_cols(0) == 3);

    // Loading again must overwrite the padding too, or a small tile would leave
    // a previous large one's values lying around to corrupt the next matmul.
    part.load_weights(w.view());
    REQUIRE(part.weight_at(3, 3, 0) == 33);
    I8Tensor tiny(1, 1);
    tiny.view().fill(4);
    part.load_weights(tiny.view());
    REQUIRE(part.weight_at(0, 0, 0) == 4);
    REQUIRE(part.weight_at(3, 3, 0) == 0);
    REQUIRE(part.weight_at(0, 1, 0) == 0);

    // With double buffering the load goes to the shadow plane and leaves the
    // active one untouched.
    Config db = small_cfg();
    db.double_buffer = true;
    Mxu two(db);
    REQUIRE(two.load_plane() == 1);
    two.load_weights(w.view());
    REQUIRE(two.weight_at(2, 2, 1) == 22);
    REQUIRE(two.weight_at(2, 2, 0) == 0);      // active plane still clear
    REQUIRE(two.switch_pending());
}

// -------------------------------------------------- @section("mxu_matmul") ---
SECTION("mxu_matmul") {
    // The first differential pass: the cycle-accurate array against the eager
    // oracle, over a spread of shapes and over data that includes both int8
    // extremes. Nothing here recomputes the matmul locally -- the expectation
    // comes from running ref.h.
    Lcg rng(12345);
    int shapes = 0, disagreements = 0;
    std::string first_bad;

    for (const uint32_t dim : std::initializer_list<uint32_t>{1u, 2u, 3u, 4u, 8u}) {
        Config cfg = small_cfg(dim, 2);
        cfg.double_buffer = false;

        for (const uint32_t len : std::initializer_list<uint32_t>{1u, 2u, 3u, dim, dim}) {
            if (len > dim) continue;             // a bank holds at most dim rows

            std::vector<i8> w(static_cast<std::size_t>(dim) * dim);
            std::vector<i8> a(static_cast<std::size_t>(len) * dim);
            for (i8& v : w) v = rng.byte();
            for (i8& v : a) v = rng.byte();

            ++shapes;
            const std::vector<int> want = ref_matmul(cfg, w, a, len);
            const std::vector<int> got  = mxu_matmul(cfg, w, a, len);
            if (got != want) {
                ++disagreements;
                if (first_bad.empty()) {
                    first_bad = "    dim " + std::to_string(dim) + " len " +
                                std::to_string(len) + "\n" + diff_vec("out", got, want);
                }
            }
        }
    }
    REQUIRE(shapes > 0);
    REQUIRE_MSG(disagreements == 0, first_bad);

    // Saturating inputs everywhere: every product is 16384 and every column sums
    // dim of them, so this is the widest the int32 chain gets in these tests.
    {
        Config cfg = small_cfg(8, 2);
        cfg.double_buffer = false;
        const std::vector<i8> w(64, -128);
        const std::vector<i8> a(64, -128);
        const std::vector<int> want = ref_matmul(cfg, w, a, 8);
        const std::vector<int> got  = mxu_matmul(cfg, w, a, 8);
        REQUIRE_MSG(got == want, diff_vec("extremes", got, want));
        REQUIRE(want[0] == 8 * 16384);
    }

    // The hand-checked case from the ref section, now through the array: row
    // [1 0 0 0] selects W row 0 and [0 1 1 0] sums W rows 1 and 2.
    {
        Config cfg = small_cfg();
        cfg.double_buffer = false;
        Mxu mxu(cfg);
        I8Tensor w(4, 4);
        for (uint32_t k = 0; k < 4; ++k) {
            for (uint32_t c = 0; c < 4; ++c) {
                w.view().at(k, c) = static_cast<i8>(k * 4 + c + 1);
            }
        }
        mxu.load_weights(w.view());

        I8Tensor a(2, 4);
        a.view().fill(0);
        a.view().at(0, 0) = 1;
        a.view().at(1, 1) = 1;
        a.view().at(1, 2) = 1;

        I32Tensor out(2, 4);
        mxu.matmul(a.view(), out.view(), false);

        std::vector<int> got;
        for (uint32_t r = 0; r < 2; ++r) {
            for (uint32_t c = 0; c < 4; ++c) got.push_back(out.view().at(r, c));
        }
        const std::vector<int> want{1, 2, 3, 4, 14, 16, 18, 20};
        REQUIRE_MSG(got == want, diff_vec("hand-checked", got, want));
    }

    // A zero-padded partial tile is correct, not merely plausible: a real 2x3 by
    // 3x2 problem in a 4x4 array.
    {
        Config cfg = small_cfg();
        cfg.double_buffer = false;
        Mxu mxu(cfg);

        I8Tensor w(3, 2);
        w.view().at(0, 0) = 1; w.view().at(0, 1) = 2;
        w.view().at(1, 0) = 3; w.view().at(1, 1) = 4;
        w.view().at(2, 0) = 5; w.view().at(2, 1) = 6;
        mxu.load_weights(w.view());

        I8Tensor a(2, 4);
        a.view().fill(0);
        a.view().at(0, 0) = 1; a.view().at(0, 2) = 1;
        a.view().at(1, 0) = 2; a.view().at(1, 1) = 1;

        I32Tensor out(2, 4);
        mxu.matmul(a.view(), out.view(), false);

        std::vector<int> got;
        for (uint32_t r = 0; r < 2; ++r) {
            for (uint32_t c = 0; c < 4; ++c) got.push_back(out.view().at(r, c));
        }
        const std::vector<int> want{6, 8, 0, 0, 5, 8, 0, 0};
        REQUIRE_MSG(got == want, diff_vec("partial tile", got, want));
    }

    // Accumulating into a bank that already holds a previous K-tile's result.
    {
        Config cfg = small_cfg();
        cfg.double_buffer = false;
        Mxu mxu(cfg);

        I8Tensor ident(4, 4);
        ident.view().fill(0);
        for (uint32_t i = 0; i < 4; ++i) ident.view().at(i, i) = 1;
        I8Tensor twos(4, 4);
        twos.view().fill(2);

        I8Tensor a0(1, 4), a1(1, 4);
        for (uint32_t k = 0; k < 4; ++k) a0.view().at(0, k) = static_cast<i8>(k + 1);
        a1.view().fill(1);

        I32Tensor out(1, 4);
        mxu.load_weights(ident.view());
        mxu.matmul(a0.view(), out.view(), false);
        mxu.load_weights(twos.view());
        mxu.matmul(a1.view(), out.view(), true);

        std::vector<int> got;
        for (uint32_t c = 0; c < 4; ++c) got.push_back(out.view().at(0, c));
        const std::vector<int> want{9, 10, 11, 12};
        REQUIRE_MSG(got == want, diff_vec("k-tiled", got, want));
    }

    // Back-to-back matmuls must not leak: the de-skew window has to reject the
    // previous stream's data still draining through the array. Large values
    // first, so a leak would be unmistakable.
    {
        Config cfg = small_cfg(4, 2);
        cfg.double_buffer = false;

        const std::vector<i8> big_w(16, 127);
        const std::vector<i8> big_a(16, 127);
        std::vector<i8>       w2(16), a2(16);
        Lcg r2(999);
        for (i8& v : w2) v = r2.byte();
        for (i8& v : a2) v = r2.byte();

        Mxu mxu(cfg);
        I32Tensor scratch(4, 4), out(4, 4);
        mxu.load_weights(ConstI8View(big_w.data(), 4, 4));
        mxu.matmul(ConstI8View(big_a.data(), 4, 4), scratch.view(), false);
        mxu.load_weights(ConstI8View(w2.data(), 4, 4));
        mxu.matmul(ConstI8View(a2.data(), 4, 4), out.view(), false);

        std::vector<int> got;
        for (uint32_t r = 0; r < 4; ++r) {
            for (uint32_t c = 0; c < 4; ++c) got.push_back(out.view().at(r, c));
        }
        const std::vector<int> want = ref_matmul(cfg, w2, a2, 4);
        REQUIRE_MSG(got == want, diff_vec("second matmul", got, want));
    }
}

// -------------------------------------------------- @section("mxu_timing") ---
SECTION("mxu_timing") {
    // The occupancy formula, over several shapes.
    for (const uint32_t dim : std::initializer_list<uint32_t>{1u, 2u, 4u, 8u}) {
        Config cfg = small_cfg(dim, 2);
        cfg.double_buffer = false;

        for (uint32_t len = 1; len <= dim; ++len) {
            Mxu mxu(cfg);
            I8Tensor w(dim, dim);
            w.view().fill(1);
            mxu.load_weights(w.view());

            I8Tensor a(len, dim);
            a.view().fill(1);
            I32Tensor out(len, dim);

            const uint64_t before = mxu.cycle();
            const MxuTiming t = mxu.matmul(a.view(), out.view(), false);

            REQUIRE(t.cycles == static_cast<uint64_t>(len) + 2u * dim - 1u);
            REQUIRE(t.start == before);
            REQUIRE(t.end == t.start + t.cycles);
            REQUIRE(mxu.cycle() == t.end);

            // Each output row completes once its last column drains out, which is
            // r + 2*dim - 1 cycles after the stream began.
            REQUIRE(t.row_valid.size() == len);
            bool valid_ok = true;
            for (uint32_t r = 0; r < len; ++r) {
                if (t.row_valid[r] != t.start + r + 2ull * dim - 1ull) valid_ok = false;
                if (r > 0 && t.row_valid[r] <= t.row_valid[r - 1]) valid_ok = false;
            }
            REQUIRE(valid_ok);

            // The last row lands on the final cycle of the occupancy window, so
            // no result is claimed after the array reports itself free.
            REQUIRE(t.row_valid.back() == t.end - 1);
            REQUIRE(t.row_valid.front() >= t.start);
        }
    }

    // Fill and drain is pure overhead, so utilization is len / (len + 2*dim - 1)
    // and only a long stream amortizes it away. Double buffering is on here to
    // keep weight-load bubbles out of the clock, leaving fill and drain as the
    // only thing being measured.
    {
        Config cfg = small_cfg(4, 2);
        cfg.double_buffer = true;

        Mxu one(cfg);
        I8Tensor w(4, 4);
        w.view().fill(1);
        I8Tensor a1(1, 4);
        a1.view().fill(1);
        I32Tensor o1(1, 4);
        one.load_weights(w.view());
        one.matmul(a1.view(), o1.view(), false);
        REQUIRE(one.stats().cycles == 1 + 7);
        REQUIRE(one.stats().fill_drain_cycles == 7);
        REQUIRE(one.stats().useful_macs == 16);
        REQUIRE(one.utilization() < 0.2);        // 16 / (16 * 8)

        // Four rows through the same weights: four times the work in less than
        // twice the time.
        Mxu four(cfg);
        I8Tensor a4(4, 4);
        a4.view().fill(1);
        I32Tensor o4(4, 4);
        four.load_weights(w.view());
        four.matmul(a4.view(), o4.view(), false);
        REQUIRE(four.stats().cycles == 4 + 7);
        REQUIRE(four.utilization() > one.utilization());

        // Weight-load bubbles belong to the clock too, so a stalled load shows up
        // as lost utilization rather than disappearing from the accounting.
        Config slow = cfg;
        slow.double_buffer = false;
        Mxu stalled(slow);
        I32Tensor os(4, 4);
        stalled.load_weights(w.view());
        stalled.matmul(a4.view(), os.view(), false);
        REQUIRE(stalled.stats().cycles == 4 + 7 + 4);
        REQUIRE(stalled.stats().useful_macs == four.stats().useful_macs);
        REQUIRE(stalled.utilization() < four.utilization());
    }

    // Padding is attributed rather than hidden: a real 2x3 tile in a 4x4 array
    // does 2*2*3 = 12 useful MACs out of 2*16 = 32 slots.
    {
        Config cfg = small_cfg(4, 2);
        cfg.double_buffer = false;
        Mxu mxu(cfg);

        I8Tensor small(2, 3);
        small.view().fill(1);
        mxu.load_weights(small.view());

        I8Tensor a(2, 4);
        a.view().fill(1);
        I32Tensor out(2, 4);
        mxu.matmul(a.view(), out.view(), false);

        REQUIRE(mxu.stats().useful_macs == 12);
        REQUIRE(mxu.stats().partial_tile_waste == 20);

        // A full tile wastes nothing.
        Mxu full(cfg);
        I8Tensor w(4, 4);
        w.view().fill(1);
        full.load_weights(w.view());
        I32Tensor fout(2, 4);
        full.matmul(a.view(), fout.view(), false);
        REQUIRE(full.stats().useful_macs == 32);
        REQUIRE(full.stats().partial_tile_waste == 0);
    }
}

// ------------------------------------------- @section("mxu_double_buffer") ---
SECTION("mxu_double_buffer") {
    // The marquee property of this phase. Two matmuls on different weight tiles:
    // the second tile's load is free when it can shift into the shadow plane, and
    // costs a full dim cycles when it cannot.
    const uint32_t dim = 4;

    I8Tensor w0(dim, dim), w1(dim, dim);
    for (uint32_t k = 0; k < dim; ++k) {
        for (uint32_t c = 0; c < dim; ++c) {
            w0.view().at(k, c) = static_cast<i8>(k + c + 1);
            w1.view().at(k, c) = static_cast<i8>(k * 2 + 1);
        }
    }
    I8Tensor a(dim, dim);
    for (uint32_t r = 0; r < dim; ++r) {
        for (uint32_t c = 0; c < dim; ++c) a.view().at(r, c) = static_cast<i8>(r + c);
    }

    // Runs the two-matmul sequence and reports the bubble charged to the *second*
    // weight load, along with both results.
    auto run_pair = [&](bool double_buffer, uint64_t& second_load_bubble,
                        std::vector<int>& r0, std::vector<int>& r1) {
        Config cfg = small_cfg(dim, 2);
        cfg.double_buffer = double_buffer;

        Mxu mxu(cfg);
        I32Tensor o0(dim, dim), o1(dim, dim);

        mxu.load_weights(w0.view());
        mxu.matmul(a.view(), o0.view(), false);

        const uint64_t before = mxu.cycle();
        mxu.load_weights(w1.view());
        second_load_bubble = mxu.cycle() - before;

        mxu.matmul(a.view(), o1.view(), false);

        r0.clear();
        r1.clear();
        for (uint32_t r = 0; r < dim; ++r) {
            for (uint32_t c = 0; c < dim; ++c) {
                r0.push_back(o0.view().at(r, c));
                r1.push_back(o1.view().at(r, c));
            }
        }
        return mxu.stats();
    };

    uint64_t db_bubble = 0, nodb_bubble = 0;
    std::vector<int> db_r0, db_r1, nodb_r0, nodb_r1;
    const MxuStats db   = run_pair(true, db_bubble, db_r0, db_r1);
    const MxuStats nodb = run_pair(false, nodb_bubble, nodb_r0, nodb_r1);

    REQUIRE(db_bubble == 0);
    REQUIRE(nodb_bubble == dim);
    REQUIRE(db.weight_load_bubble == 0);
    REQUIRE(nodb.weight_load_bubble == 2ull * dim);   // both loads are exposed

    // Two matmuls of dim rows each, and the whole difference between the two
    // configurations is the weight bubbles.
    const uint64_t compute = 2ull * (dim + 2ull * dim - 1ull);
    REQUIRE(db.cycles == compute);
    REQUIRE(nodb.cycles == compute + 2ull * dim);

    // Speed must not change the answer. This is the assertion that makes the
    // bubble counts above worth having: an array that skipped the weight load
    // entirely would also report a zero bubble.
    REQUIRE_MSG(db_r0 == nodb_r0, diff_vec("first matmul", db_r0, nodb_r0));
    REQUIRE_MSG(db_r1 == nodb_r1, diff_vec("second matmul", db_r1, nodb_r1));

    // And both agree with the oracle, so the plane switch really did make the
    // second tile resident instead of leaving the first one in place.
    {
        Config cfg = small_cfg(dim, 2);
        std::vector<i8> fw0, fw1, fa;
        for (uint32_t k = 0; k < dim; ++k) {
            for (uint32_t c = 0; c < dim; ++c) {
                fw0.push_back(w0.view().at(k, c));
                fw1.push_back(w1.view().at(k, c));
                fa.push_back(a.view().at(k, c));
            }
        }
        const std::vector<int> want0 = ref_matmul(cfg, fw0, fa, dim);
        const std::vector<int> want1 = ref_matmul(cfg, fw1, fa, dim);
        REQUIRE_MSG(db_r0 == want0, diff_vec("tile 0 vs oracle", db_r0, want0));
        REQUIRE_MSG(db_r1 == want1, diff_vec("tile 1 vs oracle", db_r1, want1));
        REQUIRE(want0 != want1);      // the two tiles really do differ
    }

    // Plane bookkeeping: with double buffering the active plane alternates, one
    // switch per load; without it, everything stays in plane 0.
    REQUIRE(db.plane_switches == 2);
    REQUIRE(nodb.plane_switches == 0);

    {
        Config cfg = small_cfg(dim, 2);
        cfg.double_buffer = true;
        Mxu mxu(cfg);
        I32Tensor out(dim, dim);

        REQUIRE(mxu.active_plane() == 0);
        REQUIRE(mxu.load_plane() == 1);
        mxu.load_weights(w0.view());
        REQUIRE(mxu.switch_pending());
        REQUIRE(mxu.active_plane() == 0);        // not yet; the matmul switches

        mxu.matmul(a.view(), out.view(), false);
        REQUIRE(!mxu.switch_pending());
        REQUIRE(mxu.active_plane() == 1);
        REQUIRE(mxu.load_plane() == 0);          // the next load goes to the other

        mxu.load_weights(w1.view());
        mxu.matmul(a.view(), out.view(), false);
        REQUIRE(mxu.active_plane() == 0);

        // The retired tile is still sitting in the shadow plane, untouched.
        REQUIRE(mxu.weight_at(0, 0, 1) == w0.view().at(0, 0));
    }

    // Two matmuls with no load in between reuse the resident tile and switch
    // nothing, so a plane switch is charged to a load and not to every matmul.
    {
        Config cfg = small_cfg(dim, 2);
        cfg.double_buffer = true;
        Mxu mxu(cfg);
        I32Tensor o0(dim, dim), o1(dim, dim);
        mxu.load_weights(w0.view());
        mxu.matmul(a.view(), o0.view(), false);
        mxu.matmul(a.view(), o1.view(), false);
        REQUIRE(mxu.stats().plane_switches == 1);
        REQUIRE(mxu.active_plane() == 1);

        std::vector<int> f0, f1;
        for (uint32_t r = 0; r < dim; ++r) {
            for (uint32_t c = 0; c < dim; ++c) {
                f0.push_back(o0.view().at(r, c));
                f1.push_back(o1.view().at(r, c));
            }
        }
        REQUIRE_MSG(f0 == f1, diff_vec("repeated matmul", f0, f1));
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

    // Two runs of the same program over the same inputs agree on every
    // comparand. This is the whole point of the phase: the comparison works
    // before there is a second implementation to point it at.
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
        return compare_runs("a", ma, ra, "b", mb, rb);
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

// -------------------------------------------------------------------- main ---
int main() {
    // Line-buffered, so a section header, its assertion failures and anything a
    // test writes to stderr stay in the order they happened even when the output
    // is piped to a file or a CI log.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    int passes = 0, fails = 0;
    for (auto& [name, fn] : test::registry()) {
        const int before = test::assertion_failures;
        std::printf("=== %s ===\n", name.c_str());
        fn();
        if (test::assertion_failures > before) ++fails;
        else                                   ++passes;
    }
    std::printf("\n%d section%s ok, %d failing (%d assertion failure%s)\n",
                passes, passes == 1 ? "" : "s",
                fails,
                test::assertion_failures, test::assertion_failures == 1 ? "" : "s");
    return test::assertion_failures ? 1 : 0;
}
