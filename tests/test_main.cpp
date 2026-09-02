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

#include "config.h"
#include "datapath.h"
#include "isa.h"
#include "loader.h"
#include "systolic_array.h"
#include "storage.h"
#include "transfer.h"
#include "tpu.h"
#include "stats.h"
#include "tpuasm.h"
#include "ref.h"
#include "workloads.h"   // the layer tiler, shared with tools/gen_examples.cpp

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

// A deterministic byte source. Taking the top bits of an LCG spreads values over
// the whole int8 range including -128, which has no positive counterpart and is
// where a sloppy negation shows up. The same generator the bundled workloads use,
// so a test and an example built from one seed hold the same bytes.
using Lcg = wl::Rng;

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

// Everything a run is compared on, lifted out of whichever model produced it so
// one comparison serves both the reference model and the timed machine.
struct RunView {
    bool     halted    = false;
    bool     trapped   = false;
    bool     budget    = false;
    uint32_t exit_code = 0;
    uint64_t retired   = 0;

    std::string trap_reason;

    std::vector<std::vector<i32>> syncs;   // accumulators at each barrier
    std::vector<i32>              acc;     // final accumulators
    std::vector<i8>               ub;
    std::vector<uint8_t>          host;
};

RunView view_of(const ref::Machine& m, const ref::Result& r) {
    RunView v;
    v.halted      = r.halted;
    v.trapped     = r.trapped;
    v.budget      = r.budget;
    v.exit_code   = r.exit_code;
    v.retired     = r.retired;
    v.trap_reason = r.trap_reason;
    for (const ref::Snapshot& s : r.syncs) v.syncs.push_back(s.acc);
    v.acc  = m.acc;
    v.ub   = m.ub;
    v.host = m.host;
    return v;
}

RunView view_of(Tpu& t, const TpuResult& r) {
    RunView v;
    v.halted      = r.halted;
    v.trapped     = r.trapped;
    v.budget      = r.budget;
    v.exit_code   = r.exit_code;
    v.retired     = r.retired;
    v.trap_reason = r.trap_reason;
    for (const TpuSnapshot& s : r.syncs) v.syncs.push_back(s.acc);
    v.acc = t.acc().raw();
    v.ub.reserve(t.ub().bytes());
    for (uint32_t i = 0; i < t.ub().bytes(); ++i) v.ub.push_back(t.ub().at(i));
    v.host = t.host();
    return v;
}

// Every comparand of the differential check: output bytes, accumulators at each
// Sync, and the retired count. Returns "" when the two runs agree.
std::string compare_runs(const char* a_name, const RunView& ra,
                         const char* b_name, const RunView& rb) {
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
            const auto& av = ra.syncs[s];
            const auto& bv = rb.syncs[s];
            if (av == bv) continue;
            const std::size_t i = first_diff(av, bv);
            os << "    sync " << s << " accumulator[" << i << "]: " << a_name << "="
               << (i < av.size() ? num(static_cast<uint64_t>(av[i])) : "-") << "  " << b_name
               << "=" << (i < bv.size() ? num(static_cast<uint64_t>(bv[i])) : "-") << "\n";
        }
    }

    if (ra.acc != rb.acc) {
        const std::size_t i = first_diff(ra.acc, rb.acc);
        note(("final accumulator[" + num(i) + "]").c_str(),
             num(static_cast<uint64_t>(ra.acc[i])), num(static_cast<uint64_t>(rb.acc[i])));
    }
    if (ra.ub != rb.ub) {
        const std::size_t i = first_diff(ra.ub, rb.ub);
        note(("unified buffer[" + num(i) + "]").c_str(),
             num(static_cast<uint64_t>(ra.ub[i])), num(static_cast<uint64_t>(rb.ub[i])));
    }
    if (ra.host != rb.host) {
        const std::size_t i = first_diff(ra.host, rb.host);
        note(("host memory[" + num(i) + "]").c_str(), num(ra.host[i]), num(rb.host[i]));
    }
    return os.str();
}

using Setup = std::function<void(ref::Machine&)>;

// The same initial state, applied to the timed machine. Kept separate from Setup
// so a test writes its inputs once and both models see them.
using TpuSetup = std::function<void(Tpu&)>;

// Two reference runs of the same program. This is what proved the comparison
// itself works before there was a second implementation to point it at.
std::string diff_run(const std::vector<RawInst>& prog, const Config& cfg,
                     const Setup& setup = {}) {
    ref::Machine ma(cfg), mb(cfg);
    if (setup) { setup(ma); setup(mb); }
    const ref::Result ra = ref::run(ma, prog);
    const ref::Result rb = ref::run(mb, prog);
    return compare_runs("ref", view_of(ma, ra), "ref-again", view_of(mb, rb));
}

// The real thing: the cycle-accurate machine against the oracle. Returns "" when
// they agree on every comparand.
std::string diff_tpu(const std::vector<RawInst>& prog, const Config& cfg,
                     const Setup& rsetup = {}, const TpuSetup& tsetup = {},
                     std::size_t host_bytes = 1u << 16,
                     std::size_t weight_bytes = 1u << 16) {
    ref::Machine m(cfg, host_bytes, weight_bytes);
    Tpu          t(cfg, host_bytes, weight_bytes);
    if (rsetup) rsetup(m);
    if (tsetup) tsetup(t);

    const ref::Result rr = ref::run(m, prog);
    const TpuResult   tr = t.run(prog);
    return compare_runs("ref", view_of(m, rr), "tpu", view_of(t, tr));
}

// Copy a byte image into host memory and weight memory on both models, so the
// two setups cannot drift apart.
struct Inputs {
    HostAddr        host_at = 0;
    std::vector<i8> host_bytes;
    uint32_t        ddr_at = 0;
    std::vector<i8> weights;
    UbAddr          ub_at = 0;
    std::vector<i8> ub_bytes;

    // Accumulator contents, row-major into one bank. Planting these lets a test
    // choose the exact int32 values the activation pipeline sees rather than
    // whatever a matmul happens to produce, which is the only way to sweep densely
    // across a clamp boundary.
    BankId           acc_at = 0;
    std::vector<i32> acc_vals;
};

Setup ref_setup(const Inputs& in) {
    return [in](ref::Machine& m) {
        for (std::size_t i = 0; i < in.host_bytes.size(); ++i) {
            m.host[in.host_at + i] = static_cast<uint8_t>(in.host_bytes[i]);
        }
        for (std::size_t i = 0; i < in.weights.size(); ++i) {
            m.weight_mem[in.ddr_at + i] = in.weights[i];
        }
        for (std::size_t i = 0; i < in.ub_bytes.size(); ++i) {
            m.ub[in.ub_at + i] = in.ub_bytes[i];
        }
        const std::size_t per = static_cast<std::size_t>(m.cfg.dim) * m.cfg.dim;
        for (std::size_t i = 0; i < in.acc_vals.size(); ++i) {
            m.acc[in.acc_at * per + i] = in.acc_vals[i];
        }
    };
}

TpuSetup tpu_setup(const Inputs& in) {
    return [in](Tpu& t) {
        for (std::size_t i = 0; i < in.host_bytes.size(); ++i) {
            t.host()[in.host_at + i] = static_cast<uint8_t>(in.host_bytes[i]);
        }
        for (std::size_t i = 0; i < in.weights.size(); ++i) {
            t.weight_mem()[in.ddr_at + i] = in.weights[i];
        }
        for (std::size_t i = 0; i < in.ub_bytes.size(); ++i) {
            t.ub().at(static_cast<UbAddr>(in.ub_at + i)) = in.ub_bytes[i];
        }
        if (!in.acc_vals.empty()) {
            const I32View bank = t.acc().bank(in.acc_at);
            const uint32_t dim = bank.cols();
            for (std::size_t i = 0; i < in.acc_vals.size(); ++i) {
                bank.at(static_cast<uint32_t>(i / dim), static_cast<uint32_t>(i % dim)) =
                    in.acc_vals[i];
            }
        }
    };
}

// Pseudo-random int8 payloads for a differential workload.
Inputs random_inputs(uint32_t seed, uint32_t dim, uint32_t len, std::size_t tiles = 1) {
    Lcg rng(seed);
    Inputs in;
    in.host_at = 0x100;
    in.host_bytes.resize(static_cast<std::size_t>(len) * dim);
    for (i8& v : in.host_bytes) v = rng.byte();
    in.weights.resize(static_cast<std::size_t>(dim) * dim * tiles);
    for (i8& v : in.weights) v = rng.byte();
    return in;
}

// ---- systolic array helpers ----------------------------------------------

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

    // Documented defaults.
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
    // oracle, over a spread of shapes and data covering both int8 extremes. Nothing
    // here recomputes the matmul locally; the expectation comes from ref.h.
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

    // Fill and drain is pure overhead, so utilization is len / (len + 2*dim - 1) and
    // only a long stream amortizes it away. Double buffering is on to keep
    // weight-load bubbles out of the clock, leaving fill and drain as the only thing
    // measured.
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

    // And both agree with the oracle, so switching planes really did make the
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

// ---------------------------------------------------------- @section("ub") ---
SECTION("ub") {
    Config cfg = small_cfg();
    cfg.ub_bytes = 256;
    cfg.ub_banks = 4;

    UnifiedBuffer ub(cfg);
    REQUIRE(ub.bytes() == 256);
    REQUIRE(ub.banks() == 4);

    // Low-order interleaving: consecutive bytes land in consecutive banks, so a
    // contiguous row spreads across every port rather than piling into one.
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

    // Bounds. The last row needs only `cols` bytes, so a tile ending flush
    // against the top of the buffer fits.
    REQUIRE(ub.tile_fits(0, 4, 4, 4));
    REQUIRE(ub.tile_fits(252, 1, 4, 4));
    // 2 rows at pitch 8 span 8 + 4 = 12 bytes, not 16: the last row needs only
    // its own columns, so this tile ends exactly at 256 and fits.
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

    // Port budget. Two reads to one bank collide; two reads to different banks
    // do not.
    using Port   = UnifiedBuffer::Port;
    using Access = UnifiedBuffer::Access;

    REQUIRE(ub.port_conflict(std::vector<UbAddr>{0, 4}));         // both bank 0
    REQUIRE(!ub.port_conflict(std::vector<UbAddr>{0, 1, 2, 3}));  // one each
    REQUIRE(!ub.port_conflict(std::vector<UbAddr>{}));
    REQUIRE(!ub.port_conflict(std::vector<UbAddr>{7}));
    REQUIRE(ub.port_conflict(std::vector<UbAddr>{0, 1, 2, 3, 8}));

    // A read and a write to the same bank proceed together: each bank has one
    // port of each. This is the distinction that makes the buffer able to feed
    // the array and take activation output in the same cycle.
    REQUIRE(!ub.port_conflict(std::vector<Access>{{0, Port::READ}, {4, Port::WRITE}}));
    REQUIRE(ub.port_conflict(std::vector<Access>{{0, Port::WRITE}, {4, Port::WRITE}}));
    REQUIRE(!ub.port_conflict(std::vector<Access>{{0, Port::READ}, {1, Port::WRITE}}));

    // A dim-wide row read is conflict-free exactly when the row spans no more
    // banks than there are: this is the property the interleaving exists for.
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

// ------------------------------------------------ @section("accumulators") ---
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

    // A locked bank refuses reads and accumulates until unlocked, and each
    // refusal is counted as a hazard.
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

    // Locking one bank leaves the others alone, which is what makes a bank the
    // unit of the interlock.
    acc.lock(0);
    REQUIRE(!acc.read(0, got.view()));
    REQUIRE(acc.read(2, got.view()));
    acc.unlock(0);

    // An invalid bank is refused but is not a hazard: it is a malformed
    // instruction, not a stall.
    const uint64_t h = acc.stats().hazards;
    REQUIRE(!acc.read(9, got.view()));
    REQUIRE(acc.stats().hazards == h);

    // ---- K-tiling ---------------------------------------------------------
    // A matmul with K four times the array width is split into four tiles that add
    // into one bank, and the result must equal the full-K product. The expectation
    // comes from a direct triple loop over the whole of K, a different computation
    // from the one the array and the banks perform between them.
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

// ------------------------------------------------- @section("weight_fifo") ---
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

    // A refill occupies its slot immediately but is not poppable until the
    // latency has elapsed: that gap is the whole reason to stage tiles early.
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

    // ---- a 1-deep FIFO serializes back-to-back weight loads ---------------
    // With depth 1 the second refill cannot start until the first tile has been
    // popped, so two loads cost two full latencies instead of overlapping into one.
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

// --------------------------------------------------------- @section("dma") ---
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

    // One transfer at a time: a second start while busy is refused rather than
    // silently interleaving.
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

    // ---- a slow bus makes a small transfer the bottleneck -----------------
    // The same 256-byte load that hides under one matmul at full bandwidth
    // dominates it at one byte per cycle, and Config::dma_bound agrees.
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

        // The roofline check reaches the same verdict from arithmetic intensity
        // alone: len*dim*dim MACs against `bytes` moved.
        const std::size_t macs = static_cast<std::size_t>(len) * dim * dim;
        REQUIRE(!fast.dma_bound(macs, bytes));
        REQUIRE(slow.dma_bound(macs, bytes));
    }
}

// --------------------------------------------------- @section("tpu_units") ---
SECTION("tpu_units") {
    // The units wired together, driven by hand and stepped through tick(): DMA
    // the activations in, stage and load a weight tile, matmul, read the bank.
    // The answer has to match the oracle running the equivalent program.
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

    // Stage the weight tile. It is not poppable until the DDR latency elapses,
    // so a load attempted immediately fails -- that is weight_fifo_empty.
    REQUIRE(tpu.stage_weights(0, dim, dim));
    REQUIRE(tpu.weight_fifo().empty());
    REQUIRE(!tpu.load_weights_from_fifo());
    REQUIRE(tpu.weight_fifo().stats().empty_stalls == 1);

    REQUIRE(tpu.run_until([&] { return !tpu.weight_fifo().empty(); }));
    REQUIRE(tpu.cycle() == 4 + 6);
    REQUIRE(tpu.load_weights_from_fifo());

    // Without double buffering the load costs the array dim cycles, and the
    // machine's clock reflects it.
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

    // DMA the result's bytes back out to prove the return path is wired: write
    // the bank's low byte per element into the buffer, then out to the host.
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

    // The array's clock never lags the machine's, across every kind of operation.
    // Tpu::matmul and load_weights_from_fifo each idle the array up to the machine
    // clock before using it, which the current API cannot violate since tick() keeps
    // the two in step. The guards exist because the sequencer advances the clock
    // while the array waits on a hazard; this assertion catches it if that stops
    // holding.
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

        // Idling is counted, so time the array spent waiting on the DMA and the
        // weight refill shows up as lost utilization rather than vanishing.
        REQUIRE(t.mxu().stats().cycles == t.cycle());
        REQUIRE(t.mxu().utilization() < 1.0);
        REQUIRE(t.mxu().stats().useful_macs == static_cast<uint64_t>(len) * dim * dim);
    }

    // ---- rejections, so a malformed operation stalls rather than corrupts ---
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

    // ---- K-tiling through the machine -------------------------------------
    // Two accumulating matmuls into one bank, driven by hand, against the oracle
    // doing the same thing eagerly.
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

// ------------------------------------------------------ tiling helpers -------

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

std::string tensor_mismatch(const std::vector<i8>& got, const std::vector<i8>& want,
                            uint32_t rows, uint32_t cols) {
    if (got == want) return "";
    std::ostringstream os;
    os << "    tiled result differs from golden (" << rows << "x" << cols << ")\n";
    int shown = 0;
    for (std::size_t i = 0; i < want.size() && shown < 6; ++i) {
        if (i < got.size() && got[i] == want[i]) continue;
        os << "      [" << (i / cols) << "," << (i % cols) << "] got "
           << (i < got.size() ? static_cast<int>(got[i]) : 0) << " want "
           << static_cast<int>(want[i]) << "\n";
        ++shown;
    }
    std::size_t bad = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        if (i >= got.size() || got[i] != want[i]) ++bad;
    }
    os << "      " << bad << " of " << want.size() << " elements differ\n";
    return os.str();
}

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

// ------------------------------------------------ characterization helpers ---

// Run a shipped workload on a given configuration and gather its statistics.
struct Measured {
    stats::Stats stats;
    std::vector<i8> got;
    bool halted = false;
    uint64_t cycles = 0;
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
        }
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
