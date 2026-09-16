#pragma once

// The section registry, the REQUIRE macros, and the helpers shared by every
// test translation unit. A SECTION() registers itself at static-init time.

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

// For failures where the expression alone says nothing useful.
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

// A machine small enough to multiply on paper.
inline Config small_cfg(uint32_t dim = 4, uint32_t banks = 2) {
    Config c;
    c.dim       = dim;
    c.acc_banks = banks;
    c.ub_bytes  = 4096;
    return c;
}

// The same generator the bundled workloads use, so one seed gives both the
// same bytes; it spans the whole int8 range, -128 included.
using Lcg = wl::Rng;

inline std::string show(const std::vector<int>& v) {
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
inline std::string diff_vec(const char* what, const std::vector<int>& got,
                            const std::vector<int>& want) {
    if (got == want) return "";
    std::ostringstream os;
    os << "    " << what << " got  " << show(got) << "\n"
       << "    " << what << " want " << show(want) << "\n";
    return os.str();
}

// The reference model's state is public, so a test can set up exactly the state
// it means to exercise rather than arranging a program to produce it.
inline void put_bytes(ref::Machine& m, UbAddr at, std::initializer_list<int> vals) {
    std::size_t i = 0;
    for (const int v : vals) m.ub[at + i++] = static_cast<i8>(v);
}

inline void put_weights(ref::Machine& m, uint32_t at, std::initializer_list<int> vals) {
    std::size_t i = 0;
    for (const int v : vals) m.weight_mem[at + i++] = static_cast<i8>(v);
}

inline void put_acc(ref::Machine& m, BankId b, std::initializer_list<int> vals) {
    const I32View bank = m.acc_bank(b);
    std::size_t i = 0;
    for (const int v : vals) {
        bank.at(static_cast<uint32_t>(i / m.dim()), static_cast<uint32_t>(i % m.dim())) = v;
        ++i;
    }
}

inline std::vector<int> ub_slice(const ref::Machine& m, UbAddr at, std::size_t n) {
    std::vector<int> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) out.push_back(m.ub[at + i]);
    return out;
}

inline std::vector<int> host_slice(const ref::Machine& m, HostAddr at, std::size_t n) {
    std::vector<int> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) out.push_back(static_cast<int8_t>(m.host[at + i]));
    return out;
}

inline std::vector<int> acc_slice(const ref::Machine& m, BankId b, uint32_t rows, uint32_t cols) {
    const ConstI32View bank = m.acc_bank(b);
    std::vector<int> out;
    out.reserve(static_cast<std::size_t>(rows) * cols);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) out.push_back(bank.at(r, c));
    }
    return out;
}

template <typename T>
std::size_t first_diff(const std::vector<T>& a, const std::vector<T>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) return i;
    }
    return n;   // a common prefix; the sizes are what differ
}

// Everything a run is compared on, lifted out of whichever model produced it.
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

inline RunView view_of(const ref::Machine& m, const ref::Result& r) {
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

inline RunView view_of(Tpu& t, const TpuResult& r) {
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

// Returns "" when the two runs agree on every comparand.
inline std::string compare_runs(const char* a_name, const RunView& ra,
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

// The same initial state, applied to the timed machine.
using TpuSetup = std::function<void(Tpu&)>;

// Two reference runs of the same program, which checks the comparison itself.
inline std::string diff_run(const std::vector<RawInst>& prog, const Config& cfg,
                            const Setup& setup = {}) {
    ref::Machine ma(cfg), mb(cfg);
    if (setup) { setup(ma); setup(mb); }
    const ref::Result ra = ref::run(ma, prog);
    const ref::Result rb = ref::run(mb, prog);
    return compare_runs("ref", view_of(ma, ra), "ref-again", view_of(mb, rb));
}

// The cycle-accurate machine against the oracle.
inline std::string diff_tpu(const std::vector<RawInst>& prog, const Config& cfg,
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

// One byte image for both models, so their setups cannot drift apart.
struct Inputs {
    HostAddr        host_at = 0;
    std::vector<i8> host_bytes;
    uint32_t        ddr_at = 0;
    std::vector<i8> weights;
    UbAddr          ub_at = 0;
    std::vector<i8> ub_bytes;

    // Accumulator contents, row-major into one bank, so a test can choose the
    // exact int32 values the activation pipeline sees.
    BankId           acc_at = 0;
    std::vector<i32> acc_vals;
};

inline Setup ref_setup(const Inputs& in) {
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

inline TpuSetup tpu_setup(const Inputs& in) {
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
inline Inputs random_inputs(uint32_t seed, uint32_t dim, uint32_t len, std::size_t tiles = 1) {
    Lcg rng(seed);
    Inputs in;
    in.host_at = 0x100;
    in.host_bytes.resize(static_cast<std::size_t>(len) * dim);
    for (i8& v : in.host_bytes) v = rng.byte();
    in.weights.resize(static_cast<std::size_t>(dim) * dim * tiles);
    for (i8& v : in.weights) v = rng.byte();
    return in;
}

// The oracle's answer, from running Read_Weights + MatMul through ref.h.
inline std::vector<int> ref_matmul(const Config& cfg, const std::vector<i8>& weights,
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

// Detail block for a tiled result that differs from its golden answer.
inline std::string tensor_mismatch(const std::vector<i8>& got, const std::vector<i8>& want,
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
