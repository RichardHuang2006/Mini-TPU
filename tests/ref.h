#pragma once

// The oracle: an eager tensor model. Every instruction is carried out in full the
// moment it is decoded, with no array, no FIFO, no double buffering and no timing.
// `Tpu` is checked against it on output bytes, accumulator snapshots at each Sync,
// and the retired count.
//
// The arithmetic below deliberately shares no code with the timed model, with one
// exception: quant.h. Two implementations that agree are evidence, one called twice
// is not, but bit-exact requantization is a contract rather than an independent
// guess, so both sides call the same helper on purpose.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "config.h"
#include "decoder.h"
#include "isa.h"
#include "quant.h"
#include "tensor.h"
#include "types.h"

namespace ref {

// ---------------------------------------------------------------------------
// Data layout. Fixed once, here, and matched by every later model.
//
//   resident weights   dim x dim int8, W[k][c]
//   activations        len x dim int8 at ub_addr, row-major, stride dim: A[r][k]
//   accumulator bank   dim x dim int32: acc[r][c] = sum_k A[r][k] * W[k][c]
//   activation output  int8 at ub_dst, row-major, stride = output columns
//
// One activation row is one input vector, and MatMul turns `len` input vectors into
// `len` output vectors. Hence `len` may not exceed a bank's row count: the bank
// holds the results.
//
// A matmul whose real K or N is smaller than the array relies on the unused weights
// being zero, which is what makes a zero-padded partial tile produce the right
// answer.
// ---------------------------------------------------------------------------

struct Options {
    uint64_t   max_insts = 1'000'000;   // runaway-program backstop
    bool       trace     = false;
    std::FILE* trace_out = stderr;
};

// Every accumulator bank, flattened, as of one Sync.
struct Snapshot {
    std::vector<i32> acc;
};

struct Result {
    std::size_t pc        = 0;   // next instruction; on halt/trap, the halting one
    uint32_t    exit_code = 0;
    uint64_t    retired   = 0;

    bool halted  = false;   // Halt
    bool trapped = false;   // illegal encoding, out-of-range operand, ran off the end
    bool budget  = false;   // hit max_insts: neither halted nor trapped

    std::string           trap_reason;
    std::vector<Snapshot> syncs;

    bool done() const { return halted || trapped; }
};

// All of the machine's state, public: this is an oracle, so being obvious matters
// more than being encapsulated.
struct Machine {
    Config cfg;

    std::vector<uint8_t> host;         // host memory
    std::vector<i8>      weight_mem;   // weight (DDR) memory
    std::vector<i8>      ub;           // Unified Buffer
    std::vector<i32>     acc;          // acc_banks * dim * dim, flattened
    std::vector<i8>      weights;      // the resident dim x dim tile

    Machine(const Config& c, std::size_t host_bytes = 1u << 16,
            std::size_t weight_bytes = 1u << 16)
        : cfg(c),
          host(host_bytes, 0),
          weight_mem(weight_bytes, 0),
          ub(c.ub_bytes, 0),
          acc(static_cast<std::size_t>(c.acc_banks) * c.dim * c.dim, 0),
          weights(static_cast<std::size_t>(c.dim) * c.dim, 0) {}

    uint32_t dim() const { return cfg.dim; }

    std::size_t bank_off(BankId b) const {
        return static_cast<std::size_t>(b) * cfg.dim * cfg.dim;
    }

    I32View acc_bank(BankId b) {
        return I32View(&acc[bank_off(b)], cfg.dim, cfg.dim);
    }
    ConstI32View acc_bank(BankId b) const {
        return ConstI32View(&acc[bank_off(b)], cfg.dim, cfg.dim);
    }

    I8View      weight_view()       { return I8View(weights.data(), cfg.dim, cfg.dim); }
    ConstI8View weight_view() const { return ConstI8View(weights.data(), cfg.dim, cfg.dim); }
};

namespace detail {

// `off + len <= size` without ever overflowing.
inline bool range_ok(std::size_t off, std::size_t len, std::size_t size) {
    return off <= size && len <= size - off;
}

}  // namespace detail

// Applied to the requantized int8 value, so ReLU6's bound is 6 in the output's
// own quantized units.
inline i8 apply_actfn(i8 v, ActFn fn) {
    switch (fn) {
        case ActFn::IDENTITY: return v;
        case ActFn::RELU:     return v < 0 ? i8{0} : v;
        case ActFn::RELU6:    return v < 0 ? i8{0} : (v > 6 ? i8{6} : v);
    }
    return v;
}

// Execute one instruction in full, updating state, the retired count, and the
// halt/trap flags.
inline void step(Machine& m, Result& st, const std::vector<RawInst>& prog,
                 const Options& opts) {
    const Decoded  d   = decode(prog[st.pc]);
    const uint32_t dim = m.cfg.dim;

    auto trap = [&](const char* why) {
        st.trapped     = true;
        st.trap_reason = why;
    };

    if (opts.trace && opts.trace_out) {
        std::fprintf(opts.trace_out, "%4zu: %s\n", st.pc, op_name(d.op));
    }

    switch (d.op) {
        case Op::READ_HOST: {
            if (!detail::range_ok(d.host_addr, d.bytes, m.host.size())) {
                trap("Read_Host_Memory: host range");
                break;
            }
            if (!detail::range_ok(d.ub_addr, d.bytes, m.ub.size())) {
                trap("Read_Host_Memory: ub range");
                break;
            }
            for (uint32_t i = 0; i < d.bytes; ++i) {
                m.ub[d.ub_addr + i] = static_cast<i8>(m.host[d.host_addr + i]);
            }
            break;
        }

        case Op::WRITE_HOST: {
            if (!detail::range_ok(d.ub_addr, d.bytes, m.ub.size())) {
                trap("Write_Host_Memory: ub range");
                break;
            }
            if (!detail::range_ok(d.host_addr, d.bytes, m.host.size())) {
                trap("Write_Host_Memory: host range");
                break;
            }
            for (uint32_t i = 0; i < d.bytes; ++i) {
                m.host[d.host_addr + i] = static_cast<uint8_t>(m.ub[d.ub_addr + i]);
            }
            break;
        }

        case Op::READ_WEIGHTS: {
            // The `tile` operand names a FIFO staging slot, a timing artifact:
            // with no FIFO to stage through, the resident weights are whatever the
            // most recent Read_Weights fetched. A timed model must end up with the
            // same tile resident.
            const std::size_t need = static_cast<std::size_t>(dim) * dim;
            if (!detail::range_ok(d.ddr_addr, need, m.weight_mem.size())) {
                trap("Read_Weights: ddr range");
                break;
            }
            for (std::size_t i = 0; i < need; ++i) {
                m.weights[i] = m.weight_mem[d.ddr_addr + i];
            }
            break;
        }

        case Op::MATMUL: {
            if (d.acc_bank >= m.cfg.acc_banks) { trap("MatMul: bank out of range"); break; }
            if (d.len > dim) { trap("MatMul: len exceeds accumulator rows"); break; }
            const std::size_t need = static_cast<std::size_t>(d.len) * dim;
            if (!detail::range_ok(d.ub_addr, need, m.ub.size())) {
                trap("MatMul: activation range");
                break;
            }
            const I32View bank = m.acc_bank(d.acc_bank);
            for (uint32_t r = 0; r < d.len; ++r) {
                for (uint32_t c = 0; c < dim; ++c) {
                    // Accumulated in int64 so the sum is well defined however many
                    // K-tiles pile into one bank; the store narrows to the int32
                    // the hardware bank holds.
                    int64_t sum = d.accumulate ? bank.at(r, c) : 0;
                    for (uint32_t k = 0; k < dim; ++k) {
                        const int64_t a = m.ub[d.ub_addr + static_cast<std::size_t>(r) * dim + k];
                        const int64_t w = m.weights[static_cast<std::size_t>(k) * dim + c];
                        sum += a * w;
                    }
                    bank.at(r, c) = static_cast<i32>(sum);
                }
            }
            break;
        }

        case Op::ACTIVATE: {
            if (d.acc_bank >= m.cfg.acc_banks) { trap("Activate: bank out of range"); break; }
            if (d.len > dim) { trap("Activate: len exceeds accumulator rows"); break; }

            const ConstI32View bank = m.acc_bank(d.acc_bank);

            // Bias, requantize, activation function: a len x dim int8 tile.
            std::vector<i8> tile(static_cast<std::size_t>(d.len) * dim, 0);
            for (uint32_t r = 0; r < d.len; ++r) {
                for (uint32_t c = 0; c < dim; ++c) {
                    const i8 q = quant::requantize_biased(bank.at(r, c), d.bias,
                                                          d.multiplier, d.shift);
                    tile[static_cast<std::size_t>(r) * dim + c] = apply_actfn(q, d.act);
                }
            }

            // Pool, if asked. Floor semantics: a window that does not divide the
            // input evenly drops the ragged edge, which is why the uneven case
            // gets a test of its own.
            uint32_t out_rows = d.len;
            uint32_t out_cols = dim;
            std::vector<i8> out;
            if (d.pool == Pool::NONE) {
                out = std::move(tile);
            } else {
                const uint32_t w = d.pool_window;
                const uint32_t s = d.pool_stride;
                if (w == 0 || s == 0)      { trap("Activate: zero pool window or stride"); break; }
                if (w > d.len || w > dim)  { trap("Activate: pool window larger than input"); break; }
                out_rows = (d.len - w) / s + 1;
                out_cols = (dim - w) / s + 1;
                out.assign(static_cast<std::size_t>(out_rows) * out_cols, 0);
                for (uint32_t orow = 0; orow < out_rows; ++orow) {
                    for (uint32_t ocol = 0; ocol < out_cols; ++ocol) {
                        const uint32_t r0 = orow * s;
                        const uint32_t c0 = ocol * s;
                        int64_t sum  = 0;
                        i8      best = tile[static_cast<std::size_t>(r0) * dim + c0];
                        for (uint32_t dr = 0; dr < w; ++dr) {
                            for (uint32_t dc = 0; dc < w; ++dc) {
                                const i8 v = tile[static_cast<std::size_t>(r0 + dr) * dim + (c0 + dc)];
                                sum += v;
                                if (v > best) best = v;
                            }
                        }
                        const std::size_t o = static_cast<std::size_t>(orow) * out_cols + ocol;
                        // Average pooling rounds the way requantization does,
                        // via the same helper.
                        out[o] = d.pool == Pool::MAX
                                   ? best
                                   : quant::saturate(quant::round_div(sum, static_cast<int64_t>(w) * w));
                    }
                }
            }

            if (!detail::range_ok(d.ub_addr, out.size(), m.ub.size())) {
                trap("Activate: output range");
                break;
            }
            for (std::size_t i = 0; i < out.size(); ++i) m.ub[d.ub_addr + i] = out[i];
            break;
        }

        case Op::SYNC: {
            // A barrier has no state of its own; it is where the differential test
            // compares accumulators mid-program.
            Snapshot s;
            s.acc = m.acc;
            st.syncs.push_back(std::move(s));
            break;
        }

        case Op::NOP:
            break;

        case Op::HALT:
            if (d.trap) trap("illegal opcode");
            else {
                st.halted    = true;
                st.exit_code = d.code;
            }
            break;
    }

    ++st.retired;
    if (!st.done()) ++st.pc;   // a halt or trap reports its own instruction
}

// Run until the program halts, traps, or runs out of budget.
inline Result run(Machine& m, const std::vector<RawInst>& prog,
                  const Options& opts = Options{}) {
    Result st;
    while (!st.done()) {
        if (st.pc >= prog.size()) {
            // A well-formed program ends in Halt; falling off the end is a bug in
            // the program, reported rather than stopped over quietly.
            st.trapped     = true;
            st.trap_reason = "ran past the end of the program";
            break;
        }
        if (st.retired >= opts.max_insts) { st.budget = true; break; }
        step(m, st, prog, opts);
    }
    return st;
}

}  // namespace ref
