#pragma once

// Header-only builder for Mini-TPU programs, so the tests need no external
// tooling to produce an instruction stream.
//
// Operands are named regions rather than bare byte offsets, so a test reads like a
// short program instead of a hand-computed set of Unified Buffer offsets.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "datapath.h"
#include "isa.h"

namespace tpuasm {

// A named slice of the Unified Buffer. Converts to its own address, so it can
// be passed anywhere an operand is expected.
struct Region {
    UbAddr   addr  = 0;
    uint32_t bytes = 0;

    operator UbAddr() const { return addr; }
};

// Bump allocator over the Unified Buffer, so a test asks for the regions it needs
// rather than computing offsets.
class UbAlloc {
public:
    explicit UbAlloc(UbAddr base = 0, uint32_t align = 4) : next_(base), align_(align) {}

    Region alloc(uint32_t bytes) {
        const Region r{next_, bytes};
        next_ += round_up(bytes);
        return r;
    }

    // An int8 tile of `rows` x `cols`.
    Region tile(uint32_t rows, uint32_t cols) { return alloc(rows * cols); }

    UbAddr next() const { return next_; }

private:
    uint32_t round_up(uint32_t n) const {
        return align_ == 0 ? n : (n + align_ - 1) / align_ * align_;
    }

    UbAddr   next_;
    uint32_t align_;
};

// Every operand of Activate, named at the call site rather than positional.
struct ActArgs {
    BankId   acc         = 0;
    UbAddr   dst         = 0;
    uint32_t len         = 0;   // accumulator rows to process
    ActFn    fn          = ActFn::IDENTITY;
    i32      bias        = 0;
    i32      multiplier  = 1;
    uint32_t shift       = 0;
    Pool     pool        = Pool::NONE;
    uint32_t pool_window = 1;
    uint32_t pool_stride = 1;
};

// Instruction stream under construction. Every method returns *this so a
// program reads as one chained statement.
class Program {
public:
    Program& read_host(HostAddr host, UbAddr ub, uint32_t bytes) {
        Decoded d;
        d.op        = Op::READ_HOST;
        d.host_addr = host;
        d.ub_addr   = ub;
        d.bytes     = bytes;
        return emit(d);
    }

    Program& read_weights(uint32_t ddr_addr, TileId tile = 0) {
        Decoded d;
        d.op       = Op::READ_WEIGHTS;
        d.ddr_addr = ddr_addr;
        d.tile     = tile;
        return emit(d);
    }

    Program& matmul(UbAddr src, uint32_t len, BankId acc, bool accumulate = false) {
        Decoded d;
        d.op         = Op::MATMUL;
        d.ub_addr    = src;
        d.len        = len;
        d.acc_bank   = acc;
        d.accumulate = accumulate;
        return emit(d);
    }

    Program& activate(const ActArgs& a) {
        Decoded d;
        d.op          = Op::ACTIVATE;
        d.acc_bank    = a.acc;
        d.ub_addr     = a.dst;
        d.len         = a.len;
        d.act         = a.fn;
        d.bias        = a.bias;
        d.multiplier  = a.multiplier;
        d.shift       = a.shift;
        d.pool        = a.pool;
        d.pool_window = a.pool_window;
        d.pool_stride = a.pool_stride;
        return emit(d);
    }

    // The common case: scale, apply a function, no bias and no pooling.
    Program& activate(BankId acc, UbAddr dst, uint32_t len, ActFn fn,
                      i32 multiplier = 1, uint32_t shift = 0) {
        ActArgs a;
        a.acc        = acc;
        a.dst        = dst;
        a.len        = len;
        a.fn         = fn;
        a.multiplier = multiplier;
        a.shift      = shift;
        return activate(a);
    }

    Program& write_host(UbAddr ub, HostAddr host, uint32_t bytes) {
        Decoded d;
        d.op        = Op::WRITE_HOST;
        d.ub_addr   = ub;
        d.host_addr = host;
        d.bytes     = bytes;
        return emit(d);
    }

    Program& sync() {
        Decoded d;
        d.op = Op::SYNC;
        return emit(d);
    }

    Program& nop() {
        Decoded d;
        d.op = Op::NOP;
        return emit(d);
    }

    Program& halt(uint32_t code = 0) {
        Decoded d;
        d.op   = Op::HALT;
        d.code = code;
        return emit(d);
    }

    // Escape hatch for the illegal-encoding tests: emit a word directly.
    Program& raw(const RawInst& inst) {
        code_.push_back(inst);
        return *this;
    }

    const std::vector<RawInst>& code() const { return code_; }
    std::vector<RawInst>        assemble() const { return code_; }
    std::size_t                 size() const { return code_.size(); }

private:
    Program& emit(const Decoded& d) {
        code_.push_back(encode(d));
        return *this;
    }

    std::vector<RawInst> code_;
};

}  // namespace tpuasm
