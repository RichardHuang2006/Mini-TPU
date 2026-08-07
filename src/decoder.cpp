#include "decoder.h"

#include "types.h"

Decoded decode(const RawInst& raw) {
    Decoded d;

    const uint32_t opcode = (raw.word[0] >> isa::OPCODE_SHIFT) & isa::OPCODE_MASK;
    if (opcode > isa::MAX_OPCODE) {
        d.op   = Op::HALT;
        d.trap = true;
        return d;
    }
    d.op = static_cast<Op>(opcode);

    // Flag fields are extracted unconditionally; they are only *meaningful* for
    // the opcodes that define them, and leaving them at their encoded value
    // keeps the decode branch-free apart from the operand switch below.
    d.accumulate  = ((raw.word[0] >> isa::ACC_SHIFT) & 1u) != 0;
    d.act         = static_cast<ActFn>((raw.word[0] >> isa::ACTFN_SHIFT) & isa::ACTFN_MASK);
    d.pool        = static_cast<Pool>((raw.word[0] >> isa::POOL_SHIFT) & isa::POOL_MASK);
    d.shift       = (raw.word[0] >> isa::SHIFT_SHIFT)  & isa::SHIFT_MASK;
    d.pool_window = (raw.word[0] >> isa::WINDOW_SHIFT) & isa::WINDOW_MASK;
    d.pool_stride = (raw.word[0] >> isa::STRIDE_SHIFT) & isa::STRIDE_MASK;

    switch (d.op) {
        case Op::READ_HOST:
            d.host_addr = raw.word[1];
            d.ub_addr   = raw.word[2];
            d.bytes     = raw.word[3];
            break;
        case Op::WRITE_HOST:
            d.ub_addr   = raw.word[1];
            d.host_addr = raw.word[2];
            d.bytes     = raw.word[3];
            break;
        case Op::READ_WEIGHTS:
            d.ddr_addr = raw.word[1];
            d.tile     = raw.word[2];
            break;
        case Op::MATMUL:
            d.ub_addr  = raw.word[1];
            d.len      = raw.word[2];
            d.acc_bank = raw.word[3];
            break;
        case Op::ACTIVATE:
            d.acc_bank   = raw.word[1];
            d.ub_addr    = raw.word[2];
            d.len        = raw.word[3];
            d.bias       = static_cast<i32>(raw.word[4]);
            d.multiplier = static_cast<i32>(raw.word[5]);
            break;
        case Op::HALT:
            d.code = raw.word[1];
            break;
        case Op::SYNC:
        case Op::NOP:
            break;
    }
    return d;
}
