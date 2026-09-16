#include "isa.h"

#include <cstdio>

Decoded decode(const RawInst& raw) {
    Decoded d;

    const uint32_t opcode = (raw.word[0] >> isa::OPCODE_SHIFT) & isa::OPCODE_MASK;
    if (opcode > isa::MAX_OPCODE) {
        d.op   = Op::HALT;
        d.trap = true;
        return d;
    }
    d.op = static_cast<Op>(opcode);

    // Extracted unconditionally; only meaningful for the opcodes that define them.
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

std::string disasm(const Decoded& d) {
    char buf[192];
    switch (d.op) {
        case Op::READ_HOST:
            std::snprintf(buf, sizeof buf, "%-18s host=0x%08X ub=0x%08X bytes=%u",
                          op_name(d.op), d.host_addr, d.ub_addr, d.bytes);
            break;
        case Op::WRITE_HOST:
            std::snprintf(buf, sizeof buf, "%-18s ub=0x%08X host=0x%08X bytes=%u",
                          op_name(d.op), d.ub_addr, d.host_addr, d.bytes);
            break;
        case Op::READ_WEIGHTS:
            std::snprintf(buf, sizeof buf, "%-18s ddr=0x%08X tile=%u",
                          op_name(d.op), d.ddr_addr, d.tile);
            break;
        case Op::MATMUL:
            std::snprintf(buf, sizeof buf, "%-18s ub=0x%08X len=%u acc=%u%s",
                          op_name(d.op), d.ub_addr, d.len, d.acc_bank,
                          d.accumulate ? " accumulate" : "");
            break;
        case Op::ACTIVATE: {
            // The window and stride only mean anything when pooling is on.
            char pool_buf[48] = "";
            if (d.pool != Pool::NONE) {
                std::snprintf(pool_buf, sizeof pool_buf, " pool=%s(%ux%u/%u)",
                              pool_name(d.pool), d.pool_window, d.pool_window,
                              d.pool_stride);
            }
            std::snprintf(buf, sizeof buf,
                          "%-18s acc=%u ub=0x%08X len=%u bias=%d mult=%d shift=%u %s%s",
                          op_name(d.op), d.acc_bank, d.ub_addr, d.len, d.bias, d.multiplier,
                          d.shift, actfn_name(d.act), pool_buf);
            break;
        }
        case Op::HALT:
            std::snprintf(buf, sizeof buf, "%-18s code=%u%s", op_name(d.op), d.code,
                          d.trap ? "  (illegal opcode)" : "");
            break;
        case Op::SYNC:
        case Op::NOP:
            std::snprintf(buf, sizeof buf, "%s", op_name(d.op));
            break;
    }
    return std::string(buf);
}
