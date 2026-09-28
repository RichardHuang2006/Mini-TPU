/// Encoding, decoding and disassembly of the TPUv1 instructions.

#include "isa/isa.h"

#include <cstdio>
#include <stdexcept>

namespace {

// Host transfers: ub_row in bytes 3-5, host_row in 6-9, rows in 10-11, all little-endian.
constexpr u32 kUbRowAt   = 3;
constexpr u32 kHostRowAt = 6;
constexpr u32 kRowsAt    = 10;

void put(InstrBytes& b, u32 at, u32 width, u64 value, const char* field) {
    if (value >> (8 * width) != 0)
        throw std::out_of_range(std::string(field) + "=" + std::to_string(value) + " does not fit in " +
                                std::to_string(8 * width) + " bits");
    for (u32 i = 0; i < width; ++i) b[at + i] = static_cast<u8>(value >> (8 * i));
}

u32 get(const InstrBytes& b, u32 at, u32 width) {
    u32 v = 0;
    for (u32 i = 0; i < width; ++i) v |= u32{b[at + i]} << (8 * i);
    return v;
}

std::string hex(u32 v) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "0x%X", v);
    return buf;
}

bool is_host_transfer(Op op) { return op == Op::ReadHostMemory || op == Op::WriteHostMemory; }

}  // namespace

const char* op_name(Op op) {
    switch (op) {
        case Op::Nop:             return "Nop";
        case Op::Halt:            return "Halt";
        case Op::Sync:            return "Sync";
        case Op::ReadHostMemory:  return "Read_Host_Memory";
        case Op::WriteHostMemory: return "Write_Host_Memory";
    }
    return "?";
}

InstrBytes encode(const Instr& in) {
    InstrBytes b{};
    b[0] = static_cast<u8>(in.op);
    if (is_host_transfer(in.op)) {
        put(b, kUbRowAt,   3, in.ub_row,   "ub");
        put(b, kHostRowAt, 4, in.host_row, "host");
        put(b, kRowsAt,    2, in.rows,     "rows");
    }
    return b;
}

Instr decode(const InstrBytes& bytes) {
    Instr in;
    in.op = static_cast<Op>(bytes[0]);
    switch (in.op) {
        case Op::Nop:
        case Op::Halt:
        case Op::Sync:
            break;
        case Op::ReadHostMemory:
        case Op::WriteHostMemory:
            in.ub_row   = get(bytes, kUbRowAt,   3);
            in.host_row = get(bytes, kHostRowAt, 4);
            in.rows     = get(bytes, kRowsAt,    2);
            break;
        default:
            throw std::invalid_argument("unknown opcode " + hex(bytes[0]));
    }
    return in;
}

std::string disasm(const Instr& in) {
    const std::string ub   = "ub=" + hex(in.ub_row);
    const std::string host = "host=" + hex(in.host_row);
    const std::string rows = "rows=" + std::to_string(in.rows);
    switch (in.op) {
        case Op::ReadHostMemory:  return std::string(op_name(in.op)) + " " + host + " " + ub + " " + rows;
        case Op::WriteHostMemory: return std::string(op_name(in.op)) + " " + ub + " " + host + " " + rows;
        case Op::Nop:
        case Op::Halt:
        case Op::Sync:            return op_name(in.op);
    }
    return "?";
}
