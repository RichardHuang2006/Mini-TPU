/// Encoding, decoding and disassembly of the TPUv1 instructions: one switch case per instruction, fields inline.

#include "isa/isa.h"

#include <cstdio>
#include <stdexcept>

namespace {

// Reads `count` bytes starting at byte `first`, lowest byte first (little-endian).
u32 read_bytes(const InstrBytes& word, u32 first, u32 count) {
    u32 value = 0;
    for (u32 i = 0; i < count; ++i) {
        const u32 byte = word[first + i];
        value = value | (byte << (8 * i));
    }
    return value;
}

// Writes `value` into `count` bytes starting at byte `first`, lowest byte first; throws if it does not fit.
void write_bytes(InstrBytes& word, u32 first, u32 count, u64 value, const char* name) {
    const u32 bits = 8 * count;
    const u64 largest = (u64{1} << bits) - 1;
    if (value > largest) {
        const std::string field = std::string(name) + "=" + std::to_string(value);
        throw std::out_of_range(field + " does not fit in " + std::to_string(bits) + " bits");
    }

    for (u32 i = 0; i < count; ++i) {
        word[first + i] = static_cast<u8>((value >> (8 * i)) & 0xFF);
    }
}

void check_flag(u32 value, const char* name) {
    if (value > 1) {
        throw std::out_of_range(std::string(name) + "=" + std::to_string(value) + " must be 0 or 1");
    }
}

std::string hex(u32 value) {
    char text[16];
    std::snprintf(text, sizeof text, "0x%X", value);
    return text;
}

}  // namespace

const char* op_name(Op op) {
    switch (op) {
        case Op::Nop:
            return "Nop";
        case Op::Halt:
            return "Halt";
        case Op::Sync:
            return "Sync";
        case Op::ReadHostMemory:
            return "Read_Host_Memory";
        case Op::WriteHostMemory:
            return "Write_Host_Memory";
        case Op::ReadWeights:
            return "Read_Weights";
        case Op::MatrixMultiply:
            return "MatrixMultiply";
    }
    return "?";
}

InstrBytes encode(const Instr& in) {
    InstrBytes word{};   // all twelve bytes start as zero
    word[0] = static_cast<u8>(in.op);

    switch (in.op) {
        case Op::Nop:
        case Op::Halt:
        case Op::Sync:
            break;

        case Op::ReadHostMemory:
        case Op::WriteHostMemory: {
            write_bytes(word, 3, 3, in.ub_row, "ub");       // bytes 3-5
            write_bytes(word, 6, 4, in.host_row, "host");   // bytes 6-9
            write_bytes(word, 10, 2, in.rows, "rows");      // bytes 10-11
            break;
        }

        case Op::ReadWeights: {
            write_bytes(word, 3, 4, in.tile, "tile");       // bytes 3-6
            break;
        }

        case Op::MatrixMultiply: {
            write_bytes(word, 3, 3, in.ub_row, "ub");       // bytes 3-5
            write_bytes(word, 6, 2, in.acc_row, "acc");     // bytes 6-7
            write_bytes(word, 8, 4, in.rows, "rows");       // bytes 8-11
            check_flag(in.accumulate, "accumulate");
            check_flag(in.new_weights, "new_weights");
            word[1] = static_cast<u8>(in.accumulate | (in.new_weights << 1));   // flag byte: bit 0, bit 1
            break;
        }
    }
    return word;
}

Instr decode(const InstrBytes& word) {
    Instr in;
    in.op = static_cast<Op>(word[0]);

    switch (in.op) {
        case Op::Nop:
        case Op::Halt:
        case Op::Sync:
            break;

        case Op::ReadHostMemory:
        case Op::WriteHostMemory: {
            in.ub_row   = read_bytes(word, 3, 3);    // bytes 3-5
            in.host_row = read_bytes(word, 6, 4);    // bytes 6-9
            in.rows     = read_bytes(word, 10, 2);   // bytes 10-11
            break;
        }

        case Op::ReadWeights: {
            in.tile = read_bytes(word, 3, 4);        // bytes 3-6
            break;
        }

        case Op::MatrixMultiply: {
            in.accumulate  = word[1] & 0x1;          // flag byte, bit 0
            in.new_weights = (word[1] >> 1) & 0x1;   // flag byte, bit 1
            in.ub_row      = read_bytes(word, 3, 3);
            in.acc_row     = read_bytes(word, 6, 2);
            in.rows        = read_bytes(word, 8, 4);
            break;
        }

        default:
            throw std::invalid_argument("unknown opcode " + hex(word[0]));
    }
    return in;
}

std::string disasm(const Instr& in) {
    std::string name = op_name(in.op);

    switch (in.op) {
        case Op::Nop:
        case Op::Halt:
        case Op::Sync:
            return name;

        case Op::ReadHostMemory: {
            return name + " host=" + hex(in.host_row) + " ub=" + hex(in.ub_row) + " rows=" + std::to_string(in.rows);
        }

        case Op::WriteHostMemory: {
            return name + " ub=" + hex(in.ub_row) + " host=" + hex(in.host_row) + " rows=" + std::to_string(in.rows);
        }

        case Op::ReadWeights: {
            return name + " tile=" + hex(in.tile);
        }

        case Op::MatrixMultiply: {
            const std::string operands = " ub=" + hex(in.ub_row) + " acc=" + hex(in.acc_row) + " rows=" + std::to_string(in.rows);
            const std::string flags    = " accumulate=" + std::to_string(in.accumulate) + " new_weights=" + std::to_string(in.new_weights);
            return name + operands + flags;
        }
    }
    return name;
}
