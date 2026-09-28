/// Encoding, decoding and disassembly of the TPUv1 instructions.

#include "isa/isa.h"

#include <cstdio>
#include <stdexcept>

namespace {

// Where each host-transfer field sits in the 12 bytes, and how many bytes it takes.
constexpr u32 kUbRowAt      = 3;
constexpr u32 kUbRowWidth   = 3;
constexpr u32 kHostRowAt    = 6;
constexpr u32 kHostRowWidth = 4;
constexpr u32 kRowsAt       = 10;
constexpr u32 kRowsWidth    = 2;

// Read_Weights: the tile number in bytes 3-6.
constexpr u32 kTileAt       = 3;
constexpr u32 kTileWidth    = 4;

// Writes value into `width` bytes starting at `at`, lowest byte first (little-endian).
void put(InstrBytes& bytes, u32 at, u32 width, u64 value, const char* field) {
    const u32 bits = 8 * width;
    const u64 largest = (u64{1} << bits) - 1;
    if (value > largest) {
        const std::string what = std::string(field) + "=" + std::to_string(value);
        throw std::out_of_range(what + " does not fit in " + std::to_string(bits) + " bits");
    }

    for (u32 i = 0; i < width; ++i) {
        const u64 shifted = value >> (8 * i);
        const u8 low_byte = static_cast<u8>(shifted & 0xFF);
        bytes[at + i] = low_byte;
    }
}

// Reads `width` bytes starting at `at`, lowest byte first, back into a number.
u32 get(const InstrBytes& bytes, u32 at, u32 width) {
    u32 value = 0;
    for (u32 i = 0; i < width; ++i) {
        const u32 byte = bytes[at + i];
        value = value | (byte << (8 * i));
    }
    return value;
}

std::string hex(u32 value) {
    char text[16];
    std::snprintf(text, sizeof text, "0x%X", value);
    return text;
}

bool is_host_transfer(Op op) {
    return op == Op::ReadHostMemory || op == Op::WriteHostMemory;
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
    }
    return "?";
}

InstrBytes encode(const Instr& in) {
    InstrBytes bytes{};   // all twelve bytes start as zero
    bytes[0] = static_cast<u8>(in.op);

    if (is_host_transfer(in.op)) {
        put(bytes, kUbRowAt, kUbRowWidth, in.ub_row, "ub");
        put(bytes, kHostRowAt, kHostRowWidth, in.host_row, "host");
        put(bytes, kRowsAt, kRowsWidth, in.rows, "rows");
    }
    if (in.op == Op::ReadWeights) {
        put(bytes, kTileAt, kTileWidth, in.tile, "tile");
    }
    return bytes;
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
            in.ub_row   = get(bytes, kUbRowAt, kUbRowWidth);
            in.host_row = get(bytes, kHostRowAt, kHostRowWidth);
            in.rows     = get(bytes, kRowsAt, kRowsWidth);
            break;
        case Op::ReadWeights:
            in.tile = get(bytes, kTileAt, kTileWidth);
            break;
        default:
            throw std::invalid_argument("unknown opcode " + hex(bytes[0]));
    }
    return in;
}

std::string disasm(const Instr& in) {
    std::string       name = op_name(in.op);
    const std::string ub   = "ub=" + hex(in.ub_row);
    const std::string host = "host=" + hex(in.host_row);
    const std::string rows = "rows=" + std::to_string(in.rows);

    // Operands are listed source first, then destination.
    if (in.op == Op::ReadHostMemory) {
        return name + " " + host + " " + ub + " " + rows;
    }
    if (in.op == Op::WriteHostMemory) {
        return name + " " + ub + " " + host + " " + rows;
    }
    if (in.op == Op::ReadWeights) {
        return name + " tile=" + hex(in.tile);
    }
    return name;
}
