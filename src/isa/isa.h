/// TPUv1's CISC instructions: opcodes, the decoded form, and 12-byte encode, decode and disassembly.

#pragma once

#include <array>
#include <string>

#include "common/types.h"

inline constexpr u32 kInstrBytes = 12;   // byte 0 opcode, bytes 1-2 flags, bytes 3-11 operands
using InstrBytes = std::array<u8, kInstrBytes>;

enum class Op : u8 {
    Nop             = 0x00,
    Halt            = 0x01,
    Sync            = 0x02,   // wait until every unit is idle
    ReadHostMemory  = 0x10,   // host memory -> Unified Buffer
    WriteHostMemory = 0x11,   // Unified Buffer -> host memory
};

struct Instr {
    Op  op       = Op::Nop;
    u32 ub_row   = 0;   // 24 bits: Unified Buffer row
    u32 host_row = 0;   // 32 bits: host memory is addressed in 256-byte rows
    u32 rows     = 0;   // 16 bits: rows moved
};

const char* op_name(Op op);

InstrBytes  encode(const Instr& in);         // throws std::out_of_range when a field does not fit
Instr       decode(const InstrBytes& bytes);  // throws std::invalid_argument on an unknown opcode
std::string disasm(const Instr& in);
