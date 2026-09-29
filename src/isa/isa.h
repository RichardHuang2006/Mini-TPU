/// TPUv1's CISC instructions: opcodes, the decoded form, and 12-byte encode, decode and disassembly.

#pragma once

#include <array>
#include <string>
#include <vector>

#include "common/types.h"

// Every instruction is 12 bytes: byte 0 opcode, bytes 1-2 flags, bytes 3-11 operands.
inline constexpr u32 kInstrBytes = 12;
using InstrBytes = std::array<u8, kInstrBytes>;

enum class Op : u8 {
    Nop             = 0x00,
    Halt            = 0x01,
    Sync            = 0x02,   // wait until every unit is idle
    ReadHostMemory  = 0x10,   // host memory -> Unified Buffer
    WriteHostMemory = 0x11,   // Unified Buffer -> host memory
    ReadWeights     = 0x20,   // Weight Memory tile -> Weight FIFO
    MatrixMultiply  = 0x30,   // UB rows x weights -> accumulator rows
};

// Every opcode, so the assembler can look names up instead of keeping its own list.
inline constexpr Op kAllOps[] = {
    Op::Nop,
    Op::Halt,
    Op::Sync,
    Op::ReadHostMemory,
    Op::WriteHostMemory,
    Op::ReadWeights,
    Op::MatrixMultiply,
};

// One instruction with its fields unpacked into plain numbers.
struct Instr {
    Op  op          = Op::Nop;
    u32 ub_row      = 0;   // 24 bits: Unified Buffer row
    u32 host_row    = 0;   // 32 bits: host memory is addressed in 256-byte rows
    u32 rows        = 0;   // rows moved (16 bits) or multiplied (32 bits)
    u32 tile        = 0;   // 32 bits: Weight Memory tile (64 KiB each)
    u32 acc_row     = 0;   // 16 bits: accumulator row
    u32 accumulate  = 0;   // flag, 0 or 1: add into the accumulators instead of overwriting
    u32 new_weights = 0;   // flag, 0 or 1: switch to the shadow weight plane first
};

// Bytes to place in a DRAM before the program runs.
struct DataBlock {
    u64             addr = 0;   // byte address
    std::vector<i8> bytes;
};

// A loadable program: what the host sends to the TPU, plus the data it expects in host memory and Weight Memory.
struct Program {
    std::vector<InstrBytes> code;
    std::vector<DataBlock>  host;
    std::vector<DataBlock>  weights;
};

// The instruction's name as the assembler and disassembler spell it, e.g. "Read_Host_Memory".
const char* op_name(Op op);

// Instr to 12 bytes; throws std::out_of_range when a field does not fit its width.
InstrBytes encode(const Instr& in);

// 12 bytes to Instr; throws std::invalid_argument on an unknown opcode.
Instr decode(const InstrBytes& bytes);

// Instr to text, e.g. "Read_Host_Memory host=0x200 ub=0x10 rows=4".
std::string disasm(const Instr& in);
