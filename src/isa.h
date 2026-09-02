#pragma once

#include <cassert>
#include <cstdint>
#include <string>

#include "datapath.h"

// The CISC instruction set: the opcode enum, the fixed six-word encoding and
// its field layout, the encoder, the decoder, and the disassembler.
//
// Fixed-width instructions: one opcode/flag word plus five operand words. No
// field straddles a word boundary, so an encoding is legible in a hex dump and a
// decode is a field extraction rather than bit-stitching. Instructions are whole
// tensor operations, so six words leave room to spare and nothing needs packing.
//
// word 0   flags and opcode
//   [7:0]    opcode (Op)
//   [8]      accumulate           (MATMUL)
//   [10:9]   activation function  (ACTIVATE)
//   [12:11]  pooling mode         (ACTIVATE)
//   [20:13]  requantization shift (ACTIVATE)
//   [26:21]  pooling window       (ACTIVATE)
//   [31:27]  pooling stride       (ACTIVATE)
//
// words 1..5  operands, per opcode
//   READ_HOST     host_addr, ub_addr, bytes
//   WRITE_HOST    ub_addr, host_addr, bytes
//   READ_WEIGHTS  ddr_addr, tile
//   MATMUL        ub_src, len, acc_bank
//   ACTIVATE      acc_bank, ub_dst, len, bias, multiplier
//   HALT          code
//   SYNC / NOP    none

// One opcode per whole-tensor operation; this class drives every dispatch
// decision from the sequencer onward.
enum class Op : uint8_t {
    READ_HOST,      // DMA host -> Unified Buffer
    READ_WEIGHTS,   // stage a weight tile into the weight FIFO / shadow plane
    MATMUL,         // stream activations through the array into an accumulator
    ACTIVATE,       // bias, requantize, activation function, optional pool
    WRITE_HOST,     // DMA Unified Buffer -> host
    SYNC,           // barrier: stall issue until all in-flight ops retire
    NOP,            // no operation
    HALT,           // stop the machine (also the illegal-opcode trap)
};

inline constexpr uint32_t ISA_WORDS      = 6;
inline constexpr uint32_t ISA_INST_BYTES = ISA_WORDS * 4;

struct RawInst {
    uint32_t word[ISA_WORDS] = {};
};

// Field positions and widths in word 0.
namespace isa {

inline constexpr uint32_t OPCODE_SHIFT = 0;
inline constexpr uint32_t OPCODE_MASK  = 0xFFu;
inline constexpr uint32_t ACC_SHIFT    = 8;
inline constexpr uint32_t ACTFN_SHIFT  = 9;
inline constexpr uint32_t ACTFN_MASK   = 0x3u;
inline constexpr uint32_t POOL_SHIFT   = 11;
inline constexpr uint32_t POOL_MASK    = 0x3u;
inline constexpr uint32_t SHIFT_SHIFT  = 13;
inline constexpr uint32_t SHIFT_MASK   = 0xFFu;
inline constexpr uint32_t WINDOW_SHIFT = 21;
inline constexpr uint32_t WINDOW_MASK  = 0x3Fu;
inline constexpr uint32_t STRIDE_SHIFT = 27;
inline constexpr uint32_t STRIDE_MASK  = 0x1Fu;

// The last legal opcode; anything above it is an illegal encoding.
inline constexpr uint32_t MAX_OPCODE = static_cast<uint32_t>(Op::HALT);

}  // namespace isa

// An instruction in field form. Every immediate is sign- or zero-extended once,
// here, and never re-derived at a use site.
struct Decoded {
    Op   op   = Op::NOP;
    bool trap = false;    // illegal encoding; halts the machine

    // DMA
    HostAddr host_addr = 0;
    UbAddr   ub_addr   = 0;    // ub_src for MATMUL, ub_dst for ACTIVATE
    uint32_t bytes     = 0;

    // Weights
    uint32_t ddr_addr = 0;
    TileId   tile     = INVALID_TILE;

    // Matmul
    uint32_t len        = 0;              // activation columns streamed
    BankId   acc_bank   = INVALID_BANK;
    bool     accumulate = false;          // add into the bank instead of overwriting

    // Activation pipeline
    ActFn    act         = ActFn::IDENTITY;
    Pool     pool        = Pool::NONE;
    uint32_t pool_window = 1;
    uint32_t pool_stride = 1;
    i32      bias        = 0;
    i32      multiplier  = 1;
    uint32_t shift       = 0;

    // Halt
    uint32_t code = 0;
};

// Encode a decoded instruction back to its wire form. Kept next to the field
// layout so encoder and decoder cannot drift apart, and so the test program
// builder has one way to emit an instruction.
inline RawInst encode(const Decoded& d) {
    assert(d.shift       <= isa::SHIFT_MASK);
    assert(d.pool_window <= isa::WINDOW_MASK);
    assert(d.pool_stride <= isa::STRIDE_MASK);

    RawInst r;
    r.word[0] = (static_cast<uint32_t>(d.op) & isa::OPCODE_MASK) << isa::OPCODE_SHIFT;
    if (d.accumulate) r.word[0] |= 1u << isa::ACC_SHIFT;
    r.word[0] |= (static_cast<uint32_t>(d.act)  & isa::ACTFN_MASK)  << isa::ACTFN_SHIFT;
    r.word[0] |= (static_cast<uint32_t>(d.pool) & isa::POOL_MASK)   << isa::POOL_SHIFT;
    r.word[0] |= (d.shift       & isa::SHIFT_MASK)  << isa::SHIFT_SHIFT;
    r.word[0] |= (d.pool_window & isa::WINDOW_MASK) << isa::WINDOW_SHIFT;
    r.word[0] |= (d.pool_stride & isa::STRIDE_MASK) << isa::STRIDE_SHIFT;

    switch (d.op) {
        case Op::READ_HOST:
            r.word[1] = d.host_addr;
            r.word[2] = d.ub_addr;
            r.word[3] = d.bytes;
            break;
        case Op::WRITE_HOST:
            r.word[1] = d.ub_addr;
            r.word[2] = d.host_addr;
            r.word[3] = d.bytes;
            break;
        case Op::READ_WEIGHTS:
            r.word[1] = d.ddr_addr;
            r.word[2] = d.tile;
            break;
        case Op::MATMUL:
            r.word[1] = d.ub_addr;
            r.word[2] = d.len;
            r.word[3] = d.acc_bank;
            break;
        case Op::ACTIVATE:
            r.word[1] = d.acc_bank;
            r.word[2] = d.ub_addr;
            r.word[3] = d.len;
            r.word[4] = static_cast<uint32_t>(d.bias);
            r.word[5] = static_cast<uint32_t>(d.multiplier);
            break;
        case Op::HALT:
            r.word[1] = d.code;
            break;
        case Op::SYNC:
        case Op::NOP:
            break;
    }
    return r;
}

// Wire form to field form. The only failure mode is an opcode outside the
// defined set, which decodes to a trapping HALT: a malformed program stops the
// machine rather than running on undefined state.
Decoded decode(const RawInst& raw);

// One line per instruction, operands named rather than positional so a listing
// can be read without the encoding table.
std::string disasm(const Decoded& d);

inline const char* op_name(Op op) {
    switch (op) {
        case Op::READ_HOST:    return "Read_Host_Memory";
        case Op::READ_WEIGHTS: return "Read_Weights";
        case Op::MATMUL:       return "MatMul";
        case Op::ACTIVATE:     return "Activate";
        case Op::WRITE_HOST:   return "Write_Host_Memory";
        case Op::SYNC:         return "Sync";
        case Op::NOP:          return "Nop";
        case Op::HALT:         return "Halt";
    }
    return "?";
}

inline const char* actfn_name(ActFn f) {
    switch (f) {
        case ActFn::IDENTITY: return "identity";
        case ActFn::RELU:     return "relu";
        case ActFn::RELU6:    return "relu6";
    }
    return "?";
}

inline const char* pool_name(Pool p) {
    switch (p) {
        case Pool::NONE: return "none";
        case Pool::MAX:  return "max";
        case Pool::AVG:  return "avg";
    }
    return "?";
}
