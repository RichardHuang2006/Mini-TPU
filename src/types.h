#pragma once

#include <cstdint>
#include <limits>

// Fixed-width datapath types. Activations and weights are int8, accumulators
// int32; the aliases below name the roles so signatures stay readable.

using i8  = int8_t;    // activation / weight element
using i32 = int32_t;   // accumulator element

// Identifier aliases. All are uint32_t; the names only make signatures
// readable and self-documenting at a call site.

using UbAddr   = uint32_t;   // byte offset into the Unified Buffer
using HostAddr = uint32_t;   // byte offset into host memory
using BankId   = uint32_t;   // accumulator (or UB) bank index
using TileId   = uint32_t;   // weight tile staged through the weight FIFO

// Out-of-band sentinels, disjoint from every legal index.
inline constexpr UbAddr   INVALID_UBADDR   = std::numeric_limits<UbAddr>::max();
inline constexpr HostAddr INVALID_HOSTADDR = std::numeric_limits<HostAddr>::max();
inline constexpr BankId   INVALID_BANK     = std::numeric_limits<BankId>::max();
inline constexpr TileId   INVALID_TILE     = std::numeric_limits<TileId>::max();

// The CISC instruction set. One opcode per whole-tensor operation; this class
// drives every dispatch decision from the sequencer onward.
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

// Activation function applied after requantization in the activation pipeline.
enum class ActFn : uint8_t {
    IDENTITY,
    RELU,
    RELU6,
};

// Optional pooling stage on the requantized int8 stream.
enum class Pool : uint8_t {
    NONE,
    MAX,
    AVG,
};
