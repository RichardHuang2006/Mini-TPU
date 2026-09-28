/// TPUv1's fixed parameters

#pragma once

#include "common/types.h"

namespace v1 {

inline constexpr u32 kClockMhz = 700;

inline constexpr u32 kMxuDim    = 256;                  // Matrix Multiply Unit: 256 x 256 8-bit MACs
inline constexpr u64 kTileBytes = kMxuDim * kMxuDim;    // one weight tile: 64 KiB

inline constexpr u32 kUbRows  = 96 * 1024;              // Unified Buffer: 24 MiB of 256-byte rows
inline constexpr u32 kAccRows = 4096;                   // Accumulators: 4 MiB of 256 x int32 rows

inline constexpr u32 kWeightFifoTiles = 4;              // Weight FIFO depth
inline constexpr u64 kWeightMemBytes  = 8 * kGiB;       // off-chip DDR3 Weight Memory

inline constexpr u32 kWeightBytesPerCycle = 48;         // DDR3 at 34 GB/s over 700 MHz
inline constexpr u32 kHostBytesPerCycle   = 22;         // PCIe Gen3 x16 at 15.75 GB/s over 700 MHz

}  // namespace v1
