/// The per-cycle activity record the visualizer draws its timeline and memory glows from.

#pragma once

#include <cstddef>

#include "common/stats.h"

// What one cycle did: whether an instruction issued (or why not), and which units worked.
struct CycleRecord {
    Cycle cycle      = 0;
    Stall stall      = Stall::None;
    int   issued_pc  = -1;      // the PC that issued this cycle, or -1
    bool  host       = false;   // the host interface moved bytes
    bool  fetching   = false;   // DDR3 filled a Weight FIFO slot
    bool  shifting   = false;   // the weight shifter moved a tile row into the shadow plane
    bool  mxu        = false;   // the systolic array took a step
    bool  activation = false;   // the activation unit took a row
};

inline constexpr std::size_t kActivityCycles = 512;   // how many recent cycles the Tpu keeps
inline constexpr u32 kMapRows = 256;                  // one visualizer map cell covers 256 rows
inline constexpr Cycle kNever = ~Cycle{0};            // a map cell nothing has written yet
