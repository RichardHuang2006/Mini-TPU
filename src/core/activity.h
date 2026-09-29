/// The per-cycle activity record the visualizer's timeline is drawn from.

#pragma once

#include <cstddef>

#include "common/stats.h"

// What one cycle did: why nothing issued (Stall::None if something did), and what each unit worked on; -1 means idle.
struct CycleRecord {
    Stall stall      = Stall::None;
    i32   issued     = -1;   // the pc that issued
    i32   host       = -1;   // the pc of the Read_Host_Memory or Write_Host_Memory the host interface is moving
    i32   fetching   = -1;   // the tile DDR3 is filling into the Weight FIFO
    i32   shifting   = -1;   // the tile the weight shifter is moving into the shadow plane
    i32   mxu        = -1;   // the pc of the MatrixMultiply the array is stepping
    i32   activation = -1;   // the pc of the Activate the activation unit is working through
};

inline constexpr std::size_t kTimelineCycles = 64;   // the visualizer's timeline shows the last 64 cycles
