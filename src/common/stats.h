/// Why a cycle issued no instruction, and the counters a run keeps.

#pragma once

#include <array>
#include <cstddef>

#include "common/types.h"

enum class Stall : u8 {
    None,                // an instruction issued this cycle
    HostInterfaceBusy,   // a host transfer waits for the one in progress
    WaitForIdle,         // Sync or Halt waits for every unit to finish
    WeightFifoFull,      // Read_Weights waits for a free FIFO slot
    MxuBusy,             // MatrixMultiply waits for the one in the array
    WeightsNotReady,     // MatrixMultiply waits for a whole tile in the shadow plane
    UbNotReady,          // an instruction waits for UB rows another unit is still reading or writing
    ActivationBusy,      // Activate waits for the one in the activation unit
    AccNotReady,         // an instruction waits for accumulator rows another unit is still reading or writing
};

inline constexpr std::size_t kStallKinds = 9;

inline const char* stall_name(Stall stall) {
    switch (stall) {
        case Stall::None:
            return "none";
        case Stall::HostInterfaceBusy:
            return "host interface busy";
        case Stall::WaitForIdle:
            return "waiting for units to finish";
        case Stall::WeightFifoFull:
            return "weight FIFO full";
        case Stall::MxuBusy:
            return "MXU busy";
        case Stall::WeightsNotReady:
            return "weights not ready";
        case Stall::UbNotReady:
            return "Unified Buffer rows not ready";
        case Stall::ActivationBusy:
            return "activation unit busy";
        case Stall::AccNotReady:
            return "accumulator rows not ready";
    }
    return "?";
}

struct Stats {
    Cycle cycles = 0;
    u64   issued = 0;   // instructions issued so far
    std::array<u64, kStallKinds> stall_cycles{};   // cycles charged to each Stall, indexed by its value
    u64   mxu_busy_cycles = 0;   // cycles the systolic array took a step

    u64 stalled(Stall stall) const {
        const std::size_t index = static_cast<std::size_t>(stall);
        return stall_cycles[index];
    }
};
