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
};

inline constexpr std::size_t kStallKinds = 4;

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
    }
    return "?";
}

struct Stats {
    Cycle cycles = 0;
    u64   issued = 0;   // instructions issued so far
    std::array<u64, kStallKinds> stall_cycles{};   // cycles charged to each Stall, indexed by its value

    u64 stalled(Stall stall) const {
        const std::size_t index = static_cast<std::size_t>(stall);
        return stall_cycles[index];
    }
};
