/// Weight FIFO: four 64 KiB tile slots, filled in order from DDR3 Weight Memory at 48 bytes per cycle.

#pragma once

#include <deque>
#include <vector>

#include "mem/dram.h"

// One tile in the FIFO: which Weight Memory tile it is, and how much of it has arrived.
struct FifoTile {
    u32             tile          = 0;
    u64             bytes_arrived = 0;
    std::vector<i8> bytes;   // 64 KiB, filled front to back as it arrives

    bool ready() const;
};

class WeightFifo {
public:
    static constexpr u32 kSlots = v1::kWeightFifoTiles;

    explicit WeightFifo(const Dram& wmem);

    bool full() const;

    // Reserves a slot and starts fetching the tile; throws if the FIFO is full or the tile is past Weight Memory.
    void push(u32 tile);

    // One cycle: moves the next 48 bytes of the oldest tile that has not fully arrived.
    void tick();

    // True while any tile is still arriving.
    bool fetching() const;

    // True when the oldest tile has fully arrived and can be taken.
    bool front_ready() const;

    // Takes the oldest tile out of the FIFO; throws unless front_ready().
    std::vector<i8> pop();

    // Empties the FIFO, as after a machine reset.
    void reset();

    const std::deque<FifoTile>& tiles() const;

private:
    const Dram&          wmem_;
    std::deque<FifoTile> tiles_;   // oldest first
};
