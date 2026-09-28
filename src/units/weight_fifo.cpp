/// The Weight FIFO's slots and its 48-bytes-per-cycle fetch from Weight Memory.

#include "units/weight_fifo.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

bool FifoTile::ready() const {
    return bytes_arrived == v1::kTileBytes;
}

WeightFifo::WeightFifo(const Dram& wmem) : wmem_(wmem) {}

bool WeightFifo::full() const {
    return tiles_.size() >= kSlots;
}

void WeightFifo::push(u32 tile) {
    if (full()) {
        throw std::logic_error("weight FIFO: all " + std::to_string(kSlots) + " slots are taken");
    }

    const u64 tile_count = wmem_.size() / v1::kTileBytes;
    if (tile >= tile_count) {
        throw std::out_of_range("weight tile " + std::to_string(tile) + " is past the last tile " +
                                std::to_string(tile_count - 1));
    }

    FifoTile entry;
    entry.tile = tile;
    entry.bytes.resize(v1::kTileBytes, 0);
    tiles_.push_back(entry);
}

void WeightFifo::tick() {
    // Tiles arrive in order, so the one to work on is the first that is not ready yet.
    for (FifoTile& entry : tiles_) {
        if (entry.ready()) {
            continue;
        }

        const u64 bytes_left = v1::kTileBytes - entry.bytes_arrived;
        const u64 piece      = std::min<u64>(bytes_left, v1::kWeightBytesPerCycle);
        const u64 source     = static_cast<u64>(entry.tile) * v1::kTileBytes + entry.bytes_arrived;

        wmem_.read(source, entry.bytes.data() + entry.bytes_arrived, piece);
        entry.bytes_arrived = entry.bytes_arrived + piece;
        return;   // one DDR3 channel: one tile moves per cycle
    }
}

bool WeightFifo::fetching() const {
    for (const FifoTile& entry : tiles_) {
        if (!entry.ready()) {
            return true;
        }
    }
    return false;
}

bool WeightFifo::front_ready() const {
    if (tiles_.empty()) {
        return false;
    }
    return tiles_.front().ready();
}

std::vector<i8> WeightFifo::pop() {
    if (!front_ready()) {
        throw std::logic_error("weight FIFO: the oldest tile has not fully arrived");
    }
    std::vector<i8> bytes = std::move(tiles_.front().bytes);
    tiles_.pop_front();
    return bytes;
}

void WeightFifo::reset() {
    tiles_.clear();
}

const std::deque<FifoTile>& WeightFifo::tiles() const {
    return tiles_;
}
