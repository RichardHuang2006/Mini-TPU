#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

#include "config.h"
#include "tensor.h"
#include "types.h"

// A staged weight tile on its way from weight memory (DDR) to the array.
struct WeightTile {
    uint32_t        ddr_addr = 0;
    TileId          tile     = 0;
    uint32_t        rows     = 0;   // the real shape; the array zero-pads the rest
    uint32_t        cols     = 0;
    std::vector<i8> data;           // rows x cols, row-major

    ConstI8View view() const { return ConstI8View(data.data(), rows, cols); }
};

// A bounded FIFO of weight tiles with a background refill.
//
// The refill latency is what makes the FIFO worth having: a tile takes
// ddr_tile_latency cycles to arrive from DDR, so a program that stages tiles
// ahead of the matmuls that need them hides that latency, and one that asks for a
// tile at the moment it needs it eats the whole thing. Popping an empty FIFO is
// the weight_fifo_empty stall.
class WeightFifo {
public:
    struct Stats {
        uint64_t pushes       = 0;
        uint64_t pops         = 0;
        uint64_t empty_stalls = 0;   // a pop that found nothing ready
        uint64_t full_rejects = 0;   // a refill with no room
    };

    explicit WeightFifo(const Config& cfg)
        : depth_(cfg.weight_fifo_depth), latency_(cfg.ddr_tile_latency) {
        assert(cfg.weight_fifo_depth > 0);
    }

    uint32_t depth() const { return depth_; }
    uint32_t latency() const { return latency_; }
    const Stats& stats() const { return stats_; }

    // Slots held, whether the tile has arrived or is still in flight. A refill
    // occupies its slot from the moment it is requested, which is what makes a
    // 1-deep FIFO serialize back-to-back loads.
    std::size_t occupancy() const { return q_.size(); }
    bool full() const { return q_.size() >= depth_; }

    // Ready tiles only: `empty()` is the question a pop asks.
    std::size_t ready() const {
        std::size_t n = 0;
        for (const Entry& e : q_) {
            if (e.ready_cycle <= now_) ++n;
            else break;               // fixed latency, so arrivals stay in order
        }
        return n;
    }
    bool empty() const { return ready() == 0; }

    // Request a refill. The tile becomes poppable ddr_tile_latency cycles later.
    bool push_refill(const WeightTile& t, uint64_t now) {
        now_ = now;
        if (full()) {
            ++stats_.full_rejects;
            return false;
        }
        q_.push_back(Entry{t, now + latency_});
        ++stats_.pushes;
        return true;
    }

    // Advance the clock, letting in-flight refills land.
    void tick(uint64_t now) { now_ = now; }

    // Pop the oldest ready tile. False means the FIFO had nothing ready and the
    // instruction must stall.
    bool pop(WeightTile& out) {
        if (empty()) {
            ++stats_.empty_stalls;
            return false;
        }
        out = q_.front().tile;
        q_.pop_front();
        ++stats_.pops;
        return true;
    }

    // When the oldest in-flight tile arrives, so a caller can decide how long to
    // wait rather than polling blindly.
    uint64_t next_ready_cycle() const {
        return q_.empty() ? now_ : q_.front().ready_cycle;
    }

private:
    struct Entry {
        WeightTile tile;
        uint64_t   ready_cycle = 0;
    };

    uint32_t          depth_;
    uint32_t          latency_;
    uint64_t          now_ = 0;
    std::deque<Entry> q_;
    Stats             stats_;
};
