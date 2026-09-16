#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

#include "config.h"
#include "datapath.h"
#include "storage.h"

// A staged weight tile on its way from weight memory (DDR) to the array.
struct WeightTile {
    uint32_t        ddr_addr = 0;
    TileId          tile     = 0;
    uint32_t        rows     = 0;   // the real shape; the array zero-pads the rest
    uint32_t        cols     = 0;
    std::vector<i8> data;           // rows x cols, row-major

    ConstI8View view() const { return ConstI8View(data.data(), rows, cols); }
};

// A bounded FIFO of weight tiles, each arriving ddr_tile_latency cycles late.
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

    // Slots held, whether the tile has arrived or is still in flight.
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

    // Pop the oldest ready tile; false means the caller must stall.
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

    // When the oldest in-flight tile arrives, so a caller can size its wait.
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

enum class DmaDir : uint8_t {
    HOST_TO_UB,   // Read_Host_Memory
    UB_TO_HOST,   // Write_Host_Memory
};

struct DmaRequest {
    DmaDir   dir   = DmaDir::HOST_TO_UB;
    HostAddr host  = 0;
    UbAddr   ub    = 0;
    uint32_t bytes = 0;
};

// Moves byte ranges between host memory and the Unified Buffer.
class Dma {
public:
    struct Stats {
        uint64_t transfers   = 0;
        uint64_t bytes_moved = 0;
        uint64_t busy_cycles = 0;
        uint64_t rejects     = 0;   // started while busy, or out of range
    };

    explicit Dma(const Config& cfg) : bw_(cfg.dma_bytes_per_cycle) {
        assert(cfg.dma_bytes_per_cycle > 0);
    }

    uint32_t bytes_per_cycle() const { return bw_; }
    const Stats& stats() const { return stats_; }

    // Cycles a transfer occupies the engine; a zero-byte transfer is free.
    uint64_t transfer_cycles(uint32_t bytes) const {
        if (bytes == 0) return 0;
        return (static_cast<uint64_t>(bytes) + bw_ - 1) / bw_;
    }

    bool busy() const { return active_ && now_ < finish_; }
    uint64_t finish_cycle() const { return finish_; }

    // The UB address the transfer is touching, for the sequencer's port budget.
    UbAddr ub_endpoint() const { return ub_; }

    // Reserve and time the transfer without moving bytes; the sequencer does that.
    bool begin(const DmaRequest& r, uint64_t now, std::size_t host_bytes,
               const UnifiedBuffer& ub) {
        now_ = now;
        if (busy()) {
            ++stats_.rejects;
            return false;
        }
        const bool host_ok = r.host <= host_bytes && r.bytes <= host_bytes - r.host;
        if (!host_ok || !ub.in_range(r.ub, r.bytes)) {
            ++stats_.rejects;
            return false;
        }

        const uint64_t cycles = transfer_cycles(r.bytes);
        active_ = cycles != 0;
        ub_     = r.ub;
        finish_ = now + cycles;

        ++stats_.transfers;
        stats_.bytes_moved += r.bytes;
        stats_.busy_cycles += cycles;
        return true;
    }

    // Reserve the engine and move the bytes immediately, for a hand-driven caller.
    bool start(const DmaRequest& r, uint64_t now, std::vector<uint8_t>& host,
               UnifiedBuffer& ub) {
        if (!begin(r, now, host.size(), ub)) return false;

        for (uint32_t i = 0; i < r.bytes; ++i) {
            if (r.dir == DmaDir::HOST_TO_UB) {
                ub.at(r.ub + i) = static_cast<i8>(host[r.host + i]);
            } else {
                host[r.host + i] = static_cast<uint8_t>(ub.at(r.ub + i));
            }
        }
        return true;
    }

    void tick(uint64_t now) {
        now_ = now;
        if (active_ && now_ >= finish_) active_ = false;
    }

private:
    uint32_t bw_;
    uint64_t now_    = 0;
    uint64_t finish_ = 0;
    bool     active_ = false;
    UbAddr   ub_     = 0;
    Stats    stats_;
};
