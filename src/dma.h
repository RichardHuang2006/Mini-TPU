#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "config.h"
#include "types.h"
#include "unified_buffer.h"

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

// The host DMA engine: moves byte ranges between host memory and the Unified
// Buffer at a configurable bandwidth.
//
// The bytes are copied at start() while the *timing* is accounted separately. A
// reader that peeked at the buffer mid-transfer would therefore see finished
// data, which is only sound because the interlock added in Phase 5 stalls any
// consumer until the DMA retires. Splitting the two this way keeps the data path
// obviously correct and leaves the schedule as the only thing the timing model
// has to get right.
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

    // Cycles a transfer occupies the engine. A zero-byte transfer is free, and
    // any nonzero one costs at least a cycle however wide the bus is.
    uint64_t transfer_cycles(uint32_t bytes) const {
        if (bytes == 0) return 0;
        return (static_cast<uint64_t>(bytes) + bw_ - 1) / bw_;
    }

    bool busy() const { return active_ && now_ < finish_; }
    uint64_t finish_cycle() const { return finish_; }

    // The UB address the transfer is touching, so the sequencer can charge it a
    // bank port for the duration.
    UbAddr ub_endpoint() const { return ub_; }

    // Reserve the engine and account for the transfer without moving any bytes.
    // The sequencer uses this one: it reads the source at issue and commits the
    // destination when the instruction retires, so that a consumer which issued
    // too early sees the old bytes rather than the new ones. That is what makes a
    // missing interlock show up as a wrong answer instead of only a wrong
    // schedule.
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

    // Reserve the engine and move the bytes immediately, for a caller driving the
    // machine by hand with nothing else in flight to observe the difference.
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
