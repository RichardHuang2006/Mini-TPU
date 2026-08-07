#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "accumulators.h"
#include "config.h"
#include "dma.h"
#include "mxu.h"
#include "types.h"
#include "unified_buffer.h"
#include "weight_fifo.h"

// The machine: the MXU, the Unified Buffer, the accumulator banks, the weight
// FIFO, the DMA engine, host and weight memory, and a program counter.
//
// In this phase the units are driven by hand, one call per operation, and tick()
// only advances time. The sequencer that turns a decoded instruction stream into
// these same calls arrives in Phase 5; keeping the units independently drivable
// is what lets that be an addition rather than a rewrite.
class Tpu {
public:
    explicit Tpu(const Config& cfg, std::size_t host_bytes = 1u << 16,
                 std::size_t weight_bytes = 1u << 16);

    const Config& config() const { return cfg_; }
    uint64_t cycle() const { return cycle_; }
    std::size_t pc() const { return pc_; }

    UnifiedBuffer& ub() { return ub_; }
    Accumulators&  acc() { return acc_; }
    WeightFifo&    weight_fifo() { return fifo_; }
    Dma&           dma() { return dma_; }
    Mxu&           mxu() { return mxu_; }

    const UnifiedBuffer& ub() const { return ub_; }
    const Accumulators&  acc() const { return acc_; }

    std::vector<uint8_t>& host() { return host_; }
    std::vector<i8>&      weight_mem() { return weight_mem_; }

    // One cycle. Stages are evaluated in reverse pipeline order so each observes
    // the previous cycle's output of its producer, making the inter-stage
    // registers behave as latches.
    void tick();

    // Run tick() until `pred` holds or `limit` cycles pass. Returns false if the
    // limit ran out, so a test cannot hang on a unit that never completes.
    template <typename Pred>
    bool run_until(Pred pred, uint64_t limit = 1u << 20) {
        for (uint64_t i = 0; i < limit; ++i) {
            if (pred()) return true;
            tick();
        }
        return pred();
    }

    // ---- hand-driven operations ------------------------------------------
    // Each is one decoded instruction's worth of work. Phase 5 calls these from
    // the sequencer instead of from a test.

    bool dma_to_ub(HostAddr host_addr, UbAddr ub_addr, uint32_t bytes);
    bool dma_to_host(UbAddr ub_addr, HostAddr host_addr, uint32_t bytes);

    // Request a rows x cols weight tile from weight memory into the FIFO. The
    // tile lands ddr_tile_latency cycles later.
    bool stage_weights(uint32_t ddr_addr, uint32_t rows, uint32_t cols, TileId tile = 0);

    // Pop a staged tile into the array's load plane. False means the FIFO had
    // nothing ready, which is the weight_fifo_empty stall.
    bool load_weights_from_fifo();

    // Stream `len` activation rows at `ub_addr` through the array into a bank.
    bool matmul(UbAddr ub_addr, uint32_t len, BankId bank, bool accumulate);

private:
    Config cfg_;

    UnifiedBuffer ub_;
    Accumulators  acc_;
    WeightFifo    fifo_;
    Dma           dma_;
    Mxu           mxu_;

    std::vector<uint8_t> host_;
    std::vector<i8>      weight_mem_;

    uint64_t    cycle_ = 0;
    std::size_t pc_    = 0;
};
