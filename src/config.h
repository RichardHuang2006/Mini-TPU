#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

struct Config {
    // Systolic array edge length: dim*dim int8 MACs per cycle at steady state.
    uint32_t dim = 32;

    // Unified Buffer size and bank count.
    uint32_t ub_bytes = 256 * 1024;
    uint32_t ub_banks = 8;

    // int32 accumulator banks, each dim x dim.
    uint32_t acc_banks = 4;

    // Weight FIFO depth in tiles, and whether the array keeps a shadow plane.
    uint32_t weight_fifo_depth = 4;
    bool     double_buffer     = true;

    uint32_t dma_bytes_per_cycle = 16;

    // Per-tile weight refill latency from weight memory (DDR), in cycles.
    uint32_t ddr_tile_latency = 8;

    // Fill depth of the bias -> requantize -> act -> pool pipeline.
    uint32_t act_pipeline_depth = 4;

    uint64_t peak_macs_per_cycle() const {
        return static_cast<uint64_t>(dim) * dim;
    }

    // Below the roofline ridge point, cross-multiplied to stay in integers.
    bool dma_bound(std::size_t macs, std::size_t bytes) const {
        const uint64_t lhs = static_cast<uint64_t>(macs) * dma_bytes_per_cycle;
        const uint64_t rhs = static_cast<uint64_t>(bytes) * peak_macs_per_cycle();
        return lhs < rhs;
    }
};

static_assert(std::is_trivially_copyable_v<Config>);
