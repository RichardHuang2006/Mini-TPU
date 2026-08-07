#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

// One field per structural knob. Downstream code reads sizes off a
// `const Config&` so a configuration sweep needs no recompilation.

struct Config {
    // Systolic array edge length: a dim x dim grid of MAC cells doing dim*dim
    // int8 MACs per cycle at steady state.
    uint32_t dim = 32;

    // Unified Buffer: total bytes and bank count (one read + one write port per
    // bank per cycle; a same-bank read pair is a conflict).
    uint32_t ub_bytes = 256 * 1024;
    uint32_t ub_banks = 8;

    // int32 accumulator banks, each dim x dim.
    uint32_t acc_banks = 4;

    // Weight staging: FIFO depth (in tiles) and whether the array keeps a
    // shadow weight plane so a tile load hides under the previous matmul.
    uint32_t weight_fifo_depth = 4;
    bool     double_buffer     = true;

    // Host DMA bandwidth (bytes moved per cycle).
    uint32_t dma_bytes_per_cycle = 16;

    // Per-tile weight refill latency from weight memory (DDR), in cycles.
    uint32_t ddr_tile_latency = 8;

    // Fill depth of the activation pipeline (bias -> requantize -> act -> pool);
    // an Activate on M elements costs M + act_pipeline_depth cycles.
    uint32_t act_pipeline_depth = 4;

    // Peak useful MACs per cycle: the whole array at steady state.
    uint64_t peak_macs_per_cycle() const {
        return static_cast<uint64_t>(dim) * dim;
    }

    // Roofline ridge point: a workload is memory-bound when its arithmetic
    // intensity (macs / bytes) sits below peak_macs_per_cycle / dma bandwidth.
    // Rearranged to an all-integer comparison to avoid rounding:
    //     macs / bytes  <  dim*dim / dma_bytes_per_cycle
    //     macs * dma_bytes_per_cycle  <  bytes * dim*dim
    bool dma_bound(std::size_t macs, std::size_t bytes) const {
        const uint64_t lhs = static_cast<uint64_t>(macs) * dma_bytes_per_cycle;
        const uint64_t rhs = static_cast<uint64_t>(bytes) * peak_macs_per_cycle();
        return lhs < rhs;
    }
};

// Configs get copied around wholesale; keep them memcpy-able.
static_assert(std::is_trivially_copyable_v<Config>);
