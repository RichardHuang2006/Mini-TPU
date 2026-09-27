/// Every structural knob of the TensorCore, with the tiny() and v4() presets.

#pragma once

#include "common/types.h"

struct CoreConfig {
    const char* name = "tiny";

    // Matrix units: each is a mxu_dim x mxu_dim weight-stationary systolic array.
    u32 mxu_dim  = 8;
    u32 num_mxus = 1;

    // A vreg is sublanes x lanes fp32 words; a row of lanes feeds one MXU input row.
    u32 sublanes = 8;
    u32 lanes    = 8;

    u32 num_vregs = 32;
    u32 num_sregs = 32;
    u32 num_sems  = 32;

    // Memory sizes in 32-bit words.
    u32 main_words = 1u << 18;
    u32 vmem_words = 1u << 16;
    u32 smem_words = 1u << 12;

    // DMA: cycles before the first word moves, then words moved per cycle.
    u32 dma_latency         = 16;
    u32 dma_words_per_cycle = 8;

    u32 vreg_words() const { return sublanes * lanes; }

    // Small enough to draw every PE and vreg element in the terminal.
    static CoreConfig tiny() { return {}; }

    // TPUv4-like TensorCore: four 128x128 MXUs and 8x128 vregs.
    static CoreConfig v4() {
        CoreConfig c;
        c.name                = "v4";
        c.mxu_dim             = 128;
        c.num_mxus            = 4;
        c.lanes               = 128;
        c.main_words          = 1u << 24;
        c.vmem_words          = 1u << 22;
        c.dma_latency         = 64;
        c.dma_words_per_cycle = 128;
        return c;
    }
};
