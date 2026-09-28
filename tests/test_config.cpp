/// TPUv1's parameters reproduce the paper's headline figures.

#include "common/config.h"
#include "test_framework.h"

TEST(config_memory_sizes_match_the_paper) {
    CHECK_EQ(v1::kTileBytes, u64{64} << 10);                          // 64 KiB weight tile
    CHECK_EQ(u64{v1::kUbRows} * v1::kMxuDim, u64{24} << 20);          // 24 MiB Unified Buffer
    CHECK_EQ(u64{v1::kAccRows} * v1::kMxuDim * 4, u64{4} << 20);      // 4 MiB of int32 accumulators
    CHECK_EQ(v1::kWeightFifoTiles * v1::kTileBytes, u64{256} << 10);  // 256 KiB Weight FIFO
}

TEST(config_peak_is_92_tops) {
    const u64 ops_per_cycle = u64{2} * v1::kMxuDim * v1::kMxuDim;     // a MAC is two ops
    const u64 gops = ops_per_cycle * v1::kClockMhz / 1000;
    CHECK_EQ(gops, u64{91750});                                        // the paper rounds to 92 TOPS
}

TEST(config_bandwidths_match_the_paper) {
    CHECK_EQ(u64{v1::kWeightBytesPerCycle} * v1::kClockMhz / 1000, u64{33});   // ~34 GB/s DDR3
    CHECK_EQ(u64{v1::kHostBytesPerCycle} * v1::kClockMhz / 1000, u64{15});     // ~15.75 GB/s PCIe
    CHECK_EQ(v1::kTileBytes / v1::kWeightBytesPerCycle, u64{1365});            // cycles to fetch one tile
}
