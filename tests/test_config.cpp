/// TPUv1's parameters reproduce the paper's headline figures.

#include "common/config.h"
#include "test_framework.h"

TEST(config_memory_sizes_match_the_paper) {
    const u64 ub_bytes   = static_cast<u64>(v1::kUbRows) * v1::kMxuDim;
    const u64 acc_bytes  = static_cast<u64>(v1::kAccRows) * v1::kMxuDim * sizeof(i32);
    const u64 fifo_bytes = v1::kWeightFifoTiles * v1::kTileBytes;

    CHECK_EQ(v1::kTileBytes, 64 * kKiB);
    CHECK_EQ(ub_bytes, 24 * kMiB);
    CHECK_EQ(acc_bytes, 4 * kMiB);
    CHECK_EQ(fifo_bytes, 256 * kKiB);
}

TEST(config_peak_is_92_tops) {
    const u64 macs_per_cycle = static_cast<u64>(v1::kMxuDim) * v1::kMxuDim;
    const u64 ops_per_cycle  = 2 * macs_per_cycle;   // a multiply-add counts as two operations
    const u64 giga_ops       = ops_per_cycle * v1::kClockMhz / 1000;
    CHECK_EQ(giga_ops, u64{91750});                  // the paper rounds this to 92 TOPS
}

TEST(config_bandwidths_match_the_paper) {
    const u64 weight_gb_per_s = static_cast<u64>(v1::kWeightBytesPerCycle) * v1::kClockMhz / 1000;
    const u64 host_gb_per_s   = static_cast<u64>(v1::kHostBytesPerCycle) * v1::kClockMhz / 1000;
    const u64 tile_fetch_cycles = v1::kTileBytes / v1::kWeightBytesPerCycle;

    CHECK_EQ(weight_gb_per_s, u64{33});        // the paper's DDR3 is 34 GB/s; 48 B/cycle rounds down
    CHECK_EQ(host_gb_per_s, u64{15});          // PCIe Gen3 x16 is 15.75 GB/s
    CHECK_EQ(tile_fetch_cycles, u64{1365});
}
