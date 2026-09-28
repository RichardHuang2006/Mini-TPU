/// DRAM: 8 GiB addressable without allocating it, page-straddling copies, zero reads, bounds.

#include <stdexcept>
#include <string>
#include <vector>

#include "mem/dram.h"
#include "test_framework.h"

TEST(dram_starts_empty_and_reads_zero) {
    const Dram wmem("wmem", v1::kWeightMemBytes);
    std::vector<i8> buf(100, 7);
    wmem.read(v1::kWeightMemBytes - 100, buf.data(), buf.size());
    CHECK(buf == std::vector<i8>(100, 0));
    CHECK_EQ(wmem.pages_allocated(), u64{0});
}

TEST(dram_allocates_only_the_pages_written) {
    Dram wmem("wmem", v1::kWeightMemBytes);
    const i8 last = 42;
    wmem.write(v1::kWeightMemBytes - 1, &last, 1);   // the very last byte of 8 GiB
    CHECK_EQ(wmem.pages_allocated(), u64{1});

    const std::vector<i8> tile(v1::kTileBytes, 3);
    wmem.write(5 * v1::kTileBytes, tile.data(), tile.size());   // an aligned tile is exactly one page
    CHECK_EQ(wmem.pages_allocated(), u64{2});

    i8 back = 0;
    wmem.read(v1::kWeightMemBytes - 1, &back, 1);
    CHECK_EQ(back, 42);
}

TEST(dram_copies_straddle_pages) {
    Dram host("host", u64{1} << 20);
    const u64 edge = Dram::kPageBytes;
    const std::vector<i8> src = {1, 2, 3, 4, 5, 6};
    host.write(edge - 3, src.data(), src.size());
    CHECK_EQ(host.pages_allocated(), u64{2});

    std::vector<i8> back(10, -1);
    host.read(edge - 5, back.data(), back.size());   // 2 zeros, the 6 bytes, 2 zeros
    CHECK(back == std::vector<i8>({0, 0, 1, 2, 3, 4, 5, 6, 0, 0}));
}

TEST(dram_out_of_range_throws_with_its_name) {
    Dram host("host", 1024);
    i8 byte = 0;
    CHECK_THROWS(host.write(1024, &byte, 1));
    CHECK_THROWS(host.read(1023, &byte, 2));
    CHECK_THROWS(host.read(1, &byte, ~u64{0}));
    host.write(1023, &byte, 1);
    host.read(0, &byte, 0);

    std::string what;
    try { host.read(2000, &byte, 1); } catch (const std::out_of_range& e) { what = e.what(); }
    CHECK(what.rfind("host bytes [2000, +1)", 0) == 0);
}
