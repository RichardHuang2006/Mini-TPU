/// Unified Buffer: size, zero start, row and byte views agree, and every out-of-range access throws.

#include <vector>

#include "mem/unified_buffer.h"
#include "test_framework.h"

TEST(ub_is_24_mib_of_zeros) {
    const UnifiedBuffer ub;
    CHECK_EQ(UnifiedBuffer::kBytes, u64{24} << 20);
    CHECK_EQ(ub.row(0)[0], 0);
    CHECK_EQ(ub.row(UnifiedBuffer::kRows - 1)[255], 0);
}

TEST(ub_row_writes_read_back) {
    UnifiedBuffer ub;
    const u32 last = UnifiedBuffer::kRows - 1;
    for (u32 c = 0; c < UnifiedBuffer::kRowBytes; ++c) ub.row(last)[c] = static_cast<i8>(c - 128);
    CHECK_EQ(ub.row(last)[0], -128);
    CHECK_EQ(ub.row(last)[255], 127);
    CHECK_EQ(ub.row(last - 1)[255], 0);   // neighbouring row untouched
}

TEST(ub_byte_ranges_are_row_major) {
    UnifiedBuffer ub;
    const std::vector<i8> src = {1, 2, 3, 4};
    ub.write(2 * 256 + 254, src.data(), src.size());   // straddles rows 2 and 3
    CHECK_EQ(ub.row(2)[254], 1);
    CHECK_EQ(ub.row(2)[255], 2);
    CHECK_EQ(ub.row(3)[0], 3);
    CHECK_EQ(ub.row(3)[1], 4);

    std::vector<i8> back(4, 0);
    ub.read(2 * 256 + 254, back.data(), back.size());
    CHECK(back == src);
}

TEST(ub_out_of_range_throws) {
    UnifiedBuffer ub;
    i8 byte = 0;
    CHECK_THROWS(ub.row(UnifiedBuffer::kRows));
    CHECK_THROWS(ub.read(UnifiedBuffer::kBytes, &byte, 1));
    CHECK_THROWS(ub.write(UnifiedBuffer::kBytes - 1, &byte, 2));
    CHECK_THROWS(ub.read(1, &byte, ~u64{0}));   // length that would wrap the address
    ub.write(UnifiedBuffer::kBytes - 1, &byte, 1);   // the very last byte is fine
    ub.read(0, &byte, 0);                            // empty range is fine
}
