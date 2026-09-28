/// Host interface: 22 bytes per cycle, data landing progressively, both directions, and range errors up front.

#include <vector>

#include "units/host_interface.h"
#include "test_framework.h"

namespace {

// Fills `rows` host rows starting at `host_row` with 1, 2, 3, ... (wrapping at 127).
void fill_host(Dram& host, u32 host_row, u32 rows) {
    std::vector<i8> bytes(static_cast<u64>(rows) * 256);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<i8>(i % 127 + 1);
    }
    host.write(static_cast<u64>(host_row) * 256, bytes.data(), bytes.size());
}

// Ticks until the transfer finishes and returns how many cycles that took.
int run_to_idle(HostInterface& link) {
    int cycles = 0;
    while (link.busy()) {
        link.tick();
        cycles = cycles + 1;
    }
    return cycles;
}

}  // namespace

TEST(host_one_row_takes_12_cycles) {
    Dram host("host", 1 * kMiB);
    UnifiedBuffer ub;
    HostInterface link(host, ub);
    fill_host(host, 0, 1);

    link.start(Direction::HostToUb, 0, 5, 1);
    CHECK(link.busy());
    CHECK_EQ(run_to_idle(link), 12);   // 256 bytes / 22 per cycle, rounded up

    std::vector<i8> expected(256);
    host.read(0, expected.data(), expected.size());
    const std::vector<i8> landed(ub.row(5), ub.row(5) + 256);
    CHECK(landed == expected);
}

TEST(host_data_lands_22_bytes_per_cycle) {
    Dram host("host", 1 * kMiB);
    UnifiedBuffer ub;
    HostInterface link(host, ub);
    fill_host(host, 0, 1);

    link.start(Direction::HostToUb, 0, 0, 1);
    link.tick();
    CHECK_EQ(link.bytes_done(), u64{22});
    CHECK_EQ(ub.row(0)[21], 22);   // the 22nd byte has arrived
    CHECK_EQ(ub.row(0)[22], 0);    // the 23rd has not
}

TEST(host_multi_row_timing) {
    Dram host("host", 1 * kMiB);
    UnifiedBuffer ub;
    HostInterface link(host, ub);

    link.start(Direction::HostToUb, 0, 0, 4);
    CHECK_EQ(link.bytes_total(), u64{1024});
    CHECK_EQ(run_to_idle(link), 47);   // 1024 / 22 = 46.5, rounded up
}

TEST(host_write_goes_ub_to_host) {
    Dram host("host", 1 * kMiB);
    UnifiedBuffer ub;
    HostInterface link(host, ub);
    ub.row(3)[0]   = -7;
    ub.row(3)[255] = 9;

    link.start(Direction::UbToHost, 10, 3, 1);
    run_to_idle(link);

    i8 first = 0;
    i8 last  = 0;
    host.read(10 * 256, &first, 1);
    host.read(10 * 256 + 255, &last, 1);
    CHECK_EQ(first, -7);
    CHECK_EQ(last, 9);
}

TEST(host_zero_rows_is_done_at_once) {
    Dram host("host", 1 * kMiB);
    UnifiedBuffer ub;
    HostInterface link(host, ub);

    link.start(Direction::HostToUb, 0, 0, 0);
    CHECK(!link.busy());
}

TEST(host_bad_ranges_throw_before_moving_anything) {
    Dram host("host", 4 * 256);   // exactly four rows
    UnifiedBuffer ub;
    HostInterface link(host, ub);
    fill_host(host, 0, 4);

    CHECK_THROWS(link.start(Direction::HostToUb, 3, 0, 2));                     // host rows 3 and 4; row 4 is past the end
    CHECK_THROWS(link.start(Direction::HostToUb, 0, UnifiedBuffer::kRows, 1));  // UB row past the last
    CHECK(!link.busy());
    CHECK_EQ(ub.row(0)[0], 0);   // nothing was copied
}

TEST(host_one_transfer_at_a_time) {
    Dram host("host", 1 * kMiB);
    UnifiedBuffer ub;
    HostInterface link(host, ub);

    link.start(Direction::HostToUb, 0, 0, 1);
    CHECK_THROWS(link.start(Direction::HostToUb, 1, 1, 1));
    CHECK_EQ(link.bytes_total(), u64{256});   // the first transfer is untouched
}
