/// State copy: it matches the machine at the cycle taken, and the endpoint bytes decode back to the exact values.

#include <string>

#include "isa/asm.h"
#include "test_framework.h"
#include "viz/state.h"

namespace {

// Two input rows times a small weight tile, stopped partway through the multiply.
const char* kMatmul =
    ".host 0\n"
    "1 2 3 4\n"
    ".host 1\n"
    "-1 0 2 5\n"
    ".weights 0 0\n"
    "1 0 0 0 2\n"
    ".weights 0 1\n"
    "0 1 0 0 2\n"
    "Read_Host_Memory host=0 ub=0 rows=2\n"
    "Read_Weights tile=0\n"
    "Read_Weights tile=1\n"
    "MatrixMultiply ub=0 acc=0 rows=2 accumulate=0 new_weights=1\n"
    "Halt\n";

i32 i32_at(const std::string& bytes, std::size_t at) {
    u32 bits = 0;
    for (u32 i = 0; i < 4; ++i) {
        const u32 byte = static_cast<u8>(bytes[at + i]);
        bits = bits | (byte << (8 * i));
    }
    return static_cast<i32>(bits);
}

int i8_at(const std::string& bytes, std::size_t at) {
    return static_cast<i8>(bytes[at]);
}

}  // namespace

TEST(state_copies_the_machine_mid_multiply) {
    Tpu tpu;
    tpu.load(assemble(kMatmul));
    tpu.next_instructions(4);   // the MatrixMultiply has just issued
    tpu.run_cycles(3);

    const auto state = capture(tpu);
    CHECK_EQ(state->cycle, tpu.cycle());
    CHECK_EQ(state->host_pages.size(), std::size_t{1});
    CHECK_EQ(state->wmem_pages.size(), std::size_t{1});   // both .weights rows are in tile 0

    // Every PE register in the copy equals the live array's.
    int mismatches = 0;
    for (u32 k = 0; k < 256; ++k) {
        for (u32 n = 0; n < 256; ++n) {
            const Pe& live = tpu.mxu().pe(k, n);
            const Pe& copy = state->pes[k * 256 + n];
            if (live.act != copy.act || live.psum != copy.psum || live.row != copy.row) {
                mismatches = mismatches + 1;
            }
        }
    }
    CHECK_EQ(mismatches, 0);

    // The copy stays as it was while the machine moves on.
    tpu.run_to_halt();
    CHECK(state->cycle < tpu.cycle());
}

TEST(state_row_bytes_decode_to_the_values) {
    Tpu tpu;
    tpu.load(assemble(kMatmul));
    tpu.run_to_halt();
    const auto state = capture(tpu);

    std::string ub;
    CHECK(rows_bytes(*state, "ub", 1, 1, ub));
    CHECK_EQ(ub.size(), std::size_t{256});
    CHECK_EQ(i8_at(ub, 0), -1);
    CHECK_EQ(i8_at(ub, 3), 5);

    std::string acc;
    CHECK(rows_bytes(*state, "acc", 0, 2, acc));
    CHECK_EQ(acc.size(), std::size_t{2 * 256 * 4});
    CHECK_EQ(i32_at(acc, 4 * 4), 2 * 1 + 2 * 2);         // row 0, column 4: x = (1, 2, 3, 4), W column 4 = (2, 2, 0, 0)
    CHECK_EQ(i32_at(acc, (256 + 4) * 4), 2 * -1 + 2 * 0);  // row 1, column 4

    std::string host;
    CHECK(rows_bytes(*state, "host", 1, 1, host));   // host rows run across the pages in use
    CHECK_EQ(i8_at(host, 3), 5);

    std::string wmem;
    CHECK(rows_bytes(*state, "wmem", 1, 1, wmem));   // tile 0, row 1
    CHECK_EQ(i8_at(wmem, 4), 2);

}

TEST(state_fifo_rows_cover_every_slot) {
    Tpu tpu;
    tpu.load(assemble(kMatmul));
    tpu.next_instructions(3);   // both Read_Weights have issued; tile 0 is arriving and tile 1 waits behind it
    const auto state = capture(tpu);

    CHECK_EQ(state->fifo_tiles.size(), std::size_t{2});
    std::string fifo;
    CHECK(rows_bytes(*state, "fifo", 511, 1, fifo));   // the last row of the second slot
    CHECK_EQ(fifo.size(), std::size_t{256});
    CHECK(!rows_bytes(*state, "fifo", 512, 1, fifo));   // there is no third slot
}

TEST(state_pe_bytes) {
    Tpu tpu;
    tpu.load(assemble(kMatmul));
    tpu.next_instructions(4);
    tpu.run_cycles(3);
    const auto state = capture(tpu);

    std::string pes;
    CHECK(pes_bytes(*state, 0, 2, pes));
    CHECK_EQ(pes.size(), std::size_t{2 * 256 * 12});
    const Pe& live = tpu.mxu().pe(1, 1);
    const std::size_t at = (256 + 1) * 12;   // PE (1, 1)
    CHECK_EQ(i8_at(pes, at), static_cast<int>(tpu.mxu().active_weights()[256 + 1]));
    CHECK_EQ(i8_at(pes, at + 2), static_cast<int>(live.act));
    CHECK_EQ(i32_at(pes, at + 4), live.psum);
    CHECK_EQ(i32_at(pes, at + 8), live.row);
}

TEST(state_rejects_bad_requests) {
    Tpu tpu;
    tpu.load(assemble(kMatmul));
    const auto state = capture(tpu);
    std::string out;
    CHECK(!rows_bytes(*state, "nope", 0, 1, out));
    CHECK(!rows_bytes(*state, "ub", 98304, 1, out));   // past the last row
    CHECK(!rows_bytes(*state, "ub", 0, 0, out));
    CHECK(!pes_bytes(*state, 255, 2, out));
}
