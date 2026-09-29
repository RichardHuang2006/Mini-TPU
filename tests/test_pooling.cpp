/// Pooling: max and average over 2x2 windows of a 4x4 map, band timing, rounding, bad shapes, encoding and syntax.

#include <string>

#include "isa/asm.h"
#include "test_framework.h"
#include "units/activation.h"

namespace {

// A 4x4 feature map in accumulator rows 0-15, pixel (y, x) in row y * 4 + x; column n holds 3 * row - 20 + n.
void fill_map(Accumulators& acc) {
    for (u32 row = 0; row < 16; ++row) {
        for (u32 n = 0; n < Accumulators::kCols; ++n) {
            acc.row(row)[n] = static_cast<i32>(3 * row) - 20 + static_cast<i32>(n);
        }
    }
}

int ub_value(const UnifiedBuffer& ub, u32 row, u32 col) {
    return ub.row(row)[col];
}

void tick_times(ActivationUnit& unit, int cycles) {
    for (int i = 0; i < cycles; ++i) {
        unit.tick();
    }
}

std::string error_of(const std::string& source) {
    try {
        assemble(source, "t.s");
    } catch (const AsmError& e) {
        return e.what();
    }
    return "no error";
}

}  // namespace

TEST(pool_average_rounds_halves_away_from_zero) {
    CHECK_EQ(static_cast<int>(average(6, 4)), 2);     // 1.5
    CHECK_EQ(static_cast<int>(average(5, 4)), 1);     // 1.25
    CHECK_EQ(static_cast<int>(average(-2, 4)), -1);   // -0.5
    CHECK_EQ(static_cast<int>(average(-6, 4)), -2);   // -1.5
    CHECK_EQ(static_cast<int>(average(4 * 127, 4)), 127);
}

TEST(pool_max_2x2_over_a_4x4_map) {
    Accumulators acc;
    UnifiedBuffer ub;
    fill_map(acc);
    ActivationUnit unit(acc, ub);

    unit.start(0, 20, 16, 0, ActivationFunction::Identity, Pooling::Max, 2, 4);
    tick_times(unit, 7);
    CHECK_EQ(ub_value(ub, 20, 0), 0);   // band 0 (image rows 0-1) is not complete until its 8th pixel
    unit.tick();
    CHECK_EQ(ub_value(ub, 20, 0), 3 * 5 - 20);   // window {0, 1, 4, 5}: the largest is row 5
    CHECK_EQ(ub_value(ub, 21, 0), 3 * 7 - 20);   // window {2, 3, 6, 7}

    tick_times(unit, 8);
    CHECK(!unit.busy());
    CHECK_EQ(ub_value(ub, 22, 0), 3 * 13 - 20);
    CHECK_EQ(ub_value(ub, 23, 0), 3 * 15 - 20);
    CHECK_EQ(ub_value(ub, 23, 10), 3 * 15 - 20 + 10);
    CHECK_EQ(ub_value(ub, 24, 0), 0);   // 16 input rows make exactly 4 pooled rows
}

TEST(pool_average_2x2_over_a_4x4_map) {
    Accumulators acc;
    UnifiedBuffer ub;
    fill_map(acc);
    ActivationUnit unit(acc, ub);

    unit.start(0, 0, 16, 0, ActivationFunction::Relu, Pooling::Average, 2, 4);
    tick_times(unit, 16);

    // Window {0, 1, 4, 5} at column 0 is -20, -17, -8, -5; relu makes them all 0.
    CHECK_EQ(ub_value(ub, 0, 0), 0);
    // Window {10, 11, 14, 15} at column 0 is 10, 13, 22, 25: sum 70, average 17.5, rounds to 18.
    CHECK_EQ(ub_value(ub, 3, 0), 18);
    // Window {2, 3, 6, 7} at column 20 is 6, 9, 18, 21: sum 54, average 13.5, rounds to 14.
    CHECK_EQ(ub_value(ub, 1, 20), 14);
}

TEST(pool_needs_whole_windows) {
    Accumulators acc;
    UnifiedBuffer ub;
    ActivationUnit unit(acc, ub);

    CHECK_THROWS(unit.start(0, 0, 20, 0, ActivationFunction::Identity, Pooling::Max, 2, 5));    // width 5 is odd
    CHECK_THROWS(unit.start(0, 0, 12, 0, ActivationFunction::Identity, Pooling::Max, 2, 4));    // 12 rows is 1.5 bands
    CHECK(!unit.busy());
    CHECK_EQ(activate_output_rows(64, Pooling::Average, 2), u32{16});
    CHECK_EQ(activate_output_rows(64, Pooling::None, 0), u32{64});
}

TEST(pool_encoding_and_syntax) {
    Instr in;
    in.op         = Op::Activate;
    in.function   = ActivationFunction::Tanh;
    in.pool       = Pooling::Max;
    in.pool_size  = 2;
    in.pool_width = 8;

    const InstrBytes word = encode(in);
    CHECK_EQ(static_cast<int>(word[1]), 0x27);   // bits 0-1 tanh (3), bits 2-3 max (1), bits 4-6 size (2)
    CHECK_EQ(static_cast<int>(word[11]), 8);

    const Instr back = decode(word);
    CHECK(back.pool == Pooling::Max);
    CHECK_EQ(back.pool_size, u32{2});
    CHECK_EQ(back.pool_width, u32{8});
    CHECK_EQ(disasm(in), std::string("Activate acc=0x0 ub=0x0 rows=0 shift=0 function=tanh pool=max size=2 width=8"));

    const Program p = assemble(disasm(in));
    CHECK(p.code.size() == 1 && p.code[0] == word);

    CHECK_EQ(error_of("Activate acc=0 ub=0 rows=4 shift=0 function=relu pool=min size=2 width=2"),
             std::string("t.s:1:55: error: unknown pooling 'min' (expects none, max, avg)"));
    CHECK_EQ(error_of("Activate acc=0 ub=0 rows=4 shift=0 function=relu pool=max size=8 width=2"),
             std::string("t.s:1:1: error: size=8 must be 1 to 7"));
    CHECK_EQ(error_of("Activate acc=0 ub=0 rows=4 shift=0 function=relu size=2"),
             std::string("t.s:1:1: error: size= and width= only go with pool=max or pool=avg"));
    CHECK_EQ(error_of("Activate acc=0 ub=0 rows=4 shift=0 function=relu pool=avg size=2"),
             std::string("t.s:1:1: error: pooling needs width=, the feature map's width in rows"));
    CHECK_EQ(error_of("Activate acc=0 ub=0 rows=4 shift=0 function=relu stride=2"),
             std::string("t.s:1:50: error: Activate has no operand 'stride' (expects acc, ub, rows, shift, function; "
                         "optional pool, size, width)"));
}
