/// bf16 conversion: exact values, round-to-nearest-even, NaN, infinity, subnormals, full round trip.

#include <cstring>

#include "common/types.h"
#include "test_framework.h"

namespace {

float f32_from_bits(u32 x) {
    float f = 0.0f;
    std::memcpy(&f, &x, sizeof f);
    return f;
}

u16 round_bits(u32 x) { return bf16::from_f32(f32_from_bits(x)).bits; }

bool is_nan(u16 b) { return (b & 0x7F80u) == 0x7F80u && (b & 0x007Fu) != 0; }

}  // namespace

TEST(bf16_exact_values) {
    CHECK_EQ(bf16::from_f32(1.0f).bits, 0x3F80);
    CHECK_EQ(bf16::from_f32(-2.0f).bits, 0xC000);
    CHECK_EQ(bf16::from_f32(0.0f).bits, 0x0000);
    CHECK_EQ(bf16::from_f32(-0.0f).bits, 0x8000);
    CHECK_EQ(bf16{0x3F80}.to_f32(), 1.0f);
    CHECK_EQ(bf16{0x4049}.to_f32(), 3.140625f);
}

TEST(bf16_rounds_ties_to_even) {
    CHECK_EQ(round_bits(0x3F808000u), 0x3F80);   // 1 + 2^-8: tie, 0x3F80 is even, round down
    CHECK_EQ(round_bits(0x3F818000u), 0x3F82);   // 1 + 3*2^-8: tie, 0x3F81 is odd, round up
    CHECK_EQ(round_bits(0xBF808000u), 0xBF80);   // negative ties behave the same
    CHECK_EQ(round_bits(0xBF818000u), 0xBF82);
}

TEST(bf16_rounds_to_nearest_off_ties) {
    CHECK_EQ(round_bits(0x3F808001u), 0x3F81);   // just above half
    CHECK_EQ(round_bits(0x3F807FFFu), 0x3F80);   // just below half
    CHECK_EQ(round_bits(0x3F80FFFFu), 0x3F81);
    CHECK_EQ(round_bits(0x3FFFFFFFu), 0x4000);   // carry ripples into the exponent
}

TEST(bf16_nan_stays_nan) {
    CHECK(is_nan(round_bits(0x7FC00000u)));      // quiet NaN
    CHECK(is_nan(round_bits(0x7F800001u)));      // payload only in the low bits must not become infinity
    CHECK(is_nan(round_bits(0xFF800001u)));
    CHECK_EQ(round_bits(0xFF800001u) & 0x8000u, 0x8000u);   // sign kept
}

TEST(bf16_infinity_and_overflow) {
    CHECK_EQ(round_bits(0x7F800000u), 0x7F80);
    CHECK_EQ(round_bits(0xFF800000u), 0xFF80);
    CHECK_EQ(round_bits(0x7F7FFFFFu), 0x7F80);   // FLT_MAX is past bf16's largest finite value
    CHECK_EQ(round_bits(0x7F7F0000u), 0x7F7F);   // bf16's largest finite value is exact
}

TEST(bf16_subnormals_round_to_even) {
    CHECK_EQ(round_bits(0x00000001u), 0x0000);
    CHECK_EQ(round_bits(0x00008000u), 0x0000);   // tie at the smallest subnormal step, even is zero
    CHECK_EQ(round_bits(0x00018000u), 0x0002);
    CHECK_EQ(round_bits(0x00010000u), 0x0001);
}

TEST(bf16_every_value_round_trips) {
    int mismatches = 0;
    for (u32 b = 0; b <= 0xFFFFu; ++b) {
        const bf16 v{static_cast<u16>(b)};
        if (is_nan(v.bits)) continue;
        if (!(bf16::from_f32(v.to_f32()) == v)) ++mismatches;
    }
    CHECK_EQ(mismatches, 0);
}
