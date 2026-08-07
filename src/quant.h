#pragma once

#include <cassert>
#include <cstdint>

#include "types.h"

// Fixed-point requantization: the int32 accumulator leaving the array is scaled
// back into the int8 activation domain by a multiply and an arithmetic shift.
//
// Three details here are each a bug magnet, and all three are pinned by
// exhaustive tests rather than spot checks, because getting one wrong perturbs
// a handful of output elements out of thousands:
//
//   - the rounding mode is half *away from zero*, in both signs;
//   - a negative accumulator must round away from zero too, which a bare
//     arithmetic shift (round toward negative infinity) does not do;
//   - saturation clamps at both ends, not just the positive one.

namespace quant {

inline constexpr i32 I8_MIN = -128;
inline constexpr i32 I8_MAX = 127;

// Largest shift the multiply-then-round sequence stays inside int64 for: the
// product of two int32 is at most 2^62 in magnitude, and the rounding term adds
// 2^(shift-1) on top of it.
inline constexpr uint32_t MAX_SHIFT = 31;

// Divide by 2^shift, rounding halves away from zero.
inline int64_t round_shift(int64_t v, uint32_t shift) {
    assert(shift <= MAX_SHIFT);
    if (shift == 0) return v;
    const int64_t half = int64_t{1} << (shift - 1);
    // Mirroring the positive case is what makes -2.5 round to -3 rather than
    // to -2; `>>` alone would floor it.
    return v >= 0 ? (v + half) >> shift : -((-v + half) >> shift);
}

// The same rounding for a divisor that is not a power of two. Average pooling
// divides by a window area, so it shares this rule and its edge-case tests.
inline int64_t round_div(int64_t num, int64_t den) {
    assert(den > 0);
    const int64_t half = den / 2;
    return num >= 0 ? (num + half) / den : -((-num + half) / den);
}

// Saturating narrow to int8.
inline i8 saturate(int64_t v) {
    if (v < I8_MIN) return static_cast<i8>(I8_MIN);
    if (v > I8_MAX) return static_cast<i8>(I8_MAX);
    return static_cast<i8>(v);
}

// acc * multiplier / 2^shift, rounded half away from zero, clamped to int8.
inline i8 requantize(i32 acc, i32 multiplier, uint32_t shift) {
    const int64_t product = static_cast<int64_t>(acc) * multiplier;
    return saturate(round_shift(product, shift));
}

// The activation pipeline's arithmetic, in the one place both the reference
// model and the timed model can call it: bias is added in int64 so a large bias
// cannot overflow before scaling. Equivalent to requantize() when bias is 0.
inline i8 requantize_biased(i32 acc, i32 bias, i32 multiplier, uint32_t shift) {
    const int64_t biased = static_cast<int64_t>(acc) + bias;
    return saturate(round_shift(biased * multiplier, shift));
}

}  // namespace quant
