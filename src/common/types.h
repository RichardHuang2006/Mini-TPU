/// Fixed-width integers, the cycle count and size units, the only types every unit shares.

#pragma once

#include <cstdint>

using u8  = std::uint8_t;
using i8  = std::int8_t;    // activations and weights
using i32 = std::int32_t;   // accumulators
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using Cycle = u64;

inline constexpr u64 kKiB = 1024;
inline constexpr u64 kMiB = 1024 * kKiB;
inline constexpr u64 kGiB = 1024 * kMiB;
