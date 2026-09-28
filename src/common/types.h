/// Fixed-width integers and the cycle count, the only types every unit shares.

#pragma once

#include <cstdint>

using u8  = std::uint8_t;
using i8  = std::int8_t;    // activations and weights
using i32 = std::int32_t;   // accumulators
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using Cycle = u64;
