/// Fundamental types: fixed-width integers, cycles, word addresses, memory spaces and bf16.

#pragma once

#include <cstdint>
#include <cstring>

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using Cycle = u64;
using Addr  = u32;   // word address: every memory holds 32-bit words

enum class Space : u8 { MAIN, VMEM, SMEM };

// Brain float: fp32's sign and 8-bit exponent with the mantissa cut to 7 bits.
struct bf16 {
    u16 bits = 0;

    // Round to nearest, ties to even; NaN stays NaN (made quiet), overflow rounds to infinity.
    static bf16 from_f32(float f) {
        u32 x = 0;
        std::memcpy(&x, &f, sizeof x);
        if ((x & 0x7FFFFFFFu) > 0x7F800000u) return {static_cast<u16>((x >> 16) | 0x0040u)};
        const u32 lsb = (x >> 16) & 1u;
        return {static_cast<u16>((x + 0x7FFFu + lsb) >> 16)};
    }

    float to_f32() const {
        const u32 x = static_cast<u32>(bits) << 16;
        float f = 0.0f;
        std::memcpy(&f, &x, sizeof f);
        return f;
    }

    friend bool operator==(bf16 a, bf16 b) { return a.bits == b.bits; }
};
