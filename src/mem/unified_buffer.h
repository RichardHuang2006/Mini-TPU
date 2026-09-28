/// Unified Buffer: 24 MiB of int8 activations in 256-byte rows, one row per MXU input vector.

#pragma once

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/config.h"

class UnifiedBuffer {
public:
    static constexpr u32 kRows     = v1::kUbRows;
    static constexpr u32 kRowBytes = v1::kMxuDim;
    static constexpr u64 kBytes    = u64{kRows} * kRowBytes;

    UnifiedBuffer() : data_(kBytes, 0) {}

    // Instructions address the buffer by row.
    i8* row(u32 r) { return data_.data() + offset(r); }
    const i8* row(u32 r) const { return data_.data() + offset(r); }

    // The host interface moves arbitrary byte ranges; addr is a byte address, row-major.
    void write(u64 addr, const i8* src, u64 n) {
        check_range(addr, n);
        std::memcpy(data_.data() + addr, src, n);
    }

    void read(u64 addr, i8* dst, u64 n) const {
        check_range(addr, n);
        std::memcpy(dst, data_.data() + addr, n);
    }

private:
    std::vector<i8> data_;

    static u64 offset(u32 r) {
        if (r >= kRows) throw std::out_of_range("ub row " + std::to_string(r) + " is past the last row " + std::to_string(kRows - 1));
        return u64{r} * kRowBytes;
    }

    static void check_range(u64 addr, u64 n) {
        if (n > kBytes || addr > kBytes - n)
            throw std::out_of_range("ub bytes [" + std::to_string(addr) + ", +" + std::to_string(n) + ") run past 24 MiB");
    }
};
