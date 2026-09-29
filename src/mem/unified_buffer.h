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
    static constexpr u64 kBytes    = static_cast<u64>(kRows) * kRowBytes;

    UnifiedBuffer() {
        bytes_.resize(kBytes, 0);
        written_.resize(kRows, false);
    }

    // Instructions address the buffer by row; the pointer covers that row's 256 bytes. Only writers ask for it.
    i8* row(u32 r) {
        check_row(r);
        written_[r] = true;
        return bytes_.data() + start_of_row(r);
    }

    const i8* row(u32 r) const {
        check_row(r);
        return bytes_.data() + start_of_row(r);
    }

    // The host interface moves byte ranges; addr is a byte address, rows laid end to end.
    void write(u64 addr, const i8* src, u64 n) {
        check_range(addr, n);
        std::memcpy(bytes_.data() + addr, src, n);
        if (n == 0) {
            return;
        }
        for (u64 r = addr / kRowBytes; r <= (addr + n - 1) / kRowBytes; ++r) {
            written_[r] = true;
        }
    }

    void read(u64 addr, i8* dst, u64 n) const {
        check_range(addr, n);
        std::memcpy(dst, bytes_.data() + addr, n);
    }

    // All 24 MiB, row after row, for the visualizer's copy.
    const std::vector<i8>& bytes() const {
        return bytes_;
    }

    // Which rows have been written since the buffer was made, so the visualizer can paint them.
    const std::vector<bool>& written_rows() const {
        return written_;
    }

private:
    std::vector<i8>   bytes_;
    std::vector<bool> written_;

    static u64 start_of_row(u32 r) {
        return static_cast<u64>(r) * kRowBytes;
    }

    static void check_row(u32 r) {
        if (r >= kRows) {
            const std::string last = std::to_string(kRows - 1);
            throw std::out_of_range("ub row " + std::to_string(r) + " is past the last row " + last);
        }
    }

    // Checked as two steps rather than `addr + n > kBytes`, which a huge n could wrap around.
    static void check_range(u64 addr, u64 n) {
        if (n > kBytes) {
            throw_range_error(addr, n);
        }
        if (addr > kBytes - n) {
            throw_range_error(addr, n);
        }
    }

    [[noreturn]] static void throw_range_error(u64 addr, u64 n) {
        const std::string range = "[" + std::to_string(addr) + ", +" + std::to_string(n) + ")";
        throw std::out_of_range("ub bytes " + range + " run past 24 MiB");
    }
};
