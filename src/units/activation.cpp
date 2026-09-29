/// The int32-to-int8 arithmetic for each function, and the activation unit's one-row-per-cycle loop.

#include "units/activation.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

i8 saturate(std::int64_t value) {
    if (value > 127) {
        return 127;
    }
    if (value < -128) {
        return -128;
    }
    return static_cast<i8>(value);
}

// value / 2^shift to the nearest integer, halves away from zero; 64 bits, so negating -2^31 cannot overflow.
std::int64_t shift_and_round(i32 value, u32 shift) {
    if (shift == 0) {
        return value;
    }

    const std::int64_t half = std::int64_t{1} << (shift - 1);
    const std::int64_t wide = value;
    if (wide >= 0) {
        return (wide + half) >> shift;
    }

    const std::int64_t magnitude = -wide;
    return -((magnitude + half) >> shift);
}

// The accumulator as the real number it stands for; dividing by a power of two is exact in a double.
double real_value(i32 value, u32 shift) {
    const double scale = static_cast<double>(std::uint64_t{1} << shift);
    return static_cast<double>(value) / scale;
}

// y in Q0.7: the int8 that best stands for y at 1/128 per step; std::round takes halves away from zero.
i8 to_q0_7(double y) {
    const double scaled = std::round(y * 128.0);
    return saturate(static_cast<std::int64_t>(scaled));
}

bool overlaps(u32 first_a, u32 count_a, u32 first_b, u32 count_b) {
    if (count_a == 0 || count_b == 0) {
        return false;
    }
    const u64 end_a = static_cast<u64>(first_a) + count_a;
    const u64 end_b = static_cast<u64>(first_b) + count_b;
    return first_a < end_b && first_b < end_a;
}

// Checked as two steps rather than `first + count > limit`, which a huge count could wrap around.
void check_rows(const std::string& memory, u32 first, u32 count, u32 limit) {
    bool fits = true;
    if (count > limit) {
        fits = false;
    } else if (first > limit - count) {
        fits = false;
    }

    if (!fits) {
        const std::string range = "[" + std::to_string(first) + ", +" + std::to_string(count) + ")";
        throw std::out_of_range(memory + " rows " + range + " run past the last row " + std::to_string(limit - 1));
    }
}

}  // namespace

i8 activate(i32 value, u32 shift, ActivationFunction function) {
    switch (function) {
        case ActivationFunction::Identity: {
            return saturate(shift_and_round(value, shift));
        }
        case ActivationFunction::Relu: {
            const i8 out = saturate(shift_and_round(value, shift));
            if (out < 0) {
                return 0;
            }
            return out;
        }
        case ActivationFunction::Sigmoid: {
            const double x = real_value(value, shift);
            const double y = 1.0 / (1.0 + std::exp(-x));
            return to_q0_7(y);
        }
        case ActivationFunction::Tanh: {
            const double x = real_value(value, shift);
            return to_q0_7(std::tanh(x));
        }
    }
    return 0;
}

ActivationUnit::ActivationUnit(const Accumulators& acc, UnifiedBuffer& ub) : acc_(acc), ub_(ub) {}

void ActivationUnit::start(u32 acc_row, u32 ub_row, u32 rows, u32 shift, ActivationFunction function) {
    if (busy()) {
        throw std::logic_error("activation unit: an Activate is already running");
    }

    // Both ranges are checked before any row moves, so a bad Activate changes nothing.
    check_rows("acc", acc_row, rows, Accumulators::kRows);
    check_rows("ub", ub_row, rows, UnifiedBuffer::kRows);

    acc_row_    = acc_row;
    ub_row_     = ub_row;
    rows_total_ = rows;
    rows_done_  = 0;
    shift_      = shift;
    function_   = function;
}

void ActivationUnit::tick() {
    if (!busy()) {
        return;
    }

    const i32* in  = acc_.row(acc_row_ + rows_done_);
    i8*        out = ub_.row(ub_row_ + rows_done_);
    for (u32 n = 0; n < Accumulators::kCols; ++n) {
        out[n] = activate(in[n], shift_, function_);
    }

    rows_done_ = rows_done_ + 1;
}

void ActivationUnit::reset() {
    acc_row_    = 0;
    ub_row_     = 0;
    rows_total_ = 0;
    rows_done_  = 0;
    shift_      = 0;
    function_   = ActivationFunction::Identity;
}

bool ActivationUnit::busy() const {
    return rows_done_ < rows_total_;
}

bool ActivationUnit::reads_acc_rows(u32 first, u32 count) const {
    if (!busy()) {
        return false;
    }
    return overlaps(acc_row_, rows_total_, first, count);
}

bool ActivationUnit::writes_ub_rows(u32 first, u32 count) const {
    if (!busy()) {
        return false;
    }
    return overlaps(ub_row_, rows_total_, first, count);
}

u32 ActivationUnit::acc_row() const {
    return acc_row_;
}

u32 ActivationUnit::ub_row() const {
    return ub_row_;
}

u32 ActivationUnit::rows_done() const {
    return rows_done_;
}

u32 ActivationUnit::rows_total() const {
    return rows_total_;
}

u32 ActivationUnit::shift() const {
    return shift_;
}

ActivationFunction ActivationUnit::function() const {
    return function_;
}
