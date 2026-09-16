#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

using i8  = int8_t;    // activation / weight element
using i32 = int32_t;   // accumulator element

using UbAddr   = uint32_t;   // byte offset into the Unified Buffer
using HostAddr = uint32_t;   // byte offset into host memory
using BankId   = uint32_t;   // accumulator (or UB) bank index
using TileId   = uint32_t;   // weight tile staged through the weight FIFO

// Out-of-band sentinels, disjoint from every legal index.
inline constexpr UbAddr   INVALID_UBADDR   = std::numeric_limits<UbAddr>::max();
inline constexpr HostAddr INVALID_HOSTADDR = std::numeric_limits<HostAddr>::max();
inline constexpr BankId   INVALID_BANK     = std::numeric_limits<BankId>::max();
inline constexpr TileId   INVALID_TILE     = std::numeric_limits<TileId>::max();

// Activation function applied after requantization in the activation pipeline.
enum class ActFn : uint8_t {
    IDENTITY,
    RELU,
    RELU6,
};

// Optional pooling stage on the requantized int8 stream.
enum class Pool : uint8_t {
    NONE,
    MAX,
    AVG,
};

// Row-major 2-D view over storage somebody else owns, with an explicit pitch.
template <typename T>
class TensorView {
public:
    TensorView() = default;

    TensorView(T* data, uint32_t rows, uint32_t cols, uint32_t stride)
        : data_(data), rows_(rows), cols_(cols), stride_(stride) {
        assert(stride >= cols);
    }

    // Contiguous rows.
    TensorView(T* data, uint32_t rows, uint32_t cols)
        : TensorView(data, rows, cols, cols) {}

    // Non-const view converts to a const one; the reverse is rejected.
    template <typename U,
              typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
    TensorView(const TensorView<U>& other)
        : data_(other.data()), rows_(other.rows()), cols_(other.cols()),
          stride_(other.stride()) {}

    uint32_t rows() const { return rows_; }
    uint32_t cols() const { return cols_; }
    uint32_t stride() const { return stride_; }
    bool empty() const { return rows_ == 0 || cols_ == 0; }
    T* data() const { return data_; }

    T& at(uint32_t r, uint32_t c) const {
        assert(r < rows_ && c < cols_);
        return data_[static_cast<std::size_t>(r) * stride_ + c];
    }

    // A window sharing the parent's storage, inheriting its stride.
    TensorView tile(uint32_t r0, uint32_t c0, uint32_t rows, uint32_t cols) const {
        assert(r0 + rows <= rows_ && c0 + cols <= cols_);
        return TensorView(&data_[static_cast<std::size_t>(r0) * stride_ + c0],
                          rows, cols, stride_);
    }

    void fill(T v) const {
        for (uint32_t r = 0; r < rows_; ++r) {
            for (uint32_t c = 0; c < cols_; ++c) at(r, c) = v;
        }
    }

    // Copy `src` into the top-left corner and zero the remainder.
    template <typename U>
    void zero_pad_from(const TensorView<U>& src) const {
        assert(src.rows() <= rows_ && src.cols() <= cols_);
        for (uint32_t r = 0; r < rows_; ++r) {
            for (uint32_t c = 0; c < cols_; ++c) {
                const bool inside = r < src.rows() && c < src.cols();
                at(r, c) = inside ? static_cast<T>(src.at(r, c)) : T{0};
            }
        }
    }

    // Copy a same-shaped source, stride differences included.
    template <typename U>
    void copy_from(const TensorView<U>& src) const {
        assert(src.rows() == rows_ && src.cols() == cols_);
        for (uint32_t r = 0; r < rows_; ++r) {
            for (uint32_t c = 0; c < cols_; ++c) {
                at(r, c) = static_cast<T>(src.at(r, c));
            }
        }
    }

private:
    T*       data_   = nullptr;
    uint32_t rows_   = 0;
    uint32_t cols_   = 0;
    uint32_t stride_ = 0;
};

// Reports a shape mismatch as inequality rather than asserting.
template <typename A, typename B>
bool same_values(const TensorView<A>& a, const TensorView<B>& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    for (uint32_t r = 0; r < a.rows(); ++r) {
        for (uint32_t c = 0; c < a.cols(); ++c) {
            if (a.at(r, c) != b.at(r, c)) return false;
        }
    }
    return true;
}

// Owning, contiguous backing store for a view.
template <typename T>
class Tensor {
public:
    Tensor() = default;

    Tensor(uint32_t rows, uint32_t cols, T init = T{0})
        : buf_(static_cast<std::size_t>(rows) * cols, init), rows_(rows), cols_(cols) {}

    uint32_t rows() const { return rows_; }
    uint32_t cols() const { return cols_; }
    std::size_t size() const { return buf_.size(); }

    TensorView<T> view() { return TensorView<T>(buf_.data(), rows_, cols_); }
    TensorView<const T> view() const { return TensorView<const T>(buf_.data(), rows_, cols_); }

    T* data() { return buf_.data(); }
    const T* data() const { return buf_.data(); }

private:
    std::vector<T> buf_;
    uint32_t       rows_ = 0;
    uint32_t       cols_ = 0;
};

using I8View       = TensorView<i8>;
using ConstI8View  = TensorView<const i8>;
using I32View      = TensorView<i32>;
using ConstI32View = TensorView<const i32>;

using I8Tensor  = Tensor<i8>;
using I32Tensor = Tensor<i32>;

// The int32-to-int8 pipeline; see README section 12 for the full sequence.
namespace quant {

inline constexpr i32 I8_MIN = -128;
inline constexpr i32 I8_MAX = 127;

// Largest shift for which multiply-then-round stays inside int64.
inline constexpr uint32_t MAX_SHIFT = 31;

// Divide by 2^shift, rounding halves away from zero.
inline int64_t round_shift(int64_t v, uint32_t shift) {
    assert(shift <= MAX_SHIFT);
    if (shift == 0) return v;
    const int64_t half = int64_t{1} << (shift - 1);
    return v >= 0 ? (v + half) >> shift : -((-v + half) >> shift);
}

// The same rounding for a divisor that is not a power of two.
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

// Bias is added in int64 so a large bias cannot overflow before scaling.
inline i8 requantize_biased(i32 acc, i32 bias, i32 multiplier, uint32_t shift) {
    const int64_t biased = static_cast<int64_t>(acc) + bias;
    return saturate(round_shift(biased * multiplier, shift));
}

// Applied to the requantized int8 value, so ReLU6's bound is the int8 value 6.
inline i8 relu(i8 v) { return v < 0 ? i8{0} : v; }
inline i8 relu6(i8 v) { return v < 0 ? i8{0} : (v > 6 ? i8{6} : v); }

// Window reductions over the requantized int8 stream.
inline i8 pool_max(i8 best, i8 v) { return v > best ? v : best; }
inline i8 pool_avg(int64_t sum, uint32_t window) {
    return saturate(round_div(sum, static_cast<int64_t>(window) * window));
}

}  // namespace quant
