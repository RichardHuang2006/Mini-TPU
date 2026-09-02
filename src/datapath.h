#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

// The foundational datapath: fixed-width element types, strided tensor views
// over somebody else's storage, and the bit-exact fixed-point arithmetic that
// turns an int32 accumulator back into an int8 activation.
//
// Everything downstream -- the array, the buffers, the activation pipeline and
// both validation models -- is built out of these three pieces.

// ============================================================================
// Fixed-width types and identifier aliases
// ============================================================================

// Activations and weights are int8, accumulators int32; the aliases below name
// the roles so signatures stay readable.

using i8  = int8_t;    // activation / weight element
using i32 = int32_t;   // accumulator element

// Identifier aliases. All are uint32_t; the names only make signatures
// readable and self-documenting at a call site.

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

// ============================================================================
// Tensor views
// ============================================================================

// Row-major 2-D views over storage somebody else owns. The row pitch is an
// explicit `stride` rather than implied by `cols`, which lets a tile be a window
// into a larger buffer instead of a copy: the Unified Buffer hands the array a
// dim x dim rectangle out of a much wider activation matrix.

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

    // Non-const view converts to a const one; the reverse is rejected because
    // the pointer conversion is.
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

    // A window sharing the parent's storage. The stride is inherited, so a
    // sub-tile addresses the same memory the parent does.
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

    // Copy `src` into the top-left corner and zero the remainder. A matmul
    // dimension smaller than the array is padded this way: the result is correct
    // and the padded MAC-cycles are wasted work.
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

// Element-wise comparison. The differential tests lean on this, so it reports
// shape mismatch as inequality rather than asserting.
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

// Owning, contiguous backing store for a view. Tensors appear only at the edges
// of the model (test inputs, host memory images); the simulator itself passes
// views around.
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

// ============================================================================
// Fixed-point requantization
// ============================================================================

// The int32 accumulator leaving the array is scaled back into the int8
// activation domain by a multiply and an arithmetic shift; the parameters are a
// bias, an int32 fixed-point multiplier, and a shift amount.
//
// Three details are pinned by exhaustive tests rather than spot checks, since
// getting one wrong perturbs a handful of output elements out of thousands:
//
//   - the rounding mode is half away from zero, in both signs;
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
    // Mirroring the positive case rounds -2.5 to -3 rather than -2; `>>` alone
    // would floor it.
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

// The activation pipeline's arithmetic, in the one place both the reference model
// and the timed model can call it. Bias is added in int64 so a large bias cannot
// overflow before scaling. Equivalent to requantize() when bias is 0.
inline i8 requantize_biased(i32 acc, i32 bias, i32 multiplier, uint32_t shift) {
    const int64_t biased = static_cast<int64_t>(acc) + bias;
    return saturate(round_shift(biased * multiplier, shift));
}

// ReLU family, applied to the requantized int8 value: the clamp works in the
// output's own quantized units, so ReLU6's bound is the int8 value 6, not 6.0
// on some notional real scale.
inline i8 relu(i8 v) { return v < 0 ? i8{0} : v; }
inline i8 relu6(i8 v) { return v < 0 ? i8{0} : (v > 6 ? i8{6} : v); }

// Window reductions over the requantized int8 stream. pool_max folds one
// element into a running maximum; pool_avg divides a window sum by its area
// with the same round-half-away-from-zero rule requantization uses, so the two
// cannot drift apart on a tie. The reference model keeps its own copy of the
// window walk on purpose -- only this rounding rule is a shared contract.
inline i8 pool_max(i8 best, i8 v) { return v > best ? v : best; }
inline i8 pool_avg(int64_t sum, uint32_t window) {
    return saturate(round_div(sum, static_cast<int64_t>(window) * window));
}

}  // namespace quant
