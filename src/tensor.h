#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "types.h"

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
