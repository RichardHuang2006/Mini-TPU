/// The compiler: lowers layers, multi-layer networks and convolutions to TPUv1 programs over 256 x 256 tiles.

#pragma once

#include <vector>

#include "isa/isa.h"
#include "mem/dram.h"
#include "ref/ref_ops.h"

// One layer: out = pool(function(round(in x weights / 2^shift))), each input row one example or one pixel.
struct Layer {
    Matrix8            weights;                                // K x N int8
    u32                shift    = 0;
    ActivationFunction function = ActivationFunction::Relu;
    Pooling            pool       = Pooling::None;             // over the output pixels, one per row
    u32                pool_size  = 0;                         // with pooling, a pool_size x pool_size window
    u32                pool_width = 0;                         // with pooling, the output feature map's width in pixels
};

// A compiled program and where it leaves its result in host memory.
struct Compiled {
    Program program;
    u32     result_host_row = 0;
    u32     result_rows     = 0;
    u32     result_cols     = 0;
};

// Chains the layers on `input` (M x K rows, M <= 4096); activations stay in the Unified Buffer between layers.
Compiled compile_layers(const Matrix8& input, const std::vector<Layer>& layers);

// A stride-1, unpadded convolution: im2col on the host, then one layer on the TPU; any pool_width is set here.
Compiled compile_conv(const Matrix8& image, u32 height, u32 width, u32 kernel, const Layer& layer);

// One row per output pixel holding its kernel x kernel window, channel fastest: column (dy * kernel + dx) * channels + c.
Matrix8 im2col(const Matrix8& image, u32 height, u32 width, u32 kernel);

// The compiled program's result, read back out of host memory.
Matrix8 read_result(const Dram& host, const Compiled& compiled);
