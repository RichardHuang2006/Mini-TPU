/// Tiling, memory layout and instruction order: weights prefetched 4 tiles ahead, accumulators double-buffered.

#include "compiler/compiler.h"

#include <stdexcept>
#include <string>

#include "common/config.h"

namespace {

constexpr u32 kDim             = v1::kMxuDim;
constexpr u32 kPrefetchTiles   = 4;                   // Read_Weights run this many tiles ahead of the MatrixMultiply using them
constexpr u32 kAccBufferRows   = v1::kAccRows / 2;    // each half of the accumulators is one buffer

// How many 256-wide column blocks a matrix with `cols` columns needs.
u32 block_count(u32 cols) {
    return (cols + kDim - 1) / kDim;
}

// Columns [256 * block, +256) of every row, zero-padded, row after row: how a matrix sits in host memory and the UB.
std::vector<i8> column_block(const Matrix8& m, u32 block) {
    std::vector<i8> bytes(static_cast<std::size_t>(m.rows) * kDim, 0);
    for (u32 r = 0; r < m.rows; ++r) {
        for (u32 c = 0; c < kDim; ++c) {
            const u32 col = block * kDim + c;
            if (col < m.cols) {
                bytes[static_cast<std::size_t>(r) * kDim + c] = m.at(r, col);
            }
        }
    }
    return bytes;
}

// Rows [256 * kb, +256) and columns [256 * nb, +256) of the weights, zero-padded: one Weight Memory tile.
std::vector<i8> weight_tile(const Matrix8& w, u32 kb, u32 nb) {
    std::vector<i8> bytes(v1::kTileBytes, 0);
    for (u32 k = 0; k < kDim; ++k) {
        for (u32 n = 0; n < kDim; ++n) {
            const u32 row = kb * kDim + k;
            const u32 col = nb * kDim + n;
            if (row < w.rows && col < w.cols) {
                bytes[static_cast<std::size_t>(k) * kDim + n] = w.at(row, col);
            }
        }
    }
    return bytes;
}

Instr host_transfer(Op op, u32 host_row, u32 ub_row, u32 rows) {
    Instr in;
    in.op       = op;
    in.host_row = host_row;
    in.ub_row   = ub_row;
    in.rows     = rows;
    return in;
}

Instr read_weights(u32 tile) {
    Instr in;
    in.op   = Op::ReadWeights;
    in.tile = tile;
    return in;
}

Instr matrix_multiply(u32 ub_row, u32 acc_row, u32 rows, bool accumulate) {
    Instr in;
    in.op          = Op::MatrixMultiply;
    in.ub_row      = ub_row;
    in.acc_row     = acc_row;
    in.rows        = rows;
    in.accumulate  = 0;
    if (accumulate) {
        in.accumulate = 1;
    }
    in.new_weights = 1;   // every multiply uses the next tile
    return in;
}

Instr activate(const Layer& layer, u32 acc_row, u32 ub_row, u32 rows) {
    Instr in;
    in.op         = Op::Activate;
    in.acc_row    = acc_row;
    in.ub_row     = ub_row;
    in.rows       = rows;
    in.shift      = layer.shift;
    in.function   = layer.function;
    in.pool       = layer.pool;
    in.pool_size  = layer.pool_size;
    in.pool_width = layer.pool_width;
    return in;
}

// Where one layer reads its input and writes its output in the Unified Buffer.
struct LayerPlan {
    u32 in_ub      = 0;
    u32 in_rows    = 0;
    u32 in_blocks  = 0;
    u32 out_ub     = 0;
    u32 out_rows   = 0;   // fewer than in_rows when the layer pools
    u32 out_blocks = 0;
};

void check_layer(const Layer& layer, u32 rows, u32 in_cols, std::size_t index) {
    const std::string name = "layer " + std::to_string(index);
    if (layer.weights.rows != in_cols) {
        throw std::invalid_argument(name + ": its weights have " + std::to_string(layer.weights.rows) + " rows but its input has " +
                                    std::to_string(in_cols) + " columns");
    }
    if (rows == 0 || rows > v1::kAccRows) {
        throw std::invalid_argument(name + ": " + std::to_string(rows) + " rows; the accumulators hold 1 to 4096");
    }
    if (layer.pool == Pooling::None) {
        return;
    }
    const bool shape_ok = layer.pool_size >= 1 && layer.pool_width >= 1 && layer.pool_width <= 255 &&
                          layer.pool_width % layer.pool_size == 0 && rows % (layer.pool_width * layer.pool_size) == 0;
    if (!shape_ok) {
        throw std::invalid_argument(name + ": pooling needs whole " + std::to_string(layer.pool_size) + "x" +
                                    std::to_string(layer.pool_size) + " windows over a map " + std::to_string(layer.pool_width) +
                                    " pixels wide");
    }
}

}  // namespace

Compiled compile_layers(const Matrix8& input, const std::vector<Layer>& layers) {
    if (layers.empty()) {
        throw std::invalid_argument("compile_layers needs at least one layer");
    }

    Compiled out;
    Program& program = out.program;

    // The input goes to host row 0, one column block after another.
    const u32 in_blocks = block_count(input.cols);
    for (u32 b = 0; b < in_blocks; ++b) {
        const u64 host_row = static_cast<u64>(b) * input.rows;
        program.host.push_back({host_row * kDim, column_block(input, b)});
    }

    // Plan every layer's UB regions, one after another, and check the shapes.
    std::vector<LayerPlan> plans;
    u32 next_ub  = 0;
    u32 rows     = input.rows;
    u32 cols     = input.cols;
    bool double_buffer = true;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const Layer& layer = layers[i];
        check_layer(layer, rows, cols, i);

        LayerPlan plan;
        plan.in_rows    = rows;
        plan.in_blocks  = block_count(cols);
        plan.out_rows   = rows;
        plan.out_blocks = block_count(layer.weights.cols);
        if (layer.pool != Pooling::None) {
            plan.out_rows = rows / (layer.pool_size * layer.pool_size);
        }
        if (i == 0) {
            plan.in_ub = next_ub;
            next_ub = next_ub + plan.in_blocks * plan.in_rows;
        } else {
            plan.in_ub = plans.back().out_ub;
        }
        plan.out_ub = next_ub;
        next_ub = next_ub + plan.out_blocks * plan.out_rows;
        if (next_ub > v1::kUbRows) {
            throw std::invalid_argument("the activations need " + std::to_string(next_ub) + " UB rows; there are 98304");
        }
        if (rows > kAccBufferRows) {
            double_buffer = false;   // a layer this tall needs the whole accumulator memory
        }

        plans.push_back(plan);
        rows = plan.out_rows;
        cols = layer.weights.cols;
    }

    // Weight tiles go to Weight Memory in the order the multiplies use them.
    u32 tile_count = 0;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        for (u32 nb = 0; nb < plans[i].out_blocks; ++nb) {
            for (u32 kb = 0; kb < plans[i].in_blocks; ++kb) {
                const u64 tile_addr = static_cast<u64>(tile_count) * v1::kTileBytes;
                program.weights.push_back({tile_addr, weight_tile(layers[i].weights, kb, nb)});
                tile_count = tile_count + 1;
            }
        }
    }

    // Start fetching the first tiles at once: DDR3 is slow, and it runs alongside the input's PCIe transfer.
    for (u32 t = 0; t < tile_count && t < kPrefetchTiles; ++t) {
        program.code.push_back(encode(read_weights(t)));
    }
    for (u32 b = 0; b < in_blocks; ++b) {
        const u32 row = b * input.rows;
        program.code.push_back(encode(host_transfer(Op::ReadHostMemory, row, plans[0].in_ub + row, input.rows)));
    }

    // Each output block: one multiply per input block, adding into the same accumulator rows, then one Activate.
    u32 multiplies = 0;
    u32 activates  = 0;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const LayerPlan& plan = plans[i];
        for (u32 nb = 0; nb < plan.out_blocks; ++nb) {
            u32 acc_row = 0;
            if (double_buffer) {
                acc_row = (activates % 2) * kAccBufferRows;   // the next block's multiplies run while this one activates
            }

            for (u32 kb = 0; kb < plan.in_blocks; ++kb) {
                const u32 ub_row = plan.in_ub + kb * plan.in_rows;
                program.code.push_back(encode(matrix_multiply(ub_row, acc_row, plan.in_rows, kb > 0)));

                const u32 ahead = multiplies + kPrefetchTiles;
                if (ahead < tile_count) {
                    program.code.push_back(encode(read_weights(ahead)));
                }
                multiplies = multiplies + 1;
            }

            const u32 out_ub = plan.out_ub + nb * plan.out_rows;
            program.code.push_back(encode(activate(layers[i], acc_row, out_ub, plan.in_rows)));
            activates = activates + 1;
        }
    }

    // The last layer's output goes back to the host, right after the input.
    const LayerPlan& last = plans.back();
    out.result_host_row = in_blocks * input.rows;
    out.result_rows     = last.out_rows;
    out.result_cols     = layers.back().weights.cols;
    for (u32 b = 0; b < last.out_blocks; ++b) {
        const u32 offset = b * last.out_rows;
        program.code.push_back(encode(host_transfer(Op::WriteHostMemory, out.result_host_row + offset, last.out_ub + offset, last.out_rows)));
    }

    Instr halt;
    halt.op = Op::Halt;
    program.code.push_back(encode(halt));
    return out;
}

Matrix8 im2col(const Matrix8& image, u32 height, u32 width, u32 kernel) {
    const u32 channels   = image.cols;
    const u32 out_height = height - kernel + 1;
    const u32 out_width  = width - kernel + 1;
    Matrix8 out(out_height * out_width, kernel * kernel * channels);

    for (u32 y = 0; y < out_height; ++y) {
        for (u32 x = 0; x < out_width; ++x) {
            for (u32 dy = 0; dy < kernel; ++dy) {
                for (u32 dx = 0; dx < kernel; ++dx) {
                    for (u32 c = 0; c < channels; ++c) {
                        const u32 pixel = (y + dy) * width + (x + dx);
                        const u32 col   = (dy * kernel + dx) * channels + c;
                        out.at(y * out_width + x, col) = image.at(pixel, c);
                    }
                }
            }
        }
    }
    return out;
}

Compiled compile_conv(const Matrix8& image, u32 height, u32 width, u32 kernel, const Layer& layer) {
    const bool shape_ok = image.rows == height * width && kernel >= 1 && kernel <= height && kernel <= width;
    if (!shape_ok) {
        throw std::invalid_argument("the image must be height x width pixels, one per row, and at least kernel x kernel");
    }

    Layer on_pixels = layer;
    if (layer.pool != Pooling::None) {
        on_pixels.pool_width = width - kernel + 1;
    }
    return compile_layers(im2col(image, height, width, kernel), {on_pixels});
}

Matrix8 read_result(const Dram& host, const Compiled& compiled) {
    Matrix8 result(compiled.result_rows, compiled.result_cols);
    std::vector<i8> row(kDim, 0);

    for (u32 b = 0; b < block_count(compiled.result_cols); ++b) {
        for (u32 r = 0; r < compiled.result_rows; ++r) {
            const u64 host_row = static_cast<u64>(compiled.result_host_row) + static_cast<u64>(b) * compiled.result_rows + r;
            host.read(host_row * kDim, row.data(), kDim);
            for (u32 c = 0; c < kDim; ++c) {
                const u32 col = b * kDim + c;
                if (col < compiled.result_cols) {
                    result.at(r, col) = row[c];
                }
            }
        }
    }
    return result;
}
