/// Compiler: dense layers, a 3-layer MLP and pooled convolutions, run on the simulator, equal the reference bit for bit.

#include <stdexcept>
#include <string>

#include "compiler/compiler.h"
#include "core/tpu.h"
#include "isa/asm.h"
#include "test_framework.h"

namespace {

Layer layer(const Matrix8& weights, u32 shift, ActivationFunction function) {
    Layer l;
    l.weights  = weights;
    l.shift    = shift;
    l.function = function;
    return l;
}

// Runs a compiled program to Halt and returns its result; `tpu` keeps the stats for the caller.
Matrix8 run(Tpu& tpu, const Compiled& compiled) {
    tpu.load(compiled.program);
    tpu.run_to_halt();
    return read_result(tpu.host(), compiled);
}

// What one layer should produce, from the reference ops alone.
Matrix8 reference(const Matrix8& input, const Layer& l) {
    const Matrix8 activated = ref_activate(ref_matmul(input, l.weights), l.shift, l.function);
    return ref_pool(activated, l.pool_width, l.pool_size, l.pool);
}

std::string compile_error(const Matrix8& input, const std::vector<Layer>& layers) {
    try {
        compile_layers(input, layers);
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "no error";
}

}  // namespace

TEST(compiler_dense_one_tile) {
    const Matrix8 input = random_matrix(8, 256, 21);
    const Layer   l     = layer(random_matrix(256, 256, 22), 10, ActivationFunction::Relu);

    Tpu tpu;
    CHECK(run(tpu, compile_layers(input, {l})) == reference(input, l));
}

TEST(compiler_dense_many_tiles) {
    // 300 inputs and 280 outputs need 2 x 2 tiles: two multiplies add into each output block, and both pad with zeros.
    const Matrix8 input = random_matrix(16, 300, 23);
    const Layer   l     = layer(random_matrix(300, 280, 24), 10, ActivationFunction::Identity);

    Tpu tpu;
    CHECK(run(tpu, compile_layers(input, {l})) == reference(input, l));

    // With 16 rows per tile the MXU finishes long before DDR3 brings the next tile: TPUv1 is bound by weight bandwidth.
    CHECK(tpu.stats().stalled(Stall::WeightsNotReady) > tpu.cycle() / 2);
}

TEST(compiler_three_layer_mlp) {
    const Matrix8 input = random_matrix(8, 64, 25);
    const Layer first  = layer(random_matrix(64, 200, 26), 9, ActivationFunction::Relu);
    const Layer second = layer(random_matrix(200, 300, 27), 14, ActivationFunction::Tanh);   // tanh inputs near -3..3
    const Layer third  = layer(random_matrix(300, 10, 28), 11, ActivationFunction::Identity);

    const Matrix8 expected = reference(reference(reference(input, first), second), third);
    Tpu tpu;
    CHECK(run(tpu, compile_layers(input, {first, second, third})) == expected);
}

TEST(compiler_convolution_with_max_pooling) {
    // An 8x8 image with 4 channels, 3x3 kernels and 8 filters: a 6x6 map, pooled 2x2 to 3x3.
    const Matrix8 image = random_matrix(8 * 8, 4, 29);
    Layer l = layer(random_matrix(3 * 3 * 4, 8, 30), 9, ActivationFunction::Relu);
    l.pool      = Pooling::Max;
    l.pool_size = 2;

    const Compiled compiled = compile_conv(image, 8, 8, 3, l);
    CHECK_EQ(compiled.result_rows, u32{9});

    const Matrix32 sums     = ref_conv2d(image, 8, 8, 3, l.weights);
    const Matrix8  expected = ref_pool(ref_activate(sums, l.shift, l.function), 6, 2, Pooling::Max);
    Tpu tpu;
    CHECK(run(tpu, compiled) == expected);
}

TEST(compiler_convolution_with_average_pooling) {
    const Matrix8 image = random_matrix(6 * 6, 3, 31);
    Layer l = layer(random_matrix(3 * 3 * 3, 5, 32), 8, ActivationFunction::Tanh);
    l.pool      = Pooling::Average;
    l.pool_size = 2;

    const Matrix32 sums     = ref_conv2d(image, 6, 6, 3, l.weights);
    const Matrix8  expected = ref_pool(ref_activate(sums, l.shift, l.function), 4, 2, Pooling::Average);
    Tpu tpu;
    CHECK(run(tpu, compile_conv(image, 6, 6, 3, l)) == expected);
}

TEST(compiler_program_text_round_trips) {
    const Matrix8 input = random_matrix(4, 40, 33);
    const Layer   l     = layer(random_matrix(40, 12, 34), 7, ActivationFunction::Relu);
    const Compiled compiled = compile_layers(input, {l});

    Compiled from_text = compiled;
    from_text.program = assemble(program_text(compiled.program, "round trip"));
    CHECK(from_text.program.code == compiled.program.code);

    Tpu tpu;
    CHECK(run(tpu, from_text) == reference(input, l));
}

TEST(compiler_rejects_bad_shapes) {
    const Matrix8 input = random_matrix(8, 64, 35);
    CHECK_EQ(compile_error(input, {layer(random_matrix(65, 8, 36), 0, ActivationFunction::Relu)}),
             std::string("layer 0: its weights have 65 rows but its input has 64 columns"));
    CHECK_EQ(compile_error(random_matrix(4097, 8, 37), {layer(random_matrix(8, 8, 38), 0, ActivationFunction::Relu)}),
             std::string("layer 0: 4097 rows; the accumulators hold 1 to 4096"));

    Layer pooled = layer(random_matrix(64, 8, 39), 0, ActivationFunction::Relu);
    pooled.pool       = Pooling::Max;
    pooled.pool_size  = 2;
    pooled.pool_width = 3;
    CHECK_EQ(compile_error(input, {pooled}), std::string("layer 0: pooling needs whole 2x2 windows over a map 3 pixels wide"));
}
