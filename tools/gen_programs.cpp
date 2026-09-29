/// gen_programs: compiles the demo networks into programs/*.s and prints what each should leave in host memory.

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "compiler/compiler.h"
#include "isa/asm.h"

namespace {

Layer layer(u32 inputs, u32 outputs, u32 seed, u32 shift, ActivationFunction function) {
    Layer l;
    l.weights  = random_matrix(inputs, outputs, seed);
    l.shift    = shift;
    l.function = function;
    return l;
}

std::string hex(u32 value) {
    char text[16];
    std::snprintf(text, sizeof text, "0x%X", value);
    return text;
}

// Where the result is, as a one-line header for the .s file and as a hint for the terminal.
std::string result_place(const Compiled& compiled) {
    const u32 last_row = compiled.result_host_row + compiled.result_rows - 1;
    return "result in host rows " + hex(compiled.result_host_row) + "-" + hex(last_row) + ", columns 0-" +
           std::to_string(compiled.result_cols - 1);
}

// Writes the program and prints its expected first result row, for checking with host[ROW:1xN] in the terminal.
bool write_program(const std::string& dir, const std::string& name, const std::string& what, const Compiled& compiled,
                   const Matrix8& expected) {
    const std::string path   = dir + "/" + name;
    const std::string header = name + " (written by gen_programs): " + what + "; " + result_place(compiled);

    std::ofstream file(path);
    if (!file) {
        std::cerr << "gen_programs: cannot write " << path << "\n";
        return false;
    }
    file << program_text(compiled.program, header);

    std::cout << path << ": " << compiled.program.code.size() << " instructions; " << result_place(compiled) << "\n";
    std::cout << "  expected host[" << hex(compiled.result_host_row) << ":1x" << expected.cols << "]:";
    for (u32 c = 0; c < expected.cols; ++c) {
        std::cout << " " << static_cast<int>(expected.at(0, c));
    }
    std::cout << "\n";
    return true;
}

Matrix8 reference(const Matrix8& input, const Layer& l) {
    return ref_activate(ref_matmul(input, l.weights), l.shift, l.function);
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = "programs";
    if (argc > 1) {
        dir = argv[1];
    }
    bool ok = true;

    // A dense layer: 16 examples of 64 features to 32 outputs.
    const Matrix8 dense_input = random_matrix(16, 64, 101);
    const Layer   dense_layer = layer(64, 32, 102, 9, ActivationFunction::Relu);
    ok = write_program(dir, "dense.s", "16x64 input, dense 64->32 relu", compile_layers(dense_input, {dense_layer}),
                       reference(dense_input, dense_layer)) && ok;

    // A three-layer MLP: 64 -> 48 -> 32 -> 10.
    const Matrix8 mlp_input = random_matrix(16, 64, 103);
    const Layer   first     = layer(64, 48, 104, 9, ActivationFunction::Relu);
    const Layer   second    = layer(48, 32, 105, 13, ActivationFunction::Tanh);        // tanh wants inputs near -3..3
    const Layer   third     = layer(32, 10, 106, 10, ActivationFunction::Identity);
    const Matrix8 mlp_expected = reference(reference(reference(mlp_input, first), second), third);
    ok = write_program(dir, "mlp.s", "16x64 input, dense 64->48 relu, 48->32 tanh, 32->10 identity",
                       compile_layers(mlp_input, {first, second, third}), mlp_expected) && ok;

    // A convolution: an 8x8 image with 4 channels, 3x3 kernels, 8 filters, relu, then 2x2 max pooling to 3x3.
    const Matrix8 image = random_matrix(8 * 8, 4, 107);
    Layer conv = layer(3 * 3 * 4, 8, 108, 9, ActivationFunction::Relu);
    conv.pool      = Pooling::Max;
    conv.pool_size = 2;
    const Matrix8 conv_sums_activated = ref_activate(ref_conv2d(image, 8, 8, 3, conv.weights), conv.shift, conv.function);
    ok = write_program(dir, "conv.s", "8x8x4 image, 3x3 conv to 8 channels, relu, 2x2 max pool",
                       compile_conv(image, 8, 8, 3, conv), ref_pool(conv_sums_activated, 6, 2, Pooling::Max)) && ok;

    if (ok) {
        return 0;
    }
    return 1;
}
