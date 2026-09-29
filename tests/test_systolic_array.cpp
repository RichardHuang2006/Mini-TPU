/// MXU: results equal a plain loop, exact wavefront timing, accumulate mode, double-buffered weights, and the UB hazard.

#include <vector>

#include "core/tpu.h"
#include "test_framework.h"

namespace {

constexpr u32 kDim = SystolicArray::kDim;

// A small fixed pseudo-random sequence of int8 values, so every run uses the same data.
std::vector<i8> random_bytes(std::size_t count, u32 seed) {
    std::vector<i8> bytes(count);
    u32 state = seed;
    for (std::size_t i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        bytes[i] = static_cast<i8>(state >> 24);
    }
    return bytes;
}

// out[r][n] = sum over k of x[r][k] * w[k][n], computed directly.
std::vector<i32> plain_matmul(const std::vector<i8>& x, const std::vector<i8>& w, u32 rows) {
    std::vector<i32> out(static_cast<std::size_t>(rows) * kDim, 0);
    for (u32 r = 0; r < rows; ++r) {
        for (u32 n = 0; n < kDim; ++n) {
            i32 sum = 0;
            for (u32 k = 0; k < kDim; ++k) {
                sum = sum + static_cast<i32>(x[r * kDim + k]) * static_cast<i32>(w[k * kDim + n]);
            }
            out[r * kDim + n] = sum;
        }
    }
    return out;
}

Instr read_host(u32 host_row, u32 ub_row, u32 rows) {
    Instr in;
    in.op       = Op::ReadHostMemory;
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

Instr matmul(u32 ub_row, u32 acc_row, u32 rows, u32 accumulate, u32 new_weights) {
    Instr in;
    in.op          = Op::MatrixMultiply;
    in.ub_row      = ub_row;
    in.acc_row     = acc_row;
    in.rows        = rows;
    in.accumulate  = accumulate;
    in.new_weights = new_weights;
    return in;
}

Instr simple(Op op) {
    Instr in;
    in.op = op;
    return in;
}

bool acc_rows_equal(const Tpu& tpu, u32 acc_row, const std::vector<i32>& expected, u32 rows) {
    for (u32 r = 0; r < rows; ++r) {
        const i32* row = tpu.acc().row(acc_row + r);
        for (u32 n = 0; n < kDim; ++n) {
            if (row[n] != expected[r * kDim + n]) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

TEST(mxu_matches_a_plain_loop) {
    const u32 rows = 8;
    const std::vector<i8> x = random_bytes(rows * kDim, 1);
    const std::vector<i8> w = random_bytes(kDim * kDim, 2);

    Program program;
    program.host.push_back({0, x});
    program.weights.push_back({0, w});
    program.code.push_back(encode(read_host(0, 0, rows)));
    program.code.push_back(encode(read_weights(0)));
    program.code.push_back(encode(matmul(0, 0, rows, 0, 1)));
    program.code.push_back(encode(simple(Op::Halt)));

    Tpu tpu;
    tpu.load(program);
    tpu.run_to_halt();
    CHECK(acc_rows_equal(tpu, 0, plain_matmul(x, w, rows), rows));
}

TEST(mxu_wavefront_timing) {
    // Every input is 1 and every weight is 1, so each finished output is 256.
    UnifiedBuffer ub;
    Accumulators acc;
    Dram wmem("wmem", v1::kWeightMemBytes);
    const std::vector<i8> ones(v1::kTileBytes, 1);
    wmem.write(0, ones.data(), ones.size());
    for (u32 r = 0; r < 3; ++r) {
        for (u32 k = 0; k < kDim; ++k) {
            ub.row(r)[k] = 1;
        }
    }

    WeightFifo fifo(wmem);
    SystolicArray mxu(ub, acc, fifo);
    fifo.push(0);
    while (!mxu.shadow_ready()) {
        fifo.tick();
        mxu.tick();
    }

    const u32 rows = 3;
    mxu.start(0, 0, rows, false, true);
    for (int i = 0; i < 255; ++i) {
        mxu.tick();
    }
    CHECK_EQ(acc.row(0)[0], 0);     // row 0, column 0 reaches the bottom on step 255
    mxu.tick();
    CHECK_EQ(acc.row(0)[0], 256);
    CHECK_EQ(acc.row(0)[1], 0);     // column 1 is one step behind

    int more_steps = 0;
    while (mxu.busy()) {
        mxu.tick();
        more_steps = more_steps + 1;
    }
    CHECK_EQ(256 + more_steps, static_cast<int>(rows + 2 * kDim - 1));   // B + 511 steps in all
    CHECK_EQ(acc.row(2)[255], 256);
}

TEST(mxu_accumulate_adds_into_the_rows) {
    const u32 rows = 2;
    const std::vector<i8> x = random_bytes(rows * kDim, 3);
    const std::vector<i8> w = random_bytes(kDim * kDim, 4);

    Program program;
    program.host.push_back({0, x});
    program.weights.push_back({0, w});
    program.code.push_back(encode(read_host(0, 0, rows)));
    program.code.push_back(encode(read_weights(0)));
    program.code.push_back(encode(matmul(0, 0, rows, 0, 1)));   // overwrite
    program.code.push_back(encode(matmul(0, 0, rows, 1, 0)));   // same weights, add on top
    program.code.push_back(encode(simple(Op::Halt)));

    Tpu tpu;
    tpu.load(program);
    tpu.run_to_halt();

    std::vector<i32> twice = plain_matmul(x, w, rows);
    for (i32& value : twice) {
        value = value * 2;
    }
    CHECK(acc_rows_equal(tpu, 0, twice, rows));
}

TEST(mxu_next_tile_shifts_in_while_the_array_works) {
    const u32 rows = 4;
    const std::vector<i8> x  = random_bytes(rows * kDim, 5);
    const std::vector<i8> w0 = random_bytes(kDim * kDim, 6);
    const std::vector<i8> w1 = random_bytes(kDim * kDim, 7);

    Program program;
    program.host.push_back({0, x});
    program.weights.push_back({0, w0});
    program.weights.push_back({v1::kTileBytes, w1});
    program.code.push_back(encode(read_host(0, 0, rows)));
    program.code.push_back(encode(read_weights(0)));
    program.code.push_back(encode(read_weights(1)));
    program.code.push_back(encode(matmul(0, 0, rows, 0, 1)));    // tile 0 into acc rows 0-3
    program.code.push_back(encode(matmul(0, 16, rows, 0, 1)));   // tile 1 into acc rows 16-19
    program.code.push_back(encode(simple(Op::Halt)));

    Tpu tpu;
    tpu.load(program);
    tpu.next_instructions(4);   // up to and including the first MatrixMultiply
    CHECK(tpu.mxu().busy());
    CHECK_EQ(tpu.mxu().active_tile(), 0);

    tpu.run_to_halt();
    CHECK(acc_rows_equal(tpu, 0, plain_matmul(x, w0, rows), rows));
    CHECK(acc_rows_equal(tpu, 16, plain_matmul(x, w1, rows), rows));
    CHECK_EQ(tpu.mxu().active_tile(), 1);
}

TEST(mxu_waits_for_its_input_rows) {
    const std::vector<i8> x = random_bytes(kDim, 8);

    Program program;
    program.host.push_back({0, x});
    program.code.push_back(encode(read_weights(0)));
    program.code.push_back(encode(simple(Op::Sync)));           // weights are in the shadow plane
    program.code.push_back(encode(read_host(0, 0, 1)));         // one row: 12 cycles
    program.code.push_back(encode(matmul(0, 0, 1, 0, 1)));      // needs that row
    program.code.push_back(encode(simple(Op::Halt)));

    Tpu tpu;
    tpu.load(program);
    tpu.run_to_halt();
    CHECK_EQ(tpu.stats().stalled(Stall::UbNotReady), u64{11});   // the read issues, then 11 more cycles of bytes
}

TEST(mxu_bad_ranges_name_the_pc) {
    Program program;
    program.code.push_back(encode(matmul(0, 4090, 8, 0, 0)));   // accumulator rows 4090-4097; the last is 4095
    program.code.push_back(encode(simple(Op::Halt)));

    Tpu tpu;
    tpu.load(program);
    std::string message;
    try {
        tpu.run_to_halt();
    } catch (const std::runtime_error& e) {
        message = e.what();
    }
    CHECK_EQ(message, std::string("pc 0: MatrixMultiply ub=0x0 acc=0xFFA rows=8 accumulate=0 new_weights=0: "
                                  "acc rows [4090, +8) run past the last row 4095"));
}
