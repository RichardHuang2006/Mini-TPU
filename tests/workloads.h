#pragma once

// Layer lowering: turns a layer spec into a Mini-TPU program, plus the host-side
// tensor layout that program expects. Shared by the test suite and by
// tools/gen_examples.cpp, so a bundled example cannot drift away from what was
// verified.
//
// Two things live here that must be kept apart in your head:
//
//   * the *tiler*, which emits instructions, and
//   * the *golden* functions, which compute the answer with plain nested loops.
//
// The golden functions know nothing about tiles, banks, or the ISA. That is what
// makes them useful: ref.h validates the machine against the oracle, but both run
// the same program, so neither can catch a tiler that lowers a layer wrongly.
// Only an independently computed answer can. Every workload here is therefore
// checked twice -- machine against oracle, and program output against golden.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "config.h"
#include "isa.h"
#include "quant.h"
#include "tpuasm.h"
#include "types.h"

namespace wl {

// A fixed generator, not the standard library's, because a bundled example has to
// be byte-identical on every machine and every toolchain. std::mt19937 would do,
// but the distributions that shape its output are not portable.
struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed) {}
    uint32_t next() {
        s = s * 1664525u + 1013904223u;
        return s;
    }
    i8 byte() { return static_cast<i8>(next() >> 24); }
};

inline std::vector<i8> random_tensor(uint32_t seed, std::size_t n) {
    Rng rng(seed);
    std::vector<i8> v(n);
    for (i8& x : v) x = rng.byte();
    return v;
}

// ---------------------------------------------------------------- packing ---
//
// A MatMul reads a `len x dim` activation tile as `len` contiguous rows of `dim`
// bytes. A row of a wider matrix is not contiguous in a row-major buffer, so
// feeding one tile from a row-major tensor would take one DMA per row.
//
// Instead the host hands over tensors already in tile-major order: every
// `dim x dim` block contiguous, blocks in row-major order. One tile is then one
// DMA, and the ragged edges of a shape that does not divide by `dim` are zero
// padded once, at pack time, instead of being special-cased in the program.
// Padding with zeros is what makes it safe: a padded activation column multiplies
// a padded weight row, so the product contributes nothing to the sum.

inline uint32_t tiles_of(uint32_t n, uint32_t dim) {
    return dim == 0 ? 0 : (n + dim - 1) / dim;
}

// Byte offset of tile (tr, tc) within a packed tensor of `cols` columns.
inline std::size_t tile_off(uint32_t tr, uint32_t tc, uint32_t cols, uint32_t dim) {
    const std::size_t per   = static_cast<std::size_t>(dim) * dim;
    const std::size_t across = tiles_of(cols, dim);
    return (static_cast<std::size_t>(tr) * across + tc) * per;
}

inline std::size_t packed_bytes(uint32_t rows, uint32_t cols, uint32_t dim) {
    return static_cast<std::size_t>(tiles_of(rows, dim)) * tiles_of(cols, dim) * dim * dim;
}

// Row-major `rows x cols` -> tile-major, zero padded to whole tiles.
inline std::vector<i8> pack(const std::vector<i8>& src, uint32_t rows, uint32_t cols,
                            uint32_t dim) {
    assert(src.size() >= static_cast<std::size_t>(rows) * cols);
    std::vector<i8> out(packed_bytes(rows, cols, dim), 0);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            const std::size_t o = tile_off(r / dim, c / dim, cols, dim) +
                                  static_cast<std::size_t>(r % dim) * dim + (c % dim);
            out[o] = src[static_cast<std::size_t>(r) * cols + c];
        }
    }
    return out;
}

// The inverse, dropping the padding.
inline std::vector<i8> unpack(const std::vector<i8>& packed, uint32_t rows, uint32_t cols,
                              uint32_t dim) {
    std::vector<i8> out(static_cast<std::size_t>(rows) * cols, 0);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            const std::size_t o = tile_off(r / dim, c / dim, cols, dim) +
                                  static_cast<std::size_t>(r % dim) * dim + (c % dim);
            assert(o < packed.size());
            out[static_cast<std::size_t>(r) * cols + c] = packed[o];
        }
    }
    return out;
}

// ----------------------------------------------------------------- golden ---
//
// Nested loops, no tiling, no timing. The independent answer every workload is
// held against.

// C[M,N] = A[M,K] * B[K,N], int8 in, int32 out.
inline std::vector<i32> golden_matmul(const std::vector<i8>& a, const std::vector<i8>& b,
                                      uint32_t M, uint32_t K, uint32_t N) {
    std::vector<i32> c(static_cast<std::size_t>(M) * N, 0);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            int64_t sum = 0;
            for (uint32_t k = 0; k < K; ++k) {
                sum += static_cast<int64_t>(a[static_cast<std::size_t>(m) * K + k]) *
                       b[static_cast<std::size_t>(k) * N + n];
            }
            c[static_cast<std::size_t>(m) * N + n] = static_cast<i32>(sum);
        }
    }
    return c;
}

inline i8 golden_actfn(i8 v, ActFn fn) {
    switch (fn) {
        case ActFn::IDENTITY: return v;
        case ActFn::RELU:     return v < 0 ? i8{0} : v;
        case ActFn::RELU6:    return v < 0 ? i8{0} : (v > 6 ? i8{6} : v);
    }
    return v;
}

// Bias, requantize, activation function -- the int32 accumulator to int8 step.
inline std::vector<i8> golden_requantize(const std::vector<i32>& acc, i32 bias, i32 multiplier,
                                         uint32_t shift, ActFn fn) {
    std::vector<i8> out(acc.size(), 0);
    for (std::size_t i = 0; i < acc.size(); ++i) {
        out[i] = golden_actfn(quant::requantize_biased(acc[i], bias, multiplier, shift), fn);
    }
    return out;
}

// ------------------------------------------------------------------ specs ---

// One dense layer: C = act(requantize(A[M,K] * B[K,N])).
struct Layer {
    uint32_t M = 0, K = 0, N = 0;

    i32      bias       = 0;
    i32      multiplier = 1;
    uint32_t shift      = 0;
    ActFn    fn         = ActFn::RELU;
};

inline std::vector<i8> golden_layer(const Layer& l, const std::vector<i8>& a,
                                    const std::vector<i8>& b) {
    return golden_requantize(golden_matmul(a, b, l.M, l.K, l.N), l.bias, l.multiplier, l.shift,
                             l.fn);
}

// ------------------------------------------------------------------- conv ---
//
// A convolution becomes a matmul by im2col: each output pixel's receptive field is
// flattened into one row, so the layer is (out_h*out_w) x (R*S*Cin) times
// (R*S*Cin) x Cout. The array never learns what a convolution is -- the whole of
// the lowering is a host-side rearrangement plus the dense tiler above.
//
// Tensors are NHWC (channels innermost), which is what makes the receptive field
// contiguous in the channel direction and the flattening a copy rather than a
// gather.

struct Conv {
    uint32_t H = 0, W = 0, Cin = 0;      // input, NHWC
    uint32_t R = 0, S = 0;               // kernel
    uint32_t Cout = 0;
    uint32_t stride = 1;
    uint32_t pad    = 0;                 // zero padding, same on all four sides

    i32      bias       = 0;
    i32      multiplier = 1;
    uint32_t shift      = 0;
    ActFn    fn         = ActFn::RELU;

    uint32_t out_h() const { return (H + 2 * pad - R) / stride + 1; }
    uint32_t out_w() const { return (W + 2 * pad - S) / stride + 1; }

    bool valid() const {
        return H && W && Cin && R && S && Cout && stride &&
               H + 2 * pad >= R && W + 2 * pad >= S;
    }

    // The dense layer this convolution turns into.
    Layer as_layer() const {
        Layer l;
        l.M = out_h() * out_w();
        l.K = R * S * Cin;
        l.N = Cout;
        l.bias       = bias;
        l.multiplier = multiplier;
        l.shift      = shift;
        l.fn         = fn;
        return l;
    }
};

// Flatten each receptive field into a row: M x K, row-major.
inline std::vector<i8> im2col(const Conv& c, const std::vector<i8>& in) {
    assert(c.valid());
    assert(in.size() >= static_cast<std::size_t>(c.H) * c.W * c.Cin);

    const uint32_t oh = c.out_h(), ow = c.out_w();
    const uint32_t K  = c.R * c.S * c.Cin;
    std::vector<i8> out(static_cast<std::size_t>(oh) * ow * K, 0);

    for (uint32_t y = 0; y < oh; ++y) {
        for (uint32_t x = 0; x < ow; ++x) {
            const std::size_t row = (static_cast<std::size_t>(y) * ow + x) * K;
            for (uint32_t r = 0; r < c.R; ++r) {
                for (uint32_t s = 0; s < c.S; ++s) {
                    // Signed, because padding puts the window off the near edge.
                    const int64_t iy = static_cast<int64_t>(y) * c.stride - c.pad + r;
                    const int64_t ix = static_cast<int64_t>(x) * c.stride - c.pad + s;
                    const bool inside = iy >= 0 && iy < c.H && ix >= 0 && ix < c.W;
                    for (uint32_t ci = 0; ci < c.Cin; ++ci) {
                        const std::size_t o = row + (static_cast<std::size_t>(r) * c.S + s) *
                                                        c.Cin + ci;
                        // Outside the input is zero, which is exactly what zero
                        // padding means -- no special case reaches the array.
                        out[o] = inside
                                   ? in[((static_cast<std::size_t>(iy) * c.W) +
                                         static_cast<std::size_t>(ix)) * c.Cin + ci]
                                   : i8{0};
                    }
                }
            }
        }
    }
    return out;
}

// Direct convolution, no im2col and no tiles: the independent answer im2col is
// held against. Weights are [(r*S + s)*Cin + ci][cout], matching the matmul's B.
inline std::vector<i32> golden_conv_acc(const Conv& c, const std::vector<i8>& in,
                                        const std::vector<i8>& w) {
    const uint32_t oh = c.out_h(), ow = c.out_w();
    std::vector<i32> out(static_cast<std::size_t>(oh) * ow * c.Cout, 0);

    for (uint32_t y = 0; y < oh; ++y) {
        for (uint32_t x = 0; x < ow; ++x) {
            for (uint32_t co = 0; co < c.Cout; ++co) {
                int64_t sum = 0;
                for (uint32_t r = 0; r < c.R; ++r) {
                    for (uint32_t s = 0; s < c.S; ++s) {
                        const int64_t iy = static_cast<int64_t>(y) * c.stride - c.pad + r;
                        const int64_t ix = static_cast<int64_t>(x) * c.stride - c.pad + s;
                        if (iy < 0 || iy >= c.H || ix < 0 || ix >= c.W) continue;
                        for (uint32_t ci = 0; ci < c.Cin; ++ci) {
                            const int64_t a =
                                in[((static_cast<std::size_t>(iy) * c.W) +
                                    static_cast<std::size_t>(ix)) * c.Cin + ci];
                            const int64_t b =
                                w[((static_cast<std::size_t>(r) * c.S + s) * c.Cin + ci) *
                                      c.Cout + co];
                            sum += a * b;
                        }
                    }
                }
                out[(static_cast<std::size_t>(y) * ow + x) * c.Cout + co] =
                    static_cast<i32>(sum);
            }
        }
    }
    return out;
}

inline std::vector<i8> golden_conv(const Conv& c, const std::vector<i8>& in,
                                   const std::vector<i8>& w) {
    return golden_requantize(golden_conv_acc(c, in, w), c.bias, c.multiplier, c.shift, c.fn);
}

// -------------------------------------------------------------- the tiler ---

// Where a lowered layer expects its tensors to live, and what it produces.
struct Lowering {
    std::vector<RawInst> code;

    HostAddr a_host = 0;        // packed activations, host memory
    HostAddr c_host = 0;        // packed int8 result, host memory
    uint32_t b_ddr  = 0;        // packed weights, weight memory

    std::size_t a_bytes = 0;
    std::size_t b_bytes = 0;
    std::size_t c_bytes = 0;

    std::size_t ub_used = 0;
    std::size_t macs    = 0;    // useful MACs, padding excluded
    std::size_t tiles   = 0;    // weight tiles loaded
};

// Where the lowered program should find and leave its tensors.
struct Placement {
    HostAddr a_host = 0x0000;
    HostAddr c_host = 0x8000;
    uint32_t b_ddr  = 0x0000;
    UbAddr   ub_base = 0;
};

// Scheduling choices, exposed so a test can turn one off and measure what it was
// worth. Both orders compute the same layer; only the cycle count moves.
struct Sched {
    // Defer each output tile's Write_Host past the next tile's MatMuls, instead of
    // emitting it right after the Activate that produced it.
    bool defer_drain = true;

    // Emit the closing Halt. Off when this layer is one stage of a larger program.
    bool terminate = true;
};

// Lower one dense layer onto a `dim x dim` array.
//
// The loop order is m, n, k with the activation tiles for one row-block hoisted
// out of the n loop: a row-block of A is read from the host once and then reused
// by every output column block, which is the whole reason to tile in this order.
// Reloading it per n would multiply the DMA traffic by the number of column
// blocks and turn an array-bound layer into a DMA-bound one.
//
// K tiling accumulates in place: the first k tile overwrites the bank and the
// rest add into it, so a K larger than the array costs bank residency rather
// than a second pass.
inline Lowering lower_layer(const Layer& l, const Config& cfg, const Placement& at = {},
                            const Sched& sched = {}) {
    const uint32_t dim = cfg.dim;
    const uint32_t mt  = tiles_of(l.M, dim);
    const uint32_t nt  = tiles_of(l.N, dim);
    const uint32_t kt  = tiles_of(l.K, dim);

    Lowering out;
    out.a_host  = at.a_host;
    out.c_host  = at.c_host;
    out.b_ddr   = at.b_ddr;
    out.a_bytes = packed_bytes(l.M, l.K, dim);
    out.b_bytes = packed_bytes(l.K, l.N, dim);
    out.c_bytes = packed_bytes(l.M, l.N, dim);
    out.macs    = static_cast<std::size_t>(l.M) * l.K * l.N;

    // One activation row-block (every k tile of it), plus two output tiles.
    //
    // The output is double buffered because a single staging tile would put the
    // next Activate behind the previous Write_Host: the Activate would be
    // overwriting bytes the DMA had not finished reading. Alternating two tiles
    // costs dim*dim bytes and lets the drain overlap the next tile's arithmetic.
    tpuasm::UbAlloc alloc(at.ub_base);
    std::vector<tpuasm::Region> a_ub(kt);
    for (uint32_t k = 0; k < kt; ++k) a_ub[k] = alloc.tile(dim, dim);
    const tpuasm::Region c_ub[2] = {alloc.tile(dim, dim), alloc.tile(dim, dim)};
    out.ub_used = alloc.next() - at.ub_base;

    // The drain of one output tile is deferred past the next tile's arithmetic.
    //
    // Issue is in order, so a stalled instruction blocks everything behind it, and
    // a Write_Host placed immediately after its Activate stalls for the whole
    // activation -- taking the next tile's MatMuls down with it. Deferring the
    // drain by one tile is what lets those MatMuls issue while the activation is
    // still draining, and it is why alternating accumulator banks buys anything at
    // all. On this machine overlap is the tiler's job as much as the hardware's.
    struct Drain {
        bool     live  = false;
        UbAddr   src   = 0;
        HostAddr dst   = 0;
        uint32_t bytes = 0;
    };
    Drain pending;

    tpuasm::Program p;
    auto flush = [&] {
        if (!pending.live) return;
        p.write_host(pending.src, pending.dst, pending.bytes);
        pending.live = false;
    };

    uint32_t stage = 0;   // which output staging tile the next Activate uses
    for (uint32_t m = 0; m < mt; ++m) {
        // Rows this block actually covers; the last one may be short, which the
        // `len` operand expresses directly with no padding needed.
        const uint32_t rows = std::min(dim, l.M - m * dim);

        // The previous row-block's last drain has to be out of the way before the
        // incoming tiles overwrite the buffer it reads from.
        flush();
        for (uint32_t k = 0; k < kt; ++k) {
            p.read_host(static_cast<HostAddr>(at.a_host + tile_off(m, k, l.K, dim)), a_ub[k],
                        rows * dim);
        }

        for (uint32_t n = 0; n < nt; ++n) {
            // Alternate banks so one tile's Activate can drain while the next
            // tile's MatMuls fill a different bank.
            const BankId bank = static_cast<BankId>(n % cfg.acc_banks);

            for (uint32_t k = 0; k < kt; ++k) {
                p.read_weights(static_cast<uint32_t>(at.b_ddr + tile_off(k, n, l.N, dim)));
                p.matmul(a_ub[k], rows, bank, /*accumulate=*/k != 0);
                ++out.tiles;
            }

            // The tile before last is drained here, after this tile's MatMuls have
            // been issued, so they get to overlap its activation.
            flush();

            tpuasm::ActArgs act;
            act.acc        = bank;
            act.dst        = c_ub[stage];
            act.len        = rows;
            act.fn         = l.fn;
            act.bias       = l.bias;
            act.multiplier = l.multiplier;
            act.shift      = l.shift;
            p.activate(act);

            pending.live  = true;
            pending.src   = c_ub[stage];
            pending.dst   = static_cast<HostAddr>(at.c_host + tile_off(m, n, l.N, dim));
            pending.bytes = rows * dim;
            if (!sched.defer_drain) flush();
            stage ^= 1;
        }
    }
    flush();
    if (sched.terminate) p.halt();

    out.code = p.code();
    return out;
}

// -------------------------------------------------------------------- MLP ---
//
// Layers chained end to end in one program. The interesting part is that no
// repacking happens between them: a layer's packed output is already in the
// tile-major layout the next layer's activations want, so stage i+1 reads exactly
// the bytes stage i wrote.
//
// That holds even when a width does not divide the array. The padding columns of a
// packed output carry whatever the activation produced from an all-zero
// accumulator, which is not necessarily zero -- but they only ever meet the padded
// rows of the next layer's packed weights, and pack() fills those with zeros. The
// junk multiplies zero and the sum is unchanged.

struct Mlp {
    std::vector<Layer> layers;

    // Consecutive layers have to agree on the batch size and on the width between
    // them, or the chaining above is meaningless.
    bool chains() const {
        if (layers.empty()) return false;
        for (std::size_t i = 0; i + 1 < layers.size(); ++i) {
            if (layers[i].N != layers[i + 1].K) return false;
            if (layers[i].M != layers[i + 1].M) return false;
        }
        return true;
    }
};

struct MlpLowering {
    std::vector<RawInst>  code;
    std::vector<Lowering> stages;   // one per layer, carrying its addresses

    HostAddr    a_host  = 0;        // packed input
    HostAddr    y_host  = 0;        // packed final output
    std::size_t y_bytes = 0;
    std::size_t macs    = 0;
};

inline MlpLowering lower_mlp(const Mlp& net, const Config& cfg, const Placement& at = {}) {
    assert(net.chains());

    MlpLowering out;
    out.a_host = at.a_host;

    HostAddr feed   = at.a_host;    // where this stage reads its activations
    HostAddr cursor = at.c_host;    // where the next stage's output goes
    uint32_t ddr    = at.b_ddr;

    for (std::size_t i = 0; i < net.layers.size(); ++i) {
        const Layer& l = net.layers[i];

        Placement place;
        place.a_host  = feed;
        place.c_host  = cursor;
        place.b_ddr   = ddr;
        place.ub_base = at.ub_base;   // stages run in sequence, so the UB is reused

        Sched sched;
        sched.terminate = false;

        Lowering stage = lower_layer(l, cfg, place, sched);
        out.code.insert(out.code.end(), stage.code.begin(), stage.code.end());
        out.macs += stage.macs;

        feed   = cursor;
        cursor = static_cast<HostAddr>(cursor + stage.c_bytes);
        ddr    = static_cast<uint32_t>(ddr + stage.b_bytes);

        out.y_host  = stage.c_host;
        out.y_bytes = stage.c_bytes;
        out.stages.push_back(std::move(stage));
    }

    tpuasm::Program tail;
    tail.halt();
    out.code.push_back(tail.code().front());
    return out;
}

// Every layer in turn, on unpacked row-major tensors.
inline std::vector<i8> golden_mlp(const Mlp& net, const std::vector<i8>& a,
                                  const std::vector<std::vector<i8>>& weights) {
    assert(weights.size() == net.layers.size());
    std::vector<i8> x = a;
    for (std::size_t i = 0; i < net.layers.size(); ++i) {
        x = golden_layer(net.layers[i], x, weights[i]);
    }
    return x;
}

// ----------------------------------------------------------------- corpus ---
//
// The shipped workloads. One definition serves the test suite and
// tools/gen_examples.cpp, so a bundled example is always a program the suite has
// verified against golden loops rather than a file someone generated once.

struct Workload {
    std::string name;
    std::string note;

    Config               cfg;
    std::vector<RawInst> code;

    HostAddr        a_host = 0;
    std::vector<i8> a;                 // packed activations
    uint32_t        b_ddr  = 0;
    std::vector<i8> b;                 // packed weights

    HostAddr    y_host  = 0;
    std::size_t y_bytes = 0;

    // Golden output, unpacked and row-major -- what the program has to reproduce.
    std::vector<i8> expect;
    uint32_t        out_rows = 0;
    uint32_t        out_cols = 0;

    std::size_t macs = 0;
};

// A dense layer as a shippable workload.
inline Workload dense_workload(std::string name, std::string note, const Layer& l,
                               const Config& cfg, uint32_t seed) {
    const std::vector<i8> a = random_tensor(seed, static_cast<std::size_t>(l.M) * l.K);
    const std::vector<i8> b = random_tensor(seed + 1, static_cast<std::size_t>(l.K) * l.N);

    const Lowering low = lower_layer(l, cfg);

    Workload w;
    w.name     = std::move(name);
    w.note     = std::move(note);
    w.cfg      = cfg;
    w.code     = low.code;
    w.a_host   = low.a_host;
    w.a        = pack(a, l.M, l.K, cfg.dim);
    w.b_ddr    = low.b_ddr;
    w.b        = pack(b, l.K, l.N, cfg.dim);
    w.y_host   = low.c_host;
    w.y_bytes  = low.c_bytes;
    w.expect   = golden_layer(l, a, b);
    w.out_rows = l.M;
    w.out_cols = l.N;
    w.macs     = low.macs;
    return w;
}

inline Workload conv_workload(std::string name, std::string note, const Conv& c,
                              const Config& cfg, uint32_t seed) {
    const Layer l = c.as_layer();

    const std::vector<i8> in =
        random_tensor(seed, static_cast<std::size_t>(c.H) * c.W * c.Cin);
    const std::vector<i8> b = random_tensor(seed + 1, static_cast<std::size_t>(l.K) * l.N);

    const Lowering low = lower_layer(l, cfg);

    Workload w;
    w.name     = std::move(name);
    w.note     = std::move(note);
    w.cfg      = cfg;
    w.code     = low.code;
    w.a_host   = low.a_host;
    w.a        = pack(im2col(c, in), l.M, l.K, cfg.dim);
    w.b_ddr    = low.b_ddr;
    w.b        = pack(b, l.K, l.N, cfg.dim);
    w.y_host   = low.c_host;
    w.y_bytes  = low.c_bytes;
    w.expect   = golden_conv(c, in, b);
    w.out_rows = l.M;
    w.out_cols = l.N;
    w.macs     = low.macs;
    return w;
}

inline Workload mlp_workload(std::string name, std::string note, const Mlp& net,
                             const Config& cfg, uint32_t seed) {
    const Layer& first = net.layers.front();
    const Layer& last  = net.layers.back();

    const std::vector<i8> a =
        random_tensor(seed, static_cast<std::size_t>(first.M) * first.K);

    std::vector<std::vector<i8>> weights;
    for (std::size_t i = 0; i < net.layers.size(); ++i) {
        const Layer& l = net.layers[i];
        weights.push_back(
            random_tensor(seed + 1 + static_cast<uint32_t>(i),
                          static_cast<std::size_t>(l.K) * l.N));
    }

    const MlpLowering low = lower_mlp(net, cfg);

    Workload w;
    w.name    = std::move(name);
    w.note    = std::move(note);
    w.cfg     = cfg;
    w.code    = low.code;
    w.a_host  = low.a_host;
    w.a       = pack(a, first.M, first.K, cfg.dim);
    w.b_ddr   = low.stages.front().b_ddr;
    w.y_host  = low.y_host;
    w.y_bytes = low.y_bytes;

    // Each stage's weights sit where its lowering expects them, end to end from
    // b_ddr, so one blob carries all three.
    for (std::size_t i = 0; i < net.layers.size(); ++i) {
        const Layer&          l = net.layers[i];
        const std::vector<i8> p = pack(weights[i], l.K, l.N, cfg.dim);
        w.b.insert(w.b.end(), p.begin(), p.end());
    }

    w.expect   = golden_mlp(net, a, weights);
    w.out_rows = last.M;
    w.out_cols = last.N;
    w.macs     = low.macs;
    return w;
}

// A configuration small enough that an example is quick to run yet still tiles in
// every direction.
inline Config example_config() {
    Config c;
    c.dim       = 16;
    c.ub_bytes  = 32 * 1024;
    c.acc_banks = 4;
    return c;
}

// What a workload *is*, separately from the array it was lowered for. The
// configuration sweep needs this: running the same layer on an 8x8 and a 256x256
// array means lowering it twice, not replaying instructions that assume a
// different tile size.
struct Spec {
    enum class Kind : uint8_t { DENSE, CONV, MLP };

    std::string name;
    std::string note;
    Kind        kind = Kind::DENSE;
    uint32_t    seed = 0;

    Layer layer;   // DENSE
    Conv  conv;    // CONV
    Mlp   mlp;     // MLP

    Config cfg;    // the configuration this workload ships with
};

inline Workload build(const Spec& s, const Config& cfg) {
    switch (s.kind) {
        case Spec::Kind::DENSE: return dense_workload(s.name, s.note, s.layer, cfg, s.seed);
        case Spec::Kind::CONV:  return conv_workload(s.name, s.note, s.conv, cfg, s.seed);
        case Spec::Kind::MLP:   return mlp_workload(s.name, s.note, s.mlp, cfg, s.seed);
    }
    return Workload{};
}

inline std::vector<Spec> specs() {
    std::vector<Spec> out;

    // Dense, tiled four ways in each of M, N and K on a 32x32 array.
    {
        Spec s;
        s.name = "matmul_128";
        s.note = "128x128 * 128x128 dense layer, relu";
        s.kind = Spec::Kind::DENSE;
        s.seed = 9001;
        s.layer.M = 128; s.layer.K = 128; s.layer.N = 128;
        s.layer.multiplier = 1; s.layer.shift = 10; s.layer.fn = ActFn::RELU;
        s.cfg.dim = 32;
        out.push_back(s);
    }

    // A ragged dense layer, where none of M, N or K divides the array.
    {
        Spec s;
        s.name = "matmul_ragged";
        s.note = "40x20 * 20x36, none of M/N/K a multiple of dim";
        s.kind = Spec::Kind::DENSE;
        s.seed = 9101;
        s.layer.M = 40; s.layer.K = 20; s.layer.N = 36;
        s.layer.bias = 250; s.layer.multiplier = 3; s.layer.shift = 9;
        s.layer.fn = ActFn::RELU6;
        s.cfg = example_config();
        out.push_back(s);
    }

    // A padded, strided convolution through im2col.
    {
        Spec s;
        s.name = "conv_3x3";
        s.note = "8x8x4 * 3x3x8 conv, stride 1, pad 1";
        s.kind = Spec::Kind::CONV;
        s.seed = 9201;
        s.conv.H = 8; s.conv.W = 8; s.conv.Cin = 4;
        s.conv.R = 3; s.conv.S = 3; s.conv.Cout = 8;
        s.conv.stride = 1; s.conv.pad = 1;
        s.conv.multiplier = 1; s.conv.shift = 8; s.conv.fn = ActFn::RELU;
        s.cfg = example_config();
        out.push_back(s);
    }

    // Three dense layers back to back, chained with no repacking between them.
    {
        auto make = [](uint32_t M, uint32_t K, uint32_t N, uint32_t shift, ActFn fn) {
            Layer l;
            l.M = M; l.K = K; l.N = N;
            l.multiplier = 1; l.shift = shift; l.fn = fn;
            return l;
        };

        Spec s;
        s.name = "mlp_3layer";
        s.note = "64 -> 48 -> 32 -> 16, batch 32";
        s.kind = Spec::Kind::MLP;
        s.seed = 9301;
        s.mlp.layers.push_back(make(32, 64, 48, 9, ActFn::RELU));
        s.mlp.layers.push_back(make(32, 48, 32, 8, ActFn::RELU));
        s.mlp.layers.push_back(make(32, 32, 16, 7, ActFn::IDENTITY));
        s.cfg = example_config();
        out.push_back(s);
    }

    return out;
}

inline std::vector<Workload> corpus() {
    std::vector<Workload> out;
    for (const Spec& s : specs()) out.push_back(build(s, s.cfg));
    return out;
}

}  // namespace wl
