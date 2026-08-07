#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "config.h"
#include "pe.h"
#include "tensor.h"
#include "types.h"

// The matrix unit: a dim x dim grid of weight-stationary processing elements,
// clocked one cycle at a time.
//
// Layout, matching the contract fixed in ref.h:
//
//   PE[k][c] holds W[k][c]. Activations enter the left edge and travel right;
//   partial sums travel down and leave the bottom edge. Column c therefore
//   computes sum_k A[r][k] * W[k][c], turning input row r into output row r.
//
// A note on names: DESIGN calls each input vector an "activation column",
// because it enters the array as a skewed vertical slice. In the row-major
// layout ref.h fixes, that vector is a *row* of the len x dim activation block.
// This file follows ref.h, so `len` counts rows and the fill/drain formula
// N + 2*dim - 1 is written with N = len.

// When each part of a matmul finished, in absolute array cycles.
struct MxuTiming {
    uint64_t start  = 0;   // array clock when the stream began
    uint64_t end    = 0;   // array free again at this cycle (exclusive)
    uint64_t cycles = 0;   // end - start, always len + 2*dim - 1

    // The cycle each output row became complete, meaning its last element left
    // the bottom edge. Results emerge skewed by column, so a row is not usable
    // the moment its first element appears.
    std::vector<uint64_t> row_valid;
};

struct MxuStats {
    uint64_t cycles = 0;   // the array clock

    uint64_t matmuls        = 0;
    uint64_t weight_loads   = 0;
    uint64_t plane_switches = 0;

    // Cycles the array stood idle waiting for weights. The point of double
    // buffering is to keep this at zero.
    uint64_t weight_load_bubble = 0;

    // The 2*dim - 1 cycles per matmul during which the array is busy but no
    // complete output row is emerging.
    uint64_t fill_drain_cycles = 0;

    // MAC slots inside the real tile, versus slots spent multiplying by the zero
    // padding of an undersized one. Splitting them is what makes a badly tiled
    // workload visible instead of merely slow.
    uint64_t useful_macs        = 0;
    uint64_t partial_tile_waste = 0;
};

class Mxu {
public:
    explicit Mxu(const Config& cfg)
        : cfg_(cfg), pe_(static_cast<std::size_t>(cfg.dim) * cfg.dim) {
        assert(cfg.dim > 0);
    }

    uint32_t dim() const { return cfg_.dim; }
    uint64_t cycle() const { return cycle_; }
    const MxuStats& stats() const { return stats_; }

    uint32_t active_plane() const { return active_; }

    // Read_Weights targets the shadow plane when double buffering is on, so a
    // load cannot disturb the tile that is currently multiplying.
    uint32_t load_plane() const { return cfg_.double_buffer ? 1u - active_ : active_; }

    bool switch_pending() const { return pending_; }

    // Cycles the array must stand idle for a weight load. With double buffering
    // the weights shift into the shadow plane while the active plane keeps
    // multiplying, so the load is free; without it the array waits dim cycles
    // for the weights to arrive. Whether the *first* load of a program is truly
    // free is a question about what it could overlap with, which only the
    // sequencer can answer; the array applies one uniform rule.
    uint32_t load_bubble() const { return cfg_.double_buffer ? 0u : cfg_.dim; }

    i8 weight_at(uint32_t k, uint32_t c, uint32_t plane) const {
        return pe(k, c).weight(plane);
    }

    // The real shape of the tile resident in a plane, which is what separates
    // useful MACs from padding.
    uint32_t plane_rows(uint32_t plane) const { return rows_[plane]; }
    uint32_t plane_cols(uint32_t plane) const { return cols_[plane]; }

    // Load a weight tile. A tile smaller than the array is zero-padded, which is
    // what makes an undersized matmul produce the right answer rather than a
    // plausible one: the padded MACs contribute exactly zero.
    void load_weights(const ConstI8View& w) {
        assert(w.rows() <= cfg_.dim && w.cols() <= cfg_.dim);
        const uint32_t plane = load_plane();

        for (uint32_t k = 0; k < cfg_.dim; ++k) {
            for (uint32_t c = 0; c < cfg_.dim; ++c) {
                const bool real = k < w.rows() && c < w.cols();
                pe(k, c).set_weight(real ? w.at(k, c) : i8{0}, plane);
            }
        }
        rows_[plane] = w.rows();
        cols_[plane] = w.cols();

        // A load into the shadow plane only takes effect at the next matmul.
        pending_ = (plane != active_);

        const uint32_t bubble = load_bubble();
        cycle_ += bubble;
        stats_.weight_load_bubble += bubble;
        stats_.cycles = cycle_;
        ++stats_.weight_loads;
    }

    // Stream `acts` (len x dim, A[r][k]) through the resident tile and de-skew
    // the results into `out` (at least len rows x dim columns of int32).
    MxuTiming matmul(const ConstI8View& acts, const I32View& out, bool accumulate);

    void clear_pipeline() {
        for (Pe& p : pe_) p.clear_pipeline();
    }

    // Idle the array forward, for when the machine's clock has moved on without
    // it -- waiting on a DMA, say, or on a weight tile that has not arrived.
    // Those cycles are counted, so time the array spent with nothing to do shows
    // up as lost utilization instead of vanishing. Never moves time backwards.
    void idle_until(uint64_t when) {
        if (when > cycle_) {
            cycle_        = when;
            stats_.cycles = cycle_;
        }
    }

    // Fraction of MAC slots that did useful work, counting every cycle the array
    // was occupied, fill and drain included. This is the number that makes the
    // case for streaming many rows through one weight load.
    double utilization() const {
        const double slots = static_cast<double>(cfg_.dim) * cfg_.dim * stats_.cycles;
        return slots == 0.0 ? 0.0 : static_cast<double>(stats_.useful_macs) / slots;
    }

private:
    Pe& pe(uint32_t k, uint32_t c) {
        return pe_[static_cast<std::size_t>(k) * cfg_.dim + c];
    }
    const Pe& pe(uint32_t k, uint32_t c) const {
        return pe_[static_cast<std::size_t>(k) * cfg_.dim + c];
    }

    Config          cfg_;
    std::vector<Pe> pe_;

    uint32_t active_  = 0;
    bool     pending_ = false;

    uint32_t rows_[Pe::PLANES] = {0, 0};
    uint32_t cols_[Pe::PLANES] = {0, 0};

    uint64_t cycle_ = 0;
    MxuStats stats_;
};

inline MxuTiming Mxu::matmul(const ConstI8View& acts, const I32View& out, bool accumulate) {
    const uint32_t dim = cfg_.dim;
    const uint32_t len = acts.rows();
    assert(acts.cols() == dim);
    assert(out.cols() == dim && out.rows() >= len);

    // A shadow load becomes active here, instantaneously. That is the entire
    // benefit of the second plane: the switch costs nothing, so the dim cycles
    // of weight shifting happened while the previous matmul was still running.
    if (pending_) {
        active_  = 1u - active_;
        pending_ = false;
        ++stats_.plane_switches;
    }

    const uint32_t plane = active_;

    MxuTiming t;
    t.start  = cycle_;
    t.cycles = static_cast<uint64_t>(len) + 2ull * dim - 1ull;
    t.end    = t.start + t.cycles;
    t.row_valid.assign(len, 0);

    for (uint64_t s = 0; s < t.cycles; ++s) {
        // Phase 1: every PE reads its neighbours as of the start of the cycle.
        for (uint32_t k = 0; k < dim; ++k) {
            // Row k is fed input row s - k. That skew is what makes all dim
            // products contributing to one output element meet the descending
            // partial sum in the right PE on the right cycle.
            i8 left = 0;
            if (s >= k) {
                const uint64_t r = s - k;
                if (r < len) left = acts.at(static_cast<uint32_t>(r), k);
            }
            for (uint32_t c = 0; c < dim; ++c) {
                const i8  act_in  = (c == 0) ? left : pe(k, c - 1).act_out();
                const i32 psum_in = (k == 0) ? 0 : pe(k - 1, c).psum_out();
                pe(k, c).tick(act_in, psum_in, plane);
            }
        }

        // Phase 2: latch the whole array at once.
        for (Pe& p : pe_) p.commit();

        // De-skew the bottom edge. An activation reaching PE[k][c] has taken k
        // hops down and c hops right, so what column c presents after this cycle
        // belongs to input row s + 1 - dim - c. Anything outside [0, len) is
        // fill, drain, or a leftover from the previous matmul still draining
        // out, and is dropped -- which is also why matmul does not need to clear
        // the pipeline first.
        for (uint32_t c = 0; c < dim; ++c) {
            const int64_t r = static_cast<int64_t>(s) + 1 - dim - c;
            if (r < 0 || r >= static_cast<int64_t>(len)) continue;

            const uint32_t row = static_cast<uint32_t>(r);
            const i32      v   = pe(dim - 1, c).psum_out();
            i32&           dst = out.at(row, c);
            dst = accumulate ? static_cast<i32>(dst + v) : v;

            // The row is complete once its last column has landed.
            if (c == dim - 1) t.row_valid[row] = t.start + s + 1;
        }
    }

    const uint64_t slots = static_cast<uint64_t>(len) * dim * dim;
    const uint64_t real  = static_cast<uint64_t>(len) * rows_[plane] * cols_[plane];
    stats_.useful_macs        += real;
    stats_.partial_tile_waste += slots - real;
    stats_.fill_drain_cycles  += 2ull * dim - 1ull;
    ++stats_.matmuls;

    cycle_        = t.end;
    stats_.cycles = cycle_;
    return t;
}
