#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "config.h"
#include "datapath.h"

// The weight-stationary systolic array: one processing element (Pe), and the
// matrix unit (Mxu) that clocks a dim x dim grid of them cycle by cycle.
//
// To trace one value through the array:
//
//   * an activation enters the left edge of its row, moves one PE to the right
//     per cycle, and falls off the right edge;
//   * a weight stays put in its PE for the whole matmul (that is what
//     "weight-stationary" means), having been shifted in by a weight load;
//   * a partial sum starts as zero at the top of a column, gains one product
//     at every PE it passes, moves one PE down per cycle, and leaves the
//     bottom edge as a finished int32 dot product.

// ============================================================================
// Processing element
// ============================================================================

// One processing element: a stationary weight, one int8 x int8 -> int32 MAC, and
// registered pass-throughs for the activation (left to right) and the partial
// sum (top to bottom).
//
//         act_in ->--+-------> act_out (next cycle)
//                    |
//                  [ w ]  <- stationary weight
//                    |
//    psum_in v-------+------v psum_out = psum_in + act_in * w
//
// Both outputs are registered, so a value handed to tick() reaches the
// neighbours only on the following cycle. That one-cycle-per-hop delay is what
// the activation skew in Mxu::matmul is computed against; combinational
// forwarding would let one activation sweep the whole array in a single cycle.
//
// Updating is split into tick() and commit() so every PE reads its neighbours'
// registers as of the start of the cycle. Fusing the two would leak PE visit
// order into the results.
class Pe {
public:
    // An active plane and a shadow plane, so a tile can be loaded while the
    // other one is still multiplying.
    static constexpr uint32_t PLANES = 2;

    void set_weight(i8 w, uint32_t plane) { w_[plane] = w; }
    i8   weight(uint32_t plane) const { return w_[plane]; }

    // The registered outputs, as of the start of the current cycle.
    i8  act_out() const { return act_; }
    i32 psum_out() const { return psum_; }

    // Multiply-accumulate into the pending half of the registers.
    //
    // int32 accumulation matches the width the hardware carries down a column.
    // One column of a dim-deep array sums dim products of magnitude at most
    // 128*128, so overflow would need a dim in the millions; UBSan reports it
    // rather than letting the result wrap silently.
    void tick(i8 act_in, i32 psum_in, uint32_t plane) {
        act_next_  = act_in;
        psum_next_ = psum_in + static_cast<i32>(act_in) * static_cast<i32>(w_[plane]);
    }

    // Latch what tick() computed.
    void commit() {
        act_  = act_next_;
        psum_ = psum_next_;
    }

    // Drops in-flight data but keeps the weights: weights are loaded by their
    // own instruction and outlive any single matmul.
    void clear_pipeline() {
        act_  = act_next_  = 0;
        psum_ = psum_next_ = 0;
    }

private:
    i8  w_[PLANES] = {0, 0};
    i8  act_ = 0,  act_next_ = 0;
    i32 psum_ = 0, psum_next_ = 0;
};

// ============================================================================
// Matrix unit
// ============================================================================

// The matrix unit: a dim x dim grid of weight-stationary processing elements,
// clocked one cycle at a time.
//
// Layout, matching the contract fixed in tests/ref.h:
//
//   PE[k][c] holds W[k][c]. Activations enter the left edge and travel right;
//   partial sums travel down and leave the bottom edge. Column c therefore
//   computes sum_k A[r][k] * W[k][c], turning input row r into output row r.
//
// On naming: an input vector is sometimes called an "activation column" because
// it enters the array as a skewed vertical slice. In the row-major layout ref.h
// fixes, that vector is a row of the len x dim activation block. This file
// follows ref.h, so `len` counts rows and the fill/drain formula N + 2*dim - 1
// is written with N = len.
//
// Why a matmul over `len` rows occupies the array for exactly
//
//     len + 2*dim - 1
//
// cycles under this schedule: one input row is admitted per cycle, and row r's
// element for PE row k enters k cycles after the row itself starts -- that is
// the activation skew, which makes all dim products of one output element meet
// the descending partial sum in the right PE on the right cycle. Row r's
// partial sum for column c then takes dim hops to descend the column and c
// hops of skew to reach that column, so its last element (column dim-1)
// leaves the bottom edge at cycle r + (dim - 1) + dim = r + 2*dim - 1,
// counting from the start of the stream. The final row is r = len - 1, whose
// last element lands after len - 1 + 2*dim - 1 complete cycles, so the array
// is occupied for (len - 1 + 2*dim - 1) + 1 = len + 2*dim - 1 cycles: `len`
// cycles of streaming plus 2*dim - 1 cycles of fill and drain overhead.

// When each part of a matmul finished, in absolute array cycles.
struct MxuTiming {
    uint64_t start  = 0;   // array clock when the stream began
    uint64_t end    = 0;   // array free again at this cycle (exclusive)
    uint64_t cycles = 0;   // end - start, always len + 2*dim - 1

    // The cycle each output row became complete, i.e. its last element left the
    // bottom edge. Results emerge skewed by column, so a row is not usable when
    // its first element appears.
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

    // MAC slots inside the real tile, versus slots spent multiplying the zero
    // padding of an undersized one. Splitting them makes a badly tiled workload
    // visible rather than merely slow.
    uint64_t useful_macs        = 0;
    uint64_t partial_tile_waste = 0;
};

class Mxu;

// An observer of the array's internal steps, for tracing. Every hook only
// reads registers; the defaults do nothing, and a null observer costs one
// branch per step, so a run without one is unchanged.
struct MxuObserver {
    virtual ~MxuObserver() = default;

    // Once per matmul, after any pending plane switch has been resolved.
    virtual void on_matmul_begin(const Mxu& mxu, uint32_t len, uint32_t plane, bool switched) {
        (void)mxu; (void)len; (void)plane; (void)switched;
    }

    // A finished partial sum left the bottom edge at column c during step s,
    // for output row `row`. `v` is the value the PE presented, `dst_after` the
    // staged output element after it was written (overwrite or accumulate).
    virtual void on_landing(uint64_t s, uint32_t c, uint32_t row, i32 v, i32 dst_after) {
        (void)s; (void)c; (void)row; (void)v; (void)dst_after;
    }

    // After step s committed and its landings were reported. `left` holds the
    // dim values fed into the left edge during this step, one per PE row.
    virtual void on_step(const Mxu& mxu, uint64_t s, const i8* left) {
        (void)mxu; (void)s; (void)left;
    }
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

    // Registered outputs of one PE as of the start of the current step.
    i8  pe_act(uint32_t k, uint32_t c) const { return pe(k, c).act_out(); }
    i32 pe_psum(uint32_t k, uint32_t c) const { return pe(k, c).psum_out(); }

    uint32_t active_plane() const { return active_; }

    // Read_Weights targets the shadow plane when double buffering is on, so a
    // load cannot disturb the tile that is currently multiplying.
    uint32_t load_plane() const { return cfg_.double_buffer ? 1u - active_ : active_; }

    bool switch_pending() const { return pending_; }

    // Cycles the array must stand idle for a weight load. With double buffering
    // the weights shift into the shadow plane while the active plane keeps
    // multiplying, so the load is free; without it the array waits dim cycles.
    // Whether a program's first load is genuinely free depends on what it could
    // overlap with, which only the sequencer knows; the array applies one
    // uniform rule.
    uint32_t load_bubble() const { return cfg_.double_buffer ? 0u : cfg_.dim; }

    i8 weight_at(uint32_t k, uint32_t c, uint32_t plane) const {
        return pe(k, c).weight(plane);
    }

    // The real shape of the tile resident in a plane, which separates useful
    // MACs from padding.
    uint32_t plane_rows(uint32_t plane) const { return rows_[plane]; }
    uint32_t plane_cols(uint32_t plane) const { return cols_[plane]; }

    // Load a weight tile and charge the array the load bubble.
    void load_weights(const ConstI8View& w) {
        load_weights_untimed(w);

        const uint32_t bubble = load_bubble();
        cycle_ += bubble;
        stats_.weight_load_bubble += bubble;
        stats_.cycles = cycle_;
    }

    // Load a weight tile without touching the clock.
    //
    // The sequencer uses this one: a Read_Weights pays both the DDR latency and
    // the load, so it folds both into the instruction's duration and lets the
    // array charge neither. Charging the array's own bubble would count it twice.
    //
    // A tile smaller than the array is zero-padded, so the padded MACs contribute
    // exactly zero and an undersized matmul still produces the right answer.
    void load_weights_untimed(const ConstI8View& w) {
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
        ++stats_.weight_loads;
    }

    // Stream `acts` (len x dim, A[r][k]) through the resident tile and de-skew
    // the results into `out` (at least len rows x dim columns of int32). An
    // observer, if given, sees every step.
    MxuTiming matmul(const ConstI8View& acts, const I32View& out, bool accumulate,
                     MxuObserver* obs = nullptr);

    void clear_pipeline() {
        for (Pe& p : pe_) p.clear_pipeline();
    }

    // Idle the array forward when the machine's clock has moved on without it,
    // e.g. while waiting on a DMA or on a weight tile that has not arrived. Those
    // cycles are counted, so idle array time shows up as lost utilization rather
    // than vanishing. Never moves time backwards.
    void idle_until(uint64_t when) {
        if (when > cycle_) {
            cycle_        = when;
            stats_.cycles = cycle_;
        }
    }

    // Fraction of MAC slots that did useful work, over every cycle the array was
    // occupied, fill and drain included.
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

inline MxuTiming Mxu::matmul(const ConstI8View& acts, const I32View& out, bool accumulate,
                             MxuObserver* obs) {
    const uint32_t dim = cfg_.dim;
    const uint32_t len = acts.rows();
    assert(acts.cols() == dim);
    assert(out.cols() == dim && out.rows() >= len);

    // A shadow load becomes active here, instantaneously. That is the benefit of
    // the second plane: the switch costs nothing, because the dim cycles of
    // weight shifting happened while the previous matmul was still running.
    const bool switched = pending_;
    if (pending_) {
        active_  = 1u - active_;
        pending_ = false;
        ++stats_.plane_switches;
    }

    const uint32_t plane = active_;
    if (obs) obs->on_matmul_begin(*this, len, plane, switched);

    MxuTiming t;
    t.start  = cycle_;
    t.cycles = static_cast<uint64_t>(len) + 2ull * dim - 1ull;
    t.end    = t.start + t.cycles;
    t.row_valid.assign(len, 0);

    std::vector<i8> left_edge;
    if (obs) left_edge.assign(dim, 0);

    for (uint64_t s = 0; s < t.cycles; ++s) {
        // First, every PE reads its neighbours as of the start of the cycle.
        for (uint32_t k = 0; k < dim; ++k) {
            // Row k is fed input row s - k. That skew makes all dim products
            // contributing to one output element meet the descending partial sum
            // in the right PE on the right cycle.
            i8 left = 0;
            if (s >= k) {
                const uint64_t r = s - k;
                if (r < len) left = acts.at(static_cast<uint32_t>(r), k);
            }
            if (obs) left_edge[k] = left;
            for (uint32_t c = 0; c < dim; ++c) {
                const i8  act_in  = (c == 0) ? left : pe(k, c - 1).act_out();
                const i32 psum_in = (k == 0) ? 0 : pe(k - 1, c).psum_out();
                pe(k, c).tick(act_in, psum_in, plane);
            }
        }

        // Then latch the whole array at once.
        for (Pe& p : pe_) p.commit();

        // De-skew the bottom edge. An activation reaching PE[k][c] has taken k
        // hops down and c hops right, so what column c presents after this cycle
        // belongs to input row s + 1 - dim - c. Anything outside [0, len) is fill,
        // drain, or a leftover from the previous matmul still draining, and is
        // dropped; that is why matmul need not clear the pipeline first.
        for (uint32_t c = 0; c < dim; ++c) {
            const int64_t r = static_cast<int64_t>(s) + 1 - dim - c;
            if (r < 0 || r >= static_cast<int64_t>(len)) continue;

            const uint32_t row = static_cast<uint32_t>(r);
            const i32      v   = pe(dim - 1, c).psum_out();
            i32&           dst = out.at(row, c);
            dst = accumulate ? static_cast<i32>(dst + v) : v;
            if (obs) obs->on_landing(s, c, row, v, dst);

            // The row is complete once its last column has landed.
            if (c == dim - 1) t.row_valid[row] = t.start + s + 1;
        }

        if (obs) obs->on_step(*this, s, left_edge.data());
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
