#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "config.h"
#include "datapath.h"

// A stationary weight, one int8 MAC, and registered neighbour pass-throughs.
class Pe {
public:
    // An active plane and a shadow plane to load into.
    static constexpr uint32_t PLANES = 2;

    void set_weight(i8 w, uint32_t plane) { w_[plane] = w; }
    i8   weight(uint32_t plane) const { return w_[plane]; }

    // The registered outputs, as of the start of the current cycle.
    i8  act_out() const { return act_; }
    i32 psum_out() const { return psum_; }

    // Multiply-accumulate into the pending half of the registers.
    void tick(i8 act_in, i32 psum_in, uint32_t plane) {
        act_next_  = act_in;
        psum_next_ = psum_in + static_cast<i32>(act_in) * static_cast<i32>(w_[plane]);
    }

    // Latch what tick() computed.
    void commit() {
        act_  = act_next_;
        psum_ = psum_next_;
    }

    // Drops in-flight data but keeps the weights.
    void clear_pipeline() {
        act_  = act_next_  = 0;
        psum_ = psum_next_ = 0;
    }

private:
    i8  w_[PLANES] = {0, 0};
    i8  act_ = 0,  act_next_ = 0;
    i32 psum_ = 0, psum_next_ = 0;
};

// When each part of a matmul finished, in absolute array cycles.
struct MxuTiming {
    uint64_t start  = 0;   // array clock when the stream began
    uint64_t end    = 0;   // array free again at this cycle (exclusive)
    uint64_t cycles = 0;   // end - start, always len + 2*dim - 1

    // The cycle each output row's last element left the bottom edge.
    std::vector<uint64_t> row_valid;
};

struct MxuStats {
    uint64_t cycles = 0;   // the array clock

    uint64_t matmuls        = 0;
    uint64_t weight_loads   = 0;
    uint64_t plane_switches = 0;

    // Cycles the array stood idle waiting for weights; zero when double buffered.
    uint64_t weight_load_bubble = 0;

    // The 2*dim - 1 busy cycles per matmul with no complete output row emerging.
    uint64_t fill_drain_cycles = 0;

    // MAC slots inside the real tile, versus slots spent on zero padding.
    uint64_t useful_macs        = 0;
    uint64_t partial_tile_waste = 0;
};

class Mxu;

// A read-only observer of the array's internal steps, for tracing.
struct MxuObserver {
    virtual ~MxuObserver() = default;

    // Once per matmul, after any pending plane switch has been resolved.
    virtual void on_matmul_begin(const Mxu& mxu, uint32_t len, uint32_t plane, bool switched) {
        (void)mxu; (void)len; (void)plane; (void)switched;
    }

    // A partial sum for `row` left the bottom edge at column c during step s.
    virtual void on_landing(uint64_t s, uint32_t c, uint32_t row, i32 v, i32 dst_after) {
        (void)s; (void)c; (void)row; (void)v; (void)dst_after;
    }

    // After step s committed; `left` holds the dim values fed into the left edge.
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

    // Read_Weights targets the shadow plane when double buffering is on.
    uint32_t load_plane() const { return cfg_.double_buffer ? 1u - active_ : active_; }

    bool switch_pending() const { return pending_; }

    // Cycles the array idles for a weight load: free into the shadow plane.
    uint32_t load_bubble() const { return cfg_.double_buffer ? 0u : cfg_.dim; }

    i8 weight_at(uint32_t k, uint32_t c, uint32_t plane) const {
        return pe(k, c).weight(plane);
    }

    // The real shape of the tile resident in a plane, excluding zero padding.
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

    // Load a zero-padded tile without touching the clock, for the sequencer.
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

    // Stream `acts` (len x dim) through the tile, de-skewing results into `out`.
    MxuTiming matmul(const ConstI8View& acts, const I32View& out, bool accumulate,
                     MxuObserver* obs = nullptr);

    void clear_pipeline() {
        for (Pe& p : pe_) p.clear_pipeline();
    }

    // Idle the array forward to the machine's clock, counting the lost cycles.
    void idle_until(uint64_t when) {
        if (when > cycle_) {
            cycle_        = when;
            stats_.cycles = cycle_;
        }
    }

    // Fraction of MAC slots that did useful work, fill and drain included.
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

    // A shadow load becomes active here, at no cost.
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
        // Every PE reads its neighbours as of the start of the cycle.
        for (uint32_t k = 0; k < dim; ++k) {
            // Activation skew: row k is fed input row s - k.
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

        // De-skew: column c now presents input row s + 1 - dim - c, and rows
        // outside [0, len) are fill, drain or a leftover, so they are dropped.
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
