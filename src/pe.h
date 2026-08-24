#pragma once

#include <cstdint>

#include "types.h"

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
