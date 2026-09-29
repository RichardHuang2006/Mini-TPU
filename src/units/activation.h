/// Activation unit: turns int32 accumulator rows into int8 Unified Buffer rows, one 256-value row per cycle.

#pragma once

#include "isa/isa.h"
#include "mem/accumulators.h"
#include "mem/unified_buffer.h"

// One value: x = value / 2^shift; identity and relu give round(x), sigmoid and tanh give round(f(x) * 128); saturated to int8.
i8 activate(i32 value, u32 shift, ActivationFunction function);

class ActivationUnit {
public:
    ActivationUnit(const Accumulators& acc, UnifiedBuffer& ub);

    // Starts activating `rows` accumulator rows into UB rows; throws if either range runs past its memory or an Activate is running.
    void start(u32 acc_row, u32 ub_row, u32 rows, u32 shift, ActivationFunction function);

    // One cycle: all 256 values of the next accumulator row go through the function into their UB row.
    void tick();

    // Drops any running Activate, as after a machine reset.
    void reset();

    bool busy() const;

    // True while the running Activate covers any of these rows.
    bool reads_acc_rows(u32 first, u32 count) const;
    bool writes_ub_rows(u32 first, u32 count) const;

    // For the terminal: what is running and how far it has got.
    u32 acc_row() const;
    u32 ub_row() const;
    u32 rows_done() const;
    u32 rows_total() const;
    u32 shift() const;
    ActivationFunction function() const;

private:
    const Accumulators& acc_;
    UnifiedBuffer&      ub_;

    u32 acc_row_    = 0;
    u32 ub_row_     = 0;
    u32 rows_total_ = 0;
    u32 rows_done_  = 0;
    u32 shift_      = 0;
    ActivationFunction function_ = ActivationFunction::Identity;
};
