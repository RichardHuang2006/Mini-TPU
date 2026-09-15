#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "isa.h"
#include "systolic_array.h"
#include "tpu.h"
#include "transfer.h"

// Structured tracing of a run: an observer the sequencer calls at every point
// where state changes or an issue attempt is decided. The sink only reads; it
// never touches the machine, and every hook site in tpu.cpp is guarded on the
// sink being non-null, so a run without a sink is the same run.
//
// The hooks follow the order of one iteration of Tpu::run, which is one cycle:
//
//   on_retire / on_weight_load    retire_completed(): staged outputs commit
//   on_prefetch                    prefetch_weights(): a tile enters the FIFO
//   on_stall | on_issue | on_trap | on_halt   issue_step(): one outcome per cycle
//   on_sync                        with an issue of Sync
//   MxuObserver hooks              inside execute() of a MatMul, one per step
//   on_cycle_end                   after the busy/idle accounting
//
// tools/tracegen.cpp implements this to write the trace container the
// visualizer under viz/ reads.

// Everything the per-cycle record needs, sampled after the cycle's accounting.
struct CycleInfo {
    std::size_t pc         = 0;      // the instruction the sequencer attempted
    bool        issued     = false;
    bool        trapped    = false;
    bool        halted     = false;
    StallReason reason     = StallReason::NONE;
    Unit        blocker    = Unit::COUNT;   // unit whose reservation blocked issue
    std::size_t blocker_pc = 0;

    // 0 = array busy, else the RunProfile idle bucket charged: 1 weights,
    // 2 bank, 3 accum, 4 dma, 5 act, 6 other.
    uint8_t idle_bucket = 0;

    uint8_t     units_active = 0;    // bitmask by Unit, after issue
    std::size_t fifo_occ     = 0;
    std::size_t fifo_ready   = 0;
    uint32_t    ub_readers   = 0;
    uint32_t    ub_writers   = 0;
    uint32_t    plane        = 0;    // active weight plane after this cycle
    bool        pending      = false;
};

struct TraceSink : MxuObserver {
    ~TraceSink() override = default;

    virtual void on_run_begin(const Tpu& tpu) { (void)tpu; }
    virtual void on_run_end(const Tpu& tpu, const TpuResult& r) { (void)tpu; (void)r; }

    virtual void on_cycle_end(const Tpu& tpu, uint64_t cycle, const CycleInfo& info) {
        (void)tpu; (void)cycle; (void)info;
    }

    // Before the staged outputs of `f` are written; `f.pending` holds them.
    virtual void on_retire(const Tpu& tpu, uint64_t cycle, Unit unit, const InFlight& f) {
        (void)tpu; (void)cycle; (void)unit; (void)f;
    }

    // A Read_Weights retired: `tile` was popped and loaded into `plane`.
    virtual void on_weight_load(const Tpu& tpu, uint64_t cycle, std::size_t pc,
                                const WeightTile& tile, uint32_t plane, bool pending_switch) {
        (void)tpu; (void)cycle; (void)pc; (void)tile; (void)plane; (void)pending_switch;
    }

    // The prefetcher pushed the tile for the Read_Weights at `for_pc`.
    virtual void on_prefetch(const Tpu& tpu, uint64_t cycle, std::size_t for_pc,
                             const WeightTile& tile, uint64_t ready_cycle,
                             std::size_t occupancy_after) {
        (void)tpu; (void)cycle; (void)for_pc; (void)tile; (void)ready_cycle; (void)occupancy_after;
    }

    virtual void on_stall(const Tpu& tpu, uint64_t cycle, std::size_t pc, const Decoded& d,
                          StallReason reason, Unit blocker, std::size_t blocker_pc) {
        (void)tpu; (void)cycle; (void)pc; (void)d; (void)reason; (void)blocker; (void)blocker_pc;
    }

    // After execute(): inputs were read (they are still readable through `tpu`,
    // since nothing commits until retire) and outputs are staged in `pw`.
    virtual void on_issue(const Tpu& tpu, uint64_t cycle, std::size_t pc, const Decoded& d,
                          Unit unit, const Reservation& res, uint64_t duration,
                          const PendingWrite& pw) {
        (void)tpu; (void)cycle; (void)pc; (void)d; (void)unit; (void)res; (void)duration; (void)pw;
    }

    virtual void on_sync(const Tpu& tpu, uint64_t cycle, std::size_t pc, std::size_t snapshot_index) {
        (void)tpu; (void)cycle; (void)pc; (void)snapshot_index;
    }
    virtual void on_halt(const Tpu& tpu, uint64_t cycle, std::size_t pc, uint32_t code) {
        (void)tpu; (void)cycle; (void)pc; (void)code;
    }
    virtual void on_trap(const Tpu& tpu, uint64_t cycle, std::size_t pc, const std::string& reason) {
        (void)tpu; (void)cycle; (void)pc; (void)reason;
    }
};
