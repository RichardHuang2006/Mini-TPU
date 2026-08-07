#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cstdio>
#include <string>

#include "accumulators.h"
#include "config.h"
#include "decoder.h"
#include "dma.h"
#include "isa.h"
#include "mxu.h"
#include "types.h"
#include "unified_buffer.h"
#include "weight_fifo.h"

// ---------------------------------------------------------------------------
// The scoreboard's view of an instruction: which Unified Buffer bytes it reads
// and writes, which accumulator bank, and whether it touches the resident
// weights. Everything the interlock needs, computed once at issue.
// ---------------------------------------------------------------------------

struct UbRegion {
    UbAddr lo = 0;
    UbAddr hi = 0;   // exclusive

    bool empty() const { return hi <= lo; }
    bool overlaps(const UbRegion& o) const {
        return !empty() && !o.empty() && lo < o.hi && o.lo < hi;
    }
};

struct Reservation {
    UbRegion ub_read;
    UbRegion ub_write;
    BankId   acc_read      = INVALID_BANK;
    BankId   acc_write     = INVALID_BANK;
    bool     reads_weights = false;    // MatMul consumes the resident tile
    bool     writes_weights = false;   // Read_Weights replaces it
};

// One instruction can be in flight per unit; that is the whole of the
// concurrency model. SEQ covers the instructions with no unit of their own.
enum class Unit : uint8_t { DMA, WEIGHT, MXU, ACT, SEQ, COUNT };

// An instruction reads its inputs when it issues and commits its outputs when it
// retires. The gap is what makes the scoreboard load-bearing for data and not
// only for the schedule: a consumer allowed to issue too early reads the state
// its producer has not replaced yet, so a missing interlock is a wrong answer
// rather than merely a wrong cycle count.
struct PendingWrite {
    UbAddr          ub_at = 0;
    std::vector<i8> ub_data;

    HostAddr             host_at = 0;
    std::vector<uint8_t> host_data;

    BankId           bank = INVALID_BANK;
    uint32_t         acc_rows = 0;
    std::vector<i32> acc_data;
};

struct InFlight {
    bool         active      = false;
    Op           op          = Op::NOP;
    Reservation  res;
    PendingWrite pending;
    uint64_t     issue_cycle = 0;
    uint64_t     done_cycle  = 0;
};

// Why issue could not proceed, per cycle. This is the breakdown Phase 8 reports:
// a slow program is slow for one of these reasons and the counters say which.
struct StallStats {
    uint64_t ub_raw       = 0;   // reader waiting on the writer of its region
    uint64_t ub_war       = 0;   // writer waiting on a reader of its region
    uint64_t ub_waw       = 0;
    uint64_t accum_hazard = 0;   // accumulator bank still being produced
    uint64_t weight_stall = 0;   // waiting on the resident weight tile

    uint64_t unit_busy         = 0;   // structural: the unit already has work
    uint64_t weight_fifo_full  = 0;
    uint64_t weight_fifo_empty = 0;
    uint64_t drain             = 0;   // Sync, Halt, or a trap waiting for quiet

    // Cycles the array stood idle for a weight load because there was no shadow
    // plane to hide it in.
    uint64_t weight_load_bubble = 0;

    uint64_t total() const {
        return ub_raw + ub_war + ub_waw + accum_hazard + weight_stall + unit_busy +
               weight_fifo_full + weight_fifo_empty + drain;
    }
};

// Every accumulator bank, flattened, as of one Sync. The mirror of ref::Snapshot.
struct TpuSnapshot {
    std::vector<i32> acc;
};

struct TpuResult {
    std::size_t pc        = 0;
    uint32_t    exit_code = 0;
    uint64_t    retired   = 0;
    uint64_t    cycles    = 0;

    bool halted  = false;
    bool trapped = false;
    bool budget  = false;

    std::string              trap_reason;
    std::vector<TpuSnapshot> syncs;

    bool done() const { return halted || trapped; }
};

struct TpuOptions {
    uint64_t   max_cycles = 10'000'000;   // runaway backstop
    bool       trace      = false;
    std::FILE* trace_out  = stderr;
};

// The machine: the MXU, the Unified Buffer, the accumulator banks, the weight
// FIFO, the DMA engine, host and weight memory, and a program counter.
//
// In this phase the units are driven by hand, one call per operation, and tick()
// only advances time. The sequencer that turns a decoded instruction stream into
// these same calls arrives in Phase 5; keeping the units independently drivable
// is what lets that be an addition rather than a rewrite.
class Tpu {
public:
    explicit Tpu(const Config& cfg, std::size_t host_bytes = 1u << 16,
                 std::size_t weight_bytes = 1u << 16);

    const Config& config() const { return cfg_; }
    uint64_t cycle() const { return cycle_; }
    std::size_t pc() const { return pc_; }

    UnifiedBuffer& ub() { return ub_; }
    Accumulators&  acc() { return acc_; }
    WeightFifo&    weight_fifo() { return fifo_; }
    Dma&           dma() { return dma_; }
    Mxu&           mxu() { return mxu_; }

    const UnifiedBuffer& ub() const { return ub_; }
    const Accumulators&  acc() const { return acc_; }

    std::vector<uint8_t>& host() { return host_; }
    std::vector<i8>&      weight_mem() { return weight_mem_; }

    // One cycle. Stages are evaluated in reverse pipeline order so each observes
    // the previous cycle's output of its producer, making the inter-stage
    // registers behave as latches.
    void tick();

    // Run tick() until `pred` holds or `limit` cycles pass. Returns false if the
    // limit ran out, so a test cannot hang on a unit that never completes.
    template <typename Pred>
    bool run_until(Pred pred, uint64_t limit = 1u << 20) {
        for (uint64_t i = 0; i < limit; ++i) {
            if (pred()) return true;
            tick();
        }
        return pred();
    }

    // ---- hand-driven operations ------------------------------------------
    // Each is one decoded instruction's worth of work. Phase 5 calls these from
    // the sequencer instead of from a test.

    bool dma_to_ub(HostAddr host_addr, UbAddr ub_addr, uint32_t bytes);
    bool dma_to_host(UbAddr ub_addr, HostAddr host_addr, uint32_t bytes);

    // Request a rows x cols weight tile from weight memory into the FIFO. The
    // tile lands ddr_tile_latency cycles later.
    bool stage_weights(uint32_t ddr_addr, uint32_t rows, uint32_t cols, TileId tile = 0);

    // Pop a staged tile into the array's load plane. False means the FIFO had
    // nothing ready, which is the weight_fifo_empty stall.
    bool load_weights_from_fifo();

    // Stream `len` activation rows at `ub_addr` through the array into a bank.
    bool matmul(UbAddr ub_addr, uint32_t len, BankId bank, bool accumulate);

    // ---- the sequencer ----------------------------------------------------
    // Fetch, decode, and issue in order, one instruction per cycle, with the
    // scoreboard gating issue and issued instructions overlapping in different
    // units. Runs until Halt, a trap, or the cycle budget.
    TpuResult run(const std::vector<RawInst>& prog, const TpuOptions& opts = TpuOptions{});

    const StallStats& stalls() const { return stalls_; }

    // Whether anything is still in flight. Sync, Halt and traps wait for this.
    bool quiet() const;

private:
    // What the instruction touches, plus whether its operands are in range. A
    // false return fills `why` with the same text the oracle uses, because a trap
    // reason is a contract between the two models rather than an independent
    // guess -- the same exception made for requantization.
    bool validate(const Decoded& d, Reservation& res, std::string& why) const;

    // Cycles the instruction occupies its unit.
    uint64_t duration_of(const Decoded& d) const;

    static Unit unit_of(Op op);
    InFlight& slot(Unit u) { return units_[static_cast<std::size_t>(u)]; }
    const InFlight& slot(Unit u) const { return units_[static_cast<std::size_t>(u)]; }

    // Does anything in flight conflict with this reservation? Bumps the matching
    // stall counter and returns true.
    bool interlocked(const Reservation& r);

    // Read the instruction's inputs and stage its outputs, returning how long it
    // occupies its unit.
    uint64_t execute(const Decoded& d, PendingWrite& pw);

    // The activation pipeline: bias, requantize, activation function, optional
    // pool. Reads the accumulator bank and stages the int8 tile it produces.
    void stage_activate(const Decoded& d, PendingWrite& pw);

    // Commit the staged outputs.
    void finish(InFlight& f);

    void issue_step(const std::vector<RawInst>& prog, TpuResult& st, const TpuOptions& opts);
    void retire_completed(TpuResult& st);
    void reset_pipeline();

    Config cfg_;

    UnifiedBuffer ub_;
    Accumulators  acc_;
    WeightFifo    fifo_;
    Dma           dma_;
    Mxu           mxu_;

    std::vector<uint8_t> host_;
    std::vector<i8>      weight_mem_;

    uint64_t    cycle_ = 0;
    std::size_t pc_    = 0;

    InFlight   units_[static_cast<std::size_t>(Unit::COUNT)];
    StallStats stalls_;
};
