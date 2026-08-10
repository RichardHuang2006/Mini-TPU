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

// Why issue could not proceed, per cycle: a slow program is slow for one of
// these reasons and the counters say which.
struct StallStats {
    uint64_t ub_raw       = 0;   // reader waiting on the writer of its region
    uint64_t ub_war       = 0;   // writer waiting on a reader of its region
    uint64_t ub_waw       = 0;
    uint64_t accum_hazard = 0;   // accumulator bank still being produced
    uint64_t weight_stall = 0;   // waiting on the resident weight tile

    uint64_t unit_busy         = 0;   // structural: the unit already has work
    uint64_t weight_fifo_full  = 0;
    uint64_t weight_fifo_empty = 0;
    uint64_t ub_bank_conflict  = 0;   // no free Unified Buffer port this cycle
    uint64_t drain             = 0;   // Sync, Halt, or a trap waiting for quiet

    // Cycles the array stood idle for a weight load because there was no shadow
    // plane to hide it in.
    uint64_t weight_load_bubble = 0;

    uint64_t total() const {
        return ub_raw + ub_war + ub_waw + accum_hazard + weight_stall + unit_busy +
               weight_fifo_full + weight_fifo_empty + ub_bank_conflict + drain;
    }
};

// What the array was doing, cycle by cycle, and why it was not doing anything
// better. src/stats.h turns this into utilization, TOPS and a stall-cause
// breakdown; keeping the raw tallies here and the arithmetic there means the
// machine never has to know how a number will be presented.
//
// The idle buckets partition `cycles - array_busy` exactly: every cycle the array
// stands still bumps precisely one of them. That is the property that makes the
// breakdown a diagnosis instead of a decoration -- if the buckets did not add up,
// a dominant cause would be an artefact of what got double counted.
struct RunProfile {
    uint64_t array_busy    = 0;   // cycles the MXU had a matmul in flight
    uint64_t stream_cycles = 0;   // of those, cycles spent streaming rows
    uint64_t matmuls       = 0;

    // MACs the array actually performed, padding included: one per PE per
    // streaming cycle. The useful subset is a property of the workload, not of
    // the machine, so the caller supplies it.
    uint64_t macs_performed = 0;

    uint64_t dma_bytes = 0;

    // Cycles the array was idle, by what the machine was waiting on.
    uint64_t idle_weights   = 0;   // the weight path: hazard, FIFO, or DDR latency
    uint64_t idle_bank      = 0;   // no free Unified Buffer port
    uint64_t idle_accum     = 0;   // an accumulator bank still in use
    uint64_t idle_dma       = 0;   // data movement in flight
    uint64_t idle_act       = 0;   // the activation pipeline in flight
    uint64_t idle_other     = 0;   // drain at the end, and everything unclaimed

    static constexpr std::size_t OPS = 8;
    uint64_t op_cycles[OPS] = {};   // cycles each opcode occupied its unit
    uint64_t op_count[OPS]  = {};

    uint64_t idle_total() const {
        return idle_weights + idle_bank + idle_accum + idle_dma + idle_act + idle_other;
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
// The units are independently drivable, one call per operation, with tick()
// advancing time; run() layers the sequencer on top, turning a decoded
// instruction stream into these same calls.
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
    // Each is one decoded instruction's worth of work. The sequencer calls
    // these; tests can also drive them directly.

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
    const RunProfile& profile() const { return profile_; }

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

    // Is there a free Unified Buffer port for this instruction's stream?
    //
    // A bank exposes one read and one write port per cycle, and an instruction
    // touching the buffer holds the port of its direction for its whole duration.
    // So the bank count is a budget on how many transfers in the
    // same direction can be in flight at once, and the model tracks streams rather
    // than the individual byte each one reaches in a given cycle: a row of a tile
    // spans every bank at these sizes, so a byte-exact check would forbid a matmul
    // and a DMA from ever overlapping. bank_of() remains the byte-exact primitive
    // for the single-cycle multi-address case.
    bool ub_port_available(const Reservation& r);

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

    // Read ahead for upcoming Read_Weights and keep their tiles arriving from DDR,
    // so the FIFO depth decides how much of the latency is hidden.
    void prefetch_weights(const std::vector<RawInst>& prog);

    // Charge one array-idle cycle to a cause, given the stall counters as they
    // stood before this cycle's issue attempt.
    void charge_idle_cycle(const StallStats& before);

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

    // How far the weight prefetcher has read ahead. Always at or ahead of pc_.
    std::size_t prefetch_pc_ = 0;

    InFlight   units_[static_cast<std::size_t>(Unit::COUNT)];
    StallStats stalls_;
    RunProfile profile_;
};
