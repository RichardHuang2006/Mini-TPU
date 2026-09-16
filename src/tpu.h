#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cstdio>
#include <string>

#include "config.h"
#include "datapath.h"
#include "isa.h"
#include "storage.h"
#include "systolic_array.h"
#include "transfer.h"

struct UbRegion {
    UbAddr lo = 0;
    UbAddr hi = 0;   // exclusive

    bool empty() const { return hi <= lo; }
    bool overlaps(const UbRegion& o) const {
        return !empty() && !o.empty() && lo < o.hi && o.lo < hi;
    }
};

// The scoreboard's view of an instruction, computed once at issue.
struct Reservation {
    UbRegion ub_read;
    UbRegion ub_write;
    BankId   acc_read      = INVALID_BANK;
    BankId   acc_write     = INVALID_BANK;
    bool     reads_weights = false;    // MatMul consumes the resident tile
    bool     writes_weights = false;   // Read_Weights replaces it
};

// One instruction in flight per unit; SEQ covers those with no unit of their own.
enum class Unit : uint8_t { DMA, WEIGHT, MXU, ACT, SEQ, COUNT };

inline const char* unit_name(Unit u) {
    switch (u) {
        case Unit::DMA:    return "DMA";
        case Unit::WEIGHT: return "WEIGHT";
        case Unit::MXU:    return "MXU";
        case Unit::ACT:    return "ACT";
        case Unit::SEQ:    return "SEQ";
        case Unit::COUNT:  break;
    }
    return "?";
}

// Why one issue attempt failed; each value names the StallStats counter it moves.
enum class StallReason : uint8_t {
    NONE,
    DRAIN,
    UNIT_BUSY,
    WEIGHT_FIFO_EMPTY,
    UB_RAW,
    UB_WAR,
    UB_WAW,
    ACCUM_HAZARD,
    WEIGHT_STALL,
    UB_BANK_CONFLICT,
};

inline const char* stall_reason_name(StallReason r) {
    switch (r) {
        case StallReason::NONE:              return "none";
        case StallReason::DRAIN:             return "drain";
        case StallReason::UNIT_BUSY:         return "unit_busy";
        case StallReason::WEIGHT_FIFO_EMPTY: return "weight_fifo_empty";
        case StallReason::UB_RAW:            return "ub_raw";
        case StallReason::UB_WAR:            return "ub_war";
        case StallReason::UB_WAW:            return "ub_waw";
        case StallReason::ACCUM_HAZARD:      return "accum_hazard";
        case StallReason::WEIGHT_STALL:      return "weight_stall";
        case StallReason::UB_BANK_CONFLICT:  return "ub_bank_conflict";
    }
    return "?";
}

// Outputs staged at issue, committed at retire.
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
    std::size_t  pc          = 0;      // which instruction this is, for tracing
    Reservation  res;
    PendingWrite pending;
    uint64_t     issue_cycle = 0;
    uint64_t     done_cycle  = 0;
};

// Why issue could not proceed, per cycle.
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

    // Cycles the array stood idle for a weight load with no shadow plane.
    uint64_t weight_load_bubble = 0;

    uint64_t total() const {
        return ub_raw + ub_war + ub_waw + accum_hazard + weight_stall + unit_busy +
               weight_fifo_full + weight_fifo_empty + ub_bank_conflict + drain;
    }
};

// Raw per-cycle tallies; the idle buckets partition `cycles - array_busy`.
struct RunProfile {
    uint64_t array_busy    = 0;   // cycles the MXU had a matmul in flight
    uint64_t stream_cycles = 0;   // of those, cycles spent streaming rows
    uint64_t matmuls       = 0;

    // MACs the array performed, padding included: one per PE per streaming cycle.
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

// Every accumulator bank, flattened, as of one Sync.
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

// Defined in src/trace.h; a null sink means no tracing.
struct TraceSink;
struct CycleInfo;

struct TpuOptions {
    uint64_t   max_cycles = 10'000'000;   // runaway backstop
    bool       trace      = false;
    std::FILE* trace_out  = stderr;
    TraceSink* sink       = nullptr;
};

// The whole machine; run() layers the sequencer over its drivable units.
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
    const WeightFifo&    weight_fifo() const { return fifo_; }
    const Dma&           dma() const { return dma_; }
    const Mxu&           mxu() const { return mxu_; }

    std::vector<uint8_t>& host() { return host_; }
    std::vector<i8>&      weight_mem() { return weight_mem_; }

    const std::vector<uint8_t>& host() const { return host_; }
    const std::vector<i8>&      weight_mem() const { return weight_mem_; }

    // Read-only views of the sequencer's state, for a trace sink.
    const InFlight& unit(Unit u) const { return slot(u); }
    std::size_t     prefetch_pc() const { return prefetch_pc_; }

    // In-flight instructions holding a Unified Buffer read / write stream.
    void ub_streams(uint32_t& readers, uint32_t& writers) const;

    // One cycle, with stages evaluated in reverse pipeline order.
    void tick();

    // Run tick() until `pred` holds; false means `limit` cycles ran out.
    template <typename Pred>
    bool run_until(Pred pred, uint64_t limit = 1u << 20) {
        for (uint64_t i = 0; i < limit; ++i) {
            if (pred()) return true;
            tick();
        }
        return pred();
    }

    // Hand-driven operations, one decoded instruction's worth of work each.
    bool dma_to_ub(HostAddr host_addr, UbAddr ub_addr, uint32_t bytes);
    bool dma_to_host(UbAddr ub_addr, HostAddr host_addr, uint32_t bytes);

    // Request a weight tile into the FIFO; it lands ddr_tile_latency cycles later.
    bool stage_weights(uint32_t ddr_addr, uint32_t rows, uint32_t cols, TileId tile = 0);

    // Pop a staged tile into the array's load plane; false is weight_fifo_empty.
    bool load_weights_from_fifo();

    // Stream `len` activation rows at `ub_addr` through the array into a bank.
    bool matmul(UbAddr ub_addr, uint32_t len, BankId bank, bool accumulate);

    // Issue in order, one instruction per cycle, until Halt, a trap, or budget.
    TpuResult run(const std::vector<RawInst>& prog, const TpuOptions& opts = TpuOptions{});

    const StallStats& stalls() const { return stalls_; }
    const RunProfile& profile() const { return profile_; }

    // Whether anything is still in flight. Sync, Halt and traps wait for this.
    bool quiet() const;

private:
    // What it touches and whether its operands are in range; `why` matches the
    // oracle's wording, which the differential tests compare.
    bool validate(const Decoded& d, Reservation& res, std::string& why) const;

    // Cycles the instruction occupies its unit.
    uint64_t duration_of(const Decoded& d) const;

    static Unit unit_of(Op op);
    InFlight& slot(Unit u) { return units_[static_cast<std::size_t>(u)]; }
    const InFlight& slot(Unit u) const { return units_[static_cast<std::size_t>(u)]; }

    // Does anything in flight conflict? Bumps the matching stall counter.
    bool interlocked(const Reservation& r, StallReason& why, Unit& blocker);

    // Is a Unified Buffer port free? The bank count budgets same-direction streams.
    bool ub_port_available(const Reservation& r);

    // Read the instruction's inputs and stage its outputs; returns its duration.
    uint64_t execute(const Decoded& d, PendingWrite& pw);

    // Bias, requantize, activation function, optional pool.
    void stage_activate(const Decoded& d, PendingWrite& pw);

    // Commit the staged outputs.
    void finish(InFlight& f);

    void issue_step(const std::vector<RawInst>& prog, TpuResult& st, const TpuOptions& opts);
    void retire_completed(TpuResult& st);
    void reset_pipeline();

    // Read ahead for upcoming Read_Weights and keep their tiles arriving from DDR.
    void prefetch_weights(const std::vector<RawInst>& prog);

    // Charge one array-idle cycle to a cause, given the stall counters before it.
    void charge_idle_cycle(const StallStats& before);

    // The per-cycle trace record, sampled after the accounting step.
    CycleInfo cycle_info(const RunProfile& before, const TpuResult& st) const;

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

    // How far the weight prefetcher has read ahead; always at or ahead of pc_.
    std::size_t prefetch_pc_ = 0;

    InFlight   units_[static_cast<std::size_t>(Unit::COUNT)];
    StallStats stalls_;
    RunProfile profile_;

    // Set for the duration of run(), plus this cycle's issue outcome for the hook.
    TraceSink*  sink_            = nullptr;
    bool        last_issued_     = false;
    bool        last_trapped_    = false;
    StallReason last_reason_     = StallReason::NONE;
    Unit        last_blocker_    = Unit::COUNT;
    std::size_t last_blocker_pc_ = 0;
};
