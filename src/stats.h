#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

#include "config.h"
#include "isa.h"
#include "tpu.h"
#include "types.h"

// Derived statistics for one run: utilization, effective TOPS, per-instruction
// cycle counts, and a stall-cause breakdown attributing every idle array-cycle to
// something.
//
// The machine collects tallies (RunProfile); everything here is arithmetic over
// them. The split matters because the breakdown has a job to do -- naming the
// bottleneck of a starved configuration -- and that job is easier to get wrong in
// the accounting than in the measurement.
//
// Two properties keep it honest, both asserted by the test suite:
//
//   * the idle buckets partition idle time exactly, so no cause can dominate by
//     being counted twice, and
//   * streaming plus fill/drain equals the time the array was busy.
//
// `partial_tile_waste` is deliberately outside that partition. It is waste inside
// cycles the array *was* busy -- PEs multiplying padding -- so adding it to the
// idle buckets would double count. It is reported alongside them, not among them.

namespace stats {

enum class Cause : uint8_t {
    NONE,
    ARRAY_FILL_DRAIN,
    PARTIAL_TILE_WASTE,
    WEIGHT_FIFO_EMPTY,
    UB_BANK_CONFLICT,
    ACCUM_HAZARD,
    DMA_BOUND,
    ACTIVATION,
    OTHER,
};

inline const char* cause_name(Cause c) {
    switch (c) {
        case Cause::NONE:               return "none";
        case Cause::ARRAY_FILL_DRAIN:   return "array_fill_drain";
        case Cause::PARTIAL_TILE_WASTE: return "partial_tile_waste";
        case Cause::WEIGHT_FIFO_EMPTY:  return "weight_fifo_empty";
        case Cause::UB_BANK_CONFLICT:   return "ub_bank_conflict";
        case Cause::ACCUM_HAZARD:       return "accum_hazard";
        case Cause::DMA_BOUND:          return "dma_bound";
        case Cause::ACTIVATION:         return "activation";
        case Cause::OTHER:              return "other";
    }
    return "?";
}

// Lost array-cycles by cause. The names are §9.4's, which describe the array's
// point of view; a few cover more than one of the machine's issue-point counters:
//
//   weight_fifo_empty  every cycle the array had no tile to multiply by, whether
//                      the sequencer was waiting on DDR latency, on FIFO space, or
//                      on the Read_Weights ahead of it
//   dma_bound          the array idle with a transfer in flight, however the
//                      hazard that blocked issue was named
struct Breakdown {
    uint64_t array_fill_drain   = 0;
    uint64_t weight_fifo_empty  = 0;
    uint64_t ub_bank_conflict   = 0;
    uint64_t accum_hazard       = 0;
    uint64_t dma_bound          = 0;
    uint64_t activation         = 0;
    uint64_t other              = 0;

    // Outside the partition above: array-cycles' worth of MAC slots spent on
    // padding while the array was busy.
    uint64_t partial_tile_waste = 0;

    uint64_t idle() const {
        return weight_fifo_empty + ub_bank_conflict + accum_hazard + dma_bound +
               activation + other;
    }
};

struct Stats {
    Config cfg;

    uint64_t cycles  = 0;
    uint64_t retired = 0;

    uint64_t array_busy    = 0;
    uint64_t stream_cycles = 0;

    uint64_t macs_performed = 0;   // including padding
    uint64_t macs_useful    = 0;   // supplied by the workload; 0 when unknown
    bool     useful_known   = false;

    uint64_t dma_bytes = 0;

    uint64_t op_cycles[RunProfile::OPS] = {};
    uint64_t op_count[RunProfile::OPS]  = {};

    Breakdown lost;

    // ---- utilization -------------------------------------------------------

    uint64_t peak_macs() const { return cfg.peak_macs_per_cycle(); }

    // Useful MACs as a fraction of what the array could have done in the same
    // wall-clock cycles. This is the number that should hurt: it counts padding,
    // fill, drain and every stall against you.
    double utilization() const {
        const uint64_t offered = cycles * peak_macs();
        if (offered == 0) return 0.0;
        return static_cast<double>(macs_useful) / static_cast<double>(offered);
    }

    // The same fraction over only the cycles the array had work, which separates
    // "the array was idle" from "the array was busy doing nothing useful".
    double busy_utilization() const {
        const uint64_t offered = array_busy * peak_macs();
        if (offered == 0) return 0.0;
        return static_cast<double>(macs_useful) / static_cast<double>(offered);
    }

    // Two ops per MAC, the usual convention, at a nominal clock. TPUv1 ran at
    // 700 MHz, so that is the default and the numbers are comparable to its 92 TOPS.
    double tops(double clock_ghz = 0.7) const {
        if (cycles == 0) return 0.0;
        const double seconds = static_cast<double>(cycles) / (clock_ghz * 1e9);
        return 2.0 * static_cast<double>(macs_useful) / seconds / 1e12;
    }

    // ---- roofline ----------------------------------------------------------

    // Useful MACs per byte moved across the host interface.
    double arithmetic_intensity() const {
        if (dma_bytes == 0) return 0.0;
        return static_cast<double>(macs_useful) / static_cast<double>(dma_bytes);
    }

    // MACs per byte at which the array and the DMA are exactly balanced. Below it a
    // workload is memory-bound and a larger array buys nothing.
    double ridge_point() const {
        if (cfg.dma_bytes_per_cycle == 0) return 0.0;
        return static_cast<double>(peak_macs()) /
               static_cast<double>(cfg.dma_bytes_per_cycle);
    }

    bool below_ridge() const { return cfg.dma_bound(macs_useful, dma_bytes); }

    // ---- diagnosis ---------------------------------------------------------

    uint64_t weight_of(Cause c) const {
        switch (c) {
            case Cause::ARRAY_FILL_DRAIN:   return lost.array_fill_drain;
            case Cause::PARTIAL_TILE_WASTE: return lost.partial_tile_waste;
            case Cause::WEIGHT_FIFO_EMPTY:  return lost.weight_fifo_empty;
            case Cause::UB_BANK_CONFLICT:   return lost.ub_bank_conflict;
            case Cause::ACCUM_HAZARD:       return lost.accum_hazard;
            case Cause::DMA_BOUND:          return lost.dma_bound;
            case Cause::ACTIVATION:         return lost.activation;
            case Cause::OTHER:              return lost.other;
            case Cause::NONE:               return 0;
        }
        return 0;
    }

    // The single largest reason the array did not do useful work. Ties go to the
    // earlier cause in the list, which is stable across runs.
    Cause dominant_cause() const {
        const Cause all[] = {
            Cause::ARRAY_FILL_DRAIN, Cause::PARTIAL_TILE_WASTE, Cause::WEIGHT_FIFO_EMPTY,
            Cause::UB_BANK_CONFLICT, Cause::ACCUM_HAZARD,       Cause::DMA_BOUND,
            Cause::ACTIVATION,       Cause::OTHER,
        };
        Cause    best  = Cause::NONE;
        uint64_t max_w = 0;
        for (const Cause c : all) {
            const uint64_t w = weight_of(c);
            if (w > max_w) { max_w = w; best = c; }
        }
        return best;
    }

    const char* dominant_name() const { return cause_name(dominant_cause()); }

    // The largest cause a *configuration* could fix, which excludes the two that
    // belong to the workload and the array shape rather than to the machine's
    // resources: fill/drain is what a systolic pipeline costs, and padding waste is
    // what the tile size costs. Those two dominate most runs, so ranking them
    // against the resource stalls would hide every provisioning problem behind
    // "array_fill_drain".
    Cause dominant_stall_cause() const {
        const Cause all[] = {
            Cause::WEIGHT_FIFO_EMPTY, Cause::UB_BANK_CONFLICT, Cause::ACCUM_HAZARD,
            Cause::DMA_BOUND,         Cause::ACTIVATION,       Cause::OTHER,
        };
        Cause    best  = Cause::NONE;
        uint64_t max_w = 0;
        for (const Cause c : all) {
            const uint64_t w = weight_of(c);
            if (w > max_w) { max_w = w; best = c; }
        }
        return best;
    }

    const char* dominant_stall_name() const { return cause_name(dominant_stall_cause()); }

    // The accounting adds up: every cycle is either array-busy or charged to one
    // idle cause, and busy time splits into streaming and fill/drain.
    bool balances() const {
        return array_busy + lost.idle() == cycles &&
               stream_cycles + lost.array_fill_drain == array_busy;
    }

    // ---- reporting ---------------------------------------------------------

    std::string report(const std::string& title = "") const;
};

// Gather a run's statistics. `useful_macs` is the workload's own MAC count with
// padding excluded -- the tiler knows it and the machine cannot, since a padded
// zero is indistinguishable from a real one at the array.
inline Stats gather(const Config& cfg, const TpuResult& r, const RunProfile& p,
                    uint64_t useful_macs = 0, bool useful_known = false) {
    Stats s;
    s.cfg     = cfg;
    s.cycles  = r.cycles;
    s.retired = r.retired;

    s.array_busy    = p.array_busy;
    s.stream_cycles = p.stream_cycles;

    s.macs_performed = p.macs_performed;
    s.useful_known   = useful_known;
    s.macs_useful    = useful_known ? useful_macs : p.macs_performed;

    s.dma_bytes = p.dma_bytes;
    for (std::size_t i = 0; i < RunProfile::OPS; ++i) {
        s.op_cycles[i] = p.op_cycles[i];
        s.op_count[i]  = p.op_count[i];
    }

    // Fill and drain is whatever busy time was not streaming, taken as a
    // difference so the partition holds by construction instead of relying on the
    // duration formula and the measured busy count agreeing.
    s.lost.array_fill_drain =
        p.array_busy > p.stream_cycles ? p.array_busy - p.stream_cycles : 0;

    s.lost.weight_fifo_empty = p.idle_weights;
    s.lost.ub_bank_conflict  = p.idle_bank;
    s.lost.accum_hazard      = p.idle_accum;
    s.lost.dma_bound         = p.idle_dma;
    s.lost.activation        = p.idle_act;
    s.lost.other             = p.idle_other;

    const uint64_t peak = cfg.peak_macs_per_cycle();
    if (peak != 0 && s.macs_performed > s.macs_useful) {
        s.lost.partial_tile_waste = (s.macs_performed - s.macs_useful) / peak;
    }
    return s;
}

inline Stats gather(const Tpu& t, const TpuResult& r, uint64_t useful_macs = 0,
                    bool useful_known = false) {
    return gather(t.config(), r, t.profile(), useful_macs, useful_known);
}

inline std::string Stats::report(const std::string& title) const {
    char     buf[256];
    std::string out;
    auto line = [&](const char* fmt, auto... args) {
        std::snprintf(buf, sizeof buf, fmt, args...);
        out += buf;
    };

    if (!title.empty()) out += title + "\n";

    line("  array %ux%u  UB %u B / %u banks  acc %u  FIFO %u  DMA %u B/cyc%s\n",
         cfg.dim, cfg.dim, cfg.ub_bytes, cfg.ub_banks, cfg.acc_banks,
         cfg.weight_fifo_depth, cfg.dma_bytes_per_cycle,
         cfg.double_buffer ? "  double-buffered" : "");
    line("  %llu cycles, %llu instructions retired, %llu DMA bytes\n",
         static_cast<unsigned long long>(cycles),
         static_cast<unsigned long long>(retired),
         static_cast<unsigned long long>(dma_bytes));
    line("  %llu useful MACs of %llu performed  (%.1f%% of array-cycles offered)\n",
         static_cast<unsigned long long>(macs_useful),
         static_cast<unsigned long long>(macs_performed), 100.0 * utilization());
    line("  utilization %.1f%% overall, %.1f%% while busy   effective %.3f TOPS @ 700 MHz\n",
         100.0 * utilization(), 100.0 * busy_utilization(), tops());
    line("  arithmetic intensity %.1f MAC/B  ridge %.1f  -> %s\n",
         arithmetic_intensity(), ridge_point(),
         below_ridge() ? "memory-bound" : "compute-bound");

    out += "  lost array-cycles:\n";
    const Cause order[] = {
        Cause::ARRAY_FILL_DRAIN, Cause::WEIGHT_FIFO_EMPTY, Cause::UB_BANK_CONFLICT,
        Cause::ACCUM_HAZARD,     Cause::DMA_BOUND,         Cause::ACTIVATION,
        Cause::OTHER,
    };
    for (const Cause c : order) {
        const uint64_t w = weight_of(c);
        const double   pct = cycles == 0 ? 0.0 : 100.0 * static_cast<double>(w) /
                                                     static_cast<double>(cycles);
        line("    %-20s %8llu  %5.1f%%\n", cause_name(c),
             static_cast<unsigned long long>(w), pct);
    }
    line("    %-20s %8llu  %5.1f%%  (inside busy cycles)\n", "partial_tile_waste",
         static_cast<unsigned long long>(lost.partial_tile_waste),
         cycles == 0 ? 0.0
                     : 100.0 * static_cast<double>(lost.partial_tile_waste) /
                           static_cast<double>(cycles));
    line("  dominant cause: %s   (largest resource stall: %s)\n", dominant_name(),
         dominant_stall_name());

    out += "  per instruction:\n";
    for (std::size_t i = 0; i < RunProfile::OPS; ++i) {
        if (op_count[i] == 0) continue;
        line("    %-18s %6llu x  %8llu cycles  (%.1f avg)\n",
             op_name(static_cast<Op>(i)),
             static_cast<unsigned long long>(op_count[i]),
             static_cast<unsigned long long>(op_cycles[i]),
             static_cast<double>(op_cycles[i]) / static_cast<double>(op_count[i]));
    }
    return out;
}

}  // namespace stats
