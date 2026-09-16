#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

#include "config.h"
#include "datapath.h"
#include "isa.h"
#include "tpu.h"

// Derived statistics: arithmetic over RunProfile, described in README section 16.
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

// Lost array-cycles by cause, from the array's point of view.
struct Breakdown {
    uint64_t array_fill_drain   = 0;
    uint64_t weight_fifo_empty  = 0;
    uint64_t ub_bank_conflict   = 0;
    uint64_t accum_hazard       = 0;
    uint64_t dma_bound          = 0;
    uint64_t activation         = 0;
    uint64_t other              = 0;

    // Outside the partition: MAC slots spent on padding while the array was busy.
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

    uint64_t peak_macs() const { return cfg.peak_macs_per_cycle(); }

    // Useful MACs over what the array could have done in the same cycles.
    double utilization() const {
        const uint64_t offered = cycles * peak_macs();
        if (offered == 0) return 0.0;
        return static_cast<double>(macs_useful) / static_cast<double>(offered);
    }

    // The same fraction over only the cycles the array had work.
    double busy_utilization() const {
        const uint64_t offered = array_busy * peak_macs();
        if (offered == 0) return 0.0;
        return static_cast<double>(macs_useful) / static_cast<double>(offered);
    }

    // Two ops per MAC, at TPUv1's 700 MHz by default.
    double tops(double clock_ghz = 0.7) const {
        if (cycles == 0) return 0.0;
        const double seconds = static_cast<double>(cycles) / (clock_ghz * 1e9);
        return 2.0 * static_cast<double>(macs_useful) / seconds / 1e12;
    }

    // Useful MACs per byte moved across the host interface.
    double arithmetic_intensity() const {
        if (dma_bytes == 0) return 0.0;
        return static_cast<double>(macs_useful) / static_cast<double>(dma_bytes);
    }

    // MACs per byte at which the array and the DMA are balanced.
    double ridge_point() const {
        if (cfg.dma_bytes_per_cycle == 0) return 0.0;
        return static_cast<double>(peak_macs()) /
               static_cast<double>(cfg.dma_bytes_per_cycle);
    }

    bool below_ridge() const { return cfg.dma_bound(macs_useful, dma_bytes); }

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

    // The largest reason the array did no useful work; ties go to the earlier.
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

    // The largest cause a configuration could fix, so not fill/drain or padding.
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

    // Every cycle is either array-busy or charged to exactly one idle cause.
    bool balances() const {
        return array_busy + lost.idle() == cycles &&
               stream_cycles + lost.array_fill_drain == array_busy;
    }

    std::string report(const std::string& title = "") const;
};

// `useful_macs` excludes padding, which only the tiler can know.
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

    // Taken as a difference, so the partition holds by construction.
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
