// Generates the performance tables in DESIGN.md §9.
//
// The tables are checked in, so they need to be reproducible rather than
// hand-entered: `make report` prints markdown that can be pasted back, and the same
// measurements are asserted by the test suite's "stats", "config_sweep" and
// "properties" sections. If a change to the model moves a number here, the tests
// that pin the reasons for it move too.

#include <cstdio>
#include <string>
#include <vector>

#include "stats.h"
#include "tpu.h"
#include "workloads.h"

namespace {

struct Run {
    stats::Stats stats;
    bool         halted = false;
};

Run execute(const wl::Spec& spec, const Config& cfg) {
    const wl::Workload w = wl::build(spec, cfg);

    const std::size_t host_bytes =
        std::max<std::size_t>(w.a_host + w.a.size(), w.y_host + w.y_bytes) + 64;
    const std::size_t weight_bytes = w.b_ddr + w.b.size() + 64;

    Tpu t(cfg, host_bytes, weight_bytes);
    for (std::size_t i = 0; i < w.a.size(); ++i) {
        t.host()[w.a_host + i] = static_cast<uint8_t>(w.a[i]);
    }
    for (std::size_t i = 0; i < w.b.size(); ++i) {
        t.weight_mem()[w.b_ddr + i] = w.b[i];
    }

    Run out;
    const TpuResult r = t.run(w.code);
    out.halted = r.halted;
    out.stats  = stats::gather(t, r, w.macs, /*useful_known=*/true);
    return out;
}

Config sized(const wl::Spec& spec, uint32_t dim) {
    Config cfg = spec.cfg;
    cfg.dim = dim;
    // Room for the tiles this array size needs, whatever the workload shipped with.
    cfg.ub_bytes = std::max<uint32_t>(cfg.ub_bytes, 16 * dim * dim);
    return cfg;
}

const uint32_t DIMS[] = {8, 32, 256};

void utilization_table() {
    std::printf("### 9.2 Utilization and TOPS by array size\n\n");
    std::printf("| Workload | 8x8 util | 32x32 util | 256x256 util | 32x32 TOPS |"
                " 32x32 dominant cause |\n");
    std::printf("|---|---|---|---|---|---|\n");

    for (const wl::Spec& spec : wl::specs()) {
        std::printf("| %s ", spec.name.c_str());

        double tops_32 = 0.0;
        std::string cause_32 = "?";
        for (const uint32_t dim : DIMS) {
            const Run r = execute(spec, sized(spec, dim));
            std::printf("| %.1f%% ", 100.0 * r.stats.utilization());
            if (dim == 32) {
                tops_32  = r.stats.tops();
                cause_32 = r.stats.dominant_name();
            }
        }
        std::printf("| %.3f | `%s` |\n", tops_32, cause_32.c_str());
    }
    std::printf("\n");
}

void roofline_table() {
    std::printf("### 9.3 Roofline\n\n");
    std::printf("| Workload | MACs | bytes moved | intensity (MAC/B) | ridge (MAC/B) |"
                " placement | dominant stall |\n");
    std::printf("|---|---|---|---|---|---|---|\n");

    for (const wl::Spec& spec : wl::specs()) {
        const Run r = execute(spec, sized(spec, 32));
        std::printf("| %s | %llu | %llu | %.1f | %.1f | %s | `%s` |\n", spec.name.c_str(),
                    static_cast<unsigned long long>(r.stats.macs_useful),
                    static_cast<unsigned long long>(r.stats.dma_bytes),
                    r.stats.arithmetic_intensity(), r.stats.ridge_point(),
                    r.stats.below_ridge() ? "memory-bound" : "compute-bound",
                    r.stats.dominant_stall_name());
    }
    std::printf("\n");
}

void breakdown_table() {
    std::printf("### 9.4 Where the cycles go (32x32)\n\n");
    std::printf("| Workload | cycles | fill/drain | activation | DMA | weights |"
                " banks | accum | padding |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|\n");

    for (const wl::Spec& spec : wl::specs()) {
        const Run r = execute(spec, sized(spec, 32));
        const stats::Breakdown& b = r.stats.lost;
        auto pct = [&](uint64_t v) {
            return r.stats.cycles == 0
                       ? 0.0
                       : 100.0 * static_cast<double>(v) / static_cast<double>(r.stats.cycles);
        };
        std::printf("| %s | %llu | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.0f%% |"
                    " %.0f%% |\n",
                    spec.name.c_str(),
                    static_cast<unsigned long long>(r.stats.cycles),
                    pct(b.array_fill_drain), pct(b.activation), pct(b.dma_bound),
                    pct(b.weight_fifo_empty), pct(b.ub_bank_conflict), pct(b.accum_hazard),
                    pct(b.partial_tile_waste));
    }
    std::printf("\n");
}

void starved_table() {
    struct Named { const char* name; Config (*make)(Config); };

    const Named configs[] = {
        {"default",             [](Config c) { return c; }},
        {"single-bank UB",      [](Config c) { c.ub_banks = 1; return c; }},
        {"1-deep FIFO",         [](Config c) { c.weight_fifo_depth = 1; return c; }},
        {"1-deep FIFO, DDR 200",[](Config c) {
             c.weight_fifo_depth = 1; c.ddr_tile_latency = 200; return c;
         }},
        {"DMA 1 B/cyc",         [](Config c) { c.dma_bytes_per_cycle = 1; return c; }},
    };

    std::printf("### Starved configurations (matmul_128, 32x32)\n\n");
    std::printf("| Configuration | cycles | vs default | dominant stall |\n");
    std::printf("|---|---|---|---|\n");

    const wl::Spec spec = wl::specs().front();
    const uint64_t base = execute(spec, sized(spec, 32)).stats.cycles;

    for (const Named& n : configs) {
        const Config cfg = n.make(sized(spec, 32));
        const Run    r   = execute(spec, cfg);
        std::printf("| %s | %llu | %+.1f%% | `%s` |\n", n.name,
                    static_cast<unsigned long long>(r.stats.cycles),
                    100.0 * (static_cast<double>(r.stats.cycles) -
                             static_cast<double>(base)) /
                        static_cast<double>(base),
                    r.stats.dominant_stall_name());
    }
    std::printf("\n");
}

}  // namespace

int main() {
    std::printf("<!-- generated by `make report`; do not edit by hand -->\n\n");
    utilization_table();
    roofline_table();
    breakdown_table();
    starved_table();
    return 0;
}
