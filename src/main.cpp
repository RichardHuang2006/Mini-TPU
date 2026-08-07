// Top-level driver for Mini-TPU.
//
// `int main()` is guarded so tests can #include this file and drive
// parse_args / print_help / disasm directly. Everything else is inline or
// file-static, so including it twice does not violate ODR.
//
// Execution arrives with the reference model; for now the driver loads a
// program and its tensors, checks them, and disassembles on request. That is
// enough to pin the loaders and the CLI surface before there is a machine to
// run anything on.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "config.h"
#include "decoder.h"
#include "isa.h"
#include "loader.h"
#include "types.h"

// ============================================================================
// CLI surface
// ============================================================================

struct CliOpts {
    std::string prog_hex_path;
    std::string prog_raw_path;
    std::string weights_path;
    std::string acts_path;

    bool dump      = false;   // disassemble the program and summarize tensors
    bool trace     = false;   // per-instruction trace once there is a machine
    bool show_help = false;

    Config cfg;
};

// One entry per uint32_t Config knob. The pointer-to-member lets the parser
// assign to any field uniformly, so adding a knob is one row here.
struct ConfigKnob {
    const char*        flag;
    uint32_t Config::* member;
    const char*        desc;
};

inline const ConfigKnob KNOBS[] = {
    {"--dim",        &Config::dim,                 "systolic array edge length (PEs per side)"},
    {"--ub",         &Config::ub_bytes,            "Unified Buffer size (bytes)"},
    {"--ub-banks",   &Config::ub_banks,            "Unified Buffer banks"},
    {"--acc-banks",  &Config::acc_banks,           "int32 accumulator banks"},
    {"--fifo",       &Config::weight_fifo_depth,   "weight FIFO depth (tiles)"},
    {"--dma",        &Config::dma_bytes_per_cycle, "host DMA bandwidth (bytes/cycle)"},
    {"--ddr-lat",    &Config::ddr_tile_latency,    "weight tile refill latency (cycles)"},
    {"--act-depth",  &Config::act_pipeline_depth,  "activation pipeline fill depth (cycles)"},
};

inline constexpr std::size_t NUM_KNOBS = sizeof(KNOBS) / sizeof(KNOBS[0]);

namespace mini_tpu_cli_detail {

inline bool parse_uint(const std::string& s, uint32_t& out) {
    try {
        std::size_t pos = 0;
        const unsigned long v = std::stoul(s, &pos, 0);   // base 0 -> auto 0x
        if (pos != s.size() || v > 0xFFFFFFFFul) return false;
        out = static_cast<uint32_t>(v);
        return true;
    } catch (...) { return false; }
}

inline std::string flag_of(const std::string& tok) {
    const auto eq = tok.find('=');
    return eq == std::string::npos ? tok : tok.substr(0, eq);
}

}  // namespace mini_tpu_cli_detail

// Returns 0 on success, non-zero on error with the message already on stderr.
inline int parse_args(int argc, char** argv, CliOpts& opts) {
    using namespace mini_tpu_cli_detail;
    for (int i = 1; i < argc; ++i) {
        const std::string tok  = argv[i];
        const std::string flag = flag_of(tok);

        auto take_value = [&](std::string& val) -> bool {
            const auto eq = tok.find('=');
            if (eq != std::string::npos) { val = tok.substr(eq + 1); return true; }
            if (i + 1 >= argc) {
                std::fprintf(stderr, "minitpu: %s requires a value\n", flag.c_str());
                return false;
            }
            val = argv[++i];
            return true;
        };

        if (flag == "--help" || flag == "-h") { opts.show_help = true; return 0; }
        if (flag == "--dump")  { opts.dump  = true; continue; }
        if (flag == "--trace") { opts.trace = true; continue; }

        if (flag == "--double-buffer")    { opts.cfg.double_buffer = true;  continue; }
        if (flag == "--no-double-buffer") { opts.cfg.double_buffer = false; continue; }

        if (flag == "--prog")     { if (!take_value(opts.prog_hex_path)) return 1; continue; }
        if (flag == "--prog-raw") { if (!take_value(opts.prog_raw_path)) return 1; continue; }
        if (flag == "--weights")  { if (!take_value(opts.weights_path))  return 1; continue; }
        if (flag == "--acts")     { if (!take_value(opts.acts_path))     return 1; continue; }

        bool matched = false;
        for (const auto& k : KNOBS) {
            if (flag == k.flag) {
                std::string v;
                if (!take_value(v)) return 1;
                uint32_t parsed = 0;
                if (!parse_uint(v, parsed)) {
                    std::fprintf(stderr, "minitpu: bad %s '%s'\n", k.flag, v.c_str());
                    return 1;
                }
                opts.cfg.*(k.member) = parsed;
                matched = true;
                break;
            }
        }
        if (matched) continue;

        std::fprintf(stderr, "minitpu: unknown flag '%s'\n", flag.c_str());
        return 1;
    }
    return 0;
}

inline void print_help() {
    std::printf("Usage: minitpu [OPTIONS]\n");
    std::printf("\nProgram input (exactly one):\n");
    std::printf("  --prog PATH       hex program (one 32-bit word per line, %u words per instruction)\n",
                ISA_WORDS);
    std::printf("  --prog-raw PATH   little-endian binary program image\n");
    std::printf("\nTensor input (self-describing MTPU containers):\n");
    std::printf("  --weights PATH    weight tensor\n");
    std::printf("  --acts PATH       activation tensor\n");
    std::printf("\nGeneral:\n");
    std::printf("  --dump            disassemble the program and summarize the tensors\n");
    std::printf("  --trace           per-instruction trace (needs the reference model)\n");
    std::printf("  --help, -h        this message\n");
    std::printf("\nMachine knobs:\n");
    for (const auto& k : KNOBS) {
        std::printf("  %-18s  %s\n", k.flag, k.desc);
    }
    std::printf("  %-18s  %s\n", "--double-buffer", "keep a shadow weight plane (default)");
    std::printf("  %-18s  %s\n", "--no-double-buffer", "load weights into the active plane");
}

// ============================================================================
// Reporting
// ============================================================================

// One line per instruction, operands named rather than positional so a listing
// can be read without the encoding table to hand.
inline std::string disasm(const Decoded& d) {
    char buf[192];
    switch (d.op) {
        case Op::READ_HOST:
            std::snprintf(buf, sizeof buf, "%-18s host=0x%08X ub=0x%08X bytes=%u",
                          op_name(d.op), d.host_addr, d.ub_addr, d.bytes);
            break;
        case Op::WRITE_HOST:
            std::snprintf(buf, sizeof buf, "%-18s ub=0x%08X host=0x%08X bytes=%u",
                          op_name(d.op), d.ub_addr, d.host_addr, d.bytes);
            break;
        case Op::READ_WEIGHTS:
            std::snprintf(buf, sizeof buf, "%-18s ddr=0x%08X tile=%u",
                          op_name(d.op), d.ddr_addr, d.tile);
            break;
        case Op::MATMUL:
            std::snprintf(buf, sizeof buf, "%-18s ub=0x%08X len=%u acc=%u%s",
                          op_name(d.op), d.ub_addr, d.len, d.acc_bank,
                          d.accumulate ? " accumulate" : "");
            break;
        case Op::ACTIVATE: {
            // The window and stride only mean anything when pooling is on.
            char pool_buf[48] = "";
            if (d.pool != Pool::NONE) {
                std::snprintf(pool_buf, sizeof pool_buf, " pool=%s(%ux%u/%u)",
                              pool_name(d.pool), d.pool_window, d.pool_window,
                              d.pool_stride);
            }
            std::snprintf(buf, sizeof buf,
                          "%-18s acc=%u ub=0x%08X len=%u bias=%d mult=%d shift=%u %s%s",
                          op_name(d.op), d.acc_bank, d.ub_addr, d.len, d.bias, d.multiplier,
                          d.shift, actfn_name(d.act), pool_buf);
            break;
        }
        case Op::HALT:
            std::snprintf(buf, sizeof buf, "%-18s code=%u%s", op_name(d.op), d.code,
                          d.trap ? "  (illegal opcode)" : "");
            break;
        case Op::SYNC:
        case Op::NOP:
            std::snprintf(buf, sizeof buf, "%s", op_name(d.op));
            break;
    }
    return std::string(buf);
}

inline void print_program(const std::vector<RawInst>& prog) {
    std::printf("program: %zu instructions\n", prog.size());
    for (std::size_t i = 0; i < prog.size(); ++i) {
        const Decoded d = decode(prog[i]);
        std::printf("  %4zu  %s\n", i, disasm(d).c_str());
    }
}

inline void print_tensor(const char* label, const TensorBlob& t) {
    std::printf("%s: %ux%u %s\n", label, t.rows, t.cols, t.wide ? "int32" : "int8");
    // A handful of leading elements is enough to tell "loaded the right file"
    // from "loaded the right shape of the wrong file".
    const std::size_t show = t.count() < 8 ? t.count() : 8;
    if (show == 0) return;
    std::printf("  head:");
    for (std::size_t i = 0; i < show; ++i) {
        if (t.wide) std::printf(" %d", t.i32v[i]);
        else        std::printf(" %d", static_cast<int>(t.i8v[i]));
    }
    std::printf("%s\n", t.count() > show ? " ..." : "");
}

inline void print_config(const Config& cfg) {
    std::printf("machine: %ux%u array  UB %u B / %u banks  acc %u banks"
                "  FIFO %u  DMA %u B/cyc  double-buffer %s\n",
                cfg.dim, cfg.dim, cfg.ub_bytes, cfg.ub_banks, cfg.acc_banks,
                cfg.weight_fifo_depth, cfg.dma_bytes_per_cycle,
                cfg.double_buffer ? "on" : "off");
}

// ============================================================================
// Entry point (compiled out when included from a test TU).
// ============================================================================

#ifndef MINI_TPU_NO_ENTRY
int main(int argc, char** argv) {
    CliOpts opts;
    if (const int rc = parse_args(argc, argv, opts); rc != 0) return rc;
    if (opts.show_help) { print_help(); return 0; }

    const int progs_given = int(!opts.prog_hex_path.empty())
                          + int(!opts.prog_raw_path.empty());
    if (progs_given != 1) {
        std::fprintf(stderr, "minitpu: give exactly one of --prog or --prog-raw\n");
        return 2;
    }

    std::vector<RawInst> prog;
    TensorBlob           weights, acts;
    try {
        if (!opts.prog_hex_path.empty()) {
            std::ifstream in(opts.prog_hex_path);
            if (!in) {
                std::fprintf(stderr, "minitpu: cannot open %s\n", opts.prog_hex_path.c_str());
                return 2;
            }
            prog = load_program_hex(in);
        } else {
            std::ifstream in(opts.prog_raw_path, std::ios::binary);
            if (!in) {
                std::fprintf(stderr, "minitpu: cannot open %s\n", opts.prog_raw_path.c_str());
                return 2;
            }
            prog = load_program_raw(in);
        }
        if (!opts.weights_path.empty()) {
            std::ifstream in(opts.weights_path, std::ios::binary);
            if (!in) {
                std::fprintf(stderr, "minitpu: cannot open %s\n", opts.weights_path.c_str());
                return 2;
            }
            weights = load_tensor_mtpu(in);
        }
        if (!opts.acts_path.empty()) {
            std::ifstream in(opts.acts_path, std::ios::binary);
            if (!in) {
                std::fprintf(stderr, "minitpu: cannot open %s\n", opts.acts_path.c_str());
                return 2;
            }
            acts = load_tensor_mtpu(in);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "minitpu: load failed: %s\n", e.what());
        return 2;
    }

    print_config(opts.cfg);
    if (opts.dump) {
        print_program(prog);
        if (!opts.weights_path.empty()) print_tensor("weights", weights);
        if (!opts.acts_path.empty())    print_tensor("acts", acts);
    } else {
        std::printf("loaded %zu instructions", prog.size());
        if (!opts.weights_path.empty()) std::printf(", weights %ux%u", weights.rows, weights.cols);
        if (!opts.acts_path.empty())    std::printf(", acts %ux%u", acts.rows, acts.cols);
        std::printf("\n");
    }

    // Any illegal encoding is worth reporting now rather than at issue time.
    std::size_t traps = 0;
    for (const RawInst& inst : prog) {
        if (decode(inst).trap) ++traps;
    }
    if (traps != 0) {
        std::fflush(stdout);   // keep the diagnosis after the listing it refers to
        std::fprintf(stderr, "minitpu: %zu illegal instruction(s)\n", traps);
        return 1;
    }

    if (opts.trace) {
        std::fprintf(stderr, "minitpu: --trace has no effect until the model can run\n");
    }
    return 0;
}
#endif
