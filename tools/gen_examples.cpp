// Writes the bundled workloads to examples/ so the CLI has something to run
// without the test binary.
//
// The programs and tensors come from wl::corpus(), the same definitions the test
// suite runs against golden loops, so an example file cannot drift away from what
// was verified. Nothing here computes a layer or emits an instruction; this is a
// writer.

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "loader.h"
#include "workloads.h"

namespace {

bool write_program(const wl::Workload& w, const std::string& dir) {
    const std::string path = dir + "/" + w.name + ".hex";
    std::ofstream out(path);
    if (!out) {
        std::fprintf(stderr, "gen_examples: cannot write %s\n", path.c_str());
        return false;
    }

    out << "# " << w.name << " -- " << w.note << "\n";
    out << "# " << w.code.size() << " instructions, " << ISA_WORDS << " words each\n";
    out << "# run: minitpu --run --prog examples/" << w.name << ".hex"
        << " --acts examples/" << w.name << ".acts.mtpu"
        << " --weights examples/" << w.name << ".weights.mtpu"
        << " --dim " << w.cfg.dim << " --ub " << w.cfg.ub_bytes << " --acc-banks "
        << w.cfg.acc_banks << " --macs " << w.macs << "\n";
    out << "# expect examples/" << w.name << ".expect.mtpu (" << w.out_rows << "x"
        << w.out_cols << " int8) at host 0x" << std::hex << w.y_host << std::dec << "\n";

    save_program_hex(out, w.code);
    if (!out) {
        std::fprintf(stderr, "gen_examples: write failed for %s\n", path.c_str());
        return false;
    }
    return true;
}

bool write_tensor(const std::string& path, const std::vector<i8>& data, uint32_t rows,
                  uint32_t cols) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "gen_examples: cannot write %s\n", path.c_str());
        return false;
    }
    TensorBlob t;
    t.rows = rows;
    t.cols = cols;
    t.wide = false;
    t.i8v  = data;
    save_tensor_mtpu(out, t);
    if (!out) {
        std::fprintf(stderr, "gen_examples: write failed for %s\n", path.c_str());
        return false;
    }
    return true;
}

// Packed tensors are shipped as a flat row of bytes: the shape that matters is the
// tiling the program was lowered for, which the program already encodes. Recording
// them as 1 x N keeps the container honest rather than implying a 2D shape the
// bytes are not in.
bool write_workload(const wl::Workload& w, const std::string& dir) {
    const std::string base = dir + "/" + w.name;
    return write_program(w, dir) &&
           write_tensor(base + ".acts.mtpu", w.a, 1,
                        static_cast<uint32_t>(w.a.size())) &&
           write_tensor(base + ".weights.mtpu", w.b, 1,
                        static_cast<uint32_t>(w.b.size())) &&
           write_tensor(base + ".expect.mtpu", w.expect, w.out_rows, w.out_cols);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "examples";

    const std::vector<wl::Workload> all = wl::corpus();
    for (const wl::Workload& w : all) {
        if (!write_workload(w, dir)) return 1;
        std::printf("examples/%-14s  %3zu instructions, %7zu MACs, %5zu B acts, %6zu B weights\n",
                    (w.name + ".hex").c_str(), w.code.size(), w.macs, w.a.size(), w.b.size());
    }
    std::printf("wrote %zu workload(s) to %s/\n", all.size(), dir.c_str());
    return 0;
}
