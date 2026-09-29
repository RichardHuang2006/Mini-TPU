/// Shell: every command's exact output, memory views in dec and hex, moving by cycles and instructions, and errors as text.

#include <filesystem>
#include <fstream>
#include <string>

#include "test_framework.h"
#include "ui/shell.h"

namespace {

// Two rows in, two rows out: the read takes 24 cycles, the write 24 more, and Halt issues on cycle 48.
const char* kCopy =
    ".host 0\n"
    "1 2 3 4\n"
    ".host 1\n"
    "-1 -2 -3 -4\n"
    "Read_Host_Memory host=0 ub=0 rows=2\n"
    "Write_Host_Memory ub=0 host=4 rows=2\n"
    "Halt\n";

std::string write_temp_program(const std::string& name, const std::string& source) {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
    std::ofstream file(path);
    file << source;
    return path.string();
}

}  // namespace

TEST(shell_load_and_run) {
    Tpu tpu;
    Shell shell(tpu);
    const std::string path = write_temp_program("mini_tpu_shell_copy.s", kCopy);

    CHECK_EQ(shell.execute("load " + path), "loaded " + path + ": 3 instructions, 8 host bytes, 0 weight bytes");
    CHECK_EQ(shell.loaded_file(), path);
    CHECK_EQ(shell.execute("run 5"), std::string("cycle 5, pc 1, stalled: host interface busy"));
    CHECK_EQ(shell.execute("run"), std::string("cycle 49, pc 2, halted"));
    CHECK_EQ(shell.execute("pc"), std::string("pc 2: Halt"));
}

TEST(shell_memory_views) {
    Tpu tpu;
    Shell shell(tpu);
    shell.execute("load " + write_temp_program("mini_tpu_shell_views.s", kCopy));
    shell.execute("run");

    CHECK_EQ(shell.execute("ub[0:2x4]"), std::string("0x0:    1    2    3    4\n0x1:   -1   -2   -3   -4"));
    CHECK_EQ(shell.execute("ub[1:1x4] hex"), std::string("0x1: FF FE FD FC"));
    CHECK_EQ(shell.execute("host[4:2x2]"), std::string("0x4:    1    2\n0x5:   -1   -2"));
    CHECK_EQ(shell.execute("wmem[0:1x2]"), std::string("0x0:    0    0"));

    const std::string one_row = shell.execute("ub[0x0]");   // a bare row shows 16 columns
    CHECK_EQ(one_row.size(), std::string("0x0:").size() + 16 * 5);
}

TEST(shell_moves_by_cycles_and_instructions) {
    Tpu tpu;
    Shell shell(tpu);
    shell.execute("load " + write_temp_program("mini_tpu_shell_move.s", kCopy));

    // step and back count cycles.
    CHECK_EQ(shell.execute("step"), std::string("cycle 1, pc 1, running"));
    CHECK_EQ(shell.execute("step 4"), std::string("cycle 5, pc 1, stalled: host interface busy"));

    // next and prev count instruction issues; the write issues on cycle 24.
    CHECK_EQ(shell.execute("next"), std::string("cycle 25, pc 2, running"));
    CHECK_EQ(shell.execute("prev"), std::string("cycle 1, pc 1, running"));
    CHECK_EQ(shell.execute("next 2"), std::string("cycle 49, pc 2, halted"));

    CHECK_EQ(shell.execute("back"), std::string("cycle 48, pc 2, stalled: waiting for units to finish"));
    CHECK_EQ(shell.execute("back 100"), std::string("cycle 0, pc 0, ready"));

    CHECK_EQ(shell.execute("step x"), std::string("error: count 'x' is not a number"));
    CHECK_EQ(shell.execute("back 1 2"), std::string("error: usage: back [N]"));
}

TEST(shell_errors_are_text) {
    Tpu tpu;
    Shell shell(tpu);

    CHECK_EQ(shell.execute("run"), std::string("error: no program loaded; use load FILE"));
    CHECK_EQ(shell.execute("load /nonexistent/prog.s"), std::string("/nonexistent/prog.s: error: cannot open file"));
    CHECK_EQ(shell.execute("foo"),
             std::string("error: unknown command 'foo'; commands: load FILE, step [N], back [N], next [N], prev [N], run [N], "
                         "TARGET [hex|dec], quit"));
    CHECK_EQ(shell.execute("mem[0]"),
             std::string("error: unknown target 'mem' (try pc, wfifo, mxu, ub[ROW:RxC], acc[ROW:RxC], host[ROW:RxC], wmem[ROW:RxC])"));
    CHECK_EQ(shell.execute("ub[0:4]"), std::string("error: expected ROWSxCOLS after ':', got '4'"));
    CHECK_EQ(shell.execute("ub[0:1x300]"), std::string("error: a view is 1 or more rows of 1 to 256 columns"));
    CHECK_EQ(shell.execute("ub[0x20000]"), std::string("error: ub row 131072 is past the last row 98303"));
    CHECK_EQ(shell.execute("ub[0] oct"), std::string("error: format must be hex or dec, got 'oct'"));
    CHECK_EQ(shell.execute("   "), std::string(""));
}

TEST(shell_weight_fifo_view) {
    Tpu tpu;
    Shell shell(tpu);
    shell.execute("load " + write_temp_program("mini_tpu_shell_fifo.s", "Read_Weights tile=0\nRead_Weights tile=1\nHalt\n"));
    CHECK_EQ(shell.execute("wfifo"), std::string("weight FIFO: 0 of 4 slots"));

    shell.execute("next 2");   // both issue on cycles 0 and 1; tile 0 has had two 48-byte cycles
    const std::string expected =
        "weight FIFO: 2 of 4 slots\n"
        "slot 0: tile 0x0, 96 of 65536 bytes\n"
        "slot 1: tile 0x1, 0 of 65536 bytes";
    CHECK_EQ(shell.execute("wfifo"), expected);

    shell.execute("run");
    CHECK_EQ(shell.execute("wfifo"), std::string("weight FIFO: 1 of 4 slots\nslot 0: tile 0x1, ready"));   // tile 0 went to the MXU
}

TEST(shell_accumulator_and_mxu_views) {
    const char* source =
        ".host 0\n"
        "1 2\n"
        ".weights 0 0\n"
        "3 -4\n"
        "Read_Host_Memory host=0 ub=0 rows=1\n"
        "Read_Weights tile=0\n"
        "MatrixMultiply ub=0 acc=0 rows=1 accumulate=0 new_weights=1\n"
        "Halt\n";
    Tpu tpu;
    Shell shell(tpu);
    shell.execute("load " + write_temp_program("mini_tpu_shell_mxu.s", source));
    CHECK_EQ(shell.execute("mxu").substr(0, 9), std::string("MXU: idle"));

    shell.execute("run");
    CHECK_EQ(shell.execute("acc[0:1x3]"), std::string("0x0:        3       -4        0"));   // x = (1, 2), row 0 of W = (3, -4)
    CHECK_EQ(shell.execute("acc[0:1x2] hex"), std::string("0x0: 00000003 FFFFFFFC"));
    CHECK_EQ(shell.execute("acc[4096]"), std::string("error: acc row 4096 is past the last row 4095"));

    const std::string mxu = shell.execute("mxu");
    CHECK_EQ(mxu.substr(0, mxu.find("PEs")),
             std::string("MXU: idle\nactive weights: tile 0x0\nshadow weights: empty\n"));
}

TEST(shell_quit) {
    Tpu tpu;
    Shell shell(tpu);
    CHECK(!shell.quit_requested());
    CHECK_EQ(shell.execute("quit"), std::string(""));
    CHECK(shell.quit_requested());
}
