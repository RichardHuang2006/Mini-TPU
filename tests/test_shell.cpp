/// Shell: every command's exact output, memory views in dec and hex, step back, and errors as text.

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

TEST(shell_step_forward_and_back) {
    Tpu tpu;
    Shell shell(tpu);
    shell.execute("load " + write_temp_program("mini_tpu_shell_step.s", kCopy));

    CHECK_EQ(shell.execute("step"), std::string("cycle 1, pc 1, running"));
    CHECK_EQ(shell.execute("step 2"), std::string("cycle 49, pc 2, halted"));
    CHECK_EQ(shell.execute("step -1"), std::string("cycle 25, pc 2, running"));
    CHECK_EQ(shell.execute("step -5"), std::string("cycle 0, pc 0, ready"));
}

TEST(shell_errors_are_text) {
    Tpu tpu;
    Shell shell(tpu);

    CHECK_EQ(shell.execute("run"), std::string("error: no program loaded; use load FILE"));
    CHECK_EQ(shell.execute("load /nonexistent/prog.s"), std::string("/nonexistent/prog.s: error: cannot open file"));
    CHECK_EQ(shell.execute("foo"),
             std::string("error: unknown command 'foo'; commands: load FILE, run [N], step [N|-N], TARGET [hex|dec], quit"));
    CHECK_EQ(shell.execute("mem[0]"),
             std::string("error: unknown target 'mem' (try pc, ub[ROW:RxC], host[ROW:RxC], wmem[ROW:RxC])"));
    CHECK_EQ(shell.execute("ub[0:4]"), std::string("error: expected ROWSxCOLS after ':', got '4'"));
    CHECK_EQ(shell.execute("ub[0:1x300]"), std::string("error: a view is 1 or more rows of 1 to 256 columns"));
    CHECK_EQ(shell.execute("ub[0x20000]"), std::string("error: ub row 131072 is past the last row 98303"));
    CHECK_EQ(shell.execute("ub[0] oct"), std::string("error: format must be hex or dec, got 'oct'"));
    CHECK_EQ(shell.execute("   "), std::string(""));
}

TEST(shell_quit) {
    Tpu tpu;
    Shell shell(tpu);
    CHECK(!shell.quit_requested());
    CHECK_EQ(shell.execute("quit"), std::string(""));
    CHECK(shell.quit_requested());
}
