/// Shell: the terminal's load and quit, button moves by cycle and by instruction, jumps to a cycle, and errors as text.

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

TEST(shell_loads_a_program) {
    Tpu tpu;
    Shell shell(tpu);
    const std::string path = write_temp_program("mini_tpu_shell_copy.s", kCopy);

    CHECK_EQ(shell.execute("load " + path), "loaded " + path + ": 3 instructions, 8 host bytes, 0 weight bytes");
    CHECK_EQ(tpu.program().code.size(), std::size_t{3});
    CHECK_EQ(shell.execute("load /nonexistent/prog.s"), std::string("/nonexistent/prog.s: error: cannot open file"));
    CHECK_EQ(shell.execute("load"), std::string("error: usage: load FILE"));
}

TEST(shell_terminal_takes_only_load_and_quit) {
    Tpu tpu;
    Shell shell(tpu);
    CHECK_EQ(shell.execute("step"),
             std::string("error: unknown command 'step': the terminal takes load FILE and quit; "
                         "step through the program in the visualizer"));
    CHECK_EQ(shell.execute("   "), std::string(""));

    CHECK(!shell.quit_requested());
    CHECK_EQ(shell.execute("quit"), std::string(""));
    CHECK(shell.quit_requested());
}

TEST(shell_moves_by_cycles_and_instructions) {
    Tpu tpu;
    Shell shell(tpu);
    shell.execute("load " + write_temp_program("mini_tpu_shell_move.s", kCopy));

    CHECK_EQ(shell.move("forward", "cycle"), std::string("cycle 1, pc 1, running"));
    CHECK_EQ(shell.move("back", "cycle"), std::string("cycle 0, pc 0, ready"));
    CHECK_EQ(shell.move("back", "cycle"), std::string("cycle 0, pc 0, ready"));   // cycle 0 is as far back as it goes

    // An instruction move runs until one more issues; the read issues on cycle 0 and the write on cycle 24.
    CHECK_EQ(shell.move("forward", "instruction"), std::string("cycle 1, pc 1, running"));
    CHECK_EQ(shell.move("forward", "instruction"), std::string("cycle 25, pc 2, running"));
    CHECK_EQ(shell.move("back", "instruction"), std::string("cycle 1, pc 1, running"));

    CHECK_EQ(shell.move("forward", "instruction"), std::string("cycle 25, pc 2, running"));
    CHECK_EQ(shell.move("forward", "instruction"), std::string("cycle 49, pc 2, halted"));
    CHECK_EQ(shell.move("forward", "cycle"), std::string("cycle 49, pc 2, halted"));   // nothing runs past Halt
    CHECK_EQ(shell.move("back", "cycle"), std::string("cycle 48, pc 2, stalled: waiting for units to finish"));
}

TEST(shell_jumps_to_a_cycle) {
    Tpu tpu;
    Shell shell(tpu);
    CHECK_EQ(shell.jump(5), std::string("error: no program loaded; use load FILE"));

    shell.execute("load " + write_temp_program("mini_tpu_shell_jump.s", kCopy));
    CHECK_EQ(shell.jump(25), std::string("cycle 25, pc 2, running"));                        // forward
    CHECK_EQ(shell.jump(5), std::string("cycle 5, pc 1, stalled: host interface busy"));     // back, by replay
    CHECK_EQ(shell.jump(5), std::string("cycle 5, pc 1, stalled: host interface busy"));     // already there
    CHECK_EQ(shell.jump(1000), std::string("cycle 49, pc 2, halted"));                       // forward stops at Halt
    CHECK_EQ(shell.jump(0), std::string("cycle 0, pc 0, ready"));
}

TEST(shell_move_errors_are_text) {
    Tpu tpu;
    Shell shell(tpu);
    CHECK_EQ(shell.move("forward", "cycle"), std::string("error: no program loaded; use load FILE"));

    shell.execute("load " + write_temp_program("mini_tpu_shell_errors.s", kCopy));
    CHECK_EQ(shell.move("up", "cycle"), std::string("error: direction must be forward or back, got 'up'"));
    CHECK_EQ(shell.move("forward", "bundle"), std::string("error: unit must be cycle or instruction, got 'bundle'"));
    CHECK_EQ(tpu.cycle(), Cycle{0});
}
