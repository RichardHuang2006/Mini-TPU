/// Terminal UI: the screen's exact geometry, what the panels show, the too-small fallback, and plain line mode.

#include <sstream>
#include <string>
#include <vector>

#include "isa/asm.h"
#include "test_framework.h"
#include "ui/tui.h"

namespace {

// Columns a line takes on screen: UTF-8 continuation bytes (10xxxxxx) do not start a new character.
int visible_width(const std::string& line) {
    int columns = 0;
    for (char c : line) {
        const bool continuation = (static_cast<u8>(c) & 0xC0) == 0x80;
        if (!continuation) {
            columns = columns + 1;
        }
    }
    return columns;
}

bool any_line_contains(const std::vector<std::string>& lines, const std::string& text) {
    for (const std::string& line : lines) {
        if (line.find(text) != std::string::npos) {
            return true;
        }
    }
    return false;
}

const char* kReadThenHalt =
    "Read_Host_Memory host=0 ub=0 rows=1\n"
    "Halt\n";

}  // namespace

TEST(tui_screen_is_exactly_the_terminal_size) {
    Tpu tpu;
    tpu.load(assemble(kReadThenHalt));
    tpu.run_to_halt();

    const std::vector<std::string> screen = render_screen(tpu, "copy.s", "hello", "run", 100, 30);
    CHECK_EQ(screen.size(), std::size_t{30});
    for (std::size_t i = 0; i + 1 < screen.size(); ++i) {
        CHECK_EQ(visible_width(screen[i]), 100);
    }
    CHECK_EQ(screen.back(), std::string("tpu> run"));
}

TEST(tui_panels_show_the_machine) {
    Tpu tpu;
    tpu.load(assemble(kReadThenHalt));
    tpu.run_to_halt();

    const std::vector<std::string> screen = render_screen(tpu, "copy.s", "some output", "", 100, 30);
    CHECK(any_line_contains(screen, "Program copy.s"));
    CHECK(any_line_contains(screen, "->   1  Halt"));       // the arrow marks the PC
    CHECK(any_line_contains(screen, " cycle    13"));
    CHECK(any_line_contains(screen, " state    halted"));
    CHECK(any_line_contains(screen, "waiting for units to finish 11"));
    CHECK(any_line_contains(screen, "some output"));
}

TEST(tui_too_small_says_so) {
    const Tpu tpu;
    const std::vector<std::string> screen = render_screen(tpu, "", "", "", 40, 10);
    CHECK_EQ(screen.size(), std::size_t{10});
    CHECK(any_line_contains(screen, "terminal too small: need 60x16, have 40x10"));
}

TEST(tui_plain_mode_echoes_and_stops_at_quit) {
    Tpu tpu;
    Shell shell(tpu);
    std::istringstream in("pc\n\nrun\nquit\nrun\n");
    std::ostringstream out;

    run_plain(shell, in, out);
    const std::string expected =
        "tpu> pc\n"
        "pc 0 (no instruction there)\n"
        "tpu> run\n"
        "error: no program loaded; use load FILE\n"
        "tpu> quit\n";
    CHECK_EQ(out.str(), expected);
}
