/// Plain line mode: echoes each command, skips blank lines, and stops at quit.

#include <sstream>
#include <string>

#include "test_framework.h"
#include "ui/repl.h"

TEST(repl_plain_mode_echoes_and_stops_at_quit) {
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
