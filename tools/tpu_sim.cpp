/// tpu_sim: the Mini-TPU terminal. `tpu_sim [program.s]`; piped input runs in plain line mode.

#include <unistd.h>

#include <iostream>
#include <string>

#include "core/tpu.h"
#include "ui/shell.h"
#include "ui/tui.h"

int main(int argc, char** argv) {
    Tpu tpu;
    Shell shell(tpu);

    std::string greeting = "Mini-TPU (TPUv1). Commands: load FILE, run [N], step [N|-N], TARGET [hex|dec], quit";
    if (argc > 1) {
        const std::string load_command = std::string("load ") + argv[1];
        greeting = "tpu> " + load_command + "\n" + shell.execute(load_command);
    }

    const bool interactive = isatty(STDIN_FILENO) != 0 && isatty(STDOUT_FILENO) != 0;
    if (interactive) {
        run_interactive(shell, tpu, greeting);
    } else {
        std::cout << greeting << "\n";
        run_plain(shell, std::cin, std::cout);
    }
    return 0;
}
