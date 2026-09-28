/// tpu_sim: the Mini-TPU terminal. Start it with ./tpu from the repo root; piped input runs in plain line mode.

#include <unistd.h>

#include <iostream>
#include <string>

#include "core/tpu.h"
#include "ui/repl.h"
#include "ui/shell.h"

int main(int argc, char** argv) {
    Tpu tpu;
    Shell shell(tpu);

    std::cout << "Mini-TPU (TPUv1). Commands: load FILE, step [N], back [N], next [N], prev [N], run [N], "
                 "TARGET [hex|dec], quit\n";

    if (argc > 1) {
        const std::string load_command = std::string("load ") + argv[1];
        std::cout << "tpu> " << load_command << "\n" << shell.execute(load_command) << "\n";
    }

    const bool interactive = isatty(STDIN_FILENO) != 0 && isatty(STDOUT_FILENO) != 0;
    if (interactive) {
        run_interactive(shell);
    } else {
        run_plain(shell, std::cin, std::cout);
    }
    return 0;
}
