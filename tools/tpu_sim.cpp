/// tpu_sim: the Mini-TPU terminal, plus a live browser visualizer. Start it with ./tpu; piped input runs in plain line mode.

#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "core/tpu.h"
#include "ui/repl.h"
#include "ui/shell.h"
#include "viz/server.h"
#include "viz/snapshot.h"

namespace {

// Hands the page to the system's browser; a failure only means the user opens the address themselves.
void open_browser(const std::string& url) {
#ifdef __APPLE__
    const std::string command = "open '" + url + "' >/dev/null 2>&1";
#else
    const std::string command = "xdg-open '" + url + "' >/dev/null 2>&1 &";
#endif
    const int ignored = std::system(command.c_str());
    (void)ignored;
}

}  // namespace

int main(int argc, char** argv) {
    // Options: --viz / --no-viz turn the visualizer on or off (on when interactive), --no-open keeps the browser closed.
    std::string program;
    bool viz_on  = isatty(STDIN_FILENO) != 0 && isatty(STDOUT_FILENO) != 0;
    bool open_it = true;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--viz") {
            viz_on = true;
        } else if (arg == "--no-viz") {
            viz_on = false;
        } else if (arg == "--no-open") {
            open_it = false;
        } else {
            program = arg;
        }
    }

    Tpu tpu;
    Shell shell(tpu);

    std::cout << "Mini-TPU (TPUv1). Commands: load FILE, step [N], back [N], next [N], prev [N], run [N], "
                 "TARGET [hex|dec], quit\n";

    std::unique_ptr<VizServer> viz;
    if (viz_on) {
        try {
            viz = std::make_unique<VizServer>(MINITPU_VIZ_DIR, 8008);
            const std::string url = "http://127.0.0.1:" + std::to_string(viz->port()) + "/";
            std::cout << "visualizer: " << url << "\n";

            shell.set_observer([&viz, &tpu, &shell]() { viz->publish(snapshot_json(tpu, shell)); });
            viz->publish(snapshot_json(tpu, shell));
            if (open_it) {
                open_browser(url);
            }
        } catch (const std::runtime_error& e) {
            std::cout << "visualizer off: " << e.what() << "\n";
        }
    }

    if (!program.empty()) {
        const std::string load_command = "load " + program;
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
