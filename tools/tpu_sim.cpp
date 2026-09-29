/// tpu_sim: the terminal loads programs, and the browser visualizer steps through them. Start it with ./tpu.

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
#include "viz/state.h"

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
    // Options: --no-open keeps the browser closed; any other argument is the program to load.
    std::string program;
    bool open_it = true;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--no-open") {
            open_it = false;
        } else {
            program = arg;
        }
    }

    Tpu tpu;
    Shell shell(tpu);

    std::cout << "Mini-TPU (TPUv1). Type load FILE, then step through it in the visualizer. quit exits.\n";

    // The page is the only way to move the machine, so there is nothing to do without it.
    std::unique_ptr<VizServer> viz;
    try {
        viz = std::make_unique<VizServer>(MINITPU_VIZ_DIR, 8008);
    } catch (const std::runtime_error& e) {
        std::cout << e.what() << "\n";
        return 1;
    }
    const std::string url = "http://127.0.0.1:" + std::to_string(viz->port()) + "/";
    std::cout << "visualizer: " << url << "\n";

    // A raw pointer, so a press still finishing while viz is destroyed publishes to a live server.
    VizServer* server = viz.get();
    shell.set_observer([server, &tpu]() { server->publish(snapshot_json(tpu), capture(tpu)); });
    viz->on_move([&shell](const std::string& direction, const std::string& unit) { return shell.move(direction, unit); });
    viz->on_jump([&shell](u64 cycle) { return shell.jump(cycle); });
    server->publish(snapshot_json(tpu), capture(tpu));
    if (open_it) {
        open_browser(url);
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
