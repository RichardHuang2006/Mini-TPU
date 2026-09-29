/// The command shell: runs one line (load, run, step, a state target, quit) and returns the text to show.

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "core/tpu.h"

// Where the machine is, in one line, e.g. "cycle 25, pc 2, halted".
std::string status_line(const Tpu& tpu);

// What the last cycle did: "ready", "running", "halted" or "stalled: <cause>".
std::string machine_state(const Tpu& tpu);

// How full the Weight FIFO is: "empty" or e.g. "2 of 4 tiles".
std::string fifo_occupancy(const WeightFifo& fifo);

// The tile arriving now, e.g. "tile 0x1, 1200 of 65536 bytes", or "idle".
std::string fifo_fetch(const WeightFifo& fifo);

// A target the user looked at, and its current text; the visualizer keeps showing it.
struct PinnedView {
    std::string target;
    std::string text;
};

class Shell {
public:
    explicit Shell(Tpu& tpu);

    // Runs one command and returns its output; errors come back as "error: ..." text, never as exceptions.
    std::string execute(const std::string& line);

    bool quit_requested() const;

    // Called after every command, and about 20 times a second during `run`, so the visualizer can redraw.
    void set_observer(std::function<void()> observer);

    // The last 4 targets looked at, each shown as it reads now.
    std::vector<PinnedView> pinned_views() const;

    // The path given to the last successful load, or "" before any load.
    const std::string& loaded_file() const;

private:
    Tpu&        tpu_;
    bool        quit_ = false;
    std::string loaded_file_;
    std::function<void()>    observer_;
    std::vector<std::string> pinned_;

    std::string run_command(const std::string& line);
    void        run_animated(u64 cycles);
    void        pin(const std::vector<std::string>& words);

    std::string load(const std::vector<std::string>& words);
    std::string run(const std::vector<std::string>& words);
    std::string move(const std::vector<std::string>& words);
    std::string show(const std::vector<std::string>& words) const;
    std::string show_pc() const;
    std::string show_fifo() const;
    std::string show_mxu() const;
    std::string show_activation() const;
    std::string show_acc(u64 first_row, u64 rows, u64 cols, bool as_hex) const;
    void        require_program() const;
};
