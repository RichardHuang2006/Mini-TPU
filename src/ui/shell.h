/// The command shell: the terminal loads programs, and the visualizer's buttons move them forward, back or to a cycle.

#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "core/tpu.h"

// Where the machine is, in one line, e.g. "cycle 25, pc 2, halted".
std::string status_line(const Tpu& tpu);

// What the last cycle did: "ready", "running", "halted" or "stalled: <cause>".
std::string machine_state(const Tpu& tpu);

class Shell {
public:
    explicit Shell(Tpu& tpu);

    // Runs one terminal line, load FILE or quit, and returns the text to show; errors come back as "error: ..." text.
    std::string execute(const std::string& line);

    // One button press: direction is forward or back, unit is cycle or instruction; returns the status line or "error: ...".
    std::string move(const std::string& direction, const std::string& unit);

    // Goes to cycle N, forward (stopping at Halt) or back from wherever the machine is; returns the status line or "error: ...".
    std::string jump(Cycle cycle);

    bool quit_requested() const;

    // Called after every line and every press, so the visualizer can redraw.
    void set_observer(std::function<void()> observer);

private:
    std::mutex            mutex_;   // the terminal and the page's buttons run on different threads; one command at a time
    Tpu&                  tpu_;
    bool                  quit_ = false;
    std::function<void()> observer_;

    std::string run_command(const std::string& line);
    std::string press(const std::function<std::string()>& work);
    std::string move_once(const std::string& direction, const std::string& unit);
    std::string jump_to(Cycle cycle);
    std::string load(const std::vector<std::string>& words);
    void        require_program() const;
};
