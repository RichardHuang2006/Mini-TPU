/// The command shell: runs one line (load, run, step, a state target, quit) and returns the text to show.

#pragma once

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

    // Runs one command and returns its output; errors come back as "error: ..." text, never as exceptions.
    std::string execute(const std::string& line);

    bool quit_requested() const;

    // The path given to the last successful load, or "" before any load.
    const std::string& loaded_file() const;

private:
    Tpu&        tpu_;
    bool        quit_ = false;
    std::string loaded_file_;

    std::string load(const std::vector<std::string>& words);
    std::string run(const std::vector<std::string>& words);
    std::string step(const std::vector<std::string>& words);
    std::string show(const std::vector<std::string>& words) const;
    std::string show_pc() const;
    void        require_program() const;
};
