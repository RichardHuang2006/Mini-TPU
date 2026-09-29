/// Command parsing for the terminal, the buttons' moves, and the text each one returns.

#include "ui/shell.h"

#include <sstream>
#include <stdexcept>
#include <utility>

#include "isa/asm.h"

namespace {

std::vector<std::string> split_words(const std::string& line) {
    std::istringstream stream(line);
    std::vector<std::string> words;
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    return words;
}

}  // namespace

std::string machine_state(const Tpu& tpu) {
    if (tpu.halted()) {
        return "halted";
    }
    if (tpu.cycle() == 0) {
        return "ready";
    }
    if (tpu.stall() == Stall::None) {
        return "running";
    }
    return std::string("stalled: ") + stall_name(tpu.stall());
}

std::string status_line(const Tpu& tpu) {
    return "cycle " + std::to_string(tpu.cycle()) + ", pc " + std::to_string(tpu.pc()) + ", " + machine_state(tpu);
}

Shell::Shell(Tpu& tpu) : tpu_(tpu) {}

std::string Shell::execute(const std::string& line) {
    const std::lock_guard<std::mutex> lock(mutex_);
    const std::string output = run_command(line);
    if (observer_) {
        observer_();
    }
    return output;
}

std::string Shell::move(const std::string& direction, const std::string& unit) {
    return press([&]() { return move_once(direction, unit); });
}

std::string Shell::jump(Cycle cycle) {
    return press([&]() { return jump_to(cycle); });
}

// A page button's work: under the lock, with an exception turned into "error: ..." text, then the observer told.
std::string Shell::press(const std::function<std::string()>& work) {
    const std::lock_guard<std::mutex> lock(mutex_);
    std::string output;
    try {
        output = work();
    } catch (const std::exception& e) {
        output = std::string("error: ") + e.what();
    }
    if (observer_) {
        observer_();
    }
    return output;
}

bool Shell::quit_requested() const {
    return quit_;
}

void Shell::set_observer(std::function<void()> observer) {
    const std::lock_guard<std::mutex> lock(mutex_);
    observer_ = std::move(observer);
}

std::string Shell::run_command(const std::string& line) {
    const std::vector<std::string> words = split_words(line);
    if (words.empty()) {
        return "";
    }

    try {
        const std::string& command = words[0];
        if (command == "quit") {
            quit_ = true;
            return "";
        }
        if (command == "load") {
            return load(words);
        }
        throw std::invalid_argument("unknown command '" + command +
                                    "': the terminal takes load FILE and quit; step through the program in the visualizer");
    } catch (const AsmError& e) {
        return e.what();   // already reads "file:line:col: error: ..."
    } catch (const std::exception& e) {
        return std::string("error: ") + e.what();
    }
}

std::string Shell::load(const std::vector<std::string>& words) {
    if (words.size() != 2) {
        throw std::invalid_argument("usage: load FILE");
    }
    const std::string& path = words[1];
    const Program program = assemble_file(path);
    tpu_.load(program);

    u64 host_bytes = 0;
    for (const DataBlock& block : program.host) {
        host_bytes += block.bytes.size();
    }
    u64 weight_bytes = 0;
    for (const DataBlock& block : program.weights) {
        weight_bytes += block.bytes.size();
    }

    return "loaded " + path + ": " + std::to_string(program.code.size()) + " instructions, " +
           std::to_string(host_bytes) + " host bytes, " + std::to_string(weight_bytes) + " weight bytes";
}

// Forward runs one cycle or until one more instruction issues; back reloads and replays to one before.
std::string Shell::move_once(const std::string& direction, const std::string& unit) {
    require_program();
    const bool forward = direction == "forward";
    if (!forward && direction != "back") {
        throw std::invalid_argument("direction must be forward or back, got '" + direction + "'");
    }

    if (unit == "cycle") {
        if (forward) {
            tpu_.run_cycles(1);
        } else {
            tpu_.back_cycles(1);
        }
    } else if (unit == "instruction") {
        if (forward) {
            tpu_.next_instructions(1);
        } else {
            tpu_.prev_instructions(1);
        }
    } else {
        throw std::invalid_argument("unit must be cycle or instruction, got '" + unit + "'");
    }
    return status_line(tpu_);
}

// Forward runs the cycles in between and stops early at Halt; back reloads and replays.
std::string Shell::jump_to(Cycle cycle) {
    require_program();
    if (cycle > tpu_.cycle()) {
        tpu_.run_cycles(cycle - tpu_.cycle());
    } else if (cycle < tpu_.cycle()) {
        tpu_.back_cycles(tpu_.cycle() - cycle);
    }
    return status_line(tpu_);
}

void Shell::require_program() const {
    if (tpu_.program().code.empty()) {
        throw std::invalid_argument("no program loaded; use load FILE");
    }
}
