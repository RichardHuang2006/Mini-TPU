/// The interactive and piped command loops.

#include "ui/repl.h"

#include <iostream>
#include <istream>
#include <ostream>

#include "ui/line_editor.h"

namespace {

const char* kPrompt = "tpu> ";

}  // namespace

void run_interactive(Shell& shell) {
    LineEditor editor;
    std::string last_command;

    while (!shell.quit_requested()) {
        std::string command;
        if (!read_line(editor, kPrompt, last_command, command)) {
            return;
        }
        if (command.empty()) {
            continue;
        }

        editor.remember(command);
        last_command = command;

        const std::string output = shell.execute(command);
        if (!output.empty()) {
            std::cout << output << "\n";
        }
    }
}

void run_plain(Shell& shell, std::istream& in, std::ostream& out) {
    std::string line;
    while (!shell.quit_requested() && std::getline(in, line)) {
        const bool blank = line.find_first_not_of(" \t\r") == std::string::npos;
        if (blank) {
            continue;
        }
        out << kPrompt << line << "\n";
        const std::string result = shell.execute(line);
        if (!result.empty()) {
            out << result << "\n";
        }
    }
}
