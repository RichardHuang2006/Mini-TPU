/// The prompt line: typing, backspace, up/down history, and Enter repeating the last command.

#pragma once

#include <string>
#include <vector>

// The line being typed and the commands already run; no terminal code, so it can be tested directly.
class LineEditor {
public:
    const std::string& text() const;

    void type(char c);
    void backspace();

    // Up arrow: the previous command in history.
    void older();

    // Down arrow: the next command in history, or an empty line after the newest.
    void newer();

    // Returns the typed line and clears it.
    std::string take();

    // Adds a command to history, unless it repeats the newest entry.
    void remember(const std::string& command);

private:
    std::string              text_;
    std::vector<std::string> history_;
    std::size_t              position_ = 0;   // history_.size() means the fresh line
};

// Reads one line at `prompt` in raw mode (Enter on an empty line gives `repeat`); false on Ctrl-C, Ctrl-D or end of input.
bool read_line(LineEditor& editor, const std::string& prompt, const std::string& repeat, std::string& line);
