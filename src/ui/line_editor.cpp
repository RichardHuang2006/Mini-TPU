/// The line editor's history, and reading keys with the terminal in raw mode.

#include "ui/line_editor.h"

#include <termios.h>
#include <unistd.h>

#include <cstdio>

const std::string& LineEditor::text() const {
    return text_;
}

void LineEditor::type(char c) {
    text_ += c;
}

void LineEditor::backspace() {
    if (!text_.empty()) {
        text_.pop_back();
    }
}

void LineEditor::older() {
    if (position_ > 0) {
        position_ = position_ - 1;
        text_ = history_[position_];
    }
}

void LineEditor::newer() {
    if (position_ < history_.size()) {
        position_ = position_ + 1;
    }
    if (position_ == history_.size()) {
        text_.clear();
    } else {
        text_ = history_[position_];
    }
}

std::string LineEditor::take() {
    std::string line = text_;
    text_.clear();
    return line;
}

void LineEditor::remember(const std::string& command) {
    const bool same_as_newest = !history_.empty() && history_.back() == command;
    if (!same_as_newest) {
        history_.push_back(command);
    }
    position_ = history_.size();
}

namespace {

// Raw mode while one line is typed; the terminal is put back when this goes out of scope.
class RawMode {
public:
    RawMode() {
        tcgetattr(STDIN_FILENO, &original_);
        termios raw = original_;
        const tcflag_t local_off = ECHO | ICANON | ISIG | IEXTEN;   // no echo, no line buffering, Ctrl-C arrives as a key
        const tcflag_t input_off = IXON | ICRNL;                    // Ctrl-S/Ctrl-Q and Enter arrive as keys too
        raw.c_lflag = raw.c_lflag & ~local_off;
        raw.c_iflag = raw.c_iflag & ~input_off;
        raw.c_cc[VMIN]  = 1;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    }

    ~RawMode() {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &original_);
    }

    RawMode(const RawMode&) = delete;
    RawMode& operator=(const RawMode&) = delete;

private:
    termios original_{};
};

void write_text(const std::string& text) {
    std::fwrite(text.data(), 1, text.size(), stdout);
    std::fflush(stdout);
}

// Rewrites the current terminal line as the prompt followed by the typed text.
void redraw(const std::string& prompt, const std::string& text) {
    write_text("\r" + prompt + text + "\x1b[K");
}

enum class Key { Char, Enter, Backspace, Up, Down, Quit, Other };

Key read_key(char& c) {
    if (read(STDIN_FILENO, &c, 1) != 1) {
        return Key::Quit;
    }

    const char ctrl_c = 3;
    const char ctrl_d = 4;
    const char escape = 27;
    if (c == ctrl_c || c == ctrl_d) {
        return Key::Quit;
    }
    if (c == '\r' || c == '\n') {
        return Key::Enter;
    }
    if (c == 127 || c == 8) {
        return Key::Backspace;
    }

    // Arrow keys arrive as ESC [ A (up) and ESC [ B (down).
    if (c == escape) {
        char sequence[2] = {0, 0};
        if (read(STDIN_FILENO, &sequence[0], 1) != 1 || read(STDIN_FILENO, &sequence[1], 1) != 1) {
            return Key::Other;
        }
        if (sequence[0] == '[' && sequence[1] == 'A') {
            return Key::Up;
        }
        if (sequence[0] == '[' && sequence[1] == 'B') {
            return Key::Down;
        }
        return Key::Other;
    }

    const bool printable = c >= 32 && c < 127;
    if (printable) {
        return Key::Char;
    }
    return Key::Other;
}

}  // namespace

bool read_line(LineEditor& editor, const std::string& prompt, const std::string& repeat, std::string& line) {
    const RawMode raw_mode;
    redraw(prompt, editor.text());

    while (true) {
        char c = 0;
        const Key key = read_key(c);

        if (key == Key::Quit) {
            write_text("\r\n");
            return false;
        }
        if (key == Key::Char) {
            editor.type(c);
        } else if (key == Key::Backspace) {
            editor.backspace();
        } else if (key == Key::Up) {
            editor.older();
        } else if (key == Key::Down) {
            editor.newer();
        } else if (key == Key::Enter) {
            line = editor.take();
            if (line.empty()) {
                line = repeat;   // Enter on an empty line repeats the last command
            }
            redraw(prompt, line);
            write_text("\r\n");
            return true;
        }
        redraw(prompt, editor.text());
    }
}
