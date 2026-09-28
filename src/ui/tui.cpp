/// Drawing the panels, reading keys in raw mode, and the line editor with history.

#include "ui/tui.h"

#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <istream>
#include <ostream>
#include <sstream>

namespace {

const char* kPrompt = "tpu> ";

// ---------------------------------------------------------------- drawing

std::string repeat(const std::string& piece, int count) {
    std::string text;
    for (int i = 0; i < count; ++i) {
        text += piece;
    }
    return text;
}

// Cuts or pads plain ASCII text to exactly `width` columns.
std::string fit(const std::string& text, int width) {
    if (width <= 0) {
        return "";
    }
    const std::size_t columns = static_cast<std::size_t>(width);
    if (text.size() >= columns) {
        return text.substr(0, columns);
    }
    return text + std::string(columns - text.size(), ' ');
}

// A bordered panel: exactly `height` lines, each exactly `width` columns.
std::vector<std::string> box(const std::string& title, const std::vector<std::string>& body, int width, int height) {
    std::vector<std::string> lines;

    // Top border: ┌─ Title ───┐
    const std::string label = fit(title, width - 5);
    const int dashes = width - 5 - static_cast<int>(label.size());
    lines.push_back("┌─ " + label + " " + repeat("─", dashes) + "┐");

    const int body_rows = height - 2;
    for (int i = 0; i < body_rows; ++i) {
        std::string text;
        if (i < static_cast<int>(body.size())) {
            text = body[static_cast<std::size_t>(i)];
        }
        lines.push_back("│" + fit(text, width - 2) + "│");
    }

    lines.push_back("└" + repeat("─", width - 2) + "┘");
    return lines;
}

std::string describe(const InstrBytes& bytes) {
    try {
        return disasm(decode(bytes));
    } catch (const std::exception&) {
        return "(unknown opcode)";
    }
}

// The listing around the PC, with an arrow on the instruction at the PC.
std::vector<std::string> program_lines(const Tpu& tpu, int rows) {
    const std::vector<InstrBytes>& code = tpu.program().code;
    if (code.empty()) {
        return {"no program loaded", "", "try: load programs/copy.s"};
    }

    const int count = static_cast<int>(code.size());
    const int pc    = static_cast<int>(tpu.pc());

    // Start half a panel above the PC, so it stays in view as the program runs.
    int first = pc - rows / 2;
    if (first < 0) {
        first = 0;
    }

    std::vector<std::string> lines;
    for (int i = first; i < count && i < first + rows; ++i) {
        std::string marker = "   ";
        if (i == pc) {
            marker = "-> ";
        }
        char number[16];
        std::snprintf(number, sizeof number, "%3d  ", i);
        lines.push_back(marker + number + describe(code[static_cast<std::size_t>(i)]));
    }
    return lines;
}

std::string field(const std::string& name, const std::string& value) {
    return " " + fit(name, 9) + value;
}

std::vector<std::string> machine_lines(const Tpu& tpu) {
    std::vector<std::string> lines;
    lines.push_back(field("cycle", std::to_string(tpu.cycle())));
    lines.push_back(field("issued", std::to_string(tpu.issued())));
    lines.push_back(field("pc", std::to_string(tpu.pc())));
    lines.push_back(field("state", machine_state(tpu)));

    const HostInterface& link = tpu.host_interface();
    if (link.busy()) {
        const std::string progress = std::to_string(link.bytes_done()) + " of " + std::to_string(link.bytes_total());
        lines.push_back(field("host i/f", progress + " bytes moved"));
    } else {
        lines.push_back(field("host i/f", "idle"));
    }

    lines.push_back("");
    lines.push_back(" stall cycles");
    const Stall kinds[] = {Stall::HostInterfaceBusy, Stall::WaitForIdle};
    for (Stall kind : kinds) {
        const std::string count = std::to_string(tpu.stats().stalled(kind));
        lines.push_back("   " + fit(stall_name(kind), 28) + count);
    }
    return lines;
}

// The last `rows` lines of the output text.
std::vector<std::string> last_lines(const std::string& text, int rows) {
    std::vector<std::string> all;
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
        all.push_back(line);
    }

    std::size_t first = 0;
    const std::size_t wanted = static_cast<std::size_t>(rows);
    if (all.size() > wanted) {
        first = all.size() - wanted;
    }
    return std::vector<std::string>(all.begin() + static_cast<long>(first), all.end());
}

// ---------------------------------------------------------------- terminal

// Puts the terminal in raw mode on an alternate screen, and puts both back when destroyed.
class RawTerminal {
public:
    RawTerminal() {
        tcgetattr(STDIN_FILENO, &original_);

        termios raw = original_;
        const tcflag_t local_off = ECHO | ICANON | ISIG | IEXTEN;   // no echo, no line buffering, Ctrl-C arrives as a key
        const tcflag_t input_off = IXON | ICRNL;                    // Ctrl-S/Ctrl-Q and Enter arrive as keys too
        raw.c_lflag = raw.c_lflag & ~local_off;
        raw.c_iflag = raw.c_iflag & ~input_off;
        raw.c_cc[VMIN]  = 1;   // read() waits for at least one byte
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

        write_text("\x1b[?1049h");   // switch to the alternate screen
    }

    ~RawTerminal() {
        write_text("\x1b[?1049l");   // back to the normal screen
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &original_);
    }

    RawTerminal(const RawTerminal&) = delete;
    RawTerminal& operator=(const RawTerminal&) = delete;

    static void write_text(const std::string& text) {
        std::fwrite(text.data(), 1, text.size(), stdout);
        std::fflush(stdout);
    }

private:
    termios original_{};
};

void terminal_size(int& width, int& height) {
    winsize size{};
    const bool known = ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col > 0;
    if (known) {
        width  = size.ws_col;
        height = size.ws_row;
    } else {
        width  = 80;
        height = 24;
    }
}

// Every panel line is exactly the screen width, so each one fully overwrites the last frame.
void draw(const std::vector<std::string>& lines, int cursor_column) {
    std::string frame = "\x1b[H";   // cursor to the top-left corner
    for (std::size_t i = 0; i < lines.size(); ++i) {
        frame += lines[i];
        if (i + 1 < lines.size()) {
            frame += "\r\n";
        }
    }
    frame += "\x1b[K";   // clear the rest of the prompt line
    frame += "\x1b[" + std::to_string(lines.size()) + ";" + std::to_string(cursor_column) + "H";
    RawTerminal::write_text(frame);
}

enum class KeyKind { Char, Enter, Backspace, Up, Down, Quit, Other };

struct Key {
    KeyKind kind = KeyKind::Other;
    char    ch   = 0;
};

Key read_key() {
    char c = 0;
    if (read(STDIN_FILENO, &c, 1) != 1) {
        return {KeyKind::Quit, 0};
    }

    const char ctrl_c = 3;
    const char ctrl_d = 4;
    const char escape = 27;
    if (c == ctrl_c || c == ctrl_d) {
        return {KeyKind::Quit, 0};
    }
    if (c == '\r' || c == '\n') {
        return {KeyKind::Enter, 0};
    }
    if (c == 127 || c == 8) {
        return {KeyKind::Backspace, 0};
    }

    // Arrow keys arrive as ESC [ A (up) and ESC [ B (down).
    if (c == escape) {
        char sequence[2] = {0, 0};
        if (read(STDIN_FILENO, &sequence[0], 1) != 1 || read(STDIN_FILENO, &sequence[1], 1) != 1) {
            return {KeyKind::Other, 0};
        }
        if (sequence[0] == '[' && sequence[1] == 'A') {
            return {KeyKind::Up, 0};
        }
        if (sequence[0] == '[' && sequence[1] == 'B') {
            return {KeyKind::Down, 0};
        }
        return {KeyKind::Other, 0};
    }

    const bool printable = c >= 32 && c < 127;
    if (printable) {
        return {KeyKind::Char, c};
    }
    return {KeyKind::Other, 0};
}

// The prompt's text, and the commands already run, reachable with the up and down arrows.
class LineEditor {
public:
    const std::string& text() const {
        return text_;
    }

    void type(char c) {
        text_ += c;
    }

    void backspace() {
        if (!text_.empty()) {
            text_.pop_back();
        }
    }

    void older() {
        if (position_ > 0) {
            position_ = position_ - 1;
            text_ = history_[position_];
        }
    }

    void newer() {
        if (position_ < history_.size()) {
            position_ = position_ + 1;
        }
        if (position_ == history_.size()) {
            text_.clear();
        } else {
            text_ = history_[position_];
        }
    }

    std::string take() {
        std::string line = text_;
        text_.clear();
        return line;
    }

    void remember(const std::string& command) {
        const bool same_as_last = !history_.empty() && history_.back() == command;
        if (!same_as_last) {
            history_.push_back(command);
        }
        position_ = history_.size();
    }

private:
    std::string              text_;
    std::vector<std::string> history_;
    std::size_t              position_ = 0;   // history_.size() means the fresh line
};

}  // namespace

std::vector<std::string> render_screen(const Tpu& tpu, const std::string& file, const std::string& output,
                                       const std::string& input, int width, int height) {
    const std::string prompt_line = kPrompt + input;
    if (width < 60 || height < 16) {
        const std::string need = "terminal too small: need 60x16, have " + std::to_string(width) + "x" +
                                 std::to_string(height);
        // The message is not cut to the width: in a terminal this narrow, wrapping it is better than hiding it.
        std::vector<std::string> lines;
        lines.push_back(need);
        for (int i = 1; i < height - 1; ++i) {
            lines.push_back("");
        }
        lines.push_back(prompt_line);
        return lines;
    }

    const int panel_height  = height - 1;
    const int output_height = std::max(6, panel_height / 3);
    const int top_height    = panel_height - output_height;
    const int left_width    = width / 2;
    const int right_width   = width - left_width;

    std::string program_title = "Program";
    if (!file.empty()) {
        program_title += " " + file;
    }

    const std::vector<std::string> left   = box(program_title, program_lines(tpu, top_height - 2), left_width, top_height);
    const std::vector<std::string> right  = box("Machine", machine_lines(tpu), right_width, top_height);
    const std::vector<std::string> bottom = box("Output", last_lines(output, output_height - 2), width, output_height);

    std::vector<std::string> lines;
    for (int i = 0; i < top_height; ++i) {
        const std::size_t row = static_cast<std::size_t>(i);
        lines.push_back(left[row] + right[row]);
    }
    for (const std::string& line : bottom) {
        lines.push_back(line);
    }
    lines.push_back(prompt_line);
    return lines;
}

void run_interactive(Shell& shell, const Tpu& tpu, const std::string& greeting) {
    RawTerminal terminal;
    LineEditor editor;
    std::string output = greeting;
    std::string last_command;

    while (!shell.quit_requested()) {
        int width  = 0;
        int height = 0;
        terminal_size(width, height);
        const std::vector<std::string> screen = render_screen(tpu, shell.loaded_file(), output, editor.text(), width, height);
        const int cursor_column = static_cast<int>(std::string(kPrompt).size() + editor.text().size()) + 1;
        draw(screen, cursor_column);

        const Key key = read_key();
        if (key.kind == KeyKind::Quit) {
            return;
        }
        if (key.kind == KeyKind::Char) {
            editor.type(key.ch);
        } else if (key.kind == KeyKind::Backspace) {
            editor.backspace();
        } else if (key.kind == KeyKind::Up) {
            editor.older();
        } else if (key.kind == KeyKind::Down) {
            editor.newer();
        } else if (key.kind == KeyKind::Enter) {
            std::string command = editor.take();
            if (command.empty()) {
                command = last_command;   // Enter on an empty line repeats the last command
            }
            if (command.empty()) {
                continue;
            }
            editor.remember(command);
            last_command = command;
            output = kPrompt + command + "\n" + shell.execute(command);
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
