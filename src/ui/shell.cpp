/// Command parsing, and the text each command prints.

#include "ui/shell.h"

#include <charconv>
#include <cstdio>
#include <sstream>
#include <stdexcept>

#include "isa/asm.h"

namespace {

const char* kCommandList = "load FILE, step [N], back [N], next [N], prev [N], run [N], TARGET [hex|dec], quit";
const char* kTargetList  = "pc, wfifo, mxu, ub[ROW:RxC], acc[ROW:RxC], host[ROW:RxC], wmem[ROW:RxC]";

std::vector<std::string> split_words(const std::string& line) {
    std::istringstream stream(line);
    std::vector<std::string> words;
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    return words;
}

// Decimal, or hex with a 0x prefix; `what` names the number in the error message.
u64 parse_number(const std::string& text, const std::string& what) {
    const char* first = text.data();
    const char* last  = text.data() + text.size();
    int base = 10;

    const bool has_hex_prefix = text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X');
    if (has_hex_prefix) {
        first += 2;
        base = 16;
    }

    u64 value = 0;
    const std::from_chars_result result = std::from_chars(first, last, value, base);
    const bool parsed      = result.ec == std::errc();
    const bool used_it_all = result.ptr == last;
    if (!parsed || !used_it_all) {
        throw std::invalid_argument(what + " '" + text + "' is not a number");
    }
    return value;
}

std::string hex(u64 value) {
    char text[24];
    std::snprintf(text, sizeof text, "0x%llX", static_cast<unsigned long long>(value));
    return text;
}

// A request to look at memory: NAME[ROW] shows 16 columns of one row, NAME[ROW:RxC] a block.
struct View {
    std::string memory;
    u64         row  = 0;
    u64         rows = 1;
    u64         cols = 16;
};

View parse_view(const std::string& text) {
    const std::size_t open = text.find('[');
    const bool has_close = !text.empty() && text.back() == ']';
    if (open == std::string::npos || !has_close) {
        throw std::invalid_argument("unknown command '" + text + "'; commands: " + kCommandList);
    }

    View view;
    view.memory = text.substr(0, open);
    const std::string inside = text.substr(open + 1, text.size() - open - 2);

    const std::size_t colon = inside.find(':');
    if (colon == std::string::npos) {
        view.row = parse_number(inside, "row");
        return view;
    }
    view.row = parse_number(inside.substr(0, colon), "row");

    // The shape is ROWSxCOLS; the last x splits it, so a hex row count like 0x4x16 still works.
    const std::string shape = inside.substr(colon + 1);
    const std::size_t x = shape.rfind('x');
    if (x == std::string::npos) {
        throw std::invalid_argument("expected ROWSxCOLS after ':', got '" + shape + "'");
    }
    view.rows = parse_number(shape.substr(0, x), "rows");
    view.cols = parse_number(shape.substr(x + 1), "columns");

    const bool rows_ok = view.rows >= 1;
    const bool cols_ok = view.cols >= 1 && view.cols <= 256;
    if (!rows_ok || !cols_ok) {
        throw std::invalid_argument("a view is 1 or more rows of 1 to 256 columns");
    }
    return view;
}

std::string format_values(const std::vector<i8>& values, bool as_hex) {
    std::string line;
    for (i8 value : values) {
        char cell[8];
        if (as_hex) {
            const unsigned bits = static_cast<u8>(value);
            std::snprintf(cell, sizeof cell, " %02X", bits);
        } else {
            std::snprintf(cell, sizeof cell, "%5d", static_cast<int>(value));
        }
        line += cell;
    }
    return line;
}

// The first `cols` bytes of one 256-byte row of the named memory.
std::vector<i8> read_row(const Tpu& tpu, const std::string& memory, u64 row, u64 cols) {
    std::vector<i8> values(cols, 0);

    if (memory == "ub") {
        if (row > 0xFFFFFFFF) {
            throw std::out_of_range("ub row " + std::to_string(row) + " is out of range");
        }
        const i8* bytes = tpu.ub().row(static_cast<u32>(row));
        for (u64 col = 0; col < cols; ++col) {
            values[col] = bytes[col];
        }
    } else if (memory == "host") {
        tpu.host().read(row * UnifiedBuffer::kRowBytes, values.data(), cols);
    } else if (memory == "wmem") {
        tpu.wmem().read(row * UnifiedBuffer::kRowBytes, values.data(), cols);
    } else {
        throw std::invalid_argument("unknown target '" + memory + "' (try " + kTargetList + ")");
    }
    return values;
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

std::string fifo_occupancy(const WeightFifo& fifo) {
    const std::size_t count = fifo.tiles().size();
    if (count == 0) {
        return "empty";
    }
    return std::to_string(count) + " of " + std::to_string(WeightFifo::kSlots) + " tiles";
}

std::string fifo_fetch(const WeightFifo& fifo) {
    // Tiles arrive in order, so the first one not ready is the one arriving now.
    for (const FifoTile& entry : fifo.tiles()) {
        if (!entry.ready()) {
            const std::string progress = std::to_string(entry.bytes_arrived) + " of " + std::to_string(v1::kTileBytes);
            return "tile " + hex(entry.tile) + ", " + progress + " bytes";
        }
    }
    return "idle";
}

std::string status_line(const Tpu& tpu) {
    return "cycle " + std::to_string(tpu.cycle()) + ", pc " + std::to_string(tpu.pc()) + ", " + machine_state(tpu);
}

Shell::Shell(Tpu& tpu) : tpu_(tpu) {}

std::string Shell::execute(const std::string& line) {
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
        if (command == "run") {
            return run(words);
        }
        const bool moves_in_time = command == "step" || command == "back" || command == "next" || command == "prev";
        if (moves_in_time) {
            return move(words);
        }
        return show(words);
    } catch (const AsmError& e) {
        return e.what();   // already reads "file:line:col: error: ..."
    } catch (const std::exception& e) {
        return std::string("error: ") + e.what();
    }
}

bool Shell::quit_requested() const {
    return quit_;
}

const std::string& Shell::loaded_file() const {
    return loaded_file_;
}

std::string Shell::load(const std::vector<std::string>& words) {
    if (words.size() != 2) {
        throw std::invalid_argument("usage: load FILE");
    }
    const std::string& path = words[1];
    const Program program = assemble_file(path);
    tpu_.load(program);
    loaded_file_ = path;

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

// `run` goes to Halt; `run N` advances N cycles.
std::string Shell::run(const std::vector<std::string>& words) {
    require_program();
    if (words.size() == 1) {
        tpu_.run_to_halt();
    } else if (words.size() == 2) {
        tpu_.run_cycles(parse_number(words[1], "cycle count"));
    } else {
        throw std::invalid_argument("usage: run [N]");
    }
    return status_line(tpu_);
}

// step and back move by cycles, next and prev by instructions; each takes a count, 1 if left out.
std::string Shell::move(const std::vector<std::string>& words) {
    require_program();
    const std::string& command = words[0];
    if (words.size() > 2) {
        throw std::invalid_argument("usage: " + command + " [N]");
    }

    u64 count = 1;
    if (words.size() == 2) {
        count = parse_number(words[1], "count");
    }

    if (command == "step") {
        tpu_.run_cycles(count);
    } else if (command == "back") {
        tpu_.back_cycles(count);
    } else if (command == "next") {
        tpu_.next_instructions(count);
    } else {
        tpu_.prev_instructions(count);
    }
    return status_line(tpu_);
}

std::string Shell::show(const std::vector<std::string>& words) const {
    if (words.size() > 2) {
        throw std::invalid_argument("usage: TARGET [hex|dec]");
    }

    bool as_hex = false;
    if (words.size() == 2) {
        if (words[1] == "hex") {
            as_hex = true;
        } else if (words[1] != "dec") {
            throw std::invalid_argument("format must be hex or dec, got '" + words[1] + "'");
        }
    }

    if (words[0] == "pc") {
        return show_pc();
    }
    if (words[0] == "wfifo") {
        return show_fifo();
    }
    if (words[0] == "mxu") {
        return show_mxu();
    }

    const View view = parse_view(words[0]);
    if (view.memory == "acc") {
        return show_acc(view.row, view.rows, view.cols, as_hex);
    }

    // Labels are padded to the widest one, so the columns line up.
    std::vector<std::string> labels;
    std::size_t label_width = 0;
    for (u64 i = 0; i < view.rows; ++i) {
        const std::string label = hex(view.row + i) + ":";
        labels.push_back(label);
        if (label.size() > label_width) {
            label_width = label.size();
        }
    }

    std::string text;
    for (u64 i = 0; i < view.rows; ++i) {
        const std::vector<i8> values = read_row(tpu_, view.memory, view.row + i, view.cols);
        std::string label = labels[i];
        label.resize(label_width, ' ');
        if (i > 0) {
            text += "\n";
        }
        text += label + format_values(values, as_hex);
    }
    return text;
}

std::string Shell::show_pc() const {
    const std::vector<InstrBytes>& code = tpu_.program().code;
    const u32 pc = tpu_.pc();
    if (pc >= code.size()) {
        return "pc " + std::to_string(pc) + " (no instruction there)";
    }
    return "pc " + std::to_string(pc) + ": " + disasm(decode(code[pc]));
}

// Accumulator rows are int32, so they get wider columns than the byte memories.
std::string Shell::show_acc(u64 first_row, u64 rows, u64 cols, bool as_hex) const {
    std::string text;
    for (u64 i = 0; i < rows; ++i) {
        const u64 row = first_row + i;
        if (row >= Accumulators::kRows) {
            throw std::out_of_range("acc row " + std::to_string(row) + " is past the last row " +
                                    std::to_string(Accumulators::kRows - 1));
        }
        const i32* values = tpu_.acc().row(static_cast<u32>(row));

        std::string line = hex(row) + ":";
        for (u64 col = 0; col < cols; ++col) {
            char cell[16];
            if (as_hex) {
                std::snprintf(cell, sizeof cell, " %08X", static_cast<u32>(values[col]));
            } else {
                std::snprintf(cell, sizeof cell, "%9d", values[col]);
            }
            line += cell;
        }
        if (i > 0) {
            text += "\n";
        }
        text += line;
    }
    return text;
}

// The MXU's state, then a map of its top-left 16x16 PEs: '#' holds data this cycle, '.' is empty.
std::string Shell::show_mxu() const {
    const SystolicArray& mxu = tpu_.mxu();

    std::string state = "idle";
    if (mxu.busy()) {
        state = "step " + std::to_string(mxu.step()) + " of " + std::to_string(mxu.total_steps());
    }

    std::string active = "none";
    if (mxu.active_tile() >= 0) {
        active = "tile " + hex(static_cast<u64>(mxu.active_tile()));
    }

    std::string shadow = "empty";
    if (mxu.shadow_ready()) {
        shadow = "tile " + hex(static_cast<u64>(mxu.shadow_tile())) + ", ready";
    } else if (mxu.shifting()) {
        shadow = "tile " + hex(static_cast<u64>(mxu.shadow_tile())) + ", " + std::to_string(mxu.rows_shifted()) +
                 " of 256 rows shifted in";
    }

    std::string text = "MXU: " + state + "\nactive weights: " + active + "\nshadow weights: " + shadow +
                       "\nPEs 0-15 x 0-15 (# = holds data):";
    for (u32 k = 0; k < 16; ++k) {
        std::string line = "  ";
        for (u32 n = 0; n < 16; ++n) {
            if (mxu.pe(k, n).row >= 0) {
                line += '#';
            } else {
                line += '.';
            }
        }
        text += "\n" + line;
    }
    return text;
}

// One line per slot, oldest first.
std::string Shell::show_fifo() const {
    const std::deque<FifoTile>& tiles = tpu_.weight_fifo().tiles();
    std::string text = "weight FIFO: " + std::to_string(tiles.size()) + " of " + std::to_string(WeightFifo::kSlots) + " slots";

    for (std::size_t slot = 0; slot < tiles.size(); ++slot) {
        const FifoTile& entry = tiles[slot];
        std::string state = "ready";
        if (!entry.ready()) {
            state = std::to_string(entry.bytes_arrived) + " of " + std::to_string(v1::kTileBytes) + " bytes";
        }
        text += "\nslot " + std::to_string(slot) + ": tile " + hex(entry.tile) + ", " + state;
    }
    return text;
}

void Shell::require_program() const {
    if (tpu_.program().code.empty()) {
        throw std::invalid_argument("no program loaded; use load FILE");
    }
}
