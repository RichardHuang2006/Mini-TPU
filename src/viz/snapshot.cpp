/// Builds the snapshot JSON by hand, one small function per block of the diagram.

#include "viz/snapshot.h"

#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kTimelineCycles = 200;   // how many recent cycles the timeline shows
constexpr std::size_t kMaxListed      = 64;    // pages or tiles listed per memory

std::string quote(const std::string& text) {
    std::string out = "\"";
    for (char c : text) {
        if (c == '"') {
            out += "\\\"";
        } else if (c == '\\') {
            out += "\\\\";
        } else if (c == '\n') {
            out += "\\n";
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char escaped[8];
            std::snprintf(escaped, sizeof escaped, "\\u%04x", static_cast<unsigned>(c));
            out += escaped;
        } else {
            out += c;
        }
    }
    return out + "\"";
}

// A JSON object built one field at a time.
class JsonObject {
public:
    void add_raw(const std::string& key, const std::string& json) {
        if (!body_.empty()) {
            body_ += ",";
        }
        body_ += quote(key) + ":" + json;
    }

    void add_text(const std::string& key, const std::string& text) {
        add_raw(key, quote(text));
    }

    void add_number(const std::string& key, long long value) {
        add_raw(key, std::to_string(value));
    }

    void add_bool(const std::string& key, bool value) {
        if (value) {
            add_raw(key, "true");
        } else {
            add_raw(key, "false");
        }
    }

    std::string text() const {
        return "{" + body_ + "}";
    }

private:
    std::string body_;
};

std::string json_array(const std::vector<std::string>& items) {
    std::string out = "[";
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i > 0) {
            out += ",";
        }
        out += items[i];
    }
    return out + "]";
}

std::string number_array(const std::vector<u64>& numbers) {
    std::vector<std::string> items;
    for (u64 n : numbers) {
        items.push_back(std::to_string(n));
    }
    return json_array(items);
}

// The block the waiting instruction is held up by, so the page can outline it.
const char* blamed_block(Stall stall) {
    switch (stall) {
        case Stall::None:
            return "";
        case Stall::HostInterfaceBusy:
            return "host";
        case Stall::WaitForIdle:
            return "units";
        case Stall::WeightFifoFull:
            return "wfifo";
        case Stall::MxuBusy:
            return "mxu";
        case Stall::WeightsNotReady:
            return "weights";
        case Stall::UbNotReady:
            return "ub";
        case Stall::ActivationBusy:
            return "act";
        case Stall::AccNotReady:
            return "acc";
    }
    return "";
}

// One letter per stall cause on the timeline's issue lane; 'I' is an issue, '.' a cycle with nothing to issue.
char stall_letter(Stall stall) {
    switch (stall) {
        case Stall::None:
            return 'I';
        case Stall::HostInterfaceBusy:
            return 'h';
        case Stall::WaitForIdle:
            return 'w';
        case Stall::WeightFifoFull:
            return 'f';
        case Stall::MxuBusy:
            return 'm';
        case Stall::WeightsNotReady:
            return 'g';
        case Stall::UbNotReady:
            return 'u';
        case Stall::ActivationBusy:
            return 'a';
        case Stall::AccNotReady:
            return 'c';
    }
    return '.';
}

std::vector<u64> first_pages(const Dram& dram) {
    std::vector<u64> pages = dram.allocated_pages();
    if (pages.size() > kMaxListed) {
        pages.resize(kMaxListed);
    }
    return pages;
}

std::string program_json(const Tpu& tpu) {
    std::vector<std::string> lines;
    for (const InstrBytes& word : tpu.program().code) {
        try {
            lines.push_back(quote(disasm(decode(word))));
        } catch (const std::exception&) {
            lines.push_back(quote("(unknown opcode)"));
        }
    }
    return json_array(lines);
}

std::string host_json(const Tpu& tpu) {
    const HostInterface& link = tpu.host_interface();
    JsonObject o;
    o.add_bool("busy", link.busy());
    o.add_bool("to_ub", link.direction() == Direction::HostToUb);
    o.add_number("host_row", link.host_row());
    o.add_number("ub_row", link.ub_row());
    o.add_number("rows", link.rows());
    o.add_number("done", static_cast<long long>(link.bytes_done()));
    o.add_number("total", static_cast<long long>(link.bytes_total()));
    o.add_number("pages", static_cast<long long>(tpu.host().pages_allocated()));
    o.add_raw("page_list", number_array(first_pages(tpu.host())));
    return o.text();
}

std::string wmem_json(const Tpu& tpu) {
    long long fetching_tile = -1;
    for (const FifoTile& entry : tpu.weight_fifo().tiles()) {
        if (!entry.ready()) {
            fetching_tile = entry.tile;
            break;
        }
    }

    JsonObject o;
    o.add_number("tiles", static_cast<long long>(tpu.wmem().pages_allocated()));   // one 64 KiB page is one tile
    o.add_raw("tile_list", number_array(first_pages(tpu.wmem())));
    o.add_number("fetching", fetching_tile);
    return o.text();
}

std::string fifo_json(const Tpu& tpu) {
    std::vector<std::string> slots;
    for (const FifoTile& entry : tpu.weight_fifo().tiles()) {
        JsonObject slot;
        slot.add_number("tile", entry.tile);
        slot.add_number("arrived", static_cast<long long>(entry.bytes_arrived));
        slots.push_back(slot.text());
    }

    JsonObject o;
    o.add_number("capacity", WeightFifo::kSlots);
    o.add_number("tile_bytes", static_cast<long long>(v1::kTileBytes));
    o.add_raw("slots", json_array(slots));
    return o.text();
}

std::string mxu_json(const Tpu& tpu) {
    const SystolicArray& mxu = tpu.mxu();
    JsonObject o;
    o.add_number("dim", SystolicArray::kDim);
    o.add_bool("busy", mxu.busy());
    o.add_number("step", mxu.step());
    o.add_number("total", mxu.total_steps());
    o.add_number("rows", mxu.rows());
    o.add_number("ub_row", mxu.ub_row());
    o.add_number("acc_row", mxu.acc_row());
    o.add_number("active_tile", mxu.active_tile());
    o.add_number("shadow_tile", mxu.shadow_tile());
    o.add_bool("shadow_ready", mxu.shadow_ready());
    o.add_bool("shifting", mxu.shifting());
    o.add_number("rows_shifted", mxu.rows_shifted());
    return o.text();
}

std::string activation_json(const Tpu& tpu) {
    const ActivationUnit& act = tpu.activation();
    JsonObject o;
    o.add_bool("busy", act.busy());
    o.add_number("acc_row", act.acc_row());
    o.add_number("ub_row", act.ub_row());
    o.add_number("done", act.rows_done());
    o.add_number("total", act.rows_total());
    o.add_number("out_rows", activate_output_rows(act.rows_total(), act.pool(), act.pool_size()));
    o.add_text("function", function_name(act.function()));
    o.add_number("shift", act.shift());
    o.add_text("pool", pooling_name(act.pool()));
    o.add_number("pool_size", act.pool_size());
    o.add_number("pool_width", act.pool_width());
    return o.text();
}

// '#' for true and '.' for false: one character per cycle on the timeline, one per cell on the maps.
char mark(bool on) {
    if (on) {
        return '#';
    }
    return '.';
}

// One character per 256-row cell of the Unified Buffer: '#' if any of its bytes is nonzero.
std::string ub_nonzero_cells(const UnifiedBuffer& ub) {
    std::string cells;
    for (u32 cell = 0; cell < UnifiedBuffer::kRows / kMapRows; ++cell) {
        bool any = false;
        for (u32 r = cell * kMapRows; r < (cell + 1) * kMapRows && !any; ++r) {
            const i8* row = ub.row(r);
            for (u32 c = 0; c < UnifiedBuffer::kRowBytes && !any; ++c) {
                any = row[c] != 0;
            }
        }
        cells += mark(any);
    }
    return cells;
}

// The same for the accumulators: '#' if any value in the cell's 256 rows is nonzero.
std::string acc_nonzero_cells(const Accumulators& acc) {
    std::string cells;
    for (u32 cell = 0; cell < Accumulators::kRows / kMapRows; ++cell) {
        bool any = false;
        for (u32 r = cell * kMapRows; r < (cell + 1) * kMapRows && !any; ++r) {
            const i32* row = acc.row(r);
            for (u32 c = 0; c < Accumulators::kCols && !any; ++c) {
                any = row[c] != 0;
            }
        }
        cells += mark(any);
    }
    return cells;
}

// How many cycles ago each map cell was last written; -1 if never.
std::string ages(const std::vector<Cycle>& written, Cycle now) {
    std::vector<std::string> items;
    for (Cycle cycle : written) {
        if (cycle == kNever) {
            items.push_back("-1");
        } else {
            items.push_back(std::to_string(now - cycle));
        }
    }
    return json_array(items);
}

std::string map_json(const std::string& nonzero, const std::vector<Cycle>& written, Cycle now) {
    JsonObject o;
    o.add_number("cell_rows", kMapRows);
    o.add_text("nonzero", nonzero);
    o.add_raw("age", ages(written, now));
    return o.text();
}

std::string timeline_json(const Tpu& tpu) {
    const std::deque<CycleRecord>& activity = tpu.activity();
    std::size_t first = 0;
    if (activity.size() > kTimelineCycles) {
        first = activity.size() - kTimelineCycles;
    }

    std::string issue;
    std::string host;
    std::string fetch;
    std::string shift;
    std::string mxu;
    std::string act;
    for (std::size_t i = first; i < activity.size(); ++i) {
        const CycleRecord& r = activity[i];
        issue += stall_letter(r.stall);
        host  += mark(r.host);
        fetch += mark(r.fetching);
        shift += mark(r.shifting);
        mxu   += mark(r.mxu);
        act   += mark(r.activation);
    }

    JsonObject o;
    long long first_cycle = 0;
    if (!activity.empty()) {
        first_cycle = static_cast<long long>(activity[first].cycle);
    }
    o.add_number("first_cycle", first_cycle);
    o.add_text("issue", issue);
    o.add_text("host", host);
    o.add_text("fetch", fetch);
    o.add_text("shift", shift);
    o.add_text("mxu", mxu);
    o.add_text("act", act);
    return o.text();
}

std::string stalls_json(const Tpu& tpu) {
    std::vector<std::string> items;
    for (std::size_t i = 1; i < kStallKinds; ++i) {
        const Stall kind = static_cast<Stall>(i);
        JsonObject o;
        o.add_text("name", stall_name(kind));
        o.add_text("letter", std::string(1, stall_letter(kind)));
        o.add_number("cycles", static_cast<long long>(tpu.stats().stalled(kind)));
        items.push_back(o.text());
    }
    return json_array(items);
}

std::string pinned_json(const Shell& shell) {
    std::vector<std::string> items;
    for (const PinnedView& view : shell.pinned_views()) {
        JsonObject o;
        o.add_text("target", view.target);
        o.add_text("text", view.text);
        items.push_back(o.text());
    }
    return json_array(items);
}

}  // namespace

std::string snapshot_json(const Tpu& tpu, const Shell& shell) {
    const Cycle now = tpu.cycle();
    const std::string ub_cells  = ub_nonzero_cells(tpu.ub());
    const std::string acc_cells = acc_nonzero_cells(tpu.acc());

    JsonObject o;
    o.add_text("file", shell.loaded_file());
    o.add_number("cycle", static_cast<long long>(now));
    o.add_number("pc", tpu.pc());
    o.add_number("issued", static_cast<long long>(tpu.issued()));
    o.add_bool("halted", tpu.halted());
    o.add_text("state", machine_state(tpu));
    o.add_text("blame", blamed_block(tpu.stall()));
    o.add_number("mxu_busy_cycles", static_cast<long long>(tpu.stats().mxu_busy_cycles));
    o.add_raw("program", program_json(tpu));
    o.add_raw("host", host_json(tpu));
    o.add_raw("wmem", wmem_json(tpu));
    o.add_raw("wfifo", fifo_json(tpu));
    o.add_raw("mxu", mxu_json(tpu));
    o.add_raw("act", activation_json(tpu));
    o.add_raw("ub", map_json(ub_cells, tpu.ub_written(), now));
    o.add_raw("acc", map_json(acc_cells, tpu.acc_written(), now));
    o.add_raw("timeline", timeline_json(tpu));
    o.add_raw("stalls", stalls_json(tpu));
    o.add_raw("pinned", pinned_json(shell));
    return o.text();
}
