/// Builds the snapshot JSON by hand, one small function per block of the diagram.

#include "viz/snapshot.h"

#include <cstdio>
#include <set>
#include <vector>

namespace {

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

// Runs of consecutive numbers, added in ascending order, as [[first, count], ...].
class RunList {
public:
    void add(u64 n) {
        const bool extends = count_ > 0 && n == first_ + count_;
        if (extends) {
            count_ = count_ + 1;
            return;
        }
        flush();
        first_ = n;
        count_ = 1;
    }

    std::string json() {
        flush();
        return json_array(runs_);
    }

private:
    std::vector<std::string> runs_;
    u64 first_ = 0;
    u64 count_ = 0;

    void flush() {
        if (count_ > 0) {
            runs_.push_back("[" + std::to_string(first_) + "," + std::to_string(count_) + "]");
        }
        count_ = 0;
    }
};

// The rows written so far, as runs, so the page can paint them.
std::string written_runs(const std::vector<bool>& written) {
    RunList runs;
    for (std::size_t row = 0; row < written.size(); ++row) {
        if (written[row]) {
            runs.add(row);
        }
    }
    return runs.json();
}

std::string written_runs(const std::set<u64>& written) {
    RunList runs;
    for (u64 row : written) {
        runs.add(row);
    }
    return runs.json();
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
    o.add_raw("pages", number_array(tpu.host().allocated_pages()));   // the host block shows these pages, in order
    o.add_raw("written", written_runs(tpu.host().written_rows()));
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
    o.add_raw("tiles", number_array(tpu.wmem().allocated_pages()));   // one 64 KiB page is one tile
    o.add_raw("written", written_runs(tpu.wmem().written_rows()));
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
    o.add_number("tile_bytes", static_cast<long long>(v1::kTileBytes));
    o.add_raw("slots", json_array(slots));
    return o.text();
}

// The PEs holding an input value this cycle, as [k, first column, count] runs along each PE row.
std::string computing_pes(const SystolicArray& mxu) {
    std::vector<std::string> runs;
    for (u32 k = 0; k < v1::kMxuDim; ++k) {
        u32 n = 0;
        while (n < v1::kMxuDim) {
            if (mxu.pe(k, n).row < 0) {
                n = n + 1;
                continue;
            }
            const u32 first = n;
            while (n < v1::kMxuDim && mxu.pe(k, n).row >= 0) {
                n = n + 1;
            }
            runs.push_back("[" + std::to_string(k) + "," + std::to_string(first) + "," + std::to_string(n - first) + "]");
        }
    }
    return json_array(runs);
}

std::string mxu_json(const Tpu& tpu) {
    const SystolicArray& mxu = tpu.mxu();
    JsonObject o;
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
    o.add_raw("pes", computing_pes(mxu));
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
    o.add_number("line_rows", static_cast<long long>(act.line().size() / Accumulators::kCols));
    return o.text();
}

// The last cycles' activity, oldest first; `first` is the cycle number of the first one.
std::string timeline_json(const Tpu& tpu) {
    std::vector<std::string> cycles;
    for (const CycleRecord& record : tpu.activity()) {
        JsonObject c;
        c.add_number("stall", static_cast<long long>(record.stall));
        c.add_number("issued", record.issued);
        c.add_number("host", record.host);
        c.add_number("fetching", record.fetching);
        c.add_number("shifting", record.shifting);
        c.add_number("mxu", record.mxu);
        c.add_number("activation", record.activation);
        cycles.push_back(c.text());
    }

    JsonObject o;
    o.add_number("first", static_cast<long long>(tpu.cycle() - tpu.activity().size()));
    o.add_number("length", static_cast<long long>(kTimelineCycles));
    o.add_raw("cycles", json_array(cycles));
    return o.text();
}

// Every stall cause's name, indexed by the stall code in each timeline cycle.
std::string stall_names_json() {
    std::vector<std::string> names;
    for (std::size_t i = 0; i < kStallKinds; ++i) {
        names.push_back(quote(stall_name(static_cast<Stall>(i))));
    }
    return json_array(names);
}

}  // namespace

std::string snapshot_json(const Tpu& tpu) {
    const Cycle now = tpu.cycle();

    JsonObject o;
    o.add_number("cycle", static_cast<long long>(now));
    o.add_number("pc", tpu.pc());
    o.add_number("issued", static_cast<long long>(tpu.issued()));
    o.add_bool("halted", tpu.halted());
    o.add_text("state", machine_state(tpu));
    o.add_text("blame", blamed_block(tpu.stall()));
    o.add_raw("timeline", timeline_json(tpu));
    o.add_raw("stall_names", stall_names_json());
    o.add_raw("program", program_json(tpu));
    o.add_raw("host", host_json(tpu));
    o.add_raw("wmem", wmem_json(tpu));
    o.add_raw("wfifo", fifo_json(tpu));
    o.add_raw("mxu", mxu_json(tpu));
    o.add_raw("act", activation_json(tpu));
    o.add_raw("ub_rows", written_runs(tpu.ub().written_rows()));
    o.add_raw("acc_rows", written_runs(tpu.acc().written_rows()));
    return o.text();
}
