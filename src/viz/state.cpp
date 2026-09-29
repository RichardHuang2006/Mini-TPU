/// Copying the machine, and turning ranges of the copy into the little-endian bytes the page reads.

#include "viz/state.h"

namespace {

constexpr u64 kRowValues      = v1::kMxuDim;   // every memory row the page shows has 256 values
constexpr u64 kRowsPerPage    = v1::kTileBytes / kRowValues;
constexpr u64 kMostRowsAsked  = 4096;          // one request returns at most this many rows

void append_i32(std::string& out, i32 value) {
    const u32 bits = static_cast<u32>(value);
    out += static_cast<char>(bits & 0xFF);
    out += static_cast<char>((bits >> 8) & 0xFF);
    out += static_cast<char>((bits >> 16) & 0xFF);
    out += static_cast<char>((bits >> 24) & 0xFF);
}

bool range_ok(u64 first, u64 count, u64 total, u64 most) {
    if (count == 0 || count > most) {
        return false;
    }
    if (first >= total) {
        return false;
    }
    return count <= total - first;
}

// Rows laid end to end across 64 KiB pieces (pages, tiles or FIFO slots), in order.
bool paged_rows(const std::vector<const std::vector<i8>*>& pieces, u64 first, u64 count, std::string& out) {
    const u64 total = pieces.size() * kRowsPerPage;
    if (!range_ok(first, count, total, kMostRowsAsked)) {
        return false;
    }
    for (u64 row = first; row < first + count; ++row) {
        const std::vector<i8>& piece = *pieces[row / kRowsPerPage];
        const u64 start = (row % kRowsPerPage) * kRowValues;
        for (u64 c = 0; c < kRowValues; ++c) {
            out += static_cast<char>(piece[start + c]);
        }
    }
    return true;
}

std::vector<const std::vector<i8>*> in_order(const std::map<u64, std::vector<i8>>& pages) {
    std::vector<const std::vector<i8>*> pieces;
    for (const auto& entry : pages) {
        pieces.push_back(&entry.second);
    }
    return pieces;
}

}  // namespace

std::shared_ptr<const MachineState> capture(const Tpu& tpu) {
    auto state = std::make_shared<MachineState>();
    state->cycle = tpu.cycle();
    state->ub    = tpu.ub().bytes();
    state->acc   = tpu.acc().values();

    for (const auto& entry : tpu.host().pages()) {
        state->host_pages[entry.first] = entry.second;
    }
    for (const auto& entry : tpu.wmem().pages()) {
        state->wmem_pages[entry.first] = entry.second;
    }
    for (const FifoTile& entry : tpu.weight_fifo().tiles()) {
        state->fifo_tiles.push_back(entry.bytes);
    }

    state->active_weights = tpu.mxu().active_weights();
    state->shadow_weights = tpu.mxu().shadow_weights();
    state->pes            = tpu.mxu().grid();
    state->line           = tpu.activation().line();
    return state;
}

bool rows_bytes(const MachineState& state, const std::string& memory, u64 first, u64 count, std::string& out) {
    if (memory == "ub") {
        const u64 total = state.ub.size() / kRowValues;
        if (!range_ok(first, count, total, kMostRowsAsked)) {
            return false;
        }
        out.assign(reinterpret_cast<const char*>(state.ub.data() + first * kRowValues), count * kRowValues);
        return true;
    }

    if (memory == "acc" || memory == "line") {
        const std::vector<i32>* values = &state.line;
        if (memory == "acc") {
            values = &state.acc;
        }
        const u64 total = values->size() / kRowValues;
        if (!range_ok(first, count, total, kMostRowsAsked)) {
            return false;
        }
        for (u64 i = first * kRowValues; i < (first + count) * kRowValues; ++i) {
            append_i32(out, (*values)[i]);
        }
        return true;
    }

    if (memory == "host") {
        return paged_rows(in_order(state.host_pages), first, count, out);
    }
    if (memory == "wmem") {
        return paged_rows(in_order(state.wmem_pages), first, count, out);
    }
    if (memory == "fifo") {
        std::vector<const std::vector<i8>*> slots;
        for (const std::vector<i8>& tile : state.fifo_tiles) {
            slots.push_back(&tile);
        }
        return paged_rows(slots, first, count, out);
    }
    return false;
}

bool pes_bytes(const MachineState& state, u64 first, u64 count, std::string& out) {
    if (!range_ok(first, count, kRowValues, kRowValues)) {
        return false;
    }
    for (u64 k = first; k < first + count; ++k) {
        for (u64 n = 0; n < kRowValues; ++n) {
            const u64 at = k * kRowValues + n;
            const Pe& pe = state.pes[at];
            out += static_cast<char>(state.active_weights[at]);
            out += static_cast<char>(state.shadow_weights[at]);
            out += static_cast<char>(pe.act);
            out += static_cast<char>(0);
            append_i32(out, pe.psum);
            append_i32(out, pe.row);
        }
    }
    return true;
}
