/// A frozen copy of every value the visualizer can show, taken at one cycle, so the server never reads the live machine.

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/tpu.h"

struct MachineState {
    Cycle cycle = 0;

    std::vector<i8>  ub;    // 98,304 rows x 256
    std::vector<i32> acc;   // 4,096 rows x 256

    std::map<u64, std::vector<i8>> host_pages;   // page number -> its 64 KiB, smallest first
    std::map<u64, std::vector<i8>> wmem_pages;   // tile number -> its 64 KiB
    std::vector<std::vector<i8>>   fifo_tiles;   // the Weight FIFO's slots, oldest first, 64 KiB each

    std::vector<i8>  active_weights;   // 256 x 256, PE (k, n) at k * 256 + n
    std::vector<i8>  shadow_weights;
    std::vector<Pe>  pes;              // 256 x 256
    std::vector<i32> line;             // the pooling line buffer, 256 values per pooled pixel
};

std::shared_ptr<const MachineState> capture(const Tpu& tpu);

// GET /rows: rows of ub, host, wmem, fifo (int8) or acc, line (int32), little-endian; host/wmem/fifo count only what is in use.
bool rows_bytes(const MachineState& state, const std::string& memory, u64 first, u64 count, std::string& out);

// GET /pes: every PE of `count` array rows from `first`, 12 bytes each: weight, shadow, act, 0, psum (4), input row (4).
bool pes_bytes(const MachineState& state, u64 first, u64 count, std::string& out);
