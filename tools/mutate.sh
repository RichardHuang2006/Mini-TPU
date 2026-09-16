#!/usr/bin/env bash
# Breaks the model on purpose, one edit at a time, and checks that the suite
# notices. A surviving mutation is a gap in the tests.

set -uo pipefail
cd "$(dirname "$0")/.."

FILES=(src/tpu.cpp src/stats.h src/systolic_array.h src/datapath.h src/transfer.h tests/workloads.h)
BACKUP=$(mktemp -d)
for f in "${FILES[@]}"; do
    mkdir -p "$BACKUP/$(dirname "$f")"
    cp "$f" "$BACKUP/$f"
done
restore() { for f in "${FILES[@]}"; do cp "$BACKUP/$f" "$f"; done; }
trap 'restore; rm -rf "$BACKUP"' EXIT

pass=0
fail=0

mutate() {
    local name=$1 file=$2 from=$3 to=$4

    if ! grep -qF -- "$from" "$file"; then
        printf '  %-46s SKIPPED (pattern not found in %s)\n' "$name" "$file"
        fail=$((fail + 1))
        return
    fi

    python3 - "$file" "$from" "$to" <<'PY'
import sys
path, frm, to = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
open(path, 'w').write(src.replace(frm, to, 1))
PY

    local out
    out=$(make test 2>&1)
    if printf '%s' "$out" | grep -qE 'error:'; then
        printf '  %-46s DID NOT COMPILE\n' "$name"
        fail=$((fail + 1))
    elif printf '%s' "$out" | grep -qE '[1-9][0-9]* failing'; then
        printf '  %-46s caught\n' "$name"
        pass=$((pass + 1))
    else
        printf '  %-46s SURVIVED\n' "$name"
        fail=$((fail + 1))
    fi

    restore
}

echo "mutating the statistics and the timing model:"

# the breakdown's partition
mutate "idle cycle charged to two buckets" src/tpu.cpp \
    '    if (weights) { ++profile_.idle_weights; return; }' \
    '    if (weights) { ++profile_.idle_weights; ++profile_.idle_other; return; }'

mutate "array-busy cycles double counted" src/tpu.cpp \
    'if (slot(Unit::MXU).active) ++profile_.array_busy;' \
    'if (slot(Unit::MXU).active) profile_.array_busy += 2;'

mutate "busy sampled before issue (off by one)" src/tpu.cpp \
    '        const StallStats before = stalls_;
        issue_step(prog, st, opts);' \
    '        const StallStats before = stalls_;
        const bool was_busy = slot(Unit::MXU).active;
        issue_step(prog, st, opts);
        if (was_busy) { ++profile_.array_busy; charge_idle_cycle(before); }'

mutate "fill/drain not counted as lost" src/stats.h \
    '    s.lost.array_fill_drain =
        p.array_busy > p.stream_cycles ? p.array_busy - p.stream_cycles : 0;' \
    '    s.lost.array_fill_drain = 0;'

# attribution
mutate "DMA claims stalls before the weight path" src/tpu.cpp \
    '    if (weights) { ++profile_.idle_weights; return; }' \
    '    if (slot(Unit::DMA).active) { ++profile_.idle_dma; return; }
    if (weights) { ++profile_.idle_weights; return; }'

mutate "bank conflicts charged to 'other'" src/tpu.cpp \
    'if (n.ub_bank_conflict != before.ub_bank_conflict) { ++profile_.idle_bank; return; }' \
    'if (n.ub_bank_conflict != before.ub_bank_conflict) { ++profile_.idle_other; return; }'

# the counters themselves
mutate "MACs counted per matmul, not per row" src/tpu.cpp \
    'profile_.macs_performed += static_cast<uint64_t>(d.len) * cfg_.dim * cfg_.dim;' \
    'profile_.macs_performed += static_cast<uint64_t>(cfg_.dim) * cfg_.dim;'

mutate "streaming cycles charged the full duration" src/tpu.cpp \
    'profile_.stream_cycles  += d.len;' \
    'profile_.stream_cycles  += d.len + 2 * cfg_.dim - 1;'

mutate "DMA bytes counted once per transfer" src/tpu.cpp \
    'profile_.dma_bytes += d.bytes;' \
    'profile_.dma_bytes += 1;'

mutate "padding waste ignored" src/stats.h \
    '        s.lost.partial_tile_waste = (s.macs_performed - s.macs_useful) / peak;' \
    '        s.lost.partial_tile_waste = 0;'

# the UB port model
mutate "UB port contention removed" src/tpu.cpp \
    '    if (!ub_port_available(res)) return;' \
    '    (void)0;'

mutate "read and write ports share one budget" src/tpu.cpp \
    '    if ((wants_read && readers >= cfg_.ub_banks) ||
        (wants_write && writers >= cfg_.ub_banks)) {' \
    '    if (readers + writers >= cfg_.ub_banks) {'

# the weight prefetcher; the depth is removed at its source because mutating the
# prefetch loop's own !fifo_.full() guard would be an equivalent mutant
mutate "the FIFO ignores its own depth" src/transfer.h \
    'bool full() const { return q_.size() >= depth_; }' \
    'bool full() const { return false; }'

mutate "Read_Weights does not wait for its tile" src/tpu.cpp \
    '    if (d.op == Op::READ_WEIGHTS && fifo_.empty()) {' \
    '    if (false) {'

# timing
mutate "weight load bubble always zero" src/systolic_array.h \
    'return cfg_.double_buffer ? 0u : cfg_.dim;' \
    'return 0u;'

mutate "activation charged per row, not per element" src/tpu.cpp \
    'return static_cast<uint64_t>(d.len) * cfg_.dim + cfg_.act_pipeline_depth;' \
    'return static_cast<uint64_t>(d.len) + cfg_.act_pipeline_depth;'

# arithmetic
mutate "requantize rounds half toward zero" src/datapath.h \
    'return v >= 0 ? (v + half) >> shift : -((-v + half) >> shift);' \
    'return v >= 0 ? (v + half - 1) >> shift : -((-v + half - 1) >> shift);'

mutate "requantize floors negatives" src/datapath.h \
    'return v >= 0 ? (v + half) >> shift : -((-v + half) >> shift);' \
    'return (v + half) >> shift;'

mutate "saturation clamps only the top" src/datapath.h \
    '    if (v < I8_MIN) return static_cast<i8>(I8_MIN);' \
    '    if (false) return static_cast<i8>(I8_MIN);'

# the tiler
mutate "last row block always uses full dim" tests/workloads.h \
    'const uint32_t rows = std::min(dim, l.M - m * dim);' \
    'const uint32_t rows = dim;'

echo
echo "$pass caught, $fail survived or unusable"
[ "$fail" -eq 0 ]
