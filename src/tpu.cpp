#include "tpu.h"

#include <algorithm>
#include <vector>

#include "datapath.h"
#include "trace.h"

Tpu::Tpu(const Config& cfg, std::size_t host_bytes, std::size_t weight_bytes)
    : cfg_(cfg), ub_(cfg), acc_(cfg), fifo_(cfg), dma_(cfg), mxu_(cfg),
      host_(host_bytes, 0), weight_mem_(weight_bytes, 0) {}

void Tpu::tick() {
    ++cycle_;

    // Reverse pipeline order, so a stage reads its producer's previous cycle.
    dma_.tick(cycle_);       // writeback / DMA completion
    mxu_.idle_until(cycle_); // compute
    fifo_.tick(cycle_);      // weight staging
}

bool Tpu::dma_to_ub(HostAddr host_addr, UbAddr ub_addr, uint32_t bytes) {
    DmaRequest r;
    r.dir   = DmaDir::HOST_TO_UB;
    r.host  = host_addr;
    r.ub    = ub_addr;
    r.bytes = bytes;
    return dma_.start(r, cycle_, host_, ub_);
}

bool Tpu::dma_to_host(UbAddr ub_addr, HostAddr host_addr, uint32_t bytes) {
    DmaRequest r;
    r.dir   = DmaDir::UB_TO_HOST;
    r.host  = host_addr;
    r.ub    = ub_addr;
    r.bytes = bytes;
    return dma_.start(r, cycle_, host_, ub_);
}

bool Tpu::stage_weights(uint32_t ddr_addr, uint32_t rows, uint32_t cols, TileId tile) {
    if (rows > cfg_.dim || cols > cfg_.dim) return false;

    const std::size_t need = static_cast<std::size_t>(rows) * cols;
    if (ddr_addr > weight_mem_.size() || need > weight_mem_.size() - ddr_addr) return false;

    WeightTile t;
    t.ddr_addr = ddr_addr;
    t.tile     = tile;
    t.rows     = rows;
    t.cols     = cols;
    t.data.assign(weight_mem_.begin() + ddr_addr,
                  weight_mem_.begin() + ddr_addr + static_cast<std::ptrdiff_t>(need));
    return fifo_.push_refill(t, cycle_);
}

bool Tpu::load_weights_from_fifo() {
    WeightTile t;
    if (!fifo_.pop(t)) return false;

    mxu_.idle_until(cycle_);
    mxu_.load_weights(t.view());
    // Without double buffering the load costs the array dim cycles.
    cycle_ = std::max(cycle_, mxu_.cycle());
    dma_.tick(cycle_);
    fifo_.tick(cycle_);
    return true;
}

bool Tpu::matmul(UbAddr ub_addr, uint32_t len, BankId bank, bool accumulate) {
    if (!acc_.valid(bank)) return false;
    if (len > cfg_.dim) return false;          // a bank holds at most dim rows
    if (acc_.locked(bank)) return false;       // accum_hazard: the caller stalls
    if (!ub_.tile_fits(ub_addr, len, cfg_.dim, cfg_.dim)) return false;

    // The array writes straight into the bank, so it stays locked for the matmul.
    acc_.lock(bank);
    mxu_.idle_until(cycle_);
    const ConstI8View acts = ub_.view(ub_addr, len, cfg_.dim, cfg_.dim);
    const MxuTiming   t    = mxu_.matmul(acts, acc_.bank(bank).tile(0, 0, len, cfg_.dim),
                                        accumulate);
    acc_.unlock(bank);

    // The array keeps its own clock; the machine's clock follows it.
    cycle_ = std::max(cycle_, t.end);
    dma_.tick(cycle_);
    fifo_.tick(cycle_);
    return true;
}

// The in-order sequencer, described in README sections 14 and 15.
Unit Tpu::unit_of(Op op) {
    switch (op) {
        case Op::READ_HOST:
        case Op::WRITE_HOST:    return Unit::DMA;
        case Op::READ_WEIGHTS:  return Unit::WEIGHT;
        case Op::MATMUL:        return Unit::MXU;
        case Op::ACTIVATE:      return Unit::ACT;
        case Op::SYNC:
        case Op::NOP:
        case Op::HALT:          return Unit::SEQ;
    }
    return Unit::SEQ;
}

bool Tpu::quiet() const {
    for (const InFlight& f : units_) {
        if (f.active) return false;
    }
    return true;
}

namespace {

// Output shape of an Activate after pooling; a ragged edge is dropped.
void act_out_shape(const Decoded& d, uint32_t dim, uint32_t& rows, uint32_t& cols) {
    if (d.pool == Pool::NONE) {
        rows = d.len;
        cols = dim;
        return;
    }
    rows = (d.len - d.pool_window) / d.pool_stride + 1;
    cols = (dim - d.pool_window) / d.pool_stride + 1;
}

bool range_ok(std::size_t off, std::size_t len, std::size_t size) {
    return off <= size && len <= size - off;
}

}  // namespace

bool Tpu::validate(const Decoded& d, Reservation& res, std::string& why) const {
    const uint32_t dim = cfg_.dim;
    res = Reservation{};

    switch (d.op) {
        case Op::READ_HOST:
            if (!range_ok(d.host_addr, d.bytes, host_.size())) {
                why = "Read_Host_Memory: host range";
                return false;
            }
            if (!range_ok(d.ub_addr, d.bytes, ub_.bytes())) {
                why = "Read_Host_Memory: ub range";
                return false;
            }
            res.ub_write = UbRegion{d.ub_addr, d.ub_addr + d.bytes};
            return true;

        case Op::WRITE_HOST:
            if (!range_ok(d.ub_addr, d.bytes, ub_.bytes())) {
                why = "Write_Host_Memory: ub range";
                return false;
            }
            if (!range_ok(d.host_addr, d.bytes, host_.size())) {
                why = "Write_Host_Memory: host range";
                return false;
            }
            res.ub_read = UbRegion{d.ub_addr, d.ub_addr + d.bytes};
            return true;

        case Op::READ_WEIGHTS: {
            const std::size_t need = static_cast<std::size_t>(dim) * dim;
            if (!range_ok(d.ddr_addr, need, weight_mem_.size())) {
                why = "Read_Weights: ddr range";
                return false;
            }
            res.writes_weights = true;
            return true;
        }

        case Op::MATMUL: {
            // Check order matches the oracle's, so both report the same reason.
            if (d.acc_bank >= cfg_.acc_banks) {
                why = "MatMul: bank out of range";
                return false;
            }
            if (d.len > dim) {
                why = "MatMul: len exceeds accumulator rows";
                return false;
            }
            const std::size_t need = static_cast<std::size_t>(d.len) * dim;
            if (!range_ok(d.ub_addr, need, ub_.bytes())) {
                why = "MatMul: activation range";
                return false;
            }
            res.ub_read       = UbRegion{d.ub_addr, static_cast<UbAddr>(d.ub_addr + need)};
            res.acc_write     = d.acc_bank;
            res.reads_weights = true;
            // Accumulating in place reads the bank as well as writing it.
            if (d.accumulate) res.acc_read = d.acc_bank;
            return true;
        }

        case Op::ACTIVATE: {
            if (d.acc_bank >= cfg_.acc_banks) {
                why = "Activate: bank out of range";
                return false;
            }
            if (d.len > dim) {
                why = "Activate: len exceeds accumulator rows";
                return false;
            }
            if (d.pool != Pool::NONE) {
                if (d.pool_window == 0 || d.pool_stride == 0) {
                    why = "Activate: zero pool window or stride";
                    return false;
                }
                if (d.pool_window > d.len || d.pool_window > dim) {
                    why = "Activate: pool window larger than input";
                    return false;
                }
            }
            uint32_t rows = 0, cols = 0;
            act_out_shape(d, dim, rows, cols);
            const std::size_t need = static_cast<std::size_t>(rows) * cols;
            if (!range_ok(d.ub_addr, need, ub_.bytes())) {
                why = "Activate: output range";
                return false;
            }
            res.acc_read  = d.acc_bank;
            res.ub_write  = UbRegion{d.ub_addr, static_cast<UbAddr>(d.ub_addr + need)};
            return true;
        }

        case Op::SYNC:
        case Op::NOP:
        case Op::HALT:
            return true;
    }
    why = "illegal opcode";
    return false;
}

uint64_t Tpu::duration_of(const Decoded& d) const {
    switch (d.op) {
        case Op::READ_HOST:
        case Op::WRITE_HOST:
            return std::max<uint64_t>(1, dma_.transfer_cycles(d.bytes));

        case Op::READ_WEIGHTS:
            // Only the shift into a plane; the FIFO already paid the DDR fetch.
            return mxu_.load_bubble();

        case Op::MATMUL:
            // Set by the array itself at issue, from its own timing model.
            return 0;

        case Op::ACTIVATE:
            // Input elements, not output ones, so pooling is not cheaper.
            return static_cast<uint64_t>(d.len) * cfg_.dim + cfg_.act_pipeline_depth;

        case Op::SYNC:
        case Op::NOP:
        case Op::HALT:
            return 1;
    }
    return 1;
}

void Tpu::ub_streams(uint32_t& readers, uint32_t& writers) const {
    readers = 0;
    writers = 0;
    for (const InFlight& f : units_) {
        if (!f.active) continue;
        if (!f.res.ub_read.empty())  ++readers;
        if (!f.res.ub_write.empty()) ++writers;
    }
}

bool Tpu::ub_port_available(const Reservation& r) {
    const bool wants_read  = !r.ub_read.empty();
    const bool wants_write = !r.ub_write.empty();
    if (!wants_read && !wants_write) return true;

    uint32_t readers = 0, writers = 0;
    ub_streams(readers, writers);

    if ((wants_read && readers >= cfg_.ub_banks) ||
        (wants_write && writers >= cfg_.ub_banks)) {
        ub_.note_bank_conflict();
        ++stalls_.ub_bank_conflict;
        return false;
    }
    return true;
}

bool Tpu::interlocked(const Reservation& r, StallReason& why, Unit& blocker) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(Unit::COUNT); ++i) {
        const InFlight& f = units_[i];
        if (!f.active) continue;
        blocker = static_cast<Unit>(i);

        // A reader waits for the writer of its region; a writer waits for both.
        if (r.ub_read.overlaps(f.res.ub_write))  { ++stalls_.ub_raw; why = StallReason::UB_RAW; return true; }
        if (r.ub_write.overlaps(f.res.ub_read))  { ++stalls_.ub_war; why = StallReason::UB_WAR; return true; }
        if (r.ub_write.overlaps(f.res.ub_write)) { ++stalls_.ub_waw; why = StallReason::UB_WAW; return true; }

        const bool acc_conflict =
            (r.acc_read != INVALID_BANK && r.acc_read == f.res.acc_write) ||
            (r.acc_write != INVALID_BANK && (r.acc_write == f.res.acc_write ||
                                             r.acc_write == f.res.acc_read));
        if (acc_conflict) { ++stalls_.accum_hazard; why = StallReason::ACCUM_HAZARD; return true; }

        // A MatMul needs the tile its Read_Weights was fetching.
        if (r.reads_weights && f.res.writes_weights) {
            ++stalls_.weight_stall;
            why = StallReason::WEIGHT_STALL;
            return true;
        }

        // A load may not overwrite a running MatMul's tile with no shadow plane.
        if (r.writes_weights && f.res.reads_weights && !cfg_.double_buffer) {
            ++stalls_.weight_stall;
            why = StallReason::WEIGHT_STALL;
            return true;
        }
        if (r.writes_weights && f.res.writes_weights) {
            ++stalls_.weight_stall;
            why = StallReason::WEIGHT_STALL;
            return true;
        }
    }
    why     = StallReason::NONE;
    blocker = Unit::COUNT;
    return false;
}

uint64_t Tpu::execute(const Decoded& d, PendingWrite& pw) {
    const uint32_t dim = cfg_.dim;
    uint64_t duration  = duration_of(d);

    switch (d.op) {
        case Op::READ_HOST: {
            DmaRequest r;
            r.dir   = DmaDir::HOST_TO_UB;
            r.host  = d.host_addr;
            r.ub    = d.ub_addr;
            r.bytes = d.bytes;
            dma_.begin(r, cycle_, host_.size(), ub_);

            pw.ub_at = d.ub_addr;
            pw.ub_data.resize(d.bytes);
            for (uint32_t i = 0; i < d.bytes; ++i) {
                pw.ub_data[i] = static_cast<i8>(host_[d.host_addr + i]);
            }
            break;
        }

        case Op::WRITE_HOST: {
            DmaRequest r;
            r.dir   = DmaDir::UB_TO_HOST;
            r.host  = d.host_addr;
            r.ub    = d.ub_addr;
            r.bytes = d.bytes;
            dma_.begin(r, cycle_, host_.size(), ub_);

            pw.host_at = d.host_addr;
            pw.host_data.resize(d.bytes);
            for (uint32_t i = 0; i < d.bytes; ++i) {
                pw.host_data[i] = static_cast<uint8_t>(ub_.at(d.ub_addr + i));
            }
            break;
        }

        case Op::READ_WEIGHTS:
            // The tile is already in the FIFO; the shift into a plane is at retire.
            stalls_.weight_load_bubble += mxu_.load_bubble();
            break;

        case Op::MATMUL: {
            mxu_.idle_until(cycle_);

            pw.bank     = d.acc_bank;
            pw.acc_rows = d.len;
            pw.acc_data.assign(static_cast<std::size_t>(d.len) * dim, 0);
            const I32View out(pw.acc_data.data(), d.len, dim);

            // Accumulating in place reads the bank at issue, like every input.
            if (d.accumulate) out.copy_from(acc_.bank(d.acc_bank).tile(0, 0, d.len, dim));

            const ConstI8View acts = ub_.view(d.ub_addr, d.len, dim, dim);
            // The array's own timing model decides the duration.
            const MxuTiming   t    = mxu_.matmul(acts, out, d.accumulate, sink_);
            duration               = t.cycles;
            break;
        }

        case Op::ACTIVATE:
            stage_activate(d, pw);
            break;

        case Op::SYNC:
        case Op::NOP:
        case Op::HALT:
            break;
    }
    return duration;
}

// Pooling sees the requantized int8 stream, not the int32 accumulators.
void Tpu::stage_activate(const Decoded& d, PendingWrite& pw) {
    const uint32_t     dim  = cfg_.dim;
    const ConstI32View bank = acc_.bank(d.acc_bank);

    std::vector<i8> tile(static_cast<std::size_t>(d.len) * dim, 0);
    for (uint32_t r = 0; r < d.len; ++r) {
        for (uint32_t c = 0; c < dim; ++c) {
            const i8 q =
                quant::requantize_biased(bank.at(r, c), d.bias, d.multiplier, d.shift);
            i8 v = q;
            switch (d.act) {
                case ActFn::IDENTITY: break;
                case ActFn::RELU:     v = quant::relu(q); break;
                case ActFn::RELU6:    v = quant::relu6(q); break;
            }
            tile[static_cast<std::size_t>(r) * dim + c] = v;
        }
    }

    uint32_t rows = 0, cols = 0;
    act_out_shape(d, dim, rows, cols);
    pw.ub_at = d.ub_addr;

    if (d.pool == Pool::NONE) {
        pw.ub_data = std::move(tile);
        return;
    }

    const uint32_t w = d.pool_window;
    const uint32_t s = d.pool_stride;
    pw.ub_data.assign(static_cast<std::size_t>(rows) * cols, 0);

    for (uint32_t orow = 0; orow < rows; ++orow) {
        for (uint32_t ocol = 0; ocol < cols; ++ocol) {
            int64_t sum  = 0;
            i8      best = tile[static_cast<std::size_t>(orow * s) * dim + ocol * s];
            for (uint32_t dr = 0; dr < w; ++dr) {
                for (uint32_t dc = 0; dc < w; ++dc) {
                    const i8 v =
                        tile[static_cast<std::size_t>(orow * s + dr) * dim + (ocol * s + dc)];
                    sum += v;
                    best = quant::pool_max(best, v);
                }
            }
            pw.ub_data[static_cast<std::size_t>(orow) * cols + ocol] =
                d.pool == Pool::MAX ? best : quant::pool_avg(sum, w);
        }
    }
}

void Tpu::finish(InFlight& f) {
    if (sink_) sink_->on_retire(*this, cycle_, static_cast<Unit>(&f - units_), f);

    const PendingWrite& pw = f.pending;
    for (std::size_t i = 0; i < pw.ub_data.size(); ++i) {
        ub_.at(static_cast<UbAddr>(pw.ub_at + i)) = pw.ub_data[i];
    }
    for (std::size_t i = 0; i < pw.host_data.size(); ++i) {
        host_[pw.host_at + i] = pw.host_data[i];
    }
    if (pw.bank != INVALID_BANK) {
        acc_.bank(pw.bank)
            .tile(0, 0, pw.acc_rows, cfg_.dim)
            .copy_from(ConstI32View(pw.acc_data.data(), pw.acc_rows, cfg_.dim));
    }

    if (f.op == Op::READ_WEIGHTS) {
        // Shift the arrived tile in; its bubble is already in the duration.
        WeightTile t;
        if (fifo_.pop(t)) {
            const uint32_t plane = mxu_.load_plane();
            mxu_.load_weights_untimed(t.view());
            if (sink_) sink_->on_weight_load(*this, cycle_, f.pc, t, plane, mxu_.switch_pending());
        } else {
            // Unreachable, but counted so a regression surfaces here.
            ++stalls_.weight_fifo_empty;
        }
    }
    f.active = false;
}

void Tpu::retire_completed(TpuResult& st) {
    // Before issue, so a unit freed this cycle can take new work in it.
    for (InFlight& f : units_) {
        if (f.active && cycle_ >= f.done_cycle) {
            finish(f);
            ++st.retired;
        }
    }
}

void Tpu::issue_step(const std::vector<RawInst>& prog, TpuResult& st,
                     const TpuOptions& opts) {
    last_issued_     = false;
    last_trapped_    = false;
    last_reason_     = StallReason::NONE;
    last_blocker_    = Unit::COUNT;
    last_blocker_pc_ = 0;

    // Recorded once, here, so a trace and the stall counters cannot disagree.
    auto stall = [&](StallReason why, Unit blocker, const Decoded* d) {
        if (why == StallReason::DRAIN) {
            uint64_t latest = 0;
            for (std::size_t i = 0; i < static_cast<std::size_t>(Unit::COUNT); ++i) {
                if (units_[i].active && units_[i].done_cycle >= latest) {
                    latest  = units_[i].done_cycle;
                    blocker = static_cast<Unit>(i);
                }
            }
        }
        last_reason_     = why;
        last_blocker_    = blocker;
        last_blocker_pc_ = blocker != Unit::COUNT ? slot(blocker).pc : 0;
        if (sink_) {
            sink_->on_stall(*this, cycle_, st.pc, d ? *d : Decoded{}, why, last_blocker_,
                            last_blocker_pc_);
        }
    };
    auto trap = [&](const std::string& reason) {
        last_trapped_ = true;
        if (sink_) sink_->on_trap(*this, cycle_, st.pc, reason);
    };

    if (st.pc >= prog.size()) {
        // A well-formed program ends in Halt; drain first, as the oracle does.
        if (!quiet()) { ++stalls_.drain; stall(StallReason::DRAIN, Unit::COUNT, nullptr); return; }
        st.trapped     = true;
        st.trap_reason = "ran past the end of the program";
        trap(st.trap_reason);
        return;
    }

    const Decoded d = decode(prog[st.pc]);

    Reservation res;
    std::string why;
    const bool  ok = !d.trap && validate(d, res, why);
    if (d.trap) why = "illegal opcode";

    if (!ok) {
        // Everything before this instruction retires before the trap is reported.
        if (!quiet()) { ++stalls_.drain; stall(StallReason::DRAIN, Unit::COUNT, &d); return; }
        st.trapped     = true;
        st.trap_reason = why;
        ++st.retired;
        trap(why);
        return;
    }

    // A barrier and a halt both need the machine quiet.
    if (d.op == Op::SYNC || d.op == Op::HALT) {
        if (!quiet()) { ++stalls_.drain; stall(StallReason::DRAIN, Unit::COUNT, &d); return; }
    }

    const Unit u = unit_of(d.op);
    if (slot(u).active) { ++stalls_.unit_busy; stall(StallReason::UNIT_BUSY, u, &d); return; }

    // The tile has to have arrived from DDR; with a shallow FIFO it may not have.
    if (d.op == Op::READ_WEIGHTS && fifo_.empty()) {
        ++stalls_.weight_fifo_empty;
        stall(StallReason::WEIGHT_FIFO_EMPTY, Unit::COUNT, &d);
        return;
    }

    {
        StallReason reason  = StallReason::NONE;
        Unit        blocker = Unit::COUNT;
        if (interlocked(res, reason, blocker)) { stall(reason, blocker, &d); return; }
    }
    if (!ub_port_available(res)) { stall(StallReason::UB_BANK_CONFLICT, Unit::COUNT, &d); return; }

    if (opts.trace && opts.trace_out) {
        std::fprintf(opts.trace_out, "%8llu  issue %4zu: %s\n",
                     static_cast<unsigned long long>(cycle_), st.pc, op_name(d.op));
    }

    PendingWrite pw;
    uint64_t     duration = execute(d, pw);
    if (duration == 0) duration = 1;

    const std::size_t slot_i = static_cast<std::size_t>(d.op);
    if (slot_i < RunProfile::OPS) {
        profile_.op_cycles[slot_i] += duration;
        ++profile_.op_count[slot_i];
    }
    if (d.op == Op::READ_HOST || d.op == Op::WRITE_HOST) profile_.dma_bytes += d.bytes;
    if (d.op == Op::MATMUL) {
        ++profile_.matmuls;
        // The steady-state part; the rest of the duration is fill and drain.
        profile_.stream_cycles  += d.len;
        profile_.macs_performed += static_cast<uint64_t>(d.len) * cfg_.dim * cfg_.dim;
    }

    last_issued_ = true;
    if (sink_) sink_->on_issue(*this, cycle_, st.pc, d, u, res, duration, pw);

    ++st.pc;

    // Halt occupies no unit, but still spends its issue cycle.
    if (d.op == Op::HALT) {
        st.halted    = true;
        st.exit_code = d.code;
        ++st.retired;
        if (sink_) sink_->on_halt(*this, cycle_, st.pc - 1, d.code);
        ++cycle_;
        return;
    }

    InFlight& f   = slot(u);
    f.active      = true;
    f.op          = d.op;
    f.pc          = st.pc - 1;
    f.res         = res;
    f.pending     = std::move(pw);
    f.issue_cycle = cycle_;
    f.done_cycle  = cycle_ + duration;

    // A barrier only issues when quiet, so its snapshot is final.
    if (d.op == Op::SYNC) {
        TpuSnapshot s;
        s.acc = acc_.raw();
        st.syncs.push_back(std::move(s));
        if (sink_) sink_->on_sync(*this, cycle_, st.pc - 1, st.syncs.size() - 1);
    }
}

// Read ahead for the tiles upcoming Read_Weights will ask for.
void Tpu::prefetch_weights(const std::vector<RawInst>& prog) {
    const uint32_t    dim  = cfg_.dim;
    const std::size_t need = static_cast<std::size_t>(dim) * dim;

    while (!fifo_.full() && prefetch_pc_ < prog.size()) {
        const Decoded d = decode(prog[prefetch_pc_]);
        if (d.trap) break;              // the sequencer will stop here anyway
        if (d.op != Op::READ_WEIGHTS) { ++prefetch_pc_; continue; }

        if (!range_ok(d.ddr_addr, need, weight_mem_.size())) break;

        WeightTile t;
        t.ddr_addr = d.ddr_addr;
        t.tile     = d.tile;
        t.rows     = dim;
        t.cols     = dim;
        t.data.assign(weight_mem_.begin() + d.ddr_addr,
                      weight_mem_.begin() + d.ddr_addr + static_cast<std::ptrdiff_t>(need));
        if (!fifo_.push_refill(t, cycle_)) break;
        if (sink_) {
            sink_->on_prefetch(*this, cycle_, prefetch_pc_, t, cycle_ + fifo_.latency(),
                               fifo_.occupancy());
        }
        ++prefetch_pc_;
    }
}

CycleInfo Tpu::cycle_info(const RunProfile& before, const TpuResult& st) const {
    CycleInfo ci;
    ci.pc         = last_issued_ ? st.pc - 1 : st.pc;
    ci.issued     = last_issued_;
    ci.trapped    = last_trapped_;
    ci.halted     = st.halted;
    ci.reason     = last_reason_;
    ci.blocker    = last_blocker_;
    ci.blocker_pc = last_blocker_pc_;

    // Which tally the accounting step moved: the busy count or one idle bucket.
    const RunProfile& p = profile_;
    if      (p.array_busy   != before.array_busy)   ci.idle_bucket = 0;
    else if (p.idle_weights != before.idle_weights) ci.idle_bucket = 1;
    else if (p.idle_bank    != before.idle_bank)    ci.idle_bucket = 2;
    else if (p.idle_accum   != before.idle_accum)   ci.idle_bucket = 3;
    else if (p.idle_dma     != before.idle_dma)     ci.idle_bucket = 4;
    else if (p.idle_act     != before.idle_act)     ci.idle_bucket = 5;
    else                                             ci.idle_bucket = 6;

    for (std::size_t i = 0; i < static_cast<std::size_t>(Unit::COUNT); ++i) {
        if (units_[i].active) ci.units_active |= static_cast<uint8_t>(1u << i);
    }
    ci.fifo_occ   = fifo_.occupancy();
    ci.fifo_ready = fifo_.ready();
    ub_streams(ci.ub_readers, ci.ub_writers);
    ci.plane   = mxu_.active_plane();
    ci.pending = mxu_.switch_pending();
    return ci;
}

void Tpu::reset_pipeline() {
    for (InFlight& f : units_) f = InFlight{};
    stalls_      = StallStats{};
    profile_     = RunProfile{};
    prefetch_pc_ = 0;
}

// Charge one array-idle cycle to a cause; test order is the attribution policy.
void Tpu::charge_idle_cycle(const StallStats& before) {
    const StallStats& n = stalls_;

    const bool weights = n.weight_stall != before.weight_stall ||
                         n.weight_fifo_full != before.weight_fifo_full ||
                         n.weight_fifo_empty != before.weight_fifo_empty;
    if (weights) { ++profile_.idle_weights; return; }

    if (n.ub_bank_conflict != before.ub_bank_conflict) { ++profile_.idle_bank; return; }
    if (n.accum_hazard != before.accum_hazard)         { ++profile_.idle_accum; return; }

    if (slot(Unit::DMA).active) { ++profile_.idle_dma; return; }
    if (slot(Unit::ACT).active) { ++profile_.idle_act; return; }

    ++profile_.idle_other;
}

// The whole machine, one cycle per loop iteration.
TpuResult Tpu::run(const std::vector<RawInst>& prog, const TpuOptions& opts) {
    TpuResult st;
    reset_pipeline();
    sink_ = opts.sink;
    if (sink_) sink_->on_run_begin(*this);

    const uint64_t start = cycle_;
    while (!st.done()) {
        if (cycle_ - start >= opts.max_cycles) {
            st.budget = true;
            break;
        }

        // Captured first, because Halt advances cycle_ itself.
        const uint64_t t = cycle_;

        retire_completed(st);
        if (st.done()) break;

        prefetch_weights(prog);

        const StallStats before = stalls_;
        RunProfile       profile_before;
        if (sink_) profile_before = profile_;
        issue_step(prog, st, opts);

        // After issue, so a matmul's first cycle is busy and its last is not.
        if (slot(Unit::MXU).active) ++profile_.array_busy;
        else                        charge_idle_cycle(before);

        if (sink_) sink_->on_cycle_end(*this, t, cycle_info(profile_before, st));

        if (st.done()) break;

        ++cycle_;
        dma_.tick(cycle_);
        fifo_.tick(cycle_);
        mxu_.idle_until(cycle_);
    }

    st.cycles = cycle_ - start;
    pc_       = st.pc;
    if (sink_) sink_->on_run_end(*this, st);
    sink_ = nullptr;
    return st;
}
