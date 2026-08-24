#include "tpu.h"

#include <algorithm>
#include <vector>

#include "quant.h"

Tpu::Tpu(const Config& cfg, std::size_t host_bytes, std::size_t weight_bytes)
    : cfg_(cfg), ub_(cfg), acc_(cfg), fifo_(cfg), dma_(cfg), mxu_(cfg),
      host_(host_bytes, 0), weight_mem_(weight_bytes, 0) {}

void Tpu::tick() {
    ++cycle_;

    // Reverse pipeline order: writeback before compute before issue before fetch,
    // so a stage reads what its producer held at the end of the previous cycle
    // rather than something produced earlier in this one.
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
    // Without double buffering the load costs the array dim cycles, and the
    // machine's clock has to reflect that.
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

    // The array writes straight into the bank, so the bank is locked for the
    // duration; a consumer reading it meanwhile would see a partial result. Driven
    // by hand the matmul completes within the call and the lock is released
    // immediately; under the sequencer it is held until the matmul retires.
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

// ===========================================================================
// The sequencer
//
// In-order issue, one instruction per cycle, overlapped execution, gated by a
// scoreboard. Nothing is speculative, so nothing is ever squashed: correctness
// reduces to whether each instruction computes the right tensor and whether the
// interlock ever lets a consumer read a region before its producer finished.
//
// An instruction's data effect lands at issue rather than spread over its
// duration. The scoreboard is what makes that sound, since no instruction able to
// observe the difference may issue until this one retires; the schedule is then
// the only thing the timing model has to get right. Read_Weights is the one
// exception: its tile is popped from the FIFO and shifted into the weight plane
// at retire, which is when the tile has actually arrived from DDR.
// ===========================================================================

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

// Output shape of an Activate, after any pooling. Floor semantics: a window that
// does not divide the input evenly drops the ragged edge.
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
            // Check order matches the oracle's, so a program with more than one
            // fault reports the same reason on both sides.
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
            // Accumulating in place reads the bank as well as writing it, so it
            // has to wait for whatever is still producing into it.
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
            // Only the shift into a plane. The FIFO already paid the DDR fetch in
            // the background, and charging that latency again would make FIFO depth
            // irrelevant. With a shadow plane the shift hides under whatever the
            // array is doing; without one it is dim exposed cycles.
            return mxu_.load_bubble();

        case Op::MATMUL:
            // Set by the array itself at issue, from its own timing model.
            return 0;

        case Op::ACTIVATE:
            // Throughput-1 after fill: one accumulator element per cycle, plus the
            // pipeline depth to fill it.
            //
            // The count is over input elements, not output ones, so pooling does
            // not make an Activate cheaper: every accumulator element is still read
            // and requantized before the window reduction sees it. Charging for
            // outputs would make a 2x2 pool look four times faster than it is.
            return static_cast<uint64_t>(d.len) * cfg_.dim + cfg_.act_pipeline_depth;

        case Op::SYNC:
        case Op::NOP:
        case Op::HALT:
            return 1;
    }
    return 1;
}

bool Tpu::ub_port_available(const Reservation& r) {
    const bool wants_read  = !r.ub_read.empty();
    const bool wants_write = !r.ub_write.empty();
    if (!wants_read && !wants_write) return true;

    uint32_t readers = 0, writers = 0;
    for (const InFlight& f : units_) {
        if (!f.active) continue;
        if (!f.res.ub_read.empty())  ++readers;
        if (!f.res.ub_write.empty()) ++writers;
    }

    if ((wants_read && readers >= cfg_.ub_banks) ||
        (wants_write && writers >= cfg_.ub_banks)) {
        ub_.note_bank_conflict();
        ++stalls_.ub_bank_conflict;
        return false;
    }
    return true;
}

bool Tpu::interlocked(const Reservation& r) {
    for (const InFlight& f : units_) {
        if (!f.active) continue;

        // A reader waits for the writer of its region; a writer waits for both
        // readers and writers. WAR matters even though effects land at issue,
        // because the array streams activations out of the buffer over many cycles
        // and a DMA overwriting that region early would corrupt a real machine.
        // Being conservative here can cost overlap but never correctness.
        if (r.ub_read.overlaps(f.res.ub_write))  { ++stalls_.ub_raw; return true; }
        if (r.ub_write.overlaps(f.res.ub_read))  { ++stalls_.ub_war; return true; }
        if (r.ub_write.overlaps(f.res.ub_write)) { ++stalls_.ub_waw; return true; }

        const bool acc_conflict =
            (r.acc_read != INVALID_BANK && r.acc_read == f.res.acc_write) ||
            (r.acc_write != INVALID_BANK && (r.acc_write == f.res.acc_write ||
                                             r.acc_write == f.res.acc_read));
        if (acc_conflict) { ++stalls_.accum_hazard; return true; }

        // A MatMul needs the tile its Read_Weights was fetching.
        if (r.reads_weights && f.res.writes_weights) { ++stalls_.weight_stall; return true; }

        // And a Read_Weights must not overwrite the tile a running MatMul is still
        // multiplying by, unless there is a shadow plane to put it in. The entire
        // benefit of double buffering falls out of this one condition rather than
        // a special case elsewhere.
        if (r.writes_weights && f.res.reads_weights && !cfg_.double_buffer) {
            ++stalls_.weight_stall;
            return true;
        }
        if (r.writes_weights && f.res.writes_weights) { ++stalls_.weight_stall; return true; }
    }
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
            // The prefetcher already put the tile in the FIFO, and issue only
            // proceeded because it had arrived. What remains is the shift into the
            // weight plane, which happens at retire.
            stalls_.weight_load_bubble += mxu_.load_bubble();
            break;

        case Op::MATMUL: {
            // The array's own timing model decides the duration, so the sequencer
            // carries no second copy of the fill/drain formula.
            mxu_.idle_until(cycle_);

            pw.bank     = d.acc_bank;
            pw.acc_rows = d.len;
            pw.acc_data.assign(static_cast<std::size_t>(d.len) * dim, 0);
            const I32View out(pw.acc_data.data(), d.len, dim);

            // Accumulating in place starts from what the bank holds now, so the
            // bank is read at issue like every other input.
            if (d.accumulate) out.copy_from(acc_.bank(d.acc_bank).tile(0, 0, d.len, dim));

            const ConstI8View acts = ub_.view(d.ub_addr, d.len, dim, dim);
            const MxuTiming   t    = mxu_.matmul(acts, out, d.accumulate);
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

// Bias add, requantize, activation function, then an optional window reduction.
// Pooling sees the requantized int8 stream rather than the int32 accumulators, so
// pooled and unpooled Activates requantize identically and differ only in what
// follows.
void Tpu::stage_activate(const Decoded& d, PendingWrite& pw) {
    const uint32_t     dim  = cfg_.dim;
    const ConstI32View bank = acc_.bank(d.acc_bank);

    std::vector<i8> tile(static_cast<std::size_t>(d.len) * dim, 0);
    for (uint32_t r = 0; r < d.len; ++r) {
        for (uint32_t c = 0; c < dim; ++c) {
            const i8 q =
                quant::requantize_biased(bank.at(r, c), d.bias, d.multiplier, d.shift);
            // Clamped in the output's own quantized units, so ReLU6's bound is the
            // int8 value 6, not 6.0 on some notional real scale.
            i8 v = q;
            switch (d.act) {
                case ActFn::IDENTITY: break;
                case ActFn::RELU:     v = q < 0 ? i8{0} : q; break;
                case ActFn::RELU6:    v = q < 0 ? i8{0} : (q > 6 ? i8{6} : q); break;
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
                    if (v > best) best = v;
                }
            }
            // Average pooling rounds through the same helper as requantization, so
            // the two cannot drift apart on a tie.
            pw.ub_data[static_cast<std::size_t>(orow) * cols + ocol] =
                d.pool == Pool::MAX
                  ? best
                  : quant::saturate(quant::round_div(sum, static_cast<int64_t>(w) * w));
        }
    }
}

void Tpu::finish(InFlight& f) {
    // Commit the staged outputs. Until now, a consumer that issued too early would
    // have read the state this instruction is replacing.
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
        // The tile has arrived; shift it into the load plane. The clock is not
        // charged here, because the load bubble was already folded into this
        // instruction's duration.
        WeightTile t;
        if (fifo_.pop(t)) {
            mxu_.load_weights_untimed(t.view());
        } else {
            // Unreachable: the duration covers the DDR latency, so the tile is
            // always ready by now. Counted rather than ignored so a change breaking
            // that assumption surfaces instead of silently loading stale weights.
            ++stalls_.weight_fifo_empty;
        }
    }
    f.active = false;
}

void Tpu::retire_completed(TpuResult& st) {
    // Reverse pipeline order: writeback before issue, so a unit freed this cycle
    // can take new work in the same cycle instead of leaving a dead cycle behind
    // every instruction.
    for (InFlight& f : units_) {
        if (f.active && cycle_ >= f.done_cycle) {
            finish(f);
            ++st.retired;
        }
    }
}

void Tpu::issue_step(const std::vector<RawInst>& prog, TpuResult& st,
                     const TpuOptions& opts) {
    if (st.pc >= prog.size()) {
        // A well-formed program ends in Halt. Wait for the machine to go quiet
        // first, so the retired count and final state match what the oracle
        // reports for the same program.
        if (!quiet()) { ++stalls_.drain; return; }
        st.trapped     = true;
        st.trap_reason = "ran past the end of the program";
        return;
    }

    const Decoded d = decode(prog[st.pc]);

    Reservation res;
    std::string why;
    const bool  ok = !d.trap && validate(d, res, why);
    if (d.trap) why = "illegal opcode";

    if (!ok) {
        // Everything before this instruction must retire before the trap is
        // reported, for the same reason as above.
        if (!quiet()) { ++stalls_.drain; return; }
        st.trapped     = true;
        st.trap_reason = why;
        ++st.retired;
        return;
    }

    // A barrier and a halt both need the machine quiet.
    if (d.op == Op::SYNC || d.op == Op::HALT) {
        if (!quiet()) { ++stalls_.drain; return; }
    }

    const Unit u = unit_of(d.op);
    if (slot(u).active) { ++stalls_.unit_busy; return; }

    // The tile has to have arrived from DDR. With a deep enough FIFO the
    // prefetcher stays far enough ahead that it always has; with a shallow one this
    // is where the array waits on memory.
    if (d.op == Op::READ_WEIGHTS && fifo_.empty()) {
        ++stalls_.weight_fifo_empty;
        return;
    }

    if (interlocked(res)) return;
    if (!ub_port_available(res)) return;

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
        // The steady-state part of the matmul: one row admitted per cycle, every PE
        // multiplying. The rest of the duration is fill and drain.
        profile_.stream_cycles  += d.len;
        profile_.macs_performed += static_cast<uint64_t>(d.len) * cfg_.dim * cfg_.dim;
    }

    ++st.pc;

    // Halt stops the machine here rather than occupying a unit: it only issued
    // because everything was quiet, so there is nothing left for it to wait on.
    // It still spends its issue cycle, like every other instruction.
    if (d.op == Op::HALT) {
        st.halted    = true;
        st.exit_code = d.code;
        ++st.retired;
        ++cycle_;
        return;
    }

    InFlight& f   = slot(u);
    f.active      = true;
    f.op          = d.op;
    f.res         = res;
    f.pending     = std::move(pw);
    f.issue_cycle = cycle_;
    f.done_cycle  = cycle_ + duration;

    // A barrier also only issues when the machine is quiet, so the accumulators
    // it snapshots are final for every instruction before it.
    if (d.op == Op::SYNC) {
        TpuSnapshot s;
        s.acc = acc_.raw();
        st.syncs.push_back(std::move(s));
    }
}

// Keep the weight FIFO as full as it will go, reading ahead in the instruction
// stream for the tiles upcoming Read_Weights will ask for.
//
// A Read_Weights that both requested and consumed its tile would pay the DDR
// latency in full every time, leaving depth irrelevant. Running the fetches ahead
// of the instructions that need them is how the latency is hidden, and the depth
// bounds how far ahead they can get.
//
// Reading ahead is safe because nothing in the ISA writes weight memory, so a tile
// fetched early cannot be stale. An out-of-range address is left alone rather than
// skipped past, so the instruction still traps when it issues.
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
        ++prefetch_pc_;
    }
}

void Tpu::reset_pipeline() {
    for (InFlight& f : units_) f = InFlight{};
    stalls_      = StallStats{};
    profile_     = RunProfile{};
    prefetch_pc_ = 0;
}

// Charge one cycle in which the array stood still to whatever the machine was
// waiting on, given which stall counter moved during this cycle's issue attempt.
//
// Test order is the attribution policy. The weight path, the buffer ports and the
// accumulators are named causes and claim their own stalls first. Anything else
// coinciding with a transfer in flight is charged to data movement, so a starved
// DMA reads as dma_bound rather than hiding behind the read-after-write it
// happens to trigger.
//
// The unit slots suffice to decide this: whichever in-flight instruction the
// scoreboard blocked on is still in flight now, since nothing retires between
// issue and here.
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

TpuResult Tpu::run(const std::vector<RawInst>& prog, const TpuOptions& opts) {
    TpuResult st;
    reset_pipeline();

    const uint64_t start = cycle_;
    while (!st.done()) {
        if (cycle_ - start >= opts.max_cycles) {
            st.budget = true;
            break;
        }

        retire_completed(st);
        if (st.done()) break;

        prefetch_weights(prog);

        const StallStats before = stalls_;
        issue_step(prog, st, opts);

        // Sampled after issue, so the cycle a matmul starts on counts as busy and
        // the cycle it retires on does not. Sampling first would undercount every
        // matmul by one cycle, leaving array_busy short of the duration the array
        // charged for it.
        if (slot(Unit::MXU).active) ++profile_.array_busy;
        else                        charge_idle_cycle(before);

        if (st.done()) break;

        ++cycle_;
        dma_.tick(cycle_);
        fifo_.tick(cycle_);
        mxu_.idle_until(cycle_);
    }

    st.cycles = cycle_ - start;
    pc_       = st.pc;
    return st;
}
