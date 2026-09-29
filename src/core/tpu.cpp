/// The machine's cycle loop: decode, stall check, issue, then every unit advances.

#include "core/tpu.h"

#include <stdexcept>

Tpu::Tpu()
    : host_("host", kHostMemBytes),
      wmem_("wmem", v1::kWeightMemBytes),
      host_interface_(host_, ub_),
      weight_fifo_(wmem_),
      mxu_(ub_, acc_, weight_fifo_),
      activation_(acc_, ub_) {}

void Tpu::load(const Program& program) {
    program_ = program;

    ub_   = UnifiedBuffer();
    acc_  = Accumulators();
    host_ = Dram("host", kHostMemBytes);
    wmem_ = Dram("wmem", v1::kWeightMemBytes);
    host_interface_.reset();
    weight_fifo_.reset();
    mxu_.reset();
    activation_.reset();

    pc_     = 0;
    halted_ = false;
    stall_  = Stall::None;
    stats_  = Stats();
    activity_.clear();
    host_pc_ = -1;
    mxu_pc_  = -1;
    act_pc_  = -1;

    for (const DataBlock& block : program_.host) {
        host_.write(block.addr, block.bytes.data(), block.bytes.size());
    }
    for (const DataBlock& block : program_.weights) {
        wmem_.write(block.addr, block.bytes.data(), block.bytes.size());
    }
}

void Tpu::tick() {
    if (halted_) {
        return;
    }
    if (pc_ >= program_.code.size()) {
        fail_at_pc("past the end of the program; is a Halt missing?");
    }

    // An error here leaves the machine exactly as it was before this cycle.
    Instr in;
    try {
        in = decode(program_.code[pc_]);
    } catch (const std::invalid_argument& e) {
        fail_at_pc(e.what());
    }

    stall_ = why_blocked(in);

    // Blocked with every unit idle means nothing can ever unblock it.
    if (stall_ != Stall::None && units_idle()) {
        fail_at_pc(disasm(in) + ": deadlock: " + stall_name(stall_) + ", but no unit is working");
    }

    i32 issued_pc = -1;
    if (stall_ == Stall::None) {
        issued_pc = static_cast<i32>(pc_);
        try {
            issue(in);
        } catch (const std::exception& e) {
            fail_at_pc(disasm(in) + ": " + e.what());
        }
    }

    const u32 shifted_before = mxu_.rows_shifted();
    const int shadow_before  = mxu_.shadow_tile();
    const bool mxu_working   = mxu_.busy();

    record_cycle(issued_pc);

    host_interface_.tick();
    weight_fifo_.tick();
    mxu_.tick();
    activation_.tick();

    // The shifter's work shows only afterwards: a tile row moved if the count changed or a new tile started.
    const bool shifted = mxu_.rows_shifted() != shifted_before || mxu_.shadow_tile() != shadow_before;
    if (shifted) {
        activity_.back().shifting = mxu_.shadow_tile();
    }
    if (mxu_working) {
        stats_.mxu_busy_cycles = stats_.mxu_busy_cycles + 1;
    }

    stats_.cycles = stats_.cycles + 1;
    if (stall_ != Stall::None) {
        const std::size_t index = static_cast<std::size_t>(stall_);
        stats_.stall_cycles[index] = stats_.stall_cycles[index] + 1;
    }
}

// What each unit works on this cycle, taken before they advance so a unit that finishes now still counts.
void Tpu::record_cycle(i32 issued_pc) {
    CycleRecord record;
    record.stall  = stall_;
    record.issued = issued_pc;
    if (host_interface_.busy()) {
        record.host = host_pc_;
    }
    if (mxu_.busy()) {
        record.mxu = mxu_pc_;
    }
    if (activation_.busy()) {
        record.activation = act_pc_;
    }

    // Tiles arrive in order, so the first one not ready is the one DDR3 is filling.
    for (const FifoTile& entry : weight_fifo_.tiles()) {
        if (!entry.ready()) {
            record.fetching = static_cast<i32>(entry.tile);
            break;
        }
    }

    activity_.push_back(record);
    if (activity_.size() > kTimelineCycles) {
        activity_.pop_front();
    }
}

void Tpu::run_cycles(u64 cycles) {
    for (u64 i = 0; i < cycles; ++i) {
        if (halted_) {
            return;
        }
        tick();
    }
}

void Tpu::back_cycles(u64 cycles) {
    Cycle target = 0;
    if (stats_.cycles > cycles) {
        target = stats_.cycles - cycles;
    }

    reload();
    while (stats_.cycles < target) {
        tick();
    }
}

void Tpu::next_instructions(u64 instructions) {
    const u64 target = stats_.issued + instructions;
    while (!halted_ && stats_.issued < target) {
        tick();
    }
}

void Tpu::prev_instructions(u64 instructions) {
    u64 target = 0;
    if (stats_.issued > instructions) {
        target = stats_.issued - instructions;
    }

    reload();
    while (stats_.issued < target) {
        tick();
    }
}

void Tpu::run_to_halt() {
    while (!halted_) {
        tick();
    }
}

// The machine is deterministic and load() is the only way data gets in, so replaying reaches the same state.
void Tpu::reload() {
    const Program program = program_;
    load(program);
}

bool Tpu::halted() const {
    return halted_;
}

Cycle Tpu::cycle() const {
    return stats_.cycles;
}

u32 Tpu::pc() const {
    return pc_;
}

u64 Tpu::issued() const {
    return stats_.issued;
}

Stall Tpu::stall() const {
    return stall_;
}

const Stats& Tpu::stats() const {
    return stats_;
}

const Program& Tpu::program() const {
    return program_;
}

const UnifiedBuffer& Tpu::ub() const {
    return ub_;
}

const Dram& Tpu::host() const {
    return host_;
}

const Dram& Tpu::wmem() const {
    return wmem_;
}

const HostInterface& Tpu::host_interface() const {
    return host_interface_;
}

const WeightFifo& Tpu::weight_fifo() const {
    return weight_fifo_;
}

const Accumulators& Tpu::acc() const {
    return acc_;
}

const SystolicArray& Tpu::mxu() const {
    return mxu_;
}

const ActivationUnit& Tpu::activation() const {
    return activation_;
}

const std::deque<CycleRecord>& Tpu::activity() const {
    return activity_;
}

bool Tpu::units_idle() const {
    const bool host_idle   = !host_interface_.busy();
    const bool weight_idle = !weight_fifo_.fetching();
    const bool mxu_idle    = !mxu_.busy() && !mxu_.shifting();
    const bool act_idle    = !activation_.busy();
    return host_idle && weight_idle && mxu_idle && act_idle;
}

// Stall::None means the instruction can issue this cycle.
Stall Tpu::why_blocked(const Instr& in) const {
    switch (in.op) {
        case Op::Nop:
            return Stall::None;
        case Op::Halt:
        case Op::Sync:
            if (!units_idle()) {
                return Stall::WaitForIdle;
            }
            return Stall::None;
        case Op::ReadHostMemory:
            if (host_interface_.busy()) {
                return Stall::HostInterfaceBusy;
            }
            if (mxu_.reads_ub_rows(in.ub_row, in.rows)) {
                return Stall::UbNotReady;   // it would overwrite rows the MXU is still reading
            }
            if (activation_.writes_ub_rows(in.ub_row, in.rows)) {
                return Stall::UbNotReady;   // its bytes must land after the Activate's
            }
            return Stall::None;
        case Op::WriteHostMemory:
            if (host_interface_.busy()) {
                return Stall::HostInterfaceBusy;
            }
            if (activation_.writes_ub_rows(in.ub_row, in.rows)) {
                return Stall::UbNotReady;   // its rows are still being written by an Activate
            }
            return Stall::None;
        case Op::ReadWeights:
            if (weight_fifo_.full()) {
                return Stall::WeightFifoFull;
            }
            return Stall::None;
        case Op::MatrixMultiply:
            if (mxu_.busy()) {
                return Stall::MxuBusy;
            }
            if (in.new_weights == 1 && !mxu_.shadow_ready()) {
                return Stall::WeightsNotReady;
            }
            if (host_interface_.writes_ub_rows(in.ub_row, in.rows)) {
                return Stall::UbNotReady;   // its input rows are still arriving from the host
            }
            if (activation_.writes_ub_rows(in.ub_row, in.rows)) {
                return Stall::UbNotReady;   // its input rows are still being written by an Activate
            }
            if (activation_.reads_acc_rows(in.acc_row, in.rows)) {
                return Stall::AccNotReady;  // it would overwrite sums an Activate has not read yet
            }
            return Stall::None;
        case Op::Activate: {
            // With pooling it writes fewer UB rows than it reads accumulator rows.
            const u32 out_rows = activate_output_rows(in.rows, in.pool, in.pool_size);
            if (activation_.busy()) {
                return Stall::ActivationBusy;
            }
            if (mxu_.writes_acc_rows(in.acc_row, in.rows)) {
                return Stall::AccNotReady;  // its sums are still coming out of the MXU
            }
            if (host_interface_.writes_ub_rows(in.ub_row, out_rows)) {
                return Stall::UbNotReady;   // its rows must land after the host's
            }
            if (host_interface_.reads_ub_rows(in.ub_row, out_rows)) {
                return Stall::UbNotReady;   // it would overwrite rows still going to the host
            }
            if (mxu_.reads_ub_rows(in.ub_row, out_rows)) {
                return Stall::UbNotReady;   // it would overwrite rows the MXU is still reading
            }
            return Stall::None;
        }
    }
    return Stall::None;
}

void Tpu::issue(const Instr& in) {
    switch (in.op) {
        case Op::Nop:
        case Op::Sync:
            break;
        case Op::Halt:
            halted_ = true;
            break;
        case Op::ReadHostMemory:
            host_interface_.start(Direction::HostToUb, in.host_row, in.ub_row, in.rows);
            host_pc_ = static_cast<i32>(pc_);
            break;
        case Op::WriteHostMemory:
            host_interface_.start(Direction::UbToHost, in.host_row, in.ub_row, in.rows);
            host_pc_ = static_cast<i32>(pc_);
            break;
        case Op::ReadWeights:
            weight_fifo_.push(in.tile);   // issues at once; the tile arrives over the next ~1366 cycles
            break;
        case Op::MatrixMultiply:
            mxu_.start(in.ub_row, in.acc_row, in.rows, in.accumulate == 1, in.new_weights == 1);
            mxu_pc_ = static_cast<i32>(pc_);
            break;
        case Op::Activate:
            activation_.start(in.acc_row, in.ub_row, in.rows, in.shift, in.function, in.pool, in.pool_size, in.pool_width);
            act_pc_ = static_cast<i32>(pc_);
            break;
    }

    stats_.issued = stats_.issued + 1;
    if (!halted_) {
        pc_ = pc_ + 1;   // Halt keeps the PC on itself
    }
}

void Tpu::fail_at_pc(const std::string& what) const {
    throw std::runtime_error("pc " + std::to_string(pc_) + ": " + what);
}
