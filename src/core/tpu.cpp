/// The machine's cycle loop: decode, stall check, issue, then every unit advances.

#include "core/tpu.h"

#include <stdexcept>

Tpu::Tpu()
    : host_("host", kHostMemBytes),
      wmem_("wmem", v1::kWeightMemBytes),
      host_interface_(host_, ub_),
      weight_fifo_(wmem_) {}

void Tpu::load(const Program& program) {
    program_ = program;

    ub_   = UnifiedBuffer();
    host_ = Dram("host", kHostMemBytes);
    wmem_ = Dram("wmem", v1::kWeightMemBytes);
    host_interface_.reset();
    weight_fifo_.reset();

    pc_     = 0;
    halted_ = false;
    stall_  = Stall::None;
    stats_  = Stats();

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

    if (stall_ == Stall::None) {
        try {
            issue(in);
        } catch (const std::exception& e) {
            fail_at_pc(disasm(in) + ": " + e.what());
        }
    }

    host_interface_.tick();
    weight_fifo_.tick();

    stats_.cycles = stats_.cycles + 1;
    if (stall_ != Stall::None) {
        const std::size_t index = static_cast<std::size_t>(stall_);
        stats_.stall_cycles[index] = stats_.stall_cycles[index] + 1;
    }
}

void Tpu::run(u64 cycles) {
    for (u64 i = 0; i < cycles; ++i) {
        if (halted_) {
            return;
        }
        tick();
    }
}

void Tpu::run_to_halt() {
    while (!halted_) {
        tick();
    }
}

void Tpu::step(u64 instructions) {
    const u64 target = stats_.issued + instructions;
    while (!halted_ && stats_.issued < target) {
        tick();
    }
}

void Tpu::step_back(u64 instructions) {
    u64 target = 0;
    if (stats_.issued > instructions) {
        target = stats_.issued - instructions;
    }

    // The machine is deterministic and load() is the only way data gets in, so replaying reaches the same state.
    const Program program = program_;
    load(program);
    while (stats_.issued < target) {
        tick();
    }
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

bool Tpu::units_idle() const {
    const bool host_idle   = !host_interface_.busy();
    const bool weight_idle = !weight_fifo_.fetching();
    return host_idle && weight_idle;
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
        case Op::WriteHostMemory:
            if (host_interface_.busy()) {
                return Stall::HostInterfaceBusy;
            }
            return Stall::None;
        case Op::ReadWeights:
            if (weight_fifo_.full()) {
                return Stall::WeightFifoFull;
            }
            return Stall::None;
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
            break;
        case Op::WriteHostMemory:
            host_interface_.start(Direction::UbToHost, in.host_row, in.ub_row, in.rows);
            break;
        case Op::ReadWeights:
            weight_fifo_.push(in.tile);   // issues at once; the tile arrives over the next ~1366 cycles
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
