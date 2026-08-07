#include "tpu.h"

#include <algorithm>

Tpu::Tpu(const Config& cfg, std::size_t host_bytes, std::size_t weight_bytes)
    : cfg_(cfg), ub_(cfg), acc_(cfg), fifo_(cfg), dma_(cfg), mxu_(cfg),
      host_(host_bytes, 0), weight_mem_(weight_bytes, 0) {}

void Tpu::tick() {
    ++cycle_;

    // Reverse pipeline order: writeback before compute before issue before fetch,
    // so a stage reads what its producer had at the end of the previous cycle
    // rather than something produced earlier in this one. Only the time-based
    // units have anything to do while the machine is driven by hand; Phase 5
    // inserts the sequencer stages between them, and the order they go in is
    // fixed here so that is a matter of filling in bodies.
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
    // duration: a consumer that read it now would see a partial result. Driven by
    // hand the matmul completes within the call, so the lock is released
    // immediately; in Phase 5 it is held until the matmul actually retires.
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
