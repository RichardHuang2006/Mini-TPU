#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "config.h"
#include "tensor.h"
#include "types.h"

// The Unified Buffer: a banked, byte-addressable scratchpad holding activations
// and intermediate results. Addresses are byte offsets and tensors are row-major
// with an explicit stride, so a tile is a strided rectangular window rather than
// a copy.
//
// Banking is low-order interleaved, bank = addr % banks, so a contiguous row of
// activations spreads across every bank instead of piling into one. That is the
// whole point of banking here: the array consumes dim consecutive bytes per
// cycle, and interleaving is what lets those arrive through dim different ports.
// A block-partitioned layout (addr / bytes_per_bank) would put an entire row in
// one bank and serialize precisely the access pattern the machine performs most.
class UnifiedBuffer {
public:
    // Each bank has one read port and one write port per cycle, so the direction
    // of an access is part of whether it collides.
    enum class Port { READ, WRITE };

    struct Access {
        UbAddr addr = 0;
        Port   port = Port::READ;
    };

    struct Stats {
        uint64_t tile_reads     = 0;
        uint64_t tile_writes    = 0;
        uint64_t bytes_read     = 0;
        uint64_t bytes_written  = 0;
        uint64_t bank_conflicts = 0;
    };

    explicit UnifiedBuffer(const Config& cfg)
        : banks_(cfg.ub_banks), mem_(cfg.ub_bytes, 0) {
        assert(cfg.ub_banks > 0);
        assert(cfg.ub_bytes > 0);
    }

    uint32_t bytes() const { return static_cast<uint32_t>(mem_.size()); }
    uint32_t banks() const { return banks_; }

    BankId bank_of(UbAddr a) const { return a % banks_; }

    const Stats& stats() const { return stats_; }

    bool in_range(UbAddr at, std::size_t n) const {
        return at <= mem_.size() && n <= mem_.size() - at;
    }

    // Does a rows x cols tile with this row pitch fit? The last row only needs
    // `cols` bytes, not a full stride, which matters for a tile that ends flush
    // against the top of the buffer.
    bool tile_fits(UbAddr at, uint32_t rows, uint32_t cols, uint32_t stride) const {
        if (rows == 0 || cols == 0) return in_range(at, 0);
        if (cols > stride) return false;
        return in_range(at, static_cast<std::size_t>(rows - 1) * stride + cols);
    }

    // A strided window onto the buffer's own storage. This is how the array gets
    // its activations: no copy, so a tile of a much wider activation matrix costs
    // nothing to address.
    I8View view(UbAddr at, uint32_t rows, uint32_t cols, uint32_t stride) {
        assert(tile_fits(at, rows, cols, stride));
        return I8View(&mem_[at], rows, cols, stride);
    }
    ConstI8View view(UbAddr at, uint32_t rows, uint32_t cols, uint32_t stride) const {
        assert(tile_fits(at, rows, cols, stride));
        return ConstI8View(&mem_[at], rows, cols, stride);
    }

    // Copy a tile out of / into the buffer. `dst` and `src` supply the shape;
    // `stride` is the pitch on the buffer side, so a narrow tile can be lifted
    // out of a wide region. Returns false rather than trapping when the tile does
    // not fit, leaving the decision about what a bad address means to the caller.
    bool read_tile(UbAddr at, uint32_t stride, const I8View& dst) const {
        if (!tile_fits(at, dst.rows(), dst.cols(), stride)) return false;
        dst.copy_from(view(at, dst.rows(), dst.cols(), stride));
        ++stats_.tile_reads;
        stats_.bytes_read += static_cast<uint64_t>(dst.rows()) * dst.cols();
        return true;
    }

    bool write_tile(UbAddr at, uint32_t stride, const ConstI8View& src) {
        if (!tile_fits(at, src.rows(), src.cols(), stride)) return false;
        view(at, src.rows(), src.cols(), stride).copy_from(src);
        ++stats_.tile_writes;
        stats_.bytes_written += static_cast<uint64_t>(src.rows()) * src.cols();
        return true;
    }

    // Would these accesses, all in one cycle, exceed a bank's port budget? Two
    // accesses in the same direction to one bank collide; a read and a write to
    // the same bank do not, because each bank has one port of each.
    //
    // Pure, so the sequencer can ask speculatively before deciding to stall.
    // Whoever actually stalls calls note_bank_conflict().
    bool port_conflict(const std::vector<Access>& accesses) const {
        std::vector<uint8_t> read_seen(banks_, 0), write_seen(banks_, 0);
        for (const Access& a : accesses) {
            const BankId  b    = bank_of(a.addr);
            uint8_t&      seen = (a.port == Port::READ) ? read_seen[b] : write_seen[b];
            if (seen) return true;
            seen = 1;
        }
        return false;
    }

    // Convenience for the common all-reads question.
    bool port_conflict(const std::vector<UbAddr>& addrs) const {
        std::vector<Access> as;
        as.reserve(addrs.size());
        for (const UbAddr a : addrs) as.push_back(Access{a, Port::READ});
        return port_conflict(as);
    }

    void note_bank_conflict() { ++stats_.bank_conflicts; }

    // Byte access, for the DMA engine and for tests setting up state.
    i8&       at(UbAddr a)       { assert(a < mem_.size()); return mem_[a]; }
    const i8& at(UbAddr a) const { assert(a < mem_.size()); return mem_[a]; }

private:
    uint32_t        banks_;
    std::vector<i8> mem_;
    mutable Stats   stats_;
};
