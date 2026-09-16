#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "config.h"
#include "datapath.h"

// A byte-addressable int8 scratchpad, low-order interleaved as addr % banks.
class UnifiedBuffer {
public:
    // Each bank has one read port and one write port per cycle.
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

    // Does a tile fit? The last row needs only `cols` bytes, not a full stride.
    bool tile_fits(UbAddr at, uint32_t rows, uint32_t cols, uint32_t stride) const {
        if (rows == 0 || cols == 0) return in_range(at, 0);
        if (cols > stride) return false;
        return in_range(at, static_cast<std::size_t>(rows - 1) * stride + cols);
    }

    // A strided window onto the buffer's own storage; no copy.
    I8View view(UbAddr at, uint32_t rows, uint32_t cols, uint32_t stride) {
        assert(tile_fits(at, rows, cols, stride));
        return I8View(&mem_[at], rows, cols, stride);
    }
    ConstI8View view(UbAddr at, uint32_t rows, uint32_t cols, uint32_t stride) const {
        assert(tile_fits(at, rows, cols, stride));
        return ConstI8View(&mem_[at], rows, cols, stride);
    }

    // Copy a tile out of / into the buffer; false when it does not fit.
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

    // Do these same-cycle accesses exceed a bank's port budget? Pure, so the
    // sequencer may ask speculatively; whoever stalls calls note_bank_conflict().
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

// int32 accumulator banks, addressed [bank][row][col].
class Accumulators {
public:
    struct Stats {
        uint64_t reads       = 0;
        uint64_t writes      = 0;
        uint64_t accumulates = 0;

        // Accesses refused because the bank was still being written.
        uint64_t hazards = 0;
    };

    explicit Accumulators(const Config& cfg)
        : banks_(cfg.acc_banks), dim_(cfg.dim),
          mem_(static_cast<std::size_t>(cfg.acc_banks) * cfg.dim * cfg.dim, 0),
          locked_(cfg.acc_banks, 0) {
        assert(cfg.acc_banks > 0);
        assert(cfg.dim > 0);
    }

    uint32_t banks() const { return banks_; }
    uint32_t dim() const { return dim_; }
    bool valid(BankId b) const { return b < banks_; }

    const Stats& stats() const { return stats_; }

    // The whole flattened array, for a Sync snapshot.
    const std::vector<i32>& raw() const { return mem_; }

    void lock(BankId b) {
        assert(valid(b));
        locked_[b] = 1;
    }
    void unlock(BankId b) {
        assert(valid(b));
        locked_[b] = 0;
    }
    bool locked(BankId b) const { return valid(b) && locked_[b] != 0; }

    // Unchecked view, for the owner of the bank.
    I32View bank(BankId b) {
        assert(valid(b));
        return I32View(&mem_[offset(b)], dim_, dim_);
    }
    ConstI32View bank(BankId b) const {
        assert(valid(b));
        return ConstI32View(&mem_[offset(b)], dim_, dim_);
    }

    void zero(BankId b) {
        assert(valid(b));
        bank(b).fill(0);
    }

    // Checked entry points; false means an invalid or locked bank.
    bool read(BankId b, const I32View& dst) const {
        if (!valid(b) || locked(b)) {
            if (valid(b)) ++stats_.hazards;
            return false;
        }
        if (dst.rows() > dim_ || dst.cols() > dim_) return false;
        dst.copy_from(bank(b).tile(0, 0, dst.rows(), dst.cols()));
        ++stats_.reads;
        return true;
    }

    bool write(BankId b, const ConstI32View& src) {
        if (!valid(b) || locked(b)) {
            if (valid(b)) ++stats_.hazards;
            return false;
        }
        if (src.rows() > dim_ || src.cols() > dim_) return false;
        bank(b).tile(0, 0, src.rows(), src.cols()).copy_from(src);
        ++stats_.writes;
        return true;
    }

    bool accumulate(BankId b, const ConstI32View& src) {
        if (!valid(b) || locked(b)) {
            if (valid(b)) ++stats_.hazards;
            return false;
        }
        if (src.rows() > dim_ || src.cols() > dim_) return false;
        const I32View dst = bank(b);
        for (uint32_t r = 0; r < src.rows(); ++r) {
            for (uint32_t c = 0; c < src.cols(); ++c) {
                dst.at(r, c) = static_cast<i32>(dst.at(r, c) + src.at(r, c));
            }
        }
        ++stats_.accumulates;
        return true;
    }

private:
    std::size_t offset(BankId b) const {
        return static_cast<std::size_t>(b) * dim_ * dim_;
    }

    uint32_t             banks_;
    uint32_t             dim_;
    std::vector<i32>     mem_;
    std::vector<uint8_t> locked_;
    mutable Stats        stats_;
};
