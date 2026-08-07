#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "config.h"
#include "tensor.h"
#include "types.h"

// int32 accumulator banks, addressed [bank][row][col].
//
// A MatMul either overwrites a bank or accumulates in place into it, and the
// second mode is how the K dimension is tiled: successive MatMuls over slices of
// K add into the same bank, and one Activate reads the finished sum out. The
// bank, not the array, is what makes K-tiling possible -- the array only ever
// sees dim of the K dimension at a time.
//
// Locking has a deliberate asymmetry. bank() hands out an unchecked view for
// whoever owns the bank (the in-flight matmul writing into it), while read(),
// write() and accumulate() are the consumer-facing entry points that refuse a
// locked bank and count a hazard. Without that split, the matmul holding the lock
// would be blocked by its own lock.
class Accumulators {
public:
    struct Stats {
        uint64_t reads       = 0;
        uint64_t writes      = 0;
        uint64_t accumulates = 0;

        // Accesses refused because the bank was still being written. This is the
        // accum_hazard of the stall breakdown.
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

    // Checked entry points. Each returns false when the access cannot proceed --
    // an invalid bank, or a locked one, in which case the caller stalls.
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
