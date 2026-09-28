/// Off-chip DRAM (host memory, Weight Memory): byte-addressed, with 64 KiB pages allocated on first write.

#pragma once

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/config.h"

class Dram {
public:
    static constexpr u64 kPageBytes = v1::kTileBytes;   // a weight tile fills exactly one page

    Dram(std::string name, u64 bytes) : name_(std::move(name)), bytes_(bytes) {}

    u64 size() const { return bytes_; }
    u64 pages_allocated() const { return pages_.size(); }

    void write(u64 addr, const i8* src, u64 n) {
        check_range(addr, n);
        while (n > 0) {
            const u64 off   = addr % kPageBytes;
            const u64 chunk = std::min(n, kPageBytes - off);
            std::vector<i8>& page = pages_[addr / kPageBytes];
            if (page.empty()) page.assign(kPageBytes, 0);
            std::memcpy(page.data() + off, src, chunk);
            addr += chunk;
            src  += chunk;
            n    -= chunk;
        }
    }

    // Bytes never written read as zero, without allocating a page.
    void read(u64 addr, i8* dst, u64 n) const {
        check_range(addr, n);
        while (n > 0) {
            const u64 off   = addr % kPageBytes;
            const u64 chunk = std::min(n, kPageBytes - off);
            const auto it   = pages_.find(addr / kPageBytes);
            if (it == pages_.end()) std::memset(dst, 0, chunk);
            else                    std::memcpy(dst, it->second.data() + off, chunk);
            addr += chunk;
            dst  += chunk;
            n    -= chunk;
        }
    }

private:
    std::string name_;   // "host" or "wmem", for error messages
    u64         bytes_;
    std::unordered_map<u64, std::vector<i8>> pages_;   // page number -> its bytes

    void check_range(u64 addr, u64 n) const {
        if (n > bytes_ || addr > bytes_ - n)
            throw std::out_of_range(name_ + " bytes [" + std::to_string(addr) + ", +" + std::to_string(n) +
                                    ") run past its " + std::to_string(bytes_) + " bytes");
    }
};
