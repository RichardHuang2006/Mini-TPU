// Writes a trace container (.mtpt) for the visualizer under viz/.
//
// The container is one file: a JSON manifest followed by 8-byte-aligned binary
// sections the manifest indexes. It holds everything the machine did, cycle by
// cycle: every issue and retire with the bytes read and committed, every stall
// with its reason and blocker, every prefetch and weight-plane load, and for
// every MatMul the whole PE grid after every array step.
//
// Nothing here simulates. A TraceSink (src/trace.h) records what Tpu::run does;
// the generator then runs the same workload untraced and asserts that tracing
// changed nothing, compares the run to the eager oracle and to the golden loops,
// and re-checks the trace against itself (PE arithmetic, landings, commits,
// requantization, stall and idle partitions). The results go into the manifest.
//
//   tracegen --small --out viz/traces/matmul_8.mtpt
//   tracegen --prog examples/matmul_128.hex --acts ... --weights ... \
//            --expect examples/matmul_128.expect.mtpu --layer 128,128,128 \
//            --dim 32 --ub 262144 --acc-banks 4 --macs 2097152 \
//            --out viz/traces/matmul_128.mtpt

#define MINI_TPU_NO_ENTRY
#include "main.cpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "ref.h"
#include "trace.h"
#include "workloads.h"

namespace {

// ============================================================================
// JSON writer
// ============================================================================

std::string jstr(const std::string& s) {
    std::string o = "\"";
    for (const char ch : s) {
        switch (ch) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof b, "\\u%04x", ch);
                    o += b;
                } else {
                    o += ch;
                }
        }
    }
    return o + "\"";
}

class JW {
public:
    std::string s;

    void obj()     { val(); s += '{'; st_.push_back(true); }
    void end_obj() { s += '}'; st_.pop_back(); }
    void arr()     { val(); s += '['; st_.push_back(true); }
    void end_arr() { s += ']'; st_.pop_back(); }

    void key(const std::string& k) { comma(); s += jstr(k); s += ':'; pending_ = true; }

    void num(long long v)           { val(); s += std::to_string(v); }
    void unum(unsigned long long v) { val(); s += std::to_string(v); }
    void dbl(double v) {
        val();
        char b[64];
        std::snprintf(b, sizeof b, "%.10g", v);
        s += b;
    }
    void str(const std::string& v) { val(); s += jstr(v); }
    void boolean(bool b)           { val(); s += b ? "true" : "false"; }
    void null()                    { val(); s += "null"; }

    // Shorthands for "key": value.
    void kv(const std::string& k, long long v)          { key(k); num(v); }
    void kvu(const std::string& k, unsigned long long v){ key(k); unum(v); }
    void kv(const std::string& k, const std::string& v) { key(k); str(v); }
    void kv(const std::string& k, const char* v)        { key(k); str(v); }
    void kvb(const std::string& k, bool v)              { key(k); boolean(v); }
    void kvd(const std::string& k, double v)            { key(k); dbl(v); }

private:
    std::vector<bool> st_;
    bool              pending_ = false;

    void comma() {
        if (st_.empty()) return;
        if (!st_.back()) s += ',';
        st_.back() = false;
    }
    void val() {
        if (pending_) { pending_ = false; return; }
        comma();
    }
};

// ============================================================================
// Container sections
// ============================================================================

struct Section {
    std::string          name;
    std::string          dtype;   // u8 i8 u32 i32
    std::size_t          count = 0;
    std::vector<uint8_t> bytes;
};

template <typename T>
Section make_section(const std::string& name, const std::string& dtype, const std::vector<T>& v) {
    Section s;
    s.name  = name;
    s.dtype = dtype;
    s.count = v.size();
    s.bytes.resize(v.size() * sizeof(T));
    if (!v.empty()) std::memcpy(s.bytes.data(), v.data(), s.bytes.size());
    return s;
}

// ============================================================================
// The recorder
// ============================================================================

struct CommitRec {
    uint64_t    cycle = 0;
    std::size_t pc    = 0;
    char        kind  = 'U';   // U unified buffer, H host, A accumulator bank, P weight plane
    uint64_t    addr  = 0;     // byte address, bank id, or plane id
    uint32_t    rows  = 0;
    uint32_t    cols  = 0;
    std::size_t len   = 0;     // bytes
    std::size_t off_after  = 0;
    std::size_t off_before = 0;
};

struct ReadRec {
    uint64_t    cycle = 0;
    std::size_t pc    = 0;
    char        kind  = 'U';   // U ub bytes, H host bytes, A bank rows (int32)
    uint64_t    addr  = 0;
    uint32_t    rows  = 0;
    uint32_t    cols  = 0;
    std::size_t len   = 0;
    std::size_t off   = 0;
};

struct IssueRec {
    uint64_t              cycle = 0;
    std::size_t           pc    = 0;
    Op                    op    = Op::NOP;
    Unit                  unit  = Unit::SEQ;
    uint64_t              duration = 0;
    uint64_t              done     = 0;
    Reservation           res;
    std::vector<uint32_t> reads;
    int                   mm  = -1;
    int                   act = -1;
};

struct RetireRec {
    uint64_t              cycle = 0;
    Unit                  unit  = Unit::SEQ;
    std::size_t           pc    = 0;
    Op                    op    = Op::NOP;
    std::vector<uint32_t> commits;
};

struct StallSpan {
    uint64_t    from = 0, to = 0;
    std::size_t pc         = 0;
    StallReason reason     = StallReason::NONE;
    Unit        blocker    = Unit::COUNT;
    std::size_t blocker_pc = 0;
};

struct PrefetchRec {
    uint64_t    cycle  = 0;
    std::size_t for_pc = 0;
    uint32_t    ddr    = 0;
    uint64_t    ready  = 0;
    std::size_t occ    = 0;
};

struct WloadRec {
    uint64_t    cycle   = 0;
    std::size_t pc      = 0;
    uint32_t    ddr     = 0;
    uint32_t    plane   = 0;
    bool        pending = false;
    uint32_t    commit  = 0;
};

struct MmRec {
    std::size_t pc     = 0;
    uint64_t    issue  = 0;
    uint64_t    retire = 0;
    uint32_t    len    = 0;
    uint32_t    plane  = 0;
    bool        switched   = false;
    bool        accumulate = false;
    uint32_t    bank   = 0;
    uint64_t    steps  = 0;
    int         acc_in_read = -1;   // read record of the bank rows, when accumulating
    std::vector<i8>  w;             // active plane, dim x dim
    std::vector<i8>  left;          // steps x dim
    std::vector<i8>  act;           // steps x dim x dim, after each step's commit
    std::vector<i32> psum;          // steps x dim x dim
    std::vector<i32> land;          // len x dim: the value that left the bottom edge
};

struct ActRec {
    std::size_t pc     = 0;
    uint64_t    issue  = 0;
    uint64_t    retire = 0;
    uint32_t    bank   = 0;
    uint32_t    len    = 0;
    uint32_t    out_rows = 0, out_cols = 0;
    int         read    = -1;
    int         commit  = -1;
};

struct CycleRow {
    uint8_t  outcome = 0;   // 0 issued, 1..9 StallReason, 10 trap, 11 halt
    uint32_t pc      = 0;
    uint8_t  blocker_unit = 255;
    uint32_t blocker_pc   = 0;
    uint8_t  idle   = 0;
    uint8_t  units  = 0;
    uint8_t  fifo_occ = 0, fifo_ready = 0;
    uint8_t  ub_r = 0, ub_w = 0;
    uint8_t  plane = 0, pending = 0;
};

struct Recorder : TraceSink {
    uint32_t dim = 0;

    // Shadow memories, so every commit can carry its before-image.
    std::vector<uint8_t> sh_host;
    std::vector<i8>      sh_ub, sh_ddr;
    std::vector<i32>     sh_acc;
    std::vector<i8>      sh_plane[2];

    std::vector<uint8_t> init_host;
    std::vector<i8>      init_ddr;

    std::vector<CommitRec>   commits;
    std::vector<uint8_t>     commit_blob;
    std::vector<ReadRec>     reads;
    std::vector<uint8_t>     read_blob;
    std::vector<IssueRec>    issues;
    std::vector<RetireRec>   retires;
    std::vector<StallSpan>   stalls;
    std::vector<PrefetchRec> prefetches;
    std::vector<WloadRec>    wloads;
    std::vector<MmRec>       mms;
    std::vector<ActRec>      acts;
    std::vector<CycleRow>    rows;
    std::vector<i32>         sync_blob;
    std::vector<std::pair<uint64_t, std::size_t>> syncs;    // cycle, pc
    std::vector<std::pair<uint64_t, std::size_t>> halts;
    std::vector<std::pair<uint64_t, std::string>> traps;
    std::map<std::size_t, int> mm_of_pc, act_of_pc;
    int cur_mm = -1;

    static void align4(std::vector<uint8_t>& b) { while (b.size() % 4) b.push_back(0); }

    template <typename T>
    static std::size_t append(std::vector<uint8_t>& b, const T* p, std::size_t n) {
        align4(b);
        const std::size_t off = b.size();
        b.resize(off + n * sizeof(T));
        if (n) std::memcpy(b.data() + off, p, n * sizeof(T));
        return off;
    }

    void on_run_begin(const Tpu& tpu) override {
        dim       = tpu.config().dim;
        init_host = tpu.host();
        init_ddr  = tpu.weight_mem();
        sh_host   = init_host;
        sh_ddr    = init_ddr;
        sh_ub.assign(tpu.ub().bytes(), 0);
        for (uint32_t a = 0; a < tpu.ub().bytes(); ++a) sh_ub[a] = tpu.ub().at(a);
        sh_acc = tpu.acc().raw();
        for (uint32_t p = 0; p < 2; ++p) {
            sh_plane[p].assign(static_cast<std::size_t>(dim) * dim, 0);
            for (uint32_t k = 0; k < dim; ++k)
                for (uint32_t c = 0; c < dim; ++c) sh_plane[p][k * dim + c] = tpu.mxu().weight_at(k, c, p);
        }
    }

    // ---- retire: commits with before-images -------------------------------
    void on_retire(const Tpu&, uint64_t cycle, Unit unit, const InFlight& f) override {
        RetireRec r;
        r.cycle = cycle;
        r.unit  = unit;
        r.pc    = f.pc;
        r.op    = f.op;
        const PendingWrite& pw = f.pending;

        if (!pw.ub_data.empty()) {
            CommitRec c;
            c.cycle = cycle; c.pc = f.pc; c.kind = 'U'; c.addr = pw.ub_at;
            c.rows = 1; c.cols = static_cast<uint32_t>(pw.ub_data.size()); c.len = pw.ub_data.size();
            c.off_after  = append(commit_blob, pw.ub_data.data(), pw.ub_data.size());
            c.off_before = append(commit_blob, sh_ub.data() + pw.ub_at, pw.ub_data.size());
            std::copy(pw.ub_data.begin(), pw.ub_data.end(), sh_ub.begin() + pw.ub_at);
            r.commits.push_back(static_cast<uint32_t>(commits.size()));
            commits.push_back(c);
        }
        if (!pw.host_data.empty()) {
            CommitRec c;
            c.cycle = cycle; c.pc = f.pc; c.kind = 'H'; c.addr = pw.host_at;
            c.rows = 1; c.cols = static_cast<uint32_t>(pw.host_data.size()); c.len = pw.host_data.size();
            c.off_after  = append(commit_blob, pw.host_data.data(), pw.host_data.size());
            c.off_before = append(commit_blob, sh_host.data() + pw.host_at, pw.host_data.size());
            std::copy(pw.host_data.begin(), pw.host_data.end(), sh_host.begin() + pw.host_at);
            r.commits.push_back(static_cast<uint32_t>(commits.size()));
            commits.push_back(c);
        }
        if (pw.bank != INVALID_BANK) {
            const std::size_t n    = static_cast<std::size_t>(pw.acc_rows) * dim;
            const std::size_t base = static_cast<std::size_t>(pw.bank) * dim * dim;
            CommitRec c;
            c.cycle = cycle; c.pc = f.pc; c.kind = 'A'; c.addr = pw.bank;
            c.rows = pw.acc_rows; c.cols = dim; c.len = n * sizeof(i32);
            c.off_after  = append(commit_blob, pw.acc_data.data(), n);
            c.off_before = append(commit_blob, sh_acc.data() + base, n);
            std::copy(pw.acc_data.begin(), pw.acc_data.begin() + static_cast<std::ptrdiff_t>(n),
                      sh_acc.begin() + static_cast<std::ptrdiff_t>(base));
            r.commits.push_back(static_cast<uint32_t>(commits.size()));
            commits.push_back(c);
        }
        if (f.op == Op::MATMUL) {
            auto it = mm_of_pc.find(f.pc);
            if (it != mm_of_pc.end()) mms[static_cast<std::size_t>(it->second)].retire = cycle;
        }
        if (f.op == Op::ACTIVATE) {
            auto it = act_of_pc.find(f.pc);
            if (it != act_of_pc.end()) {
                ActRec& a = acts[static_cast<std::size_t>(it->second)];
                a.retire = cycle;
                if (!r.commits.empty()) a.commit = static_cast<int>(r.commits.front());
            }
        }
        retires.push_back(std::move(r));
    }

    void on_weight_load(const Tpu& tpu, uint64_t cycle, std::size_t pc, const WeightTile& tile,
                        uint32_t plane, bool pending) override {
        std::vector<i8> after(static_cast<std::size_t>(dim) * dim, 0);
        for (uint32_t k = 0; k < dim; ++k)
            for (uint32_t c = 0; c < dim; ++c) after[k * dim + c] = tpu.mxu().weight_at(k, c, plane);

        CommitRec c;
        c.cycle = cycle; c.pc = pc; c.kind = 'P'; c.addr = plane;
        c.rows = dim; c.cols = dim; c.len = after.size();
        c.off_after  = append(commit_blob, after.data(), after.size());
        c.off_before = append(commit_blob, sh_plane[plane].data(), after.size());
        sh_plane[plane] = after;
        const uint32_t id = static_cast<uint32_t>(commits.size());
        commits.push_back(c);
        if (!retires.empty() && retires.back().pc == pc && retires.back().cycle == cycle) {
            retires.back().commits.push_back(id);
        }

        WloadRec w;
        w.cycle = cycle; w.pc = pc; w.ddr = tile.ddr_addr; w.plane = plane; w.pending = pending;
        w.commit = id;
        wloads.push_back(w);
    }

    void on_prefetch(const Tpu&, uint64_t cycle, std::size_t for_pc, const WeightTile& tile,
                     uint64_t ready, std::size_t occ) override {
        PrefetchRec p;
        p.cycle = cycle; p.for_pc = for_pc; p.ddr = tile.ddr_addr; p.ready = ready; p.occ = occ;
        prefetches.push_back(p);
    }

    void on_stall(const Tpu&, uint64_t cycle, std::size_t pc, const Decoded&, StallReason reason,
                  Unit blocker, std::size_t blocker_pc) override {
        if (!stalls.empty()) {
            StallSpan& s = stalls.back();
            if (s.pc == pc && s.reason == reason && s.blocker == blocker &&
                s.blocker_pc == blocker_pc && s.to + 1 == cycle) {
                s.to = cycle;
                return;
            }
        }
        StallSpan s;
        s.from = s.to = cycle; s.pc = pc; s.reason = reason; s.blocker = blocker;
        s.blocker_pc = blocker_pc;
        stalls.push_back(s);
    }

    // ---- issue: input snapshots ------------------------------------------
    uint32_t add_read(uint64_t cycle, std::size_t pc, char kind, uint64_t addr, uint32_t rows,
                      uint32_t cols, const void* data, std::size_t bytes, std::size_t elem) {
        ReadRec r;
        r.cycle = cycle; r.pc = pc; r.kind = kind; r.addr = addr; r.rows = rows; r.cols = cols;
        r.len = bytes;
        r.off = append(read_blob, static_cast<const uint8_t*>(data), bytes);
        (void)elem;
        reads.push_back(r);
        return static_cast<uint32_t>(reads.size() - 1);
    }

    void on_issue(const Tpu& tpu, uint64_t cycle, std::size_t pc, const Decoded& d, Unit unit,
                  const Reservation& res, uint64_t duration, const PendingWrite& pw) override {
        IssueRec ir;
        ir.cycle = cycle; ir.pc = pc; ir.op = d.op; ir.unit = unit; ir.duration = duration;
        ir.done = cycle + duration; ir.res = res;

        switch (d.op) {
            case Op::READ_HOST:
                ir.reads.push_back(add_read(cycle, pc, 'H', d.host_addr, 1, d.bytes,
                                            tpu.host().data() + d.host_addr, d.bytes, 1));
                break;
            case Op::WRITE_HOST: {
                std::vector<i8> b(d.bytes);
                for (uint32_t i = 0; i < d.bytes; ++i) b[i] = tpu.ub().at(d.ub_addr + i);
                ir.reads.push_back(add_read(cycle, pc, 'U', d.ub_addr, 1, d.bytes, b.data(), d.bytes, 1));
                break;
            }
            case Op::MATMUL: {
                const std::size_t n = static_cast<std::size_t>(d.len) * dim;
                std::vector<i8> a(n);
                for (std::size_t i = 0; i < n; ++i) a[i] = tpu.ub().at(static_cast<UbAddr>(d.ub_addr + i));
                ir.reads.push_back(add_read(cycle, pc, 'U', d.ub_addr, d.len, dim, a.data(), n, 1));
                if (cur_mm >= 0) {
                    MmRec& m = mms[static_cast<std::size_t>(cur_mm)];
                    m.pc = pc; m.issue = cycle; m.accumulate = d.accumulate; m.bank = d.acc_bank;
                    if (d.accumulate) {
                        const ConstI32View bank = tpu.acc().bank(d.acc_bank);
                        std::vector<i32> rows_(n);
                        for (uint32_t r = 0; r < d.len; ++r)
                            for (uint32_t c = 0; c < dim; ++c) rows_[r * dim + c] = bank.at(r, c);
                        const uint32_t id = add_read(cycle, pc, 'A', d.acc_bank, d.len, dim, rows_.data(),
                                                     n * sizeof(i32), 4);
                        ir.reads.push_back(id);
                        m.acc_in_read = static_cast<int>(id);
                    }
                    ir.mm = cur_mm;
                    mm_of_pc[pc] = cur_mm;
                }
                break;
            }
            case Op::ACTIVATE: {
                const std::size_t  n    = static_cast<std::size_t>(d.len) * dim;
                const ConstI32View bank = tpu.acc().bank(d.acc_bank);
                std::vector<i32> rows_(n);
                for (uint32_t r = 0; r < d.len; ++r)
                    for (uint32_t c = 0; c < dim; ++c) rows_[r * dim + c] = bank.at(r, c);
                const uint32_t id = add_read(cycle, pc, 'A', d.acc_bank, d.len, dim, rows_.data(),
                                             n * sizeof(i32), 4);
                ir.reads.push_back(id);
                ActRec a;
                a.pc = pc; a.issue = cycle; a.retire = cycle + duration; a.bank = d.acc_bank;
                a.len = d.len; a.read = static_cast<int>(id);
                a.out_rows = d.pool == Pool::NONE ? d.len : (d.len - d.pool_window) / d.pool_stride + 1;
                a.out_cols = d.pool == Pool::NONE ? dim : (dim - d.pool_window) / d.pool_stride + 1;
                (void)pw;
                ir.act = static_cast<int>(acts.size());
                act_of_pc[pc] = ir.act;
                acts.push_back(a);
                break;
            }
            default:
                break;
        }
        issues.push_back(std::move(ir));
    }

    // ---- the array ---------------------------------------------------------
    void on_matmul_begin(const Mxu& mxu, uint32_t len, uint32_t plane, bool switched) override {
        MmRec m;
        m.len = len; m.plane = plane; m.switched = switched;
        m.steps = static_cast<uint64_t>(len) + 2ull * dim - 1ull;
        m.w.resize(static_cast<std::size_t>(dim) * dim);
        for (uint32_t k = 0; k < dim; ++k)
            for (uint32_t c = 0; c < dim; ++c) m.w[k * dim + c] = mxu.weight_at(k, c, plane);
        m.left.reserve(m.steps * dim);
        m.act.reserve(m.steps * dim * dim);
        m.psum.reserve(m.steps * dim * dim);
        m.land.assign(static_cast<std::size_t>(len) * dim, 0);
        cur_mm = static_cast<int>(mms.size());
        mms.push_back(std::move(m));
    }

    void on_landing(uint64_t, uint32_t c, uint32_t row, i32 v, i32) override {
        mms[static_cast<std::size_t>(cur_mm)].land[static_cast<std::size_t>(row) * dim + c] = v;
    }

    void on_step(const Mxu& mxu, uint64_t, const i8* left) override {
        MmRec& m = mms[static_cast<std::size_t>(cur_mm)];
        m.left.insert(m.left.end(), left, left + dim);
        for (uint32_t k = 0; k < dim; ++k) {
            for (uint32_t c = 0; c < dim; ++c) {
                m.act.push_back(mxu.pe_act(k, c));
                m.psum.push_back(mxu.pe_psum(k, c));
            }
        }
    }

    // ---- sequencer outcomes ------------------------------------------------
    void on_sync(const Tpu& tpu, uint64_t cycle, std::size_t pc, std::size_t) override {
        syncs.emplace_back(cycle, pc);
        const std::vector<i32>& a = tpu.acc().raw();
        sync_blob.insert(sync_blob.end(), a.begin(), a.end());
    }
    void on_halt(const Tpu&, uint64_t cycle, std::size_t pc, uint32_t) override { halts.emplace_back(cycle, pc); }
    void on_trap(const Tpu&, uint64_t cycle, std::size_t, const std::string& why) override {
        traps.emplace_back(cycle, why);
    }

    void on_cycle_end(const Tpu&, uint64_t cycle, const CycleInfo& ci) override {
        CycleRow r;
        if (ci.issued)       r.outcome = ci.halted ? 11 : 0;
        else if (ci.trapped) r.outcome = 10;
        else                 r.outcome = static_cast<uint8_t>(ci.reason);
        r.pc           = static_cast<uint32_t>(ci.pc);
        r.blocker_unit = ci.blocker == Unit::COUNT ? 255 : static_cast<uint8_t>(ci.blocker);
        r.blocker_pc   = static_cast<uint32_t>(ci.blocker_pc);
        r.idle  = ci.idle_bucket;
        r.units = ci.units_active;
        r.fifo_occ = static_cast<uint8_t>(ci.fifo_occ);
        r.fifo_ready = static_cast<uint8_t>(ci.fifo_ready);
        r.ub_r = static_cast<uint8_t>(ci.ub_readers);
        r.ub_w = static_cast<uint8_t>(ci.ub_writers);
        r.plane = static_cast<uint8_t>(ci.plane);
        r.pending = ci.pending ? 1 : 0;
        if (rows.size() != cycle) {
            std::fprintf(stderr, "tracegen: cycle rows out of order (%zu vs %llu)\n", rows.size(),
                         static_cast<unsigned long long>(cycle));
        }
        rows.push_back(r);
    }
};

// ============================================================================
// Workload description
// ============================================================================

struct Workload {
    std::string name, note;
    Config      cfg;
    std::vector<RawInst> prog;
    std::vector<i8>      a_packed, b_packed;   // what host / DDR hold at start
    std::vector<i8>      expect;               // golden output, row-major M x N
    wl::Layer            layer;
    bool                 has_layer = false;
    HostAddr a_host = 0, c_host = 0x8000;
    uint32_t b_ddr  = 0;
    std::size_t macs = 0;
};

Workload small_workload() {
    Workload w;
    w.name = "matmul_8";
    w.note = "8x8 * 8x8 on a 4x4 array; hand-picked one-digit operands, shift 2, ReLU";
    w.cfg.dim = 4;
    w.cfg.ub_bytes = 4096;
    w.cfg.acc_banks = 2;
    w.cfg.dma_bytes_per_cycle = 4;
    w.layer.M = w.layer.K = w.layer.N = 8;
    w.layer.bias = 0; w.layer.multiplier = 1; w.layer.shift = 2; w.layer.fn = ActFn::RELU;
    w.has_layer = true;

    std::vector<i8> A(64), B(64);
    for (uint32_t i = 0; i < 8; ++i)
        for (uint32_t k = 0; k < 8; ++k) A[i * 8 + k] = static_cast<i8>(static_cast<int>((3 * i + 5 * k) % 7) - 3);
    for (uint32_t k = 0; k < 8; ++k)
        for (uint32_t j = 0; j < 8; ++j) B[k * 8 + j] = static_cast<i8>(static_cast<int>((2 * k + 3 * j) % 5) - 2);

    const wl::Lowering low = wl::lower_layer(w.layer, w.cfg);
    w.prog     = low.code;
    w.a_host   = low.a_host;
    w.c_host   = low.c_host;
    w.b_ddr    = low.b_ddr;
    w.a_packed = wl::pack(A, 8, 8, 4);
    w.b_packed = wl::pack(B, 8, 8, 4);
    w.expect   = wl::golden_layer(w.layer, A, B);
    w.macs     = low.macs;
    return w;
}

void load_into(Tpu& t, const Workload& w) {
    for (std::size_t i = 0; i < w.a_packed.size() && w.a_host + i < t.host().size(); ++i)
        t.host()[w.a_host + i] = static_cast<uint8_t>(w.a_packed[i]);
    for (std::size_t i = 0; i < w.b_packed.size() && w.b_ddr + i < t.weight_mem().size(); ++i)
        t.weight_mem()[w.b_ddr + i] = w.b_packed[i];
}

// Which logical tile each instruction touches, from the tiler's packing
// (tests/workloads.h). Address-based, so any dense layer lowered by
// wl::lower_layer annotates; anything else gets no tile.
struct Tile { char kind = 0; int m = -1, n = -1, k = -1; };

std::vector<Tile> annotate(const Workload& w) {
    std::vector<Tile> out(w.prog.size());
    if (!w.has_layer) return out;
    const uint32_t dim = w.cfg.dim;
    const std::size_t per = static_cast<std::size_t>(dim) * dim;
    const uint32_t across_a = wl::tiles_of(w.layer.K, dim);
    const uint32_t across_b = wl::tiles_of(w.layer.N, dim);
    const uint32_t across_c = wl::tiles_of(w.layer.N, dim);
    const std::size_t a_bytes = wl::packed_bytes(w.layer.M, w.layer.K, dim);
    const std::size_t b_bytes = wl::packed_bytes(w.layer.K, w.layer.N, dim);
    const std::size_t c_bytes = wl::packed_bytes(w.layer.M, w.layer.N, dim);

    std::map<UbAddr, std::pair<int, int>> ub_a;   // ub address -> A tile (m, k)
    std::pair<int, int> last_b{-1, -1};
    std::map<UbAddr, std::pair<int, int>> ub_c;   // staging address -> C tile (m, n)
    std::map<BankId, std::pair<int, int>> bank_c; // bank -> C tile (m, n)

    for (std::size_t pc = 0; pc < w.prog.size(); ++pc) {
        const Decoded d = decode(w.prog[pc]);
        Tile& t = out[pc];
        switch (d.op) {
            case Op::READ_HOST:
                if (d.host_addr >= w.a_host && d.host_addr < w.a_host + a_bytes) {
                    const std::size_t idx = (d.host_addr - w.a_host) / per;
                    t.kind = 'A'; t.m = static_cast<int>(idx / across_a); t.k = static_cast<int>(idx % across_a);
                    ub_a[d.ub_addr] = {t.m, t.k};
                }
                break;
            case Op::READ_WEIGHTS:
                if (d.ddr_addr >= w.b_ddr && d.ddr_addr < w.b_ddr + b_bytes) {
                    const std::size_t idx = (d.ddr_addr - w.b_ddr) / per;
                    t.kind = 'B'; t.k = static_cast<int>(idx / across_b); t.n = static_cast<int>(idx % across_b);
                    last_b = {t.k, t.n};
                }
                break;
            case Op::MATMUL: {
                auto a = ub_a.find(d.ub_addr);
                if (a != ub_a.end() && last_b.first >= 0) {
                    t.kind = 'M'; t.m = a->second.first; t.k = a->second.second; t.n = last_b.second;
                    bank_c[d.acc_bank] = {t.m, t.n};
                }
                break;
            }
            case Op::ACTIVATE: {
                auto b = bank_c.find(d.acc_bank);
                if (b != bank_c.end()) {
                    t.kind = 'C'; t.m = b->second.first; t.n = b->second.second;
                    ub_c[d.ub_addr] = {t.m, t.n};
                }
                break;
            }
            case Op::WRITE_HOST:
                if (d.host_addr >= w.c_host && d.host_addr < w.c_host + c_bytes) {
                    const std::size_t idx = (d.host_addr - w.c_host) / per;
                    t.kind = 'C'; t.m = static_cast<int>(idx / across_c); t.n = static_cast<int>(idx % across_c);
                }
                break;
            default:
                break;
        }
    }
    return out;
}

// ============================================================================
// Checks
// ============================================================================

struct Checks {
    bool trace_noop      = false;
    bool oracle          = false;
    bool golden          = false;
    bool pe_identity     = false;
    bool landings        = false;
    bool commit_replay   = false;
    bool requantize      = false;
    bool partitions      = false;
    bool intervals       = false;
    std::size_t pe_checked = 0, pe_bad = 0;
    std::size_t golden_mismatches = 0;
    uint64_t cycles_untraced = 0;
    std::string oracle_diff;
    bool all() const {
        return trace_noop && oracle && golden && pe_identity && landings && commit_replay &&
               requantize && partitions && intervals;
    }
};

bool same_stalls(const StallStats& a, const StallStats& b) {
    return a.ub_raw == b.ub_raw && a.ub_war == b.ub_war && a.ub_waw == b.ub_waw &&
           a.accum_hazard == b.accum_hazard && a.weight_stall == b.weight_stall &&
           a.unit_busy == b.unit_busy && a.weight_fifo_full == b.weight_fifo_full &&
           a.weight_fifo_empty == b.weight_fifo_empty && a.ub_bank_conflict == b.ub_bank_conflict &&
           a.drain == b.drain && a.weight_load_bubble == b.weight_load_bubble;
}

bool same_profile(const RunProfile& a, const RunProfile& b) {
    if (a.array_busy != b.array_busy || a.stream_cycles != b.stream_cycles || a.matmuls != b.matmuls ||
        a.macs_performed != b.macs_performed || a.dma_bytes != b.dma_bytes ||
        a.idle_weights != b.idle_weights || a.idle_bank != b.idle_bank || a.idle_accum != b.idle_accum ||
        a.idle_dma != b.idle_dma || a.idle_act != b.idle_act || a.idle_other != b.idle_other) return false;
    for (std::size_t i = 0; i < RunProfile::OPS; ++i)
        if (a.op_cycles[i] != b.op_cycles[i] || a.op_count[i] != b.op_count[i]) return false;
    return true;
}

bool same_machine(const Tpu& a, const Tpu& b) {
    if (a.host() != b.host()) return false;
    if (a.acc().raw() != b.acc().raw()) return false;
    for (uint32_t i = 0; i < a.ub().bytes(); ++i)
        if (a.ub().at(i) != b.ub().at(i)) return false;
    for (uint32_t p = 0; p < 2; ++p)
        for (uint32_t k = 0; k < a.config().dim; ++k)
            for (uint32_t c = 0; c < a.config().dim; ++c)
                if (a.mxu().weight_at(k, c, p) != b.mxu().weight_at(k, c, p)) return false;
    return a.mxu().active_plane() == b.mxu().active_plane() &&
           a.mxu().switch_pending() == b.mxu().switch_pending();
}

i8 act_fn(i8 v, ActFn fn) {
    switch (fn) {
        case ActFn::IDENTITY: return v;
        case ActFn::RELU:     return quant::relu(v);
        case ActFn::RELU6:    return quant::relu6(v);
    }
    return v;
}

// The activation pipeline recomputed from its written rules (the same quant::
// helpers both models share), over the bank rows the instruction read.
std::vector<i8> requantize_tile(const Decoded& d, const i32* bank, uint32_t dim) {
    std::vector<i8> tile(static_cast<std::size_t>(d.len) * dim, 0);
    for (uint32_t r = 0; r < d.len; ++r)
        for (uint32_t c = 0; c < dim; ++c)
            tile[r * dim + c] = act_fn(quant::requantize_biased(bank[r * dim + c], d.bias, d.multiplier, d.shift), d.act);
    if (d.pool == Pool::NONE) return tile;
    const uint32_t w = d.pool_window, s = d.pool_stride;
    const uint32_t rows = (d.len - w) / s + 1, cols = (dim - w) / s + 1;
    std::vector<i8> out(static_cast<std::size_t>(rows) * cols, 0);
    for (uint32_t orow = 0; orow < rows; ++orow) {
        for (uint32_t ocol = 0; ocol < cols; ++ocol) {
            int64_t sum = 0;
            i8 best = tile[static_cast<std::size_t>(orow * s) * dim + ocol * s];
            for (uint32_t dr = 0; dr < w; ++dr)
                for (uint32_t dc = 0; dc < w; ++dc) {
                    const i8 v = tile[static_cast<std::size_t>(orow * s + dr) * dim + (ocol * s + dc)];
                    sum += v;
                    best = quant::pool_max(best, v);
                }
            out[static_cast<std::size_t>(orow) * cols + ocol] = d.pool == Pool::MAX ? best : quant::pool_avg(sum, w);
        }
    }
    return out;
}

Checks run_checks(const Workload& w, const Recorder& rec, const Tpu& traced, const TpuResult& r_traced,
                  std::size_t host_bytes, std::size_t weight_bytes) {
    Checks ck;
    const uint32_t dim = w.cfg.dim;

    // 1. Tracing changed nothing.
    {
        Tpu plain(w.cfg, host_bytes, weight_bytes);
        load_into(plain, w);
        const TpuResult r0 = plain.run(w.prog);
        ck.cycles_untraced = r0.cycles;
        ck.trace_noop = r0.cycles == r_traced.cycles && r0.retired == r_traced.retired &&
                        r0.halted == r_traced.halted && r0.trapped == r_traced.trapped &&
                        r0.exit_code == r_traced.exit_code && r0.trap_reason == r_traced.trap_reason &&
                        r0.syncs.size() == r_traced.syncs.size() &&
                        same_stalls(plain.stalls(), traced.stalls()) &&
                        same_profile(plain.profile(), traced.profile()) &&
                        plain.mxu().stats().cycles == traced.mxu().stats().cycles &&
                        plain.mxu().stats().plane_switches == traced.mxu().stats().plane_switches &&
                        same_machine(plain, traced);
        for (std::size_t i = 0; i < r0.syncs.size() && ck.trace_noop; ++i)
            if (r0.syncs[i].acc != r_traced.syncs[i].acc) ck.trace_noop = false;
    }

    // 2. The eager oracle agrees on memories, snapshots and the retired count.
    {
        ref::Machine m(w.cfg, host_bytes, weight_bytes);
        m.host = rec.init_host;
        m.weight_mem = rec.init_ddr;
        const ref::Result rr = ref::run(m, w.prog);
        bool ok = m.host == traced.host() && m.acc == traced.acc().raw() &&
                  rr.retired == r_traced.retired && rr.halted == r_traced.halted &&
                  rr.trapped == r_traced.trapped && rr.syncs.size() == r_traced.syncs.size();
        for (uint32_t i = 0; i < traced.ub().bytes() && ok; ++i) if (m.ub[i] != traced.ub().at(i)) ok = false;
        for (std::size_t i = 0; i < rr.syncs.size() && ok; ++i) if (rr.syncs[i].acc != r_traced.syncs[i].acc) ok = false;
        ck.oracle = ok;
        if (!ok) ck.oracle_diff = "oracle state differs";
    }

    // 3. The output equals the golden loops' answer.
    if (w.has_layer && !w.expect.empty()) {
        const std::size_t c_bytes = wl::packed_bytes(w.layer.M, w.layer.N, dim);
        std::vector<i8> packed(c_bytes, 0);
        for (std::size_t i = 0; i < c_bytes && w.c_host + i < traced.host().size(); ++i)
            packed[i] = static_cast<i8>(traced.host()[w.c_host + i]);
        const std::vector<i8> got = wl::unpack(packed, w.layer.M, w.layer.N, dim);
        ck.golden_mismatches = 0;
        for (std::size_t i = 0; i < got.size() && i < w.expect.size(); ++i)
            if (got[i] != w.expect[i]) ++ck.golden_mismatches;
        ck.golden = got.size() == w.expect.size() && ck.golden_mismatches == 0;
    }

    // 4. Every recorded PE step satisfies psum_out = psum_in + act_in * w, and
    //    the act register carried the operand in.
    {
        std::size_t checked = 0, bad = 0;
        const std::size_t n = static_cast<std::size_t>(dim) * dim;
        for (const MmRec& m : rec.mms) {
            for (uint64_t s = 0; s < m.steps; ++s) {
                const i8*  act  = m.act.data() + s * n;
                const i32* psum = m.psum.data() + s * n;
                const i8*  pact = s ? m.act.data() + (s - 1) * n : nullptr;
                const i32* ppsum = s ? m.psum.data() + (s - 1) * n : nullptr;
                const i8*  left = m.left.data() + s * dim;
                for (uint32_t k = 0; k < dim; ++k) {
                    for (uint32_t c = 0; c < dim; ++c) {
                        const i8  act_in  = c == 0 ? left[k] : (pact ? pact[k * dim + c - 1] : 0);
                        const i32 psum_in = k == 0 ? 0 : (ppsum ? ppsum[(k - 1) * dim + c] : 0);
                        const i32 want    = static_cast<i32>(static_cast<uint32_t>(psum_in) +
                                                              static_cast<uint32_t>(static_cast<i32>(act_in) * static_cast<i32>(m.w[k * dim + c])));
                        ++checked;
                        if (psum[k * dim + c] != want || act[k * dim + c] != act_in) ++bad;
                    }
                }
            }
        }
        ck.pe_checked = checked;
        ck.pe_bad = bad;
        ck.pe_identity = bad == 0;
    }

    // 5. Landings equal the bottom row at the landing step, and the staged rows
    //    committed at retire equal (bank at issue +) landings.
    {
        bool ok = true;
        const std::size_t n = static_cast<std::size_t>(dim) * dim;
        for (std::size_t mi = 0; mi < rec.mms.size() && ok; ++mi) {
            const MmRec& m = rec.mms[mi];
            // the commit at retire for this pc
            const CommitRec* commit = nullptr;
            for (const CommitRec& c : rec.commits)
                if (c.kind == 'A' && c.pc == m.pc && c.cycle == m.retire) { commit = &c; break; }
            if (!commit) { ok = false; break; }
            const i32* after = reinterpret_cast<const i32*>(rec.commit_blob.data() + commit->off_after);
            const i32* acc_in = m.acc_in_read >= 0
                ? reinterpret_cast<const i32*>(rec.read_blob.data() + rec.reads[static_cast<std::size_t>(m.acc_in_read)].off)
                : nullptr;
            for (uint32_t row = 0; row < m.len && ok; ++row) {
                for (uint32_t c = 0; c < dim; ++c) {
                    const uint64_t s = static_cast<uint64_t>(row) + dim - 1 + c;
                    const i32 bottom = m.psum[s * n + (dim - 1) * dim + c];
                    const i32 v = m.land[row * dim + c];
                    if (bottom != v) { ok = false; break; }
                    const i32 want = m.accumulate
                        ? static_cast<i32>(static_cast<uint32_t>(acc_in[row * dim + c]) + static_cast<uint32_t>(v))
                        : v;
                    if (after[row * dim + c] != want) { ok = false; break; }
                }
            }
        }
        ck.landings = ok;
    }

    // 6. Replaying the after-images reproduces the final memories; before-images
    //    reproduce the initial ones when applied in reverse.
    {
        std::vector<uint8_t> host = rec.init_host;
        std::vector<i8> ub(traced.ub().bytes(), 0);
        std::vector<i32> acc(traced.acc().raw().size(), 0);
        std::vector<i8> plane[2];
        plane[0].assign(static_cast<std::size_t>(dim) * dim, 0);
        plane[1].assign(static_cast<std::size_t>(dim) * dim, 0);
        auto apply = [&](const CommitRec& c, std::size_t off) {
            const uint8_t* p = rec.commit_blob.data() + off;
            switch (c.kind) {
                case 'U': std::memcpy(ub.data() + c.addr, p, c.len); break;
                case 'H': std::memcpy(host.data() + c.addr, p, c.len); break;
                case 'A': std::memcpy(acc.data() + c.addr * dim * dim, p, c.len); break;
                case 'P': std::memcpy(plane[c.addr].data(), p, c.len); break;
                default: break;
            }
        };
        for (const CommitRec& c : rec.commits) apply(c, c.off_after);
        bool ok = host == traced.host() && acc == traced.acc().raw();
        for (uint32_t i = 0; i < traced.ub().bytes() && ok; ++i) if (ub[i] != traced.ub().at(i)) ok = false;
        for (uint32_t p = 0; p < 2 && ok; ++p)
            for (uint32_t k = 0; k < dim && ok; ++k)
                for (uint32_t c = 0; c < dim; ++c)
                    if (plane[p][k * dim + c] != traced.mxu().weight_at(k, c, p)) { ok = false; break; }
        for (std::size_t i = rec.commits.size(); i-- > 0;) apply(rec.commits[i], rec.commits[i].off_before);
        if (host != rec.init_host) ok = false;
        for (const i8 v : ub) if (v != 0) { ok = false; break; }
        for (const i32 v : acc) if (v != 0) { ok = false; break; }
        ck.commit_replay = ok;
    }

    // 7. Requantization recomputed from the bank rows read at issue equals the
    //    int8 tile committed at retire.
    {
        bool ok = true;
        for (const ActRec& a : rec.acts) {
            if (a.read < 0 || a.commit < 0) { ok = false; break; }
            const Decoded d = decode(w.prog[a.pc]);
            const i32* bank = reinterpret_cast<const i32*>(rec.read_blob.data() + rec.reads[static_cast<std::size_t>(a.read)].off);
            const std::vector<i8> want = requantize_tile(d, bank, dim);
            const CommitRec& c = rec.commits[static_cast<std::size_t>(a.commit)];
            if (c.kind != 'U' || c.len != want.size()) { ok = false; break; }
            if (std::memcmp(rec.commit_blob.data() + c.off_after, want.data(), want.size()) != 0) { ok = false; break; }
        }
        ck.requantize = ok;
    }

    // 8. The per-cycle table partitions the run the way the counters say.
    {
        uint64_t hist[12] = {};
        uint64_t idle[7] = {};
        for (const CycleRow& row : rec.rows) { ++hist[row.outcome]; ++idle[row.idle]; }
        const StallStats& s = traced.stalls();
        const RunProfile& p = traced.profile();
        const bool stalls_ok = hist[1] == s.drain && hist[2] == s.unit_busy && hist[3] == s.weight_fifo_empty &&
                               hist[4] == s.ub_raw && hist[5] == s.ub_war && hist[6] == s.ub_waw &&
                               hist[7] == s.accum_hazard && hist[8] == s.weight_stall && hist[9] == s.ub_bank_conflict;
        const bool idle_ok = idle[0] == p.array_busy && idle[1] == p.idle_weights && idle[2] == p.idle_bank &&
                             idle[3] == p.idle_accum && idle[4] == p.idle_dma && idle[5] == p.idle_act &&
                             idle[6] == p.idle_other;
        ck.partitions = stalls_ok && idle_ok && rec.rows.size() == r_traced.cycles &&
                        hist[0] + hist[11] == rec.issues.size();
    }

    // 9. Unit intervals never overlap, and each retire happens at issue + duration.
    {
        bool ok = true;
        std::vector<std::vector<std::pair<uint64_t, uint64_t>>> per(static_cast<std::size_t>(Unit::COUNT));
        for (const IssueRec& i : rec.issues) {
            if (i.op == Op::HALT) continue;
            per[static_cast<std::size_t>(i.unit)].emplace_back(i.cycle, i.done);
        }
        for (auto& v : per) {
            std::sort(v.begin(), v.end());
            for (std::size_t j = 1; j < v.size(); ++j) if (v[j].first < v[j - 1].second) ok = false;
        }
        for (const RetireRec& r : rec.retires) {
            bool found = false;
            for (const IssueRec& i : rec.issues) if (i.pc == r.pc && i.done == r.cycle) { found = true; break; }
            if (!found) ok = false;
        }
        ck.intervals = ok;
    }
    return ck;
}

// ============================================================================
// Manifest + container
// ============================================================================

void write_reservation(JW& j, const Reservation& r) {
    j.obj();
    j.key("ub_read"); j.arr(); j.unum(r.ub_read.lo); j.unum(r.ub_read.hi); j.end_arr();
    j.key("ub_write"); j.arr(); j.unum(r.ub_write.lo); j.unum(r.ub_write.hi); j.end_arr();
    j.key("acc_read");  if (r.acc_read == INVALID_BANK) j.null(); else j.unum(r.acc_read);
    j.key("acc_write"); if (r.acc_write == INVALID_BANK) j.null(); else j.unum(r.acc_write);
    j.kvb("reads_weights", r.reads_weights);
    j.kvb("writes_weights", r.writes_weights);
    j.end_obj();
}

const char* kind_name(char k) {
    switch (k) { case 'U': return "ub"; case 'H': return "host"; case 'A': return "acc"; case 'P': return "plane"; }
    return "?";
}

std::string build_manifest(const Workload& w, const Recorder& rec, const Tpu& tpu, const TpuResult& r,
                           const stats::Stats& st, const Checks& ck, const std::vector<Tile>& tiles,
                           const std::vector<Section>& sections, const std::vector<std::size_t>& offsets,
                           std::size_t host_bytes, std::size_t weight_bytes) {
    JW j;
    j.obj();
    j.kv("format", "mtpt/1");
    j.kv("generator", "tools/tracegen.cpp");
    j.kv("endian", "little");

    j.key("workload"); j.obj();
    j.kv("name", w.name); j.kv("note", w.note);
    j.kvb("has_layer", w.has_layer);
    if (w.has_layer) {
        j.kv("M", w.layer.M); j.kv("K", w.layer.K); j.kv("N", w.layer.N);
        j.key("layer"); j.obj();
        j.kv("bias", w.layer.bias); j.kv("multiplier", w.layer.multiplier); j.kv("shift", w.layer.shift);
        j.kv("fn", actfn_name(w.layer.fn));
        j.end_obj();
        j.key("tiles"); j.obj();
        j.kv("mt", wl::tiles_of(w.layer.M, w.cfg.dim)); j.kv("nt", wl::tiles_of(w.layer.N, w.cfg.dim));
        j.kv("kt", wl::tiles_of(w.layer.K, w.cfg.dim));
        j.end_obj();
        j.kvu("a_bytes", wl::packed_bytes(w.layer.M, w.layer.K, w.cfg.dim));
        j.kvu("b_bytes", wl::packed_bytes(w.layer.K, w.layer.N, w.cfg.dim));
        j.kvu("c_bytes", wl::packed_bytes(w.layer.M, w.layer.N, w.cfg.dim));
    }
    j.key("placement"); j.obj();
    j.kvu("a_host", w.a_host); j.kvu("c_host", w.c_host); j.kvu("b_ddr", w.b_ddr);
    j.end_obj();
    j.kvu("macs", w.macs);
    j.end_obj();

    const Config& c = w.cfg;
    j.key("config"); j.obj();
    j.kv("dim", c.dim); j.kv("ub_bytes", c.ub_bytes); j.kv("ub_banks", c.ub_banks);
    j.kv("acc_banks", c.acc_banks); j.kv("weight_fifo_depth", c.weight_fifo_depth);
    j.kvb("double_buffer", c.double_buffer); j.kv("dma_bytes_per_cycle", c.dma_bytes_per_cycle);
    j.kv("ddr_tile_latency", c.ddr_tile_latency); j.kv("act_pipeline_depth", c.act_pipeline_depth);
    j.kvu("host_bytes", host_bytes); j.kvu("weight_bytes", weight_bytes);
    j.end_obj();

    j.key("result"); j.obj();
    j.kvu("cycles", r.cycles); j.kvu("retired", r.retired); j.kvb("halted", r.halted);
    j.kv("exit_code", r.exit_code); j.kvb("trapped", r.trapped); j.kv("trap_reason", r.trap_reason);
    j.kvb("budget", r.budget);
    j.key("halt_cycle"); if (rec.halts.empty()) j.null(); else j.unum(rec.halts.front().first);
    j.end_obj();

    const StallStats& ss = tpu.stalls();
    const RunProfile& pf = tpu.profile();
    j.key("stats"); j.obj();
    j.kvu("array_busy", pf.array_busy); j.kvu("stream_cycles", pf.stream_cycles);
    j.kvu("macs_performed", pf.macs_performed); j.kvu("macs_useful", st.macs_useful);
    j.kvb("useful_known", st.useful_known); j.kvu("dma_bytes", pf.dma_bytes);
    j.key("stalls"); j.obj();
    j.kvu("ub_raw", ss.ub_raw); j.kvu("ub_war", ss.ub_war); j.kvu("ub_waw", ss.ub_waw);
    j.kvu("accum_hazard", ss.accum_hazard); j.kvu("weight_stall", ss.weight_stall);
    j.kvu("unit_busy", ss.unit_busy); j.kvu("weight_fifo_full", ss.weight_fifo_full);
    j.kvu("weight_fifo_empty", ss.weight_fifo_empty); j.kvu("ub_bank_conflict", ss.ub_bank_conflict);
    j.kvu("drain", ss.drain); j.kvu("weight_load_bubble", ss.weight_load_bubble);
    j.end_obj();
    j.key("idle"); j.obj();
    j.kvu("weights", pf.idle_weights); j.kvu("bank", pf.idle_bank); j.kvu("accum", pf.idle_accum);
    j.kvu("dma", pf.idle_dma); j.kvu("act", pf.idle_act); j.kvu("other", pf.idle_other);
    j.end_obj();
    j.key("breakdown"); j.obj();
    j.kvu("array_fill_drain", st.lost.array_fill_drain); j.kvu("weight_fifo_empty", st.lost.weight_fifo_empty);
    j.kvu("ub_bank_conflict", st.lost.ub_bank_conflict); j.kvu("accum_hazard", st.lost.accum_hazard);
    j.kvu("dma_bound", st.lost.dma_bound); j.kvu("activation", st.lost.activation);
    j.kvu("other", st.lost.other); j.kvu("partial_tile_waste", st.lost.partial_tile_waste);
    j.end_obj();
    j.kvd("utilization", st.utilization()); j.kvd("busy_utilization", st.busy_utilization());
    j.kvd("tops", st.tops()); j.kvd("arithmetic_intensity", st.arithmetic_intensity());
    j.kvd("ridge_point", st.ridge_point()); j.kvb("memory_bound", st.below_ridge());
    j.kv("dominant_cause", st.dominant_name()); j.kv("dominant_stall", st.dominant_stall_name());
    j.key("op_count"); j.arr(); for (std::size_t i = 0; i < RunProfile::OPS; ++i) j.unum(pf.op_count[i]); j.end_arr();
    j.key("op_cycles"); j.arr(); for (std::size_t i = 0; i < RunProfile::OPS; ++i) j.unum(pf.op_cycles[i]); j.end_arr();
    const MxuStats& ms = tpu.mxu().stats();
    j.key("mxu"); j.obj();
    j.kvu("cycles", ms.cycles); j.kvu("matmuls", ms.matmuls); j.kvu("weight_loads", ms.weight_loads);
    j.kvu("plane_switches", ms.plane_switches); j.kvu("weight_load_bubble", ms.weight_load_bubble);
    j.kvu("fill_drain_cycles", ms.fill_drain_cycles); j.kvu("useful_macs", ms.useful_macs);
    j.kvu("partial_tile_waste", ms.partial_tile_waste);
    j.end_obj();
    j.end_obj();

    // Program, with each instruction's lifetime and reservation.
    std::vector<long long> first_attempt(w.prog.size(), -1), issue_at(w.prog.size(), -1), retire_at(w.prog.size(), -1);
    std::vector<int> issue_index(w.prog.size(), -1);
    for (const StallSpan& s : rec.stalls)
        if (s.pc < w.prog.size() && (first_attempt[s.pc] < 0 || static_cast<long long>(s.from) < first_attempt[s.pc]))
            first_attempt[s.pc] = static_cast<long long>(s.from);
    for (std::size_t i = 0; i < rec.issues.size(); ++i) {
        const IssueRec& ir = rec.issues[i];
        issue_at[ir.pc] = static_cast<long long>(ir.cycle);
        issue_index[ir.pc] = static_cast<int>(i);
        if (first_attempt[ir.pc] < 0) first_attempt[ir.pc] = static_cast<long long>(ir.cycle);
        retire_at[ir.pc] = ir.op == Op::HALT ? static_cast<long long>(ir.cycle) : static_cast<long long>(ir.done);
    }

    j.key("program"); j.arr();
    for (std::size_t pc = 0; pc < w.prog.size(); ++pc) {
        const Decoded d = decode(w.prog[pc]);
        j.obj();
        j.kvu("pc", pc);
        j.key("words"); j.arr(); for (uint32_t k = 0; k < ISA_WORDS; ++k) j.unum(w.prog[pc].word[k]); j.end_arr();
        j.kv("asm", disasm(d));
        j.kv("op", op_name(d.op));
        j.kv("opcode", static_cast<long long>(d.op));
        j.kv("unit", unit_name(d.op == Op::READ_HOST || d.op == Op::WRITE_HOST ? Unit::DMA :
                                d.op == Op::READ_WEIGHTS ? Unit::WEIGHT :
                                d.op == Op::MATMUL ? Unit::MXU :
                                d.op == Op::ACTIVATE ? Unit::ACT : Unit::SEQ));
        j.key("fields"); j.obj();
        j.kvb("trap", d.trap);
        j.kvu("host_addr", d.host_addr); j.kvu("ub_addr", d.ub_addr); j.kvu("bytes", d.bytes);
        j.kvu("ddr_addr", d.ddr_addr); j.kvu("tile", d.tile); j.kvu("len", d.len);
        j.key("acc_bank"); if (d.acc_bank == INVALID_BANK) j.null(); else j.unum(d.acc_bank);
        j.kvb("accumulate", d.accumulate); j.kv("act", actfn_name(d.act)); j.kv("pool", pool_name(d.pool));
        j.kvu("pool_window", d.pool_window); j.kvu("pool_stride", d.pool_stride);
        j.kv("bias", d.bias); j.kv("multiplier", d.multiplier); j.kvu("shift", d.shift); j.kvu("code", d.code);
        j.end_obj();
        j.key("tile");
        if (tiles[pc].kind == 0) j.null();
        else {
            j.obj();
            j.kv("kind", std::string(1, tiles[pc].kind));
            j.kv("m", tiles[pc].m); j.kv("n", tiles[pc].n); j.kv("k", tiles[pc].k);
            j.end_obj();
        }
        j.key("first_attempt"); if (first_attempt[pc] < 0) j.null(); else j.num(first_attempt[pc]);
        j.key("issue");  if (issue_at[pc] < 0) j.null(); else j.num(issue_at[pc]);
        j.key("retire"); if (retire_at[pc] < 0) j.null(); else j.num(retire_at[pc]);
        j.key("issue_index"); if (issue_index[pc] < 0) j.null(); else j.num(issue_index[pc]);
        j.key("res");
        if (issue_index[pc] < 0) j.null(); else write_reservation(j, rec.issues[static_cast<std::size_t>(issue_index[pc])].res);
        j.end_obj();
    }
    j.end_arr();

    j.key("issues"); j.arr();
    for (const IssueRec& ir : rec.issues) {
        j.obj();
        j.kvu("cycle", ir.cycle); j.kvu("pc", ir.pc); j.kv("op", op_name(ir.op)); j.kv("unit", unit_name(ir.unit));
        j.kvu("duration", ir.duration); j.kvu("done", ir.done);
        j.key("reads"); j.arr(); for (const uint32_t id : ir.reads) j.unum(id); j.end_arr();
        j.key("mm"); if (ir.mm < 0) j.null(); else j.num(ir.mm);
        j.key("act"); if (ir.act < 0) j.null(); else j.num(ir.act);
        j.end_obj();
    }
    j.end_arr();

    j.key("retires"); j.arr();
    for (const RetireRec& rr : rec.retires) {
        j.obj();
        j.kvu("cycle", rr.cycle); j.kv("unit", unit_name(rr.unit)); j.kvu("pc", rr.pc); j.kv("op", op_name(rr.op));
        j.key("commits"); j.arr(); for (const uint32_t id : rr.commits) j.unum(id); j.end_arr();
        j.end_obj();
    }
    j.end_arr();

    j.key("stalls"); j.arr();
    for (const StallSpan& s : rec.stalls) {
        j.obj();
        j.kvu("from", s.from); j.kvu("to", s.to); j.kvu("pc", s.pc); j.kv("reason", stall_reason_name(s.reason));
        j.key("blocker_unit"); if (s.blocker == Unit::COUNT) j.null(); else j.str(unit_name(s.blocker));
        j.key("blocker_pc");   if (s.blocker == Unit::COUNT) j.null(); else j.unum(s.blocker_pc);
        j.end_obj();
    }
    j.end_arr();

    j.key("prefetches"); j.arr();
    for (const PrefetchRec& p : rec.prefetches) {
        j.obj();
        j.kvu("cycle", p.cycle); j.kvu("for_pc", p.for_pc); j.kvu("ddr", p.ddr); j.kvu("ready", p.ready); j.kvu("occ", p.occ);
        j.end_obj();
    }
    j.end_arr();

    j.key("wloads"); j.arr();
    for (const WloadRec& wl_ : rec.wloads) {
        j.obj();
        j.kvu("cycle", wl_.cycle); j.kvu("pc", wl_.pc); j.kvu("ddr", wl_.ddr); j.kvu("plane", wl_.plane);
        j.kvb("pending", wl_.pending); j.kvu("commit", wl_.commit);
        j.end_obj();
    }
    j.end_arr();

    j.key("commits"); j.arr();
    for (const CommitRec& cr : rec.commits) {
        j.obj();
        j.kvu("cycle", cr.cycle); j.kvu("pc", cr.pc); j.kv("kind", kind_name(cr.kind)); j.kvu("addr", cr.addr);
        j.kvu("rows", cr.rows); j.kvu("cols", cr.cols); j.kvu("len", cr.len);
        j.kvu("after", cr.off_after); j.kvu("before", cr.off_before);
        j.end_obj();
    }
    j.end_arr();

    j.key("reads"); j.arr();
    for (const ReadRec& rr : rec.reads) {
        j.obj();
        j.kvu("cycle", rr.cycle); j.kvu("pc", rr.pc); j.kv("kind", kind_name(rr.kind)); j.kvu("addr", rr.addr);
        j.kvu("rows", rr.rows); j.kvu("cols", rr.cols); j.kvu("len", rr.len); j.kvu("off", rr.off);
        j.end_obj();
    }
    j.end_arr();

    j.key("matmuls"); j.arr();
    for (std::size_t i = 0; i < rec.mms.size(); ++i) {
        const MmRec& m = rec.mms[i];
        j.obj();
        j.kvu("index", i); j.kvu("pc", m.pc); j.kvu("issue", m.issue); j.kvu("retire", m.retire);
        j.kvu("len", m.len); j.kvu("steps", m.steps); j.kvu("plane", m.plane); j.kvb("switched", m.switched);
        j.kvb("accumulate", m.accumulate); j.kvu("bank", m.bank);
        j.key("acc_in_read"); if (m.acc_in_read < 0) j.null(); else j.num(m.acc_in_read);
        j.end_obj();
    }
    j.end_arr();

    j.key("activates"); j.arr();
    for (std::size_t i = 0; i < rec.acts.size(); ++i) {
        const ActRec& a = rec.acts[i];
        j.obj();
        j.kvu("index", i); j.kvu("pc", a.pc); j.kvu("issue", a.issue); j.kvu("retire", a.retire);
        j.kvu("bank", a.bank); j.kvu("len", a.len); j.kvu("out_rows", a.out_rows); j.kvu("out_cols", a.out_cols);
        j.key("read"); if (a.read < 0) j.null(); else j.num(a.read);
        j.key("commit"); if (a.commit < 0) j.null(); else j.num(a.commit);
        j.end_obj();
    }
    j.end_arr();

    j.key("syncs"); j.arr();
    for (const auto& s : rec.syncs) { j.obj(); j.kvu("cycle", s.first); j.kvu("pc", s.second); j.end_obj(); }
    j.end_arr();
    j.key("traps"); j.arr();
    for (const auto& t : rec.traps) { j.obj(); j.kvu("cycle", t.first); j.kv("reason", t.second); j.end_obj(); }
    j.end_arr();

    j.key("checks"); j.obj();
    j.kvb("all", ck.all());
    j.kvb("trace_noop", ck.trace_noop); j.kvu("cycles_untraced", ck.cycles_untraced);
    j.kvb("oracle", ck.oracle); j.kvb("golden", ck.golden); j.kvu("golden_mismatches", ck.golden_mismatches);
    j.kvb("pe_identity", ck.pe_identity); j.kvu("pe_checked", ck.pe_checked); j.kvu("pe_bad", ck.pe_bad);
    j.kvb("landings", ck.landings); j.kvb("commit_replay", ck.commit_replay); j.kvb("requantize", ck.requantize);
    j.kvb("partitions", ck.partitions); j.kvb("intervals", ck.intervals);
    j.end_obj();

    j.key("sections"); j.arr();
    for (std::size_t i = 0; i < sections.size(); ++i) {
        j.obj();
        j.kv("name", sections[i].name); j.kv("dtype", sections[i].dtype);
        j.kvu("count", sections[i].count); j.kvu("off", offsets[i]); j.kvu("len", sections[i].bytes.size());
        j.end_obj();
    }
    j.end_arr();

    j.end_obj();
    return j.s;
}

void wr_u32(std::ostream& o, uint32_t v) {
    const char b[4] = {static_cast<char>(v & 0xFF), static_cast<char>((v >> 8) & 0xFF),
                       static_cast<char>((v >> 16) & 0xFF), static_cast<char>((v >> 24) & 0xFF)};
    o.write(b, 4);
}

}  // namespace

int main(int argc, char** argv) {
    // Our own flags first; everything else is the CLI's.
    std::string out_path = "trace.mtpt", expect_path, name_override, layer_spec;
    bool small = false;
    HostAddr a_host = 0, c_host = 0x8000;
    uint32_t b_ddr = 0;
    std::vector<std::string> rest;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](std::string& v) { if (i + 1 < argc) v = argv[++i]; };
        std::string v;
        if (a == "--out")         { next(out_path); continue; }
        if (a == "--expect")      { next(expect_path); continue; }
        if (a == "--name")        { next(name_override); continue; }
        if (a == "--layer")       { next(layer_spec); continue; }
        if (a == "--small")       { small = true; continue; }
        if (a == "--a-host")      { next(v); a_host = static_cast<HostAddr>(std::stoul(v, nullptr, 0)); continue; }
        if (a == "--c-host")      { next(v); c_host = static_cast<HostAddr>(std::stoul(v, nullptr, 0)); continue; }
        if (a == "--b-ddr")       { next(v); b_ddr = static_cast<uint32_t>(std::stoul(v, nullptr, 0)); continue; }
        rest.push_back(a);
    }
    std::vector<std::vector<char>> store;
    std::vector<char*> cli;
    cli.push_back(argv[0]);
    for (const std::string& s : rest) { store.emplace_back(s.begin(), s.end()); store.back().push_back('\0'); }
    for (auto& s : store) cli.push_back(s.data());
    CliOpts opts;
    if (const int rc = parse_args(static_cast<int>(cli.size()), cli.data(), opts); rc != 0) return rc;
    if (opts.show_help) {
        print_help();
        std::printf("\nTrace generator:\n  --out PATH        container to write (default trace.mtpt)\n"
                    "  --small           the built-in 8x8 on 4x4 teaching workload\n"
                    "  --expect PATH     golden output (MTPU, rows x cols int8) for the check\n"
                    "  --layer M,K,N     the dense layer shape, for tile annotation and the golden check\n"
                    "  --a-host/--c-host/--b-ddr  placement (defaults 0, 0x8000, 0)\n");
        return 0;
    }

    Workload w;
    std::size_t host_bytes = 0, weight_bytes = 0;
    if (small) {
        w = small_workload();
        host_bytes   = std::max<std::size_t>(w.a_host + w.a_packed.size(), w.c_host + wl::packed_bytes(w.layer.M, w.layer.N, w.cfg.dim)) + 64;
        weight_bytes = w.b_packed.size() + 64;
    } else {
        if (opts.prog_hex_path.empty() && opts.prog_raw_path.empty()) {
            std::fprintf(stderr, "tracegen: give --prog or --small\n");
            return 2;
        }
        w.cfg = opts.cfg;
        w.name = name_override.empty() ? "workload" : name_override;
        try {
            if (!opts.prog_hex_path.empty()) { std::ifstream in(opts.prog_hex_path); w.prog = load_program_hex(in); }
            else { std::ifstream in(opts.prog_raw_path, std::ios::binary); w.prog = load_program_raw(in); }
            TensorBlob acts, weights;
            if (!opts.acts_path.empty())    { std::ifstream in(opts.acts_path, std::ios::binary); acts = load_tensor_mtpu(in); }
            if (!opts.weights_path.empty()) { std::ifstream in(opts.weights_path, std::ios::binary); weights = load_tensor_mtpu(in); }
            w.a_packed = acts.i8v;
            w.b_packed = weights.i8v;
            if (!expect_path.empty()) { std::ifstream in(expect_path, std::ios::binary); w.expect = load_tensor_mtpu(in).i8v; }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "tracegen: load failed: %s\n", e.what());
            return 2;
        }
        w.a_host = a_host; w.c_host = c_host; w.b_ddr = b_ddr;
        w.macs = opts.macs;
        if (!layer_spec.empty()) {
            unsigned M = 0, K = 0, N = 0;
            if (std::sscanf(layer_spec.c_str(), "%u,%u,%u", &M, &K, &N) == 3) {
                w.layer.M = M; w.layer.K = K; w.layer.N = N;
                w.has_layer = true;
                // Requantization parameters from the program's first Activate.
                for (const RawInst& ri : w.prog) {
                    const Decoded d = decode(ri);
                    if (d.op == Op::ACTIVATE) { w.layer.bias = d.bias; w.layer.multiplier = d.multiplier; w.layer.shift = d.shift; w.layer.fn = d.act; break; }
                }
                if (w.name == "workload") w.name = "matmul_" + std::to_string(M);
            }
        }
        const MemNeed need = memory_needed(w.prog, w.cfg);
        host_bytes   = std::max<std::size_t>({std::size_t{1} << 12, need.host, w.a_packed.size()}) + 64;
        weight_bytes = std::max<std::size_t>({std::size_t{1} << 12, need.weight, w.b_packed.size()}) + 64;
    }

    // The traced run.
    Recorder rec;
    Tpu tpu(w.cfg, host_bytes, weight_bytes);
    load_into(tpu, w);
    TpuOptions ro;
    ro.sink = &rec;
    const auto t0 = std::chrono::steady_clock::now();
    const TpuResult r = tpu.run(w.prog, ro);
    const auto t1 = std::chrono::steady_clock::now();
    const stats::Stats st = stats::gather(tpu, r, w.macs, w.macs != 0);

    const std::vector<Tile> tiles = annotate(w);
    const Checks ck = run_checks(w, rec, tpu, r, host_bytes, weight_bytes);

    std::printf("%s: %llu cycles, %llu retired, %s\n", w.name.c_str(),
                static_cast<unsigned long long>(r.cycles), static_cast<unsigned long long>(r.retired),
                r.halted ? "halted" : r.trapped ? ("trapped: " + r.trap_reason).c_str() : "did not halt");
    std::printf("  traced run %.1f ms\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count());
    std::printf("  checks: trace_noop %d (untraced %llu cycles)  oracle %d  golden %d (%zu mismatches)\n",
                ck.trace_noop, static_cast<unsigned long long>(ck.cycles_untraced), ck.oracle, ck.golden,
                ck.golden_mismatches);
    std::printf("          pe_identity %d (%zu checked, %zu bad)  landings %d  commit_replay %d  requantize %d\n",
                ck.pe_identity, ck.pe_checked, ck.pe_bad, ck.landings, ck.commit_replay, ck.requantize);
    std::printf("          partitions %d  intervals %d  => %s\n", ck.partitions, ck.intervals,
                ck.all() ? "ALL PASS" : "FAILURES");

    // Sections.
    std::vector<Section> sections;
    {
        std::vector<uint32_t> words;
        for (const RawInst& ri : w.prog) for (uint32_t k = 0; k < ISA_WORDS; ++k) words.push_back(ri.word[k]);
        sections.push_back(make_section("prog", "u32", words));
        sections.push_back(make_section("init.host", "u8", rec.init_host));
        sections.push_back(make_section("init.ddr", "i8", rec.init_ddr));

        std::vector<uint8_t> outcome, bunit, idle, units, focc, fready, ubr, ubw, plane, pending;
        std::vector<uint32_t> pcs, bpc;
        for (const CycleRow& row : rec.rows) {
            outcome.push_back(row.outcome); pcs.push_back(row.pc); bunit.push_back(row.blocker_unit);
            bpc.push_back(row.blocker_pc); idle.push_back(row.idle); units.push_back(row.units);
            focc.push_back(row.fifo_occ); fready.push_back(row.fifo_ready); ubr.push_back(row.ub_r);
            ubw.push_back(row.ub_w); plane.push_back(row.plane); pending.push_back(row.pending);
        }
        sections.push_back(make_section("cyc.outcome", "u8", outcome));
        sections.push_back(make_section("cyc.pc", "u32", pcs));
        sections.push_back(make_section("cyc.blocker_unit", "u8", bunit));
        sections.push_back(make_section("cyc.blocker_pc", "u32", bpc));
        sections.push_back(make_section("cyc.idle", "u8", idle));
        sections.push_back(make_section("cyc.units", "u8", units));
        sections.push_back(make_section("cyc.fifo_occ", "u8", focc));
        sections.push_back(make_section("cyc.fifo_ready", "u8", fready));
        sections.push_back(make_section("cyc.ub_r", "u8", ubr));
        sections.push_back(make_section("cyc.ub_w", "u8", ubw));
        sections.push_back(make_section("cyc.plane", "u8", plane));
        sections.push_back(make_section("cyc.pending", "u8", pending));
        sections.push_back(make_section("commits", "u8", rec.commit_blob));
        sections.push_back(make_section("reads", "u8", rec.read_blob));
        sections.push_back(make_section("syncs", "i32", rec.sync_blob));
        for (std::size_t i = 0; i < rec.mms.size(); ++i) {
            const std::string p = "mm." + std::to_string(i) + ".";
            sections.push_back(make_section(p + "w", "i8", rec.mms[i].w));
            sections.push_back(make_section(p + "left", "i8", rec.mms[i].left));
            sections.push_back(make_section(p + "act", "i8", rec.mms[i].act));
            sections.push_back(make_section(p + "psum", "i32", rec.mms[i].psum));
            sections.push_back(make_section(p + "land", "i32", rec.mms[i].land));
        }
    }
    std::vector<std::size_t> offsets(sections.size());
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < sections.size(); ++i) {
        offsets[i] = cursor;
        cursor += sections[i].bytes.size();
        cursor = (cursor + 7) & ~static_cast<std::size_t>(7);
    }

    const std::string manifest = build_manifest(w, rec, tpu, r, st, ck, tiles, sections, offsets, host_bytes, weight_bytes);

    std::ofstream out(out_path, std::ios::binary);
    if (!out) { std::fprintf(stderr, "tracegen: cannot write %s\n", out_path.c_str()); return 1; }
    out.write("MTPT", 4);
    wr_u32(out, 1);
    wr_u32(out, static_cast<uint32_t>(manifest.size()));
    out.write(manifest.data(), static_cast<std::streamsize>(manifest.size()));
    std::size_t written = 12 + manifest.size();
    while (written % 8) { out.put('\0'); ++written; }
    const std::size_t base = written;
    for (std::size_t i = 0; i < sections.size(); ++i) {
        while (written - base < offsets[i]) { out.put('\0'); ++written; }
        out.write(reinterpret_cast<const char*>(sections[i].bytes.data()),
                  static_cast<std::streamsize>(sections[i].bytes.size()));
        written += sections[i].bytes.size();
    }
    out.close();
    std::size_t mm_bytes = 0;
    for (const Section& s : sections) if (s.name.rfind("mm.", 0) == 0) mm_bytes += s.bytes.size();
    std::printf("  wrote %s: manifest %zu B, sections %zu B (PE detail %zu B in %zu matmuls), total %zu B\n",
                out_path.c_str(), manifest.size(), cursor, mm_bytes, rec.mms.size(), written);
    return ck.all() ? 0 : 1;
}
