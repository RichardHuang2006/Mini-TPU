#!/usr/bin/env python3
"""Independent consistency check of a Mini-TPU trace container (.mtpt).

tools/tracegen.cpp already checks a trace against the simulator while it has
the machine in hand. This script knows nothing about the simulator: it opens
the container the visualizer opens and checks that the trace is consistent
with itself and with the arithmetic the machine defines:

  1. layout: magic, manifest, every section inside the file and aligned;
  2. cycles: one outcome per cycle, stall and idle histograms equal the stats;
  3. lifetimes: retire == issue + duration, units never double-booked, stall
     spans cover exactly the cycles that issued nothing, with the right pc;
  4. commits: replaying after-images from the initial memories makes every
     before-image match, so backward and forward scrubbing agree;
  5. reads: every input snapshot equals the replayed memory at its issue cycle;
  6. array: for every recorded PE step, psum_out == psum_in + act_in * w and
     the act register carried act_in; landings equal the bottom row at the
     landing step; staged rows equal (bank at issue +) landings;
  7. activation: bias/multiply/round-half-away/clamp/function over the bank
     rows read at issue equals the int8 tile committed at retire;
  8. weights: FIFO pushes and pops are in order and within depth, each tile
     becomes ready push + latency, planes commit at Read_Weights retire;
  9. golden: for a dense layer, A and B unpacked from the initial memories,
     multiplied and requantized here, equal the output the trace wrote to host.

Exit status 0 when everything passes. Numpy is used when present; the pure
Python path is the same arithmetic, only slower.
"""
import json
import struct
import sys

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

DTYPE_SIZE = {"u8": 1, "i8": 1, "u32": 4, "i32": 4}
STALL_NAMES = ["issued", "drain", "unit_busy", "weight_fifo_empty", "ub_raw", "ub_war", "ub_waw",
               "accum_hazard", "weight_stall", "ub_bank_conflict", "trap", "halt"]
IDLE_NAMES = ["busy", "weights", "bank", "accum", "dma", "act", "other"]
UNITS = ["DMA", "WEIGHT", "MXU", "ACT", "SEQ"]


class Trace:
    def __init__(self, path):
        self.data = open(path, "rb").read()
        d = self.data
        if d[:4] != b"MTPT":
            raise ValueError("bad magic")
        self.version, mlen = struct.unpack("<II", d[4:12])
        self.man = json.loads(d[12:12 + mlen].decode("utf-8"))
        base = 12 + mlen
        base += (8 - base % 8) % 8
        self.base = base
        self.sections = {s["name"]: s for s in self.man["sections"]}

    def sec_bytes(self, name):
        s = self.sections[name]
        off = self.base + s["off"]
        return self.data[off:off + s["len"]]

    def sec(self, name):
        """Section as a list (or numpy array) of numbers."""
        s = self.sections[name]
        raw = self.sec_bytes(name)
        fmt = {"u8": "B", "i8": "b", "u32": "I", "i32": "i"}[s["dtype"]]
        if np is not None:
            return np.frombuffer(raw, dtype={"u8": np.uint8, "i8": np.int8, "u32": np.uint32, "i32": np.int32}[s["dtype"]])
        return list(struct.unpack("<%d%s" % (s["count"], fmt), raw))


class Checker:
    def __init__(self, path):
        self.t = Trace(path)
        self.man = self.t.man
        self.failures = []
        self.notes = []

    def check(self, name, ok, detail=""):
        if ok:
            print("  ok   %-12s %s" % (name, detail))
        else:
            print("  FAIL %-12s %s" % (name, detail))
            self.failures.append(name)

    # ------------------------------------------------------------------ 1 --
    def layout(self):
        t = self.t
        total = len(t.data)
        ok = t.version == 1 and self.man.get("format") == "mtpt/1"
        for s in self.man["sections"]:
            off = t.base + s["off"]
            if off % 8 != 0 or off + s["len"] > total:
                ok = False
            if s["len"] != s["count"] * DTYPE_SIZE[s["dtype"]]:
                ok = False
        self.check("layout", ok, "%d sections, %d bytes" % (len(self.man["sections"]), total))

    # ------------------------------------------------------------------ 2 --
    def cycles(self):
        m = self.man
        cycles = m["result"]["cycles"]
        outcome = self.t.sec("cyc.outcome")
        idle = self.t.sec("cyc.idle")
        ok = len(outcome) == cycles and len(idle) == cycles
        hist = [0] * 12
        for o in outcome:
            hist[int(o)] += 1
        ihist = [0] * 7
        for i in idle:
            ihist[int(i)] += 1
        st = m["stats"]["stalls"]
        ok &= hist[1] == st["drain"] and hist[2] == st["unit_busy"] and hist[3] == st["weight_fifo_empty"]
        ok &= hist[4] == st["ub_raw"] and hist[5] == st["ub_war"] and hist[6] == st["ub_waw"]
        ok &= hist[7] == st["accum_hazard"] and hist[8] == st["weight_stall"] and hist[9] == st["ub_bank_conflict"]
        ok &= hist[0] + hist[11] == len(m["issues"])
        ok &= ihist[0] == m["stats"]["array_busy"]
        idl = m["stats"]["idle"]
        ok &= ihist[1] == idl["weights"] and ihist[2] == idl["bank"] and ihist[3] == idl["accum"]
        ok &= ihist[4] == idl["dma"] and ihist[5] == idl["act"] and ihist[6] == idl["other"]
        self.check("cycles", ok, "%d cycles: %s" % (cycles, ", ".join(
            "%s %d" % (STALL_NAMES[i], hist[i]) for i in range(12) if hist[i])))

    # ------------------------------------------------------------------ 3 --
    def lifetimes(self):
        m = self.man
        ok = True
        per_unit = {u: [] for u in UNITS}
        by_pc = {}
        for i in m["issues"]:
            by_pc[i["pc"]] = i
            if i["op"] != "Halt":
                per_unit[i["unit"]].append((i["cycle"], i["done"]))
                if i["done"] != i["cycle"] + i["duration"]:
                    ok = False
        for u, iv in per_unit.items():
            iv.sort()
            for a, b in zip(iv, iv[1:]):
                if b[0] < a[1]:
                    ok = False
        for r in m["retires"]:
            i = by_pc.get(r["pc"])
            if i is None or i["done"] != r["cycle"] or i["unit"] != r["unit"]:
                ok = False
        # Program entries agree with the issue list.
        for p in m["program"]:
            i = by_pc.get(p["pc"])
            if i is None:
                if p["issue"] is not None:
                    ok = False
                continue
            if p["issue"] != i["cycle"]:
                ok = False
            if p["op"] != "Halt" and p["retire"] != i["done"]:
                ok = False
        # Stall spans cover exactly the non-issue cycles with the same pc.
        outcome = self.t.sec("cyc.outcome")
        pcs = self.t.sec("cyc.pc")
        covered = 0
        for s in m["stalls"]:
            for c in range(s["from"], s["to"] + 1):
                o = int(outcome[c])
                if o == 0 or o == 10 or o == 11 or STALL_NAMES[o] != s["reason"] or int(pcs[c]) != s["pc"]:
                    ok = False
                covered += 1
        stalled = sum(1 for o in outcome if 1 <= int(o) <= 9)
        ok &= covered == stalled
        self.check("lifetimes", ok, "%d issues, %d retires, %d stall spans covering %d cycles" % (
            len(m["issues"]), len(m["retires"]), len(m["stalls"]), covered))

    # ------------------------------------------------------------------ 4/5 --
    def replay(self):
        m = self.man
        cfg = m["config"]
        dim = cfg["dim"]
        blob = self.t.sec_bytes("commits")
        rblob = self.t.sec_bytes("reads")
        host = bytearray(self.t.sec_bytes("init.host"))
        ub = bytearray(cfg["ub_bytes"])
        acc = bytearray(4 * cfg["acc_banks"] * dim * dim)
        planes = [bytearray(dim * dim), bytearray(dim * dim)]

        def loc(c):
            k = c["kind"]
            if k == "ub":
                return ub, c["addr"]
            if k == "host":
                return host, c["addr"]
            if k == "acc":
                return acc, c["addr"] * dim * dim * 4
            return planes[c["addr"]], 0

        events = []
        for ci, c in enumerate(m["commits"]):
            events.append((c["cycle"], 1, ci, c))
        for ri, r in enumerate(m["reads"]):
            events.append((r["cycle"], 0, ri, r))   # reads happen at issue, after that cycle's retires
        events.sort(key=lambda e: (e[0], e[1], e[2]))
        # Within a cycle, retires (commits) precede issue (reads): sort key puts
        # commits (1) after reads (0), so order commits first explicitly.
        events.sort(key=lambda e: (e[0], 0 if e[1] == 1 else 1, e[2]))

        ok_before = True
        ok_reads = True
        bad_read = None
        for cycle, kind, idx, rec in events:
            if kind == 1:
                mem, off = loc(rec)
                before = blob[rec["before"]:rec["before"] + rec["len"]]
                if bytes(mem[off:off + rec["len"]]) != bytes(before):
                    ok_before = False
                after = blob[rec["after"]:rec["after"] + rec["len"]]
                mem[off:off + rec["len"]] = after
            else:
                k = rec["kind"]
                if k == "ub":
                    mem, off = ub, rec["addr"]
                elif k == "host":
                    mem, off = host, rec["addr"]
                else:
                    mem, off = acc, rec["addr"] * dim * dim * 4
                snap = rblob[rec["off"]:rec["off"] + rec["len"]]
                if bytes(mem[off:off + rec["len"]]) != bytes(snap):
                    ok_reads = False
                    if bad_read is None:
                        bad_read = (cycle, rec["pc"], k)
        self.check("commits", ok_before, "%d commits; every before-image matches the replayed state" % len(m["commits"]))
        self.check("reads", ok_reads, "%d issue-time snapshots match the replayed state%s" % (
            len(m["reads"]), "" if bad_read is None else "; first mismatch %r" % (bad_read,)))
        self.final_host = bytes(host)
        self.final_ub = bytes(ub)
        self.final_acc = bytes(acc)

    # ------------------------------------------------------------------ 6 --
    def array(self):
        m = self.man
        dim = m["config"]["dim"]
        n = dim * dim
        checked = 0
        bad = 0
        land_ok = True
        blob = self.t.sec_bytes("commits")
        rblob = self.t.sec_bytes("reads")
        commits_by = {}
        for ci, c in enumerate(m["commits"]):
            if c["kind"] == "acc":
                commits_by[(c["pc"], c["cycle"])] = c
        for mm in m["matmuls"]:
            i = mm["index"]
            w = self.t.sec("mm.%d.w" % i)
            left = self.t.sec("mm.%d.left" % i)
            act = self.t.sec("mm.%d.act" % i)
            psum = self.t.sec("mm.%d.psum" % i)
            land = self.t.sec("mm.%d.land" % i)
            steps = mm["steps"]
            ln = mm["len"]
            if np is not None:
                W = w.reshape(dim, dim).astype(np.int32)
                A = act.reshape(steps, dim, dim)
                P = psum.reshape(steps, dim, dim)
                L = left.reshape(steps, dim)
                prev_act = np.zeros((dim, dim), dtype=np.int8)
                prev_psum = np.zeros((dim, dim), dtype=np.int32)
                for s in range(steps):
                    act_in = np.empty((dim, dim), dtype=np.int8)
                    act_in[:, 0] = L[s]
                    act_in[:, 1:] = prev_act[:, :-1]
                    psum_in = np.zeros((dim, dim), dtype=np.int32)
                    psum_in[1:, :] = prev_psum[:-1, :]
                    with np.errstate(over="ignore"):
                        want = (psum_in + act_in.astype(np.int32) * W).astype(np.int32)
                    checked += n
                    bad += int(np.count_nonzero(P[s] != want)) + int(np.count_nonzero(A[s] != act_in))
                    prev_act, prev_psum = A[s], P[s]
                # landings
                Lnd = land.reshape(ln, dim)
                for row in range(ln):
                    for c in range(dim):
                        s = row + dim - 1 + c
                        if int(P[s][dim - 1][c]) != int(Lnd[row][c]):
                            land_ok = False
            else:
                prev_act = [0] * n
                prev_psum = [0] * n
                for s in range(steps):
                    a_s = act[s * n:(s + 1) * n]
                    p_s = psum[s * n:(s + 1) * n]
                    l_s = left[s * dim:(s + 1) * dim]
                    for k in range(dim):
                        for c in range(dim):
                            act_in = l_s[k] if c == 0 else prev_act[k * dim + c - 1]
                            psum_in = 0 if k == 0 else prev_psum[(k - 1) * dim + c]
                            want = (psum_in + act_in * w[k * dim + c]) & 0xFFFFFFFF
                            if want >= 1 << 31:
                                want -= 1 << 32
                            checked += 1
                            if p_s[k * dim + c] != want or a_s[k * dim + c] != act_in:
                                bad += 1
                    prev_act, prev_psum = a_s, p_s
                for row in range(ln):
                    for c in range(dim):
                        s = row + dim - 1 + c
                        if psum[s * n + (dim - 1) * dim + c] != land[row * dim + c]:
                            land_ok = False
            # staged rows committed at retire == (acc_in +) landings
            c = commits_by.get((mm["pc"], mm["retire"]))
            if c is None:
                land_ok = False
                continue
            after = struct.unpack("<%di" % (ln * dim), blob[c["after"]:c["after"] + ln * dim * 4])
            if mm["accumulate"]:
                r = m["reads"][mm["acc_in_read"]]
                acc_in = struct.unpack("<%di" % (ln * dim), rblob[r["off"]:r["off"] + ln * dim * 4])
            else:
                acc_in = [0] * (ln * dim)
            for j in range(ln * dim):
                want = (acc_in[j] + int(land[j])) & 0xFFFFFFFF
                if want >= 1 << 31:
                    want -= 1 << 32
                if after[j] != want:
                    land_ok = False
                    break
        self.check("pe_identity", bad == 0, "%d PE steps checked, %d bad" % (checked, bad))
        self.check("landings", land_ok, "%d matmuls: bottom row == landing, staged == bank + landing" % len(m["matmuls"]))

    # ------------------------------------------------------------------ 7 --
    @staticmethod
    def round_shift(v, shift):
        if shift == 0:
            return v
        half = 1 << (shift - 1)
        return (v + half) >> shift if v >= 0 else -((-v + half) >> shift)

    @staticmethod
    def saturate(v):
        return max(-128, min(127, v))

    def act_fn(self, v, fn):
        if fn == "relu":
            return 0 if v < 0 else v
        if fn == "relu6":
            return 0 if v < 0 else (6 if v > 6 else v)
        return v

    def requantize(self, acc_rows, ln, dim, f):
        tile = [self.act_fn(self.saturate(self.round_shift((a + f["bias"]) * f["multiplier"], f["shift"])), f["act"])
                for a in acc_rows]
        if f["pool"] == "none":
            return tile
        w, s = f["pool_window"], f["pool_stride"]
        rows, cols = (ln - w) // s + 1, (dim - w) // s + 1
        out = []
        for orow in range(rows):
            for ocol in range(cols):
                vals = [tile[(orow * s + dr) * dim + (ocol * s + dc)] for dr in range(w) for dc in range(w)]
                if f["pool"] == "max":
                    out.append(max(vals))
                else:
                    num, den = sum(vals), w * w
                    half = den // 2
                    q = (num + half) // den if num >= 0 else -((-num + half) // den)
                    out.append(self.saturate(q))
        return out

    def activation(self):
        m = self.man
        dim = m["config"]["dim"]
        blob = self.t.sec_bytes("commits")
        rblob = self.t.sec_bytes("reads")
        ok = True
        for a in m["activates"]:
            f = m["program"][a["pc"]]["fields"]
            r = m["reads"][a["read"]]
            acc_rows = struct.unpack("<%di" % (a["len"] * dim), rblob[r["off"]:r["off"] + a["len"] * dim * 4])
            want = self.requantize(acc_rows, a["len"], dim, f)
            c = m["commits"][a["commit"]]
            got = struct.unpack("<%db" % c["len"], blob[c["after"]:c["after"] + c["len"]])
            if list(got) != want:
                ok = False
        self.check("activation", ok, "%d Activates re-derived from their bank rows" % len(m["activates"]))

    # ------------------------------------------------------------------ 8 --
    def weights(self):
        m = self.man
        cfg = m["config"]
        ok = True
        q = []
        pushes = sorted(m["prefetches"], key=lambda p: (p["cycle"], p["for_pc"]))
        pops = sorted(m["wloads"], key=lambda w: w["cycle"])
        pi = 0

        def push_through(limit):
            nonlocal pi, ok
            while pi < len(pushes) and pushes[pi]["cycle"] <= limit:
                p = pushes[pi]
                if p["ready"] != p["cycle"] + cfg["ddr_tile_latency"]:
                    ok = False
                q.append(p)
                if len(q) > cfg["weight_fifo_depth"]:
                    ok = False
                pi += 1

        for pop in pops:
            # Within a cycle the retire (pop) precedes the prefetch (push), so
            # pushes of earlier cycles enter first, then the pop, then this
            # cycle's pushes.
            push_through(pop["cycle"] - 1)
            if not q:
                ok = False
                break
            head = q.pop(0)
            if head["ddr"] != pop["ddr"] or head["ready"] > pop["cycle"]:
                ok = False
            push_through(pop["cycle"])
            c = m["commits"][pop["commit"]]
            if c["kind"] != "plane" or c["addr"] != pop["plane"] or c["cycle"] != pop["cycle"]:
                ok = False
        push_through(m["result"]["cycles"])
        ok &= len(pushes) == len(pops)
        self.check("weights", ok, "%d pushes, %d plane loads, depth %d, latency %d" % (
            len(pushes), len(pops), cfg["weight_fifo_depth"], cfg["ddr_tile_latency"]))

    # ------------------------------------------------------------------ 9 --
    def golden(self):
        m = self.man
        w = m["workload"]
        if not w.get("has_layer"):
            self.notes.append("no dense layer metadata; golden check skipped")
            return
        dim = m["config"]["dim"]
        M, K, N = w["M"], w["K"], w["N"]
        pl = w["placement"]
        init_host = self.t.sec_bytes("init.host")
        init_ddr = self.t.sec_bytes("init.ddr")

        def tiles_of(n):
            return (n + dim - 1) // dim

        def unpack(buf, rows, cols):
            across = tiles_of(cols)
            out = [0] * (rows * cols)
            for r in range(rows):
                for c in range(cols):
                    o = ((r // dim) * across + (c // dim)) * dim * dim + (r % dim) * dim + (c % dim)
                    v = buf[o]
                    out[r * cols + c] = v - 256 if v > 127 else v
            return out

        A = unpack(init_host[pl["a_host"]:], M, K)
        B = unpack(init_ddr[pl["b_ddr"]:], K, N)
        f = dict(m["workload"]["layer"])
        f["pool"] = "none"
        f["act"] = f["fn"]
        C = []
        for i in range(M):
            for j in range(N):
                s = 0
                for k in range(K):
                    s += A[i * K + k] * B[k * N + j]
                s &= 0xFFFFFFFF
                if s >= 1 << 31:
                    s -= 1 << 32
                C.append(s)
        want = self.requantize(C, M, N, f) if False else [
            self.act_fn(self.saturate(self.round_shift((c + f["bias"]) * f["multiplier"], f["shift"])), f["fn"]) for c in C]
        got = unpack(self.final_host[pl["c_host"]:], M, N)
        mism = sum(1 for a, b in zip(got, want) if a != b)
        self.check("golden", mism == 0, "%dx%d output recomputed from A and B: %d mismatches" % (M, N, mism))

    def run(self):
        m = self.man
        print("%s: %s" % (m["workload"]["name"], m["workload"].get("note", "")))
        print("  generator checks: %s" % ("all pass" if m["checks"]["all"] else "FAILED: " + json.dumps(m["checks"])))
        self.layout()
        self.cycles()
        self.lifetimes()
        self.replay()
        self.array()
        self.activation()
        self.weights()
        self.golden()
        for n in self.notes:
            print("  note: " + n)
        return not self.failures and m["checks"]["all"]


def main(argv):
    if len(argv) < 2:
        print("usage: check_trace.py TRACE.mtpt [...]", file=sys.stderr)
        return 2
    ok = True
    for path in argv[1:]:
        if not Checker(path).run():
            ok = False
    print("check_trace: %s" % ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
