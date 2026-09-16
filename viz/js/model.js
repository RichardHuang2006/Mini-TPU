// State at (cycle t, phase) as a pure function of the trace, over the phases
// viz/README.md's cycle contract defines. Memory images replay the commits.
(function () {
  'use strict';
  const MTV = (window.MTV = window.MTV || {});

  const UNITS = ['DMA', 'WEIGHT', 'MXU', 'ACT', 'SEQ'];
  const PHASES = ['start', 'retire', 'prefetch', 'issue', 'account'];
  const STALL_NAMES = ['issued', 'drain', 'unit_busy', 'weight_fifo_empty', 'ub_raw', 'ub_war', 'ub_waw',
    'accum_hazard', 'weight_stall', 'ub_bank_conflict', 'trap', 'halt'];
  const IDLE_NAMES = ['busy', 'weights', 'bank', 'accum', 'dma', 'act', 'other'];
  const UNIT_OF_OP = {
    Read_Host_Memory: 'DMA', Write_Host_Memory: 'DMA', Read_Weights: 'WEIGHT', MatMul: 'MXU',
    Activate: 'ACT', Sync: 'SEQ', Nop: 'SEQ', Halt: 'SEQ',
  };

  function upperBound(arr, key, value) {
    // first index i with key(arr[i]) > value
    let lo = 0, hi = arr.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (key(arr[mid]) <= value) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  class Model {
    constructor(container) {
      this.c = container;
      const m = (this.m = container.manifest);
      this.cfg = m.config;
      this.dim = m.config.dim;
      this.cycles = m.result.cycles;
      this.workload = m.workload;
      this.program = m.program;
      this.issues = m.issues;
      this.retires = m.retires;
      this.stalls = m.stalls;
      this.prefetches = m.prefetches;
      this.wloads = m.wloads;
      this.commits = m.commits;
      this.reads = m.reads;
      this.matmuls = m.matmuls;
      this.activates = m.activates;

      this.cyc = {
        outcome: container.sec('cyc.outcome'), pc: container.sec('cyc.pc'),
        bunit: container.sec('cyc.blocker_unit'), bpc: container.sec('cyc.blocker_pc'),
        idle: container.sec('cyc.idle'), units: container.sec('cyc.units'),
        focc: container.sec('cyc.fifo_occ'), fready: container.sec('cyc.fifo_ready'),
        ubr: container.sec('cyc.ub_r'), ubw: container.sec('cyc.ub_w'),
        plane: container.sec('cyc.plane'), pending: container.sec('cyc.pending'),
      };
      this.commitBlob = container.sec('commits');
      this.readBlob = container.sec('reads');
      this.initHost = container.sec('init.host');
      this.initDdr = container.sec('init.ddr');

      this.buildIndexes();
      this.resetImages();
      this.t = 0;
      this.phase = 'account';
      this.state = null;
    }

    // indexes
    buildIndexes() {
      const m = this.m;
      this.unitIv = {};
      for (const u of UNITS) this.unitIv[u] = [];
      this.issueByPc = new Map();
      this.issuesByCycle = new Map();
      this.issues.forEach((iss, idx) => {
        iss.index = idx;
        this.issueByPc.set(iss.pc, iss);
        this.issuesByCycle.set(iss.cycle, iss);
        if (iss.op !== 'Halt') this.unitIv[iss.unit].push({ pc: iss.pc, issue: iss.cycle, done: iss.done, iss });
      });
      for (const u of UNITS) this.unitIv[u].sort((a, b) => a.issue - b.issue);

      this.retiresByCycle = new Map();
      for (const r of this.retires) {
        if (!this.retiresByCycle.has(r.cycle)) this.retiresByCycle.set(r.cycle, []);
        this.retiresByCycle.get(r.cycle).push(r);
      }
      this.commitsByCycle = new Map();
      this.commits.forEach((c, i) => {
        c.index = i;
        if (!this.commitsByCycle.has(c.cycle)) this.commitsByCycle.set(c.cycle, []);
        this.commitsByCycle.get(c.cycle).push(c);
      });
      this.readsByCycle = new Map();
      this.reads.forEach((r, i) => {
        r.index = i;
        if (!this.readsByCycle.has(r.cycle)) this.readsByCycle.set(r.cycle, []);
        this.readsByCycle.get(r.cycle).push(r);
      });
      this.prefetchByCycle = new Map();
      for (const p of this.prefetches) {
        if (!this.prefetchByCycle.has(p.cycle)) this.prefetchByCycle.set(p.cycle, []);
        this.prefetchByCycle.get(p.cycle).push(p);
      }
      this.wloadByCycle = new Map();
      this.wloads.forEach((w, i) => {
        w.index = i;
        if (!this.wloadByCycle.has(w.cycle)) this.wloadByCycle.set(w.cycle, []);
        this.wloadByCycle.get(w.cycle).push(w);
      });
      this.mmByPc = new Map();
      for (const mm of this.matmuls) this.mmByPc.set(mm.pc, mm);
      this.actByPc = new Map();
      for (const a of this.activates) this.actByPc.set(a.pc, a);
      this.stalls.forEach((s, i) => { s.index = i; });
      this.stallsByPc = new Map();
      for (const s of this.stalls) {
        if (!this.stallsByPc.has(s.pc)) this.stallsByPc.set(s.pc, []);
        this.stallsByPc.get(s.pc).push(s);
      }
      for (const list of this.stallsByPc.values()) list.sort((a, b) => a.from - b.from);
      this.prefetchForPc = new Map();
      for (const p of this.prefetches) if (!this.prefetchForPc.has(p.for_pc)) this.prefetchForPc.set(p.for_pc, p);

      // Idle runs (RLE of cyc.idle) and outcome runs, for the timeline.
      this.idleRuns = [];
      this.outcomeRuns = [];
      const idle = this.cyc.idle, out = this.cyc.outcome, pcs = this.cyc.pc;
      for (let t = 0; t < this.cycles; t++) {
        const last = this.idleRuns[this.idleRuns.length - 1];
        if (last && last.bucket === idle[t]) last.to = t;
        else this.idleRuns.push({ from: t, to: t, bucket: idle[t] });
        const lo = this.outcomeRuns[this.outcomeRuns.length - 1];
        if (lo && lo.outcome === out[t] && lo.pc === pcs[t] && out[t] !== 0) lo.to = t;
        else this.outcomeRuns.push({ from: t, to: t, outcome: out[t], pc: pcs[t] });
      }

      // Bookkeeping for the dense-layer views.
      const w = this.workload;
      this.hasLayer = !!w.has_layer;
      if (this.hasLayer) {
        this.M = w.M; this.K = w.K; this.N = w.N;
        this.mt = w.tiles.mt; this.nt = w.tiles.nt; this.kt = w.tiles.kt;
        this.placement = w.placement;
        this.tilePcs = { A: new Map(), B: new Map(), M: new Map(), C: new Map() };
        for (const p of this.program) {
          if (!p.tile) continue;
          const t = p.tile;
          let key;
          if (t.kind === 'A') key = t.m + ',' + t.k;
          else if (t.kind === 'B') key = t.k + ',' + t.n;
          else if (t.kind === 'M') key = t.m + ',' + t.n;
          else key = t.m + ',' + t.n;
          const map = this.tilePcs[t.kind];
          if (!map.has(key)) map.set(key, []);
          map.get(key).push(p);
        }
      }
      // Named UB regions from the program, for the memory map.
      this.ubRegions = [];
      const seen = new Set();
      for (const p of this.program) {
        let addr = null, len = 0, kind = '';
        if (p.op === 'Read_Host_Memory') { addr = p.fields.ub_addr; len = p.fields.bytes; kind = 'A'; }
        if (p.op === 'Activate') { addr = p.fields.ub_addr; len = (this.actByPc.get(p.pc) || {}).out_rows * (this.actByPc.get(p.pc) || {}).out_cols || p.fields.len * this.dim; kind = 'C'; }
        if (addr === null) continue;
        const key = kind + ':' + addr + ':' + len;
        if (seen.has(key)) continue;
        seen.add(key);
        this.ubRegions.push({ kind, addr, len, label: (kind === 'A' ? 'A tile slot ' : 'C staging ') + '0x' + addr.toString(16) });
      }
      this.ubRegions.sort((a, b) => a.addr - b.addr);
    }

    // memory images
    resetImages() {
      const dim = this.dim;
      this.img = {
        host: new Uint8Array(this.initHost),
        ub: new Int8Array(this.cfg.ub_bytes),
        acc: new Int32Array(this.cfg.acc_banks * dim * dim),
        planes: [new Int8Array(dim * dim), new Int8Array(dim * dim)],
      };
      this.applied = 0;   // commits[0..applied) are applied
    }

    commitAfter(c) { return this.commitBlob.subarray(c.after, c.after + c.len); }
    commitBefore(c) { return this.commitBlob.subarray(c.before, c.before + c.len); }
    commitAfterI32(c) {
      const b = this.commitBlob;
      return new Int32Array(b.buffer, b.byteOffset + c.after, c.len / 4);
    }
    commitBeforeI32(c) {
      const b = this.commitBlob;
      return new Int32Array(b.buffer, b.byteOffset + c.before, c.len / 4);
    }
    readBytes(r) { return this.readBlob.subarray(r.off, r.off + r.len); }
    readI8(r) { return new Int8Array(this.readBlob.buffer, this.readBlob.byteOffset + r.off, r.len); }
    readI32(r) { return new Int32Array(this.readBlob.buffer, this.readBlob.byteOffset + r.off, r.len / 4); }

    applyOne(c, bytes) {
      const dim = this.dim;
      switch (c.kind) {
        case 'ub': this.img.ub.set(new Int8Array(bytes.buffer, bytes.byteOffset, bytes.length), c.addr); break;
        case 'host': this.img.host.set(bytes, c.addr); break;
        case 'acc': {
          const words = new Int32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);
          this.img.acc.set(words, c.addr * dim * dim);
          break;
        }
        case 'plane': this.img.planes[c.addr].set(new Int8Array(bytes.buffer, bytes.byteOffset, bytes.length), 0); break;
        default: break;
      }
    }

    // Bring the images to "all commits with cycle <= through applied".
    materialize(through) {
      const target = through < 0 ? 0 : upperBound(this.commits, (c) => c.cycle, through);
      while (this.applied < target) { const c = this.commits[this.applied]; this.applyOne(c, this.commitAfter(c)); this.applied++; }
      while (this.applied > target) { this.applied--; const c = this.commits[this.applied]; this.applyOne(c, this.commitBefore(c)); }
    }

    // queries
    unitAt(u, t, phase) {
      // Active instruction on unit u at (t, phase), or null.
      const iv = this.unitIv[u];
      const i = upperBound(iv, (x) => x.issue, t) - 1;
      for (let j = i; j >= 0 && j >= i - 1; j--) {
        const x = iv[j];
        if (!x) continue;
        const issuedYet = phase === 'issue' || phase === 'account' ? x.issue <= t : x.issue < t;
        const retiredYet = phase === 'start' ? x.done < t : x.done <= t;
        if (issuedYet && !retiredYet) return x;
      }
      return null;
    }

    unitsAt(t, phase) {
      const out = {};
      for (const u of UNITS) out[u] = this.unitAt(u, t, phase);
      return out;
    }

    fifoAt(t, phase) {
      const afterPrefetch = phase === 'prefetch' || phase === 'issue' || phase === 'account';
      const afterRetire = phase !== 'start';
      const pushes = this.prefetches;
      const nPush = upperBound(pushes, (p) => p.cycle, afterPrefetch ? t : t - 1);
      const nPop = upperBound(this.wloads, (w) => w.cycle, afterRetire ? t : t - 1);
      const entries = [];
      for (let i = nPop; i < nPush; i++) {
        const p = pushes[i];
        entries.push({ push: p, ready: p.ready <= t, slot: i - nPop, index: i });
      }
      return entries;
    }

    planesAt(t, phase) {
      // Contents from the memory image, active/pending from the per-cycle table.
      const prevPlane = t > 0 ? this.cyc.plane[t - 1] : 0;
      const prevPending = t > 0 ? !!this.cyc.pending[t - 1] : false;
      let active = prevPlane, pending = prevPending;
      const wl = this.wloadByCycle.get(t) || [];
      if (phase !== 'start' && wl.length) pending = wl[wl.length - 1].pending;
      if (phase === 'issue' || phase === 'account') { active = this.cyc.plane[t]; pending = !!this.cyc.pending[t]; }
      // Which tile each plane holds: the latest plane load applied.
      const through = phase === 'start' ? t - 1 : t;
      const tiles = [null, null];
      const n = upperBound(this.wloads, (w) => w.cycle, through);
      for (let i = n - 1; i >= 0 && (tiles[0] === null || tiles[1] === null); i--) {
        const w = this.wloads[i];
        if (tiles[w.plane] === null) tiles[w.plane] = w;
      }
      return { active, pending, tiles };
    }

    // The MatMul whose array steps cover cycle t, and which frame to show.
    mxuAt(t, phase) {
      const x = this.unitAt('MXU', t, phase === 'start' || phase === 'retire' || phase === 'prefetch' ? 'account' : phase);
      // Before issue, the matmul in flight is the one issued before t.
      const early = phase === 'start' || phase === 'retire' || phase === 'prefetch';
      let mm = null, s = -1;
      if (early) {
        const y = this.unitAt('MXU', t, 'start');
        if (y) { mm = this.mmByPc.get(y.pc); s = t - 1 - mm.issue; }   // frame after step (t-1-issue)
        if (y && y.done === t && phase !== 'start') { mm = null; s = -1; }  // retired in phase 1
      } else if (x) {
        mm = this.mmByPc.get(x.pc);
        s = t - mm.issue;
      }
      return { mm, step: s };
    }

    outcomeAt(t) {
      const o = this.cyc.outcome[t];
      const pc = this.cyc.pc[t];
      const kind = o === 0 ? 'issued' : o === 10 ? 'trap' : o === 11 ? 'halt' : 'stall';
      const bu = this.cyc.bunit[t];
      return {
        kind, pc, reason: kind === 'stall' ? STALL_NAMES[o] : null,
        blockerUnit: bu === 255 ? null : UNITS[bu], blockerPc: bu === 255 ? null : this.cyc.bpc[t],
        idle: IDLE_NAMES[this.cyc.idle[t]],
      };
    }

    stallSpanAt(t) {
      const i = upperBound(this.stalls, (s) => s.from, t) - 1;
      if (i < 0) return null;
      const s = this.stalls[i];
      return t <= s.to ? s : null;
    }

    eventsAt(t) {
      return {
        retires: this.retiresByCycle.get(t) || [],
        commits: this.commitsByCycle.get(t) || [],
        wloads: this.wloadByCycle.get(t) || [],
        prefetches: this.prefetchByCycle.get(t) || [],
        issue: this.issuesByCycle.get(t) || null,
        reads: this.readsByCycle.get(t) || [],
        outcome: this.outcomeAt(t),
      };
    }

    // setCycle
    setCycle(t, phase) {
      t = Math.max(0, Math.min(this.cycles - 1, t | 0));
      if (!PHASES.includes(phase)) phase = 'account';
      this.t = t;
      this.phase = phase;
      this.materialize(phase === 'start' ? t - 1 : t);
      const units = this.unitsAt(t, phase);
      const st = {
        t, phase,
        units,
        fifo: this.fifoAt(t, phase),
        planes: this.planesAt(t, phase),
        mxu: this.mxuAt(t, phase),
        outcome: this.outcomeAt(t),
        events: this.eventsAt(t),
        stallSpan: this.stallSpanAt(t),
        ubStreams: { readers: 0, writers: 0 },
        pcNext: this.cyc.pc[t],
      };
      for (const u of UNITS) {
        const x = units[u];
        if (!x) continue;
        const res = this.program[x.pc].res;
        if (res && res.ub_read[1] > res.ub_read[0]) st.ubStreams.readers++;
        if (res && res.ub_write[1] > res.ub_write[0]) st.ubStreams.writers++;
      }
      this.state = st;
      return st;
    }

    // navigation
    nextEventCycle(t) {
      // Next cycle where something other than a continued stall happens.
      for (let x = t + 1; x < this.cycles; x++) {
        const o = this.cyc.outcome[x];
        if (o !== this.cyc.outcome[t] || this.cyc.pc[x] !== this.cyc.pc[t] || o === 0 ||
            this.retiresByCycle.has(x) || this.prefetchByCycle.has(x)) return x;
      }
      return this.cycles - 1;
    }
    prevEventCycle(t) {
      for (let x = t - 1; x >= 0; x--) {
        const o = this.cyc.outcome[x];
        if (o === 0 || o >= 10 || this.retiresByCycle.has(x) || this.prefetchByCycle.has(x) ||
            (x > 0 && (this.cyc.outcome[x - 1] !== o || this.cyc.pc[x - 1] !== this.cyc.pc[x]))) return x;
      }
      return 0;
    }
    nextMatmulCycle(t) {
      for (const mm of this.matmuls) if (mm.issue > t) return mm.issue;
      return this.cycles - 1;
    }
    prevMatmulCycle(t) {
      for (let i = this.matmuls.length - 1; i >= 0; i--) if (this.matmuls[i].issue < t) return this.matmuls[i].issue;
      return 0;
    }
    // Tile boundaries: the first MatMul of each (m, n) tile, and each Activate.
    tileBoundaries() {
      if (this._tb) return this._tb;
      const out = [];
      let last = null;
      for (const p of this.program) {
        if (!p.tile || p.issue === null) continue;
        if (p.tile.kind === 'M') {
          const key = p.tile.m + ',' + p.tile.n;
          if (key !== last) { out.push({ cycle: p.issue, label: 'C tile (' + p.tile.m + ',' + p.tile.n + ') begins', pc: p.pc }); last = key; }
        }
        if (p.tile.kind === 'C' && p.op === 'Activate') out.push({ cycle: p.issue, label: 'Activate tile (' + p.tile.m + ',' + p.tile.n + ')', pc: p.pc });
      }
      out.sort((a, b) => a.cycle - b.cycle);
      this._tb = out;
      return out;
    }
    nextTileCycle(t) {
      for (const b of this.tileBoundaries()) if (b.cycle > t) return b.cycle;
      return this.cycles - 1;
    }
    prevTileCycle(t) {
      const tb = this.tileBoundaries();
      for (let i = tb.length - 1; i >= 0; i--) if (tb[i].cycle < t) return tb[i].cycle;
      return 0;
    }

    // dense-layer helpers
    tilesOf(n) { return Math.floor((n + this.dim - 1) / this.dim); }
    tileOff(tr, tc, cols) { return (tr * this.tilesOf(cols) + tc) * this.dim * this.dim; }

    // Logical A[i][k] -> physical locations.
    aElement(i, k) {
      const dim = this.dim;
      const m = Math.floor(i / dim), kt = Math.floor(k / dim);
      const host = this.placement.a_host + this.tileOff(m, kt, this.K) + (i % dim) * dim + (k % dim);
      const value = this.initHost[host];
      return { i, k, m, kt, host, value: value > 127 ? value - 256 : value, tilePcs: this.tilePcs.A.get(m + ',' + kt) || [] };
    }
    bElement(k, j) {
      const dim = this.dim;
      const kt = Math.floor(k / dim), n = Math.floor(j / dim);
      const ddr = this.placement.b_ddr + this.tileOff(kt, n, this.N) + (k % dim) * dim + (j % dim);
      return { k, j, kt, n, ddr, value: this.initDdr[ddr], tilePcs: this.tilePcs.B.get(kt + ',' + n) || [] };
    }
    cElement(i, j) {
      const dim = this.dim;
      const m = Math.floor(i / dim), n = Math.floor(j / dim);
      const host = this.placement.c_host + this.tileOff(m, n, this.N) + (i % dim) * dim + (j % dim);
      const chain = this.tilePcs.M.get(m + ',' + n) || [];
      const acts = (this.tilePcs.C.get(m + ',' + n) || []).filter((p) => p.op === 'Activate');
      const writes = (this.tilePcs.C.get(m + ',' + n) || []).filter((p) => p.op === 'Write_Host_Memory');
      return { i, j, m, n, r: i % dim, c: j % dim, host, chain, acts, writes };
    }

    // Prefix sums of the per-cycle columns, so any window costs O(buckets).
    counts() {
      if (this._counts) return this._counts;
      const n = this.cycles;
      const idle = IDLE_NAMES.map(() => new Uint32Array(n + 1));
      const outcome = STALL_NAMES.map(() => new Uint32Array(n + 1));
      const ci = this.cyc.idle, co = this.cyc.outcome;
      for (let t = 0; t < n; t++) {
        for (let b = 0; b < idle.length; b++) idle[b][t + 1] = idle[b][t] + (ci[t] === b ? 1 : 0);
        for (let k = 0; k < outcome.length; k++) outcome[k][t + 1] = outcome[k][t] + (co[t] === k ? 1 : 0);
      }
      this._counts = { idle, outcome };
      return this._counts;
    }

    // Counts over cycles [a, b).
    windowCounts(a, b) {
      a = Math.max(0, Math.min(this.cycles, a | 0));
      b = Math.max(a, Math.min(this.cycles, b | 0));
      const pre = this.counts();
      const idle = {}, outcome = {};
      IDLE_NAMES.forEach((name, i) => { idle[name] = pre.idle[i][b] - pre.idle[i][a]; });
      STALL_NAMES.forEach((name, i) => { outcome[name] = pre.outcome[i][b] - pre.outcome[i][a]; });
      return { a, b, n: b - a, idle, outcome };
    }

    // How many cycles in [a, b) match a filter {kind: 'idle' | 'stall', key}.
    matchCount(filter, a, b) {
      const pre = this.counts();
      const arr = filter.kind === 'idle' ? pre.idle[IDLE_NAMES.indexOf(filter.key)] : pre.outcome[STALL_NAMES.indexOf(filter.key)];
      if (!arr) return 0;
      return arr[b] - arr[a];
    }

    idleRunAt(t) {
      const i = upperBound(this.idleRuns, (r) => r.from, t) - 1;
      return i >= 0 && t <= this.idleRuns[i].to ? this.idleRuns[i] : null;
    }

    // The instruction an idle-cause cycle points at, in the charge rule's order:
    // the recorded blocker, the in-flight DMA or ACT, the MatMul, the staller.
    idleBlockerAt(t) {
      const o = this.outcomeAt(t);
      if (o.kind === 'stall' && o.blockerPc !== null) return { pc: o.blockerPc, via: 'blocker', o };
      const unitFor = { dma: 'DMA', act: 'ACT', busy: 'MXU' }[o.idle];
      if (unitFor) {
        const x = this.unitAt(unitFor, t, 'account');
        if (x) return { pc: x.pc, via: o.idle === 'busy' ? 'busy' : 'charge', unit: unitFor, o };
      }
      if (this.program[o.pc]) return { pc: o.pc, via: 'stalled', o };
      return { pc: null, via: 'none', o };
    }

    // Why pc issued when it did, walked back to the root of the critical path;
    // each hop is tagged blocked / in_order / resource / root / unissued.
    blockerChain(pc, maxHops) {
      maxHops = maxHops || 4096;
      const hops = [];
      const seen = new Set();
      let cur = pc;
      while (cur !== null && cur !== undefined && hops.length < maxHops) {
        const p = this.program[cur];
        if (!p) break;
        if (seen.has(cur)) { hops.push({ pc: cur, kind: 'loop', next: null, spans: [] }); break; }
        seen.add(cur);
        const spans = this.stallsByPc.get(cur) || [];
        const hop = {
          pc: cur, op: p.op, unit: p.unit, first: p.first_attempt, issue: p.issue, retire: p.retire, spans,
          waited: spans.reduce((sum, x) => sum + (x.to - x.from + 1), 0), next: null,
        };
        const last = spans[spans.length - 1];
        if (p.issue === null) {
          hop.kind = 'unissued';
          if (last && last.blocker_pc !== null) { hop.reason = last.reason; hop.span = last; }
        } else if (last && last.to + 1 === p.issue) {
          hop.reason = last.reason;
          hop.span = last;
          if (last.blocker_pc !== null) {
            hop.kind = 'blocked';
            hop.next = last.blocker_pc;
            hop.nextUnit = last.blocker_unit;
          } else {
            hop.kind = 'resource';
            if (last.reason === 'weight_fifo_empty') hop.prefetch = this.prefetchForPc.get(cur) || null;
          }
        } else if (cur === 0) {
          hop.kind = 'root';
        } else {
          const prev = this.program[cur - 1];
          if (prev && prev.issue !== null && prev.issue === p.issue - 1) { hop.kind = 'in_order'; hop.next = cur - 1; }
          else hop.kind = 'root';
        }
        hops.push(hop);
        cur = hop.next;
      }
      return hops;
    }

    // Every read and commit covering one location; kind 'ub' | 'host' | 'acc'.
    history(kind, addr, row, col) {
      const dim = this.dim;
      const out = [];
      const covers = (rec) => {
        if (rec.kind !== kind) return false;
        if (kind === 'acc') return rec.addr === addr && row < rec.rows;
        return addr >= rec.addr && addr < rec.addr + rec.len;
      };
      for (const r of this.reads) if (covers(r)) out.push({ cycle: r.cycle, pc: r.pc, what: 'read at issue', rec: r });
      for (const c of this.commits) if (covers(c)) out.push({ cycle: c.cycle, pc: c.pc, what: 'commit at retire', rec: c });
      out.sort((a, b) => a.cycle - b.cycle || (a.what < b.what ? 1 : -1));
      return out;
    }
  }

  MTV.Model = Model;
  MTV.UNITS = UNITS;
  MTV.PHASES = PHASES;
  MTV.STALL_NAMES = STALL_NAMES;
  MTV.IDLE_NAMES = IDLE_NAMES;
  MTV.UNIT_OF_OP = UNIT_OF_OP;
  MTV.upperBound = upperBound;
})();
