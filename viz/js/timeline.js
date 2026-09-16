// Bottom timeline: activity per unit, the weight FIFO, the Unified Buffer port
// streams, bank reservations and the idle-cause bucket. See viz/README.md.
(function () {
  'use strict';
  const MTV = (window.MTV = window.MTV || {});
  MTV.views = MTV.views || {};
  const fmt = (n) => (n === null || n === undefined ? '—' : Number(n).toLocaleString('en-US'));
  const esc = (s) => String(s).replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
  const label = (m, pc) => (MTV.instrLabel ? MTV.instrLabel(m, pc) : 'pc ' + pc);

  const ROWS = [
    { key: 'SEQ', label: 'SEQ', min: 22 },
    { key: 'DMA', label: 'DMA', min: 22 },
    { key: 'WEIGHT', label: 'WEIGHT', min: 22 },
    { key: 'MXU', label: 'MXU', min: 22 },
    { key: 'ACT', label: 'ACT', min: 22 },
    { key: 'FIFO', label: 'FIFO', min: 24 },
    { key: 'UB', label: 'UB ports', min: 26 },
    { key: 'ACC', label: 'ACC banks', min: 24 },
    { key: 'IDLE', label: 'idle cause', min: 22 },
  ];
  const UNIT_ROWS = ['DMA', 'WEIGHT', 'MXU', 'ACT'];
  const COLLAPSED_H = 10;
  const RULER_H = 20;

  // Theme tokens, read through getComputedStyle so every row follows the theme.
  const STALL_VARS = {
    unit_busy: '--st-unit-busy', ub_raw: '--st-ub-raw', ub_war: '--st-ub-war', ub_waw: '--st-ub-waw',
    accum_hazard: '--st-accum', weight_fifo_empty: '--st-fifo-empty', weight_stall: '--st-weight',
    ub_bank_conflict: '--st-bank', drain: '--st-drain',
  };
  const IDLE_VARS = {
    busy: '--idle-busy', weights: '--idle-weights', bank: '--idle-bank', accum: '--idle-accum',
    dma: '--idle-dma', act: '--idle-act', other: '--idle-other',
  };
  // charge_idle_cycle's rule for each bucket, in its test order.
  const IDLE_RULES = {
    busy: 'the MXU slot is active after this cycle’s issue',
    weights: 'a weight-path counter (weight_stall, weight_fifo_full, weight_fifo_empty) moved',
    bank: 'ub_bank_conflict moved',
    accum: 'accum_hazard moved',
    dma: 'no named counter moved and the DMA unit is active',
    act: 'no named counter moved, DMA idle, ACT active',
    other: 'no named counter moved and neither DMA nor ACT is active',
  };

  function readPalette() {
    const cs = getComputedStyle(document.documentElement);
    const v = (n) => cs.getPropertyValue(n).trim();
    const P = {
      ink: v('--ink'), muted: v('--muted'), line: v('--line'), lineStrong: v('--line-strong'),
      surface: v('--surface'), surface2: v('--surface-2'), sel: v('--sel'), accent: v('--accent'),
      ok: v('--ok'), bad: v('--bad'), onFill: v('--on-fill') || '#fff', hatch: v('--hatch'), mono: v('--mono') || 'monospace',
      ubRead: v('--ub-read'), ubWrite: v('--ub-write'), accWrite: v('--acc-write'), accRead: v('--acc-read'),
      unit: { DMA: v('--u-dma'), WEIGHT: v('--u-weight'), MXU: v('--u-mxu'), ACT: v('--u-act'), SEQ: v('--u-seq') },
      stall: {}, idle: {},
    };
    for (const k of Object.keys(STALL_VARS)) P.stall[k] = v(STALL_VARS[k]);
    for (const k of Object.keys(IDLE_VARS)) P.idle[k] = v(IDLE_VARS[k]);
    return P;
  }

  function niceStep(raw) {
    const p = Math.pow(10, Math.floor(Math.log10(Math.max(1, raw))));
    for (const k of [1, 2, 5, 10]) if (k * p >= raw) return k * p;
    return 10 * p;
  }

  const view = {
    init(app) {
      this.app = app;
      this.scroll = document.getElementById('tl-scroll');
      this.canvas = document.getElementById('tl-canvas');
      this.ruler = document.getElementById('tl-ruler');
      this.labels = document.getElementById('tl-labels');
      this.legend = document.getElementById('tl-legend');
      this.tip = document.getElementById('tl-tip');
      this.summary = document.getElementById('tl-summary');
      this.ctx = this.canvas.getContext('2d');
      this.rctx = this.ruler.getContext('2d');
      this.x0 = 0; this.x1 = 1;       // visible cycle window [x0, x1)
      this.collapsed = new Set();
      this.filter = null;             // {kind: 'idle' | 'stall', key}
      this.buildLabels();
      this.buildLegend();

      const cv = this.canvas, ru = this.ruler;
      cv.addEventListener('wheel', (e) => this.onWheel(e, true), { passive: false });
      ru.addEventListener('wheel', (e) => this.onWheel(e, false), { passive: false });
      cv.addEventListener('mousedown', (e) => this.onDown(e));
      ru.addEventListener('mousedown', (e) => this.onRulerDown(e));
      cv.addEventListener('mousemove', (e) => this.onHover(e, 'tracks'));
      ru.addEventListener('mousemove', (e) => this.onHover(e, 'ruler'));
      cv.addEventListener('mouseleave', () => { this.pointer = null; this.hideTip(); });
      ru.addEventListener('mouseleave', () => { this.pointer = null; this.hideTip(); });
      window.addEventListener('mousemove', (e) => this.onDrag(e));
      window.addEventListener('mouseup', () => { this.drag = null; });
      window.addEventListener('resize', () => this.render());
      cv.addEventListener('dblclick', () => { this.fit(); this.render(); });
      document.getElementById('tl-fit').addEventListener('click', () => { this.fit(); this.render(); });
      this.labels.addEventListener('click', (e) => {
        const b = e.target.closest('.tl-lbl');
        if (!b) return;
        const key = b.getAttribute('data-row');
        if (this.collapsed.has(key)) this.collapsed.delete(key); else this.collapsed.add(key);
        this.render();
      });
      this.summary.addEventListener('click', (e) => {
        if (e.target.closest('[data-clear]')) { this.setFilter(null); return; }
        const r = e.target.closest('.par-row');
        if (!r) return;
        const f = { kind: r.getAttribute('data-kind'), key: r.getAttribute('data-key') };
        this.setFilter(this.filter && this.filter.kind === f.kind && this.filter.key === f.key ? null : f);
      });
    },

    reset(model) {
      this.model = model;
      this.filter = null;
      this.sumKey = null;
      this.chainCache = null;
      this.hideTip();
      this.fit();
    },
    fit() {
      if (!this.model) return;
      this.x0 = 0; this.x1 = this.model.cycles;
    },
    setFilter(f) {
      this.filter = f;
      this.sumKey = null;
      this.render();
    },

    // chrome
    buildLabels() {
      this.labels.innerHTML = ROWS.map((r) =>
        '<button type="button" class="tl-lbl" data-row="' + r.key + '" title="Collapse or expand the ' + esc(r.label) + ' row">' +
        '<span class="tw">▾</span>' + esc(r.label) + '</button>').join('');
      this.labelEls = Array.from(this.labels.querySelectorAll('.tl-lbl'));
    },

    buildLegend() {
      const sw = (cssVar, cls, text) => '<span class="lg-item"><span class="sw-c ' + (cls || '') + '" style="--c:var(' + cssVar + ')"></span>' + text + '</span>';
      const group = (name, items) => '<span class="lg-group"><b>' + name + '</b>' + items.join('') + '</span>';
      const stalls = Object.keys(STALL_VARS).map((k) => sw(STALL_VARS[k], '', k));
      const idle = Object.keys(IDLE_VARS).map((k) => sw(IDLE_VARS[k], k === 'busy' ? '' : 'hatch', k));
      this.legend.innerHTML = [
        group('SEQ', [sw('--ok', 'tick', 'issued'), sw('--bad', 'tick', 'trap')].concat(stalls)),
        group('units', [sw('--u-mxu', '', 'MXU stream'), sw('--u-mxu', 'faint', 'fill/drain'), '<span class="lg-item" title="selected instruction: solid outline; its blocker chain: dashed">▭ selected · ⋯ chain</span>']),
        group('FIFO', [sw('--u-weight', 'faint', 'occupied'), sw('--u-weight', '', 'ready'), sw('--u-weight', 'band', 'min–max'), sw('--accent', 'tick', 'prefetch')]),
        group('UB', [sw('--ub-read', '', 'read ↑'), sw('--ub-write', '', 'write ↓')]),
        group('ACC', [sw('--acc-write', '', 'write'), sw('--acc-read', '', 'Activate read')]),
        group('idle', idle),
      ].join('');
    },

    // geometry
    layout() {
      const m = this.model;
      const avail = Math.max(60, (this.scroll.clientHeight || 220) - RULER_H - 1);
      const mins = ROWS.map((r) => (r.key === 'ACC' ? Math.max(r.min, m.cfg.acc_banks * 7 + 4) : r.min));
      const open = ROWS.map((r) => !this.collapsed.has(r.key));
      let fixed = 0, minOpen = 0, nOpen = 0;
      ROWS.forEach((r, i) => { if (open[i]) { minOpen += mins[i]; nOpen++; } else fixed += COLLAPSED_H; });
      const extra = nOpen ? Math.max(0, Math.floor((avail - fixed - minOpen) / nOpen)) : 0;
      let y = 0;
      const rows = ROWS.map((r, i) => {
        const h = open[i] ? mins[i] + extra : COLLAPSED_H;
        const row = { key: r.key, label: r.label, index: i, y, h, collapsed: !open[i], pad: open[i] ? 3 : 2 };
        y += h;
        return row;
      });
      return { rows, H: y };
    },

    geometry() {
      const lay = this.layout();
      const dpr = window.devicePixelRatio || 1;
      if (this.canvas.style.height !== lay.H + 'px') this.canvas.style.height = lay.H + 'px';
      const heights = lay.rows.map((r) => r.h + (r.collapsed ? 'c' : '')).join(',');
      if (heights !== this.labelHeights) {
        this.labelHeights = heights;
        lay.rows.forEach((r, i) => {
          const el = this.labelEls[i];
          el.style.height = r.h + 'px';
          el.classList.toggle('collapsed', r.collapsed);
          el.setAttribute('aria-expanded', String(!r.collapsed));
          el.querySelector('.tw').textContent = r.collapsed ? '▸' : '▾';
        });
      }
      const W = Math.max(10, Math.floor(this.canvas.clientWidth || this.canvas.getBoundingClientRect().width || 10));
      const H = lay.H;
      if (this.canvas.width !== Math.round(W * dpr) || this.canvas.height !== Math.round(H * dpr)) {
        this.canvas.width = Math.round(W * dpr); this.canvas.height = Math.round(H * dpr);
      }
      if (this.ruler.width !== Math.round(W * dpr) || this.ruler.height !== Math.round(RULER_H * dpr)) {
        this.ruler.width = Math.round(W * dpr); this.ruler.height = Math.round(RULER_H * dpr);
      }
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.rctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const g = { W, H, rows: lay.rows, byKey: {} };
      for (const r of lay.rows) g.byKey[r.key] = r;
      return g;
    },
    cx(t, g) { return (t - this.x0) / (this.x1 - this.x0) * g.W; },
    tx(x, g) { return this.x0 + x / g.W * (this.x1 - this.x0); },

    // One call per cycle when zoomed in, one per pixel when zoomed out.
    forColumns(g, fn) {
      const m = this.model;
      const a = Math.max(0, Math.floor(this.x0)), b = Math.min(m.cycles, Math.ceil(this.x1));
      if (b - a <= g.W) {
        for (let t = a; t < b; t++) { const x = this.cx(t, g); fn(x, Math.max(1, this.cx(t + 1, g) - x), t, t + 1); }
        return;
      }
      const per = (this.x1 - this.x0) / g.W;
      for (let px = 0; px < g.W; px++) {
        const t0 = Math.max(a, Math.floor(this.x0 + px * per));
        if (t0 >= b) break;
        const t1 = Math.min(b, Math.max(t0 + 1, Math.floor(this.x0 + (px + 1) * per)));
        fn(px, 1, t0, t1);
      }
    },

    // Cycles [c0, c1) covered by the pixel column under x.
    columnAt(x, g) {
      const m = this.model;
      const px = Math.floor(x);
      const c0 = Math.max(0, Math.min(m.cycles - 1, Math.floor(this.tx(px, g))));
      const c1 = Math.max(c0 + 1, Math.min(m.cycles, Math.floor(this.tx(px + 1, g))));
      return [c0, c1];
    },

    hatchPattern(color) {
      if (this.hatch && this.hatch.color === color) return this.hatch.pat;
      let pat = null;
      try {
        const c = document.createElement('canvas');
        c.width = 6; c.height = 6;
        const x = c.getContext('2d');
        if (x && typeof this.ctx.createPattern === 'function') {
          x.strokeStyle = color; x.lineWidth = 1.2;
          x.beginPath(); x.moveTo(-1, 7); x.lineTo(7, -1); x.moveTo(-1, 1); x.lineTo(1, -1); x.moveTo(5, 7); x.lineTo(7, 5); x.stroke();
          pat = this.ctx.createPattern(c, 'repeat') || null;
        }
      } catch (err) { pat = null; }
      this.hatch = { color, pat };
      return pat;
    },

    chainPcs(pc) {
      if (this.chainCache && this.chainCache.pc === pc) return this.chainCache.set;
      const set = new Set();
      if (this.model.blockerChain) for (const h of this.model.blockerChain(pc)) if (h.pc !== pc) set.add(h.pc);
      this.chainCache = { pc, set };
      return set;
    },

    // A surface halo under an ink stroke, so it reads on any fill in either theme.
    outlineRect(x, y, w, h, P, dashed) {
      const ctx = this.ctx;
      ctx.save();
      ctx.lineWidth = dashed ? 3 : 3.5; ctx.strokeStyle = P.surface;
      ctx.strokeRect(x + 1, y + 1, Math.max(1, w - 2), Math.max(1, h - 2));
      if (dashed) ctx.setLineDash([4, 3]);
      ctx.lineWidth = dashed ? 1.4 : 1.8; ctx.strokeStyle = P.ink;
      ctx.strokeRect(x + 1, y + 1, Math.max(1, w - 2), Math.max(1, h - 2));
      ctx.restore();
    },

    ensureVisible(t) {
      if (!this.model) return;
      const span = this.x1 - this.x0;
      if (t < this.x0 || t >= this.x1) {
        const nx0 = Math.max(0, Math.min(this.model.cycles - span, t - span * 0.3));
        this.x0 = nx0; this.x1 = nx0 + span;
      }
    },

    // render
    render() {
      const m = this.model;
      if (!m) return;
      const st = this.app.state;
      const sel = this.app.sel;
      const g = (this.g = this.geometry());
      const P = (this.P = readPalette());
      const ctx = this.ctx;
      const span = this.x1 - this.x0;
      const pxPerCycle = g.W / span;
      const vis = (a, b) => b > this.x0 && a < this.x1;
      const X = (t) => this.cx(Math.max(this.x0, t), g);
      const XW = (a, b) => Math.max(1, this.cx(Math.min(this.x1, b), g) - X(a));
      const R = g.byKey;
      const inner = (row) => ({ y: row.y + row.pad, h: Math.max(1, row.h - 2 * row.pad) });

      ctx.clearRect(0, 0, g.W, g.H);
      ctx.fillStyle = P.surface; ctx.fillRect(0, 0, g.W, g.H);
      const step = niceStep(span / Math.max(4, g.W / 90));
      ctx.fillStyle = P.line;
      for (let t = Math.ceil(this.x0 / step) * step; t < this.x1; t += step) ctx.fillRect(Math.round(this.cx(t, g)), 0, 1, g.H);
      for (const row of g.rows) {
        if (row.collapsed) { ctx.fillStyle = P.surface2; ctx.fillRect(0, row.y, g.W, row.h); }
        ctx.fillStyle = P.line; ctx.fillRect(0, row.y + row.h - 1, g.W, 1);
      }

      // SEQ: one run per issue or per unchanged (pc, stall reason).
      {
        const { y, h } = inner(R.SEQ);
        for (const run of m.outcomeRuns) {
          if (!vis(run.from, run.to + 1)) continue;
          const o = run.outcome;
          if (o === 0 || o === 11) { ctx.fillStyle = P.ok; ctx.fillRect(X(run.from), y, Math.max(1, pxPerCycle), h); }
          else if (o === 10) { ctx.fillStyle = P.bad; ctx.fillRect(X(run.from), y, Math.max(2, pxPerCycle), h); }
          else {
            ctx.fillStyle = P.stall[MTV.STALL_NAMES[o]] || P.muted;
            const inset = R.SEQ.collapsed ? 0 : 2;
            ctx.fillRect(X(run.from), y + inset, XW(run.from, run.to + 1), h - 2 * inset);
          }
        }
      }

      // Unit rows: one bar per instruction in flight, [issue, done).
      const chain = sel && sel.type === 'instr' ? this.chainPcs(sel.pc) : null;
      ctx.font = '10px ' + P.mono;
      for (const u of UNIT_ROWS) {
        const row = R[u];
        const { y, h } = inner(row);
        for (const iv of m.unitIv[u]) {
          if (!vis(iv.issue, iv.done)) continue;
          const x = X(iv.issue), w = XW(iv.issue, iv.done);
          ctx.fillStyle = P.unit[u];
          if (u === 'MXU') {
            const mm = m.mmByPc.get(iv.pc);
            const split = Math.min(iv.done, iv.issue + (mm ? mm.len : 0));
            ctx.fillRect(x, y, XW(iv.issue, split), h);
            if (split < iv.done && vis(split, iv.done)) { ctx.globalAlpha = 0.45; ctx.fillRect(X(split), y, XW(split, iv.done), h); ctx.globalAlpha = 1; }
          } else ctx.fillRect(x, y, w, h);
          if (sel && sel.type === 'instr' && sel.pc === iv.pc) this.outlineRect(x, y, w, h, P, false);
          else if (chain && chain.has(iv.pc)) this.outlineRect(x, y, w, h, P, true);
          if (!row.collapsed && w > 30 && h >= 12) { ctx.fillStyle = P.onFill; ctx.fillText('pc ' + iv.pc, x + 3, y + h / 2 + 3.5); }
        }
      }

      // The selected instruction's stalls on SEQ, and the chain's stalls dashed.
      if (sel && sel.type === 'instr') {
        const { y, h } = inner(R.SEQ);
        const outline = (pc, dashed) => {
          for (const s of m.stallsByPc.get(pc) || []) {
            if (vis(s.from, s.to + 1)) this.outlineRect(X(s.from), y, XW(s.from, s.to + 1), h, P, dashed);
          }
        };
        if (chain) for (const pc of chain) outline(pc, true);
        outline(sel.pc, false);
        const p = m.program[sel.pc];
        if (p && p.issue !== null && vis(p.issue, p.issue + 1)) { ctx.fillStyle = P.ink; ctx.fillRect(X(p.issue), y, Math.max(2, pxPerCycle), h); }
      }

      // Occupancy and ready slots against the depth. A pixel covering several
      // cycles draws its minimum solid and the min–max range as a lighter band.
      {
        const { y, h } = inner(R.FIFO);
        const depth = Math.max(1, m.cfg.weight_fifo_depth);
        const lvl = (v) => h * Math.min(v, depth) / depth;
        ctx.fillStyle = P.unit.WEIGHT;
        this.forColumns(g, (x, w, t0, t1) => {
          let oMin = Infinity, oMax = 0, rMin = Infinity, rMax = 0;
          for (let t = t0; t < t1; t++) {
            const o = m.cyc.focc[t], r = m.cyc.fready[t];
            if (o < oMin) oMin = o; if (o > oMax) oMax = o;
            if (r < rMin) rMin = r; if (r > rMax) rMax = r;
          }
          ctx.globalAlpha = 0.3; ctx.fillRect(x, y + h - lvl(oMin), w, lvl(oMin));
          ctx.globalAlpha = 0.14; ctx.fillRect(x, y + h - lvl(oMax), w, lvl(oMax) - lvl(oMin));
          ctx.globalAlpha = 0.9; ctx.fillRect(x, y + h - lvl(rMin), w, lvl(rMin));
          ctx.globalAlpha = 0.45; ctx.fillRect(x, y + h - lvl(rMax), w, lvl(rMax) - lvl(rMin));
        });
        ctx.globalAlpha = 1;
        ctx.fillStyle = P.accent;
        for (const p of m.prefetches) if (vis(p.cycle, p.cycle + 1)) ctx.fillRect(X(p.cycle), y, Math.max(1, pxPerCycle * 0.6), Math.min(3, h));
      }

      // Read streams up from the midline, write streams down, against ub_banks.
      {
        const { y, h } = inner(R.UB);
        const banks = Math.max(1, m.cfg.ub_banks);
        const half = h / 2, mid = y + half;
        const lvl = (v) => half * Math.min(v, banks) / banks;
        ctx.fillStyle = P.line; ctx.fillRect(0, Math.round(mid), g.W, 1);
        this.forColumns(g, (x, w, t0, t1) => {
          let rMin = Infinity, rMax = 0, wMin = Infinity, wMax = 0;
          for (let t = t0; t < t1; t++) {
            const r = m.cyc.ubr[t], wv = m.cyc.ubw[t];
            if (r < rMin) rMin = r; if (r > rMax) rMax = r;
            if (wv < wMin) wMin = wv; if (wv > wMax) wMax = wv;
          }
          ctx.fillStyle = P.ubRead;
          ctx.globalAlpha = 0.9; ctx.fillRect(x, mid - lvl(rMin), w, lvl(rMin));
          ctx.globalAlpha = 0.4; ctx.fillRect(x, mid - lvl(rMax), w, lvl(rMax) - lvl(rMin));
          ctx.fillStyle = P.ubWrite;
          ctx.globalAlpha = 0.9; ctx.fillRect(x, mid, w, lvl(wMin));
          ctx.globalAlpha = 0.4; ctx.fillRect(x, mid + lvl(wMin), w, lvl(wMax) - lvl(wMin));
        });
        ctx.globalAlpha = 1;
        if (!R.UB.collapsed) { ctx.fillStyle = P.muted; ctx.font = '9px ' + P.mono; ctx.fillText('r', 2, y + 8); ctx.fillText('w', 2, y + h - 1); }
      }

      // ACC banks: one lane per bank; write reservations and Activate reads.
      {
        const { y, h } = inner(R.ACC);
        const nb = Math.max(1, m.cfg.acc_banks), lane = h / nb;
        for (const iss of m.issues) {
          if (!vis(iss.cycle, iss.done)) continue;
          const p = m.program[iss.pc];
          if (!p.res) continue;
          if (p.res.acc_write !== null) { ctx.fillStyle = P.accWrite; ctx.fillRect(X(iss.cycle), y + p.res.acc_write * lane, XW(iss.cycle, iss.done), Math.max(1, lane - 1)); }
          if (p.res.acc_read !== null && p.op === 'Activate') { ctx.fillStyle = P.accRead; ctx.fillRect(X(iss.cycle), y + p.res.acc_read * lane, XW(iss.cycle, iss.done), Math.max(1, lane - 1)); }
        }
        if (!R.ACC.collapsed && lane >= 7) {
          ctx.font = '8px ' + P.mono;
          for (let b = 0; b < nb; b++) {
            ctx.globalAlpha = 0.85; ctx.fillStyle = P.surface; ctx.fillRect(0, y + b * lane, 12, lane - 1);
            ctx.globalAlpha = 1; ctx.fillStyle = P.muted; ctx.fillText('b' + b, 1, y + b * lane + lane - 1.5);
          }
        }
      }

      // Idle cause: charge_idle_cycle's bucket per cycle; idle buckets hatched.
      {
        const { y, h } = inner(R.IDLE);
        const pat = this.hatchPattern(P.hatch);
        for (const run of m.idleRuns) {
          if (!vis(run.from, run.to + 1)) continue;
          const name = MTV.IDLE_NAMES[run.bucket];
          const x = X(run.from), w = XW(run.from, run.to + 1);
          ctx.fillStyle = P.idle[name] || P.muted;
          ctx.fillRect(x, y, w, h);
          if (name !== 'busy' && pat) { ctx.fillStyle = pat; ctx.fillRect(x, y, w, h); }
        }
      }

      // Filter: dim every column in which no cycle matches.
      if (this.filter) {
        const pre = m.counts();
        const idx = this.filter.kind === 'idle' ? MTV.IDLE_NAMES.indexOf(this.filter.key) : MTV.STALL_NAMES.indexOf(this.filter.key);
        const arr = this.filter.kind === 'idle' ? pre.idle[idx] : pre.outcome[idx];
        if (arr) {
          ctx.fillStyle = P.surface; ctx.globalAlpha = 0.74;
          let runX = null, runEnd = 0;
          this.forColumns(g, (x, w, t0, t1) => {
            const match = arr[t1] - arr[t0] > 0;
            if (!match) { if (runX === null) runX = x; runEnd = x + w; }
            else if (runX !== null) { ctx.fillRect(runX, 0, runEnd - runX, g.H); runX = null; }
          });
          if (runX !== null) ctx.fillRect(runX, 0, runEnd - runX, g.H);
          ctx.globalAlpha = 1;
        }
      }

      // Cursor line (its label lives in the ruler band).
      if (st) { ctx.fillStyle = P.accent; ctx.fillRect(Math.round(this.cx(st.t + 0.5, g)) - 1, 0, 2, g.H); }

      this.renderRuler(g, P, step);
      this.renderSummary();
      if (this.pointer) this.refreshTip();
    },

    renderRuler(g, P, step) {
      const m = this.model, st = this.app.state, ctx = this.rctx;
      ctx.clearRect(0, 0, g.W, RULER_H);
      ctx.fillStyle = P.surface; ctx.fillRect(0, 0, g.W, RULER_H);
      ctx.font = '10px ' + P.mono;
      for (let t = Math.ceil(this.x0 / step) * step; t < this.x1; t += step) {
        const x = Math.round(this.cx(t, g));
        ctx.fillStyle = P.lineStrong; ctx.fillRect(x, RULER_H - 5, 1, 5);
        ctx.fillStyle = P.muted; ctx.fillText(fmt(t), x + 3, 12);
      }
      ctx.fillStyle = P.accent;
      for (const b of m.tileBoundaries()) if (b.cycle >= this.x0 && b.cycle < this.x1) ctx.fillRect(Math.round(this.cx(b.cycle, g)), RULER_H - 3, 1, 3);
      if (st) {
        const x = this.cx(st.t + 0.5, g);
        const text = 't = ' + fmt(st.t);
        const tw = (ctx.measureText && ctx.measureText(text).width) || text.length * 6.5;
        const bx = x + 4 + tw + 6 > g.W ? x - tw - 10 : x + 4;
        ctx.fillStyle = P.accent; ctx.fillRect(Math.round(x) - 1, 0, 2, RULER_H);
        ctx.fillRect(bx, 2, tw + 6, RULER_H - 5);
        ctx.fillStyle = P.onFill; ctx.fillText(text, bx + 3, 12);
      }
    },

    // summary
    windowBounds() {
      const m = this.model;
      const a = Math.max(0, Math.floor(this.x0));
      const b = Math.max(a + 1, Math.min(m.cycles, Math.ceil(this.x1)));
      return [a, b];
    },

    renderSummary() {
      const m = this.model;
      const [a, b] = this.windowBounds();
      const key = a + ':' + b + ':' + (this.filter ? this.filter.kind + '/' + this.filter.key : '');
      if (key === this.sumKey) return;
      this.sumKey = key;
      const wc = m.windowCounts(a, b);
      const n = wc.n;
      const pct = (v, of) => (of ? (100 * v / of).toFixed(v && 100 * v / of < 1 ? 2 : 1) : '0') + '%';
      const busy = wc.idle.busy;
      const issued = wc.outcome.issued + wc.outcome.halt;
      document.getElementById('sum-window').textContent = 'cycles ' + fmt(a) + '–' + fmt(b - 1) + ' · ' + fmt(n);
      document.getElementById('sum-sub').textContent = 'array busy ' + fmt(busy) + ' (' + pct(busy, n) + ') · idle ' + fmt(n - busy) + ' (' + pct(n - busy, n) + ') · issued ' + fmt(issued) + (wc.outcome.trap ? ' · trap ' + wc.outcome.trap : '');

      const pareto = (kind, names, counts, vars, hatched) => {
        const items = names.map((k) => ({ k, v: counts[k] })).filter((x) => x.v > 0).sort((p, q) => q.v - p.v);
        const zero = names.filter((k) => !counts[k]);
        const total = items.reduce((s, x) => s + x.v, 0);
        const max = items.length ? items[0].v : 1;
        let cum = 0;
        let html = '<div class="par-zero">' + fmt(total) + ' cycles (' + pct(total, n) + ' of window)</div>';
        html += items.map((x) => {
          cum += x.v;
          const active = this.filter && this.filter.kind === kind && this.filter.key === x.k;
          return '<button type="button" class="par-row' + (active ? ' active' : '') + '" data-kind="' + kind + '" data-key="' + x.k + '"' +
            ' title="' + esc(x.k + ': ' + fmt(x.v) + ' cycles, ' + pct(x.v, n) + ' of the window; cumulative ' + pct(cum, total) + ' of ' + (kind === 'idle' ? 'idle' : 'refused') + ' cycles. Click to dim the rest.') + '">' +
            '<span class="sw-c' + (hatched(x.k) ? ' hatch' : '') + '" style="--c:var(' + vars[x.k] + ')"></span>' +
            '<span class="par-name">' + esc(x.k) + '</span>' +
            '<span class="par-n">' + fmt(x.v) + '<span class="par-cum">Σ' + pct(cum, total) + '</span></span>' +
            '<span class="par-bar"><i style="width:' + (100 * x.v / max).toFixed(1) + '%;--c:var(' + vars[x.k] + ')"></i></span>' +
            '</button>';
        }).join('');
        if (!items.length) html += '<div class="par-zero">none in this window</div>';
        if (zero.length && items.length) html += '<div class="par-zero">zero: ' + zero.map(esc).join(', ') + '</div>';
        return html;
      };
      const idleNames = MTV.IDLE_NAMES.filter((k) => k !== 'busy');
      const stallNames = MTV.STALL_NAMES.filter((k) => STALL_VARS[k]);
      document.getElementById('sum-idle').innerHTML = pareto('idle', idleNames, wc.idle, IDLE_VARS, () => true);
      document.getElementById('sum-stall').innerHTML = pareto('stall', stallNames, wc.outcome, STALL_VARS, () => false);

      const f = document.getElementById('sum-filter');
      if (this.filter) {
        const hits = m.matchCount(this.filter, a, b);
        const what = this.filter.kind === 'idle' ? 'cycles charged to ' + this.filter.key : 'cycles refused on ' + this.filter.key;
        f.innerHTML = 'Showing ' + esc(what) + ': ' + fmt(hits) + ' of ' + fmt(n) + ' in the window; the rest is dimmed. <a data-clear>clear</a>';
      } else {
        f.textContent = 'No filter.';
      }
    },

    // input
    onWheel(e, tracks) {
      if (!this.model) return;
      if (tracks && e.altKey) return;   // let the row area scroll vertically
      e.preventDefault();
      const g = this.g || this.geometry();
      const span = this.x1 - this.x0;
      const horizontal = e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY);
      if (horizontal) {
        this.pan(((Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY) / g.W) * span);
      } else {
        const f = Math.exp(e.deltaY * 0.0015);
        const at = this.tx(e.offsetX, g);
        const ns = Math.max(Math.min(24, this.model.cycles), Math.min(this.model.cycles, span * f));
        let nx0 = at - (at - this.x0) * (ns / span);
        nx0 = Math.max(0, Math.min(this.model.cycles - ns, nx0));
        this.x0 = nx0; this.x1 = nx0 + ns;
      }
      this.render();
    },
    pan(d) {
      const span = this.x1 - this.x0;
      const nx0 = Math.max(0, Math.min(this.model.cycles - span, this.x0 + d));
      this.x0 = nx0; this.x1 = nx0 + span;
    },

    onDown(e) {
      if (!this.model || e.button !== 0) return;
      const g = this.g || this.geometry();
      const info = this.hitTest(e.offsetX, e.offsetY, g);
      if (!info) return;
      if (info.seek || e.altKey) this.app.setCycle(info.t);
      if (info.select) this.app.select(info.select);
      else this.drag = { el: this.canvas };
    },
    onRulerDown(e) {
      if (!this.model || e.button !== 0) return;
      const g = this.g || this.geometry();
      this.app.setCycle(Math.floor(this.tx(e.offsetX, g)));
      this.drag = { el: this.ruler };
    },
    onDrag(e) {
      if (!this.drag || !this.model) return;
      const g = this.g || this.geometry();
      const rect = this.drag.el.getBoundingClientRect();
      const x = Math.max(0, Math.min(g.W - 1, e.clientX - rect.left));
      this.app.setCycle(Math.floor(this.tx(x, g)));
    },

    onHover(e, where) {
      if (!this.model) return;
      this.pointer = { where, x: e.offsetX, y: e.offsetY, cx: e.clientX, cy: e.clientY };
      this.refreshTip();
    },
    refreshTip() {
      const p = this.pointer;
      if (!p || !this.model) return;
      const g = this.g || this.geometry();
      const info = p.where === 'ruler' ? this.rulerInfo(p.x, g) : this.hitTest(p.x, p.y, g);
      if (!info) { this.hideTip(); return; }
      this.showTip(info, p.cx, p.cy);
    },
    hideTip() { if (this.tip) this.tip.hidden = true; },
    showTip(info, cx, cy) {
      const tip = this.tip;
      tip.innerHTML = '<div class="tip-head">' + esc(info.label) + ' · cycle ' + fmt(info.t) +
        (info.c1 - info.c0 > 1 ? ' <span class="mono">(pixel covers ' + fmt(info.c0) + '–' + fmt(info.c1 - 1) + ')</span>' : '') + '</div>' +
        info.lines.map((l) => '<div class="tip-line">' + l + '</div>').join('') +
        (info.hint ? '<div class="tip-hint">' + esc(info.hint) + '</div>' : '');
      tip.hidden = false;
      const vw = window.innerWidth || 1200, vh = window.innerHeight || 800;
      const r = tip.getBoundingClientRect();
      const w = r.width || 300, h = r.height || 100;
      let left = cx + 14, top = cy - h - 12;
      if (left + w > vw - 8) left = cx - w - 14;
      if (top < 8) top = cy + 16;
      if (top + h > vh - 8) top = Math.max(8, vh - h - 8);
      tip.style.left = Math.max(8, left) + 'px';
      tip.style.top = top + 'px';
    },

    rulerInfo(x, g) {
      const m = this.model;
      const [c0, c1] = this.columnAt(x, g);
      const t = c0;
      const lines = [];
      const near = m.tileBoundaries().filter((b) => Math.abs(this.cx(b.cycle, g) - x) <= 4);
      for (const b of near.slice(0, 3)) lines.push(esc(b.label) + ' at cycle ' + fmt(b.cycle));
      if (!lines.length) lines.push('click or drag to seek; wheel to zoom');
      return { label: 'ruler', t, c0, c1, lines, hint: null };
    },

    // What occupies (x, y) and what a click there does.
    hitTest(x, y, g) {
      const m = this.model;
      if (!m) return null;
      g = g || this.g || this.geometry();
      const row = g.rows.find((r) => y >= r.y && y < r.y + r.h);
      if (!row) return null;
      const [c0, c1] = this.columnAt(x, g);
      const t = Math.max(0, Math.min(m.cycles - 1, Math.floor(this.tx(x, g))));
      const info = { row: row.key, label: row.label, t, c0, c1, lines: [], select: null, seek: false, hint: null };
      const L = (html) => info.lines.push(html);
      const cy = (v) => '<span class="mono">' + fmt(v) + '</span>';
      const der = (text) => ' <span class="chip der">' + esc(text) + '</span>';
      const inFlight = (u) => m.unitAt(u, t, 'account');

      switch (row.key) {
        case 'SEQ': {
          const o = m.outcomeAt(t);
          if (o.kind === 'stall') {
            const sp = m.stallSpanAt(t);
            L(esc(label(m, o.pc)) + ' cannot issue: <b>' + esc(o.reason) + '</b>');
            if (sp) L('stall span ' + cy(sp.from) + '–' + cy(sp.to) + ' (' + fmt(sp.to - sp.from + 1) + ' cycles)');
            if (o.blockerPc !== null) {
              const b = m.program[o.blockerPc];
              L('blocked by ' + esc(label(m, o.blockerPc)) + ' on ' + esc(o.blockerUnit) + (b && b.retire !== null ? ', retires at ' + cy(b.retire) : ''));
              info.select = { type: 'instr', pc: o.blockerPc };
              info.hint = 'click: seek here and select the blocker, pc ' + o.blockerPc;
            } else {
              L('no blocking instruction recorded' + (o.reason === 'weight_fifo_empty' ? ': the tile has not arrived from DDR' : o.reason === 'ub_bank_conflict' ? ': the stream budget is exhausted' : ''));
              info.select = { type: 'instr', pc: o.pc };
              info.hint = 'click: seek here and select pc ' + o.pc;
            }
          } else if (o.kind === 'issued' || o.kind === 'halt') {
            const iss = m.issuesByCycle.get(t);
            L(esc(label(m, o.pc)) + ' issued' + (iss && o.kind === 'issued' ? ' on ' + esc(iss.unit) + ', in flight ' + cy(iss.cycle) + '–' + cy(iss.done - 1) + ', retires at ' + cy(iss.done) : ': the run ends'));
            info.select = { type: 'instr', pc: o.pc };
            info.hint = 'click: seek here and select pc ' + o.pc;
          } else {
            L('trap at pc ' + o.pc + (m.m.traps && m.m.traps[0] ? ': ' + esc(m.m.traps[0].reason) : ''));
            if (m.program[o.pc]) info.select = { type: 'instr', pc: o.pc };
          }
          info.seek = true;
          break;
        }
        case 'DMA': case 'WEIGHT': case 'MXU': case 'ACT': {
          const iv = m.unitIv[row.key];
          const i = MTV.upperBound(iv, (v) => v.issue, c1 - 1) - 1;
          const hit = i >= 0 && iv[i].done > c0 ? iv[i] : null;
          if (hit) {
            const p = m.program[hit.pc];
            L(esc(label(m, hit.pc)));
            L('in flight ' + cy(hit.issue) + '–' + cy(hit.done - 1) + ', retires at ' + cy(hit.done) + ' (' + fmt(hit.done - hit.issue) + ' cycles)');
            if (row.key === 'MXU') {
              const mm = m.mmByPc.get(hit.pc);
              const s = Math.max(0, Math.min(mm.steps - 1, t - mm.issue));
              L('array step ' + s + ' of ' + mm.steps + ': ' + (s < mm.len ? 'streaming (row ' + s + ' admitted)' : 'fill/drain') + der('stream = first len steps'));
            } else if (row.key === 'DMA' || row.key === 'ACT') {
              L('cycle ' + fmt(Math.min(hit.done - hit.issue, t - hit.issue + 1)) + ' of ' + fmt(hit.done - hit.issue) + ' <span class="chip nom">nominal</span> no per-cycle data movement is modeled');
            }
            if (p && p.res) {
              const parts = [];
              if (p.res.ub_read[1] > p.res.ub_read[0]) parts.push('UB read [' + MTV.hex(p.res.ub_read[0]) + ', ' + MTV.hex(p.res.ub_read[1]) + ')');
              if (p.res.ub_write[1] > p.res.ub_write[0]) parts.push('UB write [' + MTV.hex(p.res.ub_write[0]) + ', ' + MTV.hex(p.res.ub_write[1]) + ')');
              if (p.res.acc_write !== null) parts.push('bank ' + p.res.acc_write + ' write');
              if (p.res.acc_read !== null) parts.push('bank ' + p.res.acc_read + ' read');
              if (parts.length) L('reservation: ' + esc(parts.join('; ')));
            }
            info.select = { type: 'instr', pc: hit.pc };
            info.hint = 'click: select pc ' + hit.pc + ' (alt+click also seeks)';
          } else {
            const next = iv[MTV.upperBound(iv, (v) => v.issue, t)];
            L(esc(row.key) + ' unit idle' + (next ? '; next: ' + esc(label(m, next.pc)) + ' at cycle ' + cy(next.issue) : '; no more work'));
            info.seek = true;
            info.hint = 'click: seek here';
          }
          break;
        }
        case 'FIFO': {
          const depth = m.cfg.weight_fifo_depth;
          const range = (arr) => { let lo = Infinity, hi = 0; for (let k = c0; k < c1; k++) { lo = Math.min(lo, arr[k]); hi = Math.max(hi, arr[k]); } return [lo, hi]; };
          L('occupancy <b>' + m.cyc.focc[t] + '</b> / ' + depth + ' slots · ready <b>' + m.cyc.fready[t] + '</b> / ' + depth);
          if (c1 - c0 > 1) {
            const [oLo, oHi] = range(m.cyc.focc), [rLo, rHi] = range(m.cyc.fready);
            L('over this pixel: occupancy ' + oLo + '–' + oHi + ', ready ' + rLo + '–' + rHi);
          }
          for (const p of (m.prefetchByCycle.get(t) || []).slice(0, 4)) L('prefetch pushed DDR ' + MTV.hex(p.ddr) + ' for ' + esc(label(m, p.for_pc)) + ', ready at ' + cy(p.ready));
          const w = inFlight('WEIGHT');
          if (w) L('Read_Weights in flight: ' + esc(label(m, w.pc)));
          L('<span class="muted">sampled after cycle ' + fmt(t) + '’s accounting; ready = arrived from DDR</span>');
          info.select = { type: 'fifo', slot: null };
          info.seek = true;
          info.hint = 'click: seek here and inspect the FIFO';
          break;
        }
        case 'UB': {
          const banks = m.cfg.ub_banks;
          L('read streams <b>' + m.cyc.ubr[t] + '</b> / ' + banks + ' · write streams <b>' + m.cyc.ubw[t] + '</b> / ' + banks);
          if (c1 - c0 > 1) {
            let rLo = Infinity, rHi = 0, wLo = Infinity, wHi = 0;
            for (let k = c0; k < c1; k++) { rLo = Math.min(rLo, m.cyc.ubr[k]); rHi = Math.max(rHi, m.cyc.ubr[k]); wLo = Math.min(wLo, m.cyc.ubw[k]); wHi = Math.max(wHi, m.cyc.ubw[k]); }
            L('over this pixel: read ' + rLo + '–' + rHi + ', write ' + wLo + '–' + wHi);
          }
          const readers = [], writers = [];
          for (const u of MTV.UNITS) {
            const x = inFlight(u);
            if (!x) continue;
            const res = m.program[x.pc].res;
            if (!res) continue;
            if (res.ub_read[1] > res.ub_read[0]) readers.push('pc ' + x.pc);
            if (res.ub_write[1] > res.ub_write[0]) writers.push('pc ' + x.pc);
          }
          L('holders: read ' + (readers.join(', ') || 'none') + ' · write ' + (writers.join(', ') || 'none') + der('from reservations'));
          L('<span class="muted">a stream-count model: ub_bank_conflict fires only when a same-direction count reaches ' + banks + '</span>');
          info.seek = true;
          info.hint = 'click: seek here';
          break;
        }
        case 'ACC': {
          const nb = Math.max(1, m.cfg.acc_banks);
          const lane = Math.max(0, Math.min(nb - 1, Math.floor((y - row.y - row.pad) / Math.max(1, (row.h - 2 * row.pad) / nb))));
          let writer = null, reader = null;
          for (const u of MTV.UNITS) {
            const x = inFlight(u);
            if (!x) continue;
            const p = m.program[x.pc];
            if (!p.res) continue;
            if (p.res.acc_write === lane) writer = x;
            if (p.res.acc_read === lane && p.op === 'Activate') reader = x;
          }
          L('<b>bank ' + lane + '</b>');
          if (writer) {
            const p = m.program[writer.pc];
            L('reserved for write by ' + esc(label(m, writer.pc)) + (p.fields.accumulate ? ' (accumulate: also reads it)' : '') + ', ' + cy(writer.issue) + '–' + cy(writer.done - 1));
          }
          if (reader) L('reserved for read by ' + esc(label(m, reader.pc)) + ', ' + cy(reader.issue) + '–' + cy(reader.done - 1));
          if (!writer && !reader) L('free at this cycle');
          const owner = writer || reader;
          if (owner) { info.select = { type: 'instr', pc: owner.pc }; info.hint = 'click: select pc ' + owner.pc; }
          else { info.select = { type: 'bank', bank: lane }; info.seek = true; info.hint = 'click: seek here and inspect bank ' + lane; }
          break;
        }
        case 'IDLE': {
          const run = m.idleRunAt(t);
          const ib = m.idleBlockerAt(t);
          const o = ib.o;
          L(o.idle === 'busy' ? 'array <b>busy</b>' : 'array idle, charged to <b>' + esc(o.idle) + '</b>');
          if (run) L('run ' + cy(run.from) + '–' + cy(run.to) + ' (' + fmt(run.to - run.from + 1) + ' cycles)');
          L('<span class="muted">rule: ' + esc(IDLE_RULES[o.idle] || '') + '</span>');
          if (ib.via === 'blocker') {
            L('blocking instruction: <b>' + esc(label(m, ib.pc)) + '</b>, the recorded blocker of ' + esc(label(m, o.pc)) + '’s ' + esc(o.reason) + ' stall');
          } else if (ib.via === 'charge') {
            L('blocking instruction: <b>' + esc(label(m, ib.pc)) + '</b>, the ' + esc(ib.unit) + ' instruction the charge rule found active' + der('occupancy at this cycle'));
            if (o.kind === 'stall') L(esc(label(m, o.pc)) + ' stalled on ' + esc(o.reason) + ' with no blocker recorded');
          } else if (ib.via === 'busy') {
            L('MatMul in flight: <b>' + esc(label(m, ib.pc)) + '</b>');
          } else if (ib.via === 'stalled') {
            L(o.kind === 'stall' ? esc(label(m, o.pc)) + ' stalled on ' + esc(o.reason) + ' with no blocker recorded' : esc(label(m, o.pc)) + ' ' + esc(o.kind) + ' this cycle; no instruction is blocking');
          }
          if (ib.pc !== null) { info.select = { type: 'instr', pc: ib.pc }; info.hint = 'click: seek here and select pc ' + ib.pc; }
          info.seek = true;
          break;
        }
        default:
          return null;
      }
      return info;
    },
  };

  MTV.views.timeline = view;
  MTV.timelineRows = ROWS;
})();
