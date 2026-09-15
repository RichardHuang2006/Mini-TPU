// Bottom timeline: activity per unit across cycles, the weight FIFO, the
// Unified Buffer port streams, accumulator bank reservations, and the array's
// idle-cause bucket, all drawn from trace records. Click a bar to select the
// instruction, click empty space to move the cycle, wheel to zoom, shift+wheel
// or drag on the ruler to pan.
//
// Classic script: defines window.MTV.views.timeline.
(function () {
  'use strict';
  const MTV = (window.MTV = window.MTV || {});
  MTV.views = MTV.views || {};

  const ROWS = ['SEQ', 'DMA', 'WEIGHT', 'MXU', 'ACT', 'FIFO', 'UB ports', 'ACC banks', 'idle cause'];
  const STALL_COLORS = {
    drain: '#8a8f98', unit_busy: '#c7891a', weight_fifo_empty: '#a45db8', ub_raw: '#b4383b', ub_war: '#d9736f',
    ub_waw: '#e39a97', accum_hazard: '#2c6fb0', weight_stall: '#6b5bb5', ub_bank_conflict: '#6a6a6a',
  };
  const IDLE_COLORS = { busy: null, weights: '#a45db8', bank: '#6a6a6a', accum: '#2c6fb0', dma: '#b4562a', act: '#2c6fb0', other: '#8a8f98' };

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const view = {
    init(app) {
      this.app = app;
      this.canvas = document.getElementById('tl-canvas');
      this.labels = document.getElementById('tl-labels');
      this.legend = document.getElementById('tl-legend');
      this.ctx = this.canvas.getContext('2d');
      this.x0 = 0; this.x1 = 1;   // visible cycle window [x0, x1)
      this.labels.innerHTML = ROWS.map((r) => '<div>' + r + '</div>').join('');
      this.legend.innerHTML = [['issued', css('--ok')], ['unit_busy', STALL_COLORS.unit_busy], ['ub_raw', STALL_COLORS.ub_raw], ['accum_hazard', STALL_COLORS.accum_hazard],
        ['weight_fifo_empty', STALL_COLORS.weight_fifo_empty], ['drain', STALL_COLORS.drain]]
        .map(([n, c]) => '<span><span class="sw" style="background:' + c + '"></span>' + n + '</span>').join('') +
        '<span class="muted">wheel: zoom · shift+wheel: pan · click: seek / select</span>';
      const cv = this.canvas;
      cv.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
      cv.addEventListener('mousedown', (e) => this.onDown(e));
      cv.addEventListener('mousemove', (e) => this.onMove(e));
      window.addEventListener('mouseup', () => { this.drag = null; });
      window.addEventListener('resize', () => this.render());
      cv.addEventListener('dblclick', (e) => { this.fit(); this.render(); });
    },

    reset(model) {
      this.model = model;
      this.fit();
    },
    fit() {
      if (!this.model) return;
      this.x0 = 0; this.x1 = this.model.cycles;
    },

    geometry() {
      const rect = this.canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const W = Math.max(10, Math.floor(rect.width)), H = Math.max(10, Math.floor(rect.height));
      if (this.canvas.width !== W * dpr || this.canvas.height !== H * dpr) {
        this.canvas.width = W * dpr; this.canvas.height = H * dpr;
      }
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const top = 16, bottom = 4;
      const rowH = (H - top - bottom) / ROWS.length;
      return { W, H, top, rowH };
    },
    cx(t, g) { return (t - this.x0) / (this.x1 - this.x0) * g.W; },
    tx(x, g) { return this.x0 + x / g.W * (this.x1 - this.x0); },

    onWheel(e) {
      e.preventDefault();
      const g = this.geometry();
      const span = this.x1 - this.x0;
      if (e.shiftKey) {
        const d = (e.deltaY || e.deltaX) / g.W * span;
        this.pan(d);
      } else {
        const f = Math.exp(e.deltaY * 0.0015);
        const at = this.tx(e.offsetX, g);
        let ns = Math.max(24, Math.min(this.model.cycles, span * f));
        let nx0 = at - (at - this.x0) * (ns / span);
        nx0 = Math.max(0, Math.min(this.model.cycles - ns, nx0));
        this.x0 = nx0; this.x1 = nx0 + ns;
      }
      this.render();
    },
    pan(d) {
      const span = this.x1 - this.x0;
      let nx0 = Math.max(0, Math.min(this.model.cycles - span, this.x0 + d));
      this.x0 = nx0; this.x1 = nx0 + span;
    },
    onDown(e) {
      const g = this.geometry();
      const t = Math.floor(this.tx(e.offsetX, g));
      const hit = this.hitTest(e.offsetX, e.offsetY, g);
      if (hit && hit.pc !== undefined) {
        if (hit.seq || e.altKey) this.app.setCycle(t);
        this.app.select({ type: 'instr', pc: hit.pc });
      } else {
        this.app.setCycle(t);
        this.drag = { scrub: true };
      }
    },
    onMove(e) {
      if (!this.drag || !this.drag.scrub) return;
      const g = this.geometry();
      this.app.setCycle(Math.floor(this.tx(e.offsetX, g)));
    },
    hitTest(x, y, g) {
      const row = Math.floor((y - g.top) / g.rowH);
      const t = this.tx(x, g);
      const m = this.model;
      const unit = ['SEQ', 'DMA', 'WEIGHT', 'MXU', 'ACT'][row];
      if (row >= 1 && row <= 4) {
        for (const iv of m.unitIv[unit]) if (t >= iv.issue && t < iv.done) return { pc: iv.pc };
      }
      if (row === 0) {
        const tt = Math.floor(t);
        if (tt >= 0 && tt < m.cycles) return { pc: m.cyc.pc[tt], seq: true };
      }
      return null;
    },

    ensureVisible(t) {
      const span = this.x1 - this.x0;
      if (t < this.x0 || t >= this.x1) {
        let nx0 = Math.max(0, Math.min(this.model.cycles - span, t - span * 0.3));
        this.x0 = nx0; this.x1 = nx0 + span;
      }
    },

    render() {
      const m = this.model;
      if (!m) return;
      const st = this.app.state;
      const sel = this.app.sel;
      const g = this.geometry();
      const ctx = this.ctx;
      const ink = css('--ink'), muted = css('--muted'), line = css('--line'), surface = css('--surface'), surface2 = css('--surface-2');
      const unitColor = { DMA: css('--u-dma'), WEIGHT: css('--u-weight'), MXU: css('--u-mxu'), ACT: css('--u-act'), SEQ: css('--u-seq') };
      ctx.clearRect(0, 0, g.W, g.H);
      ctx.fillStyle = surface; ctx.fillRect(0, 0, g.W, g.H);
      const span = this.x1 - this.x0;
      const pxPerCycle = g.W / span;

      // Ruler
      ctx.font = '10px ' + css('--mono');
      ctx.fillStyle = muted;
      const step = niceStep(span / Math.max(4, g.W / 90));
      for (let t = Math.ceil(this.x0 / step) * step; t < this.x1; t += step) {
        const x = this.cx(t, g);
        ctx.fillStyle = line; ctx.fillRect(Math.round(x), g.top, 1, g.H - g.top);
        ctx.fillStyle = muted; ctx.fillText(String(t), x + 3, 11);
      }
      // Row separators
      for (let r = 0; r <= ROWS.length; r++) { ctx.fillStyle = line; ctx.fillRect(0, Math.round(g.top + r * g.rowH), g.W, 1); }

      const rowY = (r) => g.top + r * g.rowH + 3;
      const rowHh = g.rowH - 6;
      const visible = (a, b) => b > this.x0 && a < this.x1;
      const X = (t) => this.cx(Math.max(this.x0, t), g);
      const XW = (a, b) => Math.max(1, this.cx(Math.min(this.x1, b), g) - X(a));

      // SEQ row: outcome runs
      for (const run of m.outcomeRuns) {
        if (!visible(run.from, run.to + 1)) continue;
        const o = run.outcome;
        if (o === 0 || o === 11) { ctx.fillStyle = css('--ok'); ctx.fillRect(X(run.from), rowY(0), Math.max(1, pxPerCycle), rowHh); }
        else if (o === 10) { ctx.fillStyle = css('--bad'); ctx.fillRect(X(run.from), rowY(0), Math.max(2, pxPerCycle), rowHh); }
        else { ctx.fillStyle = STALL_COLORS[MTV.STALL_NAMES[o]] || '#999'; ctx.globalAlpha = 0.85; ctx.fillRect(X(run.from), rowY(0) + 2, XW(run.from, run.to + 1), rowHh - 4); ctx.globalAlpha = 1; }
      }
      // Unit rows
      ['DMA', 'WEIGHT', 'MXU', 'ACT'].forEach((u, i) => {
        const r = i + 1;
        for (const iv of m.unitIv[u]) {
          if (!visible(iv.issue, iv.done)) continue;
          const x = X(iv.issue), w = XW(iv.issue, iv.done);
          ctx.fillStyle = unitColor[u];
          if (u === 'MXU') {
            const mm = m.mmByPc.get(iv.pc);
            const split = iv.issue + (mm ? mm.len : 0);
            ctx.fillRect(x, rowY(r), XW(iv.issue, Math.min(split, iv.done)), rowHh);
            ctx.globalAlpha = 0.45; ctx.fillRect(X(split), rowY(r), XW(split, iv.done), rowHh); ctx.globalAlpha = 1;
          } else ctx.fillRect(x, rowY(r), w, rowHh);
          if (sel && sel.type === 'instr' && sel.pc === iv.pc) { ctx.strokeStyle = css('--sel'); ctx.lineWidth = 2; ctx.strokeRect(x, rowY(r), w, rowHh); }
          if (w > 28) { ctx.fillStyle = '#fff'; ctx.font = '10px ' + css('--mono'); ctx.fillText('pc ' + iv.pc, x + 3, rowY(r) + rowHh / 2 + 3.5); }
        }
      });
      // Selected instruction's stall spans on the SEQ row
      if (sel && sel.type === 'instr') {
        for (const s of m.stalls) if (s.pc === sel.pc && visible(s.from, s.to + 1)) { ctx.strokeStyle = css('--sel'); ctx.lineWidth = 2; ctx.strokeRect(X(s.from), rowY(0) + 1, XW(s.from, s.to + 1), rowHh - 2); }
        const p = m.program[sel.pc];
        if (p && p.issue !== null && visible(p.issue, p.issue + 1)) { ctx.fillStyle = css('--sel'); ctx.fillRect(X(p.issue), rowY(0), Math.max(2, pxPerCycle), rowHh); }
      }
      // FIFO occupancy (sampled per pixel column)
      {
        const y = rowY(5), h = rowHh, depth = m.cfg.weight_fifo_depth;
        ctx.fillStyle = css('--u-weight');
        const cols = Math.min(g.W, Math.ceil(span));
        for (let i = 0; i < cols; i++) {
          const t0 = Math.floor(this.x0 + i / cols * span);
          const t1 = Math.max(t0 + 1, Math.floor(this.x0 + (i + 1) / cols * span));
          let occ = 0, rdy = 0;
          for (let t = t0; t < t1 && t < m.cycles; t++) { occ = Math.max(occ, m.cyc.focc[t]); rdy = Math.max(rdy, m.cyc.fready[t]); }
          const x = i / cols * g.W, w = Math.max(1, g.W / cols);
          ctx.globalAlpha = 0.35; ctx.fillRect(x, y + h - h * occ / depth, w, h * occ / depth);
          ctx.globalAlpha = 0.9; ctx.fillRect(x, y + h - h * rdy / depth, w, h * rdy / depth);
        }
        ctx.globalAlpha = 1;
        for (const p of m.prefetches) if (visible(p.cycle, p.cycle + 1)) { ctx.fillStyle = css('--accent'); ctx.fillRect(X(p.cycle), y, Math.max(1, pxPerCycle * 0.6), 4); }
      }
      // UB ports
      {
        const y = rowY(6), h = rowHh, banks = m.cfg.ub_banks;
        const cols = Math.min(g.W, Math.ceil(span));
        for (let i = 0; i < cols; i++) {
          const t0 = Math.floor(this.x0 + i / cols * span);
          let rr = 0, ww = 0;
          for (let t = t0; t < Math.max(t0 + 1, Math.floor(this.x0 + (i + 1) / cols * span)) && t < m.cycles; t++) { rr = Math.max(rr, m.cyc.ubr[t]); ww = Math.max(ww, m.cyc.ubw[t]); }
          const x = i / cols * g.W, w = Math.max(1, g.W / cols);
          ctx.fillStyle = css('--ok'); ctx.fillRect(x, y + h / 2 - h / 2 * rr / banks, w, h / 2 * rr / banks);
          ctx.fillStyle = css('--staged'); ctx.fillRect(x, y + h / 2, w, h / 2 * ww / banks);
        }
        ctx.fillStyle = muted; ctx.font = '9px ' + css('--mono'); ctx.fillText('r', 2, y + 7); ctx.fillText('w', 2, y + h);
      }
      // ACC banks lanes
      {
        const y = rowY(7), h = rowHh, nb = m.cfg.acc_banks, lane = h / nb;
        for (const iss of m.issues) {
          const p = m.program[iss.pc];
          if (!p.res) continue;
          if (!visible(iss.cycle, iss.done)) continue;
          if (p.res.acc_write !== null) { ctx.fillStyle = css('--u-mxu'); ctx.fillRect(X(iss.cycle), y + p.res.acc_write * lane, XW(iss.cycle, iss.done), lane - 1); }
          if (p.res.acc_read !== null && p.op === 'Activate') { ctx.fillStyle = css('--u-act'); ctx.fillRect(X(iss.cycle), y + p.res.acc_read * lane, XW(iss.cycle, iss.done), lane - 1); }
        }
        ctx.fillStyle = muted; ctx.font = '9px ' + css('--mono');
        for (let b = 0; b < nb; b++) ctx.fillText(String(b), 2, y + b * lane + lane - 1);
      }
      // Idle cause
      for (const run of m.idleRuns) {
        if (!visible(run.from, run.to + 1)) continue;
        const name = MTV.IDLE_NAMES[run.bucket];
        const col = IDLE_COLORS[name];
        ctx.fillStyle = col || css('--ok');
        ctx.globalAlpha = col ? 0.8 : 0.35;
        ctx.fillRect(X(run.from), rowY(8), XW(run.from, run.to + 1), rowHh);
      }
      ctx.globalAlpha = 1;
      // Tile boundary markers
      ctx.fillStyle = css('--accent');
      for (const b of m.tileBoundaries()) if (visible(b.cycle, b.cycle + 1)) ctx.fillRect(Math.round(this.cx(b.cycle, g)), 12, 1, 4);
      // Cursor
      if (st) {
        const x = this.cx(st.t + 0.5, g);
        ctx.fillStyle = css('--accent'); ctx.fillRect(Math.round(x) - 1, 0, 2, g.H);
        ctx.font = '10px ' + css('--mono'); ctx.fillStyle = css('--accent');
        const label = 't = ' + st.t;
        ctx.fillText(label, Math.min(g.W - 60, x + 4), g.H - 4);
      }
      ctx.fillStyle = ink;
    },
  };

  function niceStep(raw) {
    const p = Math.pow(10, Math.floor(Math.log10(Math.max(1, raw))));
    for (const m of [1, 2, 5, 10]) if (m * p >= raw) return m * p;
    return 10 * p;
  }

  MTV.views.timeline = view;
})();
