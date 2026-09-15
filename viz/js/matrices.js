// Logical matrices A (M×K), B (K×N) and C (M×N) for a dense layer, with the
// tiler's tile grid, the tiles in use at the current cycle, and for C the
// progress state of every element derived from the trace's issue/retire
// records: untouched → in the PE array → landed in the staged rows →
// committed partial (j of kt K-tiles) → complete int32 in a bank → requantized
// (staged by Activate) → int8 in the UB staging tile → captured by Write_Host
// → written to host memory. Click an element to inspect it.
//
// Classic script: defines window.MTV.views.matrices.
(function () {
  'use strict';
  const MTV = (window.MTV = window.MTV || {});
  MTV.views = MTV.views || {};
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const C_STATES = [
    ['untouched', 'rgba(128,128,128,0.15)'],
    ['in PE array (partial in flight)', 'rgba(30,127,110,0.45)'],
    ['landed, staged (not yet in bank)', 'rgba(107,91,181,0.55)'],
    ['committed partial', 'rgba(180,86,42,0.45)'],
    ['complete int32 in bank', 'rgba(180,86,42,0.95)'],
    ['requantized, staged by Activate', 'rgba(107,91,181,0.9)'],
    ['int8 in UB staging', 'rgba(30,127,110,0.9)'],
    ['captured by Write_Host', 'rgba(185,121,26,0.9)'],
    ['written to host', 'rgba(26,34,48,0.9)'],
  ];

  function issuedYet(p, t, phase) { return p.issue !== null && ((phase === 'issue' || phase === 'account') ? p.issue <= t : p.issue < t); }
  function retiredYet(p, t, phase) { return p.retire !== null && (phase === 'start' ? p.retire < t : p.retire <= t); }

  const view = {
    init(app) {
      this.app = app;
      this.cv = { a: document.getElementById('mat-a'), b: document.getElementById('mat-b'), c: document.getElementById('mat-c') };
      for (const [key, cv] of Object.entries(this.cv)) cv.addEventListener('click', (e) => this.onClick(key, e));
      document.getElementById('mat-legend').innerHTML = C_STATES.map(([n, col]) => '<span><span class="sw" style="background:' + col + '"></span>' + n + '</span>').join('') +
        '<span class="muted">tile grid every ' + '' + 'dim; outlines: solid = in use now, dashed = resident / staged</span>';
    },
    reset(model) { this.model = model; this.tileCache = null; },

    onClick(key, e) {
      const m = this.model;
      if (!m.hasLayer) return;
      const cv = this.cv[key];
      const r = cv.getBoundingClientRect();
      const cols = key === 'a' ? m.K : key === 'b' ? m.N : m.N;
      const rows = key === 'a' ? m.M : key === 'b' ? m.K : m.M;
      const j = Math.floor((e.clientX - r.left) / r.width * cols), i = Math.floor((e.clientY - r.top) / r.height * rows);
      if (i < 0 || j < 0 || i >= rows || j >= cols) return;
      if (key === 'a') this.app.select({ type: 'aelem', i, k: j });
      else if (key === 'b') this.app.select({ type: 'belem', k: i, j });
      else this.app.select({ type: 'celem', i, j });
    },

    // Per-tile summary of the K-chain, Activate and Write_Host progress at (t, phase).
    tileInfo(model, st, m, n) {
      const t = st.t, ph = st.phase;
      const chain = model.tilePcs.M.get(m + ',' + n) || [];
      const acts = (model.tilePcs.C.get(m + ',' + n) || []).filter((p) => p.op === 'Activate');
      const writes = (model.tilePcs.C.get(m + ',' + n) || []).filter((p) => p.op === 'Write_Host_Memory');
      const info = { chain, acts, writes, retiredMm: 0, inflight: null, actIssued: null, actRetired: null, wIssued: null, wRetired: null };
      for (const p of chain) {
        if (retiredYet(p, t, ph)) info.retiredMm++;
        else if (issuedYet(p, t, ph)) info.inflight = p;
      }
      for (const p of acts) { if (retiredYet(p, t, ph)) info.actRetired = p; else if (issuedYet(p, t, ph)) info.actIssued = p; }
      for (const p of writes) { if (retiredYet(p, t, ph)) info.wRetired = p; else if (issuedYet(p, t, ph)) info.wIssued = p; }
      return info;
    },

    // State of C[i][j] with its values and locations.
    cState(model, st, e) {
      const dim = model.dim, t = st.t, ph = st.phase;
      const ti = this.tileInfo(model, st, e.m, e.n);
      const r = e.r, c = e.c;
      const bank = ti.chain.length ? ti.chain[0].fields.acc_bank : null;
      const bankVal = bank !== null ? model.img.acc[bank * dim * dim + r * dim + c] : 0;
      const signed = (v) => (v > 127 ? v - 256 : v);
      if (ti.wRetired) return { code: 8, label: C_STATES[8][0], int8: signed(model.img.host[e.host]), where8: 'host ' + MTV.hex(e.host) };
      if (ti.wIssued) {
        const iss = model.issueByPc.get(ti.wIssued.pc);
        const rd = iss && iss.reads.length ? model.reads[iss.reads[0]] : null;
        const v = rd ? model.readI8(rd)[r * dim + c] : null;
        return { code: 7, label: C_STATES[7][0] + ' (in flight to host until ' + ti.wIssued.retire + ')', int8: v, where8: 'DMA snapshot at issue' };
      }
      if (ti.actRetired) {
        const ub = ti.actRetired.fields.ub_addr + r * dim + c;
        return { code: 6, label: C_STATES[6][0], int8: model.img.ub[ub], where8: 'UB ' + MTV.hex(ub) };
      }
      if (ti.actIssued) {
        const a = model.actByPc.get(ti.actIssued.pc);
        const cm = a && a.commit !== null ? model.commits[a.commit] : null;
        const v = cm ? new Int8Array(model.commitBlob.buffer, model.commitBlob.byteOffset + cm.after, cm.len)[r * dim + c] : null;
        return { code: 5, label: C_STATES[5][0] + ' (commits at ' + ti.actIssued.retire + ')', int8: v, where8: 'staged by Activate', int32: bankVal, where: 'bank ' + bank };
      }
      if (ti.inflight) {
        const mm = model.mmByPc.get(ti.inflight.pc);
        const sNow = (ph === 'issue' || ph === 'account') ? t - mm.issue : t - 1 - mm.issue;
        const landStep = r + dim - 1 + c;
        if (sNow >= landStep) {
          const cm = model.commits.find((x) => x.kind === 'acc' && x.pc === mm.pc);
          const v = cm ? model.commitAfterI32(cm)[r * dim + c] : null;
          return { code: 2, label: C_STATES[2][0] + ' (K-tile ' + ti.retiredMm + ' of ' + ti.chain.length + ', commits at ' + mm.retire + ')', int32: v, where: 'staged rows of pc ' + mm.pc };
        }
        return { code: 1, label: C_STATES[1][0] + ' (K-tile ' + ti.retiredMm + ' of ' + ti.chain.length + ', lands at step ' + landStep + ' = cycle ' + (mm.issue + landStep) + ')', int32: bankVal, where: 'bank ' + bank + ' still holds the previous partial' };
      }
      if (ti.chain.length && ti.retiredMm === ti.chain.length) return { code: 4, label: C_STATES[4][0] + ' ' + bank, int32: bankVal, where: 'bank ' + bank + ' [' + r + '][' + c + ']' };
      if (ti.retiredMm > 0) return { code: 3, label: C_STATES[3][0] + ' (' + ti.retiredMm + ' of ' + ti.chain.length + ' K-tiles)', int32: bankVal, where: 'bank ' + bank + ' [' + r + '][' + c + ']' };
      return { code: 0, label: C_STATES[0][0] };
    },

    render() {
      const m = this.model, st = this.app.state, sel = this.app.sel;
      if (!m || !st) return;
      if (!m.hasLayer) {
        for (const cv of Object.values(this.cv)) { const ctx = cv.getContext('2d'); cv.width = 300; cv.height = 40; ctx.fillStyle = css('--muted'); ctx.font = '12px sans-serif'; ctx.fillText('no dense-layer metadata in this trace', 6, 24); }
        return;
      }
      const dim = m.dim;
      const t = st.t, ph = st.phase;
      const mx = st.mxu;
      const inflightMm = mx.mm && mx.step >= 0 ? m.program[mx.mm.pc] : (st.units.MXU ? m.program[st.units.MXU.pc] : null);

      // A
      this.drawValues(this.cv.a, m.M, m.K, (i, k) => m.aElement(i, k).value, dim, (ctx, px) => {
        // resident tiles (Read_Host retired) dashed, tile being multiplied solid
        for (const [key, pcs] of m.tilePcs.A) {
          const [mm_, kk] = key.split(',').map(Number);
          const rh = pcs.filter((p) => p.op === 'Read_Host_Memory' && retiredYet(p, t, ph));
          if (rh.length) this.outline(ctx, px, mm_ * dim, kk * dim, dim, dim, css('--ok'), true);
        }
        if (inflightMm && inflightMm.tile) this.outline(ctx, px, inflightMm.tile.m * dim, inflightMm.tile.k * dim, dim, dim, css('--ok'), false);
        if (sel && sel.type === 'aelem') this.mark(ctx, px, sel.i, sel.k);
      });
      // B
      this.drawValues(this.cv.b, m.K, m.N, (k, j) => m.bElement(k, j).value, dim, (ctx, px) => {
        const pl = st.planes;
        for (let p = 0; p < 2; p++) {
          const w = pl.tiles[p];
          if (!w) continue;
          const tp = m.program[w.pc];
          if (!tp || !tp.tile) continue;
          this.outline(ctx, px, tp.tile.k * dim, tp.tile.n * dim, dim, dim, p === pl.active ? css('--ok') : css('--staged'), p !== pl.active);
        }
        for (const e of st.fifo) {
          const tp = m.program[e.push.for_pc];
          if (tp && tp.tile) this.outline(ctx, px, tp.tile.k * dim, tp.tile.n * dim, dim, dim, css('--accent'), true);
        }
        if (sel && sel.type === 'belem') this.mark(ctx, px, sel.k, sel.j);
      });
      // C
      const cv = this.cv.c;
      const px = this.sizeCanvas(cv, m.M, m.N);
      const ctx = cv.getContext('2d');
      // per-tile info cached per render
      const tiles = new Map();
      for (let i = 0; i < m.M; i++) {
        for (let j = 0; j < m.N; j++) {
          const tm = Math.floor(i / dim), tn = Math.floor(j / dim);
          const key = tm + ',' + tn;
          if (!tiles.has(key)) tiles.set(key, this.tileInfo(m, st, tm, tn));
          const ti = tiles.get(key);
          let code = 0;
          if (ti.wRetired) code = 8; else if (ti.wIssued) code = 7; else if (ti.actRetired) code = 6; else if (ti.actIssued) code = 5;
          else if (ti.inflight) {
            const mm = m.mmByPc.get(ti.inflight.pc);
            const sNow = (ph === 'issue' || ph === 'account') ? t - mm.issue : t - 1 - mm.issue;
            code = sNow >= (i % dim) + dim - 1 + (j % dim) ? 2 : 1;
          } else if (ti.chain.length && ti.retiredMm === ti.chain.length) code = 4;
          else if (ti.retiredMm > 0) code = 3;
          ctx.fillStyle = C_STATES[code][1];
          ctx.fillRect(j * px, i * px, px, px);
        }
      }
      this.grid(ctx, px, m.M, m.N, dim);
      if (inflightMm && inflightMm.tile && inflightMm.tile.kind === 'M') this.outline(ctx, px, inflightMm.tile.m * dim, inflightMm.tile.n * dim, dim, dim, css('--ok'), false);
      if (sel && sel.type === 'celem') this.mark(ctx, px, sel.i, sel.j);
    },

    sizeCanvas(cv, rows, cols) {
      const px = Math.max(2, Math.min(6, Math.floor(360 / Math.max(rows, cols))));
      cv.width = cols * px; cv.height = rows * px;
      cv.style.aspectRatio = cols + ' / ' + rows;
      return px;
    },
    drawValues(cv, rows, cols, get, dim, overlay) {
      const px = this.sizeCanvas(cv, rows, cols);
      const ctx = cv.getContext('2d');
      for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
        const v = get(i, j);
        const a = 0.1 + 0.9 * Math.abs(v) / 128;
        ctx.fillStyle = v >= 0 ? 'rgba(44,111,176,' + a + ')' : 'rgba(180,86,42,' + a + ')';
        ctx.fillRect(j * px, i * px, px, px);
      }
      this.grid(ctx, px, rows, cols, dim);
      overlay(ctx, px);
    },
    grid(ctx, px, rows, cols, dim) {
      ctx.strokeStyle = 'rgba(255,255,255,0.7)'; ctx.lineWidth = 1;
      for (let i = dim; i < rows; i += dim) { ctx.beginPath(); ctx.moveTo(0, i * px); ctx.lineTo(cols * px, i * px); ctx.stroke(); }
      for (let j = dim; j < cols; j += dim) { ctx.beginPath(); ctx.moveTo(j * px, 0); ctx.lineTo(j * px, rows * px); ctx.stroke(); }
    },
    outline(ctx, px, r0, c0, rows, cols, color, dashed) {
      ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 2; if (dashed) ctx.setLineDash([4, 3]);
      ctx.strokeRect(c0 * px + 1, r0 * px + 1, cols * px - 2, rows * px - 2); ctx.restore();
    },
    mark(ctx, px, i, j) {
      ctx.save(); ctx.strokeStyle = css('--sel'); ctx.lineWidth = 2; ctx.strokeRect(j * px - 1, i * px - 1, px + 2, px + 2); ctx.restore();
    },
  };

  MTV.views.matrices = view;
})();
