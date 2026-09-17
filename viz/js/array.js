// The systolic array view: the MatMul drawn inside the PE grid, one cell per PE,
// from the recorded registers after each step (mm.<i>.left/act/psum/land in the
// container). Reads left → grid → bottom → staged rows:
//
//   [A tile]  [in lane] [ dim×dim PE grid ]         [B plane chip]
//                       [landing lane] ──────────►  [staged output rows]
//
// Derived rules, labelled on screen and checked by viz/selftest.js:
//   role     r = s − k − c, useful when 0 ≤ r < len
//   input    PE row k receives A[s − k][k] at step s (left[s*dim + k])
//   landing  row r leaves column c at step s = r + dim − 1 + c (land[r*dim + c])
(function () {
  'use strict';
  const MTV = (window.MTV = window.MTV || {});
  MTV.views = MTV.views || {};
  const { fmt, hex, tileLabel, esc } = MTV;

  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  function rgb(s) {
    const m = /^#([0-9a-f]{6})$/i.exec(s);
    if (!m) return [128, 128, 128];
    const v = parseInt(m[1], 16);
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
  }
  const rgba = (c, a) => 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')';
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const TEXT_CELL = 26;                                   // smallest cell that shows a value
  const fontFor = (cell) => clamp(cell / 3, 8, 12);       // four int8 characters fit the cell

  const view = {
    init(app) {
      this.app = app;
      this.canvas = document.getElementById('array-canvas');
      this.ctx = this.canvas.getContext('2d');
      this.over = document.getElementById('array-overview');
      this.octx = this.over.getContext('2d');
      this.note = document.getElementById('array-note');
      this.tip = document.getElementById('array-tip');
      this.mode = 'flow';
      this.zoom = 1;
      this.pan = { x: 0, y: 0 };
      this.frame = null;
      this.frameFor = -1;
      this.loading = -1;
      this.hover = null;
      this.cache = new Map();   // mm index -> {A, rd, maxLand, wMax}
      this.off = {};            // offscreen canvases by name
      this._pal = null;
      document.getElementById('array-mode').addEventListener('click', (e) => {
        const b = e.target.closest('button'); if (!b) return;
        this.mode = b.getAttribute('data-mode');
        for (const x of e.currentTarget.querySelectorAll('button')) x.classList.toggle('active', x === b);
        this.render();
      });
      document.getElementById('array-zoom').addEventListener('input', (e) => { this.zoom = Math.pow(2, e.target.value / 25); this.render(); });
      document.getElementById('array-fit').addEventListener('click', () => this.setZoom(1));
      document.getElementById('array-text').addEventListener('click', () => {
        // Zoom until a cell is 30 px: the first size where int8 text fits comfortably.
        const g = this.geometry();
        this.setZoom(clamp(30 / (g.cell / this.zoom), 1, 16));
      });
      this.canvas.addEventListener('click', (e) => this.onClick(e));
      this.canvas.addEventListener('mousemove', (e) => this.onMove(e));
      this.canvas.addEventListener('mouseleave', () => { if (this.hover) { this.hover = null; this.tip.hidden = true; this.render(); } });
      this.canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const f = Math.exp(-e.deltaY * 0.002);
        this.setZoom(clamp(this.zoom * f, 1, 16));
      }, { passive: false });
      let drag = null;
      this.canvas.addEventListener('mousedown', (e) => { drag = { x: e.clientX, y: e.clientY, px: this.pan.x, py: this.pan.y, moved: false }; });
      window.addEventListener('mousemove', (e) => {
        if (!drag) return;
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
        if (drag.moved) { this.pan.x = drag.px + dx; this.pan.y = drag.py + dy; this.render(); }
      });
      window.addEventListener('mouseup', () => { drag = null; });
      this.over.addEventListener('mousedown', (e) => { this.overDrag = true; this.overPan(e); });
      this.over.addEventListener('mousemove', (e) => { if (this.overDrag) this.overPan(e); });
      window.addEventListener('mouseup', () => { this.overDrag = false; });
      window.addEventListener('resize', () => { if (this.app.view === 'array') this.render(); });
      this.note.addEventListener('click', (e) => { const a = e.target.closest('[data-action]'); if (a) { e.preventDefault(); this.app.action(a.getAttribute('data-action')); } });
      if (window.matchMedia) {
        const mq = window.matchMedia('(prefers-color-scheme: dark)');
        const onTheme = () => { this._pal = null; if (this.app.view === 'array') this.render(); };
        if (mq.addEventListener) mq.addEventListener('change', onTheme); else if (mq.addListener) mq.addListener(onTheme);
      }
    },
    reset(model) { this.model = model; this.frame = null; this.frameFor = -1; this.loading = -1; this.hover = null; this.cache.clear(); this._pal = null; this.tip.hidden = true; },
    setZoom(z) {
      this.zoom = z;
      if (z <= 1) this.pan = { x: 0, y: 0 };
      document.getElementById('array-zoom').value = String(Math.round(Math.log2(z) * 25));
      this.render();
    },

    // Theme tokens, read once per theme (not per render).
    pal() {
      if (this._pal) return this._pal;
      const p = {};
      for (const k of ['surface', 'surface-2', 'surface-3', 'ink', 'muted', 'line', 'line-strong', 'ok', 'bad', 'staged', 'accent', 'accent-ink', 'u-act', 'sel', 'warn']) p[k] = cssVar('--' + k);
      p.mono = cssVar('--mono'); p.sans = cssVar('--sans'); p.head = cssVar('--head');
      p.c = {};
      for (const k of ['surface', 'surface-2', 'surface-3', 'ok', 'bad', 'staged', 'accent', 'u-act', 'sel', 'muted', 'ink']) p.c[k] = rgb(p[k]);
      this._pal = p;
      return p;
    },

    geometry() {
      const rect = this.canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const W = Math.max(10, Math.floor(rect.width)), H = Math.max(10, Math.floor(rect.height));
      if (this.canvas.width !== W * dpr || this.canvas.height !== H * dpr) { this.canvas.width = W * dpr; this.canvas.height = H * dpr; }
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const dim = this.model.dim;
      const T = clamp(3 * dim, 64, 128);          // thumb / chip / strip size in px
      const colW = Math.max(T, 150);              // side columns: image plus two caption lines
      const top = 46;
      const leftW = 8 + colW + 8;
      const rightW = colW + 24;
      const availW = W - leftW - rightW - 8, availH = H - top - 10;
      const base = Math.max(3, Math.min(availW / (dim + 1), availH / (dim + 1)));
      const cell = base * this.zoom;
      const gridW = cell * dim;
      // keep the grid within reach of the clip window
      const minX = Math.min(0, availW - (gridW + cell)), minY = Math.min(0, availH - (gridW + cell));
      this.pan.x = clamp(this.pan.x, minX, 0);
      this.pan.y = clamp(this.pan.y, minY, 0);
      const ox = leftW + cell + this.pan.x, oy = top + this.pan.y;
      const overview = this.zoom > 1.001;
      // the right column sits just past the grid at fit, at the pane's edge when zoomed
      const rx = Math.min(W - rightW + 12, leftW + cell + gridW + 26);
      const chipY = overview ? 10 + 132 + 14 : top;
      // with the overview above them, the chip and the staged strip shrink to fit the height
      const Tr = overview ? clamp(Math.floor((H - chipY - 96) / 2), 40, T) : T;
      const stagedY = chipY + Tr + 50;
      return { W, H, dim, T, Tr, top, leftW, rightW, cell, gridW, ox, oy, clip: { x: leftW, y: top, w: Math.max(0, rx - 12 - leftW), h: H - top }, rx, chipY, stagedY, overview, thumb: { x: 8, y: top, size: T } };
    },

    // True when the PE frames are in hand; otherwise loads and re-renders.
    ensureFrame() {
      const st = this.app.state;
      const mx = st.mxu;
      if (!mx.mm) { this.frame = null; this.frameFor = -1; return true; }
      const idx = mx.mm.index;
      const c = this.model.c;
      // From a File the frames are sliced asynchronously: warm the next MatMul near the end of this one.
      if (!c.sliceSync && mx.step > mx.mm.steps - 20 && idx + 1 < this.model.matmuls.length && !c.mmLoaded(idx + 1) && this.prefetching !== idx + 1) {
        this.prefetching = idx + 1;
        c.mm(idx + 1).catch(() => {}).then(() => { if (this.prefetching === idx + 1) this.prefetching = -1; });
      }
      if (this.frameFor === idx && this.frame) return true;
      const sync = c.mmSync(idx);
      if (sync) { this.frame = sync; this.frameFor = idx; return true; }
      if (this.loading !== idx) {
        this.loading = idx;
        c.mm(idx).then((f) => {
          this.frame = f; this.frameFor = idx; this.loading = -1;
          if (this.app.view === 'array') this.render();
          if (this.app.sel && this.app.sel.type === 'pe') this.app.render();
        }).catch((err) => { console.error(err); this.loading = -1; });
      }
      return false;
    },
    // The frames for a given MatMul if already loaded (used by the inspector).
    frameOf(idx) {
      if (this.frameFor === idx && this.frame) return this.frame;
      const sync = this.model && this.model.c.mmSync(idx);
      if (sync) return sync;
      if (this.model && this.model.c.mmLoaded(idx)) return this.model.c.mmCache.get(idx);
      this.ensureFrame();
      return null;
    },

    // Per-MatMul facts that do not change with the step: the A tile read at
    // issue, the largest landed value, the largest weight.
    info(mm, f) {
      let x = this.cache.get(mm.index);
      if (x) return x;
      const m = this.model, dim = m.dim;
      const iss = m.issueByPc.get(mm.pc);
      const rd = iss ? iss.reads.map((i) => m.reads[i]).find((r) => r.kind === 'ub') : null;
      const A = rd ? m.readI8(rd) : null;
      let maxLand = 1, wMax = 1;
      for (let i = 0; i < f.land.length; i++) maxLand = Math.max(maxLand, Math.abs(f.land[i]));
      for (let i = 0; i < f.w.length; i++) wMax = Math.max(wMax, Math.abs(f.w[i]));
      const bankRead = mm.acc_in_read !== null && mm.acc_in_read !== undefined ? m.reads[mm.acc_in_read] : null;
      x = { A, rd, maxLand, wMax, bankRead, aRows: rd ? rd.rows : mm.len };
      this.cache.set(mm.index, x);
      return x;
    },
    maxPsumAt(f, s) {
      if (this.psumFor && this.psumFor.f === f && this.psumFor.s === s) return this.psumFor.v;
      const n = this.model.dim * this.model.dim;
      let v = 1;
      const ps = f.psum.subarray(s * n, (s + 1) * n);
      for (let i = 0; i < n; i++) v = Math.max(v, Math.abs(ps[i]));
      this.psumFor = { f, s, v };
      return v;
    },

    // ---- offscreen colour maps ----
    offCanvas(name, w, h) {
      let o = this.off[name];
      if (!o || o.w !== w || o.h !== h) {
        const cv = document.createElement('canvas');
        cv.width = w; cv.height = h;
        const ctx = cv.getContext('2d');
        o = this.off[name] = { cv, ctx, w, h, img: ctx && ctx.createImageData ? ctx.createImageData(w, h) : null };
      }
      return o;
    },
    // paint(w, h, fn) where fn(x, y, out) writes out[0..2] = rgb.
    paint(name, w, h, fn) {
      const o = this.offCanvas(name, w, h);
      if (!o.img || !o.ctx) return null;
      const d = o.img.data;
      const px = [0, 0, 0];
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          fn(x, y, px);
          const i = (y * w + x) * 4;
          d[i] = px[0]; d[i + 1] = px[1]; d[i + 2] = px[2]; d[i + 3] = 255;
        }
      }
      o.ctx.putImageData(o.img, 0, 0);
      return o.cv;
    },
    blit(cv, x, y, w, h) {
      if (!cv) return;
      const ctx = this.ctx;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(cv, x, y, w, h);
      ctx.imageSmoothingEnabled = true;
    },

    // ---- hit testing ----
    cellAt(g, x, y) {
      const dim = g.dim;
      if (x < g.clip.x || y < g.clip.y || x > g.clip.x + g.clip.w || y > g.clip.y + g.clip.h) return null;
      const c = Math.floor((x - g.ox) / g.cell), k = Math.floor((y - g.oy) / g.cell);
      if (k < 0 || c < 0 || k >= dim || c >= dim) return null;
      return { k, c };
    },
    onClick(e) {
      const g = this.geometry();
      const x = e.offsetX, y = e.offsetY;
      const m = this.model, st = this.app.state, dim = g.dim;
      const cell = this.cellAt(g, x, y);
      if (cell) { this.app.select({ type: 'pe', k: cell.k, c: cell.c, frame: this.frame }); return; }
      const h = this.hits || {};
      const inside = (r) => r && x >= r.x && y >= r.y && x < r.x + r.w && y < r.y + r.h;
      const mm = st.mxu.mm;
      if (inside(h.thumb) && mm) {
        const p = m.program[mm.pc];
        const i = Math.floor((y - h.thumb.y) / h.thumb.cw), k = Math.floor((x - h.thumb.x) / h.thumb.cw);
        if (m.hasLayer && p.tile) this.app.select({ type: 'aelem', i: p.tile.m * dim + i, k: p.tile.k * dim + k });
        else if (h.thumb.rd) { this.app.setView('memory'); MTV.views.memory.show('ub'); }
        return;
      }
      if (inside(h.staged) && mm) {
        const p = m.program[mm.pc];
        const r = Math.floor((y - h.staged.y) / h.staged.cw), c = Math.floor((x - h.staged.x) / h.staged.cw);
        if (m.hasLayer && p.tile) this.app.select({ type: 'celem', i: p.tile.m * dim + r, j: p.tile.n * dim + c });
        else this.app.select({ type: 'bank', bank: mm.bank });
        return;
      }
      if (inside(h.chip)) { this.app.select({ type: 'planes' }); }
    },
    onMove(e) {
      const g = this.geometry();
      const cell = this.cellAt(g, e.offsetX, e.offsetY);
      const same = (cell === null && this.hover === null) || (cell && this.hover && cell.k === this.hover.k && cell.c === this.hover.c);
      if (!same) { this.hover = cell; this.render(); }
      if (cell) this.showTip(cell, e.clientX, e.clientY); else this.tip.hidden = true;
    },
    showTip(cell, cx, cy) {
      const m = this.model, st = this.app.state, dim = m.dim;
      const mx = st.mxu;
      const f = mx.mm && this.frameFor === mx.mm.index ? this.frame : null;
      const { k, c } = cell;
      let html = '<div class="tip-head">PE[' + k + '][' + c + ']</div>';
      if (f && mx.step >= 0) {
        const v = MTV.peValues(m, mx.mm, f, mx.step, k, c);
        html += '<div class="tip-line">' + (v.live ? 'useful · A row ' + v.r + (v.coords ? ' → C[' + v.coords.i + '][' + v.coords.j + ']' : '') : v.r < 0 ? 'fill: no row here yet' : 'drain: rows are past') + ' <span class="muted">(r = s − k − c)</span></div>';
        html += '<div class="tip-line mono">w ' + v.w + (v.coords ? ' = B[' + v.coords.kk + '][' + v.coords.j + ']' : '') + ' · act_in ' + v.actIn + (v.coords ? ' = A[' + v.coords.i + '][' + v.coords.kk + ']' : '') + '</div>';
        html += '<div class="arith">' + fmt(v.psumIn) + ' + (' + v.actIn + ' × ' + v.w + ') → ' + fmt(v.psumOut) + (v.ok ? '' : ' ✗') + '</div>';
        html += '<div class="tip-line mono">act → ' + (c === dim - 1 ? 'right edge' : 'PE[' + k + '][' + (c + 1) + ']') + ' · psum → ' + (k === dim - 1 ? 'bottom edge' : 'PE[' + (k + 1) + '][' + c + ']') + ' next step</div>';
        if (v.landed !== null) html += '<div class="tip-line">lands: staged row ' + v.r + ', column ' + c + ' (' + fmt(v.landed) + ')</div>';
      } else {
        html += '<div class="tip-line">no MatMul step: registers hold 0</div>';
        html += '<div class="tip-line mono">w ' + m.img.planes[st.planes.active][k * dim + c] + ' (plane ' + st.planes.active + ', resident)</div>';
      }
      html += '<div class="tip-hint">click to open this PE in the inspector</div>';
      this.tip.innerHTML = html;
      this.tip.hidden = false;
      const vw = window.innerWidth, vh = window.innerHeight;
      const tw = this.tip.offsetWidth || 260, th = this.tip.offsetHeight || 120;
      this.tip.style.left = Math.min(cx + 14, vw - tw - 8) + 'px';
      this.tip.style.top = (cy + 16 + th > vh ? cy - th - 10 : cy + 16) + 'px';
    },
    overPan(e) {
      const g = this.geometry();
      const r = this.over.getBoundingClientRect();
      const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
      this.pan.x = -(fx * g.gridW - g.clip.w / 2);
      this.pan.y = -(fy * g.gridW - g.clip.h / 2);
      this.render();
    },

    // ---- render ----
    render() {
      const m = this.model, st = this.app.state;
      if (!m || !st) return;
      const ready = this.ensureFrame();
      const g = this.geometry();
      const ctx = this.ctx;
      const P = this.pal();
      const dim = g.dim, n = dim * dim;
      const mx = st.mxu;
      const f = mx.mm && this.frameFor === mx.mm.index ? this.frame : null;
      const s = mx.step;
      const live = !!(f && s >= 0);
      const mm = mx.mm;
      const coarse = this.app.playing && this.app.speed >= 64;
      ctx.clearRect(0, 0, g.W, g.H);
      this.hits = {};

      const info = live ? this.info(mm, f) : null;
      const maxPsum = live ? this.maxPsumAt(f, s) : 1;
      const planeW = m.img.planes[st.planes.active];
      let wMaxIdle = 1;
      if (!live) for (let i = 0; i < n; i++) wMaxIdle = Math.max(wMaxIdle, Math.abs(planeW[i]));

      // --- grid colour map (one image, scaled) ---
      const C = P.c;
      const mode = this.mode;
      const mix = (out, col, a) => { out[0] += (col[0] - out[0]) * a; out[1] += (col[1] - out[1]) * a; out[2] += (col[2] - out[2]) * a; };
      const base = C['surface-2'];
      const gridImg = this.paint('grid', dim, dim, (c, k, out) => {
        out[0] = base[0]; out[1] = base[1]; out[2] = base[2];
        if (!live) {
          const w = planeW[k * dim + c];
          if (w !== 0) mix(out, w > 0 ? C['u-act'] : C.accent, 0.08 + 0.3 * Math.abs(w) / wMaxIdle);
          return;
        }
        const r = s - k - c;
        const useful = r >= 0 && r < mm.len;
        const i = s * n + k * dim + c;
        const act = f.act[i], psum = f.psum[i];
        if (mode === 'flow') {
          mix(out, useful ? C.ok : r < 0 ? C.accent : C.staged, useful ? 0.2 : 0.1);
          if (act !== 0) mix(out, act > 0 ? C['u-act'] : C.accent, 0.2 + 0.7 * Math.abs(act) / 127);
        } else if (mode === 'act') {
          if (act !== 0) mix(out, act > 0 ? C['u-act'] : C.accent, 0.12 + 0.85 * Math.abs(act) / 127);
        } else if (mode === 'psum') {
          if (useful) mix(out, psum >= 0 ? C.ok : C.bad, 0.12 + 0.85 * Math.abs(psum) / maxPsum);
        } else if (mode === 'w') {
          const w = f.w[k * dim + c];
          if (w !== 0) mix(out, w > 0 ? C['u-act'] : C.accent, 0.12 + 0.85 * Math.abs(w) / info.wMax);
        } else {
          mix(out, useful ? C.ok : r < 0 ? C.accent : C.staged, useful ? 0.9 : r < 0 ? 0.28 : 0.35);
        }
      });

      ctx.save();
      ctx.beginPath(); ctx.rect(g.clip.x, g.clip.y, g.clip.w, g.clip.h); ctx.clip();
      this.blit(gridImg, g.ox, g.oy, g.gridW, g.gridW);
      // grid lines
      if (g.cell >= 6) {
        ctx.strokeStyle = rgba(C.surface, 0.9); ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i <= dim; i++) { const x = Math.round(g.ox + i * g.cell) + 0.5, y = Math.round(g.oy + i * g.cell) + 0.5; ctx.moveTo(x, g.oy); ctx.lineTo(x, g.oy + g.gridW); ctx.moveTo(g.ox, y); ctx.lineTo(g.ox + g.gridW, y); }
        ctx.stroke();
      }
      // visible cell range
      const c0 = Math.max(0, Math.floor((g.clip.x - g.ox) / g.cell)), c1 = Math.min(dim - 1, Math.ceil((g.clip.x + g.clip.w - g.ox) / g.cell));
      const k0 = Math.max(0, Math.floor((g.clip.y - g.oy) / g.cell)), k1 = Math.min(dim - 1, Math.ceil((g.clip.y + g.clip.h - g.oy) / g.cell));

      if (live) {
        // psum drop bars (flow mode): the sum growing toward the bottom edge
        if (mode === 'flow' && g.cell >= 12 && !coarse) {
          const bw = Math.max(2, Math.round(g.cell * 0.14));
          for (let k = k0; k <= k1; k++) for (let c = c0; c <= c1; c++) {
            const r = s - k - c;
            if (r < 0 || r >= mm.len) continue;
            const psum = f.psum[s * n + k * dim + c];
            if (psum === 0) continue;
            const h = Math.max(1, (g.cell - 4) * Math.abs(psum) / maxPsum);
            ctx.fillStyle = psum > 0 ? P.ok : P.bad;
            ctx.fillRect(g.ox + (c + 1) * g.cell - bw - 2, g.oy + k * g.cell + 2, bw, h);
          }
        }
        // text
        if (g.cell >= TEXT_CELL && !coarse) {
          const fs = fontFor(g.cell);
          ctx.font = fs + 'px ' + P.mono;
          for (let k = k0; k <= k1; k++) for (let c = c0; c <= c1; c++) {
            const x = g.ox + c * g.cell, y = g.oy + k * g.cell;
            const i = s * n + k * dim + c;
            ctx.textAlign = 'left'; ctx.fillStyle = P.ink;
            ctx.fillText(String(f.act[i]), x + 3, y + fs + 2);
            if (g.cell >= 44) {
              ctx.textAlign = 'right'; ctx.fillStyle = P.muted;
              ctx.fillText('w' + f.w[k * dim + c], x + g.cell - 4, y + fs + 2);
              ctx.fillStyle = P.ink;
              ctx.fillText(String(f.psum[i]), x + g.cell - 4, y + g.cell - 5);
            }
          }
          ctx.textAlign = 'left';
        }
        // wavefront: leading edge r = 0 (k + c = s), trailing edge r = len − 1, and their ghosts from step s − 1
        this.diag(g, s - 1, P.ok, true, 0.4, null);
        this.diag(g, s - mm.len, P.staged, true, 0.4, null);
        this.diag(g, s, P.ok, false, 1, s < mm.len ? 'row 0 of A' : 'front');
        this.diag(g, s - mm.len + 1, P.staged, false, 1, 'row ' + (mm.len - 1) + ' (last)');
      } else if (g.cell >= TEXT_CELL && !coarse) {
        const fs = fontFor(g.cell);
        ctx.font = fs + 'px ' + P.mono; ctx.fillStyle = P.muted;
        for (let k = k0; k <= k1; k++) for (let c = c0; c <= c1; c++) ctx.fillText('w' + planeW[k * dim + c], g.ox + c * g.cell + 3, g.oy + k * g.cell + fs + 2);
      }
      // hover / selection: the whole anti-diagonal shares one A row this step
      const sel = this.app.sel;
      const focus = this.hover || (sel && sel.type === 'pe' ? { k: sel.k, c: sel.c } : null);
      if (focus && live) {
        const sum = focus.k + focus.c;
        ctx.strokeStyle = rgba(C.sel, 0.55); ctx.lineWidth = 1.5;
        for (let k = Math.max(0, sum - dim + 1); k <= Math.min(dim - 1, sum); k++) ctx.strokeRect(g.ox + (sum - k) * g.cell + 1, g.oy + k * g.cell + 1, g.cell - 2, g.cell - 2);
      }
      if (sel && sel.type === 'pe') { ctx.strokeStyle = P.sel; ctx.lineWidth = 2.5; ctx.strokeRect(g.ox + sel.c * g.cell + 1, g.oy + sel.k * g.cell + 1, g.cell - 2, g.cell - 2); }
      if (this.hover) { ctx.strokeStyle = P.sel; ctx.lineWidth = 1.5; ctx.strokeRect(g.ox + this.hover.c * g.cell + 1, g.oy + this.hover.k * g.cell + 1, g.cell - 2, g.cell - 2); }

      // lanes: left (inputs this step) and bottom (landings this step)
      this.drawLanes(g, f, s, mm, live, coarse, info);
      ctx.restore();

      // idle banner over the grid
      if (!live) this.drawIdle(g, st, mm, ready);

      // margins: header, A thumb, B chip, staged rows
      this.drawHeader(g, st, live, mm, s);
      this.drawThumb(g, st, f, s, mm, live, info, focus);
      this.drawChip(g, st, f, mm, live, info, planeW, wMaxIdle);
      this.drawStaged(g, st, f, s, mm, live, info, focus);

      // toolbar note
      if (live) {
        const pm = m.program[mm.pc];
        this.note.innerHTML = '<a data-action="instr:' + mm.pc + '">pc ' + mm.pc + '</a> ' + esc(tileLabel(pm) || hex(pm.fields.ub_addr)) + ' · step ' + s + '/' + mm.steps + ' · plane ' + mm.plane + ' · ' + (s < mm.len ? 'streaming' : 'draining') + (ready === true ? '' : ' · loading PE detail…') + (coarse ? ' · coarse while playing' : '');
      } else if (mm) {
        this.note.innerHTML = '<a data-action="instr:' + mm.pc + '">pc ' + mm.pc + '</a> · loading PE detail…';
      } else {
        const next = m.unitIv.MXU.find((iv) => iv.issue >= st.t);
        this.note.innerHTML = 'array idle' + (next ? (next.issue === st.t ? ' · <a data-action="instr:' + next.pc + '">pc ' + next.pc + '</a> issues this cycle (phase issue)' : ' · <a data-action="cycle:' + next.issue + '">next MatMul pc ' + next.pc + ' at ' + fmt(next.issue) + '</a>') : ' · no more MatMuls');
      }
      // overview only while zoomed
      this.over.hidden = !g.overview;
      if (g.overview) this.drawOverview(g, gridImg);
    },

    // One anti-diagonal k + c = sum drawn as a line through the cell centres.
    diag(g, sum, color, dashed, alpha, label) {
      const dim = g.dim;
      if (sum < 0 || sum > 2 * dim - 2) return;
      const ctx = this.ctx;
      const kA = Math.max(0, sum - dim + 1), cA = sum - kA;     // upper-right end
      const kB = Math.min(dim - 1, sum), cB = sum - kB;         // lower-left end
      const xA = g.ox + (cA + 0.5) * g.cell, yA = g.oy + (kA + 0.5) * g.cell;
      const xB = g.ox + (cB + 0.5) * g.cell, yB = g.oy + (kB + 0.5) * g.cell;
      const ext = g.cell * 0.5;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color; ctx.lineWidth = dashed ? 1.5 : 2.5;
      if (dashed) ctx.setLineDash([4, 4]);
      ctx.beginPath();
      if (sum === 0) { ctx.moveTo(xA - ext, yA + ext); ctx.lineTo(xA + ext, yA - ext); }
      else { ctx.moveTo(xA + ext * 0.7, yA - ext * 0.7); ctx.lineTo(xB - ext * 0.7, yB + ext * 0.7); }
      ctx.stroke();
      if (label && g.cell >= 8) {
        ctx.setLineDash([]);
        ctx.font = '600 10px ' + this.pal().head; ctx.fillStyle = color;
        if (sum === 0 || cA >= dim - 1 && kA > 0) { ctx.textAlign = 'right'; ctx.fillText(label, xA - 6, yA - 5); }
        else { ctx.textAlign = 'left'; ctx.fillText(label, xA + 8, yA - 5); }
        ctx.textAlign = 'left';
      }
      ctx.restore();
    },

    drawLanes(g, f, s, mm, live, coarse, info) {
      const ctx = this.ctx, P = this.pal(), C = P.c, dim = g.dim, n = dim * dim, cell = g.cell;
      const lx = g.ox - cell, ly = g.oy + g.gridW;
      // lane backgrounds
      ctx.fillStyle = rgba(C['surface-3'], 0.5);
      ctx.fillRect(lx, g.oy, cell - 1, g.gridW);
      ctx.fillRect(g.ox, ly + 1, g.gridW, cell - 1);
      if (!live) return;
      const fs = fontFor(cell);
      // left lane: PE row k receives A[s − k][k]
      for (let k = 0; k < dim; k++) {
        const y = g.oy + k * cell;
        if (y + cell < g.clip.y || y > g.clip.y + g.clip.h) continue;
        const v = f.left[s * dim + k];
        const entering = s - k >= 0 && s - k < mm.len;
        if (entering) {
          ctx.fillStyle = v === 0 ? rgba(C.ok, 0.25) : rgba(v > 0 ? C['u-act'] : C.accent, 0.25 + 0.7 * Math.abs(v) / 127);
          ctx.fillRect(lx, y, cell - 1, cell - 1);
          if (cell >= 10) { ctx.fillStyle = P.ok; ctx.beginPath(); ctx.moveTo(lx + cell - 6, y + cell / 2 - 3); ctx.lineTo(lx + cell - 2, y + cell / 2); ctx.lineTo(lx + cell - 6, y + cell / 2 + 3); ctx.closePath(); ctx.fill(); }
          if (cell >= TEXT_CELL && !coarse) { ctx.font = fs + 'px ' + P.mono; ctx.fillStyle = P.ink; ctx.fillText(String(v), lx + 3, y + fs + 2); }
        }
      }
      // bottom lane: row r = s + 1 − dim − c leaves column c
      for (let c = 0; c < dim; c++) {
        const x = g.ox + c * cell;
        if (x + cell < g.clip.x || x > g.clip.x + g.clip.w) continue;
        const r = s + 1 - dim - c;
        if (r < 0 || r >= mm.len) continue;
        const v = f.psum[s * n + (dim - 1) * dim + c];
        ctx.fillStyle = rgba(v >= 0 ? C.ok : C.bad, 0.2 + 0.7 * Math.abs(v) / info.maxLand);
        ctx.fillRect(x, ly + 1, cell - 1, cell - 1);
        if (cell >= 10) { ctx.fillStyle = P.staged; ctx.beginPath(); ctx.moveTo(x + cell / 2 - 3, ly + 2); ctx.lineTo(x + cell / 2, ly + 6); ctx.lineTo(x + cell / 2 + 3, ly + 2); ctx.closePath(); ctx.fill(); }
        if (cell >= 12 && !coarse) { ctx.font = '600 ' + Math.max(7, Math.min(10, cell / 2.6)) + 'px ' + P.mono; ctx.fillStyle = P.staged; ctx.fillText('r' + r, x + 2, ly + cell - 3); }
        if (cell >= 44 && !coarse) { ctx.font = fs + 'px ' + P.mono; ctx.fillStyle = P.ink; ctx.textAlign = 'right'; ctx.fillText(String(v), x + cell - 3, ly + cell - 3); ctx.textAlign = 'left'; }
      }
    },

    drawIdle(g, st, mm, ready) {
      const ctx = this.ctx, P = this.pal(), m = this.model;
      const o = st.outcome;
      const lines = [];
      if (mm) lines.push('loading PE detail of pc ' + mm.pc + '…');
      else {
        const ev = st.events;
        const retiredMm = ev.retires.find((r) => r.unit === 'MXU');
        const busy = st.units.MXU;
        if (busy) { lines.push('pc ' + busy.pc + ' issues this cycle'); lines.push('step 0 is visible after phase issue'); }
        else if (retiredMm && st.phase !== 'start') {
          const c = ev.commits.find((x) => x.pc === retiredMm.pc && x.kind === 'acc');
          lines.push('pc ' + retiredMm.pc + ' retired: ' + (c ? MTV.commitText(m, c) : 'done'));
          lines.push('the grid is empty until the next MatMul issues');
        } else {
          lines.push('array idle · cycle ' + fmt(st.t) + (o.idle !== 'busy' ? ' · charged to ' + o.idle : ''));
          const ib = m.idleBlockerAt(st.t);
          if (ib.pc !== null && ib.via === 'charge') { const x = st.units[ib.unit]; lines.push('pc ' + ib.pc + ' ' + MTV.OP_SHORT[m.program[ib.pc].op] + ' ' + (tileLabel(m.program[ib.pc]) || '') + ' runs until ' + fmt(x ? x.done : 0)); }
          if (o.kind === 'stall' && m.program[o.pc]) lines.push('pc ' + o.pc + ' ' + MTV.OP_SHORT[m.program[o.pc].op] + ' waits: ' + o.reason + (o.blockerPc !== null ? ' ← pc ' + o.blockerPc : ''));
          const next = m.unitIv.MXU.find((iv) => iv.issue > st.t);
          lines.push(next ? 'next MatMul: pc ' + next.pc + ' at ' + fmt(next.issue) + ' (+' + fmt(next.issue - st.t) + ')' : 'no more MatMuls in this run');
        }
        lines.push('PEs hold 0; the resident weights of plane ' + st.planes.active + ' are shown faintly');
      }
      ctx.save();
      ctx.beginPath(); ctx.rect(g.clip.x, g.clip.y, g.clip.w, g.clip.h); ctx.clip();
      ctx.font = '12.5px ' + P.sans;
      let w = 0;
      for (const l of lines) w = Math.max(w, ctx.measureText(l).width);
      const bw = w + 28, bh = lines.length * 18 + 16;
      const cx = g.ox + g.gridW / 2, cy = g.oy + g.gridW / 2;
      const bx = clamp(cx - bw / 2, g.clip.x + 6, g.clip.x + g.clip.w - bw - 6), by = clamp(cy - bh / 2, g.clip.y + 6, g.clip.y + g.clip.h - bh - 6);
      ctx.fillStyle = rgba(P.c.surface, 0.92); ctx.strokeStyle = P['line-strong']; ctx.lineWidth = 1;
      ctx.fillRect(bx, by, bw, bh); ctx.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1);
      lines.forEach((l, i) => { ctx.fillStyle = i === 0 ? P.ink : P.muted; ctx.fillText(l, bx + 14, by + 22 + i * 18); });
      ctx.restore();
    },

    drawHeader(g, st, live, mm, s) {
      const ctx = this.ctx, P = this.pal(), dim = g.dim;
      ctx.font = '10.5px ' + P.mono; ctx.fillStyle = P.muted; ctx.textAlign = 'left';
      ctx.fillText('k↓ c→ · act moves →, psum moves ↓ · PE[k][c] is useful when r = s − k − c ∈ [0, len)', g.leftW, 14);
      if (live) {
        ctx.fillText('in: PE row k gets A[s−k][k] at its left edge (one step later per row) · out: row r = s+1−dim−c leaves column c at the bottom', g.leftW, 30);
      } else {
        ctx.fillText('no MatMul step at this cycle/phase: every PE holds 0 (the grid drains to zero at the end of each MatMul)', g.leftW, 30);
      }
      ctx.font = '600 10px ' + P.head; ctx.fillStyle = P.muted;
      ctx.fillText('IN', g.ox - g.cell, g.top - 4);
      ctx.fillText('A TILE', g.thumb.x, g.top - 4);
    },

    // The A tile as read at issue, with the diagonal entering this step lit and
    // the cells already inside the array dimmed.
    drawThumb(g, st, f, s, mm, live, info, focus) {
      const ctx = this.ctx, P = this.pal(), C = P.c, dim = g.dim, T = g.T;
      const t = g.thumb;
      const cw = T / dim;
      ctx.fillStyle = P['surface-2']; ctx.fillRect(t.x, t.y, T, T);
      let cap1 = 'A tile: —', cap2 = 'read at MatMul issue';
      if (live && info.A) {
        const A = info.A, rows = info.aRows;
        const focusRow = focus ? s - focus.k - focus.c : -1;
        const img = this.paint('thumb', dim, rows, (k, i, out) => {
          const v = A[i * dim + k];
          out[0] = C['surface-2'][0]; out[1] = C['surface-2'][1]; out[2] = C['surface-2'][2];
          if (v !== 0) { const col = v > 0 ? C['u-act'] : C.accent; const a = 0.15 + 0.8 * Math.abs(v) / 127; out[0] += (col[0] - out[0]) * a; out[1] += (col[1] - out[1]) * a; out[2] += (col[2] - out[2]) * a; }
          const age = s - k - i;   // steps since this element entered
          if (age > 0) { const a = 0.6; out[0] += (C.surface[0] - out[0]) * a; out[1] += (C.surface[1] - out[1]) * a; out[2] += (C.surface[2] - out[2]) * a; }
          else if (age === 0) { out[0] = C.ok[0]; out[1] = C.ok[1]; out[2] = C.ok[2]; }
          else if (age === -1) { out[0] += (C.ok[0] - out[0]) * 0.35; out[1] += (C.ok[1] - out[1]) * 0.35; out[2] += (C.ok[2] - out[2]) * 0.35; }
          if (i === focusRow) { out[0] += (C.sel[0] - out[0]) * 0.5; out[1] += (C.sel[1] - out[1]) * 0.5; out[2] += (C.sel[2] - out[2]) * 0.5; }
        });
        this.blit(img, t.x, t.y, T, cw * rows);
        this.hits.thumb = { x: t.x, y: t.y, w: T, h: cw * rows, cw, rd: info.rd };
        const pm = this.model.program[mm.pc];
        cap1 = (tileLabel(pm) ? tileLabel(pm).replace(/^C\((\d+),(\d+)\) K-tile (\d+)$/, 'A($1,$3)') : 'A') + ' · UB ' + hex(info.rd.addr);
        cap2 = 'lit: entering · dim: inside';
      } else if (live) {
        cap1 = 'A read not recorded';
      }
      ctx.strokeStyle = P.line; ctx.lineWidth = 1; ctx.strokeRect(t.x + 0.5, t.y + 0.5, T - 1, T - 1);
      ctx.font = '10px ' + P.mono; ctx.fillStyle = P.muted;
      ctx.fillText(cap1, t.x, t.y + T + 13);
      ctx.fillText(cap2, t.x, t.y + T + 26);
      // arrow from the thumb toward the input lane
      ctx.strokeStyle = P.ok; ctx.lineWidth = 1.5;
      const ay = t.y + T / 2, ax0 = t.x + T + 3, ax1 = g.leftW - 3;
      if (ax1 > ax0 + 4) { ctx.beginPath(); ctx.moveTo(ax0, ay); ctx.lineTo(ax1, ay); ctx.lineTo(ax1 - 4, ay - 3); ctx.moveTo(ax1, ay); ctx.lineTo(ax1 - 4, ay + 3); ctx.stroke(); }
    },

    drawChip(g, st, f, mm, live, info, planeW, wMaxIdle) {
      const ctx = this.ctx, P = this.pal(), C = P.c, dim = g.dim, T = g.Tr;
      const x = g.rx, y = g.chipY;
      const w = live ? f.w : planeW, wMax = live ? info.wMax : wMaxIdle;
      const img = this.paint('chip', dim, dim, (c, k, out) => {
        const v = w[k * dim + c];
        out[0] = C['surface-2'][0]; out[1] = C['surface-2'][1]; out[2] = C['surface-2'][2];
        if (v !== 0) { const col = v > 0 ? C['u-act'] : C.accent; const a = 0.15 + 0.8 * Math.abs(v) / wMax; out[0] += (col[0] - out[0]) * a; out[1] += (col[1] - out[1]) * a; out[2] += (col[2] - out[2]) * a; }
      });
      ctx.font = '600 10px ' + P.head; ctx.fillStyle = P.muted; ctx.fillText('WEIGHTS (STATIONARY)', x, y - 4);
      this.blit(img, x, y, T, T);
      ctx.strokeStyle = P.line; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, T - 1, T - 1);
      this.hits.chip = { x, y, w: T, h: T };
      const plane = live ? mm.plane : st.planes.active;
      const tl = st.planes.tiles[plane];
      const m = this.model;
      ctx.font = '10px ' + P.mono; ctx.fillStyle = P.muted;
      ctx.fillText('plane ' + plane + ' · ' + (tl ? (tileLabel(m.program[tl.pc]) || 'DDR ' + hex(tl.ddr)) : 'zeros') + (live && mm.switched ? ' · switched' : ''), x, y + T + 13);
      ctx.fillText(tl ? 'loaded ' + fmt(tl.cycle) + ' by pc ' + tl.pc : 'never loaded', x, y + T + 26);
    },

    // The staged output rows: cell (r, c) is filled once row r has left column c.
    drawStaged(g, st, f, s, mm, live, info, focus) {
      const ctx = this.ctx, P = this.pal(), C = P.c, dim = g.dim, T = g.Tr;
      const x = g.rx, y = g.stagedY;
      ctx.font = '600 10px ' + P.head; ctx.fillStyle = P.muted; ctx.fillText('STAGED ROWS', x, y - 4);
      if (!live) {
        ctx.fillStyle = P['surface-2']; ctx.fillRect(x, y, T, T);
        ctx.strokeStyle = P.line; ctx.strokeRect(x + 0.5, y + 0.5, T - 1, T - 1);
        ctx.font = '10px ' + P.mono; ctx.fillStyle = P.muted;
        ctx.fillText('nothing staged', x, y + T + 13);
        return;
      }
      const len = mm.len, cw = T / dim;
      const focusRow = focus ? s - focus.k - focus.c : -1;
      const img = this.paint('staged', dim, len, (c, r, out) => {
        const at = r + dim - 1 + c;           // the step this cell lands
        out[0] = C['surface-2'][0]; out[1] = C['surface-2'][1]; out[2] = C['surface-2'][2];
        if (s < at) { if (r === focusRow) { out[0] += (C.sel[0] - out[0]) * 0.25; out[1] += (C.sel[1] - out[1]) * 0.25; out[2] += (C.sel[2] - out[2]) * 0.25; } return; }
        const v = f.land[r * dim + c];
        const col = v >= 0 ? C.ok : C.bad; const a = 0.2 + 0.75 * Math.abs(v) / info.maxLand;
        out[0] += (col[0] - out[0]) * a; out[1] += (col[1] - out[1]) * a; out[2] += (col[2] - out[2]) * a;
        if (s === at) { out[0] = C.staged[0]; out[1] = C.staged[1]; out[2] = C.staged[2]; }
        if (r === focusRow) { out[0] += (C.sel[0] - out[0]) * 0.4; out[1] += (C.sel[1] - out[1]) * 0.4; out[2] += (C.sel[2] - out[2]) * 0.4; }
      });
      const h = cw * len;
      this.blit(img, x, y, T, h);
      ctx.strokeStyle = P.line; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, T - 1, h - 1);
      this.hits.staged = { x, y, w: T, h, cw };
      // complete rows: a tick on the left
      const done = Math.max(0, Math.min(len, s + 3 - 2 * dim));
      if (done > 0) { ctx.fillStyle = P.ok; ctx.fillRect(x - 5, y, 3, cw * done); }
      // arrow from the landing lane into the strip
      ctx.strokeStyle = P.staged; ctx.lineWidth = 1.5;
      const ly = g.oy + g.gridW + g.cell / 2;
      const ax0 = Math.min(g.clip.x + g.clip.w - 2, g.ox + g.gridW + 4);
      if (ly < g.H - 4 && ax0 < x - 8) {
        ctx.beginPath(); ctx.moveTo(ax0, ly); ctx.lineTo(x - 10, ly); ctx.lineTo(x - 10, y + h / 2); ctx.lineTo(x - 3, y + h / 2);
        ctx.moveTo(x - 7, y + h / 2 - 3); ctx.lineTo(x - 3, y + h / 2); ctx.lineTo(x - 7, y + h / 2 + 3); ctx.stroke();
      }
      ctx.font = '10px ' + P.mono; ctx.fillStyle = P.muted;
      ctx.fillText('→ bank ' + mm.bank + (mm.accumulate ? ' (+ bank at issue)' : ' (overwrite)'), x, y + h + 13);
      ctx.fillText('commit ' + fmt(mm.retire) + ' · ' + done + '/' + len + ' rows done', x, y + h + 26);
    },

    drawOverview(g, gridImg) {
      const cv = this.over, ctx = this.octx;
      const W = cv.width;
      ctx.clearRect(0, 0, W, W);
      if (gridImg) { ctx.imageSmoothingEnabled = false; ctx.drawImage(gridImg, 0, 0, W, W); }
      const vx = (g.clip.x - g.ox) / g.gridW * W, vy = (g.clip.y - g.oy) / g.gridW * W;
      const vw = g.clip.w / g.gridW * W, vh = g.clip.h / g.gridW * W;
      ctx.strokeStyle = this.pal().sel; ctx.lineWidth = 1.5;
      ctx.strokeRect(Math.max(0, vx), Math.max(0, vy), Math.min(W, vw), Math.min(W, vh));
    },
  };

  MTV.views.array = view;
})();
