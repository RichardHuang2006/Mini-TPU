// The systolic array view: one cell per PE, drawn from the recorded registers
// after each step of the MatMul in flight (mm.<i>.act / psum in the container).
(function () {
  'use strict';
  const MTV = (window.MTV = window.MTV || {});
  MTV.views = MTV.views || {};
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const view = {
    init(app) {
      this.app = app;
      this.canvas = document.getElementById('array-canvas');
      this.ctx = this.canvas.getContext('2d');
      this.over = document.getElementById('array-overview');
      this.octx = this.over.getContext('2d');
      this.note = document.getElementById('array-note');
      this.mode = 'role';
      this.zoom = 1;
      this.pan = { x: 0, y: 0 };
      this.frame = null;
      this.frameFor = -1;
      document.getElementById('array-mode').addEventListener('click', (e) => {
        const b = e.target.closest('button'); if (!b) return;
        this.mode = b.getAttribute('data-mode');
        for (const x of e.currentTarget.querySelectorAll('button')) x.classList.toggle('active', x === b);
        this.render();
      });
      document.getElementById('array-zoom').addEventListener('input', (e) => { this.zoom = Math.pow(2, e.target.value / 25); this.render(); });
      document.getElementById('array-fit').addEventListener('click', () => { this.zoom = 1; this.pan = { x: 0, y: 0 }; document.getElementById('array-zoom').value = 0; this.render(); });
      this.canvas.addEventListener('click', (e) => this.onClick(e));
      this.canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const f = Math.exp(-e.deltaY * 0.002);
        this.zoom = Math.max(1, Math.min(16, this.zoom * f));
        document.getElementById('array-zoom').value = Math.round(Math.log2(this.zoom) * 25);
        this.render();
      }, { passive: false });
      let drag = null;
      this.canvas.addEventListener('mousedown', (e) => { drag = { x: e.clientX, y: e.clientY, px: this.pan.x, py: this.pan.y }; });
      window.addEventListener('mousemove', (e) => { if (!drag) return; this.pan.x = drag.px + (e.clientX - drag.x); this.pan.y = drag.py + (e.clientY - drag.y); this.render(); });
      window.addEventListener('mouseup', () => { drag = null; });
      this.over.addEventListener('mousedown', (e) => { this.overDrag = true; this.overPan(e); });
      this.over.addEventListener('mousemove', (e) => { if (this.overDrag) this.overPan(e); });
      window.addEventListener('mouseup', () => { this.overDrag = false; });
      window.addEventListener('resize', () => this.render());
    },
    reset(model) { this.model = model; this.frame = null; this.frameFor = -1; },

    geometry() {
      const rect = this.canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const W = Math.max(10, Math.floor(rect.width)), H = Math.max(10, Math.floor(rect.height));
      if (this.canvas.width !== W * dpr || this.canvas.height !== H * dpr) { this.canvas.width = W * dpr; this.canvas.height = H * dpr; }
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const dim = this.model.dim;
      const margin = 56;
      const base = Math.max(4, Math.min((W - margin - 20) / dim, (H - margin - 20) / dim));
      const cell = base * this.zoom;
      const gridW = cell * dim;
      // keep the grid within reach
      const minX = Math.min(margin, W - gridW - 20), minY = Math.min(margin, H - gridW - 20);
      this.pan.x = Math.max(minX - margin, Math.min(0, this.pan.x));
      this.pan.y = Math.max(minY - margin, Math.min(0, this.pan.y));
      return { W, H, cell, ox: margin + this.pan.x, oy: margin + this.pan.y, margin, gridW };
    },

    // True when the PE frames are in hand; otherwise loads and re-renders.
    ensureFrame() {
      const st = this.app.state;
      const mx = st.mxu;
      if (!mx.mm) { this.frame = null; this.frameFor = -1; return true; }
      const idx = mx.mm.index;
      if (this.frameFor === idx && this.frame) return true;
      const c = this.model.c;
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

    onClick(e) {
      const g = this.geometry();
      const dim = this.model.dim;
      const c = Math.floor((e.offsetX - g.ox) / g.cell), k = Math.floor((e.offsetY - g.oy) / g.cell);
      if (k < 0 || c < 0 || k >= dim || c >= dim) return;
      this.app.select({ type: 'pe', k, c, frame: this.frame });
    },
    overPan(e) {
      const g = this.geometry();
      const r = this.over.getBoundingClientRect();
      const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
      this.pan.x = -(fx * g.gridW - (g.W - g.margin) / 2);
      this.pan.y = -(fy * g.gridW - (g.H - g.margin) / 2);
      this.render();
    },

    render() {
      const m = this.model, st = this.app.state;
      if (!m || !st) return;
      const ready = this.ensureFrame();
      const g = this.geometry();
      const ctx = this.ctx;
      const dim = m.dim, n = dim * dim;
      const mx = st.mxu;
      const sel = this.app.sel;
      ctx.clearRect(0, 0, g.W, g.H);
      const f = mx.mm && this.frameFor === mx.mm.index ? this.frame : null;
      const s = mx.step;
      const live = f && s >= 0;
      const mm = mx.mm;
      const planeW = m.img.planes[st.planes.active];
      const okC = css('--ok'), accentC = css('--accent'), stagedC = css('--staged'), muted = css('--muted'), ink = css('--ink'), line = css('--line');
      const mono = css('--mono');
      let maxAbs = 1;
      if (live && this.mode === 'psum') { const ps = f.psum.subarray(s * n, (s + 1) * n); for (let i = 0; i < n; i++) maxAbs = Math.max(maxAbs, Math.abs(ps[i])); }

      // cells
      for (let k = 0; k < dim; k++) {
        for (let c = 0; c < dim; c++) {
          const x = g.ox + c * g.cell, y = g.oy + k * g.cell;
          if (x + g.cell < g.margin - 1 || y + g.cell < g.margin - 1 || x > g.W || y > g.H) continue;
          let fill = 'rgba(128,128,128,0.12)';
          let text = null;
          if (live) {
            const r = s - k - c;
            const useful = r >= 0 && r < mm.len;
            const psum = f.psum[s * n + k * dim + c];
            const act = f.act[s * n + k * dim + c];
            if (this.mode === 'role') fill = useful ? okC : r < 0 ? 'rgba(180,86,42,0.28)' : 'rgba(107,91,181,0.35)';
            else if (this.mode === 'psum') { const a = Math.abs(psum) / maxAbs; fill = psum >= 0 ? 'rgba(30,127,110,' + (0.12 + 0.85 * a) + ')' : 'rgba(180,56,59,' + (0.12 + 0.85 * a) + ')'; }
            else { const a = Math.abs(act) / 128; fill = act >= 0 ? 'rgba(44,111,176,' + (0.12 + 0.85 * a) + ')' : 'rgba(180,86,42,' + (0.12 + 0.85 * a) + ')'; }
            if (g.cell >= 30) text = this.mode === 'act' ? String(act) : String(psum);
          }
          ctx.fillStyle = fill;
          ctx.fillRect(x, y, g.cell - 1, g.cell - 1);
          if (g.cell >= 30) {
            ctx.fillStyle = live && this.mode === 'role' ? '#fff' : ink;
            ctx.font = Math.max(8, Math.min(12, g.cell / 4)) + 'px ' + mono;
            ctx.textAlign = 'right';
            ctx.fillText(text === null ? String(planeW[k * dim + c]) : text, x + g.cell - 4, y + g.cell - 4);
            ctx.textAlign = 'left';
            ctx.fillStyle = live && this.mode === 'role' ? 'rgba(255,255,255,0.8)' : muted;
            ctx.fillText('w' + planeW[k * dim + c], x + 3, y + 11);
          }
        }
      }
      // selected PE
      if (sel && sel.type === 'pe') {
        ctx.strokeStyle = css('--sel'); ctx.lineWidth = 2;
        ctx.strokeRect(g.ox + sel.c * g.cell, g.oy + sel.k * g.cell, g.cell - 1, g.cell - 1);
      }
      // strips: left edge inputs, bottom edge outputs, axes
      ctx.fillStyle = css('--surface');
      ctx.fillRect(0, 0, g.margin - 2, g.H); ctx.fillRect(0, 0, g.W, g.margin - 2);
      ctx.fillStyle = muted; ctx.font = '10px ' + mono;
      ctx.fillText('k↓ / c→', 4, 12);
      if (live) {
        ctx.fillText('A row in ' + (s < mm.len ? '(row ' + s + ')' : '(none)'), 4, 26);
        for (let k = 0; k < dim; k++) {
          const y = g.oy + k * g.cell;
          if (y < g.margin - 2 || y > g.H) continue;
          const v = f.left[s * dim + k];
          ctx.fillStyle = v !== 0 ? okC : muted;
          if (g.cell >= 9) ctx.fillText(String(v), 6, y + Math.min(g.cell, 14) - 2);
        }
        ctx.fillStyle = muted;
        ctx.fillText('bottom edge → landings this step; row r = s+1−dim−c', g.margin, 40);
        for (let c = 0; c < dim; c++) {
          const x = g.ox + c * g.cell;
          if (x < g.margin - 2 || x > g.W) continue;
          const r = s + 1 - dim - c;
          if (r >= 0 && r < mm.len && g.cell >= 9) { ctx.fillStyle = stagedC; ctx.fillText('r' + r, x + 2, 52); }
        }
      } else {
        ctx.fillText('no MatMul step at this cycle/phase: every PE holds 0 (the grid drains to zero at the end of each matmul)', 4, 26);
        ctx.fillText('cells show the resident weights of plane ' + st.planes.active, 4, 40);
      }
      // note
      this.note.textContent = live ? 'pc ' + mm.pc + ' step ' + s + ' of ' + mm.steps + ' · plane ' + mm.plane + ' · ' + (s < mm.len ? 'streaming' : 'fill/drain') + (ready === true ? '' : ' · loading PE detail…') : (mm ? 'loading PE detail…' : 'array idle');
      this.drawOverview(g, live ? f : null, s, mm);
    },

    drawOverview(g, f, s, mm) {
      const cv = this.over, ctx = this.octx, dim = this.model.dim, n = dim * dim;
      const W = cv.width;
      ctx.clearRect(0, 0, W, W);
      const cell = W / dim;
      let maxAbs = 1;
      if (f) for (let i = 0; i < n; i++) maxAbs = Math.max(maxAbs, Math.abs(f.psum[s * n + i]));
      for (let k = 0; k < dim; k++) for (let c = 0; c < dim; c++) {
        let col = 'rgba(128,128,128,0.15)';
        if (f) {
          const r = s - k - c;
          if (this.mode === 'role') col = r >= 0 && r < mm.len ? 'rgba(30,127,110,0.9)' : r < 0 ? 'rgba(180,86,42,0.3)' : 'rgba(107,91,181,0.4)';
          else { const v = f.psum[s * n + k * dim + c]; const a = Math.abs(v) / maxAbs; col = v >= 0 ? 'rgba(30,127,110,' + (0.1 + 0.9 * a) + ')' : 'rgba(180,56,59,' + (0.1 + 0.9 * a) + ')'; }
        }
        ctx.fillStyle = col; ctx.fillRect(c * cell, k * cell, Math.ceil(cell), Math.ceil(cell));
      }
      // viewport rectangle
      const vx = (g.margin - g.ox) / g.gridW * W, vy = (g.margin - g.oy) / g.gridW * W;
      const vw = (g.W - g.margin) / g.gridW * W, vh = (g.H - g.margin) / g.gridW * W;
      ctx.strokeStyle = css('--sel'); ctx.lineWidth = 1.5;
      ctx.strokeRect(Math.max(0, vx), Math.max(0, vy), Math.min(W, vw), Math.min(W, vh));
    },
  };

  MTV.views.array = view;
})();
