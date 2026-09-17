// The controller: owns the cycle, phase and selection state, drives every view
// from it, and handles playback and keyboard navigation.
(function () {
  'use strict';
  const MTV = (window.MTV = window.MTV || {});
  const { fmt } = MTV;
  const SPEEDS = [1, 2, 4, 16, 64, 256, 1024, 4096];

  // Pointers to trace records; the views show the records, not this text.
  // [cycle, text, view]: the view opens before seeking (array steps are best
  // watched on the Systolic array tab, stalls on the block diagram).
  const BOOKMARKS = {
    matmul_128: [
      [0, 'DMA preamble: Read_Host A(0,0) issues; the prefetcher has already filled the FIFO', 'diagram'],
      [193, 'First Read_Weights issues (its tile has been ready since cycle 8)', 'diagram'],
      [194, 'First MatMul issues: step 0, A[0][0] enters PE[0][0]; plane switch 0→1', 'array'],
      [196, 'Read_Weights #6 retires: shadow plane 0 loads under the running MatMul', 'diagram'],
      [225, 'Step 31: the wavefront spans the anti-diagonal; C[0][0] is the first landing', 'array'],
      [230, 'Step 36: PE[31][0] computes 70457 + (−31 × −46) → 71883 for C[5][0]', 'array'],
      [256, 'Step 62: A[31][31] is the last input; output row 0 is complete', 'array'],
      [287, 'Step 93: the last partial sum (row 31, column 31) lands', 'array'],
      [288, 'Step 94: the grid has drained to zero; 32/32 rows are staged for bank 0', 'array'],
      [289, 'MatMul #5 retires (bank 0 commit) and #7 (accumulate) issues in the same cycle', 'array'],
      [480, 'Activate #12 waits on accum_hazard while MatMul #11 still owns bank 0', 'diagram'],
      [574, 'Activate #12 issues; the array goes idle and is charged to “activation”', 'diagram'],
      [957, 'Write_Host #21 is blocked by ub_raw behind Activate #12 for 645 cycles', 'diagram'],
      [1000, 'Idle array behind Activate #12: what the grid shows between MatMuls', 'array'],
      [1602, 'Activate #12 retires; Write_Host #21 issues next cycle', 'diagram'],
      [18949, 'Halt waits for the last Write_Host (drain)', 'diagram'],
      [19012, 'Halt issues; the run ends at 19 013 cycles', 'diagram'],
    ],
    matmul_8: [
      [0, 'Read_Host A(0,0) issues; the prefetcher pushes four tiles, ready at cycle 8', 'diagram'],
      [5, 'Read_Weights #2 stalls on weight_fifo_empty: the DDR latency is exposed', 'diagram'],
      [8, 'Read_Weights #2 issues; it loads plane 1 at retire', 'diagram'],
      [9, 'MatMul #3 issues: step 0, A[0][0] enters PE[0][0]; plane switch 0→1', 'array'],
      [11, 'Read_Weights #4 retires: shadow plane 0 loads under the running MatMul', 'diagram'],
      [12, 'Step 3: the wavefront spans the anti-diagonal; C[0][0] is the first landing', 'array'],
      [18, 'Step 9: the last partial sum lands', 'array'],
      [19, 'Step 10: the grid has drained; 4/4 rows are staged', 'array'],
      [20, 'MatMul #3 retires; MatMul #5 (accumulate) issues with a plane switch', 'array'],
      [21, 'Activate #6 stalls on accum_hazard against MatMul #5', 'diagram'],
      [31, 'Activate #6 issues (20 cycles = 4·4 + 4)', 'diagram'],
      [51, 'Write_Host #11 issues after waiting on ub_raw', 'diagram'],
      [155, 'Halt', 'diagram'],
    ],
  };

  class App {
    constructor() {
      const $ = (id) => document.getElementById(id);
      this.el = {
        wl: $('wl-select'), file: $('open-file'), reset: $('btn-reset'), play: $('btn-play'), prev: $('btn-prev'), next: $('btn-next'),
        prevEvent: $('btn-prev-event'), nextEvent: $('btn-next-event'), prevMm: $('btn-prev-mm'), nextMm: $('btn-next-mm'),
        prevTile: $('btn-prev-tile'), nextTile: $('btn-next-tile'), speed: $('speed'), speedLabel: $('speed-label'),
        cycle: $('cycle-input'), slider: $('cycle-slider'), total: $('cycle-total'), phase: $('phase'), bookmarks: $('bookmarks'),
        tabs: $('tabs'), badge: $('badge'), viewNote: $('view-note'),
      };
      this.sel = { type: 'cycle' };
      this.view = 'array';
      this.playing = false;
      this.speed = SPEEDS[3];
      this.inspector = new MTV.Inspector(this, $('cycle-strip'), $('insp-title'), $('insp'));
      for (const v of ['timeline', 'diagram', 'array', 'matrices', 'memory']) MTV.views[v].init(this);
      this.bind();
    }

    bind() {
      const e = this.el;
      e.wl.addEventListener('change', () => this.loadSelection());
      e.file.addEventListener('change', () => { if (e.file.files[0]) this.loadFile(e.file.files[0]); });
      e.reset.addEventListener('click', () => this.setCycle(0, 'account'));
      e.play.addEventListener('click', () => this.togglePlay());
      e.prev.addEventListener('click', () => this.step(-1));
      e.next.addEventListener('click', () => this.step(1));
      e.prevEvent.addEventListener('click', () => this.setCycle(this.model.prevEventCycle(this.t)));
      e.nextEvent.addEventListener('click', () => this.setCycle(this.model.nextEventCycle(this.t)));
      e.prevMm.addEventListener('click', () => this.setCycle(this.model.prevMatmulCycle(this.t)));
      e.nextMm.addEventListener('click', () => this.setCycle(this.model.nextMatmulCycle(this.t)));
      e.prevTile.addEventListener('click', () => this.setCycle(this.model.prevTileCycle(this.t)));
      e.nextTile.addEventListener('click', () => this.setCycle(this.model.nextTileCycle(this.t)));
      e.speed.addEventListener('input', () => { this.speed = SPEEDS[+e.speed.value]; e.speedLabel.textContent = this.speed + ' cyc/s'; if (this.playing) { this.pause(); this.play(); } });
      e.cycle.addEventListener('change', () => this.setCycle(+e.cycle.value));
      e.slider.addEventListener('input', () => this.setCycle(+e.slider.value));
      e.phase.addEventListener('click', (ev) => { const b = ev.target.closest('button'); if (b) this.setCycle(this.t, b.getAttribute('data-phase')); });
      e.bookmarks.addEventListener('change', () => {
        const v = e.bookmarks.value;
        if (v !== '') { const [t, view] = v.split(':'); if (view) this.setView(view); this.setCycle(+t, 'account'); }
        e.bookmarks.value = '';
      });
      e.tabs.addEventListener('click', (ev) => { const b = ev.target.closest('.tab'); if (b) this.setView(b.getAttribute('data-view')); });
      window.addEventListener('keydown', (ev) => this.onKey(ev));
      document.body.addEventListener('dragover', (ev) => { ev.preventDefault(); });
      document.body.addEventListener('drop', (ev) => { ev.preventDefault(); const f = ev.dataTransfer.files[0]; if (f) this.loadFile(f); });
    }

    async boot() {
      this.embedded = (window.MTPT_EMBEDDED || []);
      this.sources = this.embedded.map((e, i) => ({ label: e.name.replace(/\.mtpt$/, '') + ' (embedded)', kind: 'embedded', index: i }));
      // Served over HTTP, visualizer.py lists every container in viz/traces.
      const served = /^https?:$/.test(location.protocol);
      if (served) {
        try {
          const resp = await fetch('traces/index.json', { cache: 'no-store' });
          if (resp.ok) {
            for (const e of await resp.json()) {
              if (!e || !e.file) continue;
              this.sources.push({ label: e.name + ' (' + (e.bytes >= 1e6 ? (e.bytes / 1e6).toFixed(0) + ' MB' : (e.bytes / 1e3).toFixed(0) + ' kB') + ')', kind: 'url', url: e.file, name: e.file.split('/').pop() });
            }
          }
        } catch (err) { /* static host without the index: Open… still works */ }
      }
      this.refreshSources();
      const params = new URLSearchParams(location.search);
      const want = params.get('trace');
      let start = 0;
      if (want) {
        const i = this.sources.findIndex((s) => s.kind === 'url' && (s.url === want || s.name === want.split('/').pop()));
        if (i >= 0) start = i;
        else { this.sources.push({ label: want.split('/').pop(), kind: 'url', url: want, name: want.split('/').pop() }); this.refreshSources(); start = this.sources.length - 1; }
      } else {
        // The 128×128 example is the one to open when it has been traced.
        const i = this.sources.findIndex((s) => s.kind === 'url' && s.name === 'matmul_128.mtpt');
        if (i >= 0) start = i;
      }
      if (this.sources.length) await this.loadSource(start);
    }
    refreshSources() {
      this.el.wl.innerHTML = this.sources.map((s, i) => '<option value="' + i + '">' + s.label + '</option>').join('');
      if (this.current !== undefined) this.el.wl.value = String(this.current);
    }
    loadSelection() { this.loadSource(+this.el.wl.value); }
    async loadFile(file) {
      this.sources.push({ label: file.name, kind: 'file', file });
      this.refreshSources();
      await this.loadSource(this.sources.length - 1);
    }
    async loadSource(i) {
      const s = this.sources[i];
      this.current = i;
      this.el.wl.value = String(i);
      this.pause();
      this.el.badge.textContent = 'loading…'; this.el.badge.className = 'badge';
      try {
        let c;
        if (s.kind === 'embedded') c = await MTV.Container.open({ base64: this.embedded[s.index].base64 }, this.embedded[s.index].name);
        else if (s.kind === 'file') c = await MTV.Container.open(s.file, s.file.name);
        else {
          if (!s.buffer) {
            // Fetched whole into memory, so the PE frames are synchronous views (mmSync).
            this.el.badge.textContent = 'fetching ' + s.label + '…';
            const resp = await fetch(s.url);
            if (!resp.ok) throw new Error('could not fetch ' + s.url + ' (' + resp.status + ')');
            s.buffer = await resp.arrayBuffer();
          }
          c = await MTV.Container.open(s.buffer, s.name);
        }
        if (this.current !== i) return;   // another selection won the race
        this.setModel(new MTV.Model(c));
      } catch (err) {
        console.error(err);
        this.el.badge.textContent = 'failed: ' + err.message; this.el.badge.className = 'badge bad';
      }
    }

    setModel(model) {
      this.model = model;
      this.sel = { type: 'cycle' };
      for (const v of ['timeline', 'diagram', 'array', 'matrices', 'memory']) MTV.views[v].reset(model);
      this.el.total.textContent = fmt(model.cycles);
      this.el.slider.max = String(model.cycles - 1);
      this.el.cycle.max = String(model.cycles - 1);
      const ck = model.m.checks;
      const okAll = ck && ck.all;
      this.el.badge.textContent = okAll ? 'trace verified ✓' : 'trace checks FAILED';
      this.el.badge.title = 'tools/tracegen checks: run unchanged by tracing, oracle, golden loops, PE identity (' + fmt(ck.pe_checked) + ' steps), landings, commit replay, requantize, stall and idle partitions, unit intervals';
      this.el.badge.className = 'badge ' + (okAll ? 'ok' : 'bad');
      const bm = BOOKMARKS[model.workload.name] || [];
      this.el.bookmarks.innerHTML = '<option value="">jump to…</option>' + bm.filter(([t]) => t < model.cycles).map(([t, text, view]) => '<option value="' + t + ':' + (view || '') + '">' + t + ' · ' + text + '</option>').join('');
      document.title = 'Mini-TPU Cycle Visualizer · ' + model.workload.name;
      this.setCycle(0, 'account');
    }

    setCycle(t, phase) {
      if (!this.model) return;
      phase = phase || this.phase || 'account';
      t = Math.max(0, Math.min(this.model.cycles - 1, t | 0));
      this.t = t; this.phase = phase;
      this.state = this.model.setCycle(t, phase);
      this.el.cycle.value = String(t);
      this.el.slider.value = String(t);
      for (const b of this.el.phase.querySelectorAll('button')) b.classList.toggle('active', b.getAttribute('data-phase') === phase);
      MTV.views.timeline.ensureVisible(t);
      this.render();
    }
    step(d) {
      // Stepping by a cycle always lands on the after-account state of the new cycle.
      if (this.phase !== 'account' && d > 0) {
        const i = MTV.PHASES.indexOf(this.phase);
        this.setCycle(this.t, MTV.PHASES[i + 1]);
        return;
      }
      this.setCycle(this.t + d, 'account');
    }
    stepPhase(d) {
      const i = MTV.PHASES.indexOf(this.phase) + d;
      if (i < 0) this.setCycle(this.t - 1, 'account');
      else if (i >= MTV.PHASES.length) this.setCycle(this.t + 1, 'start');
      else this.setCycle(this.t, MTV.PHASES[i]);
    }

    render() {
      const st = this.state;
      this.inspector.renderStrip(this.model, st);
      this.inspector.render(this.model, st, this.sel);
      MTV.views.timeline.render();
      MTV.views[this.view].render();
      this.el.viewNote.textContent = 'cycle ' + fmt(st.t) + ' · ' + st.phase + (st.outcome.kind === 'stall' ? ' · stalled: ' + st.outcome.reason : st.outcome.kind === 'issued' ? ' · issued pc ' + st.outcome.pc : '');
    }

    select(sel) {
      this.sel = sel || { type: 'cycle' };
      if (sel && sel.type === 'instr') {
        const p = this.model.program[sel.pc];
        if (p && p.op === 'MatMul' && this.view === 'diagram') { /* stay */ }
      }
      this.render();
    }
    setView(name) {
      this.view = name;
      for (const b of this.el.tabs.querySelectorAll('.tab')) b.classList.toggle('active', b.getAttribute('data-view') === name);
      for (const v of ['diagram', 'array', 'matrices', 'memory']) document.getElementById('view-' + v).hidden = v !== name;
      this.render();
    }

    // Links from the inspector and narration.
    action(a) {
      const [kind, arg, arg2] = a.split(':');
      if (kind === 'cycle') this.setCycle(+arg);
      else if (kind === 'instr') this.select({ type: 'instr', pc: +arg });
      else if (kind === 'sel') this.select({ type: arg });
      else if (kind === 'view') this.setView(arg);
      else if (kind === 'array') { const mm = this.model.matmuls[+arg]; this.setView('array'); this.setCycle(mm.issue, 'account'); }
      else if (kind === 'mem') { this.setView('memory'); MTV.views.memory.show(arg); if (arg2 !== undefined) { MTV.views.memory.addr = 0; } this.render(); }
      else if (kind === 'read') { const r = this.model.reads[+arg]; this.showBlob('Read at issue by pc ' + r.pc, r.kind === 'acc' ? this.model.readI32(r) : this.model.readI8(r), r.cols || 16, r.kind); }
      else if (kind === 'commit') { const c = this.model.commits[+arg]; this.showBlob('Commit at retire by pc ' + c.pc, c.kind === 'acc' ? this.model.commitAfterI32(c) : new Int8Array(this.model.commitBlob.buffer, this.model.commitBlob.byteOffset + c.after, c.len), c.cols || 16, c.kind); }
    }
    showBlob(title, arr, cols, kind) {
      const rows = [];
      const n = Math.min(arr.length, 4096);
      const c = kind === 'acc' || kind === 'plane' ? Math.min(cols, 32) : 16;
      for (let i = 0; i < n; i += c) rows.push(Array.from(arr.subarray(i, Math.min(n, i + c))).map((v) => String(v).padStart(kind === 'acc' ? 7 : 5)).join(''));
      const body = document.getElementById('insp');
      body.insertAdjacentHTML('afterbegin', '<h3>' + MTV.esc(title) + '</h3><div class="vals">' + MTV.esc(rows.join('\n')) + (arr.length > n ? '\n… ' + (arr.length - n) + ' more' : '') + '</div>');
    }

    togglePlay() { if (this.playing) this.pause(); else this.play(); }
    play() {
      if (!this.model) return;
      this.playing = true;
      this.el.play.textContent = '⏸';
      const per = Math.max(1, Math.round(this.speed / 30));
      const interval = this.speed >= 30 ? 1000 / 30 : 1000 / this.speed;
      this.timer = setInterval(() => {
        if (this.t >= this.model.cycles - 1) { this.pause(); return; }
        this.setCycle(this.t + per, 'account');
      }, interval);
    }
    pause() { this.playing = false; this.el.play.textContent = '▶'; if (this.timer) clearInterval(this.timer); this.timer = null; }

    onKey(ev) {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
      if (!this.model) return;
      const k = ev.key;
      if (k === 'ArrowLeft') { this.setCycle(this.t - (ev.shiftKey ? 10 : 1), 'account'); ev.preventDefault(); }
      else if (k === 'ArrowRight') { this.setCycle(this.t + (ev.shiftKey ? 10 : 1), 'account'); ev.preventDefault(); }
      else if (k === ' ') { this.togglePlay(); ev.preventDefault(); }
      else if (k === '.') this.stepPhase(1);
      else if (k === ',') this.stepPhase(-1);
      else if (k === 'e') this.setCycle(this.model.nextEventCycle(this.t));
      else if (k === 'E') this.setCycle(this.model.prevEventCycle(this.t));
      else if (k === 'm') this.setCycle(this.model.nextMatmulCycle(this.t));
      else if (k === 'M') this.setCycle(this.model.prevMatmulCycle(this.t));
      else if (k === 't') this.setCycle(this.model.nextTileCycle(this.t));
      else if (k === 'T') this.setCycle(this.model.prevTileCycle(this.t));
      else if (k === '1') this.setView('diagram');
      else if (k === '2') this.setView('array');
      else if (k === '3') this.setView('matrices');
      else if (k === '4') this.setView('memory');
      else if (k === 'Escape') this.select({ type: 'cycle' });
    }
  }

  window.addEventListener('DOMContentLoaded', () => {
    MTV.app = new App();
    MTV.app.boot();
  });
})();
