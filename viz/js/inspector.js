// The right-hand inspector: the "What happened this cycle?" list and the
// details of whatever is selected (a unit, an instruction, a PE, a matrix
// element, a FIFO slot, a bank, a memory address, a wire). Every number shown
// is read from a trace record; derivations carry a chip saying so.
//
// Classic script: defines window.MTV.Inspector.
(function () {
  'use strict';
  const MTV = (window.MTV = window.MTV || {});
  const { hex, fmt, instrLabel, tileLabel, commitText, readText } = MTV;

  const esc = (s) => String(s).replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
  const chip = (kind, text) => '<span class="chip ' + kind + '">' + text + '</span>';
  const link = (action, text) => '<a data-action="' + esc(action) + '">' + esc(text) + '</a>';
  const pcLink = (model, pc) => link('instr:' + pc, instrLabel(model, pc));
  const cyLink = (t, text) => link('cycle:' + t, text === undefined ? 'cycle ' + fmt(t) : text);

  class Inspector {
    constructor(app, whytEl, titleEl, bodyEl) {
      this.app = app;
      this.whyt = whytEl;
      this.title = titleEl;
      this.body = bodyEl;
      const onClick = (e) => {
        const a = e.target.closest('a[data-action]');
        if (!a) return;
        e.preventDefault();
        this.app.action(a.getAttribute('data-action'));
      };
      this.whyt.addEventListener('click', onClick);
      this.body.addEventListener('click', onClick);
    }

    renderWhyt(model, st) {
      const lines = MTV.narrate(model, st);
      this.whyt.innerHTML = lines.map((l) => {
        let text = esc(l.text);
        // Link "pc N" mentions.
        text = text.replace(/\bpc (\d+)\b/g, (m0, n) => '<a data-action="instr:' + n + '">pc ' + n + '</a>');
        text = text.replace(/\bcycle ([\d,]+)\b/g, (m0, n) => '<a data-action="cycle:' + n.replace(/,/g, '') + '">cycle ' + n + '</a>');
        return '<li class="' + l.kind + '"><span class="ph">' + l.phase + '</span>' + text + '<span class="src">' + esc(l.src) + '</span></li>';
      }).join('');
    }

    render(model, st, sel) {
      let html = '';
      let title = 'Cycle ' + fmt(st.t);
      try {
        if (!sel || sel.type === 'cycle') html = this.cycle(model, st);
        else if (sel.type === 'unit') { title = sel.unit + ' unit'; html = this.unit(model, st, sel.unit); }
        else if (sel.type === 'instr') { title = instrLabel(model, sel.pc); html = this.instr(model, st, sel.pc); }
        else if (sel.type === 'pe') { title = 'PE[' + sel.k + '][' + sel.c + ']'; html = this.pe(model, st, sel); }
        else if (sel.type === 'fifo') { title = 'Weight FIFO'; html = this.fifo(model, st, sel.slot); }
        else if (sel.type === 'bank') { title = 'Accumulator bank ' + sel.bank; html = this.bank(model, st, sel.bank); }
        else if (sel.type === 'wire') { title = sel.label; html = this.wire(model, st, sel); }
        else if (sel.type === 'aelem') { title = 'A[' + sel.i + '][' + sel.k + ']'; html = this.aelem(model, st, sel); }
        else if (sel.type === 'belem') { title = 'B[' + sel.k + '][' + sel.j + ']'; html = this.belem(model, st, sel); }
        else if (sel.type === 'celem') { title = 'C[' + sel.i + '][' + sel.j + ']'; html = this.celem(model, st, sel); }
        else if (sel.type === 'mem') { title = sel.label; html = this.mem(model, st, sel); }
        else if (sel.type === 'planes') { title = 'Weight planes'; html = this.planes(model, st); }
        else if (sel.type === 'seq') { title = 'Sequencer'; html = this.seq(model, st); }
        else if (sel.type === 'stats') { title = 'Statistics'; html = this.stats(model, st); }
        else html = this.cycle(model, st);
      } catch (e) {
        html = '<p class="muted">Inspector error: ' + esc(e.message) + '</p>';
        console.error(e);
      }
      this.title.textContent = title;
      this.body.innerHTML = html;
    }

    // ------------------------------------------------------------ panels --
    kv(rows) {
      return '<dl class="kv">' + rows.map(([k, v, mono]) => '<dt>' + k + '</dt><dd' + (mono ? ' class="mono"' : '') + '>' + v + '</dd>').join('') + '</dl>';
    }

    cycle(model, st) {
      const o = st.outcome;
      const rows = [
        ['Phase', st.phase + ' <span class="muted">(run-loop step)</span>'],
        ['Outcome', o.kind === 'stall' ? 'stall: ' + o.reason + (o.blockerPc !== null ? ' by ' + pcLink(model, o.blockerPc) : '') : o.kind === 'issued' ? 'issued ' + pcLink(model, o.pc) : o.kind],
        ['Attempted', model.program[o.pc] ? pcLink(model, o.pc) : 'pc ' + o.pc + ' (past end)'],
        ['Array', o.idle === 'busy' ? 'busy' : 'idle, charged to ' + o.idle],
      ];
      for (const u of MTV.UNITS) {
        const x = st.units[u];
        rows.push([u, x ? pcLink(model, x.pc) + ' <span class="muted">until ' + fmt(x.done) + '</span>' : '<span class="muted">idle</span>']);
      }
      rows.push(['Weight FIFO', st.fifo.length + '/' + model.cfg.weight_fifo_depth + ' (' + st.fifo.filter((e) => e.ready).length + ' ready)']);
      rows.push(['Planes', 'active ' + st.planes.active + (st.planes.pending ? ', switch pending' : '')]);
      rows.push(['UB streams', st.ubStreams.readers + ' read / ' + st.ubStreams.writers + ' write of ' + model.cfg.ub_banks]);
      let html = this.kv(rows);
      html += '<h3>Running counters at ' + fmt(st.t) + '</h3>' + this.countersSoFar(model, st.t);
      html += '<div class="btnrow">' + link('sel:stats', 'Final statistics') + ' · ' + link('sel:seq', 'Sequencer') + ' · ' + link('sel:planes', 'Weight planes') + '</div>';
      return html;
    }

    countersSoFar(model, t) {
      const stallNames = MTV.STALL_NAMES;
      const hist = {};
      const idle = {};
      for (let x = 0; x <= t; x++) {
        const o = model.cyc.outcome[x];
        const nm = stallNames[o];
        hist[nm] = (hist[nm] || 0) + 1;
        const ib = MTV.IDLE_NAMES[model.cyc.idle[x]];
        idle[ib] = (idle[ib] || 0) + 1;
      }
      const rows = [];
      rows.push(['Issue-point stalls', Object.keys(hist).filter((k) => k !== 'issued' && k !== 'halt').map((k) => k + ' ' + fmt(hist[k])).join(', ') || 'none']);
      rows.push(['Array-idle buckets', Object.keys(idle).filter((k) => k !== 'busy').map((k) => k + ' ' + fmt(idle[k])).join(', ') || 'none']);
      rows.push(['Array busy', fmt(idle.busy || 0) + ' of ' + fmt(t + 1) + ' cycles']);
      return this.kv(rows) + '<p class="note">Two attributions of the same cycles: the counter that refused issue (StallStats) and the bucket the idle array was charged to (charge_idle_cycle). Neither is “the cause”.</p>';
    }

    unit(model, st, u) {
      const x = st.units[u];
      const rows = [];
      if (!x) {
        const next = model.unitIv[u].find((iv) => iv.issue > st.t);
        const blocked = next && model.cyc.pc[st.t] !== next.pc;
        rows.push(['State', 'idle' + (next ? (blocked ? ' (head-of-line blocked: ' + pcLink(model, next.pc) + ' is not at pc yet)' : '') : ' (no more work)')]);
        if (next) rows.push(['Next', pcLink(model, next.pc) + ' issues at ' + cyLink(next.issue)]);
      } else {
        const p = model.program[x.pc];
        rows.push(['State', 'busy with ' + pcLink(model, x.pc)]);
        rows.push(['Issued', cyLink(x.issue)]);
        rows.push(['Retires', cyLink(x.done) + ' (duration ' + fmt(x.done - x.issue) + ')']);
        const elapsed = st.phase === 'start' || st.phase === 'retire' || st.phase === 'prefetch' ? st.t - x.issue : st.t - x.issue + 1;
        if (u === 'MXU') {
          const mm = model.mmByPc.get(x.pc);
          rows.push(['Array step', (st.mxu.step >= 0 ? st.mxu.step : '—') + ' of ' + mm.steps + ' ' + chip('mod', 'modeled')]);
          rows.push(['Phase', st.mxu.step < 0 ? '—' : st.mxu.step < mm.len ? 'streaming (row ' + st.mxu.step + ' admitted)' : 'fill/drain (' + (mm.steps - mm.len) + ' cycles per matmul)']);
          rows.push(['Plane', mm.plane + (mm.switched ? ' (switched at issue)' : '')]);
          rows.push(['Bank', mm.bank + (mm.accumulate ? ' accumulate' : ' overwrite')]);
        } else if (u === 'DMA' || u === 'ACT') {
          rows.push(['Progress', 'cycle ' + fmt(Math.min(elapsed, x.done - x.issue)) + ' of ' + fmt(x.done - x.issue) + ' ' + chip('nom', 'nominal') + ' <span class="muted">no per-cycle data movement is modeled</span>']);
        }
        if (p.res) rows.push(['Reservation', this.resText(p.res)]);
        const iss = model.issueByPc.get(x.pc);
        if (iss && iss.reads.length) rows.push(['Read at issue', iss.reads.map((ri) => readText(model, model.reads[ri])).join('; ')]);
        const commits = model.commits.filter((c) => c.pc === x.pc);
        if (commits.length) rows.push(['Commits at retire', commits.map((c) => commitText(model, c)).join('; ')]);
      }
      const src = { DMA: 'Dma (src/transfer.h), Tpu::execute/finish', WEIGHT: 'WeightFifo (src/transfer.h), Tpu::prefetch_weights/finish', MXU: 'Mxu (src/systolic_array.h)', ACT: 'Tpu::stage_activate, quant:: (src/datapath.h)', SEQ: 'Tpu::issue_step (src/tpu.cpp)' }[u];
      rows.push(['Source', '<span class="mono">' + src + '</span>']);
      let html = this.kv(rows);
      if (u === 'WEIGHT') html += this.fifo(model, st, null);
      if (u === 'MXU') html += this.planes(model, st);
      return html;
    }

    resText(res) {
      const parts = [];
      if (res.ub_read[1] > res.ub_read[0]) parts.push('UB read [' + hex(res.ub_read[0]) + ', ' + hex(res.ub_read[1]) + ')');
      if (res.ub_write[1] > res.ub_write[0]) parts.push('UB write [' + hex(res.ub_write[0]) + ', ' + hex(res.ub_write[1]) + ')');
      if (res.acc_read !== null) parts.push('bank ' + res.acc_read + ' read');
      if (res.acc_write !== null) parts.push('bank ' + res.acc_write + ' write');
      if (res.reads_weights) parts.push('reads weight tile');
      if (res.writes_weights) parts.push('replaces weight tile');
      return parts.join('; ') || 'none';
    }

    instr(model, st, pc) {
      const p = model.program[pc];
      if (!p) return '<p>No such instruction.</p>';
      const iss = model.issueByPc.get(pc);
      const rows = [
        ['Instruction', '<span class="mono">' + esc(p.asm) + '</span>'],
        ['Unit', p.unit],
        ['Tile', tileLabel(p) || '—'],
        ['First attempt', p.first_attempt === null ? '—' : cyLink(p.first_attempt)],
        ['Issued', p.issue === null ? '—' : cyLink(p.issue)],
        ['Retired', p.retire === null ? '—' : cyLink(p.retire) + (iss && p.op !== 'Halt' ? ' (duration ' + fmt(iss.duration) + ')' : '')],
        ['Reservation', p.res ? this.resText(p.res) : '—'],
      ];
      const spans = model.stalls.filter((s) => s.pc === pc);
      if (spans.length) rows.push(['Waited', spans.map((s) => fmt(s.to - s.from + 1) + ' cycles on ' + s.reason + (s.blocker_pc !== null ? ' (' + pcLink(model, s.blocker_pc) + ')' : '') + ' at ' + cyLink(s.from)).join('; ')]);
      if (iss && iss.reads.length) rows.push(['Read at issue', iss.reads.map((ri) => link('read:' + ri, readText(model, model.reads[ri]))).join('; ')]);
      const commits = model.commits.filter((c) => c.pc === pc);
      if (commits.length) rows.push(['Commits at retire', commits.map((c) => link('commit:' + c.index, commitText(model, c))).join('; ')]);
      if (p.op === 'MatMul' && iss) {
        const mm = model.matmuls[iss.mm];
        rows.push(['Array', mm.steps + ' steps (len ' + mm.len + ' + 2·' + model.dim + ' − 1), plane ' + mm.plane + (mm.switched ? ', switched at issue' : '') + '; ' + link('array:' + mm.index, 'open in the array view')]);
      }
      let html = this.kv(rows);
      html += '<div class="btnrow">' + (p.first_attempt !== null ? link('cycle:' + p.first_attempt, '⇥ first attempt') : '') + ' ' + (p.issue !== null ? link('cycle:' + p.issue, '⇥ issue') : '') + ' ' + (p.retire !== null ? link('cycle:' + p.retire, '⇥ retire') : '') + '</div>';
      html += '<h3>Decoded fields</h3><div class="vals">' + esc(JSON.stringify(p.fields, null, 1).replace(/[{}]/g, '').trim()) + '</div>';
      return html;
    }

    fifo(model, st, slot) {
      let html = '<h3>Weight FIFO ' + st.fifo.length + '/' + model.cfg.weight_fifo_depth + '</h3>';
      if (!st.fifo.length) html += '<p class="muted">empty</p>';
      html += '<ol class="hist">' + st.fifo.map((e, i) => {
        const p = e.push;
        return '<li' + (slot === i ? ' style="font-weight:600"' : '') + '>slot ' + i + ': DDR ' + hex(p.ddr) + ' for ' + pcLink(model, p.for_pc) + ', pushed ' + cyLink(p.cycle) + ', ' + (e.ready ? 'ready since ' + cyLink(p.ready) : 'in flight, ready at ' + cyLink(p.ready)) + '</li>';
      }).join('') + '</ol>';
      html += '<p class="note">DDR is modeled as a fixed ' + model.cfg.ddr_tile_latency + '-cycle latency per tile; the prefetcher reads ahead of pc until the FIFO is full. ' + chip('mod', 'modeled') + '</p>';
      return html;
    }

    planes(model, st) {
      const pl = st.planes;
      let html = '<h3>Weight planes</h3>';
      for (let p = 0; p < 2; p++) {
        const w = pl.tiles[p];
        html += '<div>plane ' + p + ': ' + (p === pl.active ? '<b>active</b>' : 'shadow') + (pl.pending && p !== pl.active ? ' (switch pending)' : '') + ' — ' + (w ? 'tile from DDR ' + hex(w.ddr) + ' ' + (tileLabel(model.program[w.pc]) || '') + ', loaded at ' + cyLink(w.cycle) + ' by ' + pcLink(model, w.pc) : 'never loaded (zeros)') + '</div>';
      }
      html += '<p class="note">A Read_Weights pops the FIFO and loads the shadow plane at its retire; the switch happens at the next MatMul’s issue (Mxu::matmul). ' + chip('mod', 'modeled') + '</p>';
      return html;
    }

    bank(model, st, b) {
      const dim = model.dim;
      const base = b * dim * dim;
      const rows = [];
      const writer = Object.values(st.units).find((x) => x && model.program[x.pc].res && model.program[x.pc].res.acc_write === b);
      const reader = Object.values(st.units).find((x) => x && model.program[x.pc].res && model.program[x.pc].res.acc_read === b);
      rows.push(['Reserved for write', writer ? pcLink(model, writer.pc) + ' until ' + fmt(writer.done) : '—']);
      rows.push(['Reserved for read', reader ? pcLink(model, reader.pc) + ' until ' + fmt(reader.done) : '—']);
      const last = [...model.commits].reverse().find((c) => c.kind === 'acc' && c.addr === b && c.cycle <= (st.phase === 'start' ? st.t - 1 : st.t));
      rows.push(['Last commit', last ? pcLink(model, last.pc) + ' at ' + cyLink(last.cycle) + ' (' + last.rows + ' rows)' : 'never written (zeros)']);
      if (writer) {
        const mm = model.mmByPc.get(writer.pc);
        if (mm) rows.push(['Staged', 'the in-flight MatMul’s ' + mm.len + '×' + dim + ' rows, visible to no consumer until ' + cyLink(mm.retire) + ' ' + chip('mod', 'modeled')]);
      }
      let min = 0, max = 0;
      for (let i = 0; i < dim * dim; i++) { const v = model.img.acc[base + i]; if (v < min) min = v; if (v > max) max = v; }
      rows.push(['Contents', 'int32 in [' + fmt(min) + ', ' + fmt(max) + ']; ' + link('mem:acc:' + b, 'open in the memory view')]);
      rows.push(['Locking', chip('nm', 'not modeled') + ' Accumulators::lock is not used by the sequencer; the reservation above is the interlock']);
      return this.kv(rows);
    }

    wire(model, st, sel) {
      const rows = [['This cycle', sel.status || 'no transfer'], ['Carries', sel.detail || '—']];
      if (sel.pc !== undefined && sel.pc !== null) rows.push(['Instruction', pcLink(model, sel.pc)]);
      rows.push(['Rule', sel.rule || '']);
      return this.kv(rows);
    }

    seq(model, st) {
      const o = st.outcome;
      const rows = [
        ['pc', String(o.pc)],
        ['Attempting', model.program[o.pc] ? '<span class="mono">' + esc(model.program[o.pc].asm) + '</span>' : 'past end of program'],
        ['Outcome', o.kind === 'stall' ? o.reason : o.kind],
        ['Blocker', o.blockerPc !== null ? pcLink(model, o.blockerPc) + ' on ' + o.blockerUnit + ' (first conflicting unit in scan order DMA, WEIGHT, MXU, ACT, SEQ)' : '—'],
        ['Prefetch pc', 'see FIFO'],
        ['Retired so far', fmt(model.retires.filter((r) => r.cycle <= st.t).length)],
      ];
      // Other conflicts, derived: re-run the overlap rules over the in-flight set.
      if (o.kind === 'stall' && model.program[o.pc] && model.program[o.pc].res) {
        const r = model.program[o.pc].res;
        const others = [];
        for (const u of MTV.UNITS) {
          const x = st.units[u];
          if (!x || x.pc === o.blockerPc) continue;
          const f = model.program[x.pc].res;
          if (!f) continue;
          const ov = (a, b) => a[1] > a[0] && b[1] > b[0] && a[0] < b[1] && b[0] < a[1];
          const why = [];
          if (ov(r.ub_read, f.ub_write)) why.push('ub_raw');
          if (ov(r.ub_write, f.ub_read)) why.push('ub_war');
          if (ov(r.ub_write, f.ub_write)) why.push('ub_waw');
          if ((r.acc_read !== null && r.acc_read === f.acc_write) || (r.acc_write !== null && (r.acc_write === f.acc_write || r.acc_write === f.acc_read))) why.push('accum_hazard');
          if (r.reads_weights && f.writes_weights) why.push('weight_stall');
          if (why.length) others.push(pcLink(model, x.pc) + ' (' + why.join(', ') + ')');
        }
        rows.push(['Also conflicting', others.length ? others.join('; ') + ' ' + chip('der', 'derived') + ' not attributed by the model' : 'none']);
      }
      return this.kv(rows);
    }

    stats(model) {
      const s = model.m.stats;
      const rows = [
        ['Cycles', fmt(model.cycles)], ['Retired', fmt(model.m.result.retired)],
        ['Array busy', fmt(s.array_busy) + ' (stream ' + fmt(s.stream_cycles) + ' + fill/drain ' + fmt(s.breakdown.array_fill_drain) + ')'],
        ['Idle buckets', Object.entries(s.idle).map(([k, v]) => k + ' ' + fmt(v)).join(', ')],
        ['Stall counters', Object.entries(s.stalls).filter(([, v]) => v).map(([k, v]) => k + ' ' + fmt(v)).join(', ')],
        ['Utilization', (100 * s.utilization).toFixed(1) + '% overall, ' + (100 * s.busy_utilization).toFixed(1) + '% while busy'],
        ['TOPS', s.tops.toFixed(3) + ' ' + chip('nom', 'nominal 700 MHz')],
        ['Roofline', s.arithmetic_intensity.toFixed(1) + ' MAC/B vs ridge ' + s.ridge_point.toFixed(1) + ' → ' + (s.memory_bound ? 'memory-bound' : 'compute-bound')],
        ['Dominant', s.dominant_cause + ' (largest resource stall: ' + s.dominant_stall + ')'],
        ['MXU', 'matmuls ' + s.mxu.matmuls + ', plane switches ' + s.mxu.plane_switches + ', useful MACs ' + fmt(s.mxu.useful_macs) + ', padding ' + fmt(s.mxu.partial_tile_waste)],
      ];
      const ck = model.m.checks;
      rows.push(['Checks', Object.entries(ck).map(([k, v]) => k + ': ' + (typeof v === 'boolean' ? (v ? '✓' : '✗') : v)).join(', ')]);
      return this.kv(rows);
    }

    // ---- PE ----
    pe(model, st, sel) {
      const dim = model.dim;
      const mx = st.mxu;
      const k = sel.k, c = sel.c;
      const rows = [];
      const pl = st.planes;
      const wActive = model.img.planes[pl.active][k * dim + c];
      const wShadow = model.img.planes[1 - pl.active][k * dim + c];
      if (!mx.mm || mx.step < 0) {
        rows.push(['State', 'no MatMul step at this cycle/phase; registers hold ' + chip('mod', '0') + ' (every matmul drains the grid to zero)']);
        rows.push(['w (plane ' + pl.active + ', active)', String(wActive), true]);
        rows.push(['w (plane ' + (1 - pl.active) + ', shadow)', String(wShadow), true]);
        return this.kv(rows);
      }
      const mm = mx.mm;
      const s = mx.step;
      const f = MTV.views.array ? MTV.views.array.frameOf(mm.index) : null;   // {w, left, act, psum, land}
      if (!f) {
        rows.push(['State', 'loading the PE detail of pc ' + mm.pc + '…']);
        return this.kv(rows);
      }
      const n = dim * dim;
      const w = f.w[k * dim + c];
      const actIn = c === 0 ? f.left[s * dim + k] : (s > 0 ? f.act[(s - 1) * n + k * dim + c - 1] : 0);
      const psumIn = k === 0 ? 0 : (s > 0 ? f.psum[(s - 1) * n + (k - 1) * dim + c] : 0);
      const psumOut = f.psum[s * n + k * dim + c];
      const actOut = f.act[s * n + k * dim + c];
      const want = (psumIn + Math.imul(actIn, w)) | 0;
      const ok = want === psumOut && actOut === actIn;
      const r = s - k - c;
      const live = r >= 0 && r < mm.len;
      const p = model.program[mm.pc];
      const t = p.tile;
      const coords = live && t ? { i: t.m * dim + r, kk: t.k * dim + k, j: t.n * dim + c } : null;
      rows.push(['MatMul', pcLink(model, mm.pc) + ', step ' + s + ' of ' + mm.steps + ' (cycle ' + fmt(mm.issue + s) + ')']);
      rows.push(['Role', live ? 'useful: row ' + r + ' of the A tile' : (r < 0 ? 'fill: no row has reached this PE yet (multiplies 0)' : 'drain: rows are past (multiplies 0)') + ' ' + chip('der', 'r = s − k − c')]);
      rows.push(['w (plane ' + mm.plane + ')', String(w) + (coords ? '  = B[' + coords.kk + '][' + coords.j + ']' : ''), true]);
      rows.push(['act_in', String(actIn) + (coords ? '  = A[' + coords.i + '][' + coords.kk + ']' : '') + (c === 0 ? ' (left edge)' : ' (from PE[' + k + '][' + (c - 1) + '])'), true]);
      rows.push(['psum_in', String(psumIn) + (k === 0 ? ' (top edge)' : ' (from PE[' + (k - 1) + '][' + c + '])'), true]);
      let html = this.kv(rows);
      html += '<div class="arith' + (ok ? '' : ' bad') + '">psum: ' + fmt(psumIn) + ' + (' + actIn + ' × ' + w + ') → ' + fmt(psumOut) + (ok ? '' : '   ✗ trace inconsistent') + '</div>';
      const rows2 = [
        ['act_out', String(actOut) + ' → PE[' + k + '][' + (c + 1) + '] at step ' + (s + 1), true],
        ['psum_out', String(psumOut) + (k === dim - 1 ? ' → bottom edge' : ' → PE[' + (k + 1) + '][' + c + '] at step ' + (s + 1)), true],
      ];
      if (coords) rows2.push(['Contributes to', 'C[' + coords.i + '][' + coords.j + '] (tile ' + t.m + ',' + t.n + ', K-tile ' + t.k + ' of ' + model.kt + ') ' + chip('der', 'coordinates from the tiler layout')]);
      if (k === dim - 1 && live) {
        const landed = f.land[r * dim + c];
        rows2.push(['Landing', 'this value (' + fmt(landed) + ') is output row ' + r + ' column ' + c + ' of the staged rows; committed to bank ' + mm.bank + ' at ' + cyLink(mm.retire) + (mm.accumulate ? ' (added to the bank’s value at issue)' : ' (overwrite)')]);
      }
      html += this.kv(rows2);
      html += '<p class="note">Values are the recorded PE registers after this step (Mxu::matmul, commit()); the identity is re-checked here and by viz/check_trace.py. ' + chip('mod', 'modeled') + '</p>';
      return html;
    }

    // ---- matrix elements ----
    aelem(model, st, sel) {
      const e = model.aElement(sel.i, sel.k);
      const dim = model.dim;
      const rows = [['Value', String(e.value), true], ['Tile', 'A(' + e.m + ',' + e.kt + ')'], ['Host byte', hex(e.host) + ' (initial image; A is never rewritten)', true]];
      const rh = e.tilePcs.filter((p) => p.op === 'Read_Host_Memory');
      const resident = rh.filter((p) => p.retire !== null && p.retire <= st.t).pop();
      if (resident) {
        const ub = resident.fields.ub_addr + (sel.i % dim) * dim + (sel.k % dim);
        rows.push(['UB byte', hex(ub) + ' since ' + cyLink(resident.retire) + ' (' + pcLink(model, resident.pc) + '); current UB value ' + model.img.ub[ub], true]);
        rows.push(['History', '<ul class="hist">' + model.history('ub', ub).map((h) => '<li>' + cyLink(h.cycle) + ' ' + h.what + ' by ' + pcLink(model, h.pc) + '</li>').join('') + '</ul>']);
      } else rows.push(['UB', 'not resident at this cycle']);
      const mms = model.program.filter((p) => p.tile && p.tile.kind === 'M' && p.tile.m === e.m && p.tile.k === e.kt);
      rows.push(['Consumed by', mms.map((p) => pcLink(model, p.pc) + ' at ' + cyLink(p.issue)).join(', ') || '—']);
      return this.kv(rows);
    }

    belem(model, st, sel) {
      const e = model.bElement(sel.k, sel.j);
      const dim = model.dim;
      const rows = [['Value', String(e.value), true], ['Tile', 'B(' + e.kt + ',' + e.n + ')'], ['DDR byte', hex(e.ddr) + ' (weight memory is read-only)', true]];
      const loads = model.wloads.filter((w) => w.ddr === e.ddr - (sel.k % dim) * dim - (sel.j % dim) && w.cycle <= st.t);
      const last = loads[loads.length - 1];
      if (last) rows.push(['Plane', last.plane + ' (' + (st.planes.active === last.plane ? 'active' : 'shadow') + ') since ' + cyLink(last.cycle) + ', PE[' + (sel.k % dim) + '][' + (sel.j % dim) + ']; loaded ' + loads.length + '×']);
      const fifo = st.fifo.find((f) => f.push.ddr === e.ddr - (sel.k % dim) * dim - (sel.j % dim));
      if (fifo) rows.push(['FIFO', 'slot ' + fifo.slot + (fifo.ready ? ' (ready)' : ' (in flight, ready ' + fmt(fifo.push.ready) + ')')]);
      rows.push(['Read_Weights', e.tilePcs.map((p) => pcLink(model, p.pc) + ' at ' + cyLink(p.issue)).join(', ') || '—']);
      return this.kv(rows);
    }

    celem(model, st, sel) {
      const e = model.cElement(sel.i, sel.j);
      const dim = model.dim;
      const rows = [['Tile', 'C(' + e.m + ',' + e.n + ')'], ['Host byte', hex(e.host), true]];
      const stateInfo = MTV.views.matrices ? MTV.views.matrices.cState(model, st, e) : null;
      if (stateInfo) rows.push(['State', stateInfo.label + ' ' + chip('mod', 'from events')]);
      if (stateInfo && stateInfo.int32 !== undefined) rows.push(['int32 partial', fmt(stateInfo.int32) + (stateInfo.where ? ' (' + stateInfo.where + ')' : ''), true]);
      if (stateInfo && stateInfo.int8 !== undefined) rows.push(['int8', String(stateInfo.int8) + (stateInfo.where8 ? ' (' + stateInfo.where8 + ')' : ''), true]);
      rows.push(['K-chain', e.chain.map((p) => pcLink(model, p.pc) + ' (' + cyLink(p.issue) + '→' + cyLink(p.retire) + ')').join(', ') || '—']);
      rows.push(['Activate', e.acts.map((p) => pcLink(model, p.pc) + ' (' + cyLink(p.issue) + '→' + cyLink(p.retire) + ')').join(', ') || '—']);
      rows.push(['Write_Host', e.writes.map((p) => pcLink(model, p.pc) + ' (' + cyLink(p.issue) + '→' + cyLink(p.retire) + ')').join(', ') || '—']);
      const bank = e.chain.length ? e.chain[0].fields.acc_bank : null;
      if (bank !== null) rows.push(['Bank cell', 'bank ' + bank + ' [' + e.r + '][' + e.c + '], current value ' + fmt(model.img.acc[bank * dim * dim + e.r * dim + e.c]), true]);
      if (e.acts.length) rows.push(['UB staging byte', hex(e.acts[0].fields.ub_addr + e.r * dim + e.c), true]);
      let html = this.kv(rows);
      if (bank !== null) html += '<h3>History of bank ' + bank + ' [' + e.r + '][' + e.c + ']</h3><ul class="hist">' + model.history('acc', bank, e.r, e.c).map((h) => '<li>' + cyLink(h.cycle) + ' ' + h.what + ' by ' + pcLink(model, h.pc) + '</li>').join('') + '</ul>';
      return html;
    }

    mem(model, st, sel) {
      const rows = [['Address', sel.addrText, true], ['Value', sel.valueText, true]];
      if (sel.extra) for (const [k, v] of sel.extra) rows.push([k, v]);
      let html = this.kv(rows);
      const hist = model.history(sel.kind === 'ddr' ? 'none' : sel.kind, sel.addr, sel.row, sel.col);
      if (hist.length) html += '<h3>History</h3><ul class="hist">' + hist.map((h) => '<li>' + cyLink(h.cycle) + ' ' + h.what + ' by ' + pcLink(model, h.pc) + ' (' + (h.rec.kind === 'acc' ? h.rec.rows + ' rows' : fmt(h.rec.len) + ' B') + ')</li>').join('') + '</ul>';
      else if (sel.kind === 'ddr') html += '<p class="note">Weight memory is never written by the ISA; the prefetcher copies tiles from it.</p>';
      return html;
    }
  }

  MTV.Inspector = Inspector;
  MTV.esc = esc;
  MTV.chip = chip;
})();
