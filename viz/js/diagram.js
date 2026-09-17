// The microarchitecture diagram (SVG). A wire lights up only in the phase whose
// trace records a transfer on it: green read, violet commit, copper prefetch.
(function () {
  'use strict';
  const MTV = (window.MTV = window.MTV || {});
  MTV.views = MTV.views || {};
  const { hex, fmt, instrLabel, tileLabel } = MTV;
  const esc = (s) => String(s).replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));

  const WIRES = {
    host_dma: { d: 'M270,352 L330,352', label: 'host → DMA (read at Read_Host issue)' },
    dma_ub: { d: 'M480,340 L540,340', label: 'DMA → UB (commit at Read_Host retire)' },
    ub_dma: { d: 'M540,366 L480,366', label: 'UB → DMA (read at Write_Host issue)' },
    dma_host: { d: 'M330,378 L270,378', label: 'DMA → host (commit at Write_Host retire)' },
    ddr_pref: { d: 'M405,80 L405,112', label: 'DDR → prefetcher (tile copied at push)' },
    pref_fifo: { d: 'M480,138 L540,138', label: 'prefetcher → FIFO (push)' },
    fifo_mxu: { d: 'M740,95 L800,95', label: 'FIFO → weight plane (pop at Read_Weights retire)' },
    ub_mxu: { d: 'M740,300 L770,300 L770,170 L800,170', label: 'UB → MXU (A tile read at MatMul issue)' },
    mxu_acc: { d: 'M925,220 L925,268', label: 'MXU → accumulator bank (commit at MatMul retire)' },
    acc_mxu: { d: 'M960,268 L960,220', label: 'bank → MXU (bank read at accumulating MatMul issue)' },
    acc_act: { d: 'M940,338 L940,380', label: 'bank → activation (read at Activate issue)' },
    act_ub: { d: 'M800,418 L640,418 L640,390', label: 'activation → UB (commit at Activate retire)' },
  };

  function svgMarkup() {
    let s = '<svg viewBox="0 0 1100 470" role="img" aria-label="Mini-TPU block diagram">';
    s += '<defs><marker id="dg-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:context-stroke"/></marker></defs>';
    // wires first (under blocks)
    for (const [id, w] of Object.entries(WIRES)) {
      s += '<path id="wire-' + id + '" class="wire" d="' + w.d + '" marker-end="url(#dg-ah)" data-wire="' + id + '"><title>' + esc(w.label) + '</title></path>';
    }
    const block = (id, x, y, w, h, cls) => '<rect id="blk-' + id + '" class="blk ' + (cls || '') + '" data-block="' + id + '" x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="4"/>';
    const text = (id, x, y, cls, size) => '<text id="' + id + '" x="' + x + '" y="' + y + '" class="' + (cls || '') + '" font-size="' + (size || 11) + '" pointer-events="none"></text>';
    // Sequencer + scoreboard
    s += block('seq', 20, 20, 290, 152);
    s += '<text x="34" y="42" class="title" pointer-events="none">Sequencer</text>';
    s += text('seq-l1', 34, 62, 'mono', 10) + text('seq-l2', 34, 78, 'mut', 10.5) + text('seq-l3', 34, 94, 'mut', 10.5);
    s += '<text x="34" y="116" class="title" font-size="11" pointer-events="none">Scoreboard</text>';
    for (let i = 0; i < 5; i++) s += text('sb-' + i, 34, 130 + i * 8.5, 'mono', 8.5);
    // Program memory
    s += block('prog', 20, 190, 290, 50, 'mem');
    s += '<text x="34" y="210" class="title" pointer-events="none">Program memory</text>' + text('prog-l1', 34, 228, 'mut', 10.5);
    // Host
    s += block('host', 20, 320, 250, 78, 'mem');
    s += '<text x="34" y="340" class="title" pointer-events="none">Host memory</text>' + text('host-l1', 34, 357, 'mono', 10) + text('host-l2', 34, 372, 'mut', 10) + text('host-l3', 34, 387, 'mut', 10);
    // DMA
    s += block('dma', 330, 320, 150, 78);
    s += '<text x="344" y="340" class="title" pointer-events="none">DMA engine</text>' + text('dma-l1', 344, 357, 'mono', 10) + text('dma-l2', 344, 372, 'mut', 10) + text('dma-l3', 344, 387, 'mut', 10);
    // UB
    s += block('ub', 540, 270, 200, 128);
    s += '<text x="554" y="290" class="title" pointer-events="none">Unified Buffer</text>' + text('ub-l1', 554, 306, 'mut', 10) + text('ub-l2', 554, 321, 'mono', 10);
    for (let i = 0; i < 4; i++) s += text('ub-r' + i, 554, 338 + i * 14, 'mono', 9.5);
    // DDR
    s += block('ddr', 330, 20, 150, 60, 'mem');
    s += '<text x="344" y="40" class="title" pointer-events="none">Weight memory</text>' + text('ddr-l1', 344, 57, 'mut', 10) + text('ddr-l2', 344, 71, 'mono', 10);
    // Prefetcher
    s += block('pref', 330, 112, 150, 52);
    s += '<text x="344" y="131" class="title" pointer-events="none">Prefetcher</text>' + text('pref-l1', 344, 147, 'mut', 10) + text('pref-l2', 344, 159, 'mono', 9.5);
    // FIFO
    s += block('fifo', 540, 20, 200, 150);
    s += '<text x="554" y="40" class="title" pointer-events="none">Weight FIFO</text>' + text('fifo-l1', 554, 56, 'mut', 10);
    for (let i = 0; i < 6; i++) {
      s += '<rect id="fifo-slot' + i + '" class="slot" data-block="fifo" data-slot="' + i + '" x="554" y="' + (66 + i * 17) + '" width="172" height="15"/>';
      s += text('fifo-s' + i, 560, 77 + i * 17, 'mono', 9);
    }
    // MXU
    s += block('mxu', 800, 20, 280, 200);
    s += '<text x="814" y="40" class="title" pointer-events="none">Matrix unit</text>' + text('mxu-l1', 814, 57, 'mut', 10);
    s += '<rect id="mxu-grid" data-block="mxu" x="814" y="66" width="120" height="120" class="slot" style="cursor:pointer"><title>Open the Systolic array tab</title></rect>';
    s += '<foreignObject x="814" y="66" width="120" height="120" pointer-events="none"><canvas xmlns="http://www.w3.org/1999/xhtml" id="mxu-mini" width="120" height="120" style="display:block"></canvas></foreignObject>';
    for (let i = 0; i < 7; i++) s += text('mxu-r' + i, 944, 80 + i * 16, i === 0 ? 'mono' : 'mut', 10);
    s += text('mxu-l2', 814, 200, 'mut', 9.5);
    // ACC
    s += block('acc', 800, 268, 280, 70);
    s += '<text x="814" y="286" class="title" pointer-events="none">Accumulator banks</text>';
    for (let i = 0; i < 4; i++) s += '<rect id="bank-' + i + '" class="slot" data-block="bank" data-bank="' + i + '" x="' + (814 + i * 66) + '" y="294" width="60" height="36"/>' + text('bank-t' + i, 818 + i * 66, 306, 'mono', 9.5) + text('bank-u' + i, 818 + i * 66, 318, 'mut', 8.5) + text('bank-v' + i, 818 + i * 66, 328, 'mut', 8.5);
    // ACT
    s += block('act', 800, 380, 280, 70);
    s += '<text x="814" y="398" class="title" pointer-events="none">Activation pipeline</text>' + text('act-l1', 814, 414, 'mono', 10) + text('act-l2', 814, 428, 'mut', 10) + text('act-l3', 814, 442, 'mut', 9.5);
    // wire labels
    s += '<text x="300" y="344" class="wlabel" text-anchor="middle">read@issue</text><text x="510" y="332" class="wlabel" text-anchor="middle">commit@retire</text>';
    s += '<text x="770" y="88" class="wlabel" text-anchor="middle">pop@retire</text><text x="785" y="240" class="wlabel" text-anchor="middle">A tile</text>';
    s += '<text x="905" y="248" class="wlabel" text-anchor="end">commit</text><text x="975" y="248" class="wlabel">bank in</text>';
    s += '<text x="952" y="364" class="wlabel">read@issue</text><text x="720" y="432" class="wlabel" text-anchor="middle">int8 tile commit@retire</text>';
    s += '<text x="418" y="100" class="wlabel">copy</text><text x="510" y="132" class="wlabel" text-anchor="middle">push</text>';
    // legend
    s += '<g transform="translate(20,412)"><line x1="0" y1="8" x2="26" y2="8" class="wire acc"/><text x="32" y="12" font-size="10">read / accepted at issue</text>';
    s += '<line x1="170" y1="8" x2="196" y2="8" class="wire com"/><text x="202" y="12" font-size="10">commit at retire</text>';
    s += '<line x1="300" y1="8" x2="326" y2="8" class="wire pref"/><text x="332" y="12" font-size="10">prefetch push</text>';
    s += '<line x1="0" y1="30" x2="26" y2="30" class="wire req"/><text x="32" y="34" font-size="10">requested, stalled</text>';
    s += '<text x="170" y="34" font-size="10" class="mut">nothing moves on a wire unless the trace has an event on it in this phase</text></g>';
    s += '<text id="dg-note" x="20" y="462" font-size="10" class="mut"></text>';
    s += '</svg>';
    return s;
  }

  const view = {
    init(app) {
      this.app = app;
      this.el = document.getElementById('view-diagram');
      this.el.innerHTML = svgMarkup();
      this.svg = this.el.querySelector('svg');
      this.svg.addEventListener('click', (e) => this.onClick(e));
      this.mini = document.getElementById('mxu-mini');
    },
    reset(model) { this.model = model; },
    t(id, text) { const e = this.svg.querySelector('#' + id); if (e) e.textContent = text; },
    cls(id, classes) { const e = this.svg.querySelector('#' + id); if (!e) return; e.setAttribute('class', classes); },

    onClick(e) {
      const wire = e.target.closest('[data-wire]');
      if (wire) { this.selectWire(wire.getAttribute('data-wire')); return; }
      const blk = e.target.closest('[data-block]');
      if (!blk) return;
      const id = blk.getAttribute('data-block');
      if (blk.id === 'mxu-grid') { this.app.setView('array'); return; }   // the mini grid opens the array tab
      const map = { seq: { type: 'seq' }, prog: { type: 'seq' }, host: { type: 'mem', kind: 'host', label: 'Host memory', open: true }, dma: { type: 'unit', unit: 'DMA' },
        ub: { type: 'mem', kind: 'ub', label: 'Unified Buffer', open: true }, ddr: { type: 'mem', kind: 'ddr', label: 'Weight memory', open: true },
        pref: { type: 'unit', unit: 'WEIGHT' }, fifo: { type: 'fifo', slot: blk.hasAttribute('data-slot') ? +blk.getAttribute('data-slot') : null },
        mxu: { type: 'unit', unit: 'MXU' }, bank: { type: 'bank', bank: +blk.getAttribute('data-bank') }, act: { type: 'unit', unit: 'ACT' } };
      const sel = map[id];
      if (!sel) return;
      if (sel.open) { this.app.setView('memory'); MTV.views.memory.show(sel.kind); this.app.select({ type: 'cycle' }); return; }
      this.app.select(sel);
    },

    selectWire(id) {
      const w = this.wireState[id] || {};
      this.app.select({ type: 'wire', id, label: WIRES[id].label, status: w.status || 'no transfer this phase', detail: w.detail || '—', pc: w.pc, rule: w.rule || WIRES[id].label });
    },

    render() {
      const m = this.model, st = this.app.state, sel = this.app.sel;
      if (!m || !st) return;
      const t = st.t, ph = st.phase;
      const afterRetire = ph !== 'start';
      const afterPrefetch = ph === 'prefetch' || ph === 'issue' || ph === 'account';
      const afterIssue = ph === 'issue' || ph === 'account';
      const ev = st.events;
      const o = st.outcome;

      // wires
      const ws = {};
      const set = (id, status, detail, pc, cls) => { ws[id] = { status, detail, pc, cls }; };
      if (afterRetire) {
        for (const c of ev.commits) {
          const p = m.program[c.pc];
          const det = MTV.commitText(m, c);
          if (c.kind === 'ub' && p.op === 'Read_Host_Memory') set('dma_ub', 'accepted: commit at retire', det, c.pc, 'com');
          if (c.kind === 'ub' && p.op === 'Activate') set('act_ub', 'accepted: commit at retire', det, c.pc, 'com');
          if (c.kind === 'host') set('dma_host', 'accepted: commit at retire', det, c.pc, 'com');
          if (c.kind === 'acc') set('mxu_acc', 'accepted: commit at retire', det, c.pc, 'com');
          if (c.kind === 'plane') set('fifo_mxu', 'accepted: pop and plane load at retire', det + ' (DDR ' + hex((ev.wloads.find((w) => w.pc === c.pc) || {}).ddr || 0) + ')', c.pc, 'com');
        }
      }
      if (afterPrefetch) {
        for (const p of ev.prefetches) {
          set('ddr_pref', 'tile copied from DDR at push', 'DDR ' + hex(p.ddr) + ' for pc ' + p.for_pc, p.for_pc, 'pref');
          set('pref_fifo', 'pushed into the FIFO', 'slot ' + (p.occ - 1) + ', ready at ' + p.ready, p.for_pc, 'pref');
        }
      }
      if (afterIssue) {
        for (const r of ev.reads) {
          const p = m.program[r.pc];
          const det = MTV.readText(m, r);
          if (r.kind === 'host') set('host_dma', 'accepted: read at issue', det, r.pc, 'acc');
          if (r.kind === 'ub' && p.op === 'Write_Host_Memory') set('ub_dma', 'accepted: read at issue', det, r.pc, 'acc');
          if (r.kind === 'ub' && p.op === 'MatMul') set('ub_mxu', 'accepted: read at issue', det, r.pc, 'acc');
          if (r.kind === 'acc' && p.op === 'MatMul') set('acc_mxu', 'accepted: read at issue (accumulate)', det, r.pc, 'acc');
          if (r.kind === 'acc' && p.op === 'Activate') set('acc_act', 'accepted: read at issue', det, r.pc, 'acc');
        }
        if (o.kind === 'stall' && m.program[o.pc]) {
          const p = m.program[o.pc];
          const req = (id) => { if (!ws[id]) set(id, 'requested: stalled on ' + o.reason, instrLabel(m, o.pc) + ' needs this path', o.pc, 'req'); };
          if (p.op === 'MatMul') { req('ub_mxu'); if (p.fields.accumulate) req('acc_mxu'); req('mxu_acc'); }
          if (p.op === 'Read_Host_Memory') { req('host_dma'); req('dma_ub'); }
          if (p.op === 'Write_Host_Memory') { req('ub_dma'); req('dma_host'); }
          if (p.op === 'Activate') { req('acc_act'); req('act_ub'); }
          if (p.op === 'Read_Weights') req('fifo_mxu');
        }
      }
      this.wireState = ws;
      for (const id of Object.keys(WIRES)) {
        const w = ws[id];
        const selected = sel && sel.type === 'wire' && sel.id === id;
        this.cls('wire-' + id, 'wire' + (w ? ' ' + w.cls : '') + (selected ? ' selected' : ''));
      }

      // blocks
      const unitBlk = { DMA: 'dma', WEIGHT: 'fifo', MXU: 'mxu', ACT: 'act' };
      const retiredUnits = new Set(afterRetire ? ev.retires.map((r) => r.unit) : []);
      const stalledUnit = afterIssue && o.kind === 'stall' && m.program[o.pc] ? MTV.UNIT_OF_OP[m.program[o.pc].op] : null;
      for (const [u, id] of Object.entries(unitBlk)) {
        let c = 'blk';
        if (st.units[u]) c += ' active';
        if (retiredUnits.has(u)) c += ' commit';
        if (stalledUnit === u) c += ' stalled';
        if (sel && sel.type === 'unit' && sel.unit === u) c += ' selected';
        if (sel && sel.type === 'fifo' && u === 'WEIGHT') c += ' selected';
        this.cls('blk-' + id, c);
      }
      this.cls('blk-seq', 'blk' + (sel && sel.type === 'seq' ? ' selected' : '') + (afterIssue && o.kind === 'stall' ? ' stalled' : ''));
      this.cls('blk-acc', 'blk' + (sel && sel.type === 'bank' ? ' selected' : ''));
      this.cls('blk-ub', 'blk' + (afterRetire && ev.commits.some((c) => c.kind === 'ub') ? ' commit' : ''));
      this.cls('blk-host', 'blk mem' + (afterRetire && ev.commits.some((c) => c.kind === 'host') ? ' commit' : ''));
      this.cls('blk-ddr', 'blk mem');
      this.cls('blk-prog', 'blk mem');
      this.cls('blk-pref', 'blk' + (afterPrefetch && ev.prefetches.length ? ' active' : ''));

      // Sequencer text
      const p = m.program[o.pc];
      this.t('seq-l1', p ? 'pc ' + o.pc + '  ' + p.asm.replace(/\s+/g, ' ').replace(/0x0000/g, '0x').slice(0, 44) : 'pc ' + o.pc + ' (past end of program)');
      if (!afterIssue) this.t('seq-l2', 'issue not yet attempted in this phase');
      else if (o.kind === 'issued') this.t('seq-l2', 'issued on ' + (p ? p.unit : '?') + (p && p.first_attempt < t ? ' after waiting since ' + fmt(p.first_attempt) : ''));
      else if (o.kind === 'halt') this.t('seq-l2', 'Halt issued: run ends');
      else if (o.kind === 'trap') this.t('seq-l2', 'trap');
      else this.t('seq-l2', 'stalled: ' + o.reason + (o.blockerPc !== null ? ' (pc ' + o.blockerPc + ' on ' + o.blockerUnit + ')' : ''));
      this.t('seq-l3', afterIssue ? (o.idle === 'busy' ? 'array busy this cycle' : 'array idle → charged to ' + o.idle) : 'retired so far: ' + m.retires.filter((r) => r.cycle < t || (afterRetire && r.cycle === t)).length);
      MTV.UNITS.forEach((u, i) => {
        const x = st.units[u];
        this.t('sb-' + i, u.padEnd(7) + (x ? 'pc ' + String(x.pc).padStart(3) + ' ' + MTV.OP_SHORT[m.program[x.pc].op].padEnd(11) + (x.issue + '→' + x.done).padEnd(12) + shortRes(m.program[x.pc].res).slice(0, 18) : 'idle'));
      });
      this.t('prog-l1', m.program.length + ' instructions × 6 words · ' + m.workload.name);

      // Host
      const hostC = ev.commits.filter((c) => c.kind === 'host');
      this.t('host-l1', fmt(m.cfg.host_bytes) + ' B · A packed at ' + hex(m.placement ? m.placement.a_host : 0) + ' · C at ' + hex(m.placement ? m.placement.c_host : 0x8000));
      this.t('host-l2', afterRetire && hostC.length ? 'commit: ' + MTV.commitText(m, hostC[0]) : (afterIssue && ev.reads.some((r) => r.kind === 'host') ? 'read: ' + MTV.readText(m, ev.reads.find((r) => r.kind === 'host')) : 'no traffic this phase'));
      this.t('host-l3', 'written only by Write_Host retire');
      // DMA
      const dma = st.units.DMA;
      if (dma) {
        const pd = m.program[dma.pc];
        const el = Math.min(dma.done - dma.issue, (afterIssue ? t + 1 : t) - dma.issue);
        this.t('dma-l1', 'pc ' + dma.pc + ' ' + MTV.OP_SHORT[pd.op] + ' ' + (tileLabel(pd) || ''));
        this.t('dma-l2', 'cycle ' + el + ' of ' + (dma.done - dma.issue) + ' (nominal)');
        this.t('dma-l3', fmt(pd.fields.bytes) + ' B, ' + m.cfg.dma_bytes_per_cycle + ' B/cycle; commits at ' + dma.done);
      } else { this.t('dma-l1', 'idle'); this.t('dma-l2', 'one transfer in flight at a time'); this.t('dma-l3', 'bytes read at issue, written at retire'); }
      // UB
      this.t('ub-l1', 'int8 · ' + fmt(m.cfg.ub_bytes) + ' B · ' + m.cfg.ub_banks + ' banks (addr % ' + m.cfg.ub_banks + ')');
      this.t('ub-l2', 'streams: ' + st.ubStreams.readers + ' read, ' + st.ubStreams.writers + ' write of ' + m.cfg.ub_banks);
      const regs = m.ubRegions.slice(0, 4);
      regs.forEach((r, i) => {
        let owner = '';
        for (const u of MTV.UNITS) {
          const x = st.units[u];
          if (!x) continue;
          const res = m.program[x.pc].res;
          if (!res) continue;
          if (res.ub_read[0] < r.addr + r.len && r.addr < res.ub_read[1]) owner += ' R:pc' + x.pc;
          if (res.ub_write[0] < r.addr + r.len && r.addr < res.ub_write[1]) owner += ' W:pc' + x.pc;
        }
        this.t('ub-r' + i, r.label.padEnd(20) + (owner || ' free'));
      });
      for (let i = regs.length; i < 4; i++) this.t('ub-r' + i, '');
      // DDR / prefetcher
      this.t('ddr-l1', fmt(m.cfg.weight_bytes) + ' B · read-only to the ISA');
      this.t('ddr-l2', afterPrefetch && ev.prefetches.length ? 'copied tile at ' + hex(ev.prefetches[0].ddr) : 'B packed at ' + hex(m.placement ? m.placement.b_ddr : 0));
      const nextPush = m.prefetches.find((pp) => pp.cycle > t);
      this.t('pref-l1', afterPrefetch && ev.prefetches.length ? 'pushed ' + ev.prefetches.length + ' tile(s) this cycle' : (st.fifo.length >= m.cfg.weight_fifo_depth ? 'idle: FIFO full' : nextPush ? 'idle' : 'idle: nothing ahead'));
      this.t('pref-l2', 'latency ' + m.cfg.ddr_tile_latency + ' cycles/tile · depth ' + m.cfg.weight_fifo_depth);
      // FIFO
      this.t('fifo-l1', st.fifo.length + '/' + m.cfg.weight_fifo_depth + ' occupied · ' + st.fifo.filter((e) => e.ready).length + ' ready');
      for (let i = 0; i < 6; i++) {
        const e = st.fifo[i];
        const slot = this.svg.querySelector('#fifo-slot' + i);
        if (i >= m.cfg.weight_fifo_depth) { slot.setAttribute('visibility', 'hidden'); this.t('fifo-s' + i, ''); continue; }
        slot.setAttribute('visibility', 'visible');
        slot.setAttribute('class', 'slot' + (e ? (e.ready ? ' ready' : ' inflight') : '') + (sel && sel.type === 'fifo' && sel.slot === i ? ' selected' : ''));
        this.t('fifo-s' + i, e ? 'slot ' + i + ' DDR ' + hex(e.push.ddr) + ' ' + (tileLabel(m.program[e.push.for_pc]) || '') + ' pc ' + e.push.for_pc + (e.ready ? ' ready' : ' →' + e.push.ready) : 'slot ' + i + ' empty');
      }
      // MXU
      const mx = st.mxu, mxu = st.units.MXU;
      if (mx.mm && mx.step >= 0) {
        const mm = mx.mm;
        const pm = m.program[mm.pc];
        this.t('mxu-l1', 'busy · pc ' + mm.pc + ' ' + (tileLabel(pm) || '') + ' · step ' + mx.step + ' of ' + mm.steps);
        this.t('mxu-r0', 'step ' + mx.step + ' / ' + mm.steps);
        this.t('mxu-r1', mx.step < mm.len ? 'streaming: row ' + mx.step + ' in' : 'fill/drain');
        this.t('mxu-r2', 'plane ' + mm.plane + ' active' + (mm.switched && mx.step === 0 ? ' (switched)' : ''));
        this.t('mxu-r3', 'A ' + (pm.tile ? 'tile (' + pm.tile.m + ',' + pm.tile.k + ')' : hex(pm.fields.ub_addr)) + ' · ' + (pm.tile ? 'B tile (' + pm.tile.k + ',' + pm.tile.n + ')' : ''));
        this.t('mxu-r4', 'bank ' + mm.bank + (mm.accumulate ? ' accumulate' : ' overwrite'));
        this.t('mxu-r5', 'complete output rows: ' + Math.max(0, Math.min(mm.len, mx.step + 3 - 2 * m.dim)) + ' of ' + mm.len);
        this.t('mxu-r6', 'retires at ' + fmt(mm.retire));
      } else {
        this.t('mxu-l1', mxu ? 'busy · pc ' + mxu.pc + ' (issued this cycle; step 0 after issue)' : 'idle');
        this.t('mxu-r0', 'grid: all zeros');
        this.t('mxu-r1', 'plane ' + st.planes.active + ' active' + (st.planes.pending ? ', switch pending' : ''));
        const tl = st.planes.tiles;
        this.t('mxu-r2', 'p0: ' + (tl[0] ? (tileLabel(m.program[tl[0].pc]) || hex(tl[0].ddr)) : '—'));
        this.t('mxu-r3', 'p1: ' + (tl[1] ? (tileLabel(m.program[tl[1].pc]) || hex(tl[1].ddr)) : '—'));
        const next = m.unitIv.MXU.find((iv) => iv.issue > t);
        this.t('mxu-r4', next ? 'next: pc ' + next.pc + ' at ' + fmt(next.issue) : 'no more MatMuls');
        this.t('mxu-r5', ''); this.t('mxu-r6', '');
      }
      this.t('mxu-l2', m.dim + '×' + m.dim + ' PEs, weight-stationary · click the grid to watch each step (Systolic array tab, key 2)');
      this.drawMini(m, st);
      // Banks
      for (let b = 0; b < 4; b++) {
        const el = this.svg.querySelector('#bank-' + b);
        if (b >= m.cfg.acc_banks) { el.setAttribute('visibility', 'hidden'); this.t('bank-t' + b, ''); this.t('bank-u' + b, ''); this.t('bank-v' + b, ''); continue; }
        el.setAttribute('visibility', 'visible');
        let w = null, r = null;
        for (const u of MTV.UNITS) { const x = st.units[u]; if (!x) continue; const res = m.program[x.pc].res; if (!res) continue; if (res.acc_write === b) w = x; if (res.acc_read === b && m.program[x.pc].op === 'Activate') r = x; }
        const committed = afterRetire && ev.commits.some((c) => c.kind === 'acc' && c.addr === b);
        el.setAttribute('class', 'slot' + (w ? ' inflight' : r ? ' ready' : '') + (sel && sel.type === 'bank' && sel.bank === b ? ' selected' : ''));
        this.t('bank-t' + b, 'bank ' + b + (committed ? ' ✓' : ''));
        this.t('bank-u' + b, w ? 'W: pc ' + w.pc + (m.program[w.pc].fields.accumulate ? ' (+)' : '') : r ? 'R: pc ' + r.pc : 'free');
        const last = [...m.commits].reverse().find((c) => c.kind === 'acc' && c.addr === b && (c.cycle < t || (afterRetire && c.cycle === t)));
        this.t('bank-v' + b, last ? 'last pc ' + last.pc + '@' + last.cycle : 'zeros');
      }
      // ACT
      const act = st.units.ACT;
      if (act) {
        const pa = m.program[act.pc];
        const el = Math.min(act.done - act.issue, (afterIssue ? t + 1 : t) - act.issue);
        this.t('act-l1', 'pc ' + act.pc + ' ' + (tileLabel(pa) || '') + ' bank ' + pa.fields.acc_bank + ' → UB ' + hex(pa.fields.ub_addr));
        this.t('act-l2', 'cycle ' + el + ' of ' + (act.done - act.issue) + ' (nominal: whole tile computed at issue)');
        this.t('act-l3', '+' + pa.fields.bias + ' · ×' + pa.fields.multiplier + ' · ≫' + pa.fields.shift + ' round · clamp · ' + pa.fields.act + (pa.fields.pool !== 'none' ? ' · ' + pa.fields.pool : ''));
      } else { this.t('act-l1', 'idle'); this.t('act-l2', 'len·dim + ' + m.cfg.act_pipeline_depth + ' cycles per Activate'); this.t('act-l3', 'bank read at issue, int8 tile committed at retire'); }
      this.t('dg-note', 'phase: ' + ph + ' · cycle ' + fmt(t) + ' of ' + fmt(m.cycles) + ' · wires show only transfers recorded for this phase');
    },

    drawMini(m, st) {
      const cv = this.mini;
      if (!cv) return;
      const ctx = cv.getContext('2d');
      const dim = m.dim;
      const W = cv.width;
      ctx.clearRect(0, 0, W, W);
      const mx = st.mxu;
      const cell = W / dim;
      const s = mx.step;
      for (let k = 0; k < dim; k++) {
        for (let c = 0; c < dim; c++) {
          let col = 'rgba(128,128,128,0.15)';
          if (mx.mm && s >= 0) {
            const r = s - k - c;
            if (r >= 0 && r < mx.mm.len) col = 'rgba(30,127,110,0.85)';
            else if (r < 0) col = 'rgba(180,86,42,0.25)';
            else col = 'rgba(107,91,181,0.35)';
          }
          ctx.fillStyle = col;
          ctx.fillRect(c * cell, k * cell, Math.ceil(cell), Math.ceil(cell));
        }
      }
    },
  };

  function shortRes(res) {
    if (!res) return '';
    const p = [];
    if (res.ub_read[1] > res.ub_read[0]) p.push('r' + hex(res.ub_read[0], 1));
    if (res.ub_write[1] > res.ub_write[0]) p.push('w' + hex(res.ub_write[0], 1));
    if (res.acc_read !== null) p.push('b' + res.acc_read + 'r');
    if (res.acc_write !== null) p.push('b' + res.acc_write + 'w');
    if (res.reads_weights) p.push('W');
    if (res.writes_weights) p.push('W!');
    return p.join(' ');
  }

  MTV.views.diagram = view;
})();
