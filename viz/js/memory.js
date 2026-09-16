// The Unified Buffer, accumulator banks, host and weight memory and the weight
// planes, replayed at this cycle; commits are violet, issue-time reads green.
(function () {
  'use strict';
  const MTV = (window.MTV = window.MTV || {});
  MTV.views = MTV.views || {};
  const { hex, fmt } = MTV;

  const view = {
    init(app) {
      this.app = app;
      this.kind = 'ub';
      this.addr = 0;
      this.map = document.getElementById('mem-map');
      this.grid = document.getElementById('mem-grid');
      this.note = document.getElementById('mem-note');
      this.addrInput = document.getElementById('mem-addr');
      document.getElementById('mem-kind').addEventListener('click', (e) => {
        const b = e.target.closest('button'); if (!b) return;
        this.show(b.getAttribute('data-kind'));
      });
      this.addrInput.addEventListener('change', () => { this.addr = parseInt(this.addrInput.value, 16) || parseInt(this.addrInput.value, 10) || 0; this.render(); });
      this.grid.addEventListener('click', (e) => {
        const cell = e.target.closest('.cell'); if (!cell) return;
        this.selectCell(cell.dataset);
      });
      this.map.addEventListener('click', (e) => {
        const r = e.target.closest('.region'); if (!r) return;
        this.addr = +r.dataset.addr; this.addrInput.value = hex(this.addr); this.render();
      });
    },
    reset(model) { this.model = model; this.addr = 0; },
    show(kind) {
      this.kind = kind;
      for (const b of document.querySelectorAll('#mem-kind button')) b.classList.toggle('active', b.getAttribute('data-kind') === kind);
      this.render();
    },

    selectCell(d) {
      const m = this.model, st = this.app.state;
      const dim = m.dim;
      if (d.kind === 'acc') {
        const bank = +d.bank, row = +d.row, col = +d.col;
        const v = m.img.acc[bank * dim * dim + row * dim + col];
        this.app.select({ type: 'mem', kind: 'acc', label: 'bank ' + bank + ' [' + row + '][' + col + ']', addr: bank, row, col, addrText: 'bank ' + bank + ', row ' + row + ', col ' + col, valueText: fmt(v) + ' (int32)', extra: this.accExtra(bank, row, col) });
      } else if (d.kind === 'planes') {
        const p = +d.bank, k = +d.row, c = +d.col;
        this.app.select({ type: 'mem', kind: 'none', label: 'plane ' + p + ' PE[' + k + '][' + c + ']', addr: p, addrText: 'plane ' + p + ' (' + (st.planes.active === p ? 'active' : 'shadow') + '), PE[' + k + '][' + c + ']', valueText: String(m.img.planes[p][k * dim + c]) + ' (int8 weight)' });
      } else {
        const a = +d.addr;
        const arr = d.kind === 'ub' ? m.img.ub : d.kind === 'host' ? m.img.host : m.initDdr;
        let v = arr[a]; if (d.kind === 'host' && v > 127) v -= 256;
        const extra = [];
        if (d.kind === 'ub') extra.push(['Bank', (a % m.cfg.ub_banks) + ' (addr % ' + m.cfg.ub_banks + '; an annotation, the sequencer budgets streams, not banks)']);
        const lg = this.logical(d.kind, a);
        if (lg) extra.push(['Logical', lg]);
        this.app.select({ type: 'mem', kind: d.kind, label: (d.kind === 'ub' ? 'UB ' : d.kind === 'host' ? 'host ' : 'DDR ') + hex(a), addr: a, addrText: hex(a), valueText: String(v) + ' (int8)', extra });
      }
    },
    accExtra(bank, row, col) {
      const m = this.model, st = this.app.state;
      const out = [];
      const ti = m.program.find((p) => p.tile && p.tile.kind === 'M' && p.fields.acc_bank === bank && p.issue !== null && p.issue <= st.t && (p.retire === null || p.retire > st.t - 1));
      if (ti) out.push(['Logical', 'C[' + (ti.tile.m * m.dim + row) + '][' + (ti.tile.n * m.dim + col) + '] (tile ' + ti.tile.m + ',' + ti.tile.n + ', from pc ' + ti.pc + ')']);
      return out;
    },
    logical(kind, a) {
      const m = this.model;
      if (!m.hasLayer) return null;
      const dim = m.dim, per = dim * dim;
      if (kind === 'host') {
        const pl = m.placement;
        const aBytes = m.workload.a_bytes, cBytes = m.workload.c_bytes;
        if (a >= pl.a_host && a < pl.a_host + aBytes) { const idx = Math.floor((a - pl.a_host) / per), o = (a - pl.a_host) % per; const across = m.tilesOf(m.K); return 'A[' + (Math.floor(idx / across) * dim + Math.floor(o / dim)) + '][' + ((idx % across) * dim + o % dim) + ']'; }
        if (a >= pl.c_host && a < pl.c_host + cBytes) { const idx = Math.floor((a - pl.c_host) / per), o = (a - pl.c_host) % per; const across = m.tilesOf(m.N); return 'C[' + (Math.floor(idx / across) * dim + Math.floor(o / dim)) + '][' + ((idx % across) * dim + o % dim) + '] (packed output)'; }
      }
      if (kind === 'ddr') {
        const pl = m.placement, bBytes = m.workload.b_bytes;
        if (a >= pl.b_ddr && a < pl.b_ddr + bBytes) { const idx = Math.floor((a - pl.b_ddr) / per), o = (a - pl.b_ddr) % per; const across = m.tilesOf(m.N); return 'B[' + (Math.floor(idx / across) * dim + Math.floor(o / dim)) + '][' + ((idx % across) * dim + o % dim) + ']'; }
      }
      if (kind === 'ub') {
        const st = this.app.state;
        for (const p of m.program) {
          if (p.op === 'Read_Host_Memory' && p.tile && p.retire !== null && p.retire <= st.t && a >= p.fields.ub_addr && a < p.fields.ub_addr + p.fields.bytes) {
            const o = a - p.fields.ub_addr;
            return 'A[' + (p.tile.m * dim + Math.floor(o / dim)) + '][' + (p.tile.k * dim + o % dim) + '] (from pc ' + p.pc + ')';
          }
        }
      }
      return null;
    },

    render() {
      const m = this.model, st = this.app.state, sel = this.app.sel;
      if (!m || !st) return;
      const dim = m.dim;
      const ev = st.events;
      const afterRetire = st.phase !== 'start';
      const afterIssue = st.phase === 'issue' || st.phase === 'account';
      const changed = (kind, a) => afterRetire && ev.commits.some((c) => c.kind === kind && a >= c.addr && a < c.addr + c.len);
      const read = (kind, a) => afterIssue && ev.reads.some((r) => r.kind === kind && a >= r.addr && a < r.addr + r.len);
      let html = '';
      this.map.innerHTML = '';
      if (this.kind === 'ub') {
        this.map.innerHTML = m.ubRegions.map((r) => {
          let cls = '';
          for (const u of MTV.UNITS) { const x = st.units[u]; if (!x) continue; const res = m.program[x.pc].res; if (!res) continue; if (res.ub_read[0] < r.addr + r.len && r.addr < res.ub_read[1]) cls = 'rr'; if (res.ub_write[0] < r.addr + r.len && r.addr < res.ub_write[1]) cls = 'rw'; }
          return '<span class="region ' + cls + '" data-addr="' + r.addr + '">' + r.label + ' · ' + fmt(r.len) + ' B</span>';
        }).join('') + '<span class="muted">green = reserved for read, violet = reserved for write by an in-flight instruction</span>';
        html = this.bytes(m.img.ub, this.addr, 512, 'ub', (a) => a % m.cfg.ub_banks, changed, read, sel);
        this.note.textContent = fmt(m.cfg.ub_bytes) + ' B; showing 512 from ' + hex(this.addr);
      } else if (this.kind === 'host') {
        html = this.bytes(m.img.host, this.addr, 512, 'host', null, changed, read, sel, true);
        this.note.textContent = fmt(m.cfg.host_bytes) + ' B; showing 512 from ' + hex(this.addr);
      } else if (this.kind === 'ddr') {
        html = this.bytes(m.initDdr, this.addr, 512, 'ddr', null, () => false, () => false, sel);
        this.note.textContent = fmt(m.cfg.weight_bytes) + ' B (never written by the ISA); showing 512 from ' + hex(this.addr);
      } else if (this.kind === 'acc') {
        for (let b = 0; b < m.cfg.acc_banks; b++) {
          html += '<div><b>bank ' + b + '</b>' + this.bankOwner(b) + '</div>';
          const cm = afterRetire ? ev.commits.find((c) => c.kind === 'acc' && c.addr === b) : null;
          const rd = afterIssue ? ev.reads.find((r) => r.kind === 'acc' && r.addr === b) : null;
          for (let r = 0; r < dim; r++) {
            let line = '<span class="addr">r' + r + '</span>';
            for (let c = 0; c < dim; c++) {
              const v = m.img.acc[b * dim * dim + r * dim + c];
              let cls = 'cell';
              if (cm && r < cm.rows) cls += ' chg'; else if (rd && r < rd.rows) cls += ' rd';
              if (sel && sel.type === 'mem' && sel.kind === 'acc' && sel.addr === b && sel.row === r && sel.col === c) cls += ' sel';
              line += '<span class="' + cls + '" data-kind="acc" data-bank="' + b + '" data-row="' + r + '" data-col="' + c + '" style="width:8ch">' + v + '</span>';
            }
            html += line + '\n';
          }
          html += '\n';
        }
        this.note.textContent = m.cfg.acc_banks + ' banks × ' + dim + '×' + dim + ' int32; violet = committed this cycle, green = read at issue';
      } else if (this.kind === 'planes') {
        for (let p = 0; p < 2; p++) {
          html += '<div><b>plane ' + p + '</b> ' + (st.planes.active === p ? 'active' : 'shadow') + (st.planes.pending && st.planes.active !== p ? ' (switch pending)' : '') + '</div>';
          const cm = afterRetire ? ev.commits.find((c) => c.kind === 'plane' && c.addr === p) : null;
          for (let k = 0; k < dim; k++) {
            let line = '<span class="addr">k' + k + '</span>';
            for (let c = 0; c < dim; c++) {
              let cls = 'cell' + (cm ? ' chg' : '');
              line += '<span class="' + cls + '" data-kind="planes" data-bank="' + p + '" data-row="' + k + '" data-col="' + c + '">' + m.img.planes[p][k * dim + c] + '</span>';
            }
            html += line + '\n';
          }
          html += '\n';
        }
        this.note.textContent = 'PE[k][c] holds W[k][c]; a Read_Weights loads the shadow plane at retire';
      }
      this.grid.innerHTML = html;
    },
    bankOwner(b) {
      const m = this.model, st = this.app.state;
      for (const u of MTV.UNITS) { const x = st.units[u]; if (!x) continue; const res = m.program[x.pc].res; if (!res) continue; if (res.acc_write === b) return ' · reserved for write by pc ' + x.pc; if (res.acc_read === b) return ' · reserved for read by pc ' + x.pc; }
      return ' · free';
    },
    bytes(arr, start, count, kind, bankOf, changed, read, sel, unsigned) {
      let html = '';
      const perRow = 16;
      start = Math.max(0, Math.min(arr.length - 1, start)) & ~(perRow - 1);
      for (let a = start; a < Math.min(arr.length, start + count); a += perRow) {
        let line = '<span class="addr">' + hex(a, 5) + '</span>';
        for (let i = 0; i < perRow && a + i < arr.length; i++) {
          let v = arr[a + i]; if (unsigned && v > 127) v -= 256;
          let cls = 'cell';
          if (changed(kind, a + i)) cls += ' chg'; else if (read(kind, a + i)) cls += ' rd';
          if (sel && sel.type === 'mem' && sel.kind === kind && sel.addr === a + i) cls += ' sel';
          line += '<span class="' + cls + '" data-kind="' + kind + '" data-addr="' + (a + i) + '" title="' + hex(a + i) + (bankOf ? ' bank ' + bankOf(a + i) : '') + '">' + v + '</span>';
        }
        html += line + '\n';
      }
      return html;
    },
  };

  MTV.views.memory = view;
})();
