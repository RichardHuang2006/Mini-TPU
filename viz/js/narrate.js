// "What happened this cycle?": fixed templates walked in the run loop's order.
// No line is generated without a backing record in the trace.
(function () {
  'use strict';
  const MTV = (window.MTV = window.MTV || {});

  const hex = (v, w) => '0x' + (v >>> 0).toString(16).toUpperCase().padStart(w || 4, '0');
  const fmt = (n) => (n === null || n === undefined) ? '—' : n.toLocaleString('en-US');
  const OP_SHORT = {
    Read_Host_Memory: 'Read_Host', Write_Host_Memory: 'Write_Host', Read_Weights: 'Read_Weights',
    MatMul: 'MatMul', Activate: 'Activate', Sync: 'Sync', Nop: 'Nop', Halt: 'Halt',
  };

  function tileLabel(p) {
    if (!p || !p.tile) return '';
    const t = p.tile;
    if (t.kind === 'A') return 'A(' + t.m + ',' + t.k + ')';
    if (t.kind === 'B') return 'B(' + t.k + ',' + t.n + ')';
    if (t.kind === 'M') return 'C(' + t.m + ',' + t.n + ') K-tile ' + t.k;
    return 'C(' + t.m + ',' + t.n + ')';
  }

  function instrLabel(model, pc) {
    const p = model.program[pc];
    if (!p) return 'pc ' + pc;
    const tl = tileLabel(p);
    return 'pc ' + pc + ' ' + OP_SHORT[p.op] + (tl ? ' ' + tl : '');
  }

  function commitText(model, c) {
    const dim = model.dim;
    switch (c.kind) {
      case 'ub': return fmt(c.len) + ' B to UB ' + hex(c.addr) + '–' + hex(c.addr + c.len - 1);
      case 'host': return fmt(c.len) + ' B to host ' + hex(c.addr) + '–' + hex(c.addr + c.len - 1);
      case 'acc': return c.rows + '×' + dim + ' int32 (' + fmt(c.len) + ' B) to accumulator bank ' + c.addr;
      case 'plane': return 'weight tile into plane ' + c.addr;
      default: return '?';
    }
  }

  function readText(model, r) {
    const dim = model.dim;
    switch (r.kind) {
      case 'ub': return (r.rows > 1 ? r.rows + '×' + r.cols + ' int8 ' : '') + fmt(r.len) + ' B from UB ' + hex(r.addr) + '–' + hex(r.addr + r.len - 1);
      case 'host': return fmt(r.len) + ' B from host ' + hex(r.addr) + '–' + hex(r.addr + r.len - 1);
      case 'acc': return r.rows + '×' + dim + ' int32 from accumulator bank ' + r.addr;
      default: return '?';
    }
  }

  // Lines: {phase, text, src, pc?, unit?, kind}
  function narrate(model, st) {
    const t = st.t;
    const ev = st.events;
    const out = [];
    const dim = model.dim;

    // 1. Retire
    if (ev.retires.length === 0) {
      out.push({ phase: 'retire', kind: 'none', text: 'Nothing retired.', src: 'Tpu::retire_completed' });
    }
    for (const r of ev.retires) {
      const p = model.program[r.pc];
      const parts = r.commits.map((ci) => commitText(model, model.commits[ci]));
      let text = instrLabel(model, r.pc) + ' retired on ' + r.unit + (parts.length ? ': committed ' + parts.join('; ') : ' (no data to commit)') + '.';
      if (p.op === 'MatMul') text += p.fields.accumulate ? ' The rows are the bank at issue plus this tile’s landings (accumulate).' : ' The rows overwrite the bank.';
      const wl = ev.wloads.find((w) => w.pc === r.pc);
      if (wl) text += ' The FIFO’s oldest tile (DDR ' + hex(wl.ddr) + ') was popped and loaded into plane ' + wl.plane + (wl.pending ? '; the switch to it is pending until the next MatMul issues.' : '.');
      text += ' Its reservation is released.';
      out.push({ phase: 'retire', kind: 'retire', text, src: 'Tpu::finish', pc: r.pc, unit: r.unit });
    }

    // 2. Prefetch
    if (ev.prefetches.length) {
      for (const p of ev.prefetches) {
        out.push({
          phase: 'prefetch', kind: 'prefetch',
          text: 'Prefetcher pushed the tile at DDR ' + hex(p.ddr) + ' for ' + instrLabel(model, p.for_pc) + ' into FIFO slot ' + (p.occ - 1) + '; ready at cycle ' + fmt(p.ready) + ' (occupancy ' + p.occ + '/' + model.cfg.weight_fifo_depth + ').',
          src: 'Tpu::prefetch_weights', pc: p.for_pc, unit: 'WEIGHT',
        });
      }
    } else {
      const occ = model.cyc.focc[t];
      const why = occ >= model.cfg.weight_fifo_depth ? 'FIFO full' : 'no Read_Weights left to fetch ahead';
      out.push({ phase: 'prefetch', kind: 'none', text: 'Prefetcher idle: ' + why + ' (occupancy ' + occ + '/' + model.cfg.weight_fifo_depth + ').', src: 'Tpu::prefetch_weights' });
    }

    // 3. Issue / stall / trap / halt
    const o = ev.outcome;
    if (o.kind === 'issued' || o.kind === 'halt') {
      const iss = ev.issue;
      const p = model.program[iss.pc];
      const reads = iss.reads.map((ri) => readText(model, model.reads[ri]));
      let text = instrLabel(model, iss.pc) + ' issued';
      if (p.op === 'Halt') {
        text += ': the machine was quiet, so it stops here. Halt occupies no unit, spends this cycle, and ends the run (exit code ' + p.fields.code + ').';
        out.push({ phase: 'issue', kind: 'halt', text, src: 'Tpu::issue_step', pc: iss.pc, unit: 'SEQ' });
      } else {
        text += ' on ' + iss.unit;
        if (reads.length) text += ': read ' + reads.join(' and ');
        if (p.op === 'MatMul') {
          const mm = model.matmuls[iss.mm];
          text += '; ' + (mm.switched ? 'switched the weight plane to plane ' + mm.plane + ' (the tile loaded by the previous Read_Weights)' : 'kept weight plane ' + mm.plane) +
            '; ran all ' + mm.steps + ' array steps and staged ' + mm.len + '×' + dim + ' int32 rows for bank ' + mm.bank + (mm.accumulate ? ' (accumulate)' : ' (overwrite)') +
            '. Occupies MXU until cycle ' + fmt(iss.done) + ' (' + iss.duration + ' cycles = len ' + mm.len + ' + 2·dim − 1).';
        } else if (p.op === 'Activate') {
          const a = model.activates[iss.act];
          const f = p.fields;
          text += '; computed the whole ' + a.out_rows + '×' + a.out_cols + ' int8 tile (bias ' + f.bias + ', ×' + f.multiplier + ', ≫' + f.shift + ' round half away, clamp, ' + f.act + (f.pool !== 'none' ? ', pool ' + f.pool : '') +
            ') and staged it for UB ' + hex(f.ub_addr) + '. Occupies ACT until cycle ' + fmt(iss.done) + ' (' + iss.duration + ' cycles = len·dim + ' + model.cfg.act_pipeline_depth + '; the element order is not modeled).';
        } else if (p.op === 'Read_Host_Memory') {
          text += '; staged them for UB ' + hex(p.fields.ub_addr) + '. Occupies DMA until cycle ' + fmt(iss.done) + ' (' + iss.duration + ' cycles = ⌈' + p.fields.bytes + ' / ' + model.cfg.dma_bytes_per_cycle + '⌉; no per-cycle byte movement is modeled).';
        } else if (p.op === 'Write_Host_Memory') {
          text += '; staged them for host ' + hex(p.fields.host_addr) + '. Occupies DMA until cycle ' + fmt(iss.done) + ' (' + iss.duration + ' cycles).';
        } else if (p.op === 'Read_Weights') {
          text += ': a tile is ready in the FIFO; it is popped and loaded into the shadow plane when this retires at cycle ' + fmt(iss.done) + '.';
        } else if (p.op === 'Sync') {
          text += ': the machine was quiet; the accumulator banks were snapshotted. Occupies SEQ for 1 cycle.';
        } else {
          text += '. Occupies ' + iss.unit + ' for ' + iss.duration + ' cycle(s).';
        }
        if (p.first_attempt !== null && p.first_attempt < iss.cycle) {
          text += ' It had waited since cycle ' + fmt(p.first_attempt) + ' (' + fmt(iss.cycle - p.first_attempt) + ' cycles).';
        }
        out.push({ phase: 'issue', kind: 'issue', text, src: 'Tpu::issue_step → Tpu::execute', pc: iss.pc, unit: iss.unit });
      }
    } else if (o.kind === 'stall') {
      const pc = o.pc;
      const p = model.program[pc];
      const res = p ? p.res : null;
      let text = (p ? instrLabel(model, pc) : 'pc ' + pc) + ' cannot issue: ' + o.reason;
      const b = o.blockerPc !== null ? model.program[o.blockerPc] : null;
      const bIss = b ? model.issueByPc.get(b.pc) : null;
      const until = bIss ? ' (retires at cycle ' + fmt(bIss.done) + ')' : '';
      switch (o.reason) {
        case 'unit_busy': text += ' — ' + o.blockerUnit + ' still holds ' + instrLabel(model, o.blockerPc) + until + '.'; break;
        case 'ub_raw': text += ' — it reads UB [' + hex(res.ub_read[0]) + ', ' + hex(res.ub_read[1]) + ') which ' + instrLabel(model, o.blockerPc) + ' is still writing' + until + '.'; break;
        case 'ub_war': text += ' — it writes UB [' + hex(res.ub_write[0]) + ', ' + hex(res.ub_write[1]) + ') which ' + instrLabel(model, o.blockerPc) + ' is still reading' + until + '.'; break;
        case 'ub_waw': text += ' — it writes UB [' + hex(res.ub_write[0]) + ', ' + hex(res.ub_write[1]) + ') which ' + instrLabel(model, o.blockerPc) + ' is also writing' + until + '.'; break;
        case 'accum_hazard': text += ' — accumulator bank ' + (res.acc_read !== null ? res.acc_read : res.acc_write) + ' is reserved by ' + instrLabel(model, o.blockerPc) + until + '.'; break;
        case 'weight_stall': text += ' — the weight tile is reserved by ' + instrLabel(model, o.blockerPc) + until + '.'; break;
        case 'weight_fifo_empty': {
          const f = st.fifo[0];
          text += ' — no weight tile has arrived from DDR yet' + (f ? '; the oldest (DDR ' + hex(f.push.ddr) + ') is ready at cycle ' + fmt(f.push.ready) : '') + '.';
          break;
        }
        case 'drain': text += ' — it needs the machine quiet; ' + (o.blockerPc !== null ? instrLabel(model, o.blockerPc) + ' retires last' + until : 'work is still in flight') + '.'; break;
        case 'ub_bank_conflict': text += ' — the Unified Buffer already has as many same-direction streams in flight as it has banks (' + model.cfg.ub_banks + ').'; break;
        default: text += '.';
      }
      text += ' The counter that moved is StallStats::' + o.reason + '.';
      out.push({ phase: 'issue', kind: 'stall', text, src: o.reason === 'unit_busy' || o.reason === 'drain' || o.reason === 'weight_fifo_empty' ? 'Tpu::issue_step' : o.reason === 'ub_bank_conflict' ? 'Tpu::ub_port_available' : 'Tpu::interlocked', pc, unit: o.blockerUnit });
    } else if (o.kind === 'trap') {
      out.push({ phase: 'issue', kind: 'trap', text: 'Trap at pc ' + o.pc + ': ' + (model.m.traps[0] ? model.m.traps[0].reason : 'illegal or out-of-range instruction') + '. The run ends; no stall counter moves.', src: 'Tpu::issue_step', pc: o.pc });
    }

    // 4. Array step
    const mx = st.mxu;
    if (mx.mm && mx.step >= 0) {
      const mm = mx.mm;
      const s = mx.step;
      const rowIn = s < mm.len ? 'row ' + s + ' of A entered the left edge' : 'no new row (draining)';
      const landed = [];
      for (let c = 0; c < dim; c++) { const r = s + 1 - dim - c; if (r >= 0 && r < mm.len) landed.push(r); }
      let text = 'Array step ' + s + ' of ' + mm.steps + ' for ' + instrLabel(model, mm.pc) + ': ' + rowIn;
      if (landed.length) {
        text += '; ' + landed.length + ' partial sum' + (landed.length > 1 ? 's' : '') + ' left the bottom edge (rows ' + landed[0] + '…' + landed[landed.length - 1] + ')';
        const complete = s + 2 - 2 * dim;
        if (complete >= 0 && complete < mm.len) text += '; output row ' + complete + ' is now complete';
      } else {
        text += '; nothing landed yet (fill)';
      }
      text += '. Landed values sit in the staged rows until retire at cycle ' + fmt(mm.retire) + '.';
      out.push({ phase: 'issue', kind: 'step', text, src: 'Mxu::matmul', pc: mm.pc, unit: 'MXU' });
    }

    // 5. Account
    const idle = o.idle;
    let acct;
    if (idle === 'busy') acct = 'Charged to array_busy: the MXU slot is active after this cycle’s issue.';
    else {
      const rule = {
        weights: 'a weight-path counter (weight_stall, weight_fifo_full or weight_fifo_empty) moved this cycle',
        bank: 'ub_bank_conflict moved this cycle',
        accum: 'accum_hazard moved this cycle',
        dma: 'no named counter moved and the DMA unit is active',
        act: 'no named counter moved, the DMA unit is idle and the ACT unit is active',
        other: 'no named counter moved and neither DMA nor ACT is active',
      }[idle];
      acct = 'Array idle; charged to the “' + idle + '” bucket because ' + rule + ' (the attribution order in charge_idle_cycle, not a root cause).';
    }
    out.push({ phase: 'account', kind: 'account', text: acct, src: 'Tpu::run → Tpu::charge_idle_cycle' });
    return out;
  }

  // ---- The cycle strip: five fixed lines, one per run-loop step ----
  // Each line: {phase, kind, text, full, action, state}. text is short; full is
  // the untruncated form for a title attribute. state is done | now | pending
  // relative to st.phase, and the array line follows the issue step.
  const clip = (s, n) => (s.length > n ? s.slice(0, n - 1) + '…' : s);
  const shortTile = (p) => tileLabel(p).replace(' K-tile ', ' K');
  const instrShort = (model, pc) => {
    const p = model.program[pc];
    if (!p) return 'pc ' + pc;
    const tl = shortTile(p);
    return 'pc ' + pc + ' ' + OP_SHORT[p.op] + (tl ? ' ' + tl : '');
  };
  function commitShort(model, c) {
    switch (c.kind) {
      case 'ub': return 'UB ' + hex(c.addr) + ' ←' + fmt(c.len) + ' B';
      case 'host': return 'host ' + hex(c.addr) + ' ←' + fmt(c.len) + ' B';
      case 'acc': return 'bank ' + c.addr + ' ←' + c.rows + ' rows';
      case 'plane': return 'plane ' + c.addr + ' ← tile';
      default: return '?';
    }
  }

  function brief(model, st) {
    const t = st.t;
    const ev = st.events;
    const o = ev.outcome;
    const dim = model.dim;
    const LIMIT = 52;
    const cur = PHASE_INDEX[st.phase];
    const stateOf = (i) => (i < cur ? 'done' : i === cur ? 'now' : 'pending');
    const line = (phase, idx, kind, full, action) => ({ phase, kind, full, text: clip(full, LIMIT), action, state: stateOf(idx) });
    const out = [];

    // 1. retire
    if (ev.retires.length === 0) out.push(line('retire', 1, 'none', 'nothing retired', 'sel:cycle'));
    else {
      const r = ev.retires[0];
      const commits = r.commits.map((ci) => model.commits[ci]);
      const wl = ev.wloads.find((w) => w.pc === r.pc);
      let full = instrShort(model, r.pc) + ' → ' + (commits.length ? commits.map((c) => commitShort(model, c)).join(', ') : wl ? 'plane ' + wl.plane : 'done');
      if (ev.retires.length > 1) full += ' · +' + (ev.retires.length - 1) + ' more';
      out.push(line('retire', 1, 'retire', full, 'instr:' + r.pc));
    }

    // 2. prefetch
    if (ev.prefetches.length) {
      const p = ev.prefetches[0];
      const tl = shortTile(model.program[p.for_pc]);
      let full = 'push ' + (tl || 'DDR ' + hex(p.ddr)) + ' for pc ' + p.for_pc + ' → slot ' + (p.occ - 1) + ' · ready ' + fmt(p.ready);
      if (ev.prefetches.length > 1) full += ' · +' + (ev.prefetches.length - 1);
      out.push(line('prefetch', 2, 'prefetch', full, 'sel:fifo'));
    } else {
      const occ = model.cyc.focc[t];
      const why = occ >= model.cfg.weight_fifo_depth ? 'FIFO full ' + occ + '/' + model.cfg.weight_fifo_depth : 'nothing ahead · FIFO ' + occ + '/' + model.cfg.weight_fifo_depth;
      out.push(line('prefetch', 2, 'none', 'idle · ' + why, 'sel:fifo'));
    }

    // 3. issue
    if (o.kind === 'issued' || o.kind === 'halt') {
      const iss = ev.issue;
      const p = model.program[iss.pc];
      let full;
      if (p.op === 'Halt') full = 'pc ' + iss.pc + ' Halt · run ends';
      else {
        full = instrShort(model, iss.pc);
        if (p.op === 'MatMul') { const mm = model.matmuls[iss.mm]; full += ' · plane ' + mm.plane + (mm.accumulate ? ' · +bank ' + mm.bank : ' · bank ' + mm.bank); }
        else if (p.op === 'Activate') full += ' · ' + p.fields.act + ' → UB ' + hex(p.fields.ub_addr);
        else if (p.op === 'Read_Host_Memory' || p.op === 'Write_Host_Memory') full += ' · ' + fmt(p.fields.bytes) + ' B';
        if (p.first_attempt !== null && p.first_attempt < iss.cycle) full += ' · waited ' + fmt(iss.cycle - p.first_attempt);
        full += ' · ' + fmt(iss.duration) + ' cyc';
      }
      out.push(line('issue', 3, p.op === 'Halt' ? 'halt' : 'issue', full, 'instr:' + iss.pc));
    } else if (o.kind === 'stall') {
      const b = o.blockerPc !== null ? model.program[o.blockerPc] : null;
      const bIss = b ? model.issueByPc.get(b.pc) : null;
      let full = instrShort(model, o.pc) + ' · ' + o.reason;
      if (o.blockerPc !== null) full += ' ← pc ' + o.blockerPc + (bIss ? ' →' + fmt(bIss.done) : '');
      else if (o.reason === 'weight_fifo_empty' && st.fifo[0]) full += ' · ready ' + fmt(st.fifo[0].push.ready);
      out.push(line('issue', 3, 'stall', full, 'sel:seq'));
    } else {
      out.push(line('issue', 3, 'trap', 'trap at pc ' + o.pc, 'sel:seq'));
    }

    // 4. array step
    const mx = st.mxu;
    if (mx.mm && mx.step >= 0) {
      const mm = mx.mm, s = mx.step;
      let nIn = 0;
      for (let k = 0; k < dim; k++) if (s - k >= 0 && s - k < mm.len) nIn++;
      const landed = [];
      for (let c = 0; c < dim; c++) { const r = s + 1 - dim - c; if (r >= 0 && r < mm.len) landed.push(r); }
      const done = Math.max(0, Math.min(mm.len, s + 3 - 2 * dim));
      const full = (cur < 3 ? 'after step ' : 'step ') + s + '/' + mm.steps + ' · in ×' + nIn +
        ' · land ' + (landed.length ? 'r' + landed[landed.length - 1] + (landed.length > 1 ? '…r' + landed[0] : '') : '—') +
        ' · done ' + done + '/' + mm.len;
      out.push({ phase: 'array', kind: 'step', full, text: clip(full, LIMIT), action: 'view:array', state: cur < 3 ? 'done' : stateOf(3) });
    } else {
      const next = model.unitIv.MXU.find((iv) => iv.issue > t);
      const busy = st.units.MXU;
      let full;
      if (busy) full = 'pc ' + busy.pc + ' issues this cycle · step 0 after issue';
      else if (ev.retires.some((r) => r.unit === 'MXU') && cur >= 1) full = 'retired · grid empty';
      else full = 'idle · ' + (next ? 'next pc ' + next.pc + ' at ' + fmt(next.issue) : 'no more MatMuls');
      out.push({ phase: 'array', kind: 'idle', full, text: clip(full, LIMIT), action: 'view:array', state: cur < 3 ? 'done' : stateOf(3) });
    }

    // 5. account
    if (o.idle === 'busy') out.push(line('account', 4, 'busy', 'busy · MXU active', 'sel:cycle'));
    else {
      const ib = model.idleBlockerAt(t);
      const who = ib.pc !== null && ib.via !== 'stalled' ? ' · ' + instrShort(model, ib.pc) : '';
      out.push(line('account', 4, 'idle', 'idle → ' + o.idle + who, 'sel:cycle'));
    }
    return out;
  }
  const PHASE_INDEX = { start: 0, retire: 1, prefetch: 2, issue: 3, account: 4 };

  MTV.narrate = narrate;
  MTV.brief = brief;
  MTV.instrShort = instrShort;
  MTV.commitShort = commitShort;
  MTV.hex = hex;
  MTV.fmt = fmt;
  MTV.OP_SHORT = OP_SHORT;
  MTV.tileLabel = tileLabel;
  MTV.instrLabel = instrLabel;
  MTV.commitText = commitText;
  MTV.readText = readText;
})();
