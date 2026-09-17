#!/usr/bin/env node
// Runs the page's container reader and cycle model under node against a trace
// container: node viz/selftest.js viz/traces/matmul_8.mtpt [more.mtpt ...]
'use strict';
const fs = require('fs');
const path = require('path');

// The browser modules are classic scripts that hang off window.MTV.
const window = { MTV: {} };
global.window = window;
for (const f of ['container.js', 'model.js', 'narrate.js']) {
  const src = fs.readFileSync(path.join(__dirname, 'js', f), 'utf8');
  new Function('window', src)(window);
}
const MTV = window.MTV;

let failures = 0;
function check(name, ok, detail) {
  console.log('  ' + (ok ? 'ok   ' : 'FAIL ') + name + (detail ? '  ' + detail : ''));
  if (!ok) failures++;
}

async function run(file) {
  const bytes = fs.readFileSync(file);
  const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const c = await MTV.Container.open(buf, path.basename(file));
  const m = new MTV.Model(c);
  console.log(m.workload.name + ': ' + m.cycles + ' cycles, ' + m.program.length + ' instructions');

  // Every cycle materializes forward and backward to the same images.
  const stride = Math.max(1, Math.floor(m.cycles / 97));
  const sig = (img) => {
    let h = 0;
    const mix = (v) => { h = (Math.imul(h, 31) + v) | 0; };
    for (let i = 0; i < img.host.length; i += 7) mix(img.host[i]);
    for (let i = 0; i < img.ub.length; i += 13) mix(img.ub[i]);
    for (let i = 0; i < img.acc.length; i += 3) mix(img.acc[i]);
    mix(img.planes[0][0]); mix(img.planes[1][0]);
    return h;
  };
  const fwd = [];
  for (let t = 0; t < m.cycles; t += stride) { m.setCycle(t, 'account'); fwd.push(sig(m.img)); }
  let same = true;
  for (let i = fwd.length - 1, t = (fwd.length - 1) * stride; i >= 0; i--, t -= stride) {
    m.setCycle(t, 'account');
    if (sig(m.img) !== fwd[i]) same = false;
  }
  check('scrub', same, 'forward and backward materialization agree at ' + fwd.length + ' cycles');

  // Halt cycle: the last row, nothing in flight afterwards.
  m.setCycle(m.cycles - 1, 'account');
  check('halt', m.state.outcome.kind === 'halt' && Object.values(m.state.units).every((u) => u === null),
        'cycle ' + (m.cycles - 1) + ' is Halt with an empty machine');

  // Narration produces at least one line per cycle and never throws.
  let lines = 0;
  for (let t = 0; t < m.cycles; t += stride) { m.setCycle(t, 'account'); lines += MTV.narrate(m, m.state).length; }
  check('narrate', lines > 0, lines + ' lines over the sampled cycles');

  // The cycle strip is always five lines in run-loop order, each short.
  {
    let bad = 0, long = 0, n = 0;
    const order = 'retire,prefetch,issue,array,account';
    for (let t = 0; t < m.cycles; t += stride) {
      for (const ph of MTV.PHASES) {
        m.setCycle(t, ph);
        const b = MTV.brief(m, m.state);
        n++;
        if (b.length !== 5 || b.map((l) => l.phase).join(',') !== order) bad++;
        for (const l of b) { if (l.text.length > 52) long++; if (!['done', 'now', 'pending'].includes(l.state) || typeof l.action !== 'string') bad++; }
        const cur = MTV.PHASES.indexOf(ph);
        if (cur >= 1 && b[0].state === 'pending') bad++;
        if (cur < 1 && b[0].state !== 'pending') bad++;
        if (cur === 4 && b[4].state !== 'now') bad++;
      }
    }
    check('cycle strip', bad === 0 && long === 0, n + ' (cycle, phase) samples, 5 lines each, longest ≤ 52 chars, ' + bad + ' shape violations');
  }

  // The rules the array view draws hold on every recorded MatMul:
  // input skew, landing step, drained final step, complete-row count.
  {
    const dim = m.dim, n = dim * dim;
    let bad = 0, cells = 0;
    for (const mm of m.matmuls) {
      const f = c.mmSync(mm.index);
      const iss = m.issueByPc.get(mm.pc);
      const rd = iss.reads.map((i) => m.reads[i]).find((r) => r.kind === 'ub');
      const A = m.readI8(rd);
      for (let s = 0; s < mm.steps; s++) for (let k = 0; k < dim; k++) {
        const r = s - k;
        const want = r >= 0 && r < mm.len ? A[r * dim + k] : 0;
        if (f.left[s * dim + k] !== want) bad++;
      }
      for (let r = 0; r < mm.len; r++) for (let cc = 0; cc < dim; cc++) {
        const s = r + dim - 1 + cc;
        cells++;
        if (s >= mm.steps || f.psum[s * n + (dim - 1) * dim + cc] !== f.land[r * dim + cc]) bad++;
      }
      for (let i = 0; i < n; i++) if (f.psum[(mm.steps - 1) * n + i] !== 0 || f.act[(mm.steps - 1) * n + i] !== 0) bad++;
      for (const s of [0, dim - 1, 2 * dim - 2, mm.steps - 1]) {
        let done = 0;
        for (let r = 0; r < mm.len; r++) if (s >= r + 2 * dim - 2) done++;
        if (done !== Math.max(0, Math.min(mm.len, s + 3 - 2 * dim))) bad++;
      }
    }
    check('array rules', bad === 0, m.matmuls.length + ' MatMuls: left[s][k] = A[s−k][k], land[r][c] = psum[r+dim−1+c][dim−1][c] (' + cells + ' cells), final step all zero, done-row count');
  }

  // Every hop obeys the release rule it claims, and every chain terminates.
  {
    let hops = 0, bad = 0, deepest = 0;
    for (const p of m.program) {
      if (p.issue === null) continue;
      const ch = m.blockerChain(p.pc);
      deepest = Math.max(deepest, ch.length);
      const end = ch[ch.length - 1];
      if (!end || !['root', 'resource'].includes(end.kind)) bad++;
      for (const h of ch) {
        hops++;
        if (h.kind === 'blocked' && m.program[h.next].retire !== h.issue) bad++;
        if (h.kind === 'in_order' && m.program[h.next].issue !== h.issue - 1) bad++;
        if (h.kind === 'loop') bad++;
      }
    }
    check('blocker chains', bad === 0, hops + ' hops over ' + m.program.length + ' chains, deepest ' + deepest + ', violations ' + bad);
  }

  // The whole run equals the recorded statistics, a sub-window a direct count.
  {
    const wc = m.windowCounts(0, m.cycles);
    const s = m.m.stats;
    let ok = wc.idle.busy === s.array_busy;
    for (const k of ['weights', 'bank', 'accum', 'dma', 'act', 'other']) ok = ok && wc.idle[k] === s.idle[k];
    for (const k of ['drain', 'unit_busy', 'weight_fifo_empty', 'ub_raw', 'ub_war', 'ub_waw', 'accum_hazard', 'weight_stall', 'ub_bank_conflict']) ok = ok && wc.outcome[k] === s.stalls[k];
    check('window counts, run', ok, 'whole-run Pareto equals RunProfile idle buckets and StallStats');
    const a = Math.floor(m.cycles / 3), b = Math.floor((2 * m.cycles) / 3);
    const sub = m.windowCounts(a, b);
    const idle = {}, out = {};
    for (let t = a; t < b; t++) {
      const i = MTV.IDLE_NAMES[m.cyc.idle[t]]; idle[i] = (idle[i] || 0) + 1;
      const o = MTV.STALL_NAMES[m.cyc.outcome[t]]; out[o] = (out[o] || 0) + 1;
    }
    let ok2 = sub.n === b - a;
    for (const k of MTV.IDLE_NAMES) ok2 = ok2 && (idle[k] || 0) === sub.idle[k];
    for (const k of MTV.STALL_NAMES) ok2 = ok2 && (out[k] || 0) === sub.outcome[k];
    ok2 = ok2 && m.matchCount({ kind: 'idle', key: 'busy' }, a, b) === sub.idle.busy && m.matchCount({ kind: 'stall', key: 'ub_raw' }, a, b) === sub.outcome.ub_raw;
    check('window counts, sub', ok2, '[' + a + ', ' + b + ') matches a direct count');
  }

  if (m.workload.name === 'matmul_128') {
    m.setCycle(289, 'start');
    check('289 start', m.state.units.MXU && m.state.units.MXU.pc === 5 && m.state.units.WEIGHT === null &&
          m.state.units.DMA === null && m.state.units.ACT === null,
          'MXU holds pc 5; other units idle');
    check('289 bank before', m.img.acc[5 * 32 + 0] === 0, 'bank 0 [5][0] is still 0 at the start of 289');
    m.setCycle(289, 'retire');
    check('289 retire', m.state.units.MXU === null && m.img.acc[5 * 32 + 0] === 71883,
          'pc 5 retired; bank 0 [5][0] = ' + m.img.acc[5 * 32 + 0]);
    m.setCycle(289, 'issue');
    check('289 issue', m.state.units.MXU && m.state.units.MXU.pc === 7 && m.state.planes.active === 0 &&
          !m.state.planes.pending && m.state.fifo.length === 4 && m.state.fifo.every((e) => e.ready),
          'pc 7 issued on plane 0, FIFO 4/4 ready');
    check('289 fifo', m.state.fifo.map((e) => e.push.ddr).join(',') === '8192,12288,1024,5120',
          'FIFO holds DDR ' + m.state.fifo.map((e) => '0x' + e.push.ddr.toString(16)).join(' '));
    m.setCycle(230, 'account');
    const mx = m.state.mxu;
    check('230 step', mx.mm && mx.mm.pc === 5 && mx.step === 36, 'pc 5 step ' + (mx.mm && mx.step));
    const pe = await c.mm(mx.mm.index);
    const dim = 32, n = dim * dim;
    const psum = (s, k, cc) => pe.psum[s * n + k * dim + cc];
    const act = (s, k, cc) => pe.act[s * n + k * dim + cc];
    check('230 PE[31][0]', psum(36, 31, 0) === 71883 && act(36, 31, 0) === -31 && pe.left[36 * dim + 31] === -31 &&
          psum(35, 30, 0) === 70457 && pe.w[31 * dim + 0] === -46,
          'psum 70457 + (-31 x -46) -> ' + psum(36, 31, 0));
    check('230 landing', pe.land[5 * dim + 0] === 71883, 'C[5][0] partial landed = ' + pe.land[5 * dim + 0]);
    const o = m.outcomeAt(1000);
    check('1000 stall', o.kind === 'stall' && o.reason === 'ub_raw' && o.pc === 21 && o.blockerUnit === 'ACT' && o.blockerPc === 12 && o.idle === 'act',
          o.reason + ' pc ' + o.pc + ' blocked by ' + o.blockerUnit + ' pc ' + o.blockerPc + ', charged ' + o.idle);
    const ce = m.cElement(5, 0);
    check('C[5][0]', ce.chain.length === 4 && ce.chain[0].pc === 5 && ce.acts.length === 1 && ce.acts[0].pc === 12 && ce.writes[0].pc === 21,
          'chain pcs ' + ce.chain.map((p) => p.pc).join(',') + '; activate pc ' + ce.acts[0].pc + '; write pc ' + ce.writes[0].pc);
    const ch21 = m.blockerChain(21);
    check('chain pc 21', ch21[0].kind === 'blocked' && ch21[0].reason === 'ub_raw' && ch21[0].next === 12 &&
          ch21[1].pc === 12 && ch21[1].reason === 'accum_hazard' && ch21[1].next === 11 &&
          ch21[ch21.length - 1].pc === 0 && ch21[ch21.length - 1].kind === 'root',
          ch21.slice(0, 3).map((h) => 'pc ' + h.pc + ' ' + h.kind + (h.reason ? ' ' + h.reason : '')).join(' → ') + ' … root pc ' + ch21[ch21.length - 1].pc + ' (' + ch21.length + ' hops)');
    const ib = m.idleBlockerAt(1000);
    check('idle blocker 1000', ib.pc === 12 && ib.via === 'blocker' && ib.o.idle === 'act', 'cycle 1000 charged to ' + ib.o.idle + ', points at pc ' + ib.pc + ' via ' + ib.via);
    // The array-view bookmarks land on the steps they claim.
    const steps = {};
    for (const [t, want] of [[194, 0], [225, 31], [256, 62], [287, 93], [288, 94]]) { m.setCycle(t, 'account'); steps[t] = m.state.mxu.mm && m.state.mxu.mm.pc === 5 ? m.state.mxu.step : -1; if (steps[t] !== want) steps.bad = true; }
    check('bookmark steps', !steps.bad, 'pc 5 steps at 194/225/256/287/288 = ' + [194, 225, 256, 287, 288].map((t) => steps[t]).join('/'));
    m.setCycle(289, 'retire');
    const b289 = MTV.brief(m, m.state);
    check('strip 289 retire', b289[0].kind === 'retire' && /pc 5 .*bank 0/.test(b289[0].full) && b289[2].state === 'pending' && b289[3].state === 'done',
          b289.map((l) => l.phase + ': ' + l.text).join(' | '));
    m.setCycle(289, 'issue');
    const b289i = MTV.brief(m, m.state);
    check('strip 289 issue', /^pc 7 MatMul .*plane 0/.test(b289i[2].full) && /^step 0\/95/.test(b289i[3].full), b289i[2].text + ' | ' + b289i[3].text);
    m.setCycle(230, 'account');
    const b230 = MTV.brief(m, m.state);
    check('strip 230', /unit_busy/.test(b230[2].full) && /^step 36\/95 .*r0…r5/.test(b230[3].full) && b230[4].kind === 'busy', b230[2].text + ' | ' + b230[3].text);
    m.setCycle(1000, 'account');
    const b1000 = MTV.brief(m, m.state);
    const nextMm = m.unitIv.MXU.find((iv) => iv.issue > 1000);
    check('strip 1000', /ub_raw ← pc 12/.test(b1000[2].full) && /^idle → act/.test(b1000[4].full) && b1000[3].full === 'idle · next pc ' + nextMm.pc + ' at ' + nextMm.issue.toLocaleString('en-US'), b1000[2].text + ' | ' + b1000[3].text + ' | ' + b1000[4].text);
    m.setCycle(m.cycles - 1, 'account');
    check('final C[5][0]', m.img.host[ce.host] === 16, 'host byte 0x' + ce.host.toString(16) + ' = ' + m.img.host[ce.host]);
  }
  if (m.workload.name === 'matmul_8') {
    m.setCycle(5, 'account');
    const o = m.outcomeAt(5);
    check('5 fifo empty', o.kind === 'stall' && o.reason === 'weight_fifo_empty' && o.pc === 2,
          o.reason + ' at cycle 5 for pc ' + o.pc + '; next ready ' + (m.state.fifo[0] && m.state.fifo[0].push.ready));
    const ch28 = m.blockerChain(28);
    const end = ch28[ch28.length - 1];
    check('chain Halt', end.pc === 2 && end.kind === 'resource' && end.reason === 'weight_fifo_empty' && end.prefetch && end.prefetch.ready === 8,
          ch28.length + ' hops from Halt to pc ' + end.pc + ' (' + end.kind + ': ' + end.reason + ', tile ready at ' + (end.prefetch && end.prefetch.ready) + ')');
    m.setCycle(20, 'issue');
    check('20 accumulate', m.state.units.MXU && m.state.units.MXU.pc === 5 && m.state.planes.active === 0,
          'pc 5 (accumulate) issued on plane 0');
    const pe = c.mmSync(1);
    check('20 PE[0][0]', pe && pe.psum[0] === pe.left[0] * pe.w[0] && pe.act[0] === pe.left[0],
          'psum: 0 + (' + pe.left[0] + ' x ' + pe.w[0] + ') -> ' + pe.psum[0]);
    m.setCycle(5, 'account');
    const b5 = MTV.brief(m, m.state);
    check('strip 5', /weight_fifo_empty/.test(b5[2].full) && b5[2].kind === 'stall', b5[2].text);
    for (const [t, want] of [[9, 0], [12, 3], [18, 9], [19, 10]]) { m.setCycle(t, 'account'); const mx = m.state.mxu; if (!(mx.mm && mx.mm.pc === 3 && mx.step === want)) failures++, console.log('  FAIL bookmark ' + t + ' step ' + (mx.mm && mx.step)); }
    check('bookmark steps', true, 'pc 3 steps at 9/12/18/19 = 0/3/9/10');
  }
}

(async () => {
  for (const f of process.argv.slice(2)) await run(f);
  console.log(failures ? 'selftest: FAILURES' : 'selftest: ALL PASS');
  process.exit(failures ? 1 : 0);
})();
