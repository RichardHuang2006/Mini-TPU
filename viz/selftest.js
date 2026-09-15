#!/usr/bin/env node
// Runs the visualizer's container reader and cycle model under node against a
// trace container and asserts facts that are true of the simulator (and, for
// the bundled matmul_128, values checked by hand against the example files).
//
//   node viz/selftest.js viz/traces/matmul_8.mtpt viz/traces/matmul_128.mtpt
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
    m.setCycle(m.cycles - 1, 'account');
    check('final C[5][0]', m.img.host[ce.host] === 16, 'host byte 0x' + ce.host.toString(16) + ' = ' + m.img.host[ce.host]);
  }
  if (m.workload.name === 'matmul_8') {
    m.setCycle(5, 'account');
    const o = m.outcomeAt(5);
    check('5 fifo empty', o.kind === 'stall' && o.reason === 'weight_fifo_empty' && o.pc === 2,
          o.reason + ' at cycle 5 for pc ' + o.pc + '; next ready ' + (m.state.fifo[0] && m.state.fifo[0].push.ready));
    m.setCycle(20, 'issue');
    check('20 accumulate', m.state.units.MXU && m.state.units.MXU.pc === 5 && m.state.planes.active === 0,
          'pc 5 (accumulate) issued on plane 0');
    const pe = c.mmSync(1);
    check('20 PE[0][0]', pe && pe.psum[0] === pe.left[0] * pe.w[0] && pe.act[0] === pe.left[0],
          'psum: 0 + (' + pe.left[0] + ' x ' + pe.w[0] + ') -> ' + pe.psum[0]);
  }
}

(async () => {
  for (const f of process.argv.slice(2)) await run(f);
  console.log(failures ? 'selftest: FAILURES' : 'selftest: ALL PASS');
  process.exit(failures ? 1 : 0);
})();
