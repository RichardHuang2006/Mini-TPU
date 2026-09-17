# Mini-TPU cycle visualizer

An interactive, cycle-accurate view of what the simulator in `src/` does: the
systolic array with the MatMul drawn inside the PE grid, the microarchitecture
blocks and the data moving between them, the logical matrices with the
progress of every output element, the memories, a timeline of every unit
across the whole run, and a five-line strip saying what each step of the run
loop did this cycle, all driven by a trace the simulator itself records.

Nothing in the page simulates. The simulator is instrumented through the
observer in `src/trace.h`; `tools/tracegen.cpp` records a run into a container
(`.mtpt`) and proves, before writing it, that tracing changed nothing and that
the trace agrees with the eager oracle, the golden loops and itself. The page
replays the recorded commits to materialize the state at any cycle, forward or
backward, and every number it shows is read from a trace record; the few
derived annotations (logical coordinates, fill/drain roles, nominal DMA and
Activate progress) are labelled as such.

## Run

    make trace              # builds tools/tracegen, writes viz/traces/*.mtpt, checks them
    python3 visualizer.py   # serves viz/ and opens the 128×128 example

`visualizer.py` serves the directory over local HTTP, answers
`traces/index.json` with every container in `viz/traces`, and opens
`matmul_128` (128×128 · 128×128 on the 32×32 array: 64 tile MatMuls, 19,013
cycles) when it has been traced; the **Workload** dropdown lists the embedded
`matmul_8` and every container found. `--trace NAME` opens another one and
`index.html?trace=traces/x.mtpt` works with any static server. Opened from
`file://` the page starts on the small teaching workload (`matmul_8`, embedded
in `viz/traces/matmul_8.js`); **Open…** or drag-and-drop loads a container
from disk (the per-MatMul PE detail is then read lazily from the file).

The **Lessons** menu jumps to named cycles; each entry opens the view it is
best watched on (the array steps of the first MatMul, the stalls on the block
diagram).

`make` on this machine currently needs the Command Line Tools compiler:

    SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
    make CXX=/Library/Developer/CommandLineTools/usr/bin/clang++ trace

## Cycle contract

Cycle *t* is one iteration of `Tpu::run` (`src/tpu.cpp`). The **phase**
control in the top bar walks the loop's own order inside a cycle:

| phase | what has happened at this point of cycle *t* |
|---|---|
| start | nothing yet: the state left by cycle *t − 1* |
| retire | units with `done_cycle ≤ t` committed their staged outputs (`Tpu::finish`); a Read_Weights popped the FIFO and loaded the shadow plane |
| prefetch | the prefetcher pushed tiles for upcoming Read_Weights (`Tpu::prefetch_weights`) |
| issue | the instruction at `pc` issued (inputs read, outputs staged, unit occupied) or exactly one stall counter moved (`Tpu::issue_step`) |
| account | the cycle was charged to array-busy or to one idle bucket (`Tpu::charge_idle_cycle`) |

Stepping by a cycle lands on *account*. A MatMul issued at cycle *t₀* runs
array step *s* at cycle *t₀ + s*; the PE grid shown after phase *issue* of
cycle *t₀ + s* is the grid after step *s* committed. The staged rows commit to
the accumulator bank at *t₀ + len + 2·dim − 1*, which is also the cycle the
next MatMul in a K-chain issues. Halt increments the cycle counter itself, so
the last cycle shows an empty machine.

The **This cycle** strip (top right) has the same five lines for every cycle,
in the loop's order: *retire*, *prefetch*, *issue*, the *array* step that
issue ran, and *account*. Each line is one short clause built from the
records above (which instruction, what it committed, which counter moved);
lines for phases the phase control has not reached yet are dimmed. Clicking a
step name stops at that phase; clicking the text selects the instruction,
unit or view it names. The long-form narration of the same records is under
*Full narration* in the cycle panel below the strip.

## Systolic array view

The default tab draws the MatMul in flight inside the `dim × dim` PE grid,
reading left → grid → bottom → staged rows, with nothing simulated: every
cell is a recorded PE register after the step shown, and the three placement
rules are derived and checked by `viz/selftest.js`:

| rule | meaning |
|---|---|
| role `r = s − k − c` | PE[k][c] works on row *r* of the A tile at step *s*; useful when 0 ≤ r < len, *fill* before, *drain* after |
| input skew | PE row *k* receives `A[s − k][k]` at its left edge at step *s* (`left[s·dim + k]`): a whole row never enters at once |
| landing | row *r* leaves column *c* at the bottom edge at step `s = r + dim − 1 + c` (`land[r·dim + c]`); row *r* is complete at `s = r + 2·dim − 2` |

- **Grid** (*flow* colouring): the activation each PE holds as a blue (+) or
  copper (−) tint that shifts one cell right per step, a faint role tint, and
  a bar on each useful PE's right edge whose length is its partial sum, growing
  toward the bottom edge. The leading anti-diagonal (`k + c = s`, row 0 of A)
  and the trailing one (row len − 1) are drawn as lines with a dashed ghost of
  the previous step, so playing the run shows the wavefront sweeping down and
  to the right. *act*, *|psum|*, *w* and *role* colour one register or the
  role only. Values appear as text once a cell is 26 px (act) or 44 px (act,
  psum and w); the **text** button zooms there, **fit** shows the whole grid.
  Hover any PE for its registers and `psum_in + act × w → psum_out`; click to
  open it in the inspector; hovering also outlines every PE on the same
  anti-diagonal, which all work on the same A row this step.
- **In lane** (left of the grid): the value entering each PE row this step
  (`left[s·dim + k]`), and the **A tile** thumbnail as read at issue with the
  entering diagonal lit and the elements already inside dimmed.
- **Landing lane** (under the grid): the partial sums leaving the bottom edge
  this step, labelled with their output row, and the **staged rows** strip on
  the right filling diagonally as rows land; complete rows are ticked, and the
  caption says which bank they commit to and when.
- **Weights**: the resident plane (the B tile) with which Read_Weights loaded
  it. Between MatMuls the grid shows the resident plane faintly with a banner
  naming what the array is waiting on and when the next MatMul issues.

## What is and is not modeled

| shown as | meaning |
|---|---|
| modeled | read directly from a trace record: unit occupancy, reservations, stall reason and blocker, FIFO entries, plane loads, every PE register after every step, every byte read at issue and committed at retire |
| derived | computed from records by a fixed rule and cross-checked: logical coordinates A[i,k] / B[k,j] / C[i,j] and tile ids (from the tiler's packing in `tests/workloads.h`), fill / useful / drain roles (`r = s − k − c`), the requantization steps between the bank value read and the byte committed |
| nominal | a duration the model charges without simulating its interior: DMA byte progress, Activate element order |
| not modeled | Unified Buffer bank conflicts by address (the sequencer budgets read and write *streams* against `ub_banks`), accumulator locking (the reservation is the interlock), a weight tile shifting into the array (it lands whole at retire) |

Two attributions of a stalled cycle exist in the simulator and are shown side
by side, never merged: the issue-point counter that refused the instruction at
`pc` (`StallStats`) and the bucket the idle array was charged to
(`charge_idle_cycle`, whose priority order is weights › bank › accum › DMA
active › ACT active › other).

## Files

| file | role |
|---|---|
| `src/trace.h` | `TraceSink`, the observer the sequencer calls; `MxuObserver` in `src/systolic_array.h` sees every array step |
| `tools/tracegen.cpp` | records a run, runs the checks, writes the container |
| `viz/check_trace.py` | independent re-check of a container (layout, cycles, lifetimes, commits, reads, PE identity, landings, activation, weights, golden) |
| `viz/selftest.js` | runs the page's reader and model under node against a container and asserts worked-example values |
| `viz/embed_small.py` | embeds the small container into `viz/traces/matmul_8.js` |
| `viz/js/container.js` | `.mtpt` reader (eager sections, lazy PE detail) |
| `viz/js/model.js` | state at (cycle, phase) by commit replay |
| `viz/js/narrate.js` | the five-line cycle strip (`brief`) and the long-form narration (`narrate`) |
| `viz/js/diagram.js`, `array.js`, `matrices.js`, `memory.js`, `timeline.js`, `inspector.js`, `app.js` | the views and the controller |

## Container format (`mtpt/1`)

`"MTPT"`, u32 version, u32 manifest length, the JSON manifest, padding to 8
bytes, then 8-byte-aligned little-endian sections listed in
`manifest.sections` (`name`, `dtype`, `count`, `off`, `len`). Sections:
`prog`, `init.host`, `init.ddr`, the per-cycle columns `cyc.*`, `commits` and
`reads` (byte blobs indexed by the manifest's `commits[]` and `reads[]`
records, each commit with an after-image and a before-image), `syncs`, and
per MatMul `mm.<i>.w`, `.left`, `.act`, `.psum`, `.land` (the active plane,
the left-edge inputs per step, the PE registers after each step, and the
values that left the bottom edge). The manifest also carries the program with
each instruction's lifetime, reservation and tile, the issue / retire /
prefetch / plane-load / stall-span records, the statistics, and the results of
the generator's checks.

## Timeline

The bottom timeline has one row per unit (SEQ, DMA, WEIGHT, MXU, ACT) and four
diagnostic rows (FIFO, UB ports, ACC banks, idle cause), all drawn from the
per-cycle trace columns and the issue / retire / stall records.

- **Hover** any row for what occupies it at that cycle: the instruction and its
  lifetime on unit rows; occupancy and ready slots against `weight_fifo_depth`
  on FIFO; read and write streams against `ub_banks` on UB ports; the pc
  reserving each accumulator bank lane on ACC; and on idle cause, the bucket,
  its run, `charge_idle_cycle`'s rule and the instruction it points at. When a
  pixel covers several cycles the tooltip says which, and the FIFO and UB rows
  draw the minimum solid with the min–max range as a lighter band, so a drop to
  empty inside a pixel stays visible.
- **Click** a stall run on SEQ or a run on idle cause to seek there and select
  the blocker the sequencer recorded (or, for an idle cycle with no recorded
  blocker, the DMA or ACT instruction the charge rule found active). Click a
  unit bar or an ACC lane to select that instruction; alt+click also seeks.
  Click the ruler or empty track to seek; drag to scrub.
- **Wheel** zooms, shift+wheel or a horizontal swipe pans, alt+wheel scrolls
  the rows. Rows keep a minimum height and scroll when they do not fit; click a
  row label to collapse or expand it.
- **Where the cycles went** (right of the tracks) is a Pareto of the
  `charge_idle_cycle` buckets beside the `StallStats` issue-point counters for
  the visible window, recomputed on every zoom and pan. Over the whole run they
  equal the recorded statistics. Click a bar to dim every cycle that does not
  match; click it again, or *clear*, to remove the filter.
- **Blocker chain** (inspector, for any selected instruction) walks back from
  its issue to the root of its critical path. Each hop follows the blocker
  recorded for the instruction's last wait, which retired on the very cycle the
  waiter issued, or the previous instruction when a hop issued on its first
  attempt (in-order issue), and ends at pc 0 or at a wait with no blocking
  instruction such as DDR latency. It is derived from recorded records only and
  labelled as such; `viz/selftest.js` checks both release rules on every hop.

Colours for stall reasons and idle buckets are theme tokens in `app.css`
(`--st-*`, `--idle-*`); idle buckets are hatched, so no two buckets share a
fill in either theme.

## Keyboard

← → cycle (Shift: ×10) · `.` `,` phase · Space play · `e` `E` next/previous event ·
`m` `M` MatMul · `t` `T` tile boundary · `1`–`4` views · Esc clear selection.
