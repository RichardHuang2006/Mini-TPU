# Mini-TPU

A cycle-accurate C++17 simulator of a TPUv1-style int8 inference accelerator,
organized as a learning guide: one source file per hardware concept, one test
file per subsystem, and a README meant to be read in order.

The datapath is a weight-stationary systolic array of `dim x dim` int8
processing elements with int32 accumulation, fed from an explicitly managed
memory hierarchy (banked unified buffer, accumulator banks, weight FIFO, host
DMA) and driven by a CISC instruction set whose in-order sequencer uses
scoreboard interlocks to overlap DMA, matmul, and activation instructions.
Results are requantized from int32 back to int8 through a bias / requantize /
activation / pooling pipeline.

## 1. What Mini-TPU teaches

Each file answers one interview-level hardware question:

- why a systolic array multiplies matrices with no multiplexers or crossbars,
  and why its inputs must be skewed;
- why a matmul over `len` rows costs exactly `len + 2*dim - 1` cycles, and what
  that fixes about utilization;
- why the array keeps two weight planes, and what a weight load costs without
  the second one;
- why on-chip memory is banked, and when a bank conflict stalls the machine;
- how int32 accumulators become int8 activations bit-exactly, with rounding
  half away from zero and saturation at both ends;
- how an in-order sequencer with a scoreboard overlaps independent instructions
  without ever computing a wrong answer;
- how to attribute every idle cycle to exactly one cause, and how a roofline
  tells you whether more compute would even help.

Throughout, keep five things separate; the code and tests are organized around
the distinction:

| Concern | Where it lives | How it is checked |
|---|---|---|
| Functional correctness | what each instruction computes | differential tests vs. the eager oracle, plus independent golden loops |
| Cycle timing | when work starts, stalls and finishes | closed-form cycle counts pinned by tests |
| Storage state | unified buffer, accumulator banks, weight planes, host/DDR | byte and word comparisons, snapshots at `Sync` |
| Scoreboard state | regions and banks reserved by in-flight instructions | stall counters that must fire (or must not) |
| Performance statistics | derived reporting: utilization, TOPS, roofline | balance invariants and starvation experiments |

## 2. Feature summary

- **Weight-stationary systolic array** of `dim x dim` int8 processing elements,
  each holding a stationary weight and performing one int8 x int8 -> int32
  multiply-accumulate per cycle, with registered activation and partial-sum
  pass-throughs (`src/systolic_array.h`).
- **Activation skew and fill/drain timing**: activations enter the left edge
  skewed by row, partial sums descend the columns, and a matmul over `len` rows
  occupies the array for `len + 2*dim - 1` cycles.
- **Dual weight planes**: `Read_Weights` loads the shadow plane while the
  active plane keeps multiplying, and the plane switch is free. With double
  buffering disabled the load costs `dim` exposed cycles, which the model
  accounts for separately.
- **int32-to-int8 requantization pipeline**: bias add in int64, multiply and
  arithmetic shift with round-half-away-from-zero, saturating int8 clamp, then
  identity / ReLU / ReLU6 and optional max or average pooling on the
  requantized int8 stream (`src/datapath.h`, `src/tpu.cpp`).
- **Banked unified buffer**: byte-addressable int8 scratchpad, low-order
  interleaved (`bank = addr % banks`), one read and one write port per bank per
  cycle, with bank-conflict detection (`src/storage.h`).
- **int32 accumulator banks**, each `dim x dim`, with overwrite and
  accumulate-in-place modes (how K larger than the array is tiled), locking,
  and counted read-after-write hazards (`src/storage.h`).
- **Weight FIFO** with a configurable per-tile DDR refill latency and a weight
  prefetcher that reads ahead in the instruction stream, so the FIFO depth
  decides how much of the latency is hidden (`src/transfer.h`, `src/tpu.cpp`).
- **Host DMA engine** at a configurable bytes-per-cycle bandwidth, with range
  checking, in-flight state and completion timing (`src/transfer.h`).
- **CISC ISA and in-order sequencer**: fixed-width six-word instructions
  (`Read_Host_Memory`, `Read_Weights`, `MatMul`, `Activate`,
  `Write_Host_Memory`, `Sync`, `Nop`, `Halt`), one instruction issued per
  cycle, with scoreboard interlocks over unified-buffer regions (RAW, WAR,
  WAW), accumulator banks, and the resident weight tile. Issued instructions
  overlap across the DMA, weight, matmul, and activation units
  (`src/isa.h/.cpp`, `src/tpu.h/.cpp`).
- **Cycle-accurate timing model with stall accounting**: every idle array-cycle
  is attributed to exactly one cause, and the derived report gives utilization,
  effective TOPS, per-instruction cycle counts, and roofline placement
  (`src/stats.h`).

## 3. End-to-end inference dataflow

One layer of a quantized network flows through the machine like this:

```
  host memory                        weight memory (DDR)
      |                                   |
      | Read_Host_Memory                  | Read_Weights: FIFO refill,
      | (DMA, dma_bytes_per_cycle B/cyc)  | ddr_tile_latency cycles per tile
      v                                   v
 +----------------+                 +-------------+
 | Unified Buffer |                 | Weight FIFO |
 | int8, banked   |                 | depth tiles |
 +----------------+                 +-------------+
      |  MatMul streams `len` rows        |  pop -> weight plane
      v                                   v
 +--------------------------------------------------+
 |     dim x dim weight-stationary systolic array   |
 +--------------------------------------------------+
      |  int32 partial sums leave the bottom edge
      v
 +-------------------+
 | accumulator banks |   overwrite, or accumulate in place (K tiling)
 +-------------------+
      |  Activate: +bias -> x multiplier -> >>shift, round -> clamp
      |            -> identity/ReLU/ReLU6 -> optional max/avg pool
      v
 Unified Buffer (int8 activations for the next layer)
      |
      | Write_Host_Memory (DMA)
      v
  host memory
```

A multi-layer network is this loop repeated; the tiler in `tests/workloads.h`
chains layers with no repacking, because a layer's tile-major output is already
the layout the next layer's activations want.

## 4. Repository structure

```
Mini-TPU/
├── README.md
├── LICENSE
├── Makefile
├── src/
│   ├── config.h            every structural knob, one runtime struct
│   ├── datapath.h          int8/int32 types, tensor views, requantization
│   ├── systolic_array.h    the PE and the dim x dim matrix unit
│   ├── storage.h           unified buffer + accumulator banks
│   ├── transfer.h          weight FIFO + host DMA engine
│   ├── isa.h / isa.cpp     opcodes, six-word encoding, decode, disassembly
│   ├── loader.h            program (hex/raw) and tensor (MTPU) file formats
│   ├── tpu.h / tpu.cpp     the machine: sequencer, scoreboard, units, ticking
│   ├── stats.h             derived statistics: utilization, TOPS, roofline
│   ├── trace.h             structured trace observer for the visualizer
│   └── main.cpp            the CLI driver
├── tests/
│   ├── test_support.h            harness, helpers, differential scaffolding
│   ├── tpuasm.h                  program builder with named UB regions
│   ├── ref.h                     the oracle: eager, untimed reference model
│   ├── workloads.h               tiler + independent golden loop nests
│   ├── test_datapath.cpp         views, strides, requantization, rounding
│   ├── test_systolic_array.cpp   PE, skew, fill/drain, dual planes
│   ├── test_storage_transfer.cpp UB, accumulators, FIFO, DMA, hand-driven units
│   ├── test_isa_scoreboard.cpp   encode/decode, interlocks, overlap, Sync
│   ├── test_workloads.cpp        dense/conv/MLP layers vs. golden loops
│   └── test_differential.cpp     oracle diff, stats invariants, config sweep
├── tools/
│   ├── gen_examples.cpp    writes examples/ from the verified workloads
│   ├── report.cpp          regenerates the performance tables
│   ├── tracegen.cpp        records a run into a trace container, with checks
│   └── mutate.sh           mutation harness: injected bugs must be caught
└── viz/                    the cycle visualizer (static page) and its checkers
```

## 5. Recommended reading order

1. `src/config.h` -- the knobs everything else reads.
2. `src/datapath.h` -- types, tensor views, and the requantization arithmetic.
3. `src/systolic_array.h` -- the PE, then the array; the heart of the machine.
4. `src/storage.h` -- the unified buffer and the accumulator banks.
5. `src/transfer.h` -- the weight FIFO and the DMA engine.
6. `src/isa.h`, `src/isa.cpp` -- the instruction set the sequencer executes.
7. `src/tpu.h`, `src/tpu.cpp` -- the sequencer; start at `run()` and `tick()`.
8. `src/stats.h` -- what the counters mean once a program has run.
9. `tests/ref.h`, then `tests/test_differential.cpp` -- how it is all checked.

Each subsystem's tests (`tests/test_<subsystem>.cpp`) double as worked
examples: they set up tiny machines whose answers can be checked on paper.

## 6. Processing element

`Pe` in `src/systolic_array.h`. One stationary int8 weight (two planes of it),
one int8 x int8 -> int32 multiply-accumulate per cycle, and registered
pass-throughs: the activation moves one PE to the right per cycle, the partial
sum one PE down, and neither is visible to a neighbour until the next cycle.
`tick()` computes into pending registers and `commit()` latches them, so the
order PEs are visited in cannot leak into the results.

## 7. Weight-stationary systolic array

`Mxu` in `src/systolic_array.h`. PE[k][c] holds W[k][c]; activations enter the
left edge and partial sums leave the bottom edge, so column `c` computes
`sum_k A[r][k] * W[k][c]` with no interconnect beyond nearest neighbours:

```
                          columns (output features)
                    c=0        c=1        c=2        c=3
                +----------+----------+----------+----------+
  A[r][0] --->  | W[0][0]  | W[0][1]  | W[0][2]  | W[0][3]  |  k=0
                +----v-----+----v-----+----v-----+----v-----+
  A[r][1] --->  | W[1][0]  | W[1][1]  | W[1][2]  | W[1][3]  |  k=1
  (1 cycle      +----v-----+----v-----+----v-----+----v-----+
   later)       | W[2][0]  | W[2][1]  | W[2][2]  | W[2][3]  |  k=2
                +----v-----+----v-----+----v-----+----v-----+
  A[r][3] --->  | W[3][0]  | W[3][1]  | W[3][2]  | W[3][3]  |  k=3
  (3 cycles     +----v-----+----v-----+----v-----+----v-----+
   later)            v          v          v          v
                acc[r][0]  acc[r][1]  acc[r][2]  acc[r][3]

  weights stay put; activations move right, one PE per cycle;
  partial sums move down, gaining one product at every PE.
```

Weights are loaded per-plane by `load_weights()`; a tile smaller than the array
is zero-padded so an undersized matmul is still exactly correct (the padded
products are zero), with the wasted MAC slots counted as `partial_tile_waste`.

## 8. Activation skew and fill/drain

Row `r`'s element for PE row `k` enters `k` cycles after the row starts -- the
activation skew. That is what makes all `dim` products of one output element
meet the descending partial sum in the right PE on the right cycle, given the
one-cycle-per-hop registers.

The consequence is the central timing formula, asserted across shapes by
`tests/test_systolic_array.cpp` and re-derived in `src/systolic_array.h`:

```
matmul over len rows  =  len + 2*dim - 1 cycles
                         ^     ^
                         |     +-- fill/drain: the last row's last column
                         |         takes (dim-1) hops of skew plus dim hops
                         |         down before it leaves the bottom edge
                         +-- streaming: one input row admitted per cycle
```

Since a matmul's results land in one accumulator bank and a bank is `dim` rows
deep, `len <= dim`, so a single matmul can never do better than
`dim / (3*dim - 1)` busy utilization -- about a third. The array is draining,
not stalling; deeper accumulator banks, not more bandwidth, would move that.

## 9. Dual weight planes

Every PE holds two weights: the active plane multiplies while `Read_Weights`
shifts the next tile into the shadow plane, and the switch at the next matmul
is free. The whole benefit of double buffering falls out of one scoreboard
rule: a weight load may not overwrite the tile a running matmul is using
*unless* there is a shadow plane to put it in.

With `double_buffer` off, each load exposes `dim` cycles of weight shifting
(`weight_load_bubble`), and the tests measure the two-matmul sequence both
ways: same answers, `2*dim` cycles apart.

## 10. Unified buffer and accumulator banks

`src/storage.h`, tested in `tests/test_storage_transfer.cpp`.

The **unified buffer** is the byte-addressable int8 scratchpad. Banking is
low-order interleaved -- `bank = addr % banks` -- so the `dim` consecutive
bytes the array consumes per cycle arrive through `dim` different ports rather
than piling into one bank. Each bank has one read and one write port per
cycle; two same-direction accesses to one bank in one cycle are a conflict,
which the sequencer's port model turns into a counted stall. Tensors in the
buffer are strided views (`TensorView` from `src/datapath.h`), so a tile is a
window into a wider matrix, not a copy.

The **accumulator banks** are `acc_banks` separate `dim x dim` int32 arrays. A
`MatMul` either overwrites a bank or accumulates in place -- the second mode is
how K larger than the physical array is tiled: each K-slice's matmul adds into
the same bank, and one `Activate` reads the finished sum. A bank is locked
while a matmul produces into it; consumers that hit the lock stall and the
refusal is counted as an accumulator hazard. `Sync` only issues once the
machine is quiet, so its accumulator snapshot is final for everything before
it.

## 11. Weight FIFO and DMA

`src/transfer.h`, tested in `tests/test_storage_transfer.cpp`.

The **weight FIFO** holds up to `weight_fifo_depth` tiles, each identified by a
`TileId`, refilling from DDR in the background with `ddr_tile_latency` cycles
per tile. The sequencer's prefetcher reads ahead in the instruction stream and
keeps refills for upcoming `Read_Weights` in flight, so a deep FIFO pays the
DDR latency once while a 1-deep FIFO pays it per tile. Popping an empty FIFO
is the `weight_fifo_empty` stall; refilling a full one is rejected and
counted.

The **DMA engine** moves byte ranges between host memory and the unified
buffer at `dma_bytes_per_cycle`, one transfer in flight at a time, with both
endpoints range-checked. A transfer of B bytes takes `ceil(B / bandwidth)`
cycles; the engine is busy for exactly that long, and the scoreboard keeps
consumers of the destination region stalled until the transfer retires. DMA is
never an instantaneous copy: under the sequencer the data effect commits at
retire, so an interlock bug would produce wrong bytes, not just wrong timing.

## 12. Requantization, activation, and pooling

The int32-to-int8 pipeline, `quant::` in `src/datapath.h`, applied by
`Activate` in `src/tpu.cpp` (timed model) and `tests/ref.h` (oracle):

```
   int32 accumulator
        |
   (1)  + bias                  in int64, so a large bias cannot overflow
        |
   (2)  x multiplier            int32 fixed-point multiplier, int64 product
        |
   (3)  >> shift                arithmetic shift by 0..31
   (4)  round half away from 0  add 2^(shift-1) toward the value's sign first;
        |                       a bare >> would floor negatives (-2.5 -> -3,
        |                       not -2)
   (5)  clamp to [-128, 127]    saturating at BOTH ends
        |
   (6)  identity | ReLU | ReLU6 in the output's own int8 units
        |                       (ReLU6 clamps at the int8 value 6)
   (7)  optional max/avg pool   on the requantized int8 stream; average
        |                       pooling rounds by the same rule as (4)
        v
   int8 activation
```

The arithmetic is bit-exact fixed point end to end -- no floating point
anywhere in the model. It is pinned three ways: exhaustive sweeps of the
helpers, differential runs against the oracle, and an independent
re-implementation from the specification in `tests/test_differential.cpp`.
Timing: an `Activate` over `len` rows costs `len*dim + act_pipeline_depth`
cycles (one element per cycle plus the pipeline fill), and pooling does not
make it cheaper, because every accumulator element is still read.

## 13. CISC ISA

`src/isa.h` and `src/isa.cpp`, tested in `tests/test_isa_scoreboard.cpp`. One
opcode per whole-tensor operation:

| Instruction | Unit | Operands |
|---|---|---|
| `Read_Host_Memory` | DMA | host addr, UB addr, bytes |
| `Read_Weights` | weight | DDR addr, tile id |
| `MatMul` | matrix | UB src, len, accumulator bank, accumulate flag |
| `Activate` | activation | bank, UB dst, len, bias, multiplier, shift, fn, pool |
| `Write_Host_Memory` | DMA | UB addr, host addr, bytes |
| `Sync` | sequencer | barrier: wait until the machine is quiet |
| `Nop` | sequencer | nothing |
| `Halt` | sequencer | exit code |

Every instruction is six 32-bit words: word 0 carries the opcode and flag
fields (accumulate bit, activation function, pooling mode/window/stride,
requantization shift), words 1-5 carry operands, and no field straddles a word
boundary, so a hex dump is legible. `encode()`, `decode()` and `disasm()` live
next to the field layout so they cannot drift apart. An opcode outside the
defined set decodes to a trapping `Halt`: a malformed program stops the
machine, with the same trap reason the oracle reports.

## 14. Scoreboard interlocks

Issue is in order, one instruction per cycle; each instruction's `Reservation`
(computed once at issue) records the UB byte regions it reads and writes, the
accumulator banks it touches, and whether it reads or replaces the resident
weight tile. A candidate instruction stalls while any in-flight reservation
conflicts:

```
  Read_Host ── writes UB [a, a+n) ──────────────┐
                                                │ RAW: MatMul reads [a, a+n)
  Read_Weights ── replaces resident weights ──┐ │
                                              │ │  weight hazard: MatMul needs
                                              v v  the tile still in flight
  MatMul ── reads [a, a+n) + weights, writes bank 0 ──┐
                                                      │ accumulator hazard:
                                                      v Activate reads bank 0
  Activate ── reads bank 0, writes UB [c, c+m) ──┐
                                                 │ RAW: Write_Host reads
                                                 v [c, c+m)
  Write_Host ── reads [c, c+m) ── host memory

  WAR: a writer also waits for in-flight readers of its region
       (the array streams its operands over many cycles);
  WAW: two writers of one region never overlap, so the region ends up
       holding what the program said, not whichever finished last.
```

The interlock is what makes overlap safe for *data*, not only for the
schedule: inputs are read at issue and outputs commit at retire, so a missing
interlock shows up as wrong bytes in the differential tests, and each stall
counter (`ub_raw`, `ub_war`, `ub_waw`, `accum_hazard`, `weight_stall`, ...) is
asserted to fire exactly when its hazard is constructed.

## 15. Overlap among hardware units

There are four independent units -- DMA, weight, matrix, activation -- plus the
sequencer, and one instruction can be in flight per unit. Independent work
overlaps; dependent work serializes; the answer is identical either way:

```
  independent operands                     dependent chain
  cycle ->                                 cycle ->
  DMA    ██████████████ (elsewhere)        DMA    ██████████████
  WEIGHT █ (shadow)                        WEIGHT █
  MXU      ███████ (bank 0)                MXU                  ███████
  ACT      ██████████████ (bank 1)         ACT                         ██████████
           ~ cost of the longest unit             ~ cost of the sum of the units
```

The tiler in `tests/workloads.h` is written against this machine model: it
alternates accumulator banks, double-buffers its output staging tiles, and
defers each output tile's `Write_Host` past the next tile's matmuls, because
with in-order issue a stalled drain would drag everything behind it. The tests
measure exactly what each of those choices is worth in cycles.

## 16. Cycle accounting and statistics

The machine keeps raw per-cycle tallies (`RunProfile`, `StallStats` in
`src/tpu.h`); `src/stats.h` derives everything presentable from them and never
touches cycle-state transitions:

- total cycles, retired instructions, per-opcode counts and cycles;
- matrix-unit busy cycles, split into streaming and fill/drain;
- useful MACs vs. performed MACs (padding), utilization overall and while
  busy, effective TOPS at a nominal 700 MHz;
- DMA bytes and cycles, weight-refill cycles, activation cycles, scoreboard
  stall counts, bank conflicts, FIFO stalls, weight-load bubbles;
- arithmetic intensity (MACs per DMA byte), peak compute
  (`dim*dim` MACs/cycle), peak memory bandwidth (`dma_bytes_per_cycle`), the
  roofline ridge point `dim*dim / dma_bytes_per_cycle`, and a
  compute-bound vs. memory-bound classification.

Two invariants are asserted by the tests, not just documented: the idle-cause
buckets partition `cycles - array_busy` exactly (every idle cycle charged to
exactly one cause), and `streaming + fill/drain = array_busy`. Starvation
experiments then check the breakdown *diagnoses*: take away the DMA and
`dma_bound` grows by the slowdown; slow DDR behind a 1-deep FIFO and
`weight_fifo_empty` does.

## 17. Differential validation

Three layers of evidence, none redundant with the others:

1. **Oracle diff** (`tests/ref.h`, `tests/test_differential.cpp`): the timed
   machine against an eager, untimed reference model -- output bytes, host
   memory, unified buffer, accumulator snapshots at every `Sync`, retired
   count, trap reasons -- across every workload and configuration in the
   sweep, including 8x8 and 256x256 arrays, one-bank buffers, 1-deep FIFOs
   and starved DMA. The oracle shares only the `quant::` helpers with the
   timed model (bit-exactness there is a contract); everything else is
   independently implemented.
2. **Golden loops** (`tests/workloads.h`, `tests/test_workloads.cpp`): both
   models run the same program, so the diff cannot catch a mis-lowered layer.
   Every dense, convolution and MLP workload is therefore also compared
   against plain nested loops that never see a tile or an instruction.
3. **Specification checks** (`tests/test_differential.cpp`): requantization
   recomputed from its written specification, structural properties
   (`len + 2*dim - 1`, the utilization ceiling, bubble sizes), roofline
   behavior (below the ridge, time scales with bytes, not MACs), and that no
   run leaks resources -- the machine ends quiet with every bank lock
   released.

On top of that, `tools/mutate.sh` breaks the model on purpose -- rounding,
interlocks, skew, port contention, counters, the prefetcher, the tiler -- and
fails if any injected bug survives the suite.

## 18. Build and run commands

Requires a C++17 compiler and make.

```bash
make            # build build/minitpu (release, -O2)
make test       # regenerate examples/ and run the full test suite
make debug      # build and run the test suite under ASan + UBSan
make examples   # write the bundled workloads to examples/
make report     # regenerate the performance tables from the model
make trace      # record matmul_128 and a small workload for the visualizer
make clean      # remove build/ and examples/
make help       # list the targets
tools/mutate.sh # run the mutation harness (each mutation rebuilds the suite)
```

The CLI takes a program as hex (`--prog`) or a little-endian binary image
(`--prog-raw`), with optional MTPU tensor files for weights (`--weights`) and
activations (`--acts`). It can disassemble (`--dump`), execute on the timed
model (`--run`), or trace per-instruction issue (`--trace`, implies `--run`).
Every machine knob has a flag -- `--dim`, `--ub`, `--ub-banks`, `--acc-banks`,
`--fifo`, `--dma`, `--ddr-lat`, `--act-depth`, `--double-buffer`,
`--no-double-buffer` -- so a configuration sweep needs no recompilation, and
`--macs N` supplies the workload's padding-free MAC count so utilization is
reported against real work.

```bash
make examples
./build/minitpu --run --prog examples/matmul_128.hex \
    --acts examples/matmul_128.acts.mtpu \
    --weights examples/matmul_128.weights.mtpu \
    --dim 32 --macs 2097152
```

Run `./build/minitpu --help` for the full flag list. Each generated example
carries its own run command in a `# run:` comment at the top of its `.hex`
file, and the checked-in performance numbers are regenerated, never
hand-entered: `make report` prints the current tables.

## 19. Cycle visualizer

`viz/index.html` shows a recorded run cycle by cycle: the block diagram with
the transfers the trace records on each path, the systolic array one PE at a
time with the arithmetic it performed, the logical matrices with the progress
of every output element, the memories, and a timeline of every unit. The
simulator is the only source of truth: `src/trace.h` is an observer the
sequencer calls at each retire, prefetch, issue or stall, and at every array
step; `tools/tracegen.cpp` records a run and, before writing the container,
checks that tracing left the run unchanged and that the trace agrees with the
oracle, the golden loops and itself. `make trace` builds both containers and
runs `viz/check_trace.py` over them; `viz/README.md` documents the cycle
contract, what is and is not modeled, and the container format.
