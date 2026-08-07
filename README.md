<div align="center">

# Mini-TPU

**A from-scratch, cycle-accurate TPUv1-style int8 inference accelerator**

`C++17` · `Systolic array` · `Cycle-accurate`

</div>

Mini-TPU is a single-chip microarchitectural simulator of a weight-stationary systolic
matrix unit with a unified on-chip buffer, int32 accumulator banks, a weight FIFO, host DMA,
and a CISC instruction sequencer that overlaps long-running matmul, activation, and DMA
instructions. It is the accelerator sibling of [Mini-CPU](../Mini-CPU) and shares its
philosophy: an obviously-correct eager reference model is the oracle, and every cycle-accurate
component is validated by differential testing against it.

---

## Features

- **Weight-stationary systolic array** — a configurable `dim × dim` grid of int8×int8→int32
  MAC cells with activation skew, `2·dim − 1` fill/drain, and double-buffered weight planes.
- **Integer-only datapath** — int8 activations and weights, int32 accumulation, fixed-point
  requantization with round-half-away-from-zero and saturating int8 clamp.
- **Explicitly-managed memory** — a banked Unified Buffer scratchpad, int32 accumulator banks
  with accumulate-in-place K-tiling, a weight FIFO, and a fixed-bandwidth host DMA.
- **CISC sequencer with overlap** — eight instructions, in-order issue, and an interlock
  scoreboard that overlaps independent instructions across units without any speculation.
- **Activation pipeline** — bias, requantize, identity / ReLU / ReLU6, and max / average
  pooling.
- **Fully configurable** — array size, buffer size and banking, FIFO depth, DMA bandwidth, and
  latencies are all `Config` fields, swept by the test suite.
- **Characterization that diagnoses** — utilization, effective TOPS, roofline placement, and a
  stall-cause breakdown that exactly partitions every idle array-cycle, so starving a resource
  shows up in that resource's bucket and nowhere else.

---

## Quickstart

```bash
make                       # build build/minitpu (-O2, warnings on)
./build/minitpu --help     # list every configuration knob
make test                  # regenerate examples/ and run the differential suite
make debug                 # build + run the suite under ASan + UBSan
make report                # regenerate the performance tables in DESIGN.md §9
```

Every bundled workload carries its own run command in a comment at the top of its
`.hex` file, so running one takes no arguments of your own:

```bash
make examples
eval $(grep '^# run:' examples/mlp_3layer.hex | sed 's|^# run: minitpu|./build/minitpu|')
```

---

## What you get

The simulator reports where the cycles went, not just a result. Real output, from
the command above:

```text
run:
  array 16x16  UB 32768 B / 8 banks  acc 4  FIFO 4  DMA 16 B/cyc  double-buffered
  4273 cycles, 123 instructions retired, 7680 DMA bytes
  163840 useful MACs of 163840 performed  (15.0% of array-cycles offered)
  utilization 15.0% overall, 34.0% while busy   effective 0.054 TOPS @ 700 MHz
  arithmetic intensity 21.3 MAC/B  ridge 16.0  -> compute-bound
  lost array-cycles:
    array_fill_drain         1240   29.0%
    weight_fifo_empty           0    0.0%
    ub_bank_conflict            0    0.0%
    accum_hazard                0    0.0%
    dma_bound                 370    8.7%
    activation               2022   47.3%
    other                       1    0.0%
    partial_tile_waste          0    0.0%  (inside busy cycles)
  dominant cause: activation   (largest resource stall: activation)
  per instruction:
    Read_Host_Memory       18 x       288 cycles  (16.0 avg)
    Read_Weights           40 x        40 cycles  (1.0 avg)
    MatMul                 40 x      1880 cycles  (47.0 avg)
    Activate               12 x      3120 cycles  (260.0 avg)
    Write_Host_Memory      12 x       192 cycles  (16.0 avg)
    Halt                    1 x         1 cycles  (1.0 avg)
```

The idle buckets **partition** idle time — every cycle the array stands still is
charged to exactly one of them, and `busy + idle == cycles` is asserted for every
workload on every configuration. That is what makes the breakdown a diagnosis
rather than a decoration.

And it diagnoses something real. An `Activate` costs 260 cycles here against a
`MatMul`'s 47, because [§6.2](./DESIGN.md#62-activation-pipeline) specifies a
throughput-1 pipeline emitting one element per cycle while the array produces
`dim²` MACs per cycle: requantizing a tile is more expensive than computing it, at
every array size, by a factor that grows linearly with `dim`. Read
[§9.5](./DESIGN.md#95-what-the-numbers-say) for the other three findings, including
why utilization is capped near ⅓ and why four of the six configuration knobs turn
out not to matter.

---

## Repo layout

```text
Mini-TPU/
├── DESIGN.md        architecture and rationale
├── PLAN.md          34-step build roadmap
├── Makefile
├── src/             simulator sources (array, memory, sequencer, activation, statistics)
├── tests/           in-tree program builder, eager reference model, tiler, differential suite
└── tools/           example generator, performance-table generator, mutation harness
```

---

## Documentation

- [DESIGN.md](./DESIGN.md) — the architecture: goals, systolic array, memory subsystem, CISC
  instruction set, activation pipeline, testing strategy, and performance characterization.
- [PLAN.md](./PLAN.md) — the build roadmap: 34 steps across 9 phases, one source file and one
  test section per step, built around the reference-model oracle.

---

## Status

All nine phases of [PLAN.md](./PLAN.md) are implemented. 34 test sections pass clean under
ASan + UBSan; `make test` takes about 12 seconds and `make debug` about 50.

Every workload runs on six configurations — 8×8 through 256×256, single-bank buffer,
shallow FIFO, starved DMA — and each is checked against the eager reference model on the
output tensor, the accumulator contents at every `Sync`, and the retired-instruction count.
The tiler is checked separately against plain nested-loop golden implementations, since the
reference model and the timed model execute the *same* generated program and so cannot catch
a bug in generating it.

`tools/mutate.sh` breaks the model 20 different ways — rounding modes, interlocks, skew,
port contention, FIFO depth, every statistics counter — and asserts the suite catches each
one. It currently does.

---

## References

- N. P. Jouppi et al., *In-Datacenter Performance Analysis of a Tensor Processing Unit*,
  ISCA 2017 — the TPUv1 architecture this model is patterned on.
- B. Jacob et al., *gemmlowp: a small self-contained low-precision GEMM library* — the
  fixed-point requantization (multiplier + shift, round, saturate) reference.
