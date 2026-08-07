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

---

## Quickstart

```bash
make                       # build build/minitpu (-O2, warnings on)
./build/minitpu --help     # list every configuration knob
make test                  # regenerate examples/ and run the differential suite
make debug                 # build + run the suite under ASan + UBSan
```

Run a bundled workload and dump the output tensor plus statistics:

```bash
./build/minitpu --prog examples/mlp.prog --weights examples/mlp.w \
                --acts examples/mlp.x --dump --dim 32
```

---

## What you get

The simulator reports utilization and a stall-cause breakdown, not just a result:

```text
workload: mlp_dense   config: default (32x32)
  cycles            12480
  MACs              10.4 M
  array util         86.3 %
  effective TOPS      ...
  DMA bytes           192 KiB
  stall causes:
    array_fill_drain   7.1 %
    weight_fifo_empty  2.4 %
    ub_bank_conflict   1.9 %
    accum_hazard       1.2 %
    dma_bound          1.1 %
    partial_tile_waste 0.0 %
```

---

## Repo layout

```text
Mini-TPU/
├── DESIGN.md        architecture and rationale
├── PLAN.md          34-step build roadmap
├── Makefile
├── src/             simulator sources (array, memory, sequencer, activation)
├── tests/           in-tree program builder, eager reference model, differential suite
└── tools/           example / workload generator (tiler)
```

---

## Documentation

- [DESIGN.md](./DESIGN.md) — the architecture: goals, systolic array, memory subsystem, CISC
  instruction set, activation pipeline, testing strategy, and performance characterization.
- [PLAN.md](./PLAN.md) — the build roadmap: 34 steps across 9 phases, one source file and one
  test section per step, built around the reference-model oracle.

---

## Status

Documentation-first. `DESIGN.md` and `PLAN.md` are complete; implementation follows the plan
phase by phase. The plan is designed so the repo is buildable and green at the end of every
step, with the first end-to-end differential pass at [Step 3.3](./PLAN.md#phase-3--systolic-array)
and a working end-to-end accelerator at [Step 5.5](./PLAN.md#phase-5--sequencer--overlap).

---

## References

- N. P. Jouppi et al., *In-Datacenter Performance Analysis of a Tensor Processing Unit*,
  ISCA 2017 — the TPUv1 architecture this model is patterned on.
- B. Jacob et al., *gemmlowp: a small self-contained low-precision GEMM library* — the
  fixed-point requantization (multiplier + shift, round, saturate) reference.
