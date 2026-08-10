# Mini-TPU

A cycle-accurate C++17 simulator of a TPUv1-style int8 inference accelerator: a
weight-stationary systolic array with an explicitly managed memory hierarchy and
a CISC-style instruction sequencer that overlaps long-running matmul,
activation, and DMA instructions. The datapath is integer-only — int8
activations and weights, int32 accumulation, fixed-point requantization back to
int8 — and every structural parameter (array size, buffer size and banking,
FIFO depth, DMA bandwidth, latencies) is a runtime `Config` field.

## Architecture

### Simulator (`src/`)

- `types.h` — fixed-width datapath types (`i8`, `i32`) and identifier aliases.
- `config.h` — the machine configuration: one field per structural knob, read at
  runtime so a configuration sweep needs no recompilation.
- `tensor.h` — row-major 2-D views with an explicit stride, so a tile is a
  window into a larger buffer rather than a copy.
- `quant.h` — fixed-point requantization: multiply, arithmetic shift with
  round-half-away-from-zero, saturating int8 clamp.
- `pe.h` — one processing element: a stationary weight, an int8×int8→int32 MAC,
  and registered pass-throughs for the activation (left to right) and the
  partial sum (top to bottom). Two weight planes per PE support double
  buffering.
- `mxu.h` — the matrix unit: a `dim × dim` grid of PEs clocked one cycle at a
  time, with activation skew, `2·dim − 1` fill/drain, plane switching, and
  weight-load bubble accounting.
- `unified_buffer.h` — the Unified Buffer: a banked, byte-addressable int8
  scratchpad for activations and intermediate results.
- `accumulators.h` — int32 accumulator banks; a MatMul either overwrites a bank
  or accumulates in place, which is what K-tiling uses.
- `weight_fifo.h` — the weight FIFO: tiles staged from weight memory (DDR)
  arrive after a configurable refill latency, and the FIFO depth decides how
  much of that latency can be hidden.
- `dma.h` — the host DMA engine: byte ranges moved between host memory and the
  Unified Buffer at a configurable bandwidth.
- `isa.h` / `decoder.h` / `decoder.cpp` — the fixed-width instruction set
  (Read/Write_Host_Memory, Read_Weights, MatMul, Activate, Sync, Nop, Halt) and
  its decoder; an illegal opcode decodes to a trapping Halt.
- `tpu.h` / `tpu.cpp` — the machine itself: the units above plus the sequencer.
  In-order issue, one instruction per cycle, with a scoreboard interlock over
  Unified Buffer regions, accumulator banks, and the resident weights; issued
  instructions overlap across units, and a weight prefetcher keeps upcoming
  tiles arriving from DDR. Also implements the activation pipeline (bias,
  requantize, identity/ReLU/ReLU6, optional max/average pooling) and per-cycle
  stall accounting.
- `stats.h` — derived statistics for one run: utilization, effective TOPS,
  roofline placement, per-instruction cycle counts, and a stall-cause breakdown
  that attributes every idle array-cycle to exactly one cause.
- `loader.h` — file formats: hex and raw binary program images, and the
  self-describing MTPU tensor container.
- `main.cpp` — the CLI driver: loads a program and its tensors, disassembles on
  request, runs the timed model, and prints the statistics report.

### Tests (`tests/`)

The suite (`test_main.cpp`, 34 sections) is differential at its core: the timed
machine is checked against an eager reference model on output bytes,
accumulator snapshots at every Sync, and the retired-instruction count, across
workloads and configuration sweeps. Microarchitectural properties (overlap,
interlocks, fill/drain timing, stall attribution) are asserted directly on top
of that.

- `ref.h` — the oracle: an eager tensor model with no array, no FIFO, and no
  timing. Its arithmetic deliberately does not share code with the timed model,
  except for requantization, where bit-exactness is a contract.
- `tpuasm.h` — a header-only program builder, so tests read like short programs
  instead of hand-computed byte offsets.
- `workloads.h` — the tiler: lowers dense, conv, and MLP layer specs into
  Mini-TPU programs, plus plain nested-loop golden functions that validate the
  tiler itself, since the reference model and the timed model execute the same
  generated program and cannot catch a bug in generating it.

### Tools (`tools/`)

- `gen_examples.cpp` — writes the bundled workloads to `examples/`, from the
  same definitions the test suite verifies.
- `report.cpp` — regenerates the performance tables (`make report`); the same
  measurements are asserted by the test suite.
- `mutate.sh` — mutation harness: breaks the model one edit at a time (rounding,
  interlocks, skew, port contention, counters) and checks the suite notices.

## Building and running

Requires a C++17 compiler and make.

```bash
make            # build build/minitpu (release, -O2)
make test       # regenerate examples/ and run the test suite
make debug      # build and run the test suite under ASan + UBSan
make examples   # write the bundled workloads to examples/
make report     # print the performance tables
make clean      # remove build/ and examples/
make help       # list the targets
```

The CLI loads a program (`--prog` hex or `--prog-raw` binary) with optional
MTPU tensors (`--weights`, `--acts`), and can disassemble (`--dump`), execute
(`--run`), or trace (`--trace`) it; every machine knob has a flag. See
`./build/minitpu --help` for the full list. Each generated example carries its
own run command in a `# run:` comment at the top of its `.hex` file.
