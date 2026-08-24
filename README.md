# Mini-TPU

A cycle-accurate C++17 simulator of a TPUv1-style int8 inference accelerator. The
datapath is a weight-stationary systolic array of int8 processing elements with
int32 accumulation, fed from an explicitly managed memory hierarchy (banked
unified buffer, accumulator banks, weight FIFO, host DMA) and driven by a CISC
instruction set whose in-order sequencer uses scoreboard interlocks to overlap
DMA, matmul, and activation instructions. Results are requantized from int32 back
to int8 through a bias / requantize / activation / pooling pipeline. Every
structural parameter (array size, buffer size and banking, FIFO depth, DMA
bandwidth, latencies) is a runtime `Config` field, so a configuration sweep needs
no recompilation.

## Features

- **Weight-stationary systolic array** of `dim` x `dim` int8 processing elements,
  each holding a stationary weight and performing one int8 x int8 -> int32
  multiply-accumulate per cycle, with registered activation and partial-sum
  pass-throughs.
- **Activation skew and fill/drain timing**: activations enter the left edge
  skewed by row, partial sums descend the columns, and a matmul over `len` rows
  occupies the array for `len + 2*dim - 1` cycles.
- **Dual weight planes**: `Read_Weights` loads the shadow plane while the active
  plane keeps multiplying, and the plane switch is free. With double buffering
  disabled the load costs `dim` exposed cycles, which the model accounts for
  separately.
- **int32-to-int8 requantization pipeline**: bias add in int64, multiply and
  arithmetic shift with round-half-away-from-zero, saturating int8 clamp, then
  identity / ReLU / ReLU6 and optional max or average pooling on the requantized
  int8 stream.
- **Banked unified buffer**: byte-addressable int8 scratchpad, low-order
  interleaved (`bank = addr % banks`), one read and one write port per bank per
  cycle.
- **int32 accumulator banks**, each `dim` x `dim`. A MatMul either overwrites a
  bank or accumulates in place, which is how K larger than the array is tiled.
- **Weight FIFO** with a configurable per-tile DDR refill latency and a weight
  prefetcher that reads ahead in the instruction stream, so the FIFO depth
  determines how much of the latency is hidden.
- **CISC ISA and in-order sequencer**: fixed-width six-word instructions
  (`Read_Host_Memory`, `Read_Weights`, `MatMul`, `Activate`,
  `Write_Host_Memory`, `Sync`, `Nop`, `Halt`), one instruction issued per cycle,
  with scoreboard interlocks over unified-buffer regions (RAW, WAR, WAW),
  accumulator banks, and the resident weight tile. Issued instructions overlap
  across the DMA, weight, matmul, and activation units.
- **Cycle-accurate timing model** with per-cycle stall accounting: every idle
  array-cycle is attributed to exactly one cause, and the derived report gives
  utilization, effective TOPS, per-instruction cycle counts, and roofline
  placement.

## Architecture

### Simulator (`src/`)

- `types.h`: fixed-width datapath types (`i8`, `i32`), identifier aliases, and
  the opcode, activation-function, and pooling enums.
- `config.h`: one field per structural knob, plus the peak-MAC and roofline
  ridge-point arithmetic.
- `tensor.h`: row-major 2-D views with an explicit stride, so a tile is a window
  into a larger buffer rather than a copy.
- `quant.h`: fixed-point requantization by multiply, arithmetic shift with
  round-half-away-from-zero, and a saturating int8 clamp.
- `pe.h`: one processing element, holding a stationary weight, an
  int8 x int8 -> int32 MAC, registered pass-throughs, and two weight planes.
- `mxu.h`: the matrix unit, a `dim` x `dim` grid of PEs clocked one cycle at a
  time, with activation skew, `2*dim - 1` fill/drain, plane switching, and
  weight-load bubble accounting.
- `unified_buffer.h`: the banked int8 scratchpad, with strided tile views and
  per-cycle port-conflict checking.
- `accumulators.h`: int32 accumulator banks with overwrite, accumulate-in-place,
  and a lock that turns a read of an in-flight bank into a counted hazard.
- `weight_fifo.h`: bounded FIFO of weight tiles with a background DDR refill.
- `dma.h`: the host DMA engine, moving byte ranges between host memory and the
  unified buffer at a configurable bandwidth.
- `isa.h`, `decoder.h`, `decoder.cpp`: the instruction encoding, its field
  layout, the encoder, and the decoder; an illegal opcode decodes to a trapping
  `Halt`.
- `tpu.h`, `tpu.cpp`: the machine, comprising the units above, the in-order
  sequencer with its scoreboard and weight prefetcher, the activation pipeline,
  and the per-cycle stall and idle-cause accounting.
- `stats.h`: derived statistics for one run, covering utilization, effective
  TOPS, roofline placement, per-instruction cycle counts, and the stall-cause
  breakdown.
- `loader.h`: file formats, namely hex and raw binary program images and the
  self-describing MTPU tensor container.
- `main.cpp`: the CLI driver, which loads a program and its tensors,
  disassembles on request, runs the timed model, and prints the report.

### Tests (`tests/`)

- `test_main.cpp`: the suite, 34 sections in one translation unit.
- `ref.h`: the oracle, an eager tensor model with no array, no FIFO, and no
  timing. Its arithmetic shares no code with the timed model except
  requantization, where bit-exactness is a contract.
- `tpuasm.h`: a header-only program builder with named unified-buffer regions.
- `workloads.h`: the tiler, which lowers dense, convolution, and MLP layer specs
  into Mini-TPU programs, plus plain nested-loop golden functions that validate
  the tiler itself.

### Tools (`tools/`)

- `gen_examples.cpp`: writes the bundled workloads to `examples/`, from the same
  definitions the test suite verifies.
- `report.cpp`: regenerates the performance tables (`make report`).
- `mutate.sh`: mutation harness, which breaks the model one edit at a time
  (rounding, interlocks, skew, port contention, counters) and checks that the
  suite notices.

## Building

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

## Usage

The CLI takes a program as hex (`--prog`) or as a little-endian binary image
(`--prog-raw`), with optional MTPU tensor files for weights (`--weights`) and
activations (`--acts`). It can disassemble (`--dump`), execute on the timed model
(`--run`), or execute with a per-instruction issue trace (`--trace`). Every
machine knob has a flag (`--dim`, `--ub`, `--ub-banks`, `--acc-banks`, `--fifo`,
`--dma`, `--ddr-lat`, `--act-depth`, `--double-buffer`, `--no-double-buffer`),
and `--macs N` supplies the workload's padding-free MAC count so utilization can
be reported against it.

```bash
make examples
./build/minitpu --run --prog examples/matmul_128.hex \
    --acts examples/matmul_128.acts.mtpu \
    --weights examples/matmul_128.weights.mtpu \
    --dim 32 --macs 2097152
```

Run `./build/minitpu --help` for the full flag list. Each generated example
carries its own run command in a `# run:` comment at the top of its `.hex` file.

## Validation

The suite is differential at its core: the timed machine is checked against the
eager reference model on output bytes, accumulator snapshots at every `Sync`, and
the retired-instruction count, across every workload and configuration in the
sweep. Because both models execute the same program, the differential check
cannot catch a mis-lowered layer, so each workload is also compared against
independent nested-loop golden functions that never see a tile or an instruction.
Requantization gets a third opinion: it is recomputed from its specification in
the test, since the timed model and the oracle deliberately share `quant.h`.

On top of that, microarchitectural properties are asserted directly:

- fill/drain timing, activation skew, and the resulting utilization ceiling of
  `dim / (3*dim - 1)` for a single matmul;
- scoreboard behaviour for RAW, WAR, and WAW on the unified buffer, accumulator
  hazards, and the weight-tile interlock;
- overlap across units, the weight-load bubble with and without dual planes, and
  the effect of FIFO depth against DDR latency;
- that the stall-cause breakdown is an exact partition of idle time, and that a
  starved resource puts its extra cycles in its own bucket;
- roofline behaviour: below the ridge point, run time scales with bytes moved
  rather than with MACs.

`make debug` runs the whole suite under AddressSanitizer and
UndefinedBehaviorSanitizer. `tools/mutate.sh` injects targeted faults into the
rounding, the interlocks, the skew, the port model, and the counters, and fails
if any of them survives the suite.
