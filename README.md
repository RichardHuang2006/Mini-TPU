# Mini-TPU

A cycle-accurate simulator of Google's **TPUv1**, the inference chip, written in plain C++17. You drive it from a terminal and watch it in a live browser view. Every unit is modelled at its real size and speed:
- a 256×256 systolic array
- a 24 MiB Unified Buffer
- 4 MiB of accumulators
- a 4-tile Weight FIFO fed by DDR3
- a PCIe host link
- an activation unit with pooling

A small compiler turns dense layers, MLPs and convolutions into TPU programs, and every result is checked bit for bit against plain reference loops.

```
./tpu programs/matmul.s
```

`./tpu` rebuilds anything that changed, starts the simulator at a `tpu>` prompt, and opens the visualizer in your browser. From there, `step`, `next` and `run` move time forward, `back` and `prev` move it back, and the page redraws after every command.

## The machine

| Unit | Size | Speed |
|---|---|---|
| Matrix Multiply Unit | 256 × 256 int8 multiply-adds, two weight planes | a multiply over B rows takes B + 511 cycles |
| Unified Buffer | 24 MiB: 98,304 rows of 256 int8 | written and read a row at a time |
| Accumulators | 4 MiB: 4,096 rows of 256 int32 | the MXU writes, Activate reads |
| Weight FIFO | 4 tiles of 64 KiB | filled from DDR3 at 48 B/cycle: 1,366 cycles per tile |
| Weight Memory | 8 GiB DDR3 | 34 GB/s at 700 MHz |
| Host interface | PCIe Gen3 x16 | 22 B/cycle: 12 cycles per 256-byte row |
| Activation unit | int32 → int8, ReLU / sigmoid / tanh, max or average pooling | one row per cycle |

At 700 MHz the array peaks at 91.75 TOPS (65,536 multiply-adds per cycle, 2 operations each).

Instructions are 12 bytes each and issue in order, one per cycle. The units then run side by side, and an interlock holds an instruction back until what it needs is ready. [docs/microarch.md](docs/microarch.md) covers the timing and the interlocks, and [docs/isa.md](docs/isa.md) the instruction set.

## Terminal commands

| Command | Does |
|---|---|
| `load FILE` | reset the machine, assemble the file, load its code and data |
| `step [N]` / `back [N]` | forward / back N **cycles** (default 1) |
| `next [N]` / `prev [N]` | forward until N more **instructions** issue / undo the last N |
| `run` / `run N` | run to Halt / run N cycles; the page animates while it runs |
| `pc` | the instruction at the program counter |
| `wfifo`, `mxu`, `act` | the Weight FIFO's slots, the MXU (with a map of its PEs), the activation unit |
| `ub[ROW:RxC]`, `acc[..]`, `host[..]`, `wmem[..]` | R rows × C columns of a memory from ROW; add `hex` for hex. `ub[5]` is 16 columns of row 5 |
| `quit` | exit (Ctrl-C and Ctrl-D too) |

↑ and ↓ walk the command history, and pressing Enter on an empty line repeats the last command, so `step` followed by Enter, Enter walks one cycle at a time. `back` and `prev` reload the program and replay it, which is exact because the machine is deterministic. Any target you look at is pinned on the page and stays live.

## The visualizer

The page at `http://127.0.0.1:8008/` is read-only. It has no buttons, and every command is typed in the terminal. It draws TPUv1's block diagram in the order data flows:

```
 Host memory ─PCIe─▶ Unified Buffer ─rows, skewed─▶ Matrix Multiply Unit ◀─tile─ Weight FIFO ◀─DDR3─ Weight Memory
                          ▲                                 │ column sums
                          └────── int8 rows ── Activation ◀─ Accumulators
```

- **Blocks** turn teal while they work. The block an instruction is waiting on turns amber and dashed, and the header names the cause.
- **Arrows** animate on the cycles data moves along them.
- **Memory maps:** the Unified Buffer (384 cells of 256 rows) and the accumulators (16 cells) shade the cells that hold data and flash green where something was just written. Rows being written are outlined in green, rows being read in blue.
- **The MXU** is drawn PE by PE at 256 × 256. The rows of a multiply sweep across it as diagonal bands, one hop per cycle.
- **Panels:** below the diagram are the instruction listing with the PC, a timeline of the last 200 cycles (what issued or why not, and which units worked), the stall counters, and your pinned targets.

The server listens on 127.0.0.1 only, trying ports 8008 to 8017. `./tpu --no-open` skips opening a browser, and `./tpu --no-viz` runs without the page. The visualizer is off when input is piped, unless you pass `--viz`.

## Programs

| Program | What it runs | Cycles |
|---|---|---|
| `programs/copy.s` | two rows host → Unified Buffer → host | 49 |
| `programs/weights.s` | two weight tiles into the FIFO; the first moves on into the MXU's shadow plane | 2,733 |
| `programs/matmul.s` | two rows times a hand-written weight tile | 2,136 |
| `programs/activate.s` | matmul.s, then ReLU and tanh, back to the host | 2,187 |
| `programs/dense.s` | 16 × 64 inputs, a 64 → 32 ReLU layer (generated) | 2,352 |
| `programs/mlp.s` | 64 → 48 ReLU → 32 tanh → 10 (generated) | 5,084 |
| `programs/conv.s` | an 8×8×4 image, 3×3 convolution to 8 channels, ReLU, 2×2 max pool (generated) | 2,310 |

The generated programs come from `./build/gen_programs`, which compiles them with random weights and prints what each should leave in host memory, so you can check the result with `host[...]`. A `.s` file holds its data (`.host ROW`, `.weights TILE [ROW]`, then int8 values) and its instructions (`Op key=value …`). [docs/isa.md](docs/isa.md) has the full syntax.

## What the simulator shows about TPUv1

Every program above that uses weights spends most of its time waiting for them:

| Program | Cycles | MXU busy | Waiting for weights |
|---|---|---|---|
| matmul.s | 2,136 | 513 (24%) | 1,620 |
| dense.s | 2,352 | 527 (22%) | 1,619 |
| mlp.s | 5,084 | 1,581 (31%) | 3,293 |
| conv.s | 2,310 | 547 (24%) | 1,619 |

A 64 KiB tile takes 1,366 cycles to come in from DDR3, and 256 more to shift into the array. A multiply over B rows keeps the array busy for B + 511 cycles. So unless each tile is used for well over a thousand rows, the array sits idle waiting for the next one. Large batches and convolutions, which reuse each weight at every pixel, keep it busy; small-batch MLPs don't. [docs/kernels.md](docs/kernels.md) shows how the compiler overlaps what it can.

## Repository layout

```
Mini-TPU/
├── tpu                   start here: rebuild, then run the simulator
├── CMakeLists.txt
├── docs/                 isa.md, microarch.md, kernels.md
├── src/
│   ├── common/           types, TPUv1's fixed parameters, stall causes and counters
│   ├── mem/              Unified Buffer, accumulators, DRAM (host memory and Weight Memory)
│   ├── isa/              the 12-byte instructions; the assembler and its reverse, program_text
│   ├── units/            host interface (PCIe), Weight FIFO (DDR3), activation unit
│   ├── mxu/              the 256 × 256 systolic array and its weight shifter
│   ├── core/             the machine: in-order issue, interlocks, the cycle loop, the activity log
│   ├── compiler/         layers, MLPs and convolutions → programs
│   ├── ref/              reference ops: the same math as plain loops, for checking
│   ├── ui/               the terminal: commands, line editor, prompt loop
│   └── viz/              the visualizer's snapshot and its local web server
├── viz/                  the page: index.html, style.css, app.js
├── tools/                tpu_sim (the terminal), gen_programs, check_headers (a lint rule)
├── programs/             the .s programs above
└── tests/                one test file per unit, plus end-to-end compiler tests
```

## Build, test, lint

```
cmake -S . -B build && cmake --build build
./build/tpu_tests                    # everything; ./build/tpu_tests mxu runs tests whose names contain "mxu"
cmake --build build --target lint    # every file opens with a /// line, then clang-tidy
./build/gen_programs                 # rewrite programs/dense.s, mlp.s and conv.s
```

C++17 with no dependencies beyond the standard library and POSIX. The visualizer's page is plain HTML, CSS and JavaScript, with nothing loaded from the network.

**Not modelled:**
- 16-bit and unsigned operand modes: only signed int8 is modelled
- the host driver and the instruction buffer's transfer over PCIe
- im2col in hardware: convolutions are rearranged on the host by the compiler
