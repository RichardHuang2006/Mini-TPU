// app.js: draws each snapshot: blocks paint occupied rows and computing PEs, the inspector shows values, the timeline the last 64 cycles.
"use strict";

const COLS = 256;                 // every memory row, and every PE row, is 256 values wide
const ROWS_PER_PAGE = 256;        // a 64 KiB host page, a weight tile and a FIFO slot are each 256 rows
const FIFO_SLOTS = 4;
const UB_ROWS = 98304;
const ACC_ROWS = 4096;
const GAP_FROM = 6;               // cells this many pixels or larger get a gap between them
const VIEW_FONT = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
const VIEW_ROW = 15;              // pixels per row in the inspector's viewers
const VIEW_HEADER = 16;           // the viewers' pinned line of column numbers
const LABEL_CHARS = 7;            // row numbers up to 0x17FFF, the Unified Buffer's last, so every view's columns start in the same place
const MOST_ROWS_ASKED = 4096;     // the server sends at most this many rows at once
const TIMELINE_GUTTER = 92;       // the timeline's lane-name column
const TIMELINE_RULER = 16;        // the cycle ruler above the lanes

let latest = null;   // the most recent snapshot

function byId(id) {
    return document.getElementById(id);
}

function hex(value) {
    return "0x" + value.toString(16).toUpperCase();
}

function rowRange(first, count) {
    if (count <= 1) {
        return "row " + hex(first);
    }
    return "rows " + hex(first) + "-" + hex(first + count - 1);
}

function cssColor(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// "#rrggbb" to [r, g, b].
function rgb(color) {
    const value = parseInt(color.slice(1), 16);
    return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

function mix(from, to, t) {
    return [
        Math.round(from[0] + (to[0] - from[0]) * t),
        Math.round(from[1] + (to[1] - from[1]) * t),
        Math.round(from[2] + (to[2] - from[2]) * t),
    ];
}

function readPalette() {
    return {
        panel: cssColor("--panel"),
        ink: cssColor("--ink"),
        line: cssColor("--line"),
        stripe: cssColor("--stripe"),
        empty: rgb(cssColor("--empty")),
        occupied: rgb(cssColor("--occupied")),
        computing: rgb(cssColor("--computing")),
        lanes: {
            issue: cssColor("--lane-issue"),
            stall: cssColor("--lane-stall"),
            host: cssColor("--lane-host"),
            weights: cssColor("--lane-weights"),
            mxu: cssColor("--lane-mxu"),
            act: cssColor("--lane-act"),
        },
    };
}

// Sizes a canvas's drawing buffer to how big it is on screen, so lines and text stay sharp.
function fitCanvas(canvas) {
    let scale = 1;
    if (window.devicePixelRatio > 1) {
        scale = window.devicePixelRatio;
    }
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    const context = canvas.getContext("2d");
    context.setTransform(scale, 0, 0, scale, 0, 0);
    return { context: context, width: width, height: height, scale: scale };
}

// Host memory and Weight Memory blocks show only the pages in use; this finds a real row's place among them (-1: not in use).
function pagedViewRow(pages, realRow) {
    const index = pages.indexOf(Math.floor(realRow / ROWS_PER_PAGE));
    if (index < 0) {
        return -1;
    }
    return index * ROWS_PER_PAGE + (realRow % ROWS_PER_PAGE);
}

// ---------------------------------------------------------------- blocks: occupied rows and computing PEs

// How many cells go on a line (a power of two) so `count` cells come out as large as the canvas allows.
function cellLayout(count, width, height) {
    let best = { across: 1, down: count, size: 0 };
    for (let across = 1; across <= count * 2; across = across * 2) {
        const down = Math.ceil(count / across);
        const size = Math.min(width / across, height / down);
        if (size > best.size) {
            best = { across: across, down: down, size: size };
        }
    }
    return best;
}

function drawGaps(context, layout, left, color) {
    context.strokeStyle = color;
    context.lineWidth = 1;
    context.beginPath();
    for (let a = 1; a < layout.across; a++) {
        context.moveTo(left + a * layout.size, 0);
        context.lineTo(left + a * layout.size, layout.down * layout.size);
    }
    for (let d = 1; d < layout.down; d++) {
        context.moveTo(left, d * layout.size);
        context.lineTo(left + layout.across * layout.size, d * layout.size);
    }
    context.stroke();
}

// Paints `count` cells line by line: a cell whose flag is 1 in `color`, the rest empty; `across` fixes the cells per line.
function paintCells(canvas, flags, count, color, across) {
    const { context, width, height, scale } = fitCanvas(canvas);
    context.clearRect(0, 0, width, height);
    if (count === 0) {
        return;
    }
    let layout = cellLayout(count, width, height);
    if (across !== undefined) {
        const down = Math.ceil(count / across);
        layout = { across: across, down: down, size: Math.min(width / across, height / down) };
    }

    // Cells a device pixel or larger are rounded down to whole device pixels, so every cell and gap comes out the same size.
    const tiny = layout.size * scale < 1;
    if (!tiny) {
        layout.size = Math.floor(layout.size * scale) / scale;
    }

    // One pixel per cell, stretched to size; smoothing only for cells under a device pixel, so painted ones still show.
    const palette = readPalette();
    const image = context.createImageData(layout.across, layout.down);
    for (let i = 0; i < count; i++) {
        let fill = palette.empty;
        if (flags[i] === 1) {
            fill = color;
        }
        image.data[i * 4] = fill[0];
        image.data[i * 4 + 1] = fill[1];
        image.data[i * 4 + 2] = fill[2];
        image.data[i * 4 + 3] = 255;
    }
    const pixels = document.createElement("canvas");
    pixels.width = layout.across;
    pixels.height = layout.down;
    pixels.getContext("2d").putImageData(image, 0, 0);
    const left = Math.floor((width - layout.across * layout.size) / 2 * scale) / scale;   // centred, on a device pixel
    context.imageSmoothingEnabled = tiny;
    context.drawImage(pixels, left, 0, layout.across * layout.size, layout.down * layout.size);
    if (layout.size >= GAP_FROM) {
        drawGaps(context, layout, left, palette.panel);
    }
}

// One flag per cell: every row of each [first, count] run sets the cell `cellOf` maps it to (-1: not shown).
function runFlags(runs, count, cellOf) {
    const flags = new Uint8Array(count);
    for (const run of runs) {
        for (let row = run[0]; row < run[0] + run[1]; row++) {
            const cell = cellOf(row);
            if (cell >= 0 && cell < count) {
                flags[cell] = 1;
            }
        }
    }
    return flags;
}

function sameRow(row) {
    return row;
}

// The UB and accumulators show at least 256 rows, doubling until the highest written row fits.
function shownRows(runs, total) {
    let highest = -1;
    for (const run of runs) {
        highest = Math.max(highest, run[0] + run[1] - 1);
    }
    let shown = 256;
    while (shown <= highest && shown < total) {
        shown = shown * 2;
    }
    return Math.min(shown, total);
}

// A slot's rows fill in as its tile streams in from DDR3.
function fifoFlags(s) {
    const flags = new Uint8Array(FIFO_SLOTS * ROWS_PER_PAGE);
    s.wfifo.slots.forEach(function (slot, index) {
        const rows = Math.ceil(slot.arrived / COLS);
        for (let r = 0; r < rows; r++) {
            flags[index * ROWS_PER_PAGE + r] = 1;
        }
    });
    return flags;
}

// The PEs holding an input value this cycle, from [k, first column, count] runs.
function peFlags(s) {
    const flags = new Uint8Array(COLS * COLS);
    for (const run of s.mxu.pes) {
        for (let n = run[1]; n < run[1] + run[2]; n++) {
            flags[run[0] * COLS + n] = 1;
        }
    }
    return flags;
}

function paintBlocks(s) {
    const palette = readPalette();

    const ubCount = shownRows(s.ub_rows, UB_ROWS);
    paintCells(byId("view-ub"), runFlags(s.ub_rows, ubCount, sameRow), ubCount, palette.occupied);

    const accCount = shownRows(s.acc_rows, ACC_ROWS);
    paintCells(byId("view-acc"), runFlags(s.acc_rows, accCount, sameRow), accCount, palette.occupied);

    const hostCount = s.host.pages.length * ROWS_PER_PAGE;
    const hostCell = function (row) {
        return pagedViewRow(s.host.pages, row);
    };
    paintCells(byId("view-host"), runFlags(s.host.written, hostCount, hostCell), hostCount, palette.occupied);

    const wmemCount = s.wmem.tiles.length * ROWS_PER_PAGE;
    const wmemCell = function (row) {
        return pagedViewRow(s.wmem.tiles, row);
    };
    paintCells(byId("view-wmem"), runFlags(s.wmem.written, wmemCount, wmemCell), wmemCount, palette.occupied);

    paintCells(byId("view-wfifo"), fifoFlags(s), FIFO_SLOTS * ROWS_PER_PAGE, palette.occupied);

    let lineCount = 0;
    if (s.act.busy) {
        lineCount = s.act.line_rows;
    }
    paintCells(byId("view-line"), new Uint8Array(lineCount).fill(1), lineCount, palette.occupied);

    paintCells(byId("view-mxu"), peFlags(s), COLS * COLS, palette.computing, COLS);
}

// ---------------------------------------------------------------- arrows, laid across the gaps between blocks

// Each arrow runs from one block to the next, `at` of the way along the edge the two blocks share.
const ARROWS = [
    { id: "arrow-host-ub", from: "block-host", to: "block-ub", at: 0.35 },
    { id: "arrow-ub-host", from: "block-ub", to: "block-host", at: 0.65 },
    { id: "arrow-ub-mxu", from: "block-ub", to: "block-mxu", at: 0.5 },
    { id: "arrow-wmem-fifo", from: "block-wmem", to: "block-wfifo", at: 0.5 },
    { id: "arrow-fifo-mxu", from: "block-wfifo", to: "block-mxu", at: 0.5 },
    { id: "arrow-mxu-acc", from: "block-mxu", to: "block-acc", at: 0.5 },
    { id: "arrow-acc-act", from: "block-acc", to: "block-act", at: 0.5 },
    { id: "arrow-act-ub", from: "block-act", to: "block-ub", at: 0.5 },
];

// A block's box in the stage's coordinates, which are the arrow layer's.
function boxInStage(id, stage) {
    const box = byId(id).getBoundingClientRect();
    return {
        left: box.left - stage.left,
        right: box.right - stage.left,
        top: box.top - stage.top,
        bottom: box.bottom - stage.top,
    };
}

function layoutArrows() {
    const stage = byId("stage").getBoundingClientRect();
    for (const arrow of ARROWS) {
        const a = boxInStage(arrow.from, stage);
        const b = boxInStage(arrow.to, stage);
        let start = null;
        let end = null;
        const sideBySide = a.right <= b.left || b.right <= a.left;
        if (sideBySide) {
            const top = Math.max(a.top, b.top);
            const y = top + (Math.min(a.bottom, b.bottom) - top) * arrow.at;
            if (a.right <= b.left) {
                start = { x: a.right + 2, y: y };
                end = { x: b.left - 3, y: y };
            } else {
                start = { x: a.left - 2, y: y };
                end = { x: b.right + 3, y: y };
            }
        } else {
            const left = Math.max(a.left, b.left);
            const x = left + (Math.min(a.right, b.right) - left) * arrow.at;
            if (a.bottom <= b.top) {
                start = { x: x, y: a.bottom + 2 };
                end = { x: x, y: b.top - 3 };
            } else {
                start = { x: x, y: a.top - 2 };
                end = { x: x, y: b.bottom + 3 };
            }
        }
        byId(arrow.id).setAttribute("d", "M" + start.x + "," + start.y + " L" + end.x + "," + end.y);
    }
}

// ---------------------------------------------------------------- statuses, arrows and the header

function setBlock(id, busy, stalled) {
    const block = byId(id);
    block.classList.toggle("busy", busy && !stalled);
    block.classList.toggle("stalled", stalled);
}

function renderBlocks(s) {
    const blame = s.blame;
    const fetching = s.wmem.fetching >= 0;
    setBlock("block-host", s.host.busy, blame === "host");
    setBlock("block-ub", s.host.busy || s.mxu.busy || s.act.busy, blame === "ub");
    setBlock("block-mxu", s.mxu.busy || s.mxu.shifting, blame === "mxu" || blame === "weights");
    setBlock("block-wmem", fetching, false);
    setBlock("block-wfifo", fetching || s.mxu.shifting, blame === "wfifo");
    setBlock("block-acc", s.mxu.busy || s.act.busy, blame === "acc");
    setBlock("block-act", s.act.busy, blame === "act");

    const flows = {
        "arrow-host-ub": s.host.busy && s.host.to_ub,
        "arrow-ub-host": s.host.busy && !s.host.to_ub,
        "arrow-ub-mxu": s.mxu.busy,
        "arrow-wmem-fifo": fetching,
        "arrow-fifo-mxu": s.mxu.shifting,
        "arrow-mxu-acc": s.mxu.busy,
        "arrow-acc-act": s.act.busy,
        "arrow-act-ub": s.act.busy,
    };
    for (const id of Object.keys(flows)) {
        byId(id).classList.toggle("active", flows[id]);
    }
}

// The parts joined by " · ", or `fallback` when there are none.
function joinedOr(parts, fallback) {
    if (parts.length === 0) {
        return fallback;
    }
    return parts.join(" · ");
}

function renderStatuses(s) {
    const host = s.host;
    let hostText = "idle";
    if (host.busy && host.to_ub) {
        hostText = "host " + rowRange(host.host_row, host.rows) + " → UB " + rowRange(host.ub_row, host.rows);
    } else if (host.busy) {
        hostText = "UB " + rowRange(host.ub_row, host.rows) + " → host " + rowRange(host.host_row, host.rows);
    }
    if (host.busy) {
        hostText = hostText + ", " + host.done + " of " + host.total + " bytes";
    }
    byId("host-status").textContent = hostText;

    const ub = [];
    if (s.mxu.busy) {
        ub.push("MXU reads " + rowRange(s.mxu.ub_row, s.mxu.rows));
    }
    if (s.act.busy) {
        ub.push("activation writes " + rowRange(s.act.ub_row, s.act.out_rows));
    }
    if (host.busy && host.to_ub) {
        ub.push("host writes " + rowRange(host.ub_row, host.rows));
    } else if (host.busy) {
        ub.push("host reads " + rowRange(host.ub_row, host.rows));
    }
    byId("ub-status").textContent = joinedOr(ub, "idle");

    const mxu = s.mxu;
    let mxuText = "idle";
    if (mxu.busy) {
        mxuText = "step " + mxu.step + " of " + mxu.total + ", UB " + rowRange(mxu.ub_row, mxu.rows) +
                  " → acc " + rowRange(mxu.acc_row, mxu.rows);
    }
    if (mxu.active_tile >= 0) {
        mxuText = mxuText + " · weights tile " + hex(mxu.active_tile);
    }
    if (mxu.shadow_ready) {
        mxuText = mxuText + " · next tile " + hex(mxu.shadow_tile) + " ready";
    } else if (mxu.shifting) {
        mxuText = mxuText + " · next tile " + hex(mxu.shadow_tile) + " " + mxu.rows_shifted + "/256 rows in";
    }
    byId("mxu-status").textContent = mxuText;

    if (s.wmem.fetching >= 0) {
        byId("wmem-status").textContent = "fetching tile " + hex(s.wmem.fetching);
    } else {
        byId("wmem-status").textContent = "idle";
    }

    const slots = s.wfifo.slots.map(function (slot) {
        if (slot.arrived === s.wfifo.tile_bytes) {
            return "tile " + hex(slot.tile) + " ready";
        }
        return "tile " + hex(slot.tile) + " " + Math.floor(100 * slot.arrived / s.wfifo.tile_bytes) + "%";
    });
    byId("wfifo-status").textContent = joinedOr(slots, "empty");

    const act = s.act;
    let actText = "idle";
    let progress = 0;
    if (act.busy) {
        actText = act.function + ", shift " + act.shift;
        if (act.pool !== "none") {
            actText = actText + ", " + act.pool + " pool " + act.pool_size + "×" + act.pool_size;
        }
        actText = actText + ", acc " + rowRange(act.acc_row, act.total) + " → UB " + rowRange(act.ub_row, act.out_rows);
        progress = 100 * act.done / act.total;
    }
    byId("act-status").textContent = actText;
    byId("act-bar").style.width = progress + "%";

    const acc = [];
    if (s.mxu.busy) {
        acc.push("MXU writes " + rowRange(s.mxu.acc_row, s.mxu.rows));
    }
    if (act.busy) {
        acc.push("activation reads " + rowRange(act.acc_row, act.total));
    }
    byId("acc-status").textContent = joinedOr(acc, "idle");
}

function renderHeader(s) {
    byId("cycle").textContent = String(s.cycle);
    byId("pc").textContent = String(s.pc);
    byId("issued").textContent = String(s.issued);
    const state = byId("state");
    state.textContent = s.state;
    if (s.program.length === 0) {
        state.textContent = "no program: type load FILE in the terminal";
    }
    state.classList.toggle("stalled", s.state.startsWith("stalled"));
}

let programKey = "";

function renderProgram(s) {
    const list = byId("program");
    const key = s.program.join("\n");
    if (key !== programKey) {
        programKey = key;
        list.textContent = "";
        s.program.forEach(function (line, index) {
            const item = document.createElement("li");
            item.textContent = String(index).padStart(3) + "  " + line;
            list.appendChild(item);
        });
    }
    const items = list.children;
    for (let i = 0; i < items.length; i++) {
        const current = i === s.pc;
        items[i].classList.toggle("current", current);
        items[i].classList.toggle("stalled", current && s.state.startsWith("stalled"));
    }
    if (items[s.pc] !== undefined) {
        list.scrollTop = items[s.pc].offsetTop - list.clientHeight / 2;
    }
}

// ---------------------------------------------------------------- the buttons

let pressing = false;   // one press at a time; clicks while one is on its way are dropped

// Asks the simulator to move one cycle or one instruction; the new snapshot arrives over /events as usual.
async function press(direction) {
    if (pressing) {
        return;
    }
    pressing = true;
    const unit = document.querySelector("input[name='unit']:checked").value;
    try {
        await fetch("/move?direction=" + direction + "&unit=" + unit, { method: "POST" });
    } catch (error) {
        // the simulator has gone away
    }
    pressing = false;
}

// Goes to the typed cycle, forward or back; anything but a number is ignored.
async function jump(text) {
    const cycle = parseNumber(text);
    if (pressing || cycle < 0) {
        return;
    }
    pressing = true;
    try {
        await fetch("/jump?cycle=" + cycle, { method: "POST" });
    } catch (error) {
        // the simulator has gone away
    }
    pressing = false;
}

function listenButtons() {
    byId("back").addEventListener("click", function () {
        press("back");
    });
    byId("forward").addEventListener("click", function () {
        press("forward");
    });
    byId("jump").addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
            jump(byId("jump").value);
        }
    });
}

// ---------------------------------------------------------------- the row inspector: two scrollable viewers, the one place values show

// One value from a /rows or /pes reply: an int8, a little-endian int32, or a PE's 12 bytes of registers.
function decodeValue(kind, bytes, i) {
    if (kind === "i8") {
        return bytes.getInt8(i);
    }
    if (kind === "i32") {
        return bytes.getInt32(i * 4, true);
    }
    const at = i * 12;
    return {
        weight: bytes.getInt8(at),
        shadow: bytes.getInt8(at + 1),
        act: bytes.getInt8(at + 2),
        psum: bytes.getInt32(at + 4, true),
        row: bytes.getInt32(at + 8, true),
    };
}

// `count` rows of 256 values each.
function decodeRows(kind, bytes, count) {
    const rows = [];
    for (let r = 0; r < count; r++) {
        const values = [];
        for (let c = 0; c < COLS; c++) {
            values.push(decodeValue(kind, bytes, r * COLS + c));
        }
        rows.push(values);
    }
    return rows;
}

// Host memory and Weight Memory show only the pages in use; this is the real row a view row stands for.
function pagedRealRow(pages, viewRow) {
    return pages[Math.floor(viewRow / ROWS_PER_PAGE)] * ROWS_PER_PAGE + (viewRow % ROWS_PER_PAGE);
}

// Every view in data-flow order: the memories go in the left half's menu, the MXU's registers (a PE `field`) in the right's.
const VIEWS = {
    host: { name: "Host memory", kind: "i8", pages: function (s) { return s.host.pages; } },
    ub: { name: "Unified Buffer", kind: "i8", rows: function () { return UB_ROWS; } },
    wmem: { name: "Weight Memory", kind: "i8", pages: function (s) { return s.wmem.tiles; } },
    fifo: { name: "Weight FIFO", kind: "i8", rows: function (s) { return s.wfifo.slots.length * ROWS_PER_PAGE; }, atLeast: FIFO_SLOTS * ROWS_PER_PAGE },
    acc: { name: "Accumulators", kind: "i32", rows: function () { return ACC_ROWS; } },
    line: { name: "Pooling buffer", kind: "i32", rows: function (s) { return s.act.line_rows; }, atLeast: 1 },
    mxu_weight: { name: "MXU weights", kind: "i8", field: "weight", rows: function () { return COLS; } },
    mxu_shadow: { name: "MXU shadow weights", kind: "i8", field: "shadow", rows: function () { return COLS; } },
    mxu_act: { name: "MXU activations", kind: "i8", field: "act", rows: function () { return COLS; } },
    mxu_psum: { name: "MXU partial sums", kind: "i32", field: "psum", rows: function () { return COLS; } },
};

// A paged view lists the pages in use, or page 0 when none is, so an empty memory still shows a page of zeros.
function shownPages(view, s) {
    const pages = view.pages(s);
    if (pages.length === 0) {
        return [0];
    }
    return pages;
}

// The rows the simulator holds for a view; the viewer shows any rows past these as zeros.
function servedRowCount(view, s) {
    if (view.pages !== undefined) {
        return view.pages(s).length * ROWS_PER_PAGE;
    }
    return view.rows(s);
}

// How many rows a view scrolls through: at least a page, all four FIFO slots and one pooling row, so an empty memory shows as zeros.
function viewRowCount(view, s) {
    if (view.pages !== undefined) {
        return shownPages(view, s).length * ROWS_PER_PAGE;
    }
    let rows = view.rows(s);
    if (view.atLeast !== undefined) {
        rows = Math.max(rows, view.atLeast);
    }
    return rows;
}

function realRowOf(view, s, viewRow) {
    if (view.pages !== undefined) {
        return pagedRealRow(shownPages(view, s), viewRow);
    }
    return viewRow;
}

// Where a real row sits in the view, or -1 when the view does not show it.
function viewRowOf(view, s, realRow) {
    let viewRow = realRow;
    if (view.pages !== undefined) {
        viewRow = pagedViewRow(shownPages(view, s), realRow);
    }
    if (viewRow < 0 || viewRow >= viewRowCount(view, s)) {
        return -1;
    }
    return viewRow;
}

// "36" or "0x24"; -1 for anything else.
function parseNumber(text) {
    const trimmed = text.trim();
    if (/^0x[0-9a-f]+$/i.test(trimmed)) {
        return parseInt(trimmed.slice(2), 16);
    }
    if (/^[0-9]+$/.test(trimmed)) {
        return parseInt(trimmed, 10);
    }
    return -1;
}

function formatValue(value, kind, asHex) {
    if (!asHex) {
        return String(value);
    }
    if (kind === "i8") {
        return (value & 0xFF).toString(16).toUpperCase().padStart(2, "0");
    }
    return (value >>> 0).toString(16).toUpperCase().padStart(8, "0");
}

// The width value columns start at, in characters: an int32 in hex, or in decimal up to six characters.
function startingChars(asHex) {
    if (asHex) {
        return 8;
    }
    return 6;
}

// Both viewers share one column width, so every memory's columns line up; it widens to the widest value drawn, never narrows.
let columnChars = 0;

// One half of the inspector: every row of one view, 256 values across, scrolled both ways; only the visible cells are drawn.
class Viewer {
    constructor(prefix, view) {
        this.prefix = prefix;
        this.view = view;          // a key of VIEWS
        this.canvas = byId(prefix + "-values");
        this.scroller = byId(prefix + "-scroll");
        this.spacer = byId(prefix + "-spacer");
        this.values = new Map();   // view row -> its 256 values, all from `cycle`
        this.cycle = -1;
        this.marked = -1;          // the view row typed into the row box, drawn highlighted
        this.loading = false;
        this.loadAgain = false;
        const viewer = this;
        this.scroller.addEventListener("scroll", function () {
            viewer.draw();
            viewer.load();
        });
    }

    rowCount() {
        if (latest === null) {
            return 0;
        }
        return viewRowCount(VIEWS[this.view], latest);
    }

    // Memory rows are numbered in hex, PE rows in decimal, as elsewhere on the page.
    rowLabel(viewRow) {
        const view = VIEWS[this.view];
        const realRow = realRowOf(view, latest, viewRow);
        if (view.field !== undefined) {
            return String(realRow);
        }
        return hex(realRow);
    }

    // Switches to another view, starting at its top.
    show(view) {
        this.view = view;
        this.values.clear();
        this.marked = -1;
        this.scroller.scrollTop = 0;
        this.scroller.scrollLeft = 0;
        this.draw();
        this.load();
    }

    // Scrolls a typed row to the top and highlights it; a row the view does not show is ignored.
    goToRow(realRow) {
        this.marked = -1;
        if (latest !== null && realRow >= 0) {
            this.marked = viewRowOf(VIEWS[this.view], latest, realRow);
        }
        if (this.marked >= 0) {
            this.scroller.scrollTop = this.marked * VIEW_ROW;
        }
        this.draw();
        this.load();
    }

    // A new snapshot: every value held belongs to an older cycle.
    refresh(s) {
        if (s.cycle !== this.cycle) {
            this.values.clear();
            this.cycle = s.cycle;
        }
        this.draw();
        this.load();
    }

    visibleRows() {
        const first = Math.floor(this.scroller.scrollTop / VIEW_ROW);
        const shown = Math.ceil((this.scroller.clientHeight - VIEW_HEADER) / VIEW_ROW) + 1;
        const count = Math.max(0, Math.min(shown, this.rowCount() - first));
        return { first: first, count: count };
    }

    url(first, count) {
        if (VIEWS[this.view].field !== undefined) {
            return "/pes?first=" + first + "&count=" + count;
        }
        return "/rows?memory=" + this.view + "&first=" + first + "&count=" + count;
    }

    decode(bytes, count) {
        const view = VIEWS[this.view];
        if (view.field === undefined) {
            return decodeRows(view.kind, bytes, count);
        }
        return decodeRows("pe", bytes, count).map(function (row) {
            return row.map(function (pe) {
                return pe[view.field];
            });
        });
    }

    // Fetches the visible rows not held yet; a call while one is in flight runs again once it lands.
    async load() {
        if (latest === null) {
            return;
        }
        if (this.loading) {
            this.loadAgain = true;
            return;
        }
        const { first, count } = this.visibleRows();
        const served = Math.min(first + count, servedRowCount(VIEWS[this.view], latest));
        let filled = false;
        for (let r = Math.max(first, served); r < first + count; r++) {
            if (!this.values.has(r)) {
                this.values.set(r, new Array(COLS).fill(0));   // not held by the simulator, so it reads as zeros
                filled = true;
            }
        }
        let missing = -1;
        for (let r = first; r < served && missing < 0; r++) {
            if (!this.values.has(r)) {
                missing = r;
            }
        }
        if (missing < 0) {
            if (filled) {
                this.draw();
            }
            return;
        }

        this.loading = true;
        const view = this.view;
        const cycle = this.cycle;
        const asked = Math.min(served - missing, MOST_ROWS_ASKED);
        try {
            const response = await fetch(this.url(missing, asked));
            if (response.ok && view === this.view && cycle === this.cycle) {
                const rows = this.decode(new DataView(await response.arrayBuffer()), asked);
                for (let i = 0; i < rows.length; i++) {
                    this.values.set(missing + i, rows[i]);
                }
            }
        } catch (error) {
            // the simulator has gone away; the next snapshot will try again
        }
        this.loading = false;
        this.draw();
        if (this.loadAgain) {
            this.loadAgain = false;
            this.load();
        }
    }

    draw() {
        const { context, width, height } = fitCanvas(this.canvas);
        context.clearRect(0, 0, width, height);
        if (latest === null) {
            return;
        }
        const palette = readPalette();
        const view = VIEWS[this.view];
        const asHex = byId("inspect-hex").checked;
        const { first, count } = this.visibleRows();
        context.font = VIEW_FONT;
        context.textBaseline = "middle";
        context.textAlign = "right";

        // Columns fit the widest value either viewer has drawn, so every value shows in full and both halves line up.
        let widest = startingChars(asHex);
        for (let r = first; r < first + count; r++) {
            const values = this.values.get(r);
            if (values === undefined) {
                continue;
            }
            for (const value of values) {
                widest = Math.max(widest, formatValue(value, view.kind, asHex).length);
            }
        }
        if (widest > columnChars) {
            columnChars = widest;
            for (const other of viewers) {
                if (other !== this) {
                    other.draw();
                }
            }
        }
        const digit = context.measureText("0").width;
        const column = Math.ceil(digit * columnChars) + 10;
        let labelChars = LABEL_CHARS;
        if (this.rowCount() > 0) {
            labelChars = Math.max(LABEL_CHARS, this.rowLabel(this.rowCount() - 1).length);
        }
        const labels = Math.ceil(digit * labelChars) + 14;
        this.spacer.style.width = (labels + COLS * column) + "px";
        this.spacer.style.height = (VIEW_HEADER + this.rowCount() * VIEW_ROW) + "px";

        const left = this.scroller.scrollLeft;
        const top = this.scroller.scrollTop;
        const firstCol = Math.max(0, Math.floor(left / column));
        const lastCol = Math.min(COLS - 1, Math.floor((left + width - labels) / column));
        const rowY = function (r) {
            return VIEW_HEADER + r * VIEW_ROW - top;
        };

        if (this.marked >= first && this.marked < first + count) {
            context.fillStyle = cssColor("--busy-soft");
            context.fillRect(0, rowY(this.marked), width, VIEW_ROW);
        }
        context.fillStyle = palette.ink;
        for (let r = first; r < first + count; r++) {
            const values = this.values.get(r);
            if (values === undefined) {
                continue;
            }
            for (let c = firstCol; c <= lastCol; c++) {
                context.fillText(formatValue(values[c], view.kind, asHex), labels + (c + 1) * column - left - 5, rowY(r) + VIEW_ROW / 2);
            }
        }

        // Column numbers stay pinned along the top and row numbers down the left.
        context.fillStyle = palette.panel;
        context.fillRect(0, 0, width, VIEW_HEADER);
        context.fillRect(0, VIEW_HEADER, labels, height);
        if (this.marked >= first && this.marked < first + count) {
            context.fillStyle = cssColor("--busy-soft");
            context.fillRect(0, rowY(this.marked), labels, VIEW_ROW);
        }
        context.font = "bold " + VIEW_FONT;
        context.fillStyle = palette.ink;
        context.save();
        context.beginPath();
        context.rect(labels, 0, width - labels, VIEW_HEADER);
        context.clip();   // a column number half under the row labels is cut off, not left as a sliver
        for (let c = firstCol; c <= lastCol; c++) {
            context.fillText(String(c), labels + (c + 1) * column - left - 5, VIEW_HEADER / 2);
        }
        context.restore();
        for (let r = first; r < first + count; r++) {
            if (rowY(r) + VIEW_ROW > VIEW_HEADER) {
                context.fillText(this.rowLabel(r), labels - 8, rowY(r) + VIEW_ROW / 2);
            }
        }
        context.fillStyle = palette.panel;
        context.fillRect(0, 0, labels, VIEW_HEADER);
    }
}

// Memories on the left, starting at host memory; the MXU's registers on the right, starting at its weights.
const viewers = [new Viewer("left", "host"), new Viewer("right", "mxu_weight")];

// The left menu lists the memories and the right menu the MXU's registers.
function fillMenus() {
    for (const viewer of viewers) {
        const menu = byId(viewer.prefix + "-select");
        for (const key of Object.keys(VIEWS)) {
            const isMxu = VIEWS[key].field !== undefined;
            if (isMxu === (viewer.prefix === "right")) {
                const option = document.createElement("option");
                option.value = key;
                option.textContent = VIEWS[key].name;
                menu.appendChild(option);
            }
        }
        menu.value = viewer.view;
    }
}

function listenInspector() {
    for (const viewer of viewers) {
        const rowBox = byId(viewer.prefix + "-row");
        byId(viewer.prefix + "-select").addEventListener("change", function (event) {
            viewer.show(event.target.value);
            viewer.goToRow(parseNumber(rowBox.value));
        });
        rowBox.addEventListener("input", function () {
            viewer.goToRow(parseNumber(rowBox.value));
        });
    }
    byId("inspect-hex").addEventListener("change", function () {
        columnChars = 0;
        for (const viewer of viewers) {
            viewer.draw();
        }
    });
}

// ---------------------------------------------------------------- the timeline: the last 64 cycles, one lane per unit

// Each lane names what its unit works on; a run of cycles with the same work is one bar.
const LANES = [
    { name: "Issue", key: "issued" },
    { name: "PCIe", key: "host", color: "host" },
    { name: "DDR3 → FIFO", key: "fetching", color: "weights" },
    { name: "Weight shift", key: "shifting", color: "weights" },
    { name: "MXU", key: "mxu", color: "mxu" },
    { name: "Activation", key: "activation", color: "act" },
];

let timelineHover = -1;   // the hovered cycle's place in the window, or -1

function instructionText(s, pc) {
    if (pc >= 0 && pc < s.program.length) {
        return s.program[pc];
    }
    return "pc " + pc;
}

// What a lane shows in one cycle: an id that joins equal neighbours into one bar, a label, a short label and a color; null when idle.
function laneWork(s, lane, cycle) {
    if (lane.key === "issued") {
        if (cycle.issued >= 0) {
            const text = instructionText(s, cycle.issued);
            return { id: "i" + cycle.issued, label: "pc " + cycle.issued + ": " + text, short: String(cycle.issued), color: "issue" };
        }
        if (cycle.stall > 0) {
            return { id: "s" + cycle.stall, label: "stalled: " + s.stall_names[cycle.stall], short: "stall", color: "stall" };
        }
        return null;
    }
    const value = cycle[lane.key];
    if (value < 0) {
        return null;
    }
    if (lane.key === "fetching" || lane.key === "shifting") {
        return { id: String(value), label: "tile " + hex(value), short: hex(value), color: lane.color };
    }
    const text = instructionText(s, value);
    return { id: String(value), label: text, short: text.split(" ")[0], color: lane.color };
}

// The lane's bars across the window: each is { first, count, work }, with first counted from the window's start.
function laneBars(s, lane) {
    const bars = [];
    s.timeline.cycles.forEach(function (cycle, index) {
        const work = laneWork(s, lane, cycle);
        if (work === null) {
            return;
        }
        const last = bars[bars.length - 1];
        const continues = last !== undefined && last.first + last.count === index && last.work.id === work.id;
        if (continues) {
            last.count = last.count + 1;
        } else {
            bars.push({ first: index, count: 1, work: work });
        }
    });
    return bars;
}

// The longest text that fits in `room` pixels: the label, the label cut short with "…", the short label, or nothing.
function fitText(context, label, short, room) {
    if (context.measureText(label).width <= room) {
        return label;
    }
    for (let length = label.length - 1; length >= 8; length--) {
        const cut = label.slice(0, length) + "…";
        if (context.measureText(cut).width <= room) {
            return cut;
        }
    }
    if (context.measureText(short).width <= room) {
        return short;
    }
    return "";
}

function drawBar(context, palette, bar, x, top, width, laneHeight) {
    const edge = palette.lanes[bar.work.color];
    const y = top + 2.5;
    const height = laneHeight - 5;
    const fill = mix(rgb(palette.panel), rgb(edge), 0.3);
    context.beginPath();
    context.roundRect(x + 1, y, Math.max(width - 2, 1), height, 3);
    context.fillStyle = "rgb(" + fill.join(",") + ")";
    context.fill();
    context.strokeStyle = edge;
    context.lineWidth = 1;
    context.stroke();

    const text = fitText(context, bar.work.label, bar.work.short, width - 8);
    if (text !== "") {
        context.fillStyle = palette.ink;
        context.fillText(text, x + 5, y + height / 2);
    }
}

// A tick for every cycle, a longer one with the cycle number every 8.
function drawRuler(context, palette, t, cell, width) {
    context.strokeStyle = palette.line;
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(TIMELINE_GUTTER, TIMELINE_RULER - 0.5);
    context.lineTo(width, TIMELINE_RULER - 0.5);
    for (let c = 0; c <= t.length; c++) {
        const x = Math.round(TIMELINE_GUTTER + c * cell) + 0.5;
        let tick = 3;
        if ((t.first + c) % 8 === 0) {
            tick = 7;
        }
        context.moveTo(x, TIMELINE_RULER);
        context.lineTo(x, TIMELINE_RULER - tick);
    }
    context.stroke();

    context.fillStyle = palette.ink;
    for (let c = 0; c < t.length; c++) {
        const label = String(t.first + c);
        const x = TIMELINE_GUTTER + c * cell + 3;
        const fits = x + context.measureText(label).width <= width;
        if ((t.first + c) % 8 === 0 && fits) {
            context.fillText(label, x, 6);
        }
    }
}

function drawTimeline() {
    const canvas = byId("timeline");
    const { context, width, height } = fitCanvas(canvas);
    context.clearRect(0, 0, width, height);
    if (latest === null) {
        return;
    }
    const palette = readPalette();
    const t = latest.timeline;
    const cell = (width - TIMELINE_GUTTER) / t.length;
    const laneHeight = (height - TIMELINE_RULER) / LANES.length;
    context.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
    context.textBaseline = "middle";

    LANES.forEach(function (lane, index) {
        const top = TIMELINE_RULER + index * laneHeight;
        if (index % 2 === 1) {
            context.fillStyle = palette.stripe;
            context.fillRect(0, top, width, laneHeight);
        }
        context.fillStyle = palette.ink;
        context.fillText(lane.name, 2, top + laneHeight / 2);
    });

    drawRuler(context, palette, t, cell, width);

    LANES.forEach(function (lane, index) {
        const top = TIMELINE_RULER + index * laneHeight;
        for (const bar of laneBars(latest, lane)) {
            drawBar(context, palette, bar, TIMELINE_GUTTER + bar.first * cell, top, bar.count * cell, laneHeight);
        }
    });

    if (timelineHover >= 0 && timelineHover < t.cycles.length) {
        const x = Math.round(TIMELINE_GUTTER + (timelineHover + 0.5) * cell) + 0.5;
        context.strokeStyle = palette.ink;
        context.beginPath();
        context.moveTo(x, TIMELINE_RULER);
        context.lineTo(x, height);
        context.stroke();
    }
}

// The hovered cycle in one line: its number, then what each busy lane worked on.
function showTimelineHover() {
    const readout = byId("timeline-readout");
    if (latest === null || timelineHover < 0 || timelineHover >= latest.timeline.cycles.length) {
        readout.textContent = "";
        return;
    }
    const cycle = latest.timeline.cycles[timelineHover];
    const parts = ["cycle " + (latest.timeline.first + timelineHover)];
    for (const lane of LANES) {
        const work = laneWork(latest, lane, cycle);
        if (work !== null) {
            parts.push(lane.name + ": " + work.label);
        }
    }
    readout.textContent = parts.join(" · ");
}

function listenTimeline() {
    const canvas = byId("timeline");
    canvas.addEventListener("mousemove", function (event) {
        timelineHover = -1;
        if (latest !== null && event.offsetX >= TIMELINE_GUTTER) {
            const cell = (canvas.clientWidth - TIMELINE_GUTTER) / latest.timeline.length;
            timelineHover = Math.floor((event.offsetX - TIMELINE_GUTTER) / cell);
        }
        drawTimeline();
        showTimelineHover();
    });
    canvas.addEventListener("mouseleave", function () {
        timelineHover = -1;
        drawTimeline();
        showTimelineHover();
    });
}

// ---------------------------------------------------------------- snapshots

function render(s) {
    latest = s;
    renderHeader(s);
    renderBlocks(s);
    renderStatuses(s);
    renderProgram(s);
    paintBlocks(s);
    drawTimeline();
    showTimelineHover();
    for (const viewer of viewers) {
        viewer.refresh(s);
    }
}

function connect() {
    const source = new EventSource("/events");
    source.onmessage = function (event) {
        render(JSON.parse(event.data));
    };
}

window.addEventListener("resize", function () {
    layoutArrows();
    if (latest !== null) {
        paintBlocks(latest);
        drawTimeline();
    }
    for (const viewer of viewers) {
        viewer.draw();
        viewer.load();
    }
});

listenButtons();
fillMenus();
listenInspector();
listenTimeline();
layoutArrows();
connect();
