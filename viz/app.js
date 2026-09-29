// app.js: draws the TPU from each snapshot the simulator pushes over /events; nothing on the page is clickable.
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const MAP_ROWS = 256;          // one map cell covers 256 memory rows
const UB_ACROSS = 24;          // the Unified Buffer map: 384 cells, 24 across and 16 down
const FRESH_CYCLES = 40;       // a just-written cell fades out over this many cycles
const TIMELINE_LANES = [
    { key: "issue", label: "issue" },
    { key: "host",  label: "PCIe" },
    { key: "fetch", label: "DDR3" },
    { key: "shift", label: "shift" },
    { key: "mxu",   label: "MXU" },
    { key: "act",   label: "activate" },
];

// One color per timeline letter: I is an instruction issuing, the rest are the stall causes.
const LETTER_COLORS = {
    I: "#0f9d8a",
    h: "#4c7bd9",
    w: "#8b93a5",
    f: "#a064c8",
    m: "#e0782c",
    g: "#d9a400",
    u: "#d0453a",
    a: "#b0487d",
    c: "#7a8f3a",
};

const ui = {
    ubCells: [],
    accCells: [],
    fifoSlots: [],
    programKey: "",
    keyBuilt: false,
};

function byId(id) {
    return document.getElementById(id);
}

function hex(value) {
    return "0x" + value.toString(16).toUpperCase();
}

function cssColor(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// "#rrggbb" to [r, g, b].
function rgb(color) {
    const value = parseInt(color.slice(1), 16);
    return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

function svg(tag, attributes, parent) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const name of Object.keys(attributes)) {
        node.setAttribute(name, attributes[name]);
    }
    parent.appendChild(node);
    return node;
}

// "rows 0x10-0x1F", or "row 0x10" for a single row.
function rowRange(first, count) {
    if (count <= 1) {
        return "row " + hex(first);
    }
    return "rows " + hex(first) + "-" + hex(first + count - 1);
}

// True when map cell `cell` holds any of the rows [first, first + count).
function cellInRange(cell, first, count) {
    if (count === 0) {
        return false;
    }
    const firstCell = Math.floor(first / MAP_ROWS);
    const lastCell = Math.floor((first + count - 1) / MAP_ROWS);
    return cell >= firstCell && cell <= lastCell;
}

// ---------------------------------------------------------------- setup

function buildMaps() {
    const ubMap = byId("ub-map");
    for (let i = 0; i < 384; i++) {
        const x = 316 + (i % UB_ACROSS) * 11;
        const y = 62 + Math.floor(i / UB_ACROSS) * 11;
        ui.ubCells.push(svg("rect", { x: x, y: y, width: 10, height: 10, rx: 1.5, class: "cell" }, ubMap));
    }

    const accMap = byId("acc-map");
    for (let i = 0; i < 16; i++) {
        const x = 686 + i * 15;
        ui.accCells.push(svg("rect", { x: x, y: 458, width: 13, height: 30, rx: 2, class: "cell" }, accMap));
    }

    const slots = byId("fifo-slots");
    for (let i = 0; i < 4; i++) {
        const y = 250 + i * 32;
        const name = svg("text", { x: 1006, y: y + 10, class: "slot-name" }, slots);
        svg("rect", { x: 1006, y: y + 16, width: 158, height: 6, rx: 3, class: "track" }, slots);
        const fill = svg("rect", { x: 1006, y: y + 16, width: 0, height: 6, rx: 3, class: "fill" }, slots);
        ui.fifoSlots.push({ name: name, fill: fill });
    }
}

function connect() {
    const badge = byId("connection");
    const source = new EventSource("/events");
    source.onopen = function () {
        badge.textContent = "live";
        badge.className = "badge";
    };
    source.onerror = function () {
        badge.textContent = "disconnected: restart ./tpu";
        badge.className = "badge off";
    };
    source.onmessage = function (event) {
        render(JSON.parse(event.data));
    };
}

// ---------------------------------------------------------------- the diagram

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
}

function renderArrows(s) {
    const flows = {
        "arrow-host-ub": s.host.busy && s.host.to_ub,
        "arrow-ub-host": s.host.busy && !s.host.to_ub,
        "arrow-ub-mxu": s.mxu.busy,
        "arrow-wmem-fifo": s.wmem.fetching >= 0,
        "arrow-fifo-mxu": s.mxu.shifting,
        "arrow-mxu-acc": s.mxu.busy,
        "arrow-acc-act": s.act.busy,
        "arrow-act-ub": s.act.busy,
    };
    for (const id of Object.keys(flows)) {
        byId(id).classList.toggle("active", flows[id]);
    }
}

function renderHost(s) {
    const host = s.host;
    if (host.busy) {
        const hostRows = "host " + rowRange(host.host_row, host.rows);
        const ubRows = "UB " + rowRange(host.ub_row, host.rows);
        if (host.to_ub) {
            byId("host-status").textContent = hostRows;
            byId("host-status-2").textContent = "→ " + ubRows;
        } else {
            byId("host-status").textContent = ubRows;
            byId("host-status-2").textContent = "→ " + hostRows;
        }
        byId("host-detail").textContent = host.done + " of " + host.total + " bytes moved";
    } else {
        byId("host-status").textContent = "idle";
        byId("host-status-2").textContent = "";
        byId("host-detail").textContent = "";
    }
    byId("host-pages").textContent = host.pages + " × 64 KiB pages in use";
}

// Shades cells that hold data, lights cells written recently, and outlines the rows units are touching now.
function renderMap(cells, map, writing, reading) {
    for (let i = 0; i < cells.length; i++) {
        const cell = cells[i];
        const age = map.age[i];
        const fresh = age >= 0 && age < FRESH_CYCLES;

        cell.classList.toggle("data", map.nonzero[i] === "#");
        cell.classList.toggle("fresh", fresh);
        if (fresh) {
            cell.style.fillOpacity = String(1 - age / FRESH_CYCLES);
        } else {
            cell.style.fillOpacity = "";
        }

        let isWriting = false;
        for (const range of writing) {
            isWriting = isWriting || cellInRange(i, range[0], range[1]);
        }
        let isReading = false;
        for (const range of reading) {
            isReading = isReading || cellInRange(i, range[0], range[1]);
        }
        cell.classList.toggle("writing", isWriting);
        cell.classList.toggle("reading", isReading && !isWriting);
    }
}

function renderUnifiedBuffer(s) {
    const writing = [];
    const reading = [];
    const said = [];
    if (s.host.busy && s.host.to_ub) {
        writing.push([s.host.ub_row, s.host.rows]);
        said.push("host writes " + rowRange(s.host.ub_row, s.host.rows));
    }
    if (s.host.busy && !s.host.to_ub) {
        reading.push([s.host.ub_row, s.host.rows]);
        said.push("host reads " + rowRange(s.host.ub_row, s.host.rows));
    }
    if (s.mxu.busy) {
        reading.push([s.mxu.ub_row, s.mxu.rows]);
        said.push("MXU reads " + rowRange(s.mxu.ub_row, s.mxu.rows));
    }
    if (s.act.busy) {
        writing.push([s.act.ub_row, s.act.out_rows]);
        said.push("activation writes " + rowRange(s.act.ub_row, s.act.out_rows));
    }
    renderMap(ui.ubCells, s.ub, writing, reading);
    if (said.length > 0) {
        byId("ub-status").textContent = said.join(" · ");
    } else {
        byId("ub-status").textContent = "idle";
    }
}

function renderAccumulators(s) {
    const writing = [];
    const reading = [];
    const said = [];
    if (s.mxu.busy) {
        writing.push([s.mxu.acc_row, s.mxu.rows]);
        said.push("MXU writes " + rowRange(s.mxu.acc_row, s.mxu.rows));
    }
    if (s.act.busy) {
        reading.push([s.act.acc_row, s.act.total]);
        said.push("activation reads " + rowRange(s.act.acc_row, s.act.total));
    }
    renderMap(ui.accCells, s.acc, writing, reading);
    byId("acc-status").textContent = "";
    byId("acc-status-2").textContent = "";
    if (said.length > 0) {
        byId("acc-status").textContent = said[0];
    }
    if (said.length > 1) {
        byId("acc-status-2").textContent = said[1];
    }
}

// Every PE, 256 x 256: the PEs holding data this cycle form diagonal bands, one per input row, sweeping down and right.
function renderMxuCanvas(s) {
    const mxu = s.mxu;
    const canvas = byId("mxu-canvas");
    const context = canvas.getContext("2d");
    const image = context.createImageData(mxu.dim, mxu.dim);

    const empty = rgb(cssColor("--pe-empty"));
    const loaded = rgb(cssColor("--pe-loaded"));
    const active = rgb(cssColor("--pe-active"));
    const activeAlt = rgb(cssColor("--pe-active-alt"));
    let base = empty;
    if (mxu.active_tile >= 0) {
        base = loaded;
    }

    // After step s the grid holds step s - 1: PE (k, n) carries input row (s - 1) - k - n.
    const lastStep = mxu.step - 1;
    for (let k = 0; k < mxu.dim; k++) {
        for (let n = 0; n < mxu.dim; n++) {
            const row = lastStep - k - n;
            const holdsData = mxu.busy && row >= 0 && row < mxu.rows;
            let color = base;
            if (holdsData && row % 2 === 0) {
                color = active;
            } else if (holdsData) {
                color = activeAlt;
            }
            const at = (k * mxu.dim + n) * 4;
            image.data[at] = color[0];
            image.data[at + 1] = color[1];
            image.data[at + 2] = color[2];
            image.data[at + 3] = 255;
        }
    }
    context.putImageData(image, 0, 0);
}

function renderMxu(s) {
    const mxu = s.mxu;
    renderMxuCanvas(s);

    if (mxu.busy) {
        byId("mxu-status").textContent = "step " + mxu.step + " of " + mxu.total;
        byId("mxu-flow").textContent = "UB " + rowRange(mxu.ub_row, mxu.rows) + " → acc " + rowRange(mxu.acc_row, mxu.rows);
    } else {
        byId("mxu-status").textContent = "idle";
        byId("mxu-flow").textContent = "";
    }

    if (mxu.active_tile >= 0) {
        byId("mxu-weights").textContent = "active weights: tile " + hex(mxu.active_tile);
    } else {
        byId("mxu-weights").textContent = "active weights: none yet";
    }

    let shadow = "shadow weights: empty";
    let shifted = 0;
    if (mxu.shadow_ready) {
        shadow = "shadow weights: tile " + hex(mxu.shadow_tile) + ", ready";
        shifted = 256;
    } else if (mxu.shifting) {
        shadow = "shadow weights: tile " + hex(mxu.shadow_tile) + ", " + mxu.rows_shifted + " of 256 rows shifted in";
        shifted = mxu.rows_shifted;
    }
    byId("mxu-shadow").textContent = shadow;
    byId("shadow-bar").setAttribute("width", String(238 * shifted / 256));
}

function renderWeights(s) {
    const wmem = s.wmem;
    if (wmem.fetching >= 0) {
        byId("wmem-status").textContent = "fetching tile " + hex(wmem.fetching);
    } else {
        byId("wmem-status").textContent = "idle";
    }

    let tiles = wmem.tiles + " tiles hold weights";
    if (wmem.tiles === 1) {
        tiles = "1 tile holds weights";
    }
    if (wmem.tiles > 0) {
        tiles = tiles + ": " + wmem.tile_list.slice(0, 6).map(hex).join(", ");
    }
    if (wmem.tiles > 6) {
        tiles = tiles + ", …";
    }
    byId("wmem-tiles").textContent = tiles;

    const fifo = s.wfifo;
    for (let i = 0; i < ui.fifoSlots.length; i++) {
        const slot = ui.fifoSlots[i];
        const entry = fifo.slots[i];
        if (entry === undefined) {
            slot.name.textContent = "slot " + i + ": empty";
            slot.fill.setAttribute("width", "0");
            continue;
        }
        let state = "ready";
        if (entry.arrived < fifo.tile_bytes) {
            state = Math.floor(100 * entry.arrived / fifo.tile_bytes) + "% arrived";
        }
        slot.name.textContent = "slot " + i + ": tile " + hex(entry.tile) + ", " + state;
        slot.fill.setAttribute("width", String(158 * entry.arrived / fifo.tile_bytes));
    }
}

function renderActivation(s) {
    const act = s.act;
    if (!act.busy) {
        byId("act-status").textContent = "idle";
        byId("act-detail").textContent = "";
        byId("act-progress").textContent = "";
        byId("act-bar").setAttribute("width", "0");
        return;
    }

    let how = act.function + ", shift " + act.shift;
    if (act.pool !== "none") {
        how = how + ", " + act.pool + " pool " + act.pool_size + "×" + act.pool_size + " over width " + act.pool_width;
    }
    byId("act-status").textContent = how;
    byId("act-detail").textContent = "acc " + rowRange(act.acc_row, act.total) + " → UB " + rowRange(act.ub_row, act.out_rows);
    byId("act-progress").textContent = act.done + " of " + act.total + " rows read";
    byId("act-bar").setAttribute("width", String(268 * act.done / act.total));
}

// ---------------------------------------------------------------- the panels

function renderHeader(s) {
    if (s.file === "") {
        byId("file").textContent = "no program loaded";
    } else {
        byId("file").textContent = s.file;
    }
    byId("cycle").textContent = String(s.cycle);
    byId("pc").textContent = String(s.pc);
    byId("issued").textContent = String(s.issued);

    const state = byId("state");
    state.textContent = s.state;
    state.classList.toggle("stalled", s.state.startsWith("stalled"));
}

function renderProgram(s) {
    const list = byId("program");
    const key = s.file + "\n" + s.program.join("\n");
    if (key !== ui.programKey) {
        ui.programKey = key;
        list.textContent = "";
        s.program.forEach(function (line, index) {
            const item = document.createElement("li");
            item.textContent = String(index).padStart(3) + "  " + line;
            list.appendChild(item);
        });
    }

    const items = list.children;
    for (let i = 0; i < items.length; i++) {
        const item = items[i];
        const current = i === s.pc;
        item.classList.toggle("done", i < s.pc);
        item.classList.toggle("current", current);
        item.classList.toggle("stalled", current && s.state.startsWith("stalled"));
    }

    const current = items[s.pc];
    if (current !== undefined) {
        list.scrollTop = current.offsetTop - list.clientHeight / 2;
    }
}

function buildTimelineKey(s) {
    const key = byId("timeline-key");
    const entries = [["I", "issued"]];
    for (const stall of s.stalls) {
        entries.push([stall.letter, stall.name]);
    }
    for (const entry of entries) {
        const item = document.createElement("span");
        const swatch = document.createElement("i");
        swatch.style.background = LETTER_COLORS[entry[0]];
        item.appendChild(swatch);
        item.appendChild(document.createTextNode(entry[1]));
        key.appendChild(item);
    }
    ui.keyBuilt = true;
}

// One column per cycle, newest on the right; the issue lane is colored by what happened, unit lanes by busy or idle.
function renderTimeline(s) {
    if (!ui.keyBuilt) {
        buildTimelineKey(s);
    }

    const t = s.timeline;
    const canvas = byId("timeline");
    let scale = 1;
    if (window.devicePixelRatio > 1) {
        scale = window.devicePixelRatio;
    }
    const shownWidth = canvas.clientWidth;
    const shownHeight = canvas.clientHeight;
    canvas.width = Math.round(shownWidth * scale);
    canvas.height = Math.round(shownHeight * scale);
    const context = canvas.getContext("2d");
    context.setTransform(scale, 0, 0, scale, 0, 0);
    const labelWidth = 70;
    const laneHeight = 22;
    const cycles = t.issue.length;
    const width = (shownWidth - labelWidth) / 200;

    context.clearRect(0, 0, shownWidth, shownHeight);
    context.font = "12px -apple-system, sans-serif";
    context.textBaseline = "middle";

    const busy = cssColor("--busy");
    const idle = cssColor("--line");
    TIMELINE_LANES.forEach(function (lane, laneIndex) {
        const y = laneIndex * (laneHeight + 3);
        context.fillStyle = cssColor("--dim");
        context.fillText(lane.label, 0, y + laneHeight / 2);

        const marks = t[lane.key];
        for (let c = 0; c < cycles; c++) {
            const letter = marks[c];
            let color = idle;
            if (lane.key === "issue" && letter in LETTER_COLORS) {
                color = LETTER_COLORS[letter];
            } else if (lane.key !== "issue" && letter === "#") {
                color = busy;
            }
            context.fillStyle = color;
            context.fillRect(labelWidth + c * width, y, Math.max(width - 0.5, 1), laneHeight);
        }
    });

    if (cycles > 0) {
        byId("timeline-range").textContent = "cycles " + t.first_cycle + "-" + (t.first_cycle + cycles - 1);
    } else {
        byId("timeline-range").textContent = "";
    }
}

function renderStalls(s) {
    const box = byId("stalls");
    box.textContent = "";
    let largest = 1;
    for (const stall of s.stalls) {
        largest = Math.max(largest, stall.cycles);
    }
    for (const stall of s.stalls) {
        const row = document.createElement("div");
        row.className = "stall-row";

        const name = document.createElement("span");
        name.textContent = stall.name;
        const track = document.createElement("span");
        const bar = document.createElement("div");
        bar.className = "stall-bar";
        bar.style.width = (100 * stall.cycles / largest) + "%";
        bar.style.background = LETTER_COLORS[stall.letter];
        track.appendChild(bar);
        const count = document.createElement("span");
        count.className = "stall-count";
        count.textContent = String(stall.cycles);

        row.appendChild(name);
        row.appendChild(track);
        row.appendChild(count);
        box.appendChild(row);
    }

    let share = 0;
    if (s.cycle > 0) {
        share = Math.round(100 * s.mxu_busy_cycles / s.cycle);
    }
    byId("utilization").textContent = "MXU busy " + s.mxu_busy_cycles + " of " + s.cycle + " cycles (" + share + "%)";
}

function renderPinned(s) {
    const box = byId("pinned");
    box.textContent = "";
    for (const view of s.pinned) {
        const section = document.createElement("div");
        section.className = "pinned-view";
        const title = document.createElement("h3");
        title.textContent = view.target;
        const text = document.createElement("pre");
        text.textContent = view.text;
        section.appendChild(title);
        section.appendChild(text);
        box.appendChild(section);
    }
}

function render(s) {
    renderHeader(s);
    renderBlocks(s);
    renderArrows(s);
    renderHost(s);
    renderUnifiedBuffer(s);
    renderMxu(s);
    renderWeights(s);
    renderAccumulators(s);
    renderActivation(s);
    renderProgram(s);
    renderTimeline(s);
    renderStalls(s);
    renderPinned(s);
}

buildMaps();
connect();
