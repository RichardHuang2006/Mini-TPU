// Mini-TPU trace container (.mtpt) reader.
//
// Layout (tools/tracegen.cpp): "MTPT", u32 version, u32 manifest length, the
// JSON manifest, padding to 8 bytes, then 8-byte-aligned little-endian
// sections the manifest indexes by name. Everything except the per-MatMul PE
// detail ("mm.<i>.*") is loaded eagerly; PE detail is sliced on demand and
// kept in a small cache, so the 33 MB large container costs about 1 MB of
// memory until the array view is opened.
//
// Classic script: defines window.MTV.Container.
(function () {
  'use strict';
  const MTV = (window.MTV = window.MTV || {});

  const CTOR = { u8: Uint8Array, i8: Int8Array, u32: Uint32Array, i32: Int32Array };

  // Blob.arrayBuffer() where available, FileReader otherwise.
  function blobToArrayBuffer(blob) {
    if (typeof blob.arrayBuffer === 'function') return blob.arrayBuffer();
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => resolve(fr.result);
      fr.onerror = () => reject(fr.error || new Error('read failed'));
      fr.readAsArrayBuffer(blob);
    });
  }

  function base64ToBytes(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  class Container {
    constructor() {
      this.manifest = null;
      this.base = 0;          // file offset of section 0
      this.sections = {};     // name -> {name, dtype, count, off, len}
      this.eager = {};        // name -> typed array
      this.mmCache = new Map();
      this.mmOrder = [];
      this.sourceName = '';
    }

    // source: ArrayBuffer | Uint8Array | File | {base64}
    static async open(source, name) {
      const c = new Container();
      c.sourceName = name || (source && source.name) || 'trace';
      if (source && typeof source.base64 === 'string') {
        const bytes = base64ToBytes(source.base64);
        source = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
      }
      if (source instanceof Uint8Array) {
        source = source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength);
      }
      if (source instanceof ArrayBuffer) {
        c.buffer = source;
        c.file = null;
        c.slice = (off, len) => Promise.resolve(source.slice(off, off + len));
        c.sliceSync = (off, len) => source.slice(off, off + len);
      } else if (typeof Blob !== 'undefined' && source instanceof Blob) {
        c.file = source;
        c.buffer = null;
        c.slice = (off, len) => blobToArrayBuffer(source.slice(off, off + len));
        c.sliceSync = null;
      } else {
        throw new Error('unsupported trace source');
      }
      await c.parseHeader();
      await c.loadEager();
      return c;
    }

    async parseHeader() {
      const head = new DataView(await this.slice(0, 12));
      const magic = String.fromCharCode(head.getUint8(0), head.getUint8(1), head.getUint8(2), head.getUint8(3));
      if (magic !== 'MTPT') throw new Error('not a Mini-TPU trace container (bad magic)');
      const version = head.getUint32(4, true);
      if (version !== 1) throw new Error('unsupported container version ' + version);
      const mlen = head.getUint32(8, true);
      const mbytes = await this.slice(12, mlen);
      this.manifest = JSON.parse(new TextDecoder('utf-8').decode(mbytes));
      if (this.manifest.format !== 'mtpt/1') throw new Error('unsupported manifest format ' + this.manifest.format);
      let base = 12 + mlen;
      base += (8 - (base % 8)) % 8;
      this.base = base;
      for (const s of this.manifest.sections) this.sections[s.name] = s;
    }

    isLazy(name) { return name.startsWith('mm.'); }

    async loadEager() {
      // Eager sections are contiguous and precede the PE detail.
      let end = 0;
      let firstLazy = Infinity;
      for (const s of this.manifest.sections) {
        if (this.isLazy(s.name)) { firstLazy = Math.min(firstLazy, s.off); continue; }
        end = Math.max(end, s.off + s.len);
      }
      if (end > firstLazy) throw new Error('container layout: eager sections after PE detail');
      const buf = await this.slice(this.base, end);
      for (const s of this.manifest.sections) {
        if (this.isLazy(s.name)) continue;
        this.eager[s.name] = new CTOR[s.dtype](buf, s.off, s.count);
      }
    }

    // Typed array view of an eager section.
    sec(name) {
      const a = this.eager[name];
      if (!a) throw new Error('section not loaded: ' + name);
      return a;
    }

    has(name) { return Object.prototype.hasOwnProperty.call(this.sections, name); }

    // PE detail for MatMul i: {w, left, act, psum, land} typed arrays.
    async mm(i) {
      if (this.mmCache.has(i)) return this.mmCache.get(i);
      const names = ['w', 'left', 'act', 'psum', 'land'].map((k) => 'mm.' + i + '.' + k);
      const secs = names.map((n) => this.sections[n]);
      if (secs.some((s) => !s)) throw new Error('no PE detail for matmul ' + i);
      const lo = Math.min(...secs.map((s) => s.off));
      const hi = Math.max(...secs.map((s) => s.off + s.len));
      const buf = await this.slice(this.base + lo, hi - lo);
      const out = {};
      ['w', 'left', 'act', 'psum', 'land'].forEach((k, j) => {
        const s = secs[j];
        out[k] = new CTOR[s.dtype](buf, s.off - lo, s.count);
      });
      this.mmCache.set(i, out);
      this.mmOrder.push(i);
      while (this.mmOrder.length > 8) {
        const old = this.mmOrder.shift();
        this.mmCache.delete(old);
      }
      return out;
    }

    // Synchronous variant when the whole container is in memory (the embedded
    // example), so the array view can render without an await.
    mmSync(i) {
      if (this.mmCache.has(i)) return this.mmCache.get(i);
      if (!this.sliceSync) return null;
      const names = ['w', 'left', 'act', 'psum', 'land'].map((k) => 'mm.' + i + '.' + k);
      const secs = names.map((n) => this.sections[n]);
      if (secs.some((s) => !s)) return null;
      const out = {};
      ['w', 'left', 'act', 'psum', 'land'].forEach((k, j) => {
        const s = secs[j];
        out[k] = new CTOR[s.dtype](this.buffer, this.base + s.off, s.count);
      });
      this.mmCache.set(i, out);
      this.mmOrder.push(i);
      return out;
    }

    mmLoaded(i) { return this.mmCache.has(i); }
  }

  MTV.Container = Container;
  MTV.base64ToBytes = base64ToBytes;
})();
