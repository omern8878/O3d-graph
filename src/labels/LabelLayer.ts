import {
  NODE_TRANSFORM_STRIDE,
  NO_INDEX,
  NodeFlag,
  type ICameraAdapter,
  type IGraphStore,
} from '../types/index.js';

/** What a `LabelLayer` reads from. `O3DGraph` satisfies it as-is. */
export interface LabelSource {
  readonly store: IGraphStore<unknown, unknown>;
  readonly renderer: { readonly camera: ICameraAdapter };
  /** The graph canvas the labels are aligned with. */
  readonly canvas: HTMLCanvasElement;
}

/**
 * Which edge labels may appear:
 *  - `'auto'`: edges touching the hovered/selected node first, then any
 *    labeled edge long enough on screen, while the budget lasts
 *  - `'focused'`: only edges touching the hovered/selected node
 *  - `'none'`
 */
export type EdgeLabelMode = 'auto' | 'focused' | 'none';

export interface LabelLayerOptions {
  /** Overlay canvas to draw into. Omit to have one created on top of the graph canvas. */
  canvas?: HTMLCanvasElement;
  /** Max node labels per frame. Default `40`. */
  maxNodeLabels?: number;
  /** Max edge labels per frame. Default `16`. */
  maxEdgeLabels?: number;
  /** Nodes smaller than this on screen (radius, CSS px) are only labeled when focused. Default `3`. */
  minNodeRadiusPx?: number;
  /** Default `'auto'`. */
  edgeLabels?: EdgeLabelMode;
  /** Where node labels go: centered on the ball (default) or to its right. */
  nodeLabelPosition?: 'center' | 'right';
  /** Default `'ui-sans-serif, system-ui, sans-serif'`. */
  fontFamily?: string;
  /** Default `600`. */
  fontWeight?: number | string;
  /** Default `500`. */
  edgeFontWeight?: number | string;
  /**
   * Node label size relative to the ball's on-screen radius, so text grows
   * as you zoom in. Default `0.8`, clamped to `[minFontSize, maxFontSize]`.
   */
  fontScale?: number;
  /** Default `11`. */
  minFontSize?: number;
  /** Default `30`. */
  maxFontSize?: number;
  /** Edge label size relative to the endpoints' mean on-screen radius. Default `0.55`. */
  edgeFontScale?: number;
  /** Default `9`. */
  minEdgeFontSize?: number;
  /** Default `18`. */
  maxEdgeFontSize?: number;
  /** Node label color. Default `'rgba(232, 235, 242, 0.92)'`. */
  color?: string;
  /** Color for the hovered/selected node and its edges. Default `'#ffffff'`. */
  focusColor?: string;
  /** Edge label color. Default `'rgba(200, 207, 222, 0.9)'`. */
  edgeColor?: string;
  /** Halo behind node text and fill of edge label pills. Default `'rgba(7, 10, 16, 0.82)'`. */
  haloColor?: string;
  /** Minimum halo stroke width (CSS px); it grows with the font. Default `3`. */
  haloWidth?: number;
  /** Node label source. Defaults to the node's `label`. */
  nodeText?: (index: number) => string | undefined;
  /** Edge label source. Defaults to the edge's `label`. */
  edgeText?: (index: number) => string | undefined;
}

const FOCUS_SCORE = 1e9;
const NEIGHBOR_SCORE = 1e6;
const FOCUS_FLAGS = NodeFlag.HOVERED | NodeFlag.SELECTED;
const PAD = 4;
const MARGIN = 2;
const OVERSAMPLE = 4; // candidates kept per label slot (collisions drop some)
const MAX_TEXT_CACHE = 8192;
/** Widths are measured once at this size and scaled linearly. */
const REF_FONT_SIZE = 100;

/**
 * Draws node and edge labels on a 2D canvas overlay, choosing per frame
 * which ones are worth showing:
 *
 * 1. the hovered / selected nodes,
 * 2. the edges touching them,
 * 3. their neighbors, then the nodes that are largest on screen (i.e. big
 *    and close to the camera),
 * 4. other labeled edges, longest on screen first (`edgeLabels: 'auto'`).
 *
 * Labels that would overlap an already placed one are skipped, so the
 * budget goes to legible text. Node labels are centered on the ball (or
 * beside it) and sized from its on-screen radius, so zooming in enlarges
 * them; edge labels sit on the visible part of the segment, rotated along it
 * and kept upright.
 *
 * Positions come from the camera's view-projection matrix, so labels track
 * any renderer/camera. Call `render()` after each rendered frame — it does
 * no work beyond a projection pass and a bounded candidate selection, and
 * reuses all its buffers.
 */
export class LabelLayer {
  enabled = true;
  /** Which edge labels may appear; can be changed between frames. */
  edgeLabels: EdgeLabelMode;
  readonly canvas: HTMLCanvasElement;

  private readonly source: LabelSource;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly ownsCanvas: boolean;
  private readonly opts: Required<Omit<LabelLayerOptions, 'canvas' | 'nodeText' | 'edgeText' | 'edgeLabels'>>;
  private readonly nodeText: (index: number) => string | undefined;
  private readonly edgeText: (index: number) => string | undefined;

  private width = 0;
  private height = 0;
  private cleared = true;

  // Per-node projection (CSS px) and focus marks, grown with the graph.
  private sx = new Float32Array(0);
  private sy = new Float32Array(0);
  private sr = new Float32Array(0);
  private visible = new Uint8Array(0);
  private focusMark = new Uint32Array(0);
  private neighborMark = new Uint32Array(0);
  private generation = 0;

  // Bounded candidate lists (sorted descending by score).
  private readonly nodeCand: Candidates;
  private readonly edgeCand: Candidates;

  // Placed label rects for collision tests: [x0, y0, x1, y1] each.
  private readonly rects: Float32Array;
  private rectCount = 0;

  private readonly nodeWidths = new Map<string, number>();
  private readonly edgeWidths = new Map<string, number>();

  /** Indices labeled in the last frame (valid up to the counts below). */
  readonly placedNodes: Int32Array;
  readonly placedEdges: Int32Array;
  placedNodeCount = 0;
  placedEdgeCount = 0;

  constructor(source: LabelSource, options: LabelLayerOptions = {}) {
    this.source = source;
    this.opts = {
      maxNodeLabels: Math.max(0, Math.floor(options.maxNodeLabels ?? 40)),
      maxEdgeLabels: Math.max(0, Math.floor(options.maxEdgeLabels ?? 16)),
      minNodeRadiusPx: options.minNodeRadiusPx ?? 3,
      nodeLabelPosition: options.nodeLabelPosition ?? 'center',
      fontFamily: options.fontFamily ?? 'ui-sans-serif, system-ui, sans-serif',
      fontWeight: options.fontWeight ?? 600,
      edgeFontWeight: options.edgeFontWeight ?? 500,
      fontScale: options.fontScale ?? 0.8,
      minFontSize: options.minFontSize ?? 11,
      maxFontSize: options.maxFontSize ?? 30,
      edgeFontScale: options.edgeFontScale ?? 0.55,
      minEdgeFontSize: options.minEdgeFontSize ?? 9,
      maxEdgeFontSize: options.maxEdgeFontSize ?? 18,
      color: options.color ?? 'rgba(232, 235, 242, 0.92)',
      focusColor: options.focusColor ?? '#ffffff',
      edgeColor: options.edgeColor ?? 'rgba(200, 207, 222, 0.9)',
      haloColor: options.haloColor ?? 'rgba(7, 10, 16, 0.82)',
      haloWidth: options.haloWidth ?? 3,
    };
    this.edgeLabels = options.edgeLabels ?? 'auto';
    this.nodeText = options.nodeText ?? ((i) => source.store.getNodeLabel(i));
    this.edgeText = options.edgeText ?? ((e) => source.store.getEdgeLabel(e));

    this.nodeCand = new Candidates(Math.max(1, this.opts.maxNodeLabels * OVERSAMPLE));
    this.edgeCand = new Candidates(Math.max(1, this.opts.maxEdgeLabels * OVERSAMPLE));
    this.rects = new Float32Array((this.opts.maxNodeLabels + this.opts.maxEdgeLabels) * 4);
    this.placedNodes = new Int32Array(this.opts.maxNodeLabels);
    this.placedEdges = new Int32Array(this.opts.maxEdgeLabels);

    if (options.canvas) {
      this.canvas = options.canvas;
      this.ownsCanvas = false;
    } else {
      this.canvas = document.createElement('canvas');
      this.canvas.setAttribute('aria-hidden', 'true');
      Object.assign(this.canvas.style, { position: 'fixed', pointerEvents: 'none', left: '0px', top: '0px' });
      source.canvas.insertAdjacentElement('afterend', this.canvas);
      this.ownsCanvas = true;
    }
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('LabelLayer: 2D canvas context unavailable');
    this.ctx = ctx;
  }

  /** Redraws the labels for the current camera and graph state. */
  render(): void {
    this.placedNodeCount = 0;
    this.placedEdgeCount = 0;
    this.syncCanvas();
    const ctx = this.ctx;
    if (!this.cleared) {
      ctx.clearRect(0, 0, this.width, this.height);
      this.cleared = true;
    }
    const store = this.source.store;
    if (!this.enabled || store.nodeCount === 0) return;

    this.project();
    this.markFocus();
    this.collectNodes();
    this.collectEdges();

    this.rectCount = 0;
    ctx.lineJoin = 'round';
    ctx.textBaseline = 'middle';
    // Focused nodes, then their edges, then everything else by score.
    this.placeNodes(true);
    this.placeEdges(true);
    this.placeNodes(false);
    this.placeEdges(false);
    this.cleared = this.placedNodeCount === 0 && this.placedEdgeCount === 0;
  }

  dispose(): void {
    if (this.ownsCanvas) this.canvas.remove();
    else this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  // ===========================================================================
  // Frame stages
  // ===========================================================================

  private syncCanvas(): void {
    const rect = this.source.canvas.getBoundingClientRect();
    const dpr = Math.min(globalThis.devicePixelRatio || 1, 2);
    if (this.ownsCanvas) {
      const style = this.canvas.style;
      const left = `${rect.left}px`;
      const top = `${rect.top}px`;
      const width = `${rect.width}px`;
      const height = `${rect.height}px`;
      if (style.left !== left) style.left = left;
      if (style.top !== top) style.top = top;
      if (style.width !== width) style.width = width;
      if (style.height !== height) style.height = height;
    }
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.cleared = true; // resizing clears the backing store
    }
    this.width = rect.width;
    this.height = rect.height;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /** Screen position and on-screen radius of every visible node. */
  private project(): void {
    const store = this.source.store;
    const n = store.nodeCount;
    if (this.sx.length < store.nodeCapacity) {
      const cap = store.nodeCapacity;
      this.sx = new Float32Array(cap);
      this.sy = new Float32Array(cap);
      this.sr = new Float32Array(cap);
      this.visible = new Uint8Array(cap);
      this.focusMark = new Uint32Array(cap);
      this.neighborMark = new Uint32Array(cap);
      this.generation = 0;
    }
    const m = this.source.renderer.camera.getCameraViewProjMatrix();
    const p = store.getNodeTransforms();
    const flags = store.nodeFlags;
    const radii = store.nodeRadii;
    const halfW = this.width / 2;
    const halfH = this.height / 2;
    // World → clip-space y scale, independent of the view rotation.
    const yScale = Math.hypot(m[1], m[5], m[9]) * halfH;
    const { sx, sy, sr, visible } = this;

    for (let i = 0, o = 0; i < n; i++, o += NODE_TRANSFORM_STRIDE) {
      visible[i] = 0;
      if ((flags[i] & NodeFlag.HIDDEN) !== 0) continue;
      const x = p[o], y = p[o + 1], z = p[o + 2];
      const w = m[3] * x + m[7] * y + m[11] * z + m[15];
      if (w <= 1e-6) continue; // behind the camera
      const inv = 1 / w;
      const nz = (m[2] * x + m[6] * y + m[10] * z + m[14]) * inv;
      if (nz > 1 || nz < -1) continue; // outside near/far (covers GL and WebGPU depth)
      const nx = (m[0] * x + m[4] * y + m[8] * z + m[12]) * inv;
      const ny = (m[1] * x + m[5] * y + m[9] * z + m[13]) * inv;
      if (nx < -1.1 || nx > 1.1 || ny < -1.1 || ny > 1.1) continue;
      sx[i] = (nx + 1) * halfW;
      sy[i] = (1 - ny) * halfH;
      sr[i] = radii[i] * yScale * inv;
      visible[i] = 1;
    }
  }

  /** Marks hovered/selected nodes and their neighbors for this frame. */
  private markFocus(): void {
    if (++this.generation === 0xffffffff) {
      this.focusMark.fill(0);
      this.neighborMark.fill(0);
      this.generation = 1;
    }
    const gen = this.generation;
    const store = this.source.store;
    const flags = store.nodeFlags;
    const first = store.nodeFirstHalfEdge;
    const next = store.halfEdgeNext;
    for (let i = 0, n = store.nodeCount; i < n; i++) {
      if ((flags[i] & FOCUS_FLAGS) === 0) continue;
      this.focusMark[i] = gen;
      for (let h = first[i]; h !== NO_INDEX; h = next[h]) {
        const e = h >> 1;
        this.neighborMark[(h & 1) === 0 ? store.edgeTargets[e] : store.edgeSources[e]] = gen;
      }
    }
  }

  private collectNodes(): void {
    const cand = this.nodeCand;
    cand.reset();
    if (this.opts.maxNodeLabels === 0) return;
    const gen = this.generation;
    const minR = this.opts.minNodeRadiusPx;
    for (let i = 0, n = this.source.store.nodeCount; i < n; i++) {
      if (this.visible[i] === 0) continue;
      const focus = this.focusMark[i] === gen;
      const neighbor = this.neighborMark[i] === gen;
      const r = this.sr[i];
      if (!focus && !neighbor && r < minR) continue;
      const score = r + (focus ? FOCUS_SCORE : neighbor ? NEIGHBOR_SCORE : 0);
      if (!cand.wouldAccept(score)) continue; // skip the text lookup for hopeless nodes
      if (!this.nodeText(i)) continue;
      cand.push(i, score);
    }
    cand.sortDescending();
  }

  private collectEdges(): void {
    const cand = this.edgeCand;
    cand.reset();
    const mode = this.edgeLabels;
    if (mode === 'none' || this.opts.maxEdgeLabels === 0) return;
    const store = this.source.store;
    const gen = this.generation;
    const src = store.edgeSources;
    const tgt = store.edgeTargets;
    for (let e = 0, m = store.edgeCount; e < m; e++) {
      const s = src[e];
      const t = tgt[e];
      if (s === t || this.visible[s] === 0 || this.visible[t] === 0) continue;
      const focused = this.focusMark[s] === gen || this.focusMark[t] === gen;
      if (mode === 'focused' && !focused) continue;
      const len = Math.hypot(this.sx[t] - this.sx[s], this.sy[t] - this.sy[s]) - this.sr[s] - this.sr[t];
      if (len <= 0) continue;
      const score = len + (focused ? FOCUS_SCORE : 0);
      if (!cand.wouldAccept(score)) continue;
      if (!this.edgeText(e)) continue;
      cand.push(e, score);
    }
    cand.sortDescending();
  }

  // ===========================================================================
  // Placement & drawing
  // ===========================================================================

  private placeNodes(focusedPass: boolean): void {
    const cand = this.nodeCand;
    const max = this.opts.maxNodeLabels;
    for (let k = 0; k < cand.size && this.placedNodeCount < max; k++) {
      const score = cand.scores[k];
      const isFocus = score >= FOCUS_SCORE;
      if (isFocus !== focusedPass) continue;
      const i = cand.indices[k];
      const text = this.nodeText(i);
      if (text && this.drawNodeLabel(i, text, isFocus)) this.placedNodes[this.placedNodeCount++] = i;
    }
  }

  private placeEdges(focusedPass: boolean): void {
    const cand = this.edgeCand;
    const max = this.opts.maxEdgeLabels;
    for (let k = 0; k < cand.size && this.placedEdgeCount < max; k++) {
      const isFocus = cand.scores[k] >= FOCUS_SCORE;
      if (isFocus !== focusedPass) continue;
      const e = cand.indices[k];
      const text = this.edgeText(e);
      if (text && this.drawEdgeLabel(e, text, isFocus)) this.placedEdges[this.placedEdgeCount++] = e;
    }
  }

  private drawNodeLabel(i: number, text: string, focus: boolean): boolean {
    const o = this.opts;
    const x = this.sx[i];
    const y = this.sy[i];
    const r = this.sr[i];
    const size = Math.round(clamp(r * o.fontScale, o.minFontSize, o.maxFontSize));
    const w = this.measure(text, o.fontWeight, this.nodeWidths) * (size / REF_FONT_SIZE);
    const h = size * 1.25;
    const y0 = y - h / 2;
    if (y0 < 0 || y0 + h > this.height) return false;

    let x0: number;
    let align: CanvasTextAlign;
    if (o.nodeLabelPosition === 'center') {
      x0 = x - w / 2;
      align = 'center';
      if (x0 < MARGIN || x0 + w > this.width - MARGIN || this.collides(x0, y0, x0 + w, y0 + h)) return false;
    } else {
      // Prefer the right side; flip left at the edge or when the right is taken.
      x0 = x + r + PAD;
      align = 'left';
      if (x0 + w > this.width - MARGIN || this.collides(x0, y0, x0 + w, y0 + h)) {
        x0 = x - r - PAD - w;
        align = 'right';
        if (x0 < MARGIN || this.collides(x0, y0, x0 + w, y0 + h)) return false;
      }
    }

    const ctx = this.ctx;
    ctx.font = `${o.fontWeight} ${size}px ${o.fontFamily}`;
    ctx.textAlign = align;
    const tx = align === 'center' ? x : align === 'left' ? x0 : x0 + w;
    ctx.lineWidth = Math.max(o.haloWidth, size * 0.28);
    ctx.strokeStyle = o.haloColor;
    ctx.strokeText(text, tx, y);
    ctx.fillStyle = focus ? o.focusColor : o.color;
    ctx.fillText(text, tx, y);
    this.addRect(x0, y0, x0 + w, y0 + h);
    return true;
  }

  private drawEdgeLabel(e: number, text: string, focus: boolean): boolean {
    const store = this.source.store;
    const s = store.edgeSources[e];
    const t = store.edgeTargets[e];
    const ax = this.sx[s], ay = this.sy[s], bx = this.sx[t], by = this.sy[t];
    const dx = bx - ax;
    const dy = by - ay;
    const len = Math.hypot(dx, dy);
    if (len === 0) return false;

    const o = this.opts;
    const size = Math.round(
      clamp(((this.sr[s] + this.sr[t]) / 2) * o.edgeFontScale, o.minEdgeFontSize, o.maxEdgeFontSize),
    );
    const tw = this.measure(text, o.edgeFontWeight, this.edgeWidths) * (size / REF_FONT_SIZE);
    const pillW = tw + size * 0.8;
    const pillH = size * 1.5;
    const visibleLen = len - this.sr[s] - this.sr[t];
    if (visibleLen < pillW + PAD) return false; // wouldn't fit between the circles

    // Center on the visible part of the segment (between the two circles).
    const along = (this.sr[s] + (len - this.sr[t])) / 2 / len;
    const cx = ax + dx * along;
    const cy = ay + dy * along;
    let angle = Math.atan2(dy, dx);
    if (angle > Math.PI / 2) angle -= Math.PI;
    else if (angle < -Math.PI / 2) angle += Math.PI;

    // Axis-aligned bounds of the rotated pill for collision tests.
    const c = Math.abs(Math.cos(angle));
    const sn = Math.abs(Math.sin(angle));
    const hw = (pillW * c + pillH * sn) / 2;
    const hh = (pillW * sn + pillH * c) / 2;
    const x0 = cx - hw, y0 = cy - hh, x1 = cx + hw, y1 = cy + hh;
    if (x0 < 0 || y0 < 0 || x1 > this.width || y1 > this.height) return false;
    if (this.collides(x0, y0, x1, y1)) return false;

    const ctx = this.ctx;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);
    ctx.fillStyle = o.haloColor;
    roundedRect(ctx, -pillW / 2, -pillH / 2, pillW, pillH, pillH / 2);
    ctx.fill();
    ctx.font = `${o.edgeFontWeight} ${size}px ${o.fontFamily}`;
    ctx.textAlign = 'center';
    ctx.fillStyle = focus ? o.focusColor : o.edgeColor;
    ctx.fillText(text, 0, 0.5);
    ctx.restore();
    this.addRect(x0, y0, x1, y1);
    return true;
  }

  /** Width at `REF_FONT_SIZE`; callers scale it linearly to the size they draw at. */
  private measure(text: string, weight: number | string, cache: Map<string, number>): number {
    let w = cache.get(text);
    if (w === undefined) {
      if (cache.size >= MAX_TEXT_CACHE) cache.clear();
      this.ctx.font = `${weight} ${REF_FONT_SIZE}px ${this.opts.fontFamily}`;
      w = this.ctx.measureText(text).width;
      cache.set(text, w);
    }
    return w;
  }

  private collides(x0: number, y0: number, x1: number, y1: number): boolean {
    const r = this.rects;
    for (let k = 0, o = 0; k < this.rectCount; k++, o += 4) {
      if (x0 < r[o + 2] + MARGIN && x1 > r[o] - MARGIN && y0 < r[o + 3] + MARGIN && y1 > r[o + 1] - MARGIN) {
        return true;
      }
    }
    return false;
  }

  private addRect(x0: number, y0: number, x1: number, y1: number): void {
    const o = this.rectCount++ * 4;
    this.rects[o] = x0;
    this.rects[o + 1] = y0;
    this.rects[o + 2] = x1;
    this.rects[o + 3] = y1;
  }
}

/** Fixed-capacity top-k by score (min-heap while collecting, then sorted descending). */
class Candidates {
  readonly indices: Int32Array;
  readonly scores: Float64Array;
  size = 0;

  constructor(readonly capacity: number) {
    this.indices = new Int32Array(capacity);
    this.scores = new Float64Array(capacity);
  }

  reset(): void {
    this.size = 0;
  }

  wouldAccept(score: number): boolean {
    return this.size < this.capacity || score > this.scores[0];
  }

  push(index: number, score: number): void {
    const idx = this.indices;
    const sc = this.scores;
    if (this.size < this.capacity) {
      // Sift up.
      let k = this.size++;
      while (k > 0) {
        const parent = (k - 1) >> 1;
        if (sc[parent] <= score) break;
        idx[k] = idx[parent];
        sc[k] = sc[parent];
        k = parent;
      }
      idx[k] = index;
      sc[k] = score;
      return;
    }
    if (score <= sc[0]) return;
    // Replace the minimum and sift down.
    let k = 0;
    const n = this.size;
    for (;;) {
      const l = 2 * k + 1;
      if (l >= n) break;
      const r = l + 1;
      const c = r < n && sc[r] < sc[l] ? r : l;
      if (sc[c] >= score) break;
      idx[k] = idx[c];
      sc[k] = sc[c];
      k = c;
    }
    idx[k] = index;
    sc[k] = score;
  }

  /** Insertion sort, descending (capacity is small: a few hundred at most). */
  sortDescending(): void {
    const idx = this.indices;
    const sc = this.scores;
    for (let a = 1; a < this.size; a++) {
      const s = sc[a];
      const i = idx[a];
      let b = a - 1;
      while (b >= 0 && sc[b] < s) {
        sc[b + 1] = sc[b];
        idx[b + 1] = idx[b];
        b--;
      }
      sc[b + 1] = s;
      idx[b + 1] = i;
    }
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function roundedRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.lineTo(x + w - rr, y);
  ctx.arcTo(x + w, y, x + w, y + rr, rr);
  ctx.lineTo(x + w, y + h - rr);
  ctx.arcTo(x + w, y + h, x + w - rr, y + h, rr);
  ctx.lineTo(x + rr, y + h);
  ctx.arcTo(x, y + h, x, y + h - rr, rr);
  ctx.lineTo(x, y + rr);
  ctx.arcTo(x, y, x + rr, y, rr);
  ctx.closePath();
}
