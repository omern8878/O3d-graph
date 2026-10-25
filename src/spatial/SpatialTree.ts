import {
  NODE_TRANSFORM_STRIDE,
  NO_INDEX,
  NodeFlag,
  type Dimensions,
  type IGraphBuffers,
  type ISpatialTree,
} from '../types/index.js';
import { growFloat32, growInt32, nextCapacity } from '../utils/capacity.js';

/**
 * Maximum subdivision depth. Bodies that still share a cell at this depth are
 * kept together in the leaf's body list (exact pairwise interaction), which
 * bounds the tree for near-coincident floats and sizes traversal stacks.
 */
export const SPATIAL_TREE_MAX_DEPTH = 32;

/** Relative padding of the root cell so bodies on the bounding box edge stay inside. */
const ROOT_PADDING = 1e-4;
const MIN_HALF_SIZE = 1e-3;

/**
 * Non-allocating flat-array Barnes-Hut tree: quadtree (4 children) in 2D,
 * octree (8 children) in 3D.
 *
 * Layout (all indexed by cell index `c`):
 *  - `cellChildren[c * branching + k]`: child in octant `k`
 *    (`k = (x ≥ cx) | (y ≥ cy) << 1 | (z ≥ cz) << 2`), `NO_INDEX` if empty.
 *  - `cellFirstBody[c]`: head of the leaf's body list, `NO_INDEX` for internal
 *    cells; `bodyNext[i]` chains coincident / depth-capped bodies.
 *  - geometry (`cellCenter`, `cellHalfSize`) and aggregates
 *    (`cellMassCenter`, `cellMass`, `cellMaxRadius`).
 *
 * Children are always allocated after their parent, so aggregation is a single
 * reverse sweep over `[0, cellCount)` — no recursion, no stack.
 *
 * Buffers grow geometrically on demand and are reused across builds, so a
 * steady-state simulation performs zero allocation.
 */
export class SpatialTree implements ISpatialTree {
  private _dimensions: Dimensions;
  private _branching: 4 | 8;
  private _root = NO_INDEX;
  private _cellCount = 0;
  private cellCapacity: number;

  private _cellChildren: Int32Array;
  private _cellCenter: Float32Array;
  private _cellHalfSize: Float32Array;
  private _cellMassCenter: Float32Array;
  private _cellMass: Float32Array;
  /** Σ|mass|, used as the center-of-mass weight (charges may be negative). */
  private cellWeight: Float32Array;
  private _cellMaxRadius: Float32Array;
  private _cellFirstBody: Int32Array;
  private _cellBodyCount: Int32Array;
  private _cellOrderStart: Int32Array;
  private _bodyNext: Int32Array;
  private _bodyOrder: Int32Array;
  private _bodyCount = 0;
  private readonly dfsStack = new Int32Array(8 * (SPATIAL_TREE_MAX_DEPTH + 2));

  /** Positions of the graph currently being built (valid only inside `build`). */
  private positions: Float32Array = new Float32Array(0);

  constructor(dimensions: Dimensions = 3, initialCellCapacity = 1024) {
    this._dimensions = dimensions;
    this._branching = dimensions === 2 ? 4 : 8;
    const cap = nextCapacity(0, initialCellCapacity);
    this.cellCapacity = cap;
    this._cellChildren = new Int32Array(cap * this._branching).fill(NO_INDEX);
    this._cellCenter = new Float32Array(cap * 3);
    this._cellHalfSize = new Float32Array(cap);
    this._cellMassCenter = new Float32Array(cap * 3);
    this._cellMass = new Float32Array(cap);
    this.cellWeight = new Float32Array(cap);
    this._cellMaxRadius = new Float32Array(cap);
    this._cellFirstBody = new Int32Array(cap).fill(NO_INDEX);
    this._cellBodyCount = new Int32Array(cap);
    this._cellOrderStart = new Int32Array(cap);
    this._bodyNext = new Int32Array(0);
    this._bodyOrder = new Int32Array(0);
  }

  get dimensions(): Dimensions {
    return this._dimensions;
  }
  get branching(): 4 | 8 {
    return this._branching;
  }
  get root(): number {
    return this._root;
  }
  get cellCount(): number {
    return this._cellCount;
  }
  get cellChildren(): Int32Array {
    return this._cellChildren;
  }
  get cellCenter(): Float32Array {
    return this._cellCenter;
  }
  get cellHalfSize(): Float32Array {
    return this._cellHalfSize;
  }
  get cellMassCenter(): Float32Array {
    return this._cellMassCenter;
  }
  get cellMass(): Float32Array {
    return this._cellMass;
  }
  get cellMaxRadius(): Float32Array {
    return this._cellMaxRadius;
  }
  get cellFirstBody(): Int32Array {
    return this._cellFirstBody;
  }
  get bodyNext(): Int32Array {
    return this._bodyNext;
  }
  get cellBodyCount(): Int32Array {
    return this._cellBodyCount;
  }
  get cellOrderStart(): Int32Array {
    return this._cellOrderStart;
  }
  get bodyOrder(): Int32Array {
    return this._bodyOrder;
  }
  get bodyCount(): number {
    return this._bodyCount;
  }

  setDimensions(dimensions: Dimensions): void {
    if (dimensions === this._dimensions) return;
    this._dimensions = dimensions;
    this._branching = dimensions === 2 ? 4 : 8;
    this._cellChildren = new Int32Array(this.cellCapacity * this._branching).fill(NO_INDEX);
    this._root = NO_INDEX;
    this._cellCount = 0;
  }

  build(graph: IGraphBuffers): void {
    const n = graph.nodeCount;
    if (this._bodyNext.length < graph.nodeCapacity) {
      this._bodyNext = new Int32Array(graph.nodeCapacity);
      this._bodyOrder = new Int32Array(graph.nodeCapacity);
    }
    this._cellCount = 0;
    this._bodyCount = 0;
    this._root = NO_INDEX;
    if (n === 0) return;

    const p = graph.getNodeTransforms();
    const flags = graph.nodeFlags;
    const is3D = this._dimensions === 3;
    this.positions = p;

    // 1. Bounding cube over visible, finite bodies.
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    let visible = 0;
    for (let i = 0, o = 0; i < n; i++, o += NODE_TRANSFORM_STRIDE) {
      if ((flags[i] & NodeFlag.HIDDEN) !== 0) continue;
      const x = p[o], y = p[o + 1], z = is3D ? p[o + 2] : 0;
      if (x - x !== 0 || y - y !== 0 || z - z !== 0) continue; // NaN / ±Infinity
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
      visible++;
    }
    if (visible === 0) return;

    let half = Math.max(maxX - minX, maxY - minY, is3D ? maxZ - minZ : 0) * 0.5;
    half = Math.max(half * (1 + ROOT_PADDING), MIN_HALF_SIZE);
    this._root = this.allocCell((minX + maxX) * 0.5, (minY + maxY) * 0.5, is3D ? (minZ + maxZ) * 0.5 : 0, half);

    // 2. Insert bodies.
    let rootEmpty = true;
    for (let i = 0, o = 0; i < n; i++, o += NODE_TRANSFORM_STRIDE) {
      if ((flags[i] & NodeFlag.HIDDEN) !== 0) continue;
      const x = p[o], y = p[o + 1], z = is3D ? p[o + 2] : 0;
      if (x - x !== 0 || y - y !== 0 || z - z !== 0) continue;
      if (rootEmpty) {
        this._cellFirstBody[this._root] = i;
        this._bodyNext[i] = NO_INDEX;
        rootEmpty = false;
      } else {
        this.insert(i, x, y, z);
      }
    }

    // 3. Bottom-up aggregation.
    this.aggregate(graph.nodeMasses, graph.nodeRadii);

    // 4. Spatially coherent body order for force passes.
    this.collectBodyOrder();
  }

  // ---------------------------------------------------------------------------

  private insert(i: number, x: number, y: number, z: number): void {
    const p = this.positions;
    const is3D = this._dimensions === 3;
    const B = this._branching;
    let c = this._root;
    let depth = 0;

    for (;;) {
      const head = this._cellFirstBody[c];
      if (head !== NO_INDEX) {
        // Leaf: stack coincident bodies, otherwise split and push the
        // existing body list one level down.
        const ho = head * NODE_TRANSFORM_STRIDE;
        const hx = p[ho], hy = p[ho + 1], hz = is3D ? p[ho + 2] : 0;
        if (depth >= SPATIAL_TREE_MAX_DEPTH || (hx === x && hy === y && hz === z)) {
          this._bodyNext[i] = head;
          this._cellFirstBody[c] = i;
          return;
        }
        this._cellFirstBody[c] = NO_INDEX;
        const child = this.allocChild(c, this.octant(c, hx, hy, hz));
        this._cellFirstBody[child] = head;
        // `c` is now internal: fall through on the next iteration.
        continue;
      }

      const slot = c * B + this.octant(c, x, y, z);
      const child = this._cellChildren[slot];
      if (child === NO_INDEX) {
        const leaf = this.allocChild(c, slot - c * B);
        this._cellFirstBody[leaf] = i;
        this._bodyNext[i] = NO_INDEX;
        return;
      }
      c = child;
      depth++;
    }
  }

  private octant(c: number, x: number, y: number, z: number): number {
    const o = c * 3;
    const cc = this._cellCenter;
    let k = x >= cc[o] ? 1 : 0;
    if (y >= cc[o + 1]) k |= 2;
    if (this._dimensions === 3 && z >= cc[o + 2]) k |= 4;
    return k;
  }

  private allocChild(parent: number, k: number): number {
    const o = parent * 3;
    const h = this._cellHalfSize[parent] * 0.5;
    const cx = this._cellCenter[o] + ((k & 1) !== 0 ? h : -h);
    const cy = this._cellCenter[o + 1] + ((k & 2) !== 0 ? h : -h);
    const cz = this._dimensions === 3 ? this._cellCenter[o + 2] + ((k & 4) !== 0 ? h : -h) : 0;
    const child = this.allocCell(cx, cy, cz, h); // may reallocate arrays
    this._cellChildren[parent * this._branching + k] = child;
    return child;
  }

  private allocCell(cx: number, cy: number, cz: number, half: number): number {
    if (this._cellCount === this.cellCapacity) this.grow();
    const c = this._cellCount++;
    const o = c * 3;
    this._cellCenter[o] = cx;
    this._cellCenter[o + 1] = cy;
    this._cellCenter[o + 2] = cz;
    this._cellHalfSize[c] = half;
    this._cellFirstBody[c] = NO_INDEX;
    const B = this._branching;
    this._cellChildren.fill(NO_INDEX, c * B, c * B + B);
    return c;
  }

  private grow(): void {
    const cap = nextCapacity(this.cellCapacity, this.cellCapacity + 1);
    this._cellChildren = growInt32(this._cellChildren, cap * this._branching, NO_INDEX);
    this._cellCenter = growFloat32(this._cellCenter, cap * 3);
    this._cellHalfSize = growFloat32(this._cellHalfSize, cap);
    this._cellMassCenter = growFloat32(this._cellMassCenter, cap * 3);
    this._cellMass = growFloat32(this._cellMass, cap);
    this.cellWeight = growFloat32(this.cellWeight, cap);
    this._cellMaxRadius = growFloat32(this._cellMaxRadius, cap);
    this._cellFirstBody = growInt32(this._cellFirstBody, cap, NO_INDEX);
    this._cellBodyCount = growInt32(this._cellBodyCount, cap, 0);
    this._cellOrderStart = growInt32(this._cellOrderStart, cap, 0);
    this.cellCapacity = cap;
  }

  private collectBodyOrder(): void {
    const B = this._branching;
    const children = this._cellChildren;
    const first = this._cellFirstBody;
    const next = this._bodyNext;
    const order = this._bodyOrder;
    const orderStart = this._cellOrderStart;
    const stack = this.dfsStack;
    let count = 0;
    let sp = 0;
    stack[sp++] = this._root;
    while (sp > 0) {
      const c = stack[--sp];
      // Pre-order DFS keeps every subtree contiguous in `order`.
      orderStart[c] = count;
      const head = first[c];
      if (head !== NO_INDEX) {
        for (let j = head; j !== NO_INDEX; j = next[j]) order[count++] = j;
        continue;
      }
      // Push in reverse so octant 0 is visited first.
      for (let k = B - 1, base = c * B; k >= 0; k--) {
        const ch = children[base + k];
        if (ch !== NO_INDEX) stack[sp++] = ch;
      }
    }
    this._bodyCount = count;
  }

  private aggregate(masses: Float32Array, radii: Float32Array): void {
    const p = this.positions;
    const is3D = this._dimensions === 3;
    const B = this._branching;
    const children = this._cellChildren;
    const first = this._cellFirstBody;
    const next = this._bodyNext;
    const center = this._cellCenter;
    const mc = this._cellMassCenter;
    const mass = this._cellMass;
    const weight = this.cellWeight;
    const maxR = this._cellMaxRadius;
    const bodies = this._cellBodyCount;

    for (let c = this._cellCount - 1; c >= 0; c--) {
      let m = 0, w = 0, sx = 0, sy = 0, sz = 0, r = 0, nb = 0;
      const head = first[c];
      if (head !== NO_INDEX) {
        for (let j = head; j !== NO_INDEX; j = next[j]) {
          const mj = masses[j];
          const aw = mj < 0 ? -mj : mj;
          const o = j * NODE_TRANSFORM_STRIDE;
          m += mj;
          w += aw;
          sx += aw * p[o];
          sy += aw * p[o + 1];
          if (is3D) sz += aw * p[o + 2];
          if (radii[j] > r) r = radii[j];
          nb++;
        }
        if (w === 0) {
          // Massless bodies: fall back to the head body's position.
          const o = head * NODE_TRANSFORM_STRIDE;
          sx = p[o];
          sy = p[o + 1];
          sz = is3D ? p[o + 2] : 0;
        }
      } else {
        for (let k = 0, base = c * B; k < B; k++) {
          const ch = children[base + k];
          if (ch === NO_INDEX) continue;
          const aw = weight[ch];
          const o = ch * 3;
          m += mass[ch];
          w += aw;
          sx += aw * mc[o];
          sy += aw * mc[o + 1];
          sz += aw * mc[o + 2];
          if (maxR[ch] > r) r = maxR[ch];
          nb += bodies[ch];
        }
        if (w === 0) {
          const o = c * 3;
          sx = center[o];
          sy = center[o + 1];
          sz = center[o + 2];
        }
      }
      const inv = w > 0 ? 1 / w : 1;
      const o = c * 3;
      mc[o] = sx * inv;
      mc[o + 1] = sy * inv;
      mc[o + 2] = sz * inv;
      mass[c] = m;
      weight[c] = w;
      maxR[c] = r;
      bodies[c] = nb;
    }
  }
}
