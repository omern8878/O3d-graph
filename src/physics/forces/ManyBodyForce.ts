import {
  NODE_TRANSFORM_STRIDE,
  NO_INDEX,
  VEC3_STRIDE,
  type Dimensions,
  type ForceTickContext,
  type IForce,
  type IGraphBuffers,
  type ISpatialTree,
} from '../../types/index.js';
import { SPATIAL_TREE_MAX_DEPTH } from '../../spatial/SpatialTree.js';
import { Lcg, jiggle } from '../../utils/random.js';

export interface ManyBodyForceOptions {
  /** Per-unit-mass strength; negative repels, positive attracts. Default `-30`. */
  strength?: number;
  /** Barnes-Hut opening angle θ. Default `0.8`. */
  theta?: number;
  /** Interactions closer than this are softened. Default `1`. */
  distanceMin?: number;
  /** Interactions beyond this are ignored. Default `Infinity`. */
  distanceMax?: number;
  /** Max bodies per interaction-list group. Default `64`. */
  groupSize?: number;
  seed?: number;
}

/** `[x, y, z, mass]` per interaction-list entry. */
const ENTRY = 4;

/**
 * Barnes-Hut n-body repulsion (or gravity) using grouped interaction lists
 * (Barnes 1990, "A modified tree code").
 *
 * Instead of walking the tree once per body, the tree is cut into groups of
 * at most `groupSize` spatially adjacent bodies (whole subtrees, contiguous in
 * `tree.bodyOrder`). Each group walks the tree once and collects:
 *  - a far list of cell pseudo-bodies `(center of mass, mass)`, accepted when
 *    `w² / θ² < d²` with `d` the distance from the cell's center of mass to the
 *    group's bounding box — conservative for every body in the group;
 *  - a near list of individual bodies from opened leaves.
 * Every body in the group then sums both lists in a tight, nearly branch-free
 * loop. This trades a few more (cheap, predictable) interactions for far fewer
 * (expensive, branch-mispredicting) tree visits.
 *
 * Force law matches d3-force: `F += Δ · strength · m · α / l²` (∝ 1/l), with
 * `l²` softened to `√(dMin² · l²)` below `distanceMin` and contributions at or
 * beyond `distanceMax` dropped.
 */
export class ManyBodyForce implements IForce {
  readonly name: string;
  readonly requiresSpatialTree = true;
  enabled = true;

  strength: number;
  groupSize: number;
  private _theta2: number;
  private _distanceMin2: number;
  private _distanceMax2: number;

  private readonly rng: Lcg;
  /** Worst case: every level of a full-depth path pushes `branching` children. */
  private readonly stack = new Int32Array(8 * (SPATIAL_TREE_MAX_DEPTH + 2));
  private readonly groupStack = new Int32Array(8 * (SPATIAL_TREE_MAX_DEPTH + 2));

  private far: Float64Array = new Float64Array(1024 * ENTRY);
  private near: Float64Array = new Float64Array(256 * ENTRY);
  private nearIndex: Int32Array = new Int32Array(256);
  private farCount = 0;
  private nearCount = 0;

  constructor(options: ManyBodyForceOptions = {}, name = 'charge') {
    this.name = name;
    this.strength = options.strength ?? -30;
    this.groupSize = Math.max(1, Math.floor(options.groupSize ?? 64));
    const theta = options.theta ?? 0.8;
    this._theta2 = theta * theta;
    const dMin = options.distanceMin ?? 1;
    const dMax = options.distanceMax ?? Infinity;
    this._distanceMin2 = dMin * dMin;
    this._distanceMax2 = dMax * dMax;
    this.rng = new Lcg(options.seed ?? 1);
  }

  get theta(): number {
    return Math.sqrt(this._theta2);
  }
  set theta(v: number) {
    this._theta2 = v * v;
  }
  get distanceMin(): number {
    return Math.sqrt(this._distanceMin2);
  }
  set distanceMin(v: number) {
    this._distanceMin2 = v * v;
  }
  get distanceMax(): number {
    return Math.sqrt(this._distanceMax2);
  }
  set distanceMax(v: number) {
    this._distanceMax2 = v * v;
  }

  initialize(_graph: IGraphBuffers, _dimensions: Dimensions): void {
    // Stateless between ticks: everything is derived from the tree.
  }

  apply(ctx: ForceTickContext): void {
    const tree = ctx.tree;
    if (tree.root === NO_INDEX || this.strength === 0) return;

    const B = tree.branching;
    const children = tree.cellChildren;
    const firstBody = tree.cellFirstBody;
    const bodyCount = tree.cellBodyCount;
    const G = this.groupSize;
    const stack = this.groupStack;

    // Cut the tree into groups: the highest cells holding ≤ G bodies (or leaves).
    let sp = 0;
    stack[sp++] = tree.root;
    while (sp > 0) {
      const c = stack[--sp];
      if (bodyCount[c] > G && firstBody[c] === NO_INDEX) {
        for (let k = 0, base = c * B; k < B; k++) {
          const ch = children[base + k];
          if (ch !== NO_INDEX) stack[sp++] = ch;
        }
        continue;
      }
      this.buildLists(tree, c, ctx.positions, ctx.graph.nodeMasses, ctx.dimensions === 3);
      this.evaluateGroup(tree, c, ctx);
    }
  }

  /** Collects far (cell) and near (body) interaction lists for group cell `g`. */
  private buildLists(tree: ISpatialTree, g: number, p: Float32Array, masses: Float32Array, is3D: boolean): void {
    const B = tree.branching;
    const children = tree.cellChildren;
    const center = tree.cellCenter;
    const half = tree.cellHalfSize;
    const mc = tree.cellMassCenter;
    const cellMass = tree.cellMass;
    const firstBody = tree.cellFirstBody;
    const bodyNext = tree.bodyNext;
    const invTheta2 = 1 / this._theta2;
    const stack = this.stack;

    const go = g * 3;
    const gx = center[go], gy = center[go + 1], gz = center[go + 2];
    const gh = half[g];

    let far = this.far;
    let near = this.near;
    let nearIndex = this.nearIndex;
    let nf = 0;
    let nn = 0;

    let sp = 0;
    stack[sp++] = tree.root;
    while (sp > 0) {
      const c = stack[--sp];
      const m = cellMass[c];
      if (m === 0) continue;

      const co = c * 3;
      const mx = mc[co], my = mc[co + 1], mz = mc[co + 2];
      let ax = Math.abs(mx - gx) - gh;
      let ay = Math.abs(my - gy) - gh;
      let az = is3D ? Math.abs(mz - gz) - gh : 0;
      if (ax < 0) ax = 0;
      if (ay < 0) ay = 0;
      if (az < 0) az = 0;
      const d2 = ax * ax + ay * ay + az * az;
      const w = 2 * half[c];

      if (w * w * invTheta2 < d2) {
        if ((nf + 1) * ENTRY > far.length) far = this.far = growF64(far, (nf + 1) * ENTRY);
        const o = nf++ * ENTRY;
        far[o] = mx;
        far[o + 1] = my;
        far[o + 2] = mz;
        far[o + 3] = m;
        continue;
      }

      const head = firstBody[c];
      if (head === NO_INDEX) {
        for (let k = 0, base = c * B; k < B; k++) {
          const ch = children[base + k];
          if (ch !== NO_INDEX) stack[sp++] = ch;
        }
        continue;
      }

      for (let j = head; j !== NO_INDEX; j = bodyNext[j]) {
        if (nn === nearIndex.length) {
          near = this.near = growF64(near, (nn + 1) * ENTRY);
          nearIndex = this.nearIndex = growI32(nearIndex, nn + 1);
        }
        const jo = j * NODE_TRANSFORM_STRIDE;
        const o = nn * ENTRY;
        near[o] = p[jo];
        near[o + 1] = p[jo + 1];
        near[o + 2] = is3D ? p[jo + 2] : 0;
        near[o + 3] = masses[j];
        nearIndex[nn++] = j;
      }
    }
    this.farCount = nf;
    this.nearCount = nn;
  }

  /** Sums both interaction lists for every body of group cell `g`. */
  private evaluateGroup(tree: ISpatialTree, g: number, ctx: ForceTickContext): void {
    const p = ctx.positions;
    const f = ctx.forces;
    const is3D = ctx.dimensions === 3;
    const kAlpha = this.strength * ctx.alpha;
    const dMin2 = this._distanceMin2;
    const dMax2 = this._distanceMax2;
    const unbounded = dMax2 === Infinity;
    const rng = this.rng;
    const far = this.far;
    const near = this.near;
    const nearIndex = this.nearIndex;
    const farEnd = this.farCount * ENTRY;
    const nn = this.nearCount;
    const order = tree.bodyOrder;
    const start = tree.cellOrderStart[g];
    const end = start + tree.cellBodyCount[g];

    for (let q = start; q < end; q++) {
      const i = order[q];
      const io = i * NODE_TRANSFORM_STRIDE;
      const xi = p[io], yi = p[io + 1], zi = is3D ? p[io + 2] : 0;
      let fx = 0, fy = 0, fz = 0;

      if (unbounded) {
        // Hot path (default `distanceMax = Infinity`): no range test.
        for (let o = 0; o < farEnd; o += ENTRY) {
          const dx = far[o] - xi;
          const dy = far[o + 1] - yi;
          const dz = far[o + 2] - zi;
          let l2 = dx * dx + dy * dy + dz * dz;
          if (l2 < dMin2) l2 = Math.sqrt(dMin2 * l2);
          const s = far[o + 3] / l2;
          fx += dx * s;
          fy += dy * s;
          fz += dz * s;
        }
      } else {
        for (let o = 0; o < farEnd; o += ENTRY) {
          const dx = far[o] - xi;
          const dy = far[o + 1] - yi;
          const dz = far[o + 2] - zi;
          let l2 = dx * dx + dy * dy + dz * dz;
          if (l2 >= dMax2) continue;
          if (l2 < dMin2) l2 = Math.sqrt(dMin2 * l2);
          const s = far[o + 3] / l2;
          fx += dx * s;
          fy += dy * s;
          fz += dz * s;
        }
      }

      for (let k = 0, o = 0; k < nn; k++, o += ENTRY) {
        if (nearIndex[k] === i) continue;
        let dx = near[o] - xi;
        let dy = near[o + 1] - yi;
        let dz = near[o + 2] - zi;
        let l2 = dx * dx + dy * dy + dz * dz;
        if (l2 >= dMax2) continue;
        if (l2 === 0) {
          dx = jiggle(rng);
          dy = jiggle(rng);
          if (is3D) dz = jiggle(rng);
          l2 = dx * dx + dy * dy + dz * dz;
        }
        if (l2 < dMin2) l2 = Math.sqrt(dMin2 * l2);
        const s = near[o + 3] / l2;
        fx += dx * s;
        fy += dy * s;
        fz += dz * s;
      }

      const v = i * VEC3_STRIDE;
      f[v] += fx * kAlpha;
      f[v + 1] += fy * kAlpha;
      if (is3D) f[v + 2] += fz * kAlpha;
    }
  }
}

function growF64(src: Float64Array, required: number): Float64Array {
  const out = new Float64Array(Math.max(required, src.length * 2));
  out.set(src);
  return out;
}

function growI32(src: Int32Array, required: number): Int32Array {
  const out = new Int32Array(Math.max(required, src.length * 2));
  out.set(src);
  return out;
}
