import {
  NODE_TRANSFORM_STRIDE,
  NO_INDEX,
  VEC3_STRIDE,
  type Dimensions,
  type ForceTickContext,
  type IForce,
  type IGraphBuffers,
} from '../../types/index.js';
import { SPATIAL_TREE_MAX_DEPTH } from '../../spatial/SpatialTree.js';
import { Lcg, jiggle } from '../../utils/random.js';

export interface CollideForceOptions {
  /** Overlap resolution strength in `[0, 1]`. Default `0.7`. */
  strength?: number;
  /** Extra clearance added to every radius. Default `1`. */
  padding?: number;
  /** Relaxation passes per tick. Default `1`. */
  iterations?: number;
  seed?: number;
}

/**
 * Sphere (3D) / circle (2D) non-overlap constraint.
 *
 * Each visible node walks the shared tree, pruning any cell whose bounding
 * box lies farther than `rᵢ + maxRadius(cell) + 2·padding` from it. Each
 * overlapping pair is visited once (`j > i`) and pushed apart along the
 * center line, split by `r²` so small nodes yield to large ones:
 *
 *   `push = (R − l) / l · strength`, `wᵢ = rⱼ² / (rᵢ² + rⱼ²)`, `wⱼ = 1 − wᵢ`
 *
 * Not scaled by alpha: overlap is resolved even in a cooled simulation.
 * Extra iterations read `P + F` predicted positions; pruning keeps using the
 * tree built at the start of the tick.
 */
export class CollideForce implements IForce {
  readonly name: string;
  readonly requiresSpatialTree = true;
  enabled = true;

  strength: number;
  padding: number;
  iterations: number;

  private readonly rng: Lcg;
  private readonly stack = new Int32Array(8 * (SPATIAL_TREE_MAX_DEPTH + 2));

  constructor(options: CollideForceOptions = {}, name = 'collide') {
    this.name = name;
    this.strength = options.strength ?? 0.7;
    this.padding = options.padding ?? 1;
    this.iterations = Math.max(1, Math.floor(options.iterations ?? 1));
    this.rng = new Lcg(options.seed ?? 1);
  }

  initialize(_graph: IGraphBuffers, _dimensions: Dimensions): void {}

  apply(ctx: ForceTickContext): void {
    const tree = ctx.tree;
    const root = tree.root;
    if (root === NO_INDEX || this.strength === 0) return;

    const order = tree.bodyOrder;
    const bodyCount = tree.bodyCount;
    const p = ctx.positions;
    const f = ctx.forces;
    const radii = ctx.graph.nodeRadii;
    const is3D = ctx.dimensions === 3;
    const pad = this.padding;
    const strength = this.strength;
    const rng = this.rng;

    const B = tree.branching;
    const children = tree.cellChildren;
    const center = tree.cellCenter;
    const half = tree.cellHalfSize;
    const maxR = tree.cellMaxRadius;
    const firstBody = tree.cellFirstBody;
    const bodyNext = tree.bodyNext;
    const stack = this.stack;

    for (let iter = 0; iter < this.iterations; iter++) {
      for (let q = 0; q < bodyCount; q++) {
        const i = order[q];
        const io = i * NODE_TRANSFORM_STRIDE;
        const i3 = i * VEC3_STRIDE;
        const xi = p[io] + f[i3];
        const yi = p[io + 1] + f[i3 + 1];
        const zi = is3D ? p[io + 2] + f[i3 + 2] : 0;
        const ri = radii[i] + pad;
        const ri2 = ri * ri;

        let sp = 0;
        stack[sp++] = root;
        while (sp > 0) {
          const c = stack[--sp];
          const reach = ri + maxR[c] + pad;
          const h = half[c];
          const co = c * 3;

          // Squared distance from node to the cell's AABB.
          let ax = Math.abs(xi - center[co]) - h;
          let ay = Math.abs(yi - center[co + 1]) - h;
          let az = is3D ? Math.abs(zi - center[co + 2]) - h : 0;
          if (ax < 0) ax = 0;
          if (ay < 0) ay = 0;
          if (az < 0) az = 0;
          if (ax * ax + ay * ay + az * az > reach * reach) continue;

          const head = firstBody[c];
          if (head === NO_INDEX) {
            for (let k = 0, base = c * B; k < B; k++) {
              const ch = children[base + k];
              if (ch !== NO_INDEX) stack[sp++] = ch;
            }
            continue;
          }

          for (let j = head; j !== NO_INDEX; j = bodyNext[j]) {
            if (j <= i) continue;
            const jo = j * NODE_TRANSFORM_STRIDE;
            const j3 = j * VEC3_STRIDE;
            const rj = radii[j] + pad;
            const R = ri + rj;
            let dx = xi - (p[jo] + f[j3]);
            let dy = yi - (p[jo + 1] + f[j3 + 1]);
            let dz = is3D ? zi - (p[jo + 2] + f[j3 + 2]) : 0;
            let l2 = dx * dx + dy * dy + dz * dz;
            if (l2 >= R * R) continue;
            if (l2 === 0) {
              dx = jiggle(rng);
              dy = jiggle(rng);
              if (is3D) dz = jiggle(rng);
              l2 = dx * dx + dy * dy + dz * dz;
            }
            const l = Math.sqrt(l2);
            const push = ((R - l) / l) * strength;
            dx *= push;
            dy *= push;
            dz *= push;
            const rj2 = rj * rj;
            const wi = rj2 / (ri2 + rj2);
            const wj = 1 - wi;
            f[i3] += dx * wi;
            f[i3 + 1] += dy * wi;
            f[j3] -= dx * wj;
            f[j3 + 1] -= dy * wj;
            if (is3D) {
              f[i3 + 2] += dz * wi;
              f[j3 + 2] -= dz * wj;
            }
          }
        }
      }
    }
  }
}
