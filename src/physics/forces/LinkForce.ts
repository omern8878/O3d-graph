import {
  NODE_TRANSFORM_STRIDE,
  NodeFlag,
  VEC3_STRIDE,
  type Dimensions,
  type ForceTickContext,
  type IForce,
  type IGraphBuffers,
} from '../../types/index.js';
import { Lcg, jiggle } from '../../utils/random.js';

export interface LinkForceOptions {
  /** Relaxation passes per tick. Default `1`. */
  iterations?: number;
  /** Global multiplier on every edge's strength. Default `1`. */
  strengthScale?: number;
  seed?: number;
}

/**
 * Hooke's-law springs along edges.
 *
 * Per edge `(s, t)` with rest length `d` and stiffness `k`:
 *   `Δ = (P_t + F_t) − (P_s + F_s)`, `m = (|Δ| − d) / |Δ| · α · k`
 *   `F_t −= Δ·m·b`, `F_s += Δ·m·(1 − b)`, with `b = deg(s) / (deg(s) + deg(t))`.
 *
 * - Auto stiffness (`EDGE_STRENGTH_AUTO`) is `1 / min(deg(s), deg(t))`, so a
 *   hub with hundreds of springs isn't yanked around by each one.
 * - The degree bias `b` moves the low-degree endpoint more than the hub.
 * - Reading `P + F` (the d3-style predicted position) lets repeated
 *   `iterations` relax the springs against each other within one tick.
 *
 * Bias and auto-stiffness are derived from `nodeDegrees` on the fly, so the
 * force keeps no per-edge cache and needs no rebuild on topology change.
 */
export class LinkForce implements IForce {
  readonly name: string;
  readonly requiresSpatialTree = false;
  enabled = true;

  iterations: number;
  strengthScale: number;
  private readonly rng: Lcg;

  constructor(options: LinkForceOptions = {}, name = 'link') {
    this.name = name;
    this.iterations = Math.max(1, Math.floor(options.iterations ?? 1));
    this.strengthScale = options.strengthScale ?? 1;
    this.rng = new Lcg(options.seed ?? 1);
  }

  initialize(_graph: IGraphBuffers, _dimensions: Dimensions): void {
    // Stateless: degrees and strengths are read directly from the store.
  }

  apply(ctx: ForceTickContext): void {
    const E = ctx.edgeCount;
    if (E === 0 || this.strengthScale === 0) return;

    const g = ctx.graph;
    const p = ctx.positions;
    const f = ctx.forces;
    const src = g.edgeSources;
    const tgt = g.edgeTargets;
    const dist = g.edgeDistances;
    const str = g.edgeStrengths;
    const deg = g.nodeDegrees;
    const flags = g.nodeFlags;
    const is3D = ctx.dimensions === 3;
    const alphaScale = ctx.alpha * this.strengthScale;
    const rng = this.rng;

    for (let iter = 0; iter < this.iterations; iter++) {
      for (let e = 0; e < E; e++) {
        const s = src[e];
        const t = tgt[e];
        if (s === t || ((flags[s] | flags[t]) & NodeFlag.HIDDEN) !== 0) continue;

        const so = s * NODE_TRANSFORM_STRIDE;
        const to = t * NODE_TRANSFORM_STRIDE;
        const s3 = s * VEC3_STRIDE;
        const t3 = t * VEC3_STRIDE;

        let dx = p[to] + f[t3] - p[so] - f[s3];
        let dy = p[to + 1] + f[t3 + 1] - p[so + 1] - f[s3 + 1];
        let dz = is3D ? p[to + 2] + f[t3 + 2] - p[so + 2] - f[s3 + 2] : 0;
        let l2 = dx * dx + dy * dy + dz * dz;
        if (l2 === 0) {
          dx = jiggle(rng);
          dy = jiggle(rng);
          if (is3D) dz = jiggle(rng);
          l2 = dx * dx + dy * dy + dz * dz;
        }
        const l = Math.sqrt(l2);

        const ds = deg[s];
        const dt = deg[t];
        const k = str[e] < 0 ? 1 / (ds < dt ? ds : dt) : str[e];
        const m = ((l - dist[e]) / l) * alphaScale * k;
        dx *= m;
        dy *= m;
        dz *= m;

        const b = ds / (ds + dt);
        const ib = 1 - b;
        f[t3] -= dx * b;
        f[t3 + 1] -= dy * b;
        f[s3] += dx * ib;
        f[s3 + 1] += dy * ib;
        if (is3D) {
          f[t3 + 2] -= dz * b;
          f[s3 + 2] += dz * ib;
        }
      }
    }
  }
}
