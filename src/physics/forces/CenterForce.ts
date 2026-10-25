import {
  NODE_TRANSFORM_STRIDE,
  NodeFlag,
  VEC3_STRIDE,
  type Dimensions,
  type ForceTickContext,
  type IForce,
  type IGraphBuffers,
} from '../../types/index.js';

const FROZEN = NodeFlag.PINNED | NodeFlag.HIDDEN;

export interface CenterForceOptions {
  x?: number;
  y?: number;
  z?: number;
  /** Fraction of the centroid offset removed per tick. Default `1`. */
  strength?: number;
}

/**
 * Translates the free nodes so their centroid sits on `(x, y, z)`.
 *
 * This is a positional constraint (as in d3-force), not an acceleration: it
 * removes global drift without distorting relative layout. Pinned and hidden
 * nodes neither contribute to the centroid nor move.
 */
export class CenterForce implements IForce {
  readonly name: string;
  readonly requiresSpatialTree = false;
  enabled = true;

  x: number;
  y: number;
  z: number;
  strength: number;

  constructor(options: CenterForceOptions = {}, name = 'center') {
    this.name = name;
    this.x = options.x ?? 0;
    this.y = options.y ?? 0;
    this.z = options.z ?? 0;
    this.strength = options.strength ?? 1;
  }

  initialize(_graph: IGraphBuffers, _dimensions: Dimensions): void {}

  apply(ctx: ForceTickContext): void {
    const n = ctx.nodeCount;
    if (n === 0 || this.strength === 0) return;
    const p = ctx.positions;
    const flags = ctx.graph.nodeFlags;
    const is3D = ctx.dimensions === 3;

    let sx = 0, sy = 0, sz = 0, count = 0;
    for (let i = 0, o = 0; i < n; i++, o += NODE_TRANSFORM_STRIDE) {
      if ((flags[i] & FROZEN) !== 0) continue;
      sx += p[o];
      sy += p[o + 1];
      sz += p[o + 2];
      count++;
    }
    if (count === 0) return;

    const k = this.strength / count;
    const mx = sx * k - this.x * this.strength;
    const my = sy * k - this.y * this.strength;
    const mz = is3D ? sz * k - this.z * this.strength : 0;
    if (mx === 0 && my === 0 && mz === 0) return;

    for (let i = 0, o = 0; i < n; i++, o += NODE_TRANSFORM_STRIDE) {
      if ((flags[i] & FROZEN) !== 0) continue;
      p[o] -= mx;
      p[o + 1] -= my;
      if (is3D) p[o + 2] -= mz;
    }
  }
}

export interface RadialForceOptions {
  /** Target distance from the center. */
  radius: number;
  x?: number;
  y?: number;
  z?: number;
  /** Default `0.1`. */
  strength?: number;
}

/**
 * Pulls every free node toward a circle (2D) / sphere (3D) of `radius`
 * around `(x, y, z)`: `F += (r − d) / d · strength · α · Δ`.
 * Useful for keeping disconnected components from drifting apart.
 */
export class RadialForce implements IForce {
  readonly name: string;
  readonly requiresSpatialTree = false;
  enabled = true;

  radius: number;
  x: number;
  y: number;
  z: number;
  strength: number;

  constructor(options: RadialForceOptions, name = 'radial') {
    this.name = name;
    this.radius = options.radius;
    this.x = options.x ?? 0;
    this.y = options.y ?? 0;
    this.z = options.z ?? 0;
    this.strength = options.strength ?? 0.1;
  }

  initialize(_graph: IGraphBuffers, _dimensions: Dimensions): void {}

  apply(ctx: ForceTickContext): void {
    const n = ctx.nodeCount;
    if (n === 0 || this.strength === 0) return;
    const p = ctx.positions;
    const f = ctx.forces;
    const flags = ctx.graph.nodeFlags;
    const is3D = ctx.dimensions === 3;
    const ka = this.strength * ctx.alpha;
    const r = this.radius;
    const cx = this.x, cy = this.y, cz = this.z;

    for (let i = 0, o = 0, v = 0; i < n; i++, o += NODE_TRANSFORM_STRIDE, v += VEC3_STRIDE) {
      if ((flags[i] & FROZEN) !== 0) continue;
      const dx = p[o] - cx || 1e-6;
      const dy = p[o + 1] - cy;
      const dz = is3D ? p[o + 2] - cz : 0;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const k = ((r - d) * ka) / d;
      f[v] += dx * k;
      f[v + 1] += dy * k;
      if (is3D) f[v + 2] += dz * k;
    }
  }
}
