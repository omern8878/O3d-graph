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

export interface ContainForceOptions {
  /** Target outer radius of the layout (world units). */
  radius: number;
  x?: number;
  y?: number;
  z?: number;
  /** Starting pull strength. Default `0.12`. */
  strength?: number;
  /** Lower / upper bounds for the adapted strength. Defaults `0.01` / `8`. */
  minStrength?: number;
  maxStrength?: number;
}

/**
 * Keeps the whole layout within `radius` of a center, whatever the node
 * count, by adapting a linear pull toward the center (`F = −k·α·Δ`).
 *
 * Each tick it measures the outermost free node and nudges `k` up when the
 * layout is too large and down when it's smaller than `radius` (bounded
 * multiplicative steps), so the layout settles with its outer edge at about
 * `radius` while link, charge and collision forces shape the interior. Unlike
 * a hard wall it doesn't pile nodes against a boundary.
 *
 * Useful when the renderer's view volume is bounded (e.g. a fixed far
 * plane) or to make layouts of any size frame the same way.
 */
export class ContainForce implements IForce {
  readonly name: string;
  readonly requiresSpatialTree = false;
  enabled = true;

  radius: number;
  x: number;
  y: number;
  z: number;
  minStrength: number;
  maxStrength: number;
  private readonly initialStrength: number;
  private _strength: number;

  constructor(options: ContainForceOptions, name = 'contain') {
    this.name = name;
    this.radius = options.radius;
    this.x = options.x ?? 0;
    this.y = options.y ?? 0;
    this.z = options.z ?? 0;
    this.initialStrength = options.strength ?? 0.12;
    this._strength = this.initialStrength;
    this.minStrength = options.minStrength ?? 0.01;
    this.maxStrength = options.maxStrength ?? 8;
  }

  /** The currently adapted pull strength. */
  get strength(): number {
    return this._strength;
  }

  /** Keeps the adapted strength across topology changes (it keeps adapting anyway). */
  initialize(_graph: IGraphBuffers, _dimensions: Dimensions): void {}

  /** Restarts adaptation from the initial strength (e.g. after loading unrelated data). */
  reset(): void {
    this._strength = this.initialStrength;
  }

  apply(ctx: ForceTickContext): void {
    const n = ctx.nodeCount;
    if (n === 0 || this.radius <= 0) return;
    const p = ctx.positions;
    const f = ctx.forces;
    const flags = ctx.graph.nodeFlags;
    const is3D = ctx.dimensions === 3;
    const cx = this.x, cy = this.y, cz = is3D ? this.z : 0;

    let far2 = 0;
    for (let i = 0, o = 0; i < n; i++, o += NODE_TRANSFORM_STRIDE) {
      if ((flags[i] & FROZEN) !== 0) continue;
      const dx = p[o] - cx, dy = p[o + 1] - cy, dz = is3D ? p[o + 2] - cz : 0;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > far2) far2 = d2;
    }
    if (far2 === 0) return;

    // Bounded multiplicative steering toward "outermost node at radius".
    const ratio = Math.sqrt(far2) / this.radius;
    const step = Math.min(1.25, Math.max(0.8, Math.pow(ratio, 0.35)));
    this._strength = Math.min(this.maxStrength, Math.max(this.minStrength, this._strength * step));

    const k = this._strength * ctx.alpha;
    for (let i = 0, o = 0, v = 0; i < n; i++, o += NODE_TRANSFORM_STRIDE, v += VEC3_STRIDE) {
      if ((flags[i] & FROZEN) !== 0) continue;
      f[v] -= (p[o] - cx) * k;
      f[v + 1] -= (p[o + 1] - cy) * k;
      if (is3D) f[v + 2] -= (p[o + 2] - cz) * k;
    }
  }
}
