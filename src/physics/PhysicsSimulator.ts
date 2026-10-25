import {
  NODE_TRANSFORM_STRIDE,
  NodeFlag,
  VEC3_STRIDE,
  type DimensionMode,
  type Dimensions,
  type ForceTickContext,
  type IForce,
  type IGraphBuffers,
  type IPhysicsEngine,
  type ISpatialTree,
  type PhysicsOptions,
} from '../types/index.js';
import { SpatialTree } from '../spatial/SpatialTree.js';
import { growFloat32 } from '../utils/capacity.js';

export type ResolvedPhysicsOptions = Required<PhysicsOptions>;

/** Fills every unset physics option with its d3-compatible default. */
export function resolvePhysicsOptions(options: PhysicsOptions = {}): ResolvedPhysicsOptions {
  const alphaMin = options.alphaMin ?? 0.001;
  return {
    alpha: options.alpha ?? 1,
    alphaMin,
    alphaDecay: options.alphaDecay ?? 1 - Math.pow(alphaMin, 1 / 300),
    alphaTarget: options.alphaTarget ?? 0,
    velocityDecay: options.velocityDecay ?? 0.4,
    timeStep: options.timeStep ?? 1,
    theta: options.theta ?? 0.8,
    chargeStrength: options.chargeStrength ?? -30,
    chargeDistanceMin: options.chargeDistanceMin ?? 1,
    chargeDistanceMax: options.chargeDistanceMax ?? Infinity,
    linkIterations: options.linkIterations ?? 1,
    centerStrength: options.centerStrength ?? 1,
    radialRadius: options.radialRadius ?? 0,
    radialStrength: options.radialStrength ?? 0.1,
    collide: options.collide ?? true,
    collideStrength: options.collideStrength ?? 0.7,
    collidePadding: options.collidePadding ?? 1,
    collideIterations: options.collideIterations ?? 1,
  };
}

/** Alpha set by `setDimensionMode` so the layout re-settles in the new space. */
const DIMENSION_REHEAT_ALPHA = 0.3;
const DEFAULT_REHEAT_ALPHA = 0.3;
const FROZEN = NodeFlag.PINNED | NodeFlag.HIDDEN;

/** Mutable backing object for the read-only `ForceTickContext`, reused every tick. */
class TickContext implements ForceTickContext {
  alpha = 1;
  dimensions: Dimensions = 3;
  nodeCount = 0;
  edgeCount = 0;
  positions: Float32Array = new Float32Array(0);
  velocities: Float32Array = new Float32Array(0);
  forces: Float32Array = new Float32Array(0);

  constructor(
    readonly graph: IGraphBuffers,
    readonly tree: ISpatialTree,
  ) {}
}

export interface PhysicsSimulatorDeps {
  /** Override the Barnes-Hut tree (e.g. for testing). */
  tree?: ISpatialTree;
}

/**
 * Velocity-Verlet integrator with simulated annealing.
 *
 * Each tick, for every free node (not pinned, not hidden):
 *  1. `α ← α + (α_target − α) · α_decay`
 *  2. drift: `x ← x + v·dt + ½·a·dt²` (written straight into WASM memory)
 *  3. zero `F`, rebuild the spatial tree once if any force needs it, run all
 *     enabled forces in registration order (they accumulate into `F`)
 *  4. kick: `v ← (v + ½·(a + F)·dt) · (1 − velocityDecay)`, `a ← F`
 *
 * Pinned nodes keep their position (drag writes it) and have `v = a = 0`.
 * In 2D the Z components of position, velocity and acceleration stay 0.
 *
 * The simulator only touches physics state; syncing edge vertices and
 * uploading to the renderer is the caller's job (see `O3DGraph`).
 */
export class PhysicsSimulator implements IPhysicsEngine {
  alpha: number;
  alphaMin: number;
  alphaDecay: number;
  alphaTarget: number;
  velocityDecay: number;
  timeStep: number;

  private readonly graph: IGraphBuffers;
  private readonly _tree: ISpatialTree;
  private readonly ctx: TickContext;
  private readonly forces: IForce[] = [];
  private readonly forcesByName = new Map<string, IForce>();

  /** Acceleration from the previous tick (Verlet `a(t)`), stride 3. */
  private prevAccel: Float32Array;
  private _dimensions: Dimensions;
  private _tickCount = 0;
  private running = true;
  private seenTopology = -1;

  constructor(graph: IGraphBuffers, options: PhysicsOptions = {}, deps: PhysicsSimulatorDeps = {}) {
    const o = resolvePhysicsOptions(options);
    this.graph = graph;
    this.alpha = o.alpha;
    this.alphaMin = o.alphaMin;
    this.alphaDecay = o.alphaDecay;
    this.alphaTarget = o.alphaTarget;
    this.velocityDecay = o.velocityDecay;
    this.timeStep = o.timeStep;

    this._dimensions = graph.dimensionMode === '2d' ? 2 : 3;
    this._tree = deps.tree ?? new SpatialTree(this._dimensions);
    this._tree.setDimensions(this._dimensions);
    this.ctx = new TickContext(graph, this._tree);
    this.prevAccel = new Float32Array(graph.nodeCapacity * VEC3_STRIDE);
  }

  get dimensions(): Dimensions {
    return this._dimensions;
  }
  get tickCount(): number {
    return this._tickCount;
  }
  get isActive(): boolean {
    return this.running && (this.alpha >= this.alphaMin || this.alphaTarget >= this.alphaMin);
  }
  get tree(): ISpatialTree {
    return this._tree;
  }

  // ---- Force registry (OCP) ---------------------------------------------------

  /** Registers a force; a force with the same name is disposed and replaced in place. */
  registerForce(force: IForce): void {
    const existing = this.forcesByName.get(force.name);
    if (existing !== undefined) {
      const idx = this.forces.indexOf(existing);
      this.forces[idx] = force;
      if (existing !== force) existing.dispose?.();
    } else {
      this.forces.push(force);
    }
    this.forcesByName.set(force.name, force);
    force.initialize(this.graph, this._dimensions);
  }

  unregisterForce(name: string): boolean {
    const force = this.forcesByName.get(name);
    if (force === undefined) return false;
    this.forces.splice(this.forces.indexOf(force), 1);
    this.forcesByName.delete(name);
    force.dispose?.();
    return true;
  }

  getForce<T extends IForce = IForce>(name: string): T | undefined {
    return this.forcesByName.get(name) as T | undefined;
  }

  get forceNames(): string[] {
    return this.forces.map((f) => f.name);
  }

  // ---- Lifecycle ----------------------------------------------------------------

  setDimensionMode(mode: DimensionMode): void {
    const dims: Dimensions = mode === '2d' ? 2 : 3;
    if (dims === this._dimensions) return;
    this._dimensions = dims;
    this._tree.setDimensions(dims);
    this.prevAccel.fill(0);
    this.initializeForces();
    this.reheat(Math.max(this.alpha, DIMENSION_REHEAT_ALPHA));
  }

  notifyTopologyChanged(): void {
    this.seenTopology = -1;
  }

  reheat(alpha?: number): void {
    this.alpha = alpha ?? Math.max(this.alpha, DEFAULT_REHEAT_ALPHA);
    this.running = true;
  }

  restart(): void {
    this.running = true;
  }

  stop(): void {
    this.running = false;
  }

  dispose(): void {
    this.running = false;
    for (const force of this.forces) force.dispose?.();
    this.forces.length = 0;
    this.forcesByName.clear();
  }

  // ---- Integration --------------------------------------------------------------

  tick(iterations = 1): boolean {
    for (let k = 0; k < iterations; k++) {
      if (!this.isActive) {
        this.running = false;
        return false;
      }
      this.alpha += (this.alphaTarget - this.alpha) * this.alphaDecay;
      this.step();
      this._tickCount++;
    }
    if (!this.isActive) {
      this.running = false;
      return false;
    }
    return true;
  }

  /** Advances one integration step regardless of `alpha` (for manual stepping / tests). */
  step(): void {
    const g = this.graph;
    const n = g.nodeCount;
    if (g.topologyVersion !== this.seenTopology) this.onTopologyChanged();
    if (n === 0) return;

    const p = g.getNodeTransforms();
    const vel = g.nodeVelocities;
    const f = g.nodeForces;
    const flags = g.nodeFlags;
    const a = this.prevAccel;
    const is3D = this._dimensions === 3;
    const dt = this.timeStep;
    const halfDt2 = 0.5 * dt * dt;

    // 1. Drift.
    for (let i = 0, o = 0, v = 0; i < n; i++, o += NODE_TRANSFORM_STRIDE, v += VEC3_STRIDE) {
      if ((flags[i] & FROZEN) !== 0) {
        vel[v] = vel[v + 1] = vel[v + 2] = 0;
        a[v] = a[v + 1] = a[v + 2] = 0;
        continue;
      }
      p[o] += vel[v] * dt + a[v] * halfDt2;
      p[o + 1] += vel[v + 1] * dt + a[v + 1] * halfDt2;
      if (is3D) p[o + 2] += vel[v + 2] * dt + a[v + 2] * halfDt2;
      else p[o + 2] = 0;
    }

    // 2. Forces at the new positions.
    f.fill(0, 0, n * VEC3_STRIDE);
    const forces = this.forces;
    let needsTree = false;
    for (let k = 0; k < forces.length; k++) {
      if (forces[k].enabled && forces[k].requiresSpatialTree) {
        needsTree = true;
        break;
      }
    }
    if (needsTree) this._tree.build(g);

    const ctx = this.ctx;
    ctx.alpha = this.alpha;
    ctx.dimensions = this._dimensions;
    ctx.nodeCount = n;
    ctx.edgeCount = g.edgeCount;
    ctx.positions = p;
    ctx.velocities = vel;
    ctx.forces = f;
    for (let k = 0; k < forces.length; k++) {
      if (forces[k].enabled) forces[k].apply(ctx);
    }

    // 3. Kick.
    const damp = 1 - this.velocityDecay;
    const halfDt = 0.5 * dt;
    for (let i = 0, v = 0; i < n; i++, v += VEC3_STRIDE) {
      if ((flags[i] & FROZEN) !== 0) {
        vel[v] = vel[v + 1] = vel[v + 2] = 0;
        continue;
      }
      const fx = f[v], fy = f[v + 1], fz = is3D ? f[v + 2] : 0;
      let vx = (vel[v] + (a[v] + fx) * halfDt) * damp;
      let vy = (vel[v + 1] + (a[v + 1] + fy) * halfDt) * damp;
      let vz = is3D ? (vel[v + 2] + (a[v + 2] + fz) * halfDt) * damp : 0;
      // Guard against a single NaN/Infinity poisoning the whole layout.
      if (vx - vx !== 0 || vy - vy !== 0 || vz - vz !== 0) {
        vx = vy = vz = 0;
        a[v] = a[v + 1] = a[v + 2] = 0;
      } else {
        a[v] = fx;
        a[v + 1] = fy;
        a[v + 2] = fz;
      }
      vel[v] = vx;
      vel[v + 1] = vy;
      vel[v + 2] = vz;
    }
  }

  // ---- Internals ------------------------------------------------------------------

  private onTopologyChanged(): void {
    this.seenTopology = this.graph.topologyVersion;
    const needed = this.graph.nodeCapacity * VEC3_STRIDE;
    if (this.prevAccel.length < needed) this.prevAccel = growFloat32(this.prevAccel, needed);
    // Swap-and-pop moved nodes between slots; stale accelerations would be misattributed.
    this.prevAccel.fill(0);
    this.initializeForces();
  }

  private initializeForces(): void {
    for (let k = 0; k < this.forces.length; k++) {
      this.forces[k].initialize(this.graph, this._dimensions);
    }
  }
}
