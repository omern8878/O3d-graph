import {
  NO_INDEX,
  NodeFlag,
  type ColorInput,
  type DimensionMode,
  type EdgeId,
  type EdgeInput,
  type EdgePatch,
  type GraphData,
  type GraphOptions,
  type GraphStoreOptions,
  type IForce,
  type IGraphEventMap,
  type IGraphRenderer,
  type IGraphStore,
  type IO3DEngineAdapter,
  type IPhysicsEngine,
  type ModeChangePayload,
  type NodeId,
  type NodeInput,
  type NodePatch,
  type TickPayload,
  type Unsubscribe,
} from './types/index.js';
import { EventBus, type Listener } from './events/EventBus.js';
import { O3DEngineAdapter, O3DRendererAdapter, type O3DWasmBindings } from './renderer/O3DRendererAdapter.js';
import { DEFAULT_STORE_OPTIONS, GraphStore } from './memory/GraphStore.js';
import { PhysicsSimulator, resolvePhysicsOptions } from './physics/PhysicsSimulator.js';
import { createDefaultForces } from './physics/forces/index.js';
import { InteractionController } from './interaction/InteractionController.js';
import { parseColor } from './utils/color.js';

/** Optional overrides for every collaborator (Dependency Injection / testing). */
export interface O3DGraphDeps<TNodeMeta, TEdgeMeta> {
  store?: IGraphStore<TNodeMeta, TEdgeMeta>;
  physics?: IPhysicsEngine;
  events?: EventBus<IGraphEventMap<TNodeMeta>>;
  /** Forces to register instead of `createDefaultForces(options.physics)`. */
  forces?: IForce[];
  /** `false` disables pointer handling entirely (headless / custom input). */
  interaction?: false;
  /** Scheduler override (defaults to `requestAnimationFrame`, or a 16 ms timer outside browsers). */
  scheduler?: FrameScheduler;
}

export interface FrameScheduler {
  request(cb: (time: number) => void): number;
  cancel(handle: number): void;
}

export interface MutationOptions {
  /** Reheat the simulation after a structural change. Default `true`. */
  reheat?: boolean;
}

/** Alpha used to re-settle after incremental structural edits. */
const MUTATION_REHEAT_ALPHA = 0.3;

/**
 * What the graph renders through:
 *  - an `IGraphRenderer` — e.g. `O3DBatchRenderer` over the `@omernach/o3d`
 *    WebGPU renderer, or your own backend;
 *  - an `IO3DEngineAdapter`, or the raw handle-based WASM bindings
 *    (`O3DWasmBindings`), wrapped in an `O3DRendererAdapter`.
 */
export type GraphRenderSource = IGraphRenderer | IO3DEngineAdapter | O3DWasmBindings;

/**
 * Public facade for `@omernach/o3d-graph`.
 *
 * Wires the renderer adapter, SoA store, physics simulator and interaction
 * controller together and drives them from an on-demand animation loop:
 *
 *  - **Active**: up to `ticksPerFrame` physics ticks within `tickBudgetMs`,
 *    then one contiguous edge-vertex sync and one buffer upload per frame.
 *  - **Idle**: once the simulation has stabilized and no pointer gesture is in
 *    progress, the loop stops scheduling frames entirely (0% CPU). Any
 *    mutation, reheat or pointer event wakes it.
 *
 * ```ts
 * const renderer = new O3DBatchRenderer({ o3d: { InstancedMesh, LineBatch }, device, nodeMesh, camera });
 * const graph = new O3DGraph(renderer, { canvas });
 * graph.setData({ nodes, edges });
 * graph.on('node:click', ({ id }) => console.log(id));
 * ```
 */
export class O3DGraph<TNodeMeta = unknown, TEdgeMeta = unknown> {
  readonly renderer: IGraphRenderer;
  readonly store: IGraphStore<TNodeMeta, TEdgeMeta>;
  readonly physics: IPhysicsEngine;
  readonly events: EventBus<IGraphEventMap<TNodeMeta>>;
  readonly interaction: InteractionController<TNodeMeta> | null;
  /** The canvas the graph renders into and listens on. */
  readonly canvas: HTMLCanvasElement;

  private readonly scheduler: FrameScheduler;
  private readonly ticksPerFrame: number;
  private readonly tickBudgetMs: number;
  private readonly initialAlpha: number;

  private frameHandle = 0;
  private frameScheduled = false;
  private running = false;
  private wasActive = false;
  private disposed = false;
  private readonly tickPayload: TickPayload = { alpha: 0, tickCount: 0 };
  private readonly modePayload: ModeChangePayload = { mode: '3d', previous: '3d' };

  constructor(
    source: GraphRenderSource,
    options: GraphOptions,
    deps: O3DGraphDeps<TNodeMeta, TEdgeMeta> = {},
  ) {
    if (options === undefined || options.canvas === undefined) {
      throw new TypeError('O3DGraph: options.canvas is required');
    }
    this.canvas = options.canvas;
    const dimension: DimensionMode = options.dimension ?? '3d';
    const physicsOptions = resolvePhysicsOptions(options.physics);
    const seed = options.seed ?? 1;

    this.ticksPerFrame = Math.max(1, Math.floor(options.ticksPerFrame ?? 1));
    this.tickBudgetMs = Math.max(0, options.tickBudgetMs ?? 12);
    this.initialAlpha = physicsOptions.alpha;
    this.scheduler = deps.scheduler ?? defaultScheduler();
    this.events = deps.events ?? new EventBus<IGraphEventMap<TNodeMeta>>();

    this.renderer = isGraphRenderer(source)
      ? source
      : new O3DRendererAdapter(isRawBindings(source) ? new O3DEngineAdapter(source) : source, {
          nodeGeometryId: options.nodeGeometryId,
          initialNodeCapacity: options.initialNodeCapacity,
          initialEdgeCapacity: options.initialEdgeCapacity,
        });

    this.store = deps.store ?? new GraphStore<TNodeMeta, TEdgeMeta>(this.renderer, resolveStoreOptions(options), dimension);
    if (deps.store !== undefined) this.store.setDimensionMode(dimension);

    this.physics = deps.physics ?? new PhysicsSimulator(this.store, physicsOptions);
    this.physics.setDimensionMode(dimension);
    for (const force of deps.forces ?? createDefaultForces(physicsOptions, seed)) {
      this.physics.registerForce(force);
    }

    this.renderer.camera.setCamera2DMode(dimension === '2d');
    this.renderer.camera.setCameraOrbitEnabled(true);

    this.interaction =
      deps.interaction === false
        ? null
        : new InteractionController<TNodeMeta>({
            element: options.canvas,
            renderer: this.renderer,
            store: this.store as IGraphStore<TNodeMeta, unknown>,
            physics: this.physics,
            events: this.events,
            options: options.interaction,
            requestFrame: this.requestFrame,
          });

    // Nothing to simulate until data arrives (setData / addNode reheat it).
    if (this.store.nodeCount === 0) this.physics.stop();
    if (options.autoStart ?? true) this.start();
  }

  // ===========================================================================
  // Data
  // ===========================================================================

  /** Replaces the whole graph and restarts the simulation from `physics.alpha`. */
  setData(data: GraphData<TNodeMeta, TEdgeMeta>): this {
    this.assertAlive();
    this.interaction?.reset();
    this.store.setData(data);
    this.physics.notifyTopologyChanged();
    this.physics.reheat(this.initialAlpha);
    this.requestFrame();
    return this;
  }

  addNode(input: NodeInput<TNodeMeta>, opts?: MutationOptions): number {
    this.assertAlive();
    const i = this.store.addNode(input);
    this.afterMutation(opts);
    return i;
  }

  addNodes(inputs: ReadonlyArray<NodeInput<TNodeMeta>>, opts?: MutationOptions): this {
    this.assertAlive();
    for (let k = 0; k < inputs.length; k++) this.store.addNode(inputs[k]);
    this.afterMutation(opts);
    return this;
  }

  addEdge(input: EdgeInput<TEdgeMeta>, opts?: MutationOptions): number {
    this.assertAlive();
    const e = this.store.addEdge(input);
    this.afterMutation(opts);
    return e;
  }

  addEdges(inputs: ReadonlyArray<EdgeInput<TEdgeMeta>>, opts?: MutationOptions): this {
    this.assertAlive();
    for (let k = 0; k < inputs.length; k++) this.store.addEdge(inputs[k]);
    this.afterMutation(opts);
    return this;
  }

  updateNode(id: NodeId, patch: NodePatch<TNodeMeta>, opts?: MutationOptions): boolean {
    this.assertAlive();
    const ok = this.store.updateNode(id, patch);
    if (ok) this.afterMutation({ reheat: opts?.reheat ?? false });
    return ok;
  }

  updateEdge(id: EdgeId, patch: EdgePatch<TEdgeMeta>, opts?: MutationOptions): boolean {
    this.assertAlive();
    const ok = this.store.updateEdge(id, patch);
    if (ok) this.afterMutation({ reheat: opts?.reheat ?? (patch.distance !== undefined || patch.strength !== undefined) });
    return ok;
  }

  /** Removes a node and every edge incident to it. */
  removeNode(id: NodeId, opts?: MutationOptions): boolean {
    this.assertAlive();
    const ok = this.store.removeNode(id);
    if (ok) this.afterMutation(opts);
    return ok;
  }

  removeEdge(id: EdgeId, opts?: MutationOptions): boolean {
    this.assertAlive();
    const ok = this.store.removeEdge(id);
    if (ok) this.afterMutation(opts);
    return ok;
  }

  clear(): this {
    this.assertAlive();
    this.interaction?.reset();
    this.store.clear();
    this.physics.notifyTopologyChanged();
    this.requestFrame();
    return this;
  }

  hasNode(id: NodeId): boolean {
    return this.store.hasNode(id);
  }

  hasEdge(id: EdgeId): boolean {
    return this.store.hasEdge(id);
  }

  get nodeCount(): number {
    return this.store.nodeCount;
  }

  get edgeCount(): number {
    return this.store.edgeCount;
  }

  /** Writes the node's `[x, y, z]` into `out` (allocates a new array when omitted). Returns `null` if absent. */
  getNodePosition(id: NodeId, out: Float32Array | number[] = [0, 0, 0]): Float32Array | number[] | null {
    const i = this.store.getNodeIndex(id);
    if (i === NO_INDEX) return null;
    this.store.readNodePosition(i, out);
    return out;
  }

  getNodeMeta(id: NodeId): TNodeMeta | undefined {
    const i = this.store.getNodeIndex(id);
    return i === NO_INDEX ? undefined : this.store.getNodeMeta(i);
  }

  // ===========================================================================
  // Node state
  // ===========================================================================

  setNodePosition(id: NodeId, x: number, y: number, z = 0): boolean {
    return this.updateNode(id, { x, y, z });
  }

  pin(id: NodeId, pinned = true): boolean {
    return this.setFlag(id, NodeFlag.PINNED, pinned);
  }

  select(id: NodeId, selected = true): boolean {
    return this.setFlag(id, NodeFlag.SELECTED, selected);
  }

  clearSelection(): this {
    const n = this.store.nodeCount;
    for (let i = 0; i < n; i++) {
      if (this.store.hasFlag(i, NodeFlag.SELECTED)) this.store.setNodeFlag(i, NodeFlag.SELECTED, false);
    }
    this.requestFrame();
    return this;
  }

  setHidden(id: NodeId, hidden = true): boolean {
    return this.setFlag(id, NodeFlag.HIDDEN, hidden);
  }

  setNodeColor(id: NodeId, color: ColorInput): boolean {
    return this.updateNode(id, { color });
  }

  setEdgeColor(id: EdgeId, color: ColorInput, targetColor?: ColorInput): boolean {
    return this.updateEdge(id, { color, targetColor: targetColor ?? color });
  }

  /** Sets (or clears, with `undefined`) the text a `LabelLayer` draws next to the node. */
  setNodeLabel(id: NodeId, label: string | undefined): boolean {
    return this.updateNode(id, { label });
  }

  /** Sets (or clears, with `undefined`) the text a `LabelLayer` draws along the edge. */
  setEdgeLabel(id: EdgeId, label: string | undefined): boolean {
    return this.updateEdge(id, { label });
  }

  // ===========================================================================
  // Mode & physics
  // ===========================================================================

  get dimensionMode(): DimensionMode {
    return this.store.dimensionMode;
  }

  /**
   * Switches between 2D and 3D at runtime: flattens or Z-jitters positions,
   * swaps the quadtree/octree, toggles the engine's 2D camera and reheats.
   */
  setDimensionMode(mode: DimensionMode): this {
    this.assertAlive();
    const previous = this.store.dimensionMode;
    if (mode === previous) return this;
    this.interaction?.reset();
    this.store.setDimensionMode(mode);
    this.physics.setDimensionMode(mode);
    this.renderer.camera.setCamera2DMode(mode === '2d');
    this.modePayload.mode = mode;
    this.modePayload.previous = previous;
    this.events.emit('mode:change', this.modePayload);
    this.requestFrame();
    return this;
  }

  registerForce(force: IForce): this {
    this.physics.registerForce(force);
    this.reheat();
    return this;
  }

  unregisterForce(name: string): boolean {
    const ok = this.physics.unregisterForce(name);
    if (ok) this.reheat();
    return ok;
  }

  getForce<T extends IForce = IForce>(name: string): T | undefined {
    return this.physics.getForce<T>(name);
  }

  /** Re-energizes the layout (default: alpha raised to at least 0.3). */
  reheat(alpha?: number): this {
    this.physics.reheat(alpha);
    this.requestFrame();
    return this;
  }

  /** Freezes the simulation (rendering and interaction continue). */
  stopSimulation(): this {
    this.physics.stop();
    return this;
  }

  /**
   * Runs `ticks` physics steps synchronously (e.g. to pre-layout before the
   * first paint), then syncs and uploads once.
   */
  warmup(ticks: number): this {
    this.assertAlive();
    this.physics.tick(Math.max(0, Math.floor(ticks)));
    this.store.syncAllEdgeVertices();
    this.store.markAllNodesDirty();
    this.store.flush();
    return this;
  }

  // ===========================================================================
  // Events
  // ===========================================================================

  on<K extends keyof IGraphEventMap<TNodeMeta>>(event: K, listener: Listener<IGraphEventMap<TNodeMeta>[K]>): Unsubscribe {
    return this.events.on(event, listener);
  }

  once<K extends keyof IGraphEventMap<TNodeMeta>>(event: K, listener: Listener<IGraphEventMap<TNodeMeta>[K]>): Unsubscribe {
    return this.events.once(event, listener);
  }

  off<K extends keyof IGraphEventMap<TNodeMeta>>(event: K, listener: Listener<IGraphEventMap<TNodeMeta>[K]>): boolean {
    return this.events.off(event, listener);
  }

  // ===========================================================================
  // Loop
  // ===========================================================================

  /** Starts (or resumes) the animation loop. */
  start(): this {
    this.assertAlive();
    this.running = true;
    this.requestFrame();
    return this;
  }

  /** Stops the animation loop; buffers keep their last uploaded state. */
  stop(): this {
    this.running = false;
    if (this.frameScheduled) {
      this.scheduler.cancel(this.frameHandle);
      this.frameScheduled = false;
    }
    return this;
  }

  get isRunning(): boolean {
    return this.running;
  }

  dispose(): void {
    if (this.disposed) return;
    this.stop();
    this.interaction?.dispose();
    this.physics.dispose();
    this.renderer.dispose();
    this.events.clear();
    this.disposed = true;
  }

  /** Schedules one frame if none is pending. Stable identity: safe to pass as a callback. */
  readonly requestFrame = (): void => {
    if (!this.running || this.frameScheduled || this.disposed) return;
    this.frameScheduled = true;
    this.frameHandle = this.scheduler.request(this.frame);
  };

  private readonly frame = (): void => {
    this.frameScheduled = false;
    if (!this.running || this.disposed) return;

    this.interaction?.update();

    const physics = this.physics;
    const active = physics.isActive;
    if (active) {
      const start = now();
      let ticks = 0;
      do {
        physics.tick(1);
        ticks++;
      } while (ticks < this.ticksPerFrame && physics.isActive && now() - start < this.tickBudgetMs);

      // One contiguous pass for all edges instead of k-incremental syncs.
      this.store.syncAllEdgeVertices();
      this.store.markAllNodesDirty();

      if (this.events.hasListeners('engine:tick')) {
        this.tickPayload.alpha = physics.alpha;
        this.tickPayload.tickCount = physics.tickCount;
        this.events.emit('engine:tick', this.tickPayload);
      }
    }

    this.store.flush();

    const stillActive = physics.isActive;
    if (this.wasActive && !stillActive) {
      this.tickPayload.alpha = physics.alpha;
      this.tickPayload.tickCount = physics.tickCount;
      this.events.emit('engine:stabilized', this.tickPayload);
    }
    this.wasActive = stillActive;

    // Keep the loop alive only while there is work; otherwise sleep until woken.
    if (stillActive || (this.interaction !== null && this.interaction.isBusy)) this.requestFrame();
  };

  // ===========================================================================
  // Internals
  // ===========================================================================

  private afterMutation(opts?: MutationOptions): void {
    if (opts?.reheat ?? true) {
      this.physics.reheat(Math.max(this.physics.alpha, MUTATION_REHEAT_ALPHA));
    }
    this.requestFrame();
  }

  private setFlag(id: NodeId, flag: (typeof NodeFlag)[keyof typeof NodeFlag], on: boolean): boolean {
    this.assertAlive();
    const i = this.store.getNodeIndex(id);
    if (i === NO_INDEX) return false;
    this.store.setNodeFlag(i, flag, on);
    this.requestFrame();
    return true;
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error('O3DGraph: used after dispose()');
  }
}

function resolveStoreOptions(options: GraphOptions): Partial<GraphStoreOptions> {
  const d = DEFAULT_STORE_OPTIONS;
  return {
    initialNodeCapacity: options.initialNodeCapacity ?? d.initialNodeCapacity,
    initialEdgeCapacity: options.initialEdgeCapacity ?? d.initialEdgeCapacity,
    defaultNodeRadius: options.node?.radius ?? d.defaultNodeRadius,
    defaultNodeMass: options.node?.mass ?? d.defaultNodeMass,
    defaultNodeColor: parseColor(options.node?.color, d.defaultNodeColor),
    hoverColor: parseColor(options.node?.hoverColor, d.hoverColor),
    selectedColor: parseColor(options.node?.selectedColor, d.selectedColor),
    defaultEdgeColor: parseColor(options.edge?.color, d.defaultEdgeColor),
    defaultLinkDistance: options.edge?.distance ?? d.defaultLinkDistance,
    initialLayoutRadius: options.initialLayoutRadius ?? d.initialLayoutRadius,
    seed: options.seed ?? d.seed,
  };
}

function isGraphRenderer(source: GraphRenderSource): source is IGraphRenderer {
  return typeof (source as Partial<IGraphRenderer>).getNodeTransforms === 'function';
}

function isRawBindings(source: IO3DEngineAdapter | O3DWasmBindings): source is O3DWasmBindings {
  return typeof (source as Partial<O3DWasmBindings>).get_wasm_memory === 'function';
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function defaultScheduler(): FrameScheduler {
  if (typeof requestAnimationFrame === 'function') {
    return {
      request: (cb) => requestAnimationFrame(cb),
      cancel: (h) => cancelAnimationFrame(h),
    };
  }
  return {
    request: (cb) => setTimeout(() => cb(now()), 16) as unknown as number,
    cancel: (h) => clearTimeout(h as unknown as ReturnType<typeof setTimeout>),
  };
}
