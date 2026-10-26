/**
 * Core contracts for @omernach/o3d-graph.
 *
 * Every module depends on the abstractions declared here rather than on each
 * other's concrete classes (Dependency Inversion). Interfaces are deliberately
 * narrow (Interface Segregation): forces only see `IGraphBuffers`, the
 * interaction layer only sees `IGraphRenderer` + `IGraphStore`, and nothing but
 * the renderer adapter ever touches `IO3DEngineAdapter`.
 */

// ---------------------------------------------------------------------------
// Primitive aliases & constants
// ---------------------------------------------------------------------------

export type DimensionMode = '2d' | '3d';

/** Spatial dimensionality as a number, used by the physics / spatial layers. */
export type Dimensions = 2 | 3;

export type NodeId = string | number;
export type EdgeId = string | number;

/**
 * Accepted color inputs:
 *  - `number`: packed `0xRRGGBBAA` — 8 hex digits, alpha last (`0xff8800ff`)
 *  - `string`: CSS hex `#rgb`, `#rgba`, `#rrggbb`, `#rrggbbaa`, or
 *    `rgb(r, g, b)` / `rgba(r, g, b, a)`
 */
export type ColorInput = number | string;

/** Bitmask stored per node in `nodeFlags` (`Uint8Array`). */
export const NodeFlag = {
  NONE: 0,
  PINNED: 1 << 0,
  HOVERED: 1 << 1,
  SELECTED: 1 << 2,
  HIDDEN: 1 << 3,
} as const;

export type NodeFlagBit = (typeof NodeFlag)[keyof typeof NodeFlag];

/** `[x, y, z, scale]` per instance in the o3d InstancedMesh transform buffer. */
export const NODE_TRANSFORM_STRIDE = 4;
/** `[x, y, z]` for velocity / force SoA buffers. */
export const VEC3_STRIDE = 3;
/** `[x1, y1, z1, x2, y2, z2]` per segment in the o3d DynamicLineBuffer. */
export const EDGE_VERTEX_STRIDE = 6;
/** `[rgba_start, rgba_end]` per segment in the o3d DynamicLineBuffer. */
export const EDGE_COLOR_STRIDE = 2;

/** Sentinel for "no index" in Int32 index buffers (adjacency, tree children). */
export const NO_INDEX = -1;

/**
 * Sentinel stored in `edgeStrengths` meaning "derive the spring strength from
 * node degrees" (`1 / min(deg(source), deg(target))`), so hubs don't dominate.
 */
export const EDGE_STRENGTH_AUTO = -1;

// ---------------------------------------------------------------------------
// @omernach/o3d engine contract (DIP boundary)
// ---------------------------------------------------------------------------

/**
 * Camera capabilities the graph needs: drag-plane normal, cursor rays, orbit
 * locking while a node is dragged, and the 2D/3D projection switch.
 */
export interface ICameraAdapter {
  /** `[ox, oy, oz, dx, dy, dz]` with a unit direction. */
  unprojectNdcRay(ndcX: number, ndcY: number): Float32Array;
  /** Normalized `[fx, fy, fz]`. */
  getCameraForward(): Float32Array;
  /** 16 floats, column-major. */
  getCameraViewProjMatrix(): Float32Array;
  setCameraOrbitEnabled(enabled: boolean): void;
  setCamera2DMode(is2D: boolean): void;
}

/**
 * One-to-one, camel-cased mirror of the handle-based WASM primitives
 * (`create_instanced_mesh`, `get_instance_transforms_ptr`, …). All pointers
 * are byte offsets into `getWasmMemory().buffer`.
 *
 * Implement this interface to unit-test the graph layer without WASM or to
 * retarget it to an engine with this function-style API. For the
 * object-based `@omernach/o3d` renderer use `O3DBatchRenderer` instead.
 */
export interface IO3DEngineAdapter extends ICameraAdapter {
  getWasmMemory(): WebAssembly.Memory;

  // Hardware instancing (nodes)
  createInstancedMesh(geometryId: number, maxInstances: number): number;
  /** Pointer to `f32` buffer, stride 4: `[x, y, z, scale]`. */
  getInstanceTransformsPtr(meshHandle: number): number;
  /** Pointer to `u32` buffer, stride 1: packed color. */
  getInstanceColorsPtr(meshHandle: number): number;
  updateInstancedMesh(meshHandle: number, activeCount: number): void;
  updateInstanceSubRange(meshHandle: number, startIdx: number, count: number): void;
  /** Optional: release GPU/WASM resources of a mesh (used on capacity growth / dispose). */
  destroyInstancedMesh?(meshHandle: number): void;

  // Dynamic line batching (edges)
  createLineBuffer(maxSegments: number): number;
  /** Pointer to `f32` buffer, stride 6: `[x1, y1, z1, x2, y2, z2]`. */
  getLineVerticesPtr(lineHandle: number): number;
  /** Pointer to `u32` buffer, stride 2: `[rgba_start, rgba_end]`. */
  getLineColorsPtr(lineHandle: number): number;
  updateLineBuffer(lineHandle: number, activeSegments: number): void;
  updateLineSubRange(lineHandle: number, startSegment: number, segmentCount: number): void;
  /** Optional: release GPU/WASM resources of a line buffer. */
  destroyLineBuffer?(lineHandle: number): void;

  /** Closest hit instance index or `-1`. `spheresPtr` points at `[x, y, z, r]` records. */
  raycastSpheresNdc(ndcX: number, ndcY: number, spheresPtr: number, count: number): number;
}

/**
 * Byte order of packed `u32` colors in the backend's buffers.
 *  - `'rgba'`: `0xRRGGBBAA` (red in the highest byte)
 *  - `'abgr'`: red in the lowest byte (`r | g << 8 | b << 16 | a << 24`), i.e.
 *    WebGPU `unorm8x4` on little-endian — what `@omernach/o3d` uses.
 */
export type ColorLayout = 'rgba' | 'abgr';

// ---------------------------------------------------------------------------
// Renderer abstraction (consumed by GraphStore / InteractionController)
// ---------------------------------------------------------------------------

/**
 * Graph-level view of the rendering backend: owns the node mesh + edge line
 * buffer and hands out zero-copy TypedArray views over WASM memory.
 *
 * Views returned by the `get*` methods are validated on every call (cheap
 * identity check) and transparently re-bound after `memory.grow()` or capacity
 * growth. **Never cache a view across an engine call or across frames** —
 * fetch it once at the top of a hot loop instead.
 */
export interface IGraphRenderer {
  readonly camera: ICameraAdapter;
  readonly nodeCapacity: number;
  readonly edgeCapacity: number;
  /** Incremented every time views are re-bound (memory growth / reallocation). */
  readonly viewVersion: number;

  getNodeTransforms(): Float32Array;
  getNodeColors(): Uint32Array;
  getEdgeVertices(): Float32Array;
  getEdgeColors(): Uint32Array;
  getNodeTransformsPtr(): number;

  /** Grows the node mesh to hold at least `required` instances. Returns `true` if it reallocated. */
  ensureNodeCapacity(required: number): boolean;
  /** Grows the line buffer to hold at least `required` segments. Returns `true` if it reallocated. */
  ensureEdgeCapacity(required: number): boolean;

  commitNodes(activeCount: number): void;
  commitNodeRange(start: number, count: number): void;
  commitEdges(activeCount: number): void;
  commitEdgeRange(start: number, count: number): void;

  /** Ray-sphere pick over the first `count` node transforms. Returns node index or `-1`. */
  raycastNodes(ndcX: number, ndcY: number, count: number): number;

  /**
   * Converts a public `0xRRGGBBAA` color into the backend's native packed
   * layout. The store calls this for every color it writes to the buffers.
   */
  encodeColor(rgba: number): number;

  dispose(): void;
}

// ---------------------------------------------------------------------------
// Public data model
// ---------------------------------------------------------------------------

export interface NodeInput<TMeta = unknown> {
  id: NodeId;
  x?: number;
  y?: number;
  z?: number;
  radius?: number;
  color?: ColorInput;
  /** Barnes-Hut charge weight. Default `1`. */
  mass?: number;
  pinned?: boolean;
  hidden?: boolean;
  /** Text drawn next to the node by a `LabelLayer`. */
  label?: string;
  meta?: TMeta;
}

export interface EdgeInput<TMeta = unknown> {
  /** Auto-generated when omitted. */
  id?: EdgeId;
  source: NodeId;
  target: NodeId;
  /** Spring rest length. Defaults to `GraphStoreOptions.defaultLinkDistance`. */
  distance?: number;
  /** Spring stiffness in `[0, 1]`. Omit for degree-weighted auto strength. */
  strength?: number;
  /** Color at the source end. */
  color?: ColorInput;
  /** Color at the target end. Defaults to `color`. */
  targetColor?: ColorInput;
  /** Text drawn along the edge by a `LabelLayer`. */
  label?: string;
  meta?: TMeta;
}

export type NodePatch<TMeta = unknown> = Partial<Omit<NodeInput<TMeta>, 'id'>>;
export type EdgePatch<TMeta = unknown> = Partial<
  Pick<EdgeInput<TMeta>, 'distance' | 'strength' | 'color' | 'targetColor' | 'label' | 'meta'>
>;

export interface GraphData<TNodeMeta = unknown, TEdgeMeta = unknown> {
  nodes: ReadonlyArray<NodeInput<TNodeMeta>>;
  edges?: ReadonlyArray<EdgeInput<TEdgeMeta>>;
}

// ---------------------------------------------------------------------------
// Store contracts
// ---------------------------------------------------------------------------

/** Fully-resolved store configuration (the facade fills defaults). */
export interface GraphStoreOptions {
  initialNodeCapacity: number;
  initialEdgeCapacity: number;
  defaultNodeRadius: number;
  defaultNodeMass: number;
  defaultNodeColor: number;
  hoverColor: number;
  selectedColor: number;
  defaultEdgeColor: number;
  defaultLinkDistance: number;
  /** Scale of the deterministic phyllotaxis layout used for nodes without coordinates. */
  initialLayoutRadius: number;
  /** Seed for deterministic jitter (e.g. 2D → 3D Z injection). */
  seed: number;
}

/**
 * Read-mostly SoA view of the graph, sufficient for forces and spatial
 * indexing. Buffers may be reallocated on capacity growth — re-read the
 * properties once per tick, never cache them across ticks.
 *
 * Adjacency is an intrusive doubly-linked half-edge list:
 * edge `e` owns half-edges `2e` (source side) and `2e + 1` (target side).
 * Iterate a node's incident edges with zero allocation:
 *
 * ```ts
 * for (let h = g.nodeFirstHalfEdge[i]; h !== NO_INDEX; h = g.halfEdgeNext[h]) {
 *   const e = h >> 1;
 *   const other = (h & 1) === 0 ? g.edgeTargets[e] : g.edgeSources[e];
 * }
 * ```
 */
export interface IGraphBuffers {
  readonly nodeCount: number;
  readonly edgeCount: number;
  readonly nodeCapacity: number;
  readonly edgeCapacity: number;
  /** Bumped on any node/edge insertion or removal (not on position changes). */
  readonly topologyVersion: number;
  readonly hiddenCount: number;
  readonly dimensionMode: DimensionMode;

  /** Zero-copy WASM view, stride 4 `[x, y, z, scale]`. See `IGraphRenderer` caveats. */
  getNodeTransforms(): Float32Array;

  readonly nodeVelocities: Float32Array;
  readonly nodeForces: Float32Array;
  readonly nodeMasses: Float32Array;
  /** Logical radius (kept even while a node is hidden and rendered at scale 0). */
  readonly nodeRadii: Float32Array;
  readonly nodeDegrees: Uint32Array;
  readonly nodeFlags: Uint8Array;
  readonly nodeFirstHalfEdge: Int32Array;

  readonly edgeSources: Uint32Array;
  readonly edgeTargets: Uint32Array;
  readonly edgeDistances: Float32Array;
  /** `EDGE_STRENGTH_AUTO` (negative) means degree-weighted. */
  readonly edgeStrengths: Float32Array;
  readonly halfEdgeNext: Int32Array;
  readonly halfEdgePrev: Int32Array;
}

export interface IGraphStore<TNodeMeta = unknown, TEdgeMeta = unknown> extends IGraphBuffers {
  // Identity
  hasNode(id: NodeId): boolean;
  hasEdge(id: EdgeId): boolean;
  getNodeIndex(id: NodeId): number;
  getNodeId(index: number): NodeId;
  getNodeMeta(index: number): TNodeMeta | undefined;
  getEdgeIndex(id: EdgeId): number;
  getEdgeId(index: number): EdgeId;
  getEdgeMeta(index: number): TEdgeMeta | undefined;
  getNodeLabel(index: number): string | undefined;
  getEdgeLabel(index: number): string | undefined;

  // Mutation
  setData(data: GraphData<TNodeMeta, TEdgeMeta>): void;
  addNode(input: NodeInput<TNodeMeta>): number;
  addEdge(input: EdgeInput<TEdgeMeta>): number;
  updateNode(id: NodeId, patch: NodePatch<TNodeMeta>): boolean;
  updateEdge(id: EdgeId, patch: EdgePatch<TEdgeMeta>): boolean;
  removeNode(id: NodeId): boolean;
  removeEdge(id: EdgeId): boolean;
  clear(): void;

  // Per-index hot-path helpers
  hasFlag(index: number, flag: NodeFlagBit): boolean;
  setNodeFlag(index: number, flag: NodeFlagBit, on: boolean): boolean;
  setNodePosition(index: number, x: number, y: number, z: number): void;
  /** Writes `[x, y, z]` into `out` at `offset` without allocating. */
  readNodePosition(index: number, out: Float32Array | number[], offset?: number): void;
  setNodeColor(index: number, color: ColorInput): void;
  setEdgeColor(index: number, sourceColor: ColorInput, targetColor?: ColorInput): void;

  // Edge vertex synchronization
  syncAllEdgeVertices(): void;
  /** O(k) update of only the edges incident to `index`. Returns k. */
  syncNodeEdges(index: number): number;

  // Dirty tracking → renderer upload
  markNodeDirty(index: number): void;
  markAllNodesDirty(): void;
  markEdgeDirty(index: number): void;
  markAllEdgesDirty(): void;
  /** Uploads dirty node/edge ranges to the renderer (full or sub-range). */
  flush(): void;

  /** Clamps or jitters Z as required. Returns `false` if the mode was unchanged. */
  setDimensionMode(mode: DimensionMode): boolean;
}

// ---------------------------------------------------------------------------
// Spatial indexing contract (implemented in Stage 2)
// ---------------------------------------------------------------------------

/**
 * Flat-array Barnes-Hut tree. Quadtree (4 children) in 2D, octree (8 children)
 * in 3D. Cells are cubic/square, so a single half-size describes the bounds.
 *
 * All arrays are indexed by cell index `c` (children by `c * branching + k`)
 * and remain valid until the next `build()` / `setDimensions()`.
 */
export interface ISpatialTree {
  readonly dimensions: Dimensions;
  readonly branching: 4 | 8;
  /** Root cell index, or `NO_INDEX` when the tree is empty. */
  readonly root: number;
  readonly cellCount: number;

  /** Child cell indices, stride `branching`, `NO_INDEX` for empty octants. */
  readonly cellChildren: Int32Array;
  /** Geometric cell center, stride 3. */
  readonly cellCenter: Float32Array;
  /** Half the cell edge length, stride 1. */
  readonly cellHalfSize: Float32Array;
  /** Mass-weighted center, stride 3. */
  readonly cellMassCenter: Float32Array;
  /** Aggregated mass (charge weight), stride 1. */
  readonly cellMass: Float32Array;
  /** Largest body radius inside the cell (used by collision pruning). */
  readonly cellMaxRadius: Float32Array;
  /** First body in a leaf, `NO_INDEX` for internal cells. */
  readonly cellFirstBody: Int32Array;
  /** Next coincident body in the same leaf, indexed by node index. */
  readonly bodyNext: Int32Array;
  /**
   * Every inserted body in depth-first (spatially coherent) order. Iterating
   * nodes in this order keeps consecutive traversals on the same tree paths,
   * which is markedly more cache-friendly than index order.
   */
  readonly bodyOrder: Int32Array;
  /** Number of valid entries in `bodyOrder` (= inserted, visible bodies). */
  readonly bodyCount: number;
  /** Number of bodies in each cell's subtree. */
  readonly cellBodyCount: Int32Array;
  /**
   * Offset of each cell's first body in `bodyOrder`; a cell's subtree occupies
   * the contiguous range `[cellOrderStart[c], cellOrderStart[c] + cellBodyCount[c])`.
   */
  readonly cellOrderStart: Int32Array;

  setDimensions(dimensions: Dimensions): void;
  /** Rebuilds the tree in place, skipping nodes flagged `HIDDEN`. */
  build(graph: IGraphBuffers): void;
}

// ---------------------------------------------------------------------------
// Physics contracts (implemented in Stage 2)
// ---------------------------------------------------------------------------

/**
 * Per-tick context handed to every force. The simulator reuses a single
 * instance for the lifetime of the simulation — never retain it.
 */
export interface ForceTickContext {
  readonly alpha: number;
  readonly dimensions: Dimensions;
  readonly nodeCount: number;
  readonly edgeCount: number;
  /** Zero-copy WASM node transforms, stride 4. */
  readonly positions: Float32Array;
  readonly velocities: Float32Array;
  /** Force accumulator, stride 3; zeroed by the simulator before forces run. */
  readonly forces: Float32Array;
  readonly graph: IGraphBuffers;
  /** Built once per tick if any enabled force sets `requiresSpatialTree`. */
  readonly tree: ISpatialTree;
}

/** Pluggable force strategy (Open/Closed: add forces without touching the simulator). */
export interface IForce {
  readonly name: string;
  enabled: boolean;
  readonly requiresSpatialTree: boolean;
  /** Called on registration, topology change and dimension change. */
  initialize(graph: IGraphBuffers, dimensions: Dimensions): void;
  /** Accumulates into `ctx.forces` (or adjusts `ctx.velocities` for constraint forces). */
  apply(ctx: ForceTickContext): void;
  dispose?(): void;
}

export interface PhysicsOptions {
  /** Initial alpha (temperature). Default `1`. */
  alpha?: number;
  /** Simulation idles once `alpha` drops below this. Default `0.001`. */
  alphaMin?: number;
  /** Per-tick cooling rate. Default `1 - alphaMin^(1/300)` (~300 ticks). */
  alphaDecay?: number;
  /** Alpha converges toward this target. Default `0`. */
  alphaTarget?: number;
  /** Fraction of velocity removed per tick (friction). Default `0.4`. */
  velocityDecay?: number;
  /** Integration time step. Default `1`. */
  timeStep?: number;
  /** Barnes-Hut opening angle. Default `0.8`. */
  theta?: number;
  /** Many-body strength; negative repels. Default `-30`. */
  chargeStrength?: number;
  /** Minimum interaction distance (avoids singularities). Default `1`. */
  chargeDistanceMin?: number;
  /** Maximum interaction distance. Default `Infinity`. */
  chargeDistanceMax?: number;
  /** Link relaxation iterations per tick. Default `1`. */
  linkIterations?: number;
  /** Center force strength. Default `1`. */
  centerStrength?: number;
  /** Optional radial force radius; `0` disables. Default `0`. */
  radialRadius?: number;
  /** Radial force strength. Default `0.1`. */
  radialStrength?: number;
  /** Enables collision resolution. Default `true`. */
  collide?: boolean;
  /** Collision strength in `[0, 1]`. Default `0.7`. */
  collideStrength?: number;
  /** Extra spacing added to radii. Default `1`. */
  collidePadding?: number;
  /** Collision iterations per tick. Default `1`. */
  collideIterations?: number;
}

export interface IPhysicsEngine {
  alpha: number;
  alphaMin: number;
  alphaDecay: number;
  alphaTarget: number;
  velocityDecay: number;
  readonly dimensions: Dimensions;
  readonly tickCount: number;
  /** `true` while `alpha >= alphaMin` or `alphaTarget > alphaMin`. */
  readonly isActive: boolean;
  readonly tree: ISpatialTree;

  registerForce(force: IForce): void;
  unregisterForce(name: string): boolean;
  getForce<T extends IForce = IForce>(name: string): T | undefined;

  setDimensionMode(mode: DimensionMode): void;
  notifyTopologyChanged(): void;

  /** Runs up to `iterations` ticks. Returns `true` if still active afterward. */
  tick(iterations?: number): boolean;
  /** Sets alpha (default: current `alpha` raised to at least `0.3`) and marks active. */
  reheat(alpha?: number): void;
  restart(): void;
  stop(): void;
  dispose(): void;
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

export interface NodeEventPayload<TNodeMeta = unknown> {
  id: NodeId;
  index: number;
  meta: TNodeMeta | undefined;
  /** Originating DOM event, when there is one. */
  event: PointerEvent | null;
}

export interface NodeDragPayload<TNodeMeta = unknown> extends NodeEventPayload<TNodeMeta> {
  x: number;
  y: number;
  z: number;
}

export interface TickPayload {
  alpha: number;
  tickCount: number;
}

export interface ModeChangePayload {
  mode: DimensionMode;
  previous: DimensionMode;
}

/**
 * Event map for the graph's EventBus. High-frequency payloads (`engine:tick`,
 * `node:drag`) are pooled and reused between emissions — copy fields if you
 * need to keep them beyond the listener call.
 */
export interface IGraphEventMap<TNodeMeta = unknown> {
  'node:click': NodeEventPayload<TNodeMeta>;
  'node:hover': NodeEventPayload<TNodeMeta>;
  'node:leave': NodeEventPayload<TNodeMeta>;
  'node:dragstart': NodeDragPayload<TNodeMeta>;
  'node:drag': NodeDragPayload<TNodeMeta>;
  'node:dragend': NodeDragPayload<TNodeMeta>;
  'engine:tick': TickPayload;
  'engine:stabilized': TickPayload;
  'mode:change': ModeChangePayload;
}

export type Unsubscribe = () => void;

// ---------------------------------------------------------------------------
// Public facade options (consumed in Stage 3)
// ---------------------------------------------------------------------------

export interface InteractionOptions {
  /** Master switch for pointer handling. Default `true`. */
  enabled?: boolean;
  hover?: boolean;
  drag?: boolean;
  /** Keep the node pinned where it was dropped. Default `false`. */
  pinOnDragEnd?: boolean;
  /** `alphaTarget` while dragging. Default `0.3`. */
  dragAlphaTarget?: number;
  /**
   * Reheat the simulation while dragging so neighbors react. When `false`
   * (static mode) only the dragged node and its k incident edges are updated,
   * in O(k). Default `true`.
   */
  reheatOnDrag?: boolean;
  /** Max pointer travel (px) for a press to still count as a click. Default `4`. */
  clickTolerancePx?: number;
}

export interface GraphOptions {
  /** Canvas the o3d engine renders into; pointer events are bound to it. */
  canvas: HTMLCanvasElement;
  dimension?: DimensionMode;
  initialNodeCapacity?: number;
  initialEdgeCapacity?: number;
  /** o3d geometry used for node instances. Default `0` (sphere). */
  nodeGeometryId?: number;
  node?: {
    radius?: number;
    mass?: number;
    color?: ColorInput;
    hoverColor?: ColorInput;
    selectedColor?: ColorInput;
  };
  edge?: {
    color?: ColorInput;
    distance?: number;
  };
  physics?: PhysicsOptions;
  interaction?: InteractionOptions;
  /** Start the rAF loop immediately. Default `true`. */
  autoStart?: boolean;
  /** Max physics ticks per animation frame while active. Default `1`. */
  ticksPerFrame?: number;
  /**
   * Wall-clock budget (ms) for physics per frame. Further ticks in the same
   * frame are skipped once exceeded; at least one tick always runs.
   * Default `12`.
   */
  tickBudgetMs?: number;
  /** Seed for every deterministic pseudo-random operation. Default `1`. */
  seed?: number;
  initialLayoutRadius?: number;
}
