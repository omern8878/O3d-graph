import {
  EDGE_COLOR_STRIDE,
  EDGE_STRENGTH_AUTO,
  EDGE_VERTEX_STRIDE,
  NODE_TRANSFORM_STRIDE,
  NO_INDEX,
  NodeFlag,
  VEC3_STRIDE,
  type ColorInput,
  type DimensionMode,
  type EdgeId,
  type EdgeInput,
  type EdgePatch,
  type GraphData,
  type GraphStoreOptions,
  type IGraphRenderer,
  type IGraphStore,
  type NodeFlagBit,
  type NodeId,
  type NodeInput,
  type NodePatch,
} from '../types/index.js';
import { growFloat32, growInt32, growUint32, growUint8, nextCapacity } from '../utils/capacity.js';
import { parseColor } from '../utils/color.js';
import { Lcg } from '../utils/random.js';

export const DEFAULT_STORE_OPTIONS: Readonly<GraphStoreOptions> = Object.freeze({
  initialNodeCapacity: 1024,
  initialEdgeCapacity: 2048,
  defaultNodeRadius: 4,
  defaultNodeMass: 1,
  defaultNodeColor: 0x4f9dffff,
  hoverColor: 0xffc857ff,
  selectedColor: 0xff5d73ff,
  defaultEdgeColor: 0x8a93a666,
  defaultLinkDistance: 30,
  initialLayoutRadius: 10,
  seed: 1,
});

/** Z jitter amplitude injected when leaving 2D mode with co-planar nodes. */
const Z_JITTER = 5;
const COPLANAR_EPSILON = 1e-6;
/** Phyllotaxis constants (same as d3-force / d3-force-3d). */
const ANGLE_ROLL = Math.PI * (3 - Math.sqrt(5));
const ANGLE_YAW = (Math.PI * 20) / (9 + Math.sqrt(221));

const INT_MAX = 0x7fffffff;

/**
 * Structure-of-Arrays graph storage.
 *
 * Positions/radii live in the renderer's WASM transform buffer
 * (`[x, y, z, scale]`) and edge endpoints in its line vertex buffer — the
 * store writes there directly and never copies them into JS memory. Physics
 * state (velocity, force, mass, flags…) lives in JS-owned TypedArrays.
 *
 * Indices are dense `[0, count)`; deletion is O(1) swap-and-pop (O(k) for a
 * node's k incident edges). Incident edges are kept in an intrusive
 * doubly-linked half-edge list, so adjacency iteration, insertion and removal
 * are allocation-free.
 *
 * Index stability: a removal moves the last element into the freed slot.
 * Consumers that hold indices across mutations (hover state, selections)
 * should compare `topologyVersion` and re-resolve through IDs.
 */
export class GraphStore<TNodeMeta = unknown, TEdgeMeta = unknown>
  implements IGraphStore<TNodeMeta, TEdgeMeta>
{
  private readonly renderer: IGraphRenderer;
  private readonly opts: GraphStoreOptions;
  private readonly rng: Lcg;

  private _nodeCount = 0;
  private _edgeCount = 0;
  private _nodeCapacity: number;
  private _edgeCapacity: number;
  private _topologyVersion = 0;
  private _hiddenCount = 0;
  private _dimensionMode: DimensionMode = '3d';

  // Node SoA
  private _nodeVelocities: Float32Array;
  private _nodeForces: Float32Array;
  private _nodeMasses: Float32Array;
  private _nodeRadii: Float32Array;
  private _nodeDegrees: Uint32Array;
  private _nodeFlags: Uint8Array;
  private _nodeBaseColors: Uint32Array;
  private _nodeFirstHalfEdge: Int32Array;

  // Edge SoA (+ half-edge adjacency, 2 slots per edge)
  private _edgeSources: Uint32Array;
  private _edgeTargets: Uint32Array;
  private _edgeDistances: Float32Array;
  private _edgeStrengths: Float32Array;
  private _halfEdgeNext: Int32Array;
  private _halfEdgePrev: Int32Array;

  // Identity & metadata (index-aligned dense arrays)
  private readonly nodeIdToIndex = new Map<NodeId, number>();
  private readonly nodeIds: NodeId[] = [];
  private readonly nodeMeta: Array<TNodeMeta | undefined> = [];
  private readonly nodeLabels: Array<string | undefined> = [];
  private readonly edgeIdToIndex = new Map<EdgeId, number>();
  private readonly edgeIds: EdgeId[] = [];
  private readonly edgeMeta: Array<TEdgeMeta | undefined> = [];
  private readonly edgeLabels: Array<string | undefined> = [];
  private edgeAutoId = 0;

  // Dirty ranges for renderer upload
  private nodesAllDirty = true;
  private nodeDirtyMin = INT_MAX;
  private nodeDirtyMax = -1;
  private edgesAllDirty = true;
  private edgeDirtyMin = INT_MAX;
  private edgeDirtyMax = -1;

  constructor(
    renderer: IGraphRenderer,
    options: Partial<GraphStoreOptions> = {},
    dimensionMode: DimensionMode = '3d',
  ) {
    this.renderer = renderer;
    this.opts = { ...DEFAULT_STORE_OPTIONS, ...options };
    this.rng = new Lcg(this.opts.seed);
    this._dimensionMode = dimensionMode;

    const nCap = nextCapacity(0, Math.max(1, this.opts.initialNodeCapacity));
    const eCap = nextCapacity(0, Math.max(1, this.opts.initialEdgeCapacity));
    this._nodeCapacity = nCap;
    this._edgeCapacity = eCap;

    this._nodeVelocities = new Float32Array(nCap * VEC3_STRIDE);
    this._nodeForces = new Float32Array(nCap * VEC3_STRIDE);
    this._nodeMasses = new Float32Array(nCap);
    this._nodeRadii = new Float32Array(nCap);
    this._nodeDegrees = new Uint32Array(nCap);
    this._nodeFlags = new Uint8Array(nCap);
    this._nodeBaseColors = new Uint32Array(nCap);
    this._nodeFirstHalfEdge = new Int32Array(nCap).fill(NO_INDEX);

    this._edgeSources = new Uint32Array(eCap);
    this._edgeTargets = new Uint32Array(eCap);
    this._edgeDistances = new Float32Array(eCap);
    this._edgeStrengths = new Float32Array(eCap);
    this._halfEdgeNext = new Int32Array(eCap * 2).fill(NO_INDEX);
    this._halfEdgePrev = new Int32Array(eCap * 2).fill(NO_INDEX);

    renderer.ensureNodeCapacity(nCap);
    renderer.ensureEdgeCapacity(eCap);
  }

  // ===========================================================================
  // IGraphBuffers
  // ===========================================================================

  get nodeCount(): number {
    return this._nodeCount;
  }
  get edgeCount(): number {
    return this._edgeCount;
  }
  get nodeCapacity(): number {
    return this._nodeCapacity;
  }
  get edgeCapacity(): number {
    return this._edgeCapacity;
  }
  get topologyVersion(): number {
    return this._topologyVersion;
  }
  get hiddenCount(): number {
    return this._hiddenCount;
  }
  get dimensionMode(): DimensionMode {
    return this._dimensionMode;
  }

  getNodeTransforms(): Float32Array {
    return this.renderer.getNodeTransforms();
  }

  get nodeVelocities(): Float32Array {
    return this._nodeVelocities;
  }
  get nodeForces(): Float32Array {
    return this._nodeForces;
  }
  get nodeMasses(): Float32Array {
    return this._nodeMasses;
  }
  get nodeRadii(): Float32Array {
    return this._nodeRadii;
  }
  get nodeDegrees(): Uint32Array {
    return this._nodeDegrees;
  }
  get nodeFlags(): Uint8Array {
    return this._nodeFlags;
  }
  get nodeFirstHalfEdge(): Int32Array {
    return this._nodeFirstHalfEdge;
  }
  get edgeSources(): Uint32Array {
    return this._edgeSources;
  }
  get edgeTargets(): Uint32Array {
    return this._edgeTargets;
  }
  get edgeDistances(): Float32Array {
    return this._edgeDistances;
  }
  get edgeStrengths(): Float32Array {
    return this._edgeStrengths;
  }
  get halfEdgeNext(): Int32Array {
    return this._halfEdgeNext;
  }
  get halfEdgePrev(): Int32Array {
    return this._halfEdgePrev;
  }

  // ===========================================================================
  // Identity
  // ===========================================================================

  hasNode(id: NodeId): boolean {
    return this.nodeIdToIndex.has(id);
  }
  hasEdge(id: EdgeId): boolean {
    return this.edgeIdToIndex.has(id);
  }
  getNodeIndex(id: NodeId): number {
    const i = this.nodeIdToIndex.get(id);
    return i === undefined ? NO_INDEX : i;
  }
  getNodeId(index: number): NodeId {
    this.assertNodeIndex(index);
    return this.nodeIds[index];
  }
  getNodeMeta(index: number): TNodeMeta | undefined {
    this.assertNodeIndex(index);
    return this.nodeMeta[index];
  }
  getEdgeIndex(id: EdgeId): number {
    const e = this.edgeIdToIndex.get(id);
    return e === undefined ? NO_INDEX : e;
  }
  getEdgeId(index: number): EdgeId {
    this.assertEdgeIndex(index);
    return this.edgeIds[index];
  }
  getEdgeMeta(index: number): TEdgeMeta | undefined {
    this.assertEdgeIndex(index);
    return this.edgeMeta[index];
  }
  getNodeLabel(index: number): string | undefined {
    return this.nodeLabels[index];
  }
  getEdgeLabel(index: number): string | undefined {
    return this.edgeLabels[index];
  }

  // ===========================================================================
  // Bulk load
  // ===========================================================================

  /**
   * Replaces the whole graph. Capacity is reserved once up front, so loading
   * N nodes / E edges performs at most one reallocation per buffer.
   * Throws on duplicate IDs or dangling edge endpoints (state is then partial).
   */
  setData(data: GraphData<TNodeMeta, TEdgeMeta>): void {
    this.clear();
    const nodes = data.nodes;
    const edges = data.edges ?? [];
    this.reserve(nodes.length, edges.length);
    for (let i = 0; i < nodes.length; i++) this.insertNode(nodes[i]);
    for (let i = 0; i < edges.length; i++) this.insertEdge(edges[i]);
    this._topologyVersion++;
    this.syncAllEdgeVertices();
    this.nodesAllDirty = true;
    this.edgesAllDirty = true;
  }

  /** Pre-allocates capacity for `nodeCount` nodes and `edgeCount` edges in total. */
  reserve(nodeCount: number, edgeCount: number): void {
    this.ensureNodeCapacity(nodeCount);
    this.ensureEdgeCapacity(edgeCount);
  }

  clear(): void {
    this._nodeCount = 0;
    this._edgeCount = 0;
    this._hiddenCount = 0;
    this.nodeIdToIndex.clear();
    this.nodeIds.length = 0;
    this.nodeMeta.length = 0;
    this.nodeLabels.length = 0;
    this.edgeIdToIndex.clear();
    this.edgeIds.length = 0;
    this.edgeMeta.length = 0;
    this.edgeLabels.length = 0;
    this.edgeAutoId = 0;
    this._topologyVersion++;
    this.nodesAllDirty = true;
    this.edgesAllDirty = true;
  }

  // ===========================================================================
  // Nodes
  // ===========================================================================

  addNode(input: NodeInput<TNodeMeta>): number {
    const i = this.insertNode(input);
    this._topologyVersion++;
    this.nodesAllDirty = true;
    return i;
  }

  updateNode(id: NodeId, patch: NodePatch<TNodeMeta>): boolean {
    const i = this.nodeIdToIndex.get(id);
    if (i === undefined) return false;
    const t = this.renderer.getNodeTransforms();
    const o = i * NODE_TRANSFORM_STRIDE;

    if (patch.x !== undefined || patch.y !== undefined || patch.z !== undefined) {
      this.setNodePosition(
        i,
        patch.x ?? t[o],
        patch.y ?? t[o + 1],
        patch.z ?? t[o + 2],
      );
    }
    if (patch.radius !== undefined) {
      const r = positiveOr(patch.radius, this.opts.defaultNodeRadius);
      this._nodeRadii[i] = r;
      if ((this._nodeFlags[i] & NodeFlag.HIDDEN) === 0) {
        this.renderer.getNodeTransforms()[o + 3] = r;
        this.markNodeDirty(i);
      }
    }
    if (patch.mass !== undefined) {
      this._nodeMasses[i] = finiteOr(patch.mass, this.opts.defaultNodeMass);
    }
    if (patch.color !== undefined) this.setNodeColor(i, patch.color);
    if (patch.pinned !== undefined) this.setNodeFlag(i, NodeFlag.PINNED, patch.pinned);
    if (patch.hidden !== undefined) this.setNodeFlag(i, NodeFlag.HIDDEN, patch.hidden);
    if ('label' in patch) {
      this.nodeLabels[i] = patch.label;
      this.markNodeDirty(i); // schedules a frame so label overlays refresh
    }
    if ('meta' in patch) this.nodeMeta[i] = patch.meta;
    return true;
  }

  removeNode(id: NodeId): boolean {
    const i = this.nodeIdToIndex.get(id);
    if (i === undefined) return false;

    // Detach incident edges first. Each removal unlinks the head half-edge,
    // so re-reading the head terminates even when swap-and-pop relocates edges.
    while (this._nodeFirstHalfEdge[i] !== NO_INDEX) {
      this.removeEdgeAt(this._nodeFirstHalfEdge[i] >> 1);
    }

    if ((this._nodeFlags[i] & NodeFlag.HIDDEN) !== 0) this._hiddenCount--;

    const last = this._nodeCount - 1;
    if (i !== last) this.moveNode(last, i);

    this.nodeIdToIndex.delete(id);
    this.nodeIds.pop();
    this.nodeMeta.pop();
    this.nodeLabels.pop();
    this._nodeCount = last;
    this._topologyVersion++;
    this.nodesAllDirty = true;
    return true;
  }

  hasFlag(index: number, flag: NodeFlagBit): boolean {
    return (this._nodeFlags[index] & flag) !== 0;
  }

  /** Sets/clears a flag with all side effects (color refresh, visibility). Returns `true` if changed. */
  setNodeFlag(index: number, flag: NodeFlagBit, on: boolean): boolean {
    this.assertNodeIndex(index);
    const prev = this._nodeFlags[index];
    const next = on ? prev | flag : prev & ~flag;
    if (next === prev) return false;
    this._nodeFlags[index] = next;

    if ((flag & NodeFlag.HIDDEN) !== 0) this.applyVisibility(index, on);
    if ((flag & (NodeFlag.HOVERED | NodeFlag.SELECTED)) !== 0) this.refreshNodeColor(index);
    if ((flag & NodeFlag.PINNED) !== 0) {
      const v = index * VEC3_STRIDE;
      this._nodeVelocities[v] = 0;
      this._nodeVelocities[v + 1] = 0;
      this._nodeVelocities[v + 2] = 0;
    }
    return true;
  }

  /** Writes a position straight into WASM memory, zeroes velocity, and updates incident edges (O(k)). */
  setNodePosition(index: number, x: number, y: number, z: number): void {
    const t = this.renderer.getNodeTransforms();
    const o = index * NODE_TRANSFORM_STRIDE;
    t[o] = x;
    t[o + 1] = y;
    t[o + 2] = this._dimensionMode === '2d' ? 0 : z;
    const v = index * VEC3_STRIDE;
    this._nodeVelocities[v] = 0;
    this._nodeVelocities[v + 1] = 0;
    this._nodeVelocities[v + 2] = 0;
    this.markNodeDirty(index);
    this.syncNodeEdges(index);
  }

  readNodePosition(index: number, out: Float32Array | number[], offset = 0): void {
    const t = this.renderer.getNodeTransforms();
    const o = index * NODE_TRANSFORM_STRIDE;
    out[offset] = t[o];
    out[offset + 1] = t[o + 1];
    out[offset + 2] = t[o + 2];
  }

  setNodeColor(index: number, color: ColorInput): void {
    this.assertNodeIndex(index);
    this._nodeBaseColors[index] = parseColor(color, this.opts.defaultNodeColor);
    this.refreshNodeColor(index);
  }

  getNodeBaseColor(index: number): number {
    return this._nodeBaseColors[index];
  }

  // ===========================================================================
  // Edges
  // ===========================================================================

  addEdge(input: EdgeInput<TEdgeMeta>): number {
    const e = this.insertEdge(input);
    this.writeEdgeVertices(e, this.renderer.getNodeTransforms(), this.renderer.getEdgeVertices());
    this._topologyVersion++;
    this.edgesAllDirty = true;
    return e;
  }

  updateEdge(id: EdgeId, patch: EdgePatch<TEdgeMeta>): boolean {
    const e = this.edgeIdToIndex.get(id);
    if (e === undefined) return false;
    if (patch.distance !== undefined) {
      this._edgeDistances[e] = positiveOr(patch.distance, this.opts.defaultLinkDistance);
    }
    if (patch.strength !== undefined) {
      this._edgeStrengths[e] = clampStrength(patch.strength);
    }
    if (patch.color !== undefined || patch.targetColor !== undefined) {
      const r = this.renderer;
      const c = r.getEdgeColors();
      const o = e * EDGE_COLOR_STRIDE;
      // Buffer values are already in the backend layout; only fresh input is encoded.
      const src =
        patch.color !== undefined ? r.encodeColor(parseColor(patch.color, this.opts.defaultEdgeColor)) : c[o];
      const dst =
        patch.targetColor !== undefined
          ? r.encodeColor(parseColor(patch.targetColor, this.opts.defaultEdgeColor))
          : patch.color !== undefined
            ? src
            : c[o + 1];
      c[o] = src;
      c[o + 1] = dst;
      this.markEdgeDirty(e);
    }
    if ('label' in patch) {
      this.edgeLabels[e] = patch.label;
      this.markEdgeDirty(e);
    }
    if ('meta' in patch) this.edgeMeta[e] = patch.meta;
    return true;
  }

  removeEdge(id: EdgeId): boolean {
    const e = this.edgeIdToIndex.get(id);
    if (e === undefined) return false;
    this.removeEdgeAt(e);
    return true;
  }

  setEdgeColor(index: number, sourceColor: ColorInput, targetColor?: ColorInput): void {
    this.assertEdgeIndex(index);
    const r = this.renderer;
    const c = r.getEdgeColors();
    const o = index * EDGE_COLOR_STRIDE;
    const src = parseColor(sourceColor, this.opts.defaultEdgeColor);
    c[o] = r.encodeColor(src);
    c[o + 1] = r.encodeColor(targetColor === undefined ? src : parseColor(targetColor, this.opts.defaultEdgeColor));
    this.markEdgeDirty(index);
  }

  // ===========================================================================
  // Edge vertex synchronization
  // ===========================================================================

  /** Tight contiguous pass copying node positions into every edge segment. */
  syncAllEdgeVertices(): void {
    const n = this._edgeCount;
    if (n === 0) return;
    const p = this.renderer.getNodeTransforms();
    const v = this.renderer.getEdgeVertices();
    const src = this._edgeSources;
    const tgt = this._edgeTargets;

    if (this._hiddenCount === 0) {
      for (let e = 0, o = 0; e < n; e++, o += EDGE_VERTEX_STRIDE) {
        const s = src[e] * NODE_TRANSFORM_STRIDE;
        const t = tgt[e] * NODE_TRANSFORM_STRIDE;
        v[o] = p[s];
        v[o + 1] = p[s + 1];
        v[o + 2] = p[s + 2];
        v[o + 3] = p[t];
        v[o + 4] = p[t + 1];
        v[o + 5] = p[t + 2];
      }
    } else {
      for (let e = 0; e < n; e++) this.writeEdgeVertices(e, p, v);
    }
    this.edgesAllDirty = true;
  }

  /** O(k): rewrites only the segments incident to `index` via the half-edge list. */
  syncNodeEdges(index: number): number {
    let h = this._nodeFirstHalfEdge[index];
    if (h === NO_INDEX) return 0;
    const p = this.renderer.getNodeTransforms();
    const v = this.renderer.getEdgeVertices();
    const next = this._halfEdgeNext;
    let k = 0;
    while (h !== NO_INDEX) {
      const e = h >> 1;
      this.writeEdgeVertices(e, p, v);
      this.markEdgeDirty(e);
      h = next[h];
      k++;
    }
    return k;
  }

  // ===========================================================================
  // Dirty tracking & upload
  // ===========================================================================

  markNodeDirty(index: number): void {
    if (index < this.nodeDirtyMin) this.nodeDirtyMin = index;
    if (index > this.nodeDirtyMax) this.nodeDirtyMax = index;
  }

  markAllNodesDirty(): void {
    this.nodesAllDirty = true;
  }

  markEdgeDirty(index: number): void {
    if (index < this.edgeDirtyMin) this.edgeDirtyMin = index;
    if (index > this.edgeDirtyMax) this.edgeDirtyMax = index;
  }

  markAllEdgesDirty(): void {
    this.edgesAllDirty = true;
  }

  flush(): void {
    if (this.nodesAllDirty) {
      this.renderer.commitNodes(this._nodeCount);
    } else if (this.nodeDirtyMax >= this.nodeDirtyMin) {
      const end = Math.min(this.nodeDirtyMax + 1, this._nodeCount);
      if (end > this.nodeDirtyMin) this.renderer.commitNodeRange(this.nodeDirtyMin, end - this.nodeDirtyMin);
    }
    this.nodesAllDirty = false;
    this.nodeDirtyMin = INT_MAX;
    this.nodeDirtyMax = -1;

    if (this.edgesAllDirty) {
      this.renderer.commitEdges(this._edgeCount);
    } else if (this.edgeDirtyMax >= this.edgeDirtyMin) {
      const end = Math.min(this.edgeDirtyMax + 1, this._edgeCount);
      if (end > this.edgeDirtyMin) this.renderer.commitEdgeRange(this.edgeDirtyMin, end - this.edgeDirtyMin);
    }
    this.edgesAllDirty = false;
    this.edgeDirtyMin = INT_MAX;
    this.edgeDirtyMax = -1;
  }

  // ===========================================================================
  // Dimension mode
  // ===========================================================================

  /**
   * `'2d'`: flattens Z, Vz and Fz to 0.
   * `'3d'`: if every visible node is co-planar on Z = 0, injects deterministic
   * Z jitter in `[-5, 5]` so repulsion can expand along Z.
   */
  setDimensionMode(mode: DimensionMode): boolean {
    if (mode === this._dimensionMode) return false;
    this._dimensionMode = mode;

    const n = this._nodeCount;
    const t = this.renderer.getNodeTransforms();
    const vel = this._nodeVelocities;
    const f = this._nodeForces;

    if (mode === '2d') {
      for (let i = 0, o = 2, v = 2; i < n; i++, o += NODE_TRANSFORM_STRIDE, v += VEC3_STRIDE) {
        t[o] = 0;
        vel[v] = 0;
        f[v] = 0;
      }
    } else {
      let coplanar = true;
      for (let o = 2, end = n * NODE_TRANSFORM_STRIDE; o < end; o += NODE_TRANSFORM_STRIDE) {
        if (Math.abs(t[o]) > COPLANAR_EPSILON) {
          coplanar = false;
          break;
        }
      }
      if (coplanar) {
        this.rng.reseed(this.opts.seed);
        const flags = this._nodeFlags;
        for (let i = 0, o = 2, v = 2; i < n; i++, o += NODE_TRANSFORM_STRIDE, v += VEC3_STRIDE) {
          // Draw for every node so the jitter of a node doesn't depend on its neighbors' pins.
          const jitter = this.rng.range(-Z_JITTER, Z_JITTER);
          if ((flags[i] & NodeFlag.PINNED) === 0) t[o] = jitter;
          vel[v] = 0;
        }
      }
    }

    this.nodesAllDirty = true;
    this.syncAllEdgeVertices();
    return true;
  }

  // ===========================================================================
  // Internals: insertion
  // ===========================================================================

  private insertNode(input: NodeInput<TNodeMeta>): number {
    const id = input.id;
    if (id === undefined || id === null) throw new TypeError('GraphStore: node id is required');
    if (this.nodeIdToIndex.has(id)) throw new Error(`GraphStore: duplicate node id "${String(id)}"`);

    this.ensureNodeCapacity(this._nodeCount + 1);
    const i = this._nodeCount++;

    const opts = this.opts;
    const radius = positiveOr(input.radius, opts.defaultNodeRadius);
    const hidden = input.hidden === true;
    let flags: number = NodeFlag.NONE;
    if (input.pinned === true) flags |= NodeFlag.PINNED;
    if (hidden) flags |= NodeFlag.HIDDEN;

    const t = this.renderer.getNodeTransforms();
    const o = i * NODE_TRANSFORM_STRIDE;
    if (isFiniteNumber(input.x) || isFiniteNumber(input.y) || isFiniteNumber(input.z)) {
      t[o] = finiteOr(input.x, 0);
      t[o + 1] = finiteOr(input.y, 0);
      t[o + 2] = this._dimensionMode === '2d' ? 0 : finiteOr(input.z, 0);
    } else {
      this.writeInitialPosition(i, t, o);
    }
    t[o + 3] = hidden ? 0 : radius;

    const v = i * VEC3_STRIDE;
    this._nodeVelocities[v] = 0;
    this._nodeVelocities[v + 1] = 0;
    this._nodeVelocities[v + 2] = 0;
    this._nodeForces[v] = 0;
    this._nodeForces[v + 1] = 0;
    this._nodeForces[v + 2] = 0;
    this._nodeMasses[i] = finiteOr(input.mass, opts.defaultNodeMass);
    this._nodeRadii[i] = radius;
    this._nodeDegrees[i] = 0;
    this._nodeFlags[i] = flags;
    this._nodeBaseColors[i] = parseColor(input.color, opts.defaultNodeColor);
    this._nodeFirstHalfEdge[i] = NO_INDEX;
    this.refreshNodeColor(i);

    this.nodeIdToIndex.set(id, i);
    this.nodeIds.push(id);
    this.nodeMeta.push(input.meta);
    this.nodeLabels.push(input.label);
    if (hidden) this._hiddenCount++;
    return i;
  }

  private insertEdge(input: EdgeInput<TEdgeMeta>): number {
    const s = this.nodeIdToIndex.get(input.source);
    if (s === undefined) throw new Error(`GraphStore: edge source "${String(input.source)}" not found`);
    const t = this.nodeIdToIndex.get(input.target);
    if (t === undefined) throw new Error(`GraphStore: edge target "${String(input.target)}" not found`);

    let id = input.id;
    if (id === undefined) {
      do id = `__e${this.edgeAutoId++}`;
      while (this.edgeIdToIndex.has(id));
    } else if (this.edgeIdToIndex.has(id)) {
      throw new Error(`GraphStore: duplicate edge id "${String(id)}"`);
    }

    this.ensureEdgeCapacity(this._edgeCount + 1);
    const e = this._edgeCount++;

    this._edgeSources[e] = s;
    this._edgeTargets[e] = t;
    this._edgeDistances[e] = positiveOr(input.distance, this.opts.defaultLinkDistance);
    this._edgeStrengths[e] = input.strength === undefined ? EDGE_STRENGTH_AUTO : clampStrength(input.strength);

    this.linkHalfEdge(e << 1, s);
    this.linkHalfEdge((e << 1) | 1, t);
    this._nodeDegrees[s]++;
    this._nodeDegrees[t]++;

    const r = this.renderer;
    const c = r.getEdgeColors();
    const co = e * EDGE_COLOR_STRIDE;
    const srcColor = parseColor(input.color, this.opts.defaultEdgeColor);
    c[co] = r.encodeColor(srcColor);
    c[co + 1] = r.encodeColor(input.targetColor === undefined ? srcColor : parseColor(input.targetColor, srcColor));

    this.edgeIdToIndex.set(id, e);
    this.edgeIds.push(id);
    this.edgeMeta.push(input.meta);
    this.edgeLabels.push(input.label);
    return e;
  }

  /** Deterministic phyllotaxis (2D) / spherical spiral (3D) seed layout. */
  private writeInitialPosition(i: number, t: Float32Array, o: number): void {
    const R = this.opts.initialLayoutRadius;
    if (this._dimensionMode === '2d') {
      const r = R * Math.sqrt(0.5 + i);
      const a = i * ANGLE_ROLL;
      t[o] = r * Math.cos(a);
      t[o + 1] = r * Math.sin(a);
      t[o + 2] = 0;
    } else {
      const r = R * Math.cbrt(0.5 + i);
      const roll = i * ANGLE_ROLL;
      const yaw = i * ANGLE_YAW;
      const sr = Math.sin(roll);
      t[o] = r * sr * Math.cos(yaw);
      t[o + 1] = r * Math.cos(roll);
      t[o + 2] = r * sr * Math.sin(yaw);
    }
  }

  // ===========================================================================
  // Internals: removal (swap-and-pop)
  // ===========================================================================

  private removeEdgeAt(e: number): void {
    const s = this._edgeSources[e];
    const t = this._edgeTargets[e];
    this.unlinkHalfEdge(e << 1, s);
    this.unlinkHalfEdge((e << 1) | 1, t);
    this._nodeDegrees[s]--;
    this._nodeDegrees[t]--;

    const id = this.edgeIds[e];
    const last = this._edgeCount - 1;
    if (e !== last) this.moveEdge(last, e);

    this.edgeIdToIndex.delete(id);
    this.edgeIds.pop();
    this.edgeMeta.pop();
    this.edgeLabels.pop();
    this._edgeCount = last;
    this._topologyVersion++;
    this.edgesAllDirty = true;
  }

  /** Relocates node `from` into slot `to`, fixing every reference to it. */
  private moveNode(from: number, to: number): void {
    const t = this.renderer.getNodeTransforms();
    const fo = from * NODE_TRANSFORM_STRIDE;
    const toff = to * NODE_TRANSFORM_STRIDE;
    t[toff] = t[fo];
    t[toff + 1] = t[fo + 1];
    t[toff + 2] = t[fo + 2];
    t[toff + 3] = t[fo + 3];
    const colors = this.renderer.getNodeColors();
    colors[to] = colors[from];

    const fv = from * VEC3_STRIDE;
    const tv = to * VEC3_STRIDE;
    const vel = this._nodeVelocities;
    const frc = this._nodeForces;
    vel[tv] = vel[fv];
    vel[tv + 1] = vel[fv + 1];
    vel[tv + 2] = vel[fv + 2];
    frc[tv] = frc[fv];
    frc[tv + 1] = frc[fv + 1];
    frc[tv + 2] = frc[fv + 2];
    this._nodeMasses[to] = this._nodeMasses[from];
    this._nodeRadii[to] = this._nodeRadii[from];
    this._nodeDegrees[to] = this._nodeDegrees[from];
    this._nodeFlags[to] = this._nodeFlags[from];
    this._nodeBaseColors[to] = this._nodeBaseColors[from];

    const head = this._nodeFirstHalfEdge[from];
    this._nodeFirstHalfEdge[to] = head;
    this._nodeFirstHalfEdge[from] = NO_INDEX;

    // Re-point edge endpoints (O(k)); self-loops have both halves in this list.
    const next = this._halfEdgeNext;
    for (let h = head; h !== NO_INDEX; h = next[h]) {
      if ((h & 1) === 0) this._edgeSources[h >> 1] = to;
      else this._edgeTargets[h >> 1] = to;
    }

    const id = this.nodeIds[from];
    this.nodeIds[to] = id;
    this.nodeMeta[to] = this.nodeMeta[from];
    this.nodeLabels[to] = this.nodeLabels[from];
    this.nodeIdToIndex.set(id, to);
  }

  /** Relocates edge `from` into slot `to`, relinking both of its half-edges. */
  private moveEdge(from: number, to: number): void {
    const s = this._edgeSources[from];
    const t = this._edgeTargets[from];
    this._edgeSources[to] = s;
    this._edgeTargets[to] = t;
    this._edgeDistances[to] = this._edgeDistances[from];
    this._edgeStrengths[to] = this._edgeStrengths[from];

    const v = this.renderer.getEdgeVertices();
    const fo = from * EDGE_VERTEX_STRIDE;
    const toff = to * EDGE_VERTEX_STRIDE;
    for (let k = 0; k < EDGE_VERTEX_STRIDE; k++) v[toff + k] = v[fo + k];
    const c = this.renderer.getEdgeColors();
    c[to * EDGE_COLOR_STRIDE] = c[from * EDGE_COLOR_STRIDE];
    c[to * EDGE_COLOR_STRIDE + 1] = c[from * EDGE_COLOR_STRIDE + 1];

    // Order matters for self-loops whose two halves are adjacent: the second
    // relink observes the first one's updated prev pointer.
    this.relocateHalfEdge(from << 1, to << 1, s);
    this.relocateHalfEdge((from << 1) | 1, (to << 1) | 1, t);

    const id = this.edgeIds[from];
    this.edgeIds[to] = id;
    this.edgeMeta[to] = this.edgeMeta[from];
    this.edgeLabels[to] = this.edgeLabels[from];
    this.edgeIdToIndex.set(id, to);
  }

  // ===========================================================================
  // Internals: intrusive half-edge list
  // ===========================================================================

  private linkHalfEdge(h: number, node: number): void {
    const head = this._nodeFirstHalfEdge[node];
    this._halfEdgeNext[h] = head;
    this._halfEdgePrev[h] = NO_INDEX;
    if (head !== NO_INDEX) this._halfEdgePrev[head] = h;
    this._nodeFirstHalfEdge[node] = h;
  }

  private unlinkHalfEdge(h: number, node: number): void {
    const next = this._halfEdgeNext[h];
    const prev = this._halfEdgePrev[h];
    if (prev !== NO_INDEX) this._halfEdgeNext[prev] = next;
    else this._nodeFirstHalfEdge[node] = next;
    if (next !== NO_INDEX) this._halfEdgePrev[next] = prev;
    this._halfEdgeNext[h] = NO_INDEX;
    this._halfEdgePrev[h] = NO_INDEX;
  }

  private relocateHalfEdge(from: number, to: number, node: number): void {
    const next = this._halfEdgeNext[from];
    const prev = this._halfEdgePrev[from];
    this._halfEdgeNext[to] = next;
    this._halfEdgePrev[to] = prev;
    if (prev !== NO_INDEX) this._halfEdgeNext[prev] = to;
    else this._nodeFirstHalfEdge[node] = to;
    if (next !== NO_INDEX) this._halfEdgePrev[next] = to;
    this._halfEdgeNext[from] = NO_INDEX;
    this._halfEdgePrev[from] = NO_INDEX;
  }

  // ===========================================================================
  // Internals: rendering state
  // ===========================================================================

  private writeEdgeVertices(e: number, p: Float32Array, v: Float32Array): void {
    const s = this._edgeSources[e];
    const tg = this._edgeTargets[e];
    const so = s * NODE_TRANSFORM_STRIDE;
    const o = e * EDGE_VERTEX_STRIDE;
    const flags = this._nodeFlags;
    // Collapse segments with a hidden endpoint to a zero-length point.
    const to =
      ((flags[s] | flags[tg]) & NodeFlag.HIDDEN) !== 0 ? so : tg * NODE_TRANSFORM_STRIDE;
    v[o] = p[so];
    v[o + 1] = p[so + 1];
    v[o + 2] = p[so + 2];
    v[o + 3] = p[to];
    v[o + 4] = p[to + 1];
    v[o + 5] = p[to + 2];
  }

  private refreshNodeColor(i: number): void {
    const f = this._nodeFlags[i];
    const color =
      (f & NodeFlag.SELECTED) !== 0
        ? this.opts.selectedColor
        : (f & NodeFlag.HOVERED) !== 0
          ? this.opts.hoverColor
          : this._nodeBaseColors[i];
    this.renderer.getNodeColors()[i] = this.renderer.encodeColor(color);
    this.markNodeDirty(i);
  }

  private applyVisibility(i: number, hidden: boolean): void {
    this._hiddenCount += hidden ? 1 : -1;
    this.renderer.getNodeTransforms()[i * NODE_TRANSFORM_STRIDE + 3] = hidden ? 0 : this._nodeRadii[i];
    this.markNodeDirty(i);
    this.syncNodeEdges(i);
  }

  // ===========================================================================
  // Internals: capacity
  // ===========================================================================

  private ensureNodeCapacity(required: number): void {
    if (required > this._nodeCapacity) {
      const cap = nextCapacity(this._nodeCapacity, required);
      this._nodeVelocities = growFloat32(this._nodeVelocities, cap * VEC3_STRIDE);
      this._nodeForces = growFloat32(this._nodeForces, cap * VEC3_STRIDE);
      this._nodeMasses = growFloat32(this._nodeMasses, cap);
      this._nodeRadii = growFloat32(this._nodeRadii, cap);
      this._nodeDegrees = growUint32(this._nodeDegrees, cap);
      this._nodeFlags = growUint8(this._nodeFlags, cap);
      this._nodeBaseColors = growUint32(this._nodeBaseColors, cap);
      this._nodeFirstHalfEdge = growInt32(this._nodeFirstHalfEdge, cap, NO_INDEX);
      this._nodeCapacity = cap;
    }
    if (this.renderer.ensureNodeCapacity(this._nodeCapacity)) this.nodesAllDirty = true;
  }

  private ensureEdgeCapacity(required: number): void {
    if (required > this._edgeCapacity) {
      const cap = nextCapacity(this._edgeCapacity, required);
      this._edgeSources = growUint32(this._edgeSources, cap);
      this._edgeTargets = growUint32(this._edgeTargets, cap);
      this._edgeDistances = growFloat32(this._edgeDistances, cap);
      this._edgeStrengths = growFloat32(this._edgeStrengths, cap);
      this._halfEdgeNext = growInt32(this._halfEdgeNext, cap * 2, NO_INDEX);
      this._halfEdgePrev = growInt32(this._halfEdgePrev, cap * 2, NO_INDEX);
      this._edgeCapacity = cap;
    }
    if (this.renderer.ensureEdgeCapacity(this._edgeCapacity)) this.edgesAllDirty = true;
  }

  private assertNodeIndex(index: number): void {
    if (!(index >= 0 && index < this._nodeCount)) {
      throw new RangeError(`GraphStore: node index ${index} out of range [0, ${this._nodeCount})`);
    }
  }

  private assertEdgeIndex(index: number): void {
    if (!(index >= 0 && index < this._edgeCount)) {
      throw new RangeError(`GraphStore: edge index ${index} out of range [0, ${this._edgeCount})`);
    }
  }
}

function isFiniteNumber(v: number | undefined): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

function finiteOr(v: number | undefined, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

function positiveOr(v: number | undefined, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : fallback;
}

function clampStrength(v: number): number {
  if (!Number.isFinite(v)) return EDGE_STRENGTH_AUTO;
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
