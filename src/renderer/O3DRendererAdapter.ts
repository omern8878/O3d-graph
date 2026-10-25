import {
  EDGE_COLOR_STRIDE,
  EDGE_VERTEX_STRIDE,
  NODE_TRANSFORM_STRIDE,
  type ColorLayout,
  type ICameraAdapter,
  type IGraphRenderer,
  type IO3DEngineAdapter,
} from '../types/index.js';
import { nextCapacity } from '../utils/capacity.js';
import { encodeColorLayout } from '../utils/color.js';

export { packRGBA, packRGBAf, parseColor, withAlpha } from '../utils/color.js';

// ---------------------------------------------------------------------------
// Raw @omernach/o3d bindings → IO3DEngineAdapter
// ---------------------------------------------------------------------------

/**
 * Structural type of the wasm-bindgen exports of `@omernach/o3d` that this
 * package relies on. Pass the imported module namespace directly:
 *
 * ```ts
 * import * as o3d from '@omernach/o3d';
 * const engine = new O3DEngineAdapter(o3d);
 * ```
 */
export interface O3DWasmBindings {
  get_wasm_memory(): WebAssembly.Memory;

  create_instanced_mesh(geometry_id: number, max_instances: number): number;
  get_instance_transforms_ptr(mesh_handle: number): number;
  get_instance_colors_ptr(mesh_handle: number): number;
  update_instanced_mesh(mesh_handle: number, active_count: number): void;
  update_instance_sub_range(mesh_handle: number, start_idx: number, count: number): void;
  destroy_instanced_mesh?(mesh_handle: number): void;

  create_line_buffer(max_segments: number): number;
  get_line_vertices_ptr(line_handle: number): number;
  get_line_colors_ptr(line_handle: number): number;
  update_line_buffer(line_handle: number, active_segments: number): void;
  update_line_sub_range(line_handle: number, start_segment: number, segment_count: number): void;
  destroy_line_buffer?(line_handle: number): void;

  raycast_spheres_ndc(ndc_x: number, ndc_y: number, spheres_ptr: number, count: number): number;
  unproject_ndc_ray(ndc_x: number, ndc_y: number): Float32Array;
  get_camera_forward(): Float32Array;
  get_camera_view_proj_matrix(): Float32Array;
  set_camera_orbit_enabled(enabled: boolean): void;
  set_camera_2d_mode(is_2d: boolean): void;
}

const REQUIRED_BINDINGS: ReadonlyArray<keyof O3DWasmBindings> = [
  'get_wasm_memory',
  'create_instanced_mesh',
  'get_instance_transforms_ptr',
  'get_instance_colors_ptr',
  'update_instanced_mesh',
  'update_instance_sub_range',
  'create_line_buffer',
  'get_line_vertices_ptr',
  'get_line_colors_ptr',
  'update_line_buffer',
  'update_line_sub_range',
  'raycast_spheres_ndc',
  'unproject_ndc_ray',
  'get_camera_forward',
  'get_camera_view_proj_matrix',
  'set_camera_orbit_enabled',
  'set_camera_2d_mode',
];

/** Thin camel-case adapter over the raw `@omernach/o3d` exports. */
export class O3DEngineAdapter implements IO3DEngineAdapter {
  private readonly b: O3DWasmBindings;
  private readonly memory: WebAssembly.Memory;

  constructor(bindings: O3DWasmBindings) {
    const missing = REQUIRED_BINDINGS.filter(
      (name) => typeof (bindings as unknown as Record<string, unknown>)[name] !== 'function',
    );
    if (missing.length > 0) {
      throw new TypeError(`O3DEngineAdapter: @omernach/o3d is missing exports: ${missing.join(', ')}`);
    }
    this.b = bindings;
    this.memory = bindings.get_wasm_memory();
  }

  getWasmMemory(): WebAssembly.Memory {
    return this.memory;
  }

  createInstancedMesh(geometryId: number, maxInstances: number): number {
    return this.b.create_instanced_mesh(geometryId, maxInstances);
  }
  getInstanceTransformsPtr(meshHandle: number): number {
    return this.b.get_instance_transforms_ptr(meshHandle);
  }
  getInstanceColorsPtr(meshHandle: number): number {
    return this.b.get_instance_colors_ptr(meshHandle);
  }
  updateInstancedMesh(meshHandle: number, activeCount: number): void {
    this.b.update_instanced_mesh(meshHandle, activeCount);
  }
  updateInstanceSubRange(meshHandle: number, startIdx: number, count: number): void {
    this.b.update_instance_sub_range(meshHandle, startIdx, count);
  }
  destroyInstancedMesh(meshHandle: number): void {
    this.b.destroy_instanced_mesh?.(meshHandle);
  }

  createLineBuffer(maxSegments: number): number {
    return this.b.create_line_buffer(maxSegments);
  }
  getLineVerticesPtr(lineHandle: number): number {
    return this.b.get_line_vertices_ptr(lineHandle);
  }
  getLineColorsPtr(lineHandle: number): number {
    return this.b.get_line_colors_ptr(lineHandle);
  }
  updateLineBuffer(lineHandle: number, activeSegments: number): void {
    this.b.update_line_buffer(lineHandle, activeSegments);
  }
  updateLineSubRange(lineHandle: number, startSegment: number, segmentCount: number): void {
    this.b.update_line_sub_range(lineHandle, startSegment, segmentCount);
  }
  destroyLineBuffer(lineHandle: number): void {
    this.b.destroy_line_buffer?.(lineHandle);
  }

  raycastSpheresNdc(ndcX: number, ndcY: number, spheresPtr: number, count: number): number {
    return this.b.raycast_spheres_ndc(ndcX, ndcY, spheresPtr, count);
  }
  unprojectNdcRay(ndcX: number, ndcY: number): Float32Array {
    return this.b.unproject_ndc_ray(ndcX, ndcY);
  }
  getCameraForward(): Float32Array {
    return this.b.get_camera_forward();
  }
  getCameraViewProjMatrix(): Float32Array {
    return this.b.get_camera_view_proj_matrix();
  }
  setCameraOrbitEnabled(enabled: boolean): void {
    this.b.set_camera_orbit_enabled(enabled);
  }
  setCamera2DMode(is2D: boolean): void {
    this.b.set_camera_2d_mode(is2D);
  }
}

// ---------------------------------------------------------------------------
// Graph renderer: handle management + zero-copy views
// ---------------------------------------------------------------------------

export interface O3DRendererOptions {
  /** o3d geometry for node instances. Default `0` (sphere). */
  nodeGeometryId?: number;
  initialNodeCapacity?: number;
  initialEdgeCapacity?: number;
  /** Packed color byte order the engine expects. Default `'rgba'` (`0xRRGGBBAA`). */
  colorLayout?: ColorLayout;
}

const EMPTY_F32 = new Float32Array(0);
const EMPTY_U32 = new Uint32Array(0);
const F32_BYTES = 4;
const U32_BYTES = 4;

/**
 * Owns the node `InstancedMesh` and edge `DynamicLineBuffer` and exposes
 * TypedArray views that alias WASM linear memory directly.
 *
 * Detachment safety: every `get*` call compares the live `memory.buffer`
 * against the one the views were built on (identity + byteLength + detached
 * flag). After `memory.grow()` — triggered by any allocation inside the
 * engine, including `unproject_ndc_ray` — the old ArrayBuffer is detached and
 * the views are rebuilt from fresh pointers. The check is a couple of property
 * reads, so hot loops should fetch a view once per pass.
 *
 * Capacity growth allocates a larger mesh/line buffer, memcpy's the old
 * contents across inside WASM memory, and retires the old handle.
 */
export class O3DRendererAdapter implements IGraphRenderer {
  readonly engine: IO3DEngineAdapter;

  private readonly memory: WebAssembly.Memory;
  private readonly geometryId: number;
  private readonly colorLayout: ColorLayout;

  private meshHandle: number;
  private lineHandle: number;
  private _nodeCapacity: number;
  private _edgeCapacity: number;

  private boundBuffer: ArrayBufferLike | null = null;
  private boundByteLength = -1;
  private _viewVersion = 0;

  private transformsPtr = 0;
  private transformsView: Float32Array = EMPTY_F32;
  private nodeColorsView: Uint32Array = EMPTY_U32;
  private edgeVerticesView: Float32Array = EMPTY_F32;
  private edgeColorsView: Uint32Array = EMPTY_U32;

  private disposed = false;

  constructor(engine: IO3DEngineAdapter, options: O3DRendererOptions = {}) {
    this.engine = engine;
    this.memory = engine.getWasmMemory();
    this.geometryId = options.nodeGeometryId ?? 0;
    this.colorLayout = options.colorLayout ?? 'rgba';

    this._nodeCapacity = nextCapacity(0, Math.max(1, options.initialNodeCapacity ?? 1024));
    this._edgeCapacity = nextCapacity(0, Math.max(1, options.initialEdgeCapacity ?? 2048));

    this.meshHandle = engine.createInstancedMesh(this.geometryId, this._nodeCapacity);
    this.lineHandle = engine.createLineBuffer(this._edgeCapacity);

    this.rebindViews();
    engine.updateInstancedMesh(this.meshHandle, 0);
    engine.updateLineBuffer(this.lineHandle, 0);
  }

  get camera(): ICameraAdapter {
    return this.engine;
  }
  get nodeCapacity(): number {
    return this._nodeCapacity;
  }
  get edgeCapacity(): number {
    return this._edgeCapacity;
  }
  get viewVersion(): number {
    return this._viewVersion;
  }
  get nodeMeshHandle(): number {
    return this.meshHandle;
  }
  get edgeLineHandle(): number {
    return this.lineHandle;
  }

  // ---- Zero-copy views ------------------------------------------------------

  getNodeTransforms(): Float32Array {
    this.ensureViews();
    return this.transformsView;
  }

  getNodeColors(): Uint32Array {
    this.ensureViews();
    return this.nodeColorsView;
  }

  getEdgeVertices(): Float32Array {
    this.ensureViews();
    return this.edgeVerticesView;
  }

  getEdgeColors(): Uint32Array {
    this.ensureViews();
    return this.edgeColorsView;
  }

  getNodeTransformsPtr(): number {
    this.ensureViews();
    return this.transformsPtr;
  }

  // ---- Capacity --------------------------------------------------------------

  ensureNodeCapacity(required: number): boolean {
    this.assertAlive();
    if (required <= this._nodeCapacity) return false;

    const oldHandle = this.meshHandle;
    const oldCap = this._nodeCapacity;
    const newCap = nextCapacity(oldCap, required);
    const newHandle = this.engine.createInstancedMesh(this.geometryId, newCap);

    // Creating the mesh may have grown memory: build every view on the live buffer.
    const buf = this.memory.buffer;
    const oldT = f32View(buf, this.engine.getInstanceTransformsPtr(oldHandle), oldCap * NODE_TRANSFORM_STRIDE);
    const newT = f32View(buf, this.engine.getInstanceTransformsPtr(newHandle), newCap * NODE_TRANSFORM_STRIDE);
    newT.set(oldT);
    const oldC = u32View(buf, this.engine.getInstanceColorsPtr(oldHandle), oldCap);
    const newC = u32View(buf, this.engine.getInstanceColorsPtr(newHandle), newCap);
    newC.set(oldC);

    this.engine.updateInstancedMesh(oldHandle, 0);
    this.engine.destroyInstancedMesh?.(oldHandle);

    this.meshHandle = newHandle;
    this._nodeCapacity = newCap;
    this.rebindViews();
    return true;
  }

  ensureEdgeCapacity(required: number): boolean {
    this.assertAlive();
    if (required <= this._edgeCapacity) return false;

    const oldHandle = this.lineHandle;
    const oldCap = this._edgeCapacity;
    const newCap = nextCapacity(oldCap, required);
    const newHandle = this.engine.createLineBuffer(newCap);

    const buf = this.memory.buffer;
    const oldV = f32View(buf, this.engine.getLineVerticesPtr(oldHandle), oldCap * EDGE_VERTEX_STRIDE);
    const newV = f32View(buf, this.engine.getLineVerticesPtr(newHandle), newCap * EDGE_VERTEX_STRIDE);
    newV.set(oldV);
    const oldC = u32View(buf, this.engine.getLineColorsPtr(oldHandle), oldCap * EDGE_COLOR_STRIDE);
    const newC = u32View(buf, this.engine.getLineColorsPtr(newHandle), newCap * EDGE_COLOR_STRIDE);
    newC.set(oldC);

    this.engine.updateLineBuffer(oldHandle, 0);
    this.engine.destroyLineBuffer?.(oldHandle);

    this.lineHandle = newHandle;
    this._edgeCapacity = newCap;
    this.rebindViews();
    return true;
  }

  // ---- Upload ---------------------------------------------------------------

  commitNodes(activeCount: number): void {
    this.assertAlive();
    this.engine.updateInstancedMesh(this.meshHandle, clampCount(activeCount, this._nodeCapacity));
  }

  commitNodeRange(start: number, count: number): void {
    this.assertAlive();
    const s = start < 0 ? 0 : start;
    const n = Math.min(count, this._nodeCapacity - s);
    if (n > 0) this.engine.updateInstanceSubRange(this.meshHandle, s, n);
  }

  commitEdges(activeCount: number): void {
    this.assertAlive();
    this.engine.updateLineBuffer(this.lineHandle, clampCount(activeCount, this._edgeCapacity));
  }

  commitEdgeRange(start: number, count: number): void {
    this.assertAlive();
    const s = start < 0 ? 0 : start;
    const n = Math.min(count, this._edgeCapacity - s);
    if (n > 0) this.engine.updateLineSubRange(this.lineHandle, s, n);
  }

  // ---- Picking ----------------------------------------------------------------

  raycastNodes(ndcX: number, ndcY: number, count: number): number {
    this.assertAlive();
    const n = clampCount(count, this._nodeCapacity);
    if (n === 0) return -1;
    const ptr = this.getNodeTransformsPtr();
    const hit = this.engine.raycastSpheresNdc(ndcX, ndcY, ptr, n);
    return hit >= 0 && hit < n ? hit : -1;
  }

  encodeColor(rgba: number): number {
    return encodeColorLayout(rgba, this.colorLayout);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.engine.updateInstancedMesh(this.meshHandle, 0);
    this.engine.updateLineBuffer(this.lineHandle, 0);
    this.engine.destroyInstancedMesh?.(this.meshHandle);
    this.engine.destroyLineBuffer?.(this.lineHandle);
    this.transformsView = EMPTY_F32;
    this.nodeColorsView = EMPTY_U32;
    this.edgeVerticesView = EMPTY_F32;
    this.edgeColorsView = EMPTY_U32;
    this.boundBuffer = null;
  }

  // ---- Internals ------------------------------------------------------------

  private ensureViews(): void {
    const buf = this.memory.buffer;
    if (buf === this.boundBuffer && buf.byteLength === this.boundByteLength && !isDetached(buf)) return;
    this.assertAlive();
    this.rebindViews();
  }

  private rebindViews(): void {
    const buf = this.memory.buffer;
    this.transformsPtr = this.engine.getInstanceTransformsPtr(this.meshHandle);
    this.transformsView = f32View(buf, this.transformsPtr, this._nodeCapacity * NODE_TRANSFORM_STRIDE);
    this.nodeColorsView = u32View(buf, this.engine.getInstanceColorsPtr(this.meshHandle), this._nodeCapacity);
    this.edgeVerticesView = f32View(
      buf,
      this.engine.getLineVerticesPtr(this.lineHandle),
      this._edgeCapacity * EDGE_VERTEX_STRIDE,
    );
    this.edgeColorsView = u32View(
      buf,
      this.engine.getLineColorsPtr(this.lineHandle),
      this._edgeCapacity * EDGE_COLOR_STRIDE,
    );
    this.boundBuffer = buf;
    this.boundByteLength = buf.byteLength;
    this._viewVersion++;
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error('O3DRendererAdapter: used after dispose()');
  }
}

function f32View(buf: ArrayBufferLike, ptr: number, length: number): Float32Array {
  checkRegion(buf, ptr, length * F32_BYTES, 'f32');
  return new Float32Array(buf, ptr, length);
}

function u32View(buf: ArrayBufferLike, ptr: number, length: number): Uint32Array {
  checkRegion(buf, ptr, length * U32_BYTES, 'u32');
  return new Uint32Array(buf, ptr, length);
}

function checkRegion(buf: ArrayBufferLike, ptr: number, bytes: number, kind: string): void {
  if (!Number.isInteger(ptr) || ptr <= 0) {
    throw new RangeError(`O3DRendererAdapter: invalid ${kind} pointer ${ptr}`);
  }
  if ((ptr & 3) !== 0) {
    throw new RangeError(`O3DRendererAdapter: ${kind} pointer ${ptr} is not 4-byte aligned`);
  }
  if (ptr + bytes > buf.byteLength) {
    throw new RangeError(
      `O3DRendererAdapter: ${kind} region [${ptr}, ${ptr + bytes}) exceeds WASM memory (${buf.byteLength} bytes)`,
    );
  }
}

function isDetached(buf: ArrayBufferLike): boolean {
  // `detached` is ES2024; fall back to the zero-length heuristic on older engines.
  const d = (buf as { detached?: boolean }).detached;
  return d === true || (d === undefined && buf.byteLength === 0);
}

function clampCount(count: number, capacity: number): number {
  return count <= 0 ? 0 : count > capacity ? capacity : count | 0;
}
