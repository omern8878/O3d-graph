import type { ICameraAdapter, IGraphRenderer } from '../types/index.js';
import { nextCapacity } from '../utils/capacity.js';
import { encodeColorLayout } from '../utils/color.js';

// ---------------------------------------------------------------------------
// Structural subset of @omernach/o3d this renderer relies on. Declared here
// (instead of imported) so the graph library builds and tests without the
// engine; the real classes satisfy these shapes as-is.
// ---------------------------------------------------------------------------

/** `GpuMesh`: only its bounding radius matters (picking radius per unit scale). */
export interface O3DGpuMeshLike {
  readonly boundingRadius: number;
}

type InstanceChannel = 'transform' | 'rotation' | 'color';

/** `InstancedMesh`: zero-copy `[x, y, z, scale]` transforms + packed colors in WASM memory. */
export interface O3DInstancedMeshLike {
  readonly capacity: number;
  getCount(): number;
  setCount(count: number): number;
  getTransforms(): Float32Array;
  getColors(): Uint32Array;
  markDirty(first: number, last?: number, channels?: readonly InstanceChannel[]): void;
  raycastNdc(camera: unknown, ndcX: number, ndcY: number): number;
  destroy(): void;
}

/** `LineBatch`: zero-copy `[x1, y1, z1, x2, y2, z2]` segments + `[start, end]` colors. */
export interface O3DLineBatchLike {
  readonly capacity: number;
  getCount(): number;
  setCount(count: number): number;
  getVertices(): Float32Array;
  getColors(): Uint32Array;
  markDirty(first: number, last?: number): void;
  destroy(): void;
}

/** `CameraController`. */
export interface O3DCameraControllerLike {
  getForward(): ArrayLike<number>;
  unprojectNdc(ndcX: number, ndcY: number): { origin: ArrayLike<number>; direction: ArrayLike<number> } | null;
  getViewProjection(): ArrayLike<number>;
  setOrbitEnabled(enabled: boolean): void;
  setMode(mode: '2d' | '3d'): void;
  lockToXYPlane(locked: boolean): void;
}

export interface O3DBatchRendererOptions<
  TDevice,
  TMesh extends O3DGpuMeshLike,
  TInstanced extends O3DInstancedMeshLike,
  TLines extends O3DLineBatchLike,
> {
  /** The batch classes from `@omernach/o3d` (pass `{ InstancedMesh, LineBatch }`). */
  o3d: {
    InstancedMesh: new (device: TDevice, mesh: TMesh, capacity: number, options?: { shading?: 'lit' | 'flat' }) => TInstanced;
    LineBatch: new (device: TDevice, capacity: number) => TLines;
  };
  device: TDevice;
  /** Node geometry, e.g. `GpuMesh.sphere(device)`. Use a unit bounding radius so scale = radius. */
  nodeMesh: TMesh;
  camera: O3DCameraControllerLike;
  /** Default `'lit'`. Use `'flat'` with disc geometry for a pure 2D look. */
  shading?: 'lit' | 'flat';
  initialNodeCapacity?: number;
  initialEdgeCapacity?: number;
  /**
   * Called whenever buffers, counts, the camera mode or the batch objects
   * change — schedule a `BatchRenderer.render(camera, renderer.scene)` here.
   */
  onChange?: () => void;
}

/** Channels the graph ever writes; rotations stay identity and are never re-uploaded. */
const GRAPH_CHANNELS: readonly InstanceChannel[] = ['transform', 'color'];

/**
 * `IGraphRenderer` over the `@omernach/o3d` WebGPU batch renderer.
 *
 * Nodes are one `InstancedMesh` and edges one `LineBatch`; the graph writes
 * positions and colors straight into their WASM views, and `commit*` calls
 * become o3d dirty ranges, so each frame uploads only what changed. o3d's
 * `WasmArrayView`s re-bind themselves after WASM memory growth.
 *
 * o3d batches have a fixed capacity: growing allocates a larger batch, copies
 * the old contents inside WASM memory, destroys the old one and replaces it
 * in `scene` (always render `renderer.scene`, never a batch you cached).
 *
 * Colors are written in o3d's layout (red in the lowest byte).
 */
export class O3DBatchRenderer<
  TDevice = unknown,
  TMesh extends O3DGpuMeshLike = O3DGpuMeshLike,
  TInstanced extends O3DInstancedMeshLike = O3DInstancedMeshLike,
  TLines extends O3DLineBatchLike = O3DLineBatchLike,
> implements IGraphRenderer
{
  readonly camera: ICameraAdapter;
  /** Pass to `BatchRenderer.render(camera, renderer.scene)`. Contents change on capacity growth. */
  readonly scene: { meshes: TInstanced[]; lines: TLines[] };

  private readonly o3d: O3DBatchRendererOptions<TDevice, TMesh, TInstanced, TLines>['o3d'];
  private readonly device: TDevice;
  private readonly nodeMesh: TMesh;
  private readonly cameraController: O3DCameraControllerLike;
  private readonly shading: 'lit' | 'flat';
  private readonly onChange: () => void;

  private nodes: TInstanced;
  private edges: TLines;
  private _viewVersion = 0;
  private lastTransformsBuffer: ArrayBufferLike | null = null;
  private disposed = false;

  constructor(options: O3DBatchRendererOptions<TDevice, TMesh, TInstanced, TLines>) {
    this.o3d = options.o3d;
    this.device = options.device;
    this.nodeMesh = options.nodeMesh;
    this.cameraController = options.camera;
    this.shading = options.shading ?? 'lit';
    this.onChange = options.onChange ?? noop;

    this.nodes = this.createNodes(nextCapacity(0, Math.max(1, options.initialNodeCapacity ?? 1024)));
    this.edges = new this.o3d.LineBatch(this.device, nextCapacity(0, Math.max(1, options.initialEdgeCapacity ?? 2048)));
    this.scene = { meshes: [this.nodes], lines: [this.edges] };
    this.camera = new CameraAdapter(options.camera, this.onChange);
  }

  get nodeCapacity(): number {
    return this.nodes.capacity;
  }
  get edgeCapacity(): number {
    return this.edges.capacity;
  }
  get viewVersion(): number {
    return this._viewVersion;
  }

  // ---- Zero-copy views (o3d re-binds them after memory growth) -------------------

  getNodeTransforms(): Float32Array {
    const view = this.nodes.getTransforms();
    if (view.buffer !== this.lastTransformsBuffer) {
      this.lastTransformsBuffer = view.buffer;
      this._viewVersion++;
    }
    return view;
  }

  getNodeColors(): Uint32Array {
    return this.nodes.getColors();
  }

  getEdgeVertices(): Float32Array {
    return this.edges.getVertices();
  }

  getEdgeColors(): Uint32Array {
    return this.edges.getColors();
  }

  getNodeTransformsPtr(): number {
    return this.nodes.getTransforms().byteOffset;
  }

  // ---- Capacity -------------------------------------------------------------------

  ensureNodeCapacity(required: number): boolean {
    this.assertAlive();
    if (required <= this.nodes.capacity) return false;
    const old = this.nodes;
    const next = this.createNodes(nextCapacity(old.capacity, required));
    // Re-fetch the old views after allocating: the allocation may have grown memory.
    next.getTransforms().set(old.getTransforms());
    next.getColors().set(old.getColors());
    const count = old.getCount();
    next.setCount(count);
    if (count > 0) next.markDirty(0, count - 1);
    old.destroy();
    this.nodes = next;
    this.scene.meshes[0] = next;
    this._viewVersion++;
    this.onChange();
    return true;
  }

  ensureEdgeCapacity(required: number): boolean {
    this.assertAlive();
    if (required <= this.edges.capacity) return false;
    const old = this.edges;
    const next = new this.o3d.LineBatch(this.device, nextCapacity(old.capacity, required));
    next.getVertices().set(old.getVertices());
    next.getColors().set(old.getColors());
    const count = old.getCount();
    next.setCount(count);
    if (count > 0) next.markDirty(0, count - 1);
    old.destroy();
    this.edges = next;
    this.scene.lines[0] = next;
    this._viewVersion++;
    this.onChange();
    return true;
  }

  // ---- Upload (→ o3d dirty ranges) ---------------------------------------------------

  commitNodes(activeCount: number): void {
    this.assertAlive();
    const n = this.nodes.setCount(activeCount);
    if (n > 0) this.nodes.markDirty(0, n - 1, GRAPH_CHANNELS);
    this.onChange();
  }

  commitNodeRange(start: number, count: number): void {
    this.assertAlive();
    const end = Math.min(start + count, this.nodes.capacity);
    if (end > start) this.nodes.markDirty(Math.max(0, start), end - 1, GRAPH_CHANNELS);
    this.onChange();
  }

  commitEdges(activeCount: number): void {
    this.assertAlive();
    const n = this.edges.setCount(activeCount);
    if (n > 0) this.edges.markDirty(0, n - 1);
    this.onChange();
  }

  commitEdgeRange(start: number, count: number): void {
    this.assertAlive();
    const end = Math.min(start + count, this.edges.capacity);
    if (end > start) this.edges.markDirty(Math.max(0, start), end - 1);
    this.onChange();
  }

  // ---- Picking & colors ---------------------------------------------------------------

  /** In-place WASM raycast over the active instances (radius = |scale| × mesh bounding radius). */
  raycastNodes(ndcX: number, ndcY: number, count: number): number {
    this.assertAlive();
    if (count <= 0) return -1;
    const hit = this.nodes.raycastNdc(this.cameraController, ndcX, ndcY);
    return hit >= 0 && hit < count ? hit : -1;
  }

  encodeColor(rgba: number): number {
    return encodeColorLayout(rgba, 'abgr');
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.nodes.destroy();
    this.edges.destroy();
    this.scene.meshes.length = 0;
    this.scene.lines.length = 0;
  }

  private createNodes(capacity: number): TInstanced {
    return new this.o3d.InstancedMesh(this.device, this.nodeMesh, capacity, { shading: this.shading });
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error('O3DBatchRenderer: used after dispose()');
  }
}

/** `ICameraAdapter` over an o3d `CameraController`. */
class CameraAdapter implements ICameraAdapter {
  constructor(
    private readonly camera: O3DCameraControllerLike,
    private readonly onChange: () => void,
  ) {}

  unprojectNdcRay(ndcX: number, ndcY: number): Float32Array {
    const ray = this.camera.unprojectNdc(ndcX, ndcY);
    const out = new Float32Array(6);
    if (ray === null) return out; // zero direction: the drag code treats it as parallel and skips
    out[0] = ray.origin[0];
    out[1] = ray.origin[1];
    out[2] = ray.origin[2];
    out[3] = ray.direction[0];
    out[4] = ray.direction[1];
    out[5] = ray.direction[2];
    return out;
  }

  getCameraForward(): Float32Array {
    const f = this.camera.getForward();
    return new Float32Array([f[0], f[1], f[2]]);
  }

  getCameraViewProjMatrix(): Float32Array {
    return Float32Array.from(this.camera.getViewProjection());
  }

  setCameraOrbitEnabled(enabled: boolean): void {
    this.camera.setOrbitEnabled(enabled);
  }

  /** 2D: orthographic, facing the XY plane head-on with orbit locked. 3D: perspective orbit. */
  setCamera2DMode(is2D: boolean): void {
    this.camera.setMode(is2D ? '2d' : '3d');
    this.camera.lockToXYPlane(is2D);
    this.onChange();
  }
}

function noop(): void {}
