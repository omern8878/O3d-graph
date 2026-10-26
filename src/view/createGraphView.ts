import type * as O3D from '@omernach/o3d';
import type { ColorInput, DimensionMode, GraphOptions, NodeId } from '../types/index.js';
import { O3DGraph } from '../O3DGraph.js';
import { O3DBatchRenderer } from '../renderer/O3DBatchRenderer.js';
import { CameraRig, type CameraRigOptions, type Vec3 } from '../camera/CameraRig.js';
import { LabelLayer, type LabelLayerOptions } from '../labels/LabelLayer.js';
import { ContainForce } from '../physics/forces/ContainForce.js';
import { parseColor } from '../utils/color.js';

/** The parts of `@omernach/o3d` a graph view uses — pass the imported module. */
export type O3DModule = Pick<
  typeof O3D,
  'initEngine' | 'GpuContext' | 'BatchRenderer' | 'CameraController' | 'GpuMesh' | 'InstancedMesh' | 'LineBatch'
>;

/** Space (CSS px) covered by UI on each side of the canvas; framing keeps the graph clear of it. */
export interface ViewInsets {
  top?: number;
  right?: number;
  bottom?: number;
  left?: number;
}

export interface GraphViewOptions extends GraphOptions {
  /** Canvas clear color. Default `'#070a10'`. */
  background?: ColorInput;
  /** Node geometry: `'sphere'` (default) or `'disc'` (flat circles, nice for 2D). */
  nodeShape?: 'sphere' | 'disc';
  /** `'lit'` (default for spheres) or `'flat'` (default for discs). */
  shading?: 'lit' | 'flat';
  /** Node / edge labels. `true` (default) uses defaults; `false` disables. */
  labels?: boolean | LabelLayerOptions;
  /** Built-in camera gestures. `true` (default); `false` leaves the camera to you. */
  cameraControls?: boolean | CameraRigOptions;
  /** Clicking a node focuses it (select + camera follow); clicking empty space clears. Default `true`. */
  focusOnClick?: boolean;
  /** Keep the camera framing the layout while it first expands, until the user takes over. Default `true`. */
  autoFit?: boolean;
  /** UI insets to keep the graph clear of when fitting (static value or a function read at fit time). */
  fitInsets?: ViewInsets | (() => ViewInsets);
  /**
   * Outer radius the layout adapts to (see `ContainForce`). o3d 1.0's camera
   * has a fixed 1000-unit far plane, so layouts must stay compact to be
   * fully visible in 3D. Default `200`; `false` disables.
   */
  containRadius?: number | false;
  /** Called whenever the focused node changes (click, `focus()`, `clearFocus()`, `fit()`, new data). */
  onFocusChange?: (id: NodeId | null) => void;
}

export interface FocusOptions {
  /** Also zoom so the node's whole neighborhood fills the view. Default `false`. */
  zoom?: boolean;
}

export interface GraphView<TNodeMeta = unknown, TEdgeMeta = unknown> {
  readonly graph: O3DGraph<TNodeMeta, TEdgeMeta>;
  readonly renderer: O3DBatchRenderer<GPUDevice, O3D.GpuMesh, O3D.InstancedMesh, O3D.LineBatch>;
  /** `null` when `cameraControls: false`. */
  readonly camera: CameraRig | null;
  /** `null` when `labels: false`. */
  readonly labels: LabelLayer | null;
  readonly cameraController: O3D.CameraController;
  readonly gpu: O3D.GpuContext;
  /** The node the camera currently follows, if any. */
  readonly focusedId: NodeId | null;
  /** Frames rendered during the last second (0 while idle). */
  readonly fps: number;

  /** Frames the whole (visible) graph. Releases focus. */
  fit(): void;
  /** Selects a node and makes it the camera's orbit center, following it while it moves. */
  focus(id: NodeId, options?: FocusOptions): void;
  clearFocus(): void;
  /** Switches 2D / 3D for the graph and the camera, keeping the focus or re-fitting. */
  setDimensionMode(mode: DimensionMode): void;
  /** Schedules a frame (the loop otherwise sleeps while nothing changes). */
  requestRender(): void;
  /** Stops rendering and frees GPU, WASM and DOM resources. */
  dispose(): void;
}

const FIT_EVERY_TICKS = 20;

/**
 * One-call setup of an interactive graph on `@omernach/o3d`: WebGPU
 * context, batch renderer, camera + gestures, labels, click-to-focus and an
 * on-demand render loop that sleeps once the layout settles.
 *
 * ```ts
 * import * as o3d from '@omernach/o3d';
 * import { createGraphView } from '@omernach/o3d-graph';
 *
 * const view = await createGraphView(o3d, { canvas });
 * view.graph.setData({ nodes, edges });
 * view.graph.on('node:click', ({ id }) => console.log(id));
 * ```
 *
 * Rejects if WebGPU is unavailable (the error message says so).
 */
export async function createGraphView<TNodeMeta = unknown, TEdgeMeta = unknown>(
  o3d: O3DModule,
  options: GraphViewOptions,
): Promise<GraphView<TNodeMeta, TEdgeMeta>> {
  const { canvas } = options;
  if (!canvas) throw new TypeError('createGraphView: options.canvas is required');
  const dimension: DimensionMode = options.dimension ?? '3d';

  // ---- o3d ------------------------------------------------------------------------
  await o3d.initEngine();
  const bg = parseColor(options.background, 0x070a10ff);
  const gpu = await o3d.GpuContext.create({
    canvas,
    clearColor: {
      r: ((bg >>> 24) & 255) / 255,
      g: ((bg >>> 16) & 255) / 255,
      b: ((bg >>> 8) & 255) / 255,
      a: (bg & 255) / 255,
    },
  });
  const cameraController = new o3d.CameraController({
    mode: dimension,
    position: { x: 0, y: 0, z: 600 },
    target: { x: 0, y: 0, z: 0 },
  });
  const batchRenderer = new o3d.BatchRenderer(gpu);
  const shape = options.nodeShape ?? 'sphere';
  const nodeMesh = shape === 'disc' ? o3d.GpuMesh.disc(gpu.device) : o3d.GpuMesh.sphere(gpu.device, 20, 14);

  // ---- Render loop (on demand) -----------------------------------------------------
  let disposed = false;
  let frameQueued = false;
  let frameHandle = 0;
  let lastFrame = 0;
  let width = 0;
  let height = 0;
  const frameTimes: number[] = [];
  const requestRender = () => {
    if (frameQueued || disposed) return;
    frameQueued = true;
    frameHandle = requestAnimationFrame(draw);
  };

  const renderer = new O3DBatchRenderer({
    o3d: { InstancedMesh: o3d.InstancedMesh, LineBatch: o3d.LineBatch },
    device: gpu.device,
    nodeMesh,
    camera: cameraController,
    shading: options.shading ?? (shape === 'disc' ? 'flat' : 'lit'),
    initialNodeCapacity: options.initialNodeCapacity,
    initialEdgeCapacity: options.initialEdgeCapacity,
    onChange: requestRender,
  });

  // ---- Graph (defaults sized for o3d's view volume) --------------------------------
  const graph = new O3DGraph<TNodeMeta, TEdgeMeta>(renderer, {
    ...options,
    dimension,
    node: { radius: 3, ...options.node },
    edge: { distance: 10, ...options.edge },
    physics: { chargeStrength: -8, collidePadding: 0.6, ...options.physics },
    initialLayoutRadius: options.initialLayoutRadius ?? 4,
    ticksPerFrame: options.ticksPerFrame ?? 2,
  });
  if (options.containRadius !== false) {
    graph.registerForce(new ContainForce({ radius: options.containRadius ?? 200 }));
  }
  const store = graph.store;

  const camera =
    options.cameraControls === false
      ? null
      : new CameraRig(cameraController, canvas, options.cameraControls === true ? {} : options.cameraControls);
  if (camera) camera.setMode(dimension);

  const labels =
    options.labels === false
      ? null
      : new LabelLayer(graph, options.labels === true || options.labels === undefined ? {} : options.labels);

  function draw(now: number): void {
    frameQueued = false;
    if (disposed) return;
    gpu.resizeToDisplaySize();
    if (canvas.width !== width || canvas.height !== height) {
      width = canvas.width;
      height = canvas.height;
      cameraController.resize(width, height);
    }
    if (camera) {
      // Hold the camera still while a node is dragged so it stays under the cursor.
      camera.followPaused = graph.interaction?.isDragging ?? false;
      camera.tick(lastFrame === 0 ? 16 : Math.min(50, now - lastFrame));
    }
    batchRenderer.render(cameraController, renderer.scene);
    labels?.render();
    lastFrame = now;
    frameTimes.push(now);
    while (frameTimes.length > 0 && now - frameTimes[0] > 1000) frameTimes.shift();
  }

  const resizeObserver = new ResizeObserver(requestRender);
  resizeObserver.observe(canvas);

  // ---- Helpers -----------------------------------------------------------------------
  const positionOf = (id: NodeId) => (): Vec3 | null => {
    const i = store.getNodeIndex(id);
    if (i < 0) return null;
    const p = store.getNodeTransforms();
    return [p[i * 4], p[i * 4 + 1], p[i * 4 + 2]];
  };

  function layoutBounds(): { center: Vec3; radius: number } {
    const p = store.getNodeTransforms();
    const n = store.nodeCount;
    let cx = 0, cy = 0, cz = 0, count = 0;
    for (let i = 0; i < n; i++) {
      if (p[i * 4 + 3] <= 0) continue; // hidden
      cx += p[i * 4];
      cy += p[i * 4 + 1];
      cz += p[i * 4 + 2];
      count++;
    }
    if (count === 0) return { center: [0, 0, 0], radius: 100 };
    cx /= count;
    cy /= count;
    cz /= count;
    let r = 0;
    for (let i = 0; i < n; i++) {
      if (p[i * 4 + 3] <= 0) continue;
      const d = Math.hypot(p[i * 4] - cx, p[i * 4 + 1] - cy, p[i * 4 + 2] - cz) + p[i * 4 + 3];
      if (d > r) r = d;
    }
    return { center: [cx, cy, cz], radius: Math.max(20, r + 4) };
  }

  const insets = (): ViewInsets =>
    typeof options.fitInsets === 'function' ? options.fitInsets() : (options.fitInsets ?? {});

  let focusedId: NodeId | null = null;
  let userTookOver = false;
  let fitWhenSettled = options.autoFit ?? true;

  function setFocused(id: NodeId | null): void {
    if (focusedId === id) return;
    focusedId = id;
    options.onFocusChange?.(id);
  }

  function fit(): void {
    clearSelection();
    setFocused(null);
    if (!camera) return;
    const b = layoutBounds();
    camera.fit(b.center, b.radius, insets());
  }

  function clearSelection(): void {
    if (focusedId !== null && graph.hasNode(focusedId)) graph.select(focusedId, false);
  }

  function focus(id: NodeId, opts: FocusOptions = {}): void {
    if (!graph.hasNode(id)) return;
    clearSelection();
    graph.select(id, true);
    setFocused(id);
    if (!camera) return;
    let radius: number | undefined;
    if (opts.zoom) {
      const i = store.getNodeIndex(id);
      const p = store.getNodeTransforms();
      radius = 20;
      for (let h = store.nodeFirstHalfEdge[i]; h !== -1; h = store.halfEdgeNext[h]) {
        const e = h >> 1;
        const j = (h & 1) === 0 ? store.edgeTargets[e] : store.edgeSources[e];
        radius = Math.max(radius, Math.hypot(p[j * 4] - p[i * 4], p[j * 4 + 1] - p[i * 4 + 1], p[j * 4 + 2] - p[i * 4 + 2]) + 6);
      }
    }
    camera.followNode(positionOf(id), radius);
  }

  function clearFocus(): void {
    clearSelection();
    setFocused(null);
    camera?.unfollow();
  }

  function setDimensionMode(mode: DimensionMode): void {
    if (graph.dimensionMode === mode) return;
    graph.setDimensionMode(mode);
    camera?.setMode(mode);
    if (focusedId === null) {
      fit();
      fitWhenSettled = true;
    }
  }

  // ---- Wiring ----------------------------------------------------------------------
  const disposers: Array<() => void> = [];
  if (camera) {
    camera.onChange = requestRender;
    // Zoom anchors on the node under the cursor (exact at any depth).
    camera.zoomAnchor = (clientX, clientY) => {
      const r = canvas.getBoundingClientRect();
      const i = renderer.raycastNodes(((clientX - r.left) / r.width) * 2 - 1, 1 - ((clientY - r.top) / r.height) * 2, store.nodeCount);
      if (i < 0) return null;
      const p = store.getNodeTransforms();
      return [p[i * 4], p[i * 4 + 1], p[i * 4 + 2]];
    };
    const takeOver = () => {
      userTookOver = true;
    };
    canvas.addEventListener('pointerdown', takeOver);
    canvas.addEventListener('wheel', takeOver, { passive: true });
    disposers.push(() => {
      canvas.removeEventListener('pointerdown', takeOver);
      canvas.removeEventListener('wheel', takeOver);
    });
  }

  if (options.focusOnClick ?? true) {
    disposers.push(graph.on('node:click', ({ id }) => (focusedId === id ? clearFocus() : focus(id))));
    if (camera) camera.onBackgroundTap = clearFocus;
  }
  disposers.push(
    graph.on('engine:tick', ({ tickCount }) => {
      if (fitWhenSettled && !userTookOver && focusedId === null && tickCount % FIT_EVERY_TICKS === 0) fit();
    }),
    graph.on('engine:stabilized', () => {
      if (fitWhenSettled && focusedId === null && !userTookOver) fit();
      fitWhenSettled = false;
    }),
  );
  // A new dataset re-enables auto-fit.
  const setData = graph.setData.bind(graph);
  graph.setData = (data) => {
    setFocused(null);
    camera?.unfollow();
    userTookOver = false;
    fitWhenSettled = options.autoFit ?? true;
    return setData(data);
  };

  requestRender();

  return {
    graph,
    renderer,
    camera,
    labels,
    cameraController,
    gpu,
    get focusedId() {
      return focusedId !== null && graph.hasNode(focusedId) ? focusedId : null;
    },
    get fps() {
      const now = performance.now();
      return frameTimes.filter((t) => now - t <= 1000).length;
    },
    fit,
    focus,
    clearFocus,
    setDimensionMode,
    requestRender,
    dispose() {
      if (disposed) return;
      disposed = true;
      cancelAnimationFrame(frameHandle);
      resizeObserver.disconnect();
      for (const d of disposers) d();
      labels?.dispose();
      camera?.dispose();
      graph.dispose(); // also destroys the renderer's batches
      batchRenderer.destroy();
      nodeMesh.destroy();
      cameraController.destroy();
    },
  };
}
