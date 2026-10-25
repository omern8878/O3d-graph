# @omernach/o3d-graph

Interactive 2D/3D force-directed network graphs on top of the
[`@omernach/o3d`](#engine-contract) WebAssembly renderer.

- **Zero-copy:** node positions are written straight into o3d's instanced-mesh
  transform buffer, and edge endpoints straight into its line buffer. Nothing
  is copied between JS and WASM.
- **No allocation in hot loops:** the physics tick, spatial tree, edge sync and
  picking all run on preallocated TypedArrays, which only grow when needed.
- **Barnes-Hut physics:** flat-array quadtree (2D) / octree (3D),
  velocity-Verlet integration, d3-compatible alpha cooling, and pluggable forces.
- **2D ⇄ 3D at runtime** with `setDimensionMode`.
- **Hover, click and 3D drag:** picking uses the engine's WASM raycast; dragging
  moves the node along a plane facing the camera.
- **Zero CPU when idle:** the frame loop stops once the layout has settled and
  wakes on interaction or changes.

```ts
import { BatchRenderer, CameraController, GpuContext, GpuMesh, InstancedMesh, LineBatch, initEngine } from '@omernach/o3d';
import { O3DBatchRenderer, O3DGraph } from '@omernach/o3d-graph';

await initEngine();
const gpu = await GpuContext.create({ canvas });
const camera = new CameraController({ mode: '3d', position: { x: 0, y: 0, z: 600 } });
const batches = new BatchRenderer(gpu);

let queued = false;
const renderer = new O3DBatchRenderer({
  o3d: { InstancedMesh, LineBatch },
  device: gpu.device,
  nodeMesh: GpuMesh.sphere(gpu.device),
  camera,
  // Called whenever buffers or the camera mode change: draw one frame on demand.
  onChange: () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      batches.render(camera, renderer.scene);
    });
  },
});

const graph = new O3DGraph(renderer, { canvas });

graph.setData({
  nodes: [{ id: 'a', color: '#4f9dff' }, { id: 'b' }, { id: 'c', radius: 8 }],
  edges: [{ source: 'a', target: 'b' }, { source: 'b', target: 'c', distance: 60 }],
});

graph.on('node:click', ({ id, meta }) => console.log('clicked', id, meta));
graph.setDimensionMode('2d');
```

`@omernach/o3d` is an optional peer dependency. The library never imports it:
`O3DBatchRenderer` takes o3d's classes as arguments, using structural types, so
this package builds and tests on its own. Camera gestures (orbit, pan, zoom)
belong to your app; while a node is being dragged, o3d-graph disables orbit on
the `CameraController`. See `demo/main.ts` for a complete setup.

## Demo

A 1,000-person mock organisation (10 teams) on the published
[`@omernach/o3d`](https://www.npmjs.com/package/@omernach/o3d) WebGPU renderer:

- hover for details, click to inspect, and see the selected person's
  neighbourhood with everything else dimmed
- select someone and the camera glides to them and keeps them at the centre
  of orbit and zoom, even while the layout moves (pan, Fit or Esc releases
  them)
- the scroll wheel zooms towards whatever is under the cursor, all the way in
  until the text is easy to read
- search people (`/`), jump to them, and walk through their connections
- drag nodes in 3D or 2D, and pin them in place
- switch 3D ⇄ 2D (`3` / `2`), fit (`F`), shake the layout (`R`), auto-rotate (`A`)
- names on the most prominent people and relationship labels on links,
  chosen per frame; toggle them with `L`
- hide or show teams from the legend
- the GPU sleeps once the layout settles

```sh
npm install
npm run demo        # → http://localhost:5173 (needs a WebGPU browser: Chrome, Edge, Safari 26+)
```

The demo source is a good starting point for your own app: `demo/main.ts`
covers GPU setup, the graph and the UI, and `demo/cameraRig.ts` covers camera
gestures and framing within o3d's fixed far plane.

## API

| Method | |
|---|---|
| `setData({ nodes, edges })` | Replace the graph and restart the layout |
| `addNode` / `addNodes` / `addEdge` / `addEdges` | Add to the graph and reheat (`{ reheat: false }` to skip) |
| `updateNode(id, patch)` / `updateEdge(id, patch)` | Change position, radius, color, mass, pinned, hidden or meta |
| `removeNode(id)` / `removeEdge(id)` / `clear()` | O(1) swap-and-pop removal (O(k) for a node with k edges) |
| `pin` / `select` / `setHidden` / `setNodeColor` / `setEdgeColor` | Per-node state |
| `getNodePosition(id, out?)` / `getNodeMeta(id)` | Read back without allocating |
| `setDimensionMode('2d' \| '3d')` | Flatten or Z-jitter, swap tree type, switch camera, reheat |
| `registerForce(force)` / `unregisterForce(name)` / `getForce(name)` | Add or replace physics forces (see `IForce`) |
| `reheat(alpha?)` / `stopSimulation()` / `warmup(ticks)` | Simulation control |
| `start()` / `stop()` / `dispose()` | Loop control and teardown |
| `on` / `once` / `off` | `node:click`, `node:hover`, `node:leave`, `node:dragstart`, `node:drag`, `node:dragend`, `engine:tick`, `engine:stabilized`, `mode:change` |

High-frequency event payloads (`engine:tick`, `node:*`) are pooled and reused.
Copy any fields you need to keep after the listener returns.

### Labels

Give nodes and edges a `label`, then draw them with a `LabelLayer`, a 2D
canvas laid over the graph:

```ts
import { LabelLayer } from '@omernach/o3d-graph';

graph.setData({
  nodes: [{ id: 'a', label: 'Ada' }, { id: 'b', label: 'Bo' }],
  edges: [{ source: 'a', target: 'b', label: 'mentors' }],
});
const labels = new LabelLayer(graph, { maxNodeLabels: 40, maxEdgeLabels: 16 });

// After each rendered frame:
batches.render(camera, renderer.scene);
labels.render();

graph.setNodeLabel('a', 'Ada L.'); // or updateNode / updateEdge with { label }
```

Each frame, the layer shows only the most relevant labels, in this order:

1. the hovered and selected nodes,
2. the edges touching them,
3. their neighbours, then the nodes that are largest on screen (big and close),
4. other labeled edges, longest on screen first.

Labels that would overlap one already placed are skipped. Node labels are
centred on the ball and sized from its on-screen radius, so zooming in makes
them bigger. Edge labels sit between the two circles, rotated along the line
and kept upright.

Options:

- `edgeLabels`: `'auto'` (default), `'focused'` (only edges of the hovered or
  selected node) or `'none'`. It can be changed between frames, as can
  `enabled`.
- `nodeText` / `edgeText`: compute label text instead of reading `label`.
- `nodeLabelPosition`: `'center'` (default) or `'right'` (beside the ball,
  flipping left near the edge of the screen).
- Text size: `fontScale` (relative to the ball's on-screen radius, default
  `0.8`) clamped to `minFontSize`–`maxFontSize` (11–30 px); for edges,
  `edgeFontScale`, `minEdgeFontSize` and `maxEdgeFontSize`.
- Styling: `fontFamily`, `fontWeight`, `edgeFontWeight`, `color`,
  `focusColor`, `edgeColor`, `haloColor`, `haloWidth`, `minNodeRadiusPx`.

The layer creates its own overlay canvas unless you pass `canvas`. Its
buffers are reused between frames; at 1000 nodes a frame takes about 0.1 ms.

### Options (selected)

```ts
new O3DGraph(o3d, {
  canvas,
  dimension: '3d',
  node: { radius: 4, color: '#4f9dff', hoverColor: '#ffc857', selectedColor: '#ff5d73' },
  edge: { color: 0x8a93a666, distance: 30 },
  physics: { chargeStrength: -30, theta: 0.8, velocityDecay: 0.4, collide: true },
  interaction: { pinOnDragEnd: false, reheatOnDrag: true, clickTolerancePx: 4 },
  ticksPerFrame: 1,   // max physics ticks per frame
  tickBudgetMs: 12,   // physics time budget per frame
});
```

Colors are either numbers in `0xRRGGBBAA` form or CSS hex strings (`#rgb`,
`#rgba`, `#rrggbb`, `#rrggbbaa`).

### Custom forces

```ts
import type { IForce, ForceTickContext } from '@omernach/o3d-graph';

class GravityWell implements IForce {
  readonly name = 'well';
  readonly requiresSpatialTree = false;
  enabled = true;
  initialize() {}
  apply({ positions: p, forces: f, nodeCount, alpha }: ForceTickContext) {
    for (let i = 0; i < nodeCount; i++) {
      f[i * 3]     -= p[i * 4]     * 0.01 * alpha;
      f[i * 3 + 1] -= p[i * 4 + 1] * 0.01 * alpha;
    }
  }
}
graph.registerForce(new GravityWell());
```

## Architecture

```
O3DGraph (facade, frame loop, DI)
├── IGraphRenderer       zero-copy node/edge views, dirty-range commits, picking, camera
│     ├── O3DBatchRenderer    ← @omernach/o3d InstancedMesh + LineBatch + CameraController
│     └── O3DRendererAdapter  ← IO3DEngineAdapter (handle-based WASM bindings)
├── GraphStore           SoA buffers, id↔index maps, half-edge adjacency, dirty ranges
├── PhysicsSimulator     velocity Verlet + alpha cooling + force registry
│     ├── SpatialTree    flat-array quadtree / octree
│     └── forces         Link · ManyBody (grouped Barnes-Hut) · Radial · Collide · Center
├── InteractionController  pointer → raycast hover/click, ray-plane drag
└── EventBus             typed emitter that never allocates on emit
```

The first constructor argument is any `IGraphRenderer` (or handle-based
bindings). The other collaborators can be swapped through the third argument
(`{ store, physics, events, forces, scheduler, interaction: false }`).

### Renderer backends

- **`O3DBatchRenderer`** for `@omernach/o3d`. Nodes are one `InstancedMesh`,
  edges one `LineBatch`, and the camera is a `CameraController`. Commits
  become o3d dirty ranges, and colors are written in o3d's byte order (red in
  the lowest byte). o3d's perspective camera has a fixed 1000-unit far plane,
  so keep layouts within a radius of a few hundred units (use short links and
  a `RadialForce` with `radius: 0` as a pull towards the center).
- **`O3DRendererAdapter`** for an engine with the handle-based function API
  described below. It defaults to `0xRRGGBBAA` colors; set `colorLayout` to
  change that.
- **Your own backend:** implement `IGraphRenderer`. Return zero-copy
  `[x, y, z, radius]` / segment / color views, accept dirty ranges, pick
  nodes, and translate colors with `encodeColor`.

Public colors are always `0xRRGGBBAA` numbers or CSS hex strings, whatever
the backend uses internally.

#### Handle-based contract

`O3DEngineAdapter` expects these exports:
`get_wasm_memory`, `create_instanced_mesh`, `get_instance_transforms_ptr`,
`get_instance_colors_ptr`, `update_instanced_mesh`, `update_instance_sub_range`,
`create_line_buffer`, `get_line_vertices_ptr`, `get_line_colors_ptr`,
`update_line_buffer`, `update_line_sub_range`, `raycast_spheres_ndc`,
`unproject_ndc_ray`, `get_camera_forward`, `get_camera_view_proj_matrix`,
`set_camera_orbit_enabled`, `set_camera_2d_mode`.

`destroy_instanced_mesh` / `destroy_line_buffer` are optional. If they're
missing, buffers replaced during capacity growth are hidden but not freed.

## Performance

Physics cost per tick, measured single-threaded in Node on Apple Silicon:

| Graph | 3D | 2D |
|---|---|---|
| 5k nodes / 10k edges | ~10 ms | — |
| 10k / 20k | ~19 ms | — |
| 20k / 40k | ~42 ms | — |
| 50k / 100k | ~114 ms | ~53 ms |

Rendering and dragging stay at frame rate at any size: after an interaction,
only the changed sub-ranges are uploaded, and a settled graph costs nothing.
The layout itself runs in real time up to about 5–8k nodes in 3D. Above that,
`tickBudgetMs` keeps each frame responsive, but the layout settles at a few
ticks per second. To run physics in real time at 50k+ nodes, the n-body loop
would need to move into WASM SIMD or a worker.

## Development

```sh
npm install
npm run typecheck
npm test        # vitest; uses a mock engine over a real WebAssembly.Memory
npm run build   # → dist/
```
