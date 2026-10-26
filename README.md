# @omernach/o3d-graph

Interactive 2D/3D force-directed network graphs on the
[`@omernach/o3d`](https://www.npmjs.com/package/@omernach/o3d) WebGPU renderer.

**[Live demo →](https://omern8878.github.io/O3d-graph/)** (needs a WebGPU browser)

- **One call to get going:** `createGraphView()` sets up WebGPU, rendering,
  camera controls, labels and click-to-focus.
- **Barnes-Hut physics:** flat-array quadtree (2D) / octree (3D),
  velocity-Verlet integration, d3-compatible cooling and pluggable forces.
- **Interaction:** hover, click, and drag nodes in 3D or 2D; orbit, pan, and
  zoom towards whatever is under the cursor; focus a node and the camera
  follows it.
- **Labels:** names on nodes and text along edges. Each frame shows the most
  relevant ones, without overlaps, and they grow as you zoom in.
- **Fast:** node and edge data is written straight into o3d's WASM buffers,
  there's no allocation in hot loops, only changed ranges are uploaded, and
  nothing runs while the graph is idle.

```sh
npm install @omernach/o3d-graph @omernach/o3d
```

```ts
import * as o3d from '@omernach/o3d';
import { createGraphView } from '@omernach/o3d-graph';

const view = await createGraphView(o3d, { canvas: document.querySelector('canvas')! });

view.graph.setData({
  nodes: [
    { id: 'ada', label: 'Ada', color: '#4f9dff' },
    { id: 'bo', label: 'Bo', radius: 5 },
    { id: 'cy', label: 'Cy' },
  ],
  edges: [
    { source: 'ada', target: 'bo', label: 'mentors' },
    { source: 'bo', target: 'cy' },
  ],
});

view.graph.on('node:click', ({ id, meta }) => console.log('clicked', id, meta));
view.focus('ada', { zoom: true }); // select it and make it the orbit center
view.setDimensionMode('2d');
```

`createGraphView` rejects with a descriptive error if WebGPU isn't available
(it needs recent Chrome or Edge, or Safari 26+, served over localhost or
HTTPS). Call `view.dispose()` to free GPU, WASM and DOM resources.

### Requirements

- `@omernach/o3d` ^1.0.1 (peer dependency).
- TypeScript users: o3d's type declarations need `@webgpu/types` and
  `"lib": ["ESNext", "DOM"]` (or `"skipLibCheck": true`).

## Demo

A 1,000-person mock organisation (10 teams) built on `createGraphView`. It
covers search (`/`), focus with the rest of the graph dimmed, an inspector,
labels chosen per frame, hiding teams, 3D ⇄ 2D (`3` / `2`), fit (`F`), shake
(`R`), auto-rotate (`A`) and toggling labels (`L`).

[Try it online](https://omern8878.github.io/O3d-graph/), or run it locally:

```sh
npm install
npm run demo        # → http://localhost:5173
```

`demo/main.ts` contains only app UI, so it's a good template.

## `createGraphView(o3d, options)`

Takes every `O3DGraph` option (below), plus:

| Option | Default | |
|---|---|---|
| `labels` | `true` | `false`, or `LabelLayerOptions` |
| `cameraControls` | `true` | `false` to drive the camera yourself, or `CameraRigOptions` |
| `focusOnClick` | `true` | clicking a node focuses it; clicking empty space clears the focus |
| `onFocusChange` | — | `(id \| null) => void` |
| `autoFit` | `true` | keeps the layout framed while it first spreads out |
| `fitInsets` | — | `{ top, right, bottom, left }` in CSS px (or a function), so framing avoids overlaid UI |
| `containRadius` | `200` | outer radius the layout adapts to (see [Layout size](#layout-size)); `false` disables it |
| `background` | `'#070a10'` | clear color |
| `nodeShape` / `shading` | `'sphere'` / `'lit'` | `'disc'` / `'flat'` give a flat 2D look |

The returned `GraphView` gives you `graph` (an `O3DGraph`), `camera` (a
`CameraRig`), `labels`, `renderer`, `fit()`, `focus(id, { zoom })`,
`clearFocus()`, `focusedId`, `setDimensionMode()`, `requestRender()`, `fps`
and `dispose()`.

## `O3DGraph`

| Method | |
|---|---|
| `setData({ nodes, edges })` | Replace the graph and restart the layout. Atomic: invalid data (duplicate or missing ids, edges to unknown nodes) throws and leaves the current graph untouched |
| `addNode` / `addNodes` / `addEdge` / `addEdges` | Add to the graph and reheat (`{ reheat: false }` to skip) |
| `updateNode(id, patch)` / `updateEdge(id, patch)` | Change position, radius, color, mass, pinned, hidden, label or meta |
| `removeNode(id)` / `removeEdge(id)` / `clear()` | O(1) removal (O(k) for a node with k edges) |
| `pin` / `select` / `setHidden` / `setNodeColor` / `setEdgeColor` / `setNodeLabel` / `setEdgeLabel` | Per-item state |
| `getNodePosition(id, out?)` / `getNodeMeta(id)` | Read back without allocating |
| `setDimensionMode('2d' \| '3d')` | Flatten or Z-jitter positions and switch the camera projection |
| `registerForce(force)` / `unregisterForce(name)` / `getForce(name)` | Add or replace physics forces |
| `reheat(alpha?)` / `stopSimulation()` / `warmup(ticks)` | Simulation control |
| `on` / `once` / `off` | `node:click`, `node:hover`, `node:leave`, `node:dragstart`, `node:drag`, `node:dragend`, `engine:tick`, `engine:stabilized`, `mode:change` |

Event payloads that fire often (`engine:tick`, `node:*`) are pooled and reused.
Copy any fields you need to keep after the listener returns.

Node inputs: `id`, `label`, `x`/`y`/`z`, `radius`, `color`, `mass`,
`pinned`, `hidden`, `meta`. Edge inputs: `id`, `source`, `target`, `label`,
`distance`, `strength`, `color`, `targetColor`, `meta`. Ids are matched by
value, so `1` and `'1'` are different ids.

**Colors** can be CSS hex (`'#f80'`, `'#ff8800'`, `'#ff880080'`),
`'rgb(…)'` / `'rgba(…)'`, or numbers in `0xRRGGBBAA` form. Numbers need
eight hex digits with alpha last (`0xff8800ff`), not `0xff8800`.

### Options (selected)

```ts
{
  canvas,
  dimension: '3d',
  node: { radius: 3, color: '#4f9dff', hoverColor: '#ffc857', selectedColor: '#ff5d73' },
  edge: { color: 0x8a93a666, distance: 10 },
  physics: { chargeStrength: -8, theta: 0.8, velocityDecay: 0.4, collide: true },
  interaction: { pinOnDragEnd: false, reheatOnDrag: true, clickTolerancePx: 4 },
  ticksPerFrame: 2,   // max physics ticks per frame
  tickBudgetMs: 12,   // physics time budget per frame
}
```

## Labels

Give nodes and edges a `label`. `createGraphView` draws them with a
`LabelLayer`; to use one directly, create it as
`new LabelLayer(graph, options)` and call `labels.render()` after each
rendered frame.

Each frame, the layer shows only the most relevant labels, in this order:

1. the hovered and selected nodes,
2. the edges touching them,
3. their neighbours, then the nodes largest on screen,
4. other labeled edges, longest on screen first.

Labels that would overlap one already placed are skipped. Node labels are
centred on the ball and sized from its on-screen radius, so zooming in makes
them bigger. Edge labels sit between the two circles, rotated along the line
and kept upright.

Options:

- `maxNodeLabels` (40) and `maxEdgeLabels` (16).
- `edgeLabels`: `'auto'`, `'focused'` or `'none'`. It can be changed between
  frames, as can `enabled`.
- `nodeLabelPosition`: `'center'` or `'right'`.
- Text size: `fontScale` with `minFontSize` / `maxFontSize`, and the
  matching `edgeFontScale` / `minEdgeFontSize` / `maxEdgeFontSize`.
- `nodeText` / `edgeText`: compute label text instead of reading `label`.
- Styling: `fontFamily`, `fontWeight`, `color`, `focusColor`, `edgeColor`,
  `haloColor`, `haloWidth`.

At 1,000 nodes the layer takes about 0.1 ms per frame.

## Camera

`createGraphView` drives o3d's camera with a `CameraRig`:

- drag to orbit (pan in 2D; shift- or right-drag to pan in 3D),
- wheel, trackpad pinch or two-finger pinch to zoom towards the point (or
  node) under the cursor,
- `followNode()` / `view.focus()` to keep a node at the orbit centre while it
  moves,
- `fit()` with insets, and `autoRotate`.

Options: `fovY`, `far`, `minDistance`, `orbitSpeed`, `zoomSpeed`, `easeMs`
and `autoRotateSpeed`. The rig works with any camera that implements
`CameraRigCamera`.

## Layout size

o3d 1.0's perspective camera has a fixed 1000-unit far plane, so in 3D a
layout only stays fully visible if it's compact. Left alone, a
force-directed layout's radius grows with node count: about 180 units at 1k
nodes, 320 at 3k and 590 at 10k.

`createGraphView` registers a `ContainForce`. It adjusts a pull towards the
centre so that any graph settles with its outer edge at `containRadius`
(200 by default), and the camera rig never zooms out past the far plane. You
can use `ContainForce` on its own for consistent framing at any size. If o3d
later lets you set the far plane, pass `cameraControls: { far }` (or
`Infinity`) and adjust `containRadius` or turn it off.

## Custom forces

```ts
import type { IForce, ForceTickContext } from '@omernach/o3d-graph';

class Wind implements IForce {
  readonly name = 'wind';
  readonly requiresSpatialTree = false;
  enabled = true;
  initialize() {}
  apply({ forces, nodeCount, alpha }: ForceTickContext) {
    for (let i = 0; i < nodeCount; i++) forces[i * 3] += 0.5 * alpha;
  }
}
view.graph.registerForce(new Wind());
```

Built-in forces: `LinkForce`, `ManyBodyForce` (grouped Barnes-Hut),
`CollideForce`, `CenterForce`, `RadialForce` and `ContainForce`.

## Architecture

```
createGraphView        one-call setup: o3d, render loop, camera, labels, focus
└── O3DGraph           facade: data API, frame loop, events, DI
    ├── IGraphRenderer       zero-copy node/edge views, dirty-range commits, picking
    │     ├── O3DBatchRenderer   ← @omernach/o3d InstancedMesh + LineBatch + CameraController
    │     └── O3DRendererAdapter ← handle-based WASM bindings (IO3DEngineAdapter)
    ├── GraphStore           SoA buffers, id↔index maps, half-edge adjacency, labels
    ├── PhysicsSimulator     velocity Verlet + cooling + force registry
    │     ├── SpatialTree    flat-array quadtree / octree
    │     └── forces         Link · ManyBody · Collide · Center · Radial · Contain
    └── InteractionController  hover/click picking, ray-plane drag
CameraRig · LabelLayer · EventBus
```

Every part can be swapped. `O3DGraph` takes any `IGraphRenderer`, and its
third argument accepts `{ store, physics, events, forces, scheduler,
interaction: false }`.

## Performance

Physics cost per tick (single-threaded, Apple Silicon, default forces plus
`ContainForce`, 2 edges per node):

| Nodes | 3D | 2D |
|---|---|---|
| 1,000 | 2.6 ms | 1.6 ms |
| 5,000 | 12 ms | 8.4 ms |
| 10,000 | 23 ms | 16 ms |
| 50,000 | 149 ms | 101 ms |

Rendering, picking, labels and camera movement stay at frame rate. A settled
graph uses no CPU or GPU. The layout settles in real time up to about 5k
nodes (around 10k in 2D). Above that, `tickBudgetMs` keeps frames
responsive while the layout settles more slowly. At 50k+ nodes, settling takes
tens of seconds, and interaction stutters while it runs.

## Development

```sh
npm install
npm run check       # typecheck (lib + demo), tests, library build, demo build
npm test            # vitest, against a mock engine on a real WebAssembly.Memory
npm run demo
```

MIT licensed.
