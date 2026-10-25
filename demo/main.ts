import {
  BatchRenderer,
  CameraController,
  GpuContext,
  GpuMesh,
  InstancedMesh,
  LineBatch,
  initEngine,
} from '@omernach/o3d';
import { LabelLayer, NO_INDEX, NodeFlag, O3DBatchRenderer, O3DGraph, RadialForce } from '../src/index.js';
import { CameraRig } from './cameraRig.js';
import { EDGE_ALPHA, TEAMS, createMockGraph, withAlpha, type LinkMeta, type PersonMeta } from './mockData.js';

/**
 * o3d-graph on the published @omernach/o3d WebGPU renderer: 1,000 mock
 * people in 10 teams. The graph library does layout, picking and dragging
 * and writes straight into o3d's WASM buffers; this file wires the GPU,
 * camera and UI.
 *
 * Layout scale: o3d's perspective camera has a fixed 1000-unit far plane, so
 * the physics is tuned to settle within a radius of ~200 (short links, low
 * repulsion, a weak pull to the center) and the camera rig clamps zoom-out.
 */

const NODE_COUNT = 1000;
const DIM_NODE = 0x232a38ff;
const FOCUS_EDGE_ALPHA = 0xe6;
const DIM_EDGE_ALPHA = 0x10;

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('view');
const css = (rgba: number) => `#${(rgba >>> 8).toString(16).padStart(6, '0')}`;
const escapeHtml = (s: string) => s.replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);

boot().catch((err: unknown) => {
  console.error(err);
  showFatal('Something went wrong', err instanceof Error ? err.message : String(err));
});

function showFatal(title: string, message: string): void {
  const fatal = $('fatal');
  fatal.innerHTML = `<div><h2>${escapeHtml(title)}</h2><p>${escapeHtml(message)}</p></div>`;
  fatal.hidden = false;
}

async function boot(): Promise<void> {
  if (!('gpu' in navigator)) {
    showFatal(
      'WebGPU is not available',
      'This demo renders with WebGPU. Open it in a recent Chrome or Edge (or Safari 26+), served from localhost or HTTPS.',
    );
    return;
  }

  // Portrait phones start in 2D: fitting the 3D view's width there would need a camera
  // distance beyond o3d's fixed far plane, while the orthographic 2D view has no such limit.
  const startMode: '2d' | '3d' = innerWidth < 640 && innerHeight > innerWidth ? '2d' : '3d';

  // ---- o3d ---------------------------------------------------------------------------
  await initEngine();
  const gpu = await GpuContext.create({ canvas, clearColor: { r: 0.027, g: 0.039, b: 0.063, a: 1 } });
  const camera = new CameraController({ mode: startMode, position: { x: 0, y: 0, z: 600 }, target: { x: 0, y: 0, z: 0 } });
  const batchRenderer = new BatchRenderer(gpu);
  const sphere = GpuMesh.sphere(gpu.device, 20, 14);
  const rig = new CameraRig(camera, canvas);
  rig.setMode(startMode);

  // ---- On-demand render loop -------------------------------------------------------------
  let frameQueued = false;
  let lastFrame = performance.now();
  const frameTimes: number[] = [];
  let size = { w: 0, h: 0 };

  const requestRender = () => {
    if (frameQueued) return;
    frameQueued = true;
    requestAnimationFrame(draw);
  };

  const renderer = new O3DBatchRenderer({
    o3d: { InstancedMesh, LineBatch },
    device: gpu.device,
    nodeMesh: sphere,
    camera,
    initialNodeCapacity: 1024,
    initialEdgeCapacity: 2048,
    onChange: requestRender,
  });

  function draw(now: number): void {
    frameQueued = false;
    gpu.resizeToDisplaySize();
    if (canvas.width !== size.w || canvas.height !== size.h) {
      size = { w: canvas.width, h: canvas.height };
      camera.resize(size.w, size.h);
    }
    // Hold the camera still while a node is dragged, so the node stays under the cursor.
    rig.followPaused = graphRef?.interaction?.isDragging ?? false;
    rig.tick(Math.min(50, now - lastFrame)); // while easing/following/rotating, onChange queues the next frame
    batchRenderer.render(camera, renderer.scene);
    labels?.render();
    lastFrame = now;
    frameTimes.push(now);
    while (frameTimes.length > 0 && now - frameTimes[0] > 1000) frameTimes.shift();
  }
  rig.onChange = requestRender;
  let labels: LabelLayer | null = null; // created once the graph exists
  let graphRef: O3DGraph<PersonMeta, LinkMeta> | null = null;
  new ResizeObserver(requestRender).observe(canvas);

  // ---- Graph ---------------------------------------------------------------------------------
  const graph = new O3DGraph<PersonMeta, LinkMeta>(renderer, {
    canvas,
    dimension: startMode,
    node: { hoverColor: '#ffffff', selectedColor: '#ffffff' },
    edge: { distance: 8 },
    physics: { chargeStrength: -6, collidePadding: 0.6 },
    initialLayoutRadius: 4,
    ticksPerFrame: 2,
  });
  graph.registerForce(new RadialForce({ radius: 0, strength: 0.14 }, 'gravity'));
  graph.setData(createMockGraph(NODE_COUNT));
  const store = graph.store;
  graphRef = graph;

  $('count-nodes').textContent = graph.nodeCount.toLocaleString();
  $('count-edges').textContent = graph.edgeCount.toLocaleString();

  // ---- Helpers -----------------------------------------------------------------------------------
  const meta = (index: number) => store.getNodeMeta(index)!;
  const teamColor = (index: number) => TEAMS[meta(index).team].color;

  function neighborsOf(index: number): number[] {
    const out: number[] = [];
    for (let h = store.nodeFirstHalfEdge[index]; h !== NO_INDEX; h = store.halfEdgeNext[h]) {
      const e = h >> 1;
      out.push((h & 1) === 0 ? store.edgeTargets[e] : store.edgeSources[e]);
    }
    return out;
  }

  /** Centroid + radius enclosing every visible node. */
  function layoutBounds(): { center: [number, number, number]; radius: number } {
    const p = store.getNodeTransforms();
    let cx = 0, cy = 0, cz = 0, n = 0;
    for (let i = 0; i < store.nodeCount; i++) {
      if (p[i * 4 + 3] <= 0) continue;
      cx += p[i * 4];
      cy += p[i * 4 + 1];
      cz += p[i * 4 + 2];
      n++;
    }
    if (n === 0) return { center: [0, 0, 0], radius: 100 };
    cx /= n;
    cy /= n;
    cz /= n;
    const d: number[] = [];
    for (let i = 0; i < store.nodeCount; i++) {
      if (p[i * 4 + 3] > 0) d.push(Math.hypot(p[i * 4] - cx, p[i * 4 + 1] - cy, p[i * 4 + 2] - cz));
    }
    d.sort((a, b) => a - b);
    return { center: [cx, cy, cz], radius: Math.max(30, d[d.length - 1] + 8) };
  }

  /**
   * Frame the graph in the part of the canvas no UI covers: below the brand
   * card when it spans the width (phones), above the bottom dock, and within
   * the width (the limiting side in portrait).
   */
  const dock = document.querySelector<HTMLElement>('.dock')!;
  const brand = document.querySelector<HTMLElement>('.brand')!;
  const fit = () => {
    const b = layoutBounds();
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const brandRect = brand.getBoundingClientRect();
    const top = brandRect.width > w * 0.6 ? brandRect.bottom + 8 : 0;
    const bottom = h - dock.getBoundingClientRect().top + 8;
    const fraction = Math.min((h - top - bottom) / h, (w - 24) / h);
    rig.fit(b.center, b.radius, fraction, (bottom - top) / 2);
  };

  // ---- Focus mode: selected node + neighbors in color, everything else dimmed ---------------------
  let selectedId: string | number | null = null;
  let near = new Uint8Array(0); // 1 = selected node or one of its neighbors

  // Labels: names on the most prominent nodes each frame, relationships on edges. While a
  // person is selected, only their neighborhood is labeled (the rest is dimmed anyway).
  labels = new LabelLayer(graph, {
    maxNodeLabels: 36,
    maxEdgeLabels: 14,
    nodeText: (i) => (selectedId !== null && near[i] === 0 ? undefined : store.getNodeLabel(i)),
  });

  function paint(): void {
    const focus = selectedId === null ? NO_INDEX : store.getNodeIndex(selectedId);
    near = new Uint8Array(store.nodeCount);
    if (focus !== NO_INDEX) {
      near[focus] = 1;
      for (const j of neighborsOf(focus)) near[j] = 1;
    }
    labels!.edgeLabels = focus === NO_INDEX ? 'auto' : 'focused';
    for (let i = 0; i < store.nodeCount; i++) {
      store.setNodeColor(i, focus === NO_INDEX || near[i] ? teamColor(i) : DIM_NODE);
    }
    for (let e = 0; e < store.edgeCount; e++) {
      const s = store.edgeSources[e];
      const t = store.edgeTargets[e];
      const alpha = focus === NO_INDEX ? EDGE_ALPHA : s === focus || t === focus ? FOCUS_EDGE_ALPHA : DIM_EDGE_ALPHA;
      store.setEdgeColor(e, withAlpha(teamColor(s), alpha), withAlpha(teamColor(t), alpha));
    }
    graph.requestFrame();
  }

  const inspector = $('inspector');

  /** Live position of a node, for the camera to follow (null once it's removed). */
  const positionOf = (id: string | number) => (): [number, number, number] | null => {
    const i = store.getNodeIndex(id);
    if (i === NO_INDEX) return null;
    const p = store.getNodeTransforms();
    return [p[i * 4], p[i * 4 + 1], p[i * 4 + 2]];
  };

  /** Follows a node and zooms so its whole neighborhood (and its labels) fit. */
  function focusNeighborhood(id: string | number): void {
    const i = store.getNodeIndex(id);
    const p = store.getNodeTransforms();
    const o = i * 4;
    let radius = 25;
    for (const j of neighborsOf(i)) {
      radius = Math.max(radius, Math.hypot(p[j * 4] - p[o], p[j * 4 + 1] - p[o + 1], p[j * 4 + 2] - p[o + 2]) + 6);
    }
    rig.followNode(positionOf(id), radius);
  }

  /**
   * Selecting a node makes it the camera's orbit center (it stays on screen while you orbit,
   * zoom or the layout moves). `zoomToNeighborhood` also frames its connections.
   */
  function select(id: string | number | null, opts: { zoomToNeighborhood?: boolean } = {}): void {
    if (selectedId !== null && graph.hasNode(selectedId)) graph.select(selectedId, false);
    selectedId = id !== null && graph.hasNode(id) ? id : null;
    if (selectedId !== null) {
      graph.select(selectedId, true);
      if (opts.zoomToNeighborhood) focusNeighborhood(selectedId);
      else rig.followNode(positionOf(selectedId));
    } else {
      rig.unfollow();
    }
    paint();
    renderInspector();
  }

  function renderInspector(): void {
    if (selectedId === null) {
      inspector.hidden = true;
      return;
    }
    const i = store.getNodeIndex(selectedId);
    const m = meta(i);
    const team = TEAMS[m.team];
    const nbrs = neighborsOf(i).sort((a, b) => store.nodeDegrees[b] - store.nodeDegrees[a]);
    const crossTeam = nbrs.filter((j) => meta(j).team !== m.team).length;
    const pinned = store.hasFlag(i, NodeFlag.PINNED);
    const shown = nbrs.slice(0, 14);
    inspector.innerHTML = `
      <div>
        <h2>${escapeHtml(m.name)}</h2>
        <p class="muted" style="margin:4px 0 0">${escapeHtml(m.role)}</p>
      </div>
      <span class="chip"><span class="dot" style="background:${css(team.color)}"></span>${team.name}</span>
      <dl class="facts">
        <div><dt>Links</dt><dd>${nbrs.length}</dd></div>
        <div><dt>Cross-team</dt><dd>${crossTeam}</dd></div>
        <div><dt>Rank</dt><dd>#${rankOf(i)}</dd></div>
      </dl>
      <div class="actions">
        <button type="button" data-action="focus">Center</button>
        <button type="button" data-action="pin" aria-pressed="${pinned}">${pinned ? 'Pinned' : 'Pin'}</button>
      </div>
      <section>
        <h3>Connections</h3>
        <ul class="neighbors">
          ${shown
            .map((j) => {
              const mj = meta(j);
              return `<li><button type="button" data-id="${escapeHtml(String(store.getNodeId(j)))}">
                <span class="dot" style="background:${css(TEAMS[mj.team].color)}"></span>${escapeHtml(mj.name)}
                <span class="sub">${store.nodeDegrees[j]}</span></button></li>`;
            })
            .join('')}
        </ul>
        ${nbrs.length > shown.length ? `<p class="muted" style="margin:6px 0 0">and ${nbrs.length - shown.length} more</p>` : ''}
      </section>`;
    inspector.hidden = false;
  }

  let degreeRank: Map<number, number> | null = null;
  function rankOf(index: number): number {
    if (degreeRank === null) {
      const order = Array.from({ length: store.nodeCount }, (_, i) => i).sort(
        (a, b) => store.nodeDegrees[b] - store.nodeDegrees[a],
      );
      degreeRank = new Map(order.map((i, r) => [i, r + 1]));
    }
    return degreeRank.get(index) ?? store.nodeCount;
  }

  inspector.addEventListener('click', (e) => {
    const target = (e.target as HTMLElement).closest('button');
    if (!target || selectedId === null) return;
    if (target.dataset.id) {
      select(target.dataset.id, { zoomToNeighborhood: true });
    } else if (target.dataset.action === 'focus') {
      focusNeighborhood(selectedId);
    } else if (target.dataset.action === 'pin') {
      const i = store.getNodeIndex(selectedId);
      graph.pin(selectedId, !store.hasFlag(i, NodeFlag.PINNED));
      renderInspector();
    }
  });

  // ---- Graph events ---------------------------------------------------------------------------------
  const tooltip = $('tooltip');
  let pointer = { x: 0, y: 0 };
  canvas.addEventListener('pointermove', (e) => {
    pointer = { x: e.clientX, y: e.clientY };
    if (!tooltip.hidden) placeTooltip();
  });
  function placeTooltip(): void {
    const pad = 14;
    const w = tooltip.offsetWidth;
    const h = tooltip.offsetHeight;
    const x = pointer.x + pad + w > innerWidth ? pointer.x - pad - w : pointer.x + pad;
    const y = pointer.y + pad + h > innerHeight ? pointer.y - pad - h : pointer.y + pad;
    tooltip.style.transform = `translate(${x}px, ${y}px)`;
  }

  graph.on('node:hover', ({ index }) => {
    const m = meta(index);
    tooltip.innerHTML = `<strong>${escapeHtml(m.name)}</strong>
      <span class="muted"><span class="dot" style="display:inline-block;margin-right:6px;background:${css(TEAMS[m.team].color)}"></span>${TEAMS[m.team].name} · ${store.nodeDegrees[index]} links</span>`;
    tooltip.hidden = false;
    placeTooltip();
  });
  graph.on('node:leave', () => (tooltip.hidden = true));
  graph.on('node:dragstart', () => (tooltip.hidden = true));
  graph.on('node:click', ({ id }) => select(selectedId === id ? null : id));
  rig.onBackgroundTap = () => select(null);
  // Zoom anchors on the node under the cursor (exact at any depth), else on the focus plane.
  rig.zoomAnchor = (clientX, clientY) => {
    const r = canvas.getBoundingClientRect();
    const i = renderer.raycastNodes(((clientX - r.left) / r.width) * 2 - 1, 1 - ((clientY - r.top) / r.height) * 2, store.nodeCount);
    if (i < 0) return null;
    const p = store.getNodeTransforms();
    return [p[i * 4], p[i * 4 + 1], p[i * 4 + 2]];
  };

  // ---- Status + first fit -------------------------------------------------------------------------------
  const status = $('status');
  let fitAfterSettle = true;
  graph.on('engine:stabilized', () => {
    if (fitAfterSettle) {
      fitAfterSettle = false;
      if (!rig.isFollowing) fit();
    }
  });

  // Keep framing the layout while it expands on load, until the user takes over.
  let userMovedCamera = false;
  canvas.addEventListener('pointerdown', () => (userMovedCamera = true));
  canvas.addEventListener('wheel', () => (userMovedCamera = true), { passive: true });
  addEventListener('resize', () => {
    if (!userMovedCamera) fit();
  });
  graph.on('engine:tick', ({ tickCount }) => {
    if (!userMovedCamera && !rig.isFollowing && fitAfterSettle && tickCount % 20 === 0) fit();
  });

  // ---- Toolbar -------------------------------------------------------------------------------------------
  const mode3d = $<HTMLButtonElement>('mode-3d');
  const mode2d = $<HTMLButtonElement>('mode-2d');
  const rotateBtn = $<HTMLButtonElement>('btn-rotate');

  function setMode(mode: '2d' | '3d'): void {
    if (graph.dimensionMode === mode) return;
    graph.setDimensionMode(mode);
    rig.setMode(mode);
    mode3d.setAttribute('aria-pressed', String(mode === '3d'));
    mode2d.setAttribute('aria-pressed', String(mode === '2d'));
    rotateBtn.disabled = mode === '2d';
    if (!rig.isFollowing) fit(); // a followed node stays the center across the switch
    fitAfterSettle = true;
  }
  function setRotate(on: boolean): void {
    rig.autoRotate = on && graph.dimensionMode === '3d';
    rotateBtn.setAttribute('aria-pressed', String(rig.autoRotate));
    requestRender();
  }
  const reheat = () => {
    graph.reheat(0.7);
  };

  mode3d.setAttribute('aria-pressed', String(startMode === '3d'));
  mode2d.setAttribute('aria-pressed', String(startMode === '2d'));
  rotateBtn.disabled = startMode === '2d';
  mode3d.addEventListener('click', () => setMode('3d'));
  mode2d.addEventListener('click', () => setMode('2d'));
  $('btn-fit').addEventListener('click', fit);
  $('btn-reheat').addEventListener('click', reheat);
  rotateBtn.addEventListener('click', () => setRotate(!rig.autoRotate));
  const labelsBtn = $<HTMLButtonElement>('btn-labels');
  const setLabels = (on: boolean) => {
    labels!.enabled = on;
    labelsBtn.setAttribute('aria-pressed', String(on));
    requestRender();
  };
  labelsBtn.addEventListener('click', () => setLabels(!labels!.enabled));

  // ---- Team legend (hide / show = NodeFlag.HIDDEN: scale 0, edges collapsed, not pickable) ---------------
  const legend = $('legend');
  const hiddenTeams = new Set<number>();
  const teamSizes = TEAMS.map(() => 0);
  for (let i = 0; i < store.nodeCount; i++) teamSizes[meta(i).team]++;
  legend.innerHTML = TEAMS.map(
    (t, k) => `<button type="button" data-team="${k}" aria-pressed="true">
      <span class="dot" style="background:${css(t.color)}"></span>${t.name} <span class="n">${teamSizes[k]}</span></button>`,
  ).join('');
  legend.addEventListener('click', (e) => {
    const button = (e.target as HTMLElement).closest<HTMLButtonElement>('button[data-team]');
    if (!button) return;
    const team = Number(button.dataset.team);
    const hide = !hiddenTeams.has(team);
    if (hide) hiddenTeams.add(team);
    else hiddenTeams.delete(team);
    button.setAttribute('aria-pressed', String(!hide));
    for (let i = 0; i < store.nodeCount; i++) {
      if (meta(i).team === team) graph.setHidden(store.getNodeId(i), hide);
    }
    if (hide && selectedId !== null && meta(store.getNodeIndex(selectedId)).team === team) select(null);
  });

  // ---- Search ---------------------------------------------------------------------------------------------
  const search = $<HTMLInputElement>('search');
  const results = $<HTMLUListElement>('results');
  let matches: number[] = [];
  let active = 0;

  function renderResults(): void {
    results.innerHTML = matches
      .map((i, k) => {
        const m = meta(i);
        return `<li><button type="button" role="option" data-index="${i}" aria-selected="${k === active}">
          <span class="dot" style="background:${css(TEAMS[m.team].color)}"></span>${escapeHtml(m.name)}
          <span class="sub">${TEAMS[m.team].name}</span></button></li>`;
      })
      .join('');
    results.hidden = matches.length === 0;
  }
  function choose(index: number): void {
    search.value = meta(index).name;
    results.hidden = true;
    search.blur();
    select(store.getNodeId(index), { zoomToNeighborhood: true });
  }
  search.addEventListener('input', () => {
    const q = search.value.trim().toLowerCase();
    matches = [];
    if (q.length > 0) {
      for (let i = 0; i < store.nodeCount && matches.length < 8; i++) {
        if (!hiddenTeams.has(meta(i).team) && meta(i).name.toLowerCase().includes(q)) matches.push(i);
      }
    }
    active = 0;
    renderResults();
  });
  search.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      if (matches.length === 0) return;
      active = (active + (e.key === 'ArrowDown' ? 1 : matches.length - 1)) % matches.length;
      renderResults();
    } else if (e.key === 'Enter' && matches.length > 0) {
      choose(matches[active]);
    } else if (e.key === 'Escape') {
      search.value = '';
      results.hidden = true;
      search.blur();
    }
  });
  results.addEventListener('pointerdown', (e) => e.preventDefault()); // keep focus until click
  results.addEventListener('click', (e) => {
    const b = (e.target as HTMLElement).closest<HTMLButtonElement>('button[data-index]');
    if (b) choose(Number(b.dataset.index));
  });

  // ---- Keyboard ---------------------------------------------------------------------------------------------
  addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === 'escape') select(null);
    else if (k === 'f') fit();
    else if (k === 'r') reheat();
    else if (k === 'a') setRotate(!rig.autoRotate);
    else if (k === 'l') setLabels(!labels!.enabled);
    else if (k === '2') setMode('2d');
    else if (k === '3') setMode('3d');
    else if (k === '/') {
      e.preventDefault();
      search.focus();
    }
  });

  // ---- Stats (4 Hz) ------------------------------------------------------------------------------------------
  const stats = $('stats');
  setInterval(() => {
    const now = performance.now();
    const fps = frameTimes.filter((t) => now - t <= 1000).length;
    const simulating = graph.physics.isActive;
    status.textContent = simulating ? 'simulating' : 'settled';
    status.classList.toggle('live', simulating);
    stats.textContent = fps === 0 ? 'idle · GPU asleep' : `${fps} fps · α ${graph.physics.alpha.toFixed(2)}`;
  }, 250);

  Object.assign(window, { graph, camera, rig, renderer, labels });
}
