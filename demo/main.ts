import * as o3d from '@omernach/o3d';
import { NO_INDEX, NodeFlag, createGraphView, type NodeId } from '../src/index.js';
import { EDGE_ALPHA, TEAMS, createMockGraph, withAlpha, type LinkMeta, type PersonMeta } from './mockData.js';

/**
 * Demo: 1,000 mock people in 10 teams on the published @omernach/o3d
 * WebGPU renderer. `createGraphView` provides the graph, camera gestures,
 * labels, click-to-focus and the render loop; everything below is app UI.
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
  const message = err instanceof Error ? err.message : String(err);
  const webgpu = !('gpu' in navigator) || /webgpu/i.test(message);
  showFatal(
    webgpu ? 'WebGPU is not available' : 'Something went wrong',
    webgpu
      ? 'This demo renders with WebGPU. Open it in a recent Chrome or Edge (or Safari 26+), served from localhost or HTTPS.'
      : message,
  );
});

function showFatal(title: string, message: string): void {
  const fatal = $('fatal');
  fatal.innerHTML = `<div><h2>${escapeHtml(title)}</h2><p>${escapeHtml(message)}</p></div>`;
  fatal.hidden = false;
}

async function boot(): Promise<void> {
  // Portrait phones start in 2D: the orthographic view fits narrow screens best.
  const startMode = innerWidth < 640 && innerHeight > innerWidth ? '2d' : '3d';
  const dock = document.querySelector<HTMLElement>('.dock')!;
  const brand = document.querySelector<HTMLElement>('.brand')!;

  let focusId: NodeId | null = null;
  let near = new Uint8Array(0); // 1 = focused node or one of its neighbors

  const view = await createGraphView<PersonMeta, LinkMeta>(o3d, {
    canvas,
    dimension: startMode,
    node: { hoverColor: '#ffffff', selectedColor: '#ffffff' },
    // While someone is focused, only their neighborhood is labeled (the rest is dimmed).
    labels: {
      maxNodeLabels: 36,
      maxEdgeLabels: 14,
      nodeText: (i): string | undefined =>
        focusId !== null && near[i] === 0 ? undefined : view.graph.store.getNodeLabel(i),
    },
    // Keep the graph clear of the bottom dock (and the search card on phones).
    fitInsets: () => {
      const r = brand.getBoundingClientRect();
      return {
        top: r.width > innerWidth * 0.6 ? r.bottom + 8 : 0,
        bottom: innerHeight - dock.getBoundingClientRect().top + 8,
      };
    },
    onFocusChange: (id) => {
      focusId = id;
      paint();
      renderInspector();
    },
  });

  const { graph } = view;
  const store = graph.store;
  graph.setData(createMockGraph(NODE_COUNT));
  $('count-nodes').textContent = graph.nodeCount.toLocaleString();
  $('count-edges').textContent = graph.edgeCount.toLocaleString();

  // ---- Helpers -------------------------------------------------------------------------
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

  // ---- Focus mode: focused node + neighbors in color, the rest dimmed --------------------
  function paint(): void {
    const focus = focusId === null ? NO_INDEX : store.getNodeIndex(focusId);
    near = new Uint8Array(store.nodeCount);
    if (focus !== NO_INDEX) {
      near[focus] = 1;
      for (const j of neighborsOf(focus)) near[j] = 1;
    }
    if (view.labels) view.labels.edgeLabels = focus === NO_INDEX ? 'auto' : 'focused';
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

  // ---- Inspector ----------------------------------------------------------------------------
  const inspector = $('inspector');
  let degreeRank: Map<number, number> | null = null;
  const rankOf = (index: number) => {
    degreeRank ??= new Map(
      Array.from({ length: store.nodeCount }, (_, i) => i)
        .sort((a, b) => store.nodeDegrees[b] - store.nodeDegrees[a])
        .map((i, r) => [i, r + 1]),
    );
    return degreeRank.get(index) ?? store.nodeCount;
  };

  function renderInspector(): void {
    if (focusId === null) {
      inspector.hidden = true;
      return;
    }
    const i = store.getNodeIndex(focusId);
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
        <button type="button" data-action="zoom">Zoom to</button>
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

  inspector.addEventListener('click', (e) => {
    const target = (e.target as HTMLElement).closest('button');
    if (!target || focusId === null) return;
    if (target.dataset.id) view.focus(target.dataset.id, { zoom: true });
    else if (target.dataset.action === 'zoom') view.focus(focusId, { zoom: true });
    else if (target.dataset.action === 'pin') {
      graph.pin(focusId, !store.hasFlag(store.getNodeIndex(focusId), NodeFlag.PINNED));
      renderInspector();
    }
  });

  // ---- Tooltip --------------------------------------------------------------------------------
  const tooltip = $('tooltip');
  let pointer = { x: 0, y: 0 };
  const placeTooltip = () => {
    const pad = 14;
    const w = tooltip.offsetWidth;
    const h = tooltip.offsetHeight;
    const x = pointer.x + pad + w > innerWidth ? pointer.x - pad - w : pointer.x + pad;
    const y = pointer.y + pad + h > innerHeight ? pointer.y - pad - h : pointer.y + pad;
    tooltip.style.transform = `translate(${x}px, ${y}px)`;
  };
  canvas.addEventListener('pointermove', (e) => {
    pointer = { x: e.clientX, y: e.clientY };
    if (!tooltip.hidden) placeTooltip();
  });
  graph.on('node:hover', ({ index }) => {
    const m = meta(index);
    tooltip.innerHTML = `<strong>${escapeHtml(m.name)}</strong>
      <span class="muted"><span class="dot" style="display:inline-block;margin-right:6px;background:${css(TEAMS[m.team].color)}"></span>${TEAMS[m.team].name} · ${store.nodeDegrees[index]} links</span>`;
    tooltip.hidden = false;
    placeTooltip();
  });
  graph.on('node:leave', () => (tooltip.hidden = true));
  graph.on('node:dragstart', () => (tooltip.hidden = true));

  // ---- Toolbar --------------------------------------------------------------------------------
  const mode3d = $<HTMLButtonElement>('mode-3d');
  const mode2d = $<HTMLButtonElement>('mode-2d');
  const rotateBtn = $<HTMLButtonElement>('btn-rotate');
  const labelsBtn = $<HTMLButtonElement>('btn-labels');

  function setMode(mode: '2d' | '3d'): void {
    view.setDimensionMode(mode);
    mode3d.setAttribute('aria-pressed', String(mode === '3d'));
    mode2d.setAttribute('aria-pressed', String(mode === '2d'));
    rotateBtn.disabled = mode === '2d';
    if (mode === '2d') setRotate(false);
  }
  function setRotate(on: boolean): void {
    if (!view.camera) return;
    view.camera.autoRotate = on && graph.dimensionMode === '3d';
    rotateBtn.setAttribute('aria-pressed', String(view.camera.autoRotate));
    view.requestRender();
  }
  function setLabels(on: boolean): void {
    if (!view.labels) return;
    view.labels.enabled = on;
    labelsBtn.setAttribute('aria-pressed', String(on));
    view.requestRender();
  }
  setMode(startMode);
  mode3d.addEventListener('click', () => setMode('3d'));
  mode2d.addEventListener('click', () => setMode('2d'));
  $('btn-fit').addEventListener('click', () => view.fit());
  $('btn-reheat').addEventListener('click', () => graph.reheat(0.7));
  rotateBtn.addEventListener('click', () => setRotate(!view.camera?.autoRotate));
  labelsBtn.addEventListener('click', () => setLabels(!view.labels?.enabled));

  // ---- Team legend (hide / show) --------------------------------------------------------------
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
    if (hide && focusId !== null && meta(store.getNodeIndex(focusId)).team === team) view.clearFocus();
  });

  // ---- Search ---------------------------------------------------------------------------------
  const search = $<HTMLInputElement>('search');
  const results = $<HTMLUListElement>('results');
  let matches: number[] = [];
  let active = 0;
  const renderResults = () => {
    results.innerHTML = matches
      .map((i, k) => {
        const m = meta(i);
        return `<li><button type="button" role="option" data-index="${i}" aria-selected="${k === active}">
          <span class="dot" style="background:${css(TEAMS[m.team].color)}"></span>${escapeHtml(m.name)}
          <span class="sub">${TEAMS[m.team].name}</span></button></li>`;
      })
      .join('');
    results.hidden = matches.length === 0;
  };
  const choose = (index: number) => {
    search.value = meta(index).name;
    results.hidden = true;
    search.blur();
    view.focus(store.getNodeId(index), { zoom: true });
  };
  search.addEventListener('input', () => {
    const q = search.value.trim().toLowerCase();
    matches = [];
    for (let i = 0; q && i < store.nodeCount && matches.length < 8; i++) {
      if (!hiddenTeams.has(meta(i).team) && meta(i).name.toLowerCase().includes(q)) matches.push(i);
    }
    active = 0;
    renderResults();
  });
  search.addEventListener('keydown', (e) => {
    if ((e.key === 'ArrowDown' || e.key === 'ArrowUp') && matches.length > 0) {
      e.preventDefault();
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
  results.addEventListener('pointerdown', (e) => e.preventDefault());
  results.addEventListener('click', (e) => {
    const b = (e.target as HTMLElement).closest<HTMLButtonElement>('button[data-index]');
    if (b) choose(Number(b.dataset.index));
  });

  // ---- Keyboard -------------------------------------------------------------------------------
  addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === 'escape') view.clearFocus();
    else if (k === 'f') view.fit();
    else if (k === 'r') graph.reheat(0.7);
    else if (k === 'a') setRotate(!view.camera?.autoRotate);
    else if (k === 'l') setLabels(!view.labels?.enabled);
    else if (k === '2') setMode('2d');
    else if (k === '3') setMode('3d');
    else if (k === '/') {
      e.preventDefault();
      search.focus();
    }
  });
  addEventListener('resize', () => {
    if (focusId === null && !view.camera?.isInteracting) view.fit();
  });

  // ---- Stats (4 Hz) ---------------------------------------------------------------------------
  const status = $('status');
  const stats = $('stats');
  setInterval(() => {
    const simulating = graph.physics.isActive;
    status.textContent = simulating ? 'simulating' : 'settled';
    status.classList.toggle('live', simulating);
    const fps = view.fps;
    stats.textContent = fps === 0 ? 'idle · GPU asleep' : `${fps} fps · α ${graph.physics.alpha.toFixed(2)}`;
  }, 250);

  // Handy for poking around in devtools (and used by the browser tests).
  Object.assign(window, { view, graph, labels: view.labels, camera: view.cameraController, rig: view.camera, renderer: view.renderer });
}
