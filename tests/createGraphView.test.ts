import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NodeFlag, createGraphView } from '../src/index.js';
import { FakeCanvas } from './helpers/mockEngine.js';
import { createFakeO3DModule, fakeLog } from './helpers/fakeO3D.js';

/** Manually stepped requestAnimationFrame. */
const frames: Array<(t: number) => void> = [];
let clock = 0;
function step(n = 1): number {
  let k = 0;
  for (; k < n && frames.length > 0; k++) {
    clock += 16;
    frames.shift()!(clock);
  }
  return k;
}

function overlayCanvas(): HTMLCanvasElement {
  const ctx = new Proxy({ measureText: (t: string) => ({ width: t.length * 6 }) } as Record<string, unknown>, {
    get: (target, key) => (key in target ? target[key as string] : () => {}),
    set: () => true,
  });
  return { width: 0, height: 0, getContext: () => ctx, remove() {} } as unknown as HTMLCanvasElement;
}

beforeEach(() => {
  frames.length = 0;
  vi.stubGlobal('requestAnimationFrame', (cb: (t: number) => void) => frames.push(cb));
  vi.stubGlobal('cancelAnimationFrame', () => void (frames.length = 0));
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} });
});
afterEach(() => vi.unstubAllGlobals());

async function setup(options: Record<string, unknown> = {}) {
  const { o3d, created } = createFakeO3DModule();
  const canvas = new FakeCanvas();
  const view = await createGraphView(o3d, {
    canvas: canvas as unknown as HTMLCanvasElement,
    labels: { canvas: overlayCanvas() },
    ...options,
  });
  const nodes = Array.from({ length: 200 }, (_, i) => ({ id: `n${i}`, label: `Node ${i}` }));
  const edges = Array.from({ length: 199 }, (_, i) => ({ source: `n${i + 1}`, target: `n${i >> 1}` }));
  return { view, canvas, created, nodes, edges, camera: () => created.cameras[0] };
}

describe('createGraphView', () => {
  it('wires o3d, the graph, camera, labels and an on-demand render loop', async () => {
    const { view, created, nodes, edges, camera } = await setup();
    expect(created.batchRenderers).toBe(1);
    expect(view.graph.getForce('contain')).toBeDefined();
    expect(view.labels).not.toBeNull();

    const before = fakeLog.renders;
    view.graph.setData({ nodes, edges });
    step(3000);
    expect(fakeLog.renders).toBeGreaterThan(before);
    expect(frames).toHaveLength(0); // settled → loop asleep
    expect(view.graph.physics.isActive).toBe(false);

    // Auto-fit framed the layout around its centroid.
    const p = view.graph.store.getNodeTransforms();
    let cx = 0;
    for (let i = 0; i < 200; i++) cx += p[i * 4];
    expect(Math.abs(camera().target[0] - cx / 200)).toBeLessThan(1);
  });

  it('focus selects a node and keeps it as the orbit center; clicks toggle focus', async () => {
    const { view, nodes, edges, camera } = await setup();
    view.graph.setData({ nodes, edges });
    step(3000);

    view.focus('n7', { zoom: true });
    step(200);
    const i = view.graph.store.getNodeIndex('n7');
    expect(view.graph.store.hasFlag(i, NodeFlag.SELECTED)).toBe(true);
    expect(view.focusedId).toBe('n7');
    const p = view.graph.getNodePosition('n7')!;
    expect(camera().target.map((v) => Math.round(v * 100) / 100)).toEqual(Array.from(p).map((v) => Math.round(v * 100) / 100));

    // node:click on the focused node clears focus (focusOnClick).
    view.graph.events.emit('node:click', { id: 'n7', index: i, meta: undefined, event: null });
    expect(view.focusedId).toBeNull();
    expect(view.graph.store.hasFlag(i, NodeFlag.SELECTED)).toBe(false);
    view.graph.events.emit('node:click', { id: 'n3', index: view.graph.store.getNodeIndex('n3'), meta: undefined, event: null });
    expect(view.focusedId).toBe('n3');
  });

  it('switches dimension mode for graph and camera', async () => {
    const { view, nodes, edges, camera } = await setup();
    view.graph.setData({ nodes, edges });
    step(3000);
    view.setDimensionMode('2d');
    step(3000);
    expect(camera().mode).toBe('2d');
    const p = view.graph.store.getNodeTransforms();
    for (let i = 0; i < 200; i++) expect(p[i * 4 + 2]).toBe(0);
  });

  it('honors labels: false / cameraControls: false', async () => {
    const { view } = await setup({ labels: false, cameraControls: false });
    expect(view.labels).toBeNull();
    expect(view.camera).toBeNull();
  });

  it('dispose frees GPU/WASM resources and removes every listener', async () => {
    const { view, canvas, created, nodes, edges } = await setup();
    view.graph.setData({ nodes, edges });
    step(10);
    view.dispose();
    expect(canvas.listenerCount()).toBe(0);
    expect(created.batchDestroyed).toBe(1);
    expect(created.meshDestroyed).toBe(1);
    expect(created.cameras[0].destroyed).toBe(true);
    view.requestRender();
    expect(frames).toHaveLength(0);
  });
});
