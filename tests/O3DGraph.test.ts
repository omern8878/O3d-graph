import { beforeEach, describe, expect, it } from 'vitest';
import { NodeFlag, O3DGraph } from '../src/index.js';
import { FakeCanvas, createManualScheduler, createMockEngine, project, type MockEngine } from './helpers/mockEngine.js';

type Meta = { i: number };

let engine: MockEngine;
let canvas: FakeCanvas;
let sched: ReturnType<typeof createManualScheduler>;
let graph: O3DGraph<Meta>;
let events: Array<[string, unknown]>;

const ndcOf = (id: string): [number, number] => {
  const p = graph.getNodePosition(id)!;
  return project(p[0], p[1], p[2]);
};
const flag = (id: string, f: number) => graph.store.hasFlag(graph.store.getNodeIndex(id), f as never);

beforeEach(() => {
  engine = createMockEngine();
  canvas = new FakeCanvas();
  sched = createManualScheduler();
  graph = new O3DGraph<Meta>(
    engine,
    { canvas: canvas as unknown as HTMLCanvasElement, node: { radius: 6 }, initialNodeCapacity: 16, initialEdgeCapacity: 16 },
    { scheduler: sched },
  );
  events = [];
  for (const t of ['node:hover', 'node:leave', 'node:click', 'node:dragstart', 'node:drag', 'node:dragend', 'engine:stabilized', 'mode:change'] as const) {
    graph.on(t, (p) => events.push([t, 'id' in p ? p.id : 'mode' in p ? p.mode : p.alpha]));
  }
  const N = 200;
  graph.setData({
    nodes: Array.from({ length: N }, (_, i) => ({ id: 'n' + i, meta: { i } })),
    edges: Array.from({ length: N - 1 }, (_, i) => ({ source: 'n' + (i + 1), target: 'n' + (i >> 1) })),
  });
  sched.step(5000);
});

describe('O3DGraph loop', () => {
  it('stabilizes once and then stops scheduling frames (idle = 0% CPU)', () => {
    expect(sched.queue).toHaveLength(0);
    expect(events.filter((e) => e[0] === 'engine:stabilized')).toHaveLength(1);
    graph.reheat();
    expect(sched.queue).toHaveLength(1);
  });
});

describe('InteractionController', () => {
  it('hovers via WASM raycast, coalesced into the next frame', () => {
    canvas.fire('pointermove', ndcOf('n37'));
    expect(sched.queue).toHaveLength(1);
    sched.step();
    expect(events.at(-1)).toEqual(['node:hover', 'n37']);
    expect(flag('n37', NodeFlag.HOVERED)).toBe(true);
    expect(canvas.style.cursor).toBe('pointer');
    canvas.fire('pointermove', [0.99, 0.99]);
    sched.step();
    expect(events.at(-1)).toEqual(['node:leave', 'n37']);
  });

  it('treats a still press as a click without reheating or leaving the node pinned', () => {
    const ndc = ndcOf('n37');
    expect(canvas.fire('pointerdown', ndc).stopped).toBe(true);
    expect(engine.calls.orbit.at(-1)).toBe(false);
    canvas.fire('pointerup', ndc);
    sched.step(5);
    expect(events.at(-1)).toEqual(['node:click', 'n37']);
    expect(graph.physics.isActive).toBe(false);
    expect(flag('n37', NodeFlag.PINNED)).toBe(false);
    expect(engine.calls.orbit.at(-1)).toBe(true);
  });

  it('drags along the camera plane: node stays under the cursor and neighbors follow', () => {
    const ndc = ndcOf('n37');
    const z0 = graph.getNodePosition('n37')![2];
    const neighbor0 = [...graph.getNodePosition('n18')!];
    canvas.fire('pointerdown', ndc);
    const dest: [number, number] = [ndc[0] + 0.3, ndc[1] - 0.2];
    for (let k = 1; k <= 10; k++) {
      canvas.fire('pointermove', [ndc[0] + 0.03 * k, ndc[1] - 0.02 * k]);
      sched.step();
    }
    expect(events.some((e) => e[0] === 'node:dragstart')).toBe(true);
    expect(graph.physics.alphaTarget).toBe(0.3);
    expect(flag('n37', NodeFlag.PINNED)).toBe(true);
    expect(canvas.style.cursor).toBe('grabbing');
    for (let k = 0; k < 40; k++) {
      canvas.fire('pointermove', dest);
      sched.step();
    }
    const [x, y] = ndcOf('n37');
    expect(Math.abs(x - dest[0])).toBeLessThan(1e-3);
    expect(Math.abs(y - dest[1])).toBeLessThan(1e-3);
    expect(graph.getNodePosition('n37')![2]).toBeCloseTo(z0, 3); // plane ⟂ camera forward
    const n1 = graph.getNodePosition('n18')!;
    expect(Math.hypot(n1[0] - neighbor0[0], n1[1] - neighbor0[1])).toBeGreaterThan(20);

    canvas.fire('pointerup', dest);
    sched.step();
    expect(events.at(-1)![0]).toBe('node:dragend');
    expect(graph.physics.alphaTarget).toBe(0);
    expect(flag('n37', NodeFlag.PINNED)).toBe(false);
    expect(engine.calls.orbit.at(-1)).toBe(true);
  });

  it('static drag (reheatOnDrag: false) rewrites only the k incident edges', () => {
    graph.interaction!.options.reheatOnDrag = false;
    const ndc = ndcOf('n37');
    const before = graph.renderer.getEdgeVertices().slice(0, graph.edgeCount * 6);
    canvas.fire('pointerdown', ndc);
    canvas.fire('pointermove', [ndc[0] + 0.1, ndc[1]]);
    sched.step();
    canvas.fire('pointerup', [ndc[0] + 0.1, ndc[1]]);
    sched.step();
    const after = graph.renderer.getEdgeVertices();
    let changed = 0;
    for (let e = 0; e < graph.edgeCount; e++) {
      for (let k = 0; k < 6; k++) {
        if (after[e * 6 + k] !== before[e * 6 + k]) {
          changed++;
          break;
        }
      }
    }
    expect(graph.physics.isActive).toBe(false);
    expect(changed).toBe(graph.store.nodeDegrees[graph.store.getNodeIndex('n37')]);
  });

  it('pinOnDragEnd keeps the dropped node pinned', () => {
    graph.interaction!.options.pinOnDragEnd = true;
    const ndc = ndcOf('n12');
    canvas.fire('pointerdown', ndc);
    canvas.fire('pointermove', [ndc[0] + 0.1, ndc[1]]);
    sched.step();
    canvas.fire('pointerup', [ndc[0] + 0.1, ndc[1]]);
    expect(flag('n12', NodeFlag.PINNED)).toBe(true);
  });

  it('survives removal of the hovered and the dragged node', () => {
    canvas.fire('pointermove', ndcOf('n5'));
    sched.step();
    graph.removeNode('n5');
    canvas.fire('pointermove', [0.99, 0.99]);
    sched.step();
    const ndc = ndcOf('n9');
    canvas.fire('pointerdown', ndc);
    canvas.fire('pointermove', [ndc[0] + 0.1, ndc[1]]);
    sched.step();
    graph.removeNode('n9');
    canvas.fire('pointermove', [ndc[0] + 0.2, ndc[1]]);
    expect(() => sched.step(3)).not.toThrow();
    expect(engine.calls.orbit.at(-1)).toBe(true);
  });
});

describe('O3DGraph modes & lifecycle', () => {
  it('switches to 2D: flat layout, 2D camera, mode:change event', () => {
    graph.setDimensionMode('2d');
    sched.step(500);
    for (let i = 0; i < graph.nodeCount; i++) expect(graph.store.getNodeTransforms()[i * 4 + 2]).toBe(0);
    expect(engine.calls.camera2D.at(-1)).toBe(true);
    expect(events).toContainEqual(['mode:change', '2d']);
  });

  it('dispose removes every listener and rejects further use', () => {
    graph.dispose();
    expect(canvas.listenerCount()).toBe(0);
    expect(() => graph.addNode({ id: 'x' })).toThrow(/dispose/);
  });
});
