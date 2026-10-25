import { describe, expect, it } from 'vitest';
import { EventBus, GraphStore, NO_INDEX, NodeFlag, O3DEngineAdapter, O3DRendererAdapter } from '../src/index.js';
import { createMockEngine, rng } from './helpers/mockEngine.js';

function setup() {
  const engine = createMockEngine();
  const renderer = new O3DRendererAdapter(new O3DEngineAdapter(engine), { initialNodeCapacity: 16, initialEdgeCapacity: 16 });
  const store = new GraphStore(renderer, { initialNodeCapacity: 16, initialEdgeCapacity: 16 });
  return { engine, renderer, store };
}

/** Adjacency, degree, identity and edge-vertex invariants. */
function checkIntegrity({ store, renderer }: ReturnType<typeof setup>) {
  const N = store.nodeCount, E = store.edgeCount;
  const seen = new Uint8Array(E * 2);
  for (let i = 0; i < N; i++) {
    expect(store.getNodeIndex(store.getNodeId(i))).toBe(i);
    let prev = NO_INDEX, deg = 0;
    for (let h = store.nodeFirstHalfEdge[i]; h !== NO_INDEX; h = store.halfEdgeNext[h]) {
      expect(store.halfEdgePrev[h]).toBe(prev);
      prev = h;
      expect(h >> 1).toBeLessThan(E);
      expect(seen[h]).toBe(0);
      seen[h] = 1;
      expect((h & 1) ? store.edgeTargets[h >> 1] : store.edgeSources[h >> 1]).toBe(i);
      deg++;
    }
    expect(store.nodeDegrees[i]).toBe(deg);
  }
  expect(seen.every((x) => x === 1)).toBe(true);
  const p = store.getNodeTransforms(), v = renderer.getEdgeVertices();
  for (let e = 0; e < E; e++) {
    const s = store.edgeSources[e] * 4, t = store.edgeTargets[e] * 4;
    for (let k = 0; k < 3; k++) {
      expect(v[e * 6 + k]).toBe(p[s + k]);
      expect(v[e * 6 + 3 + k]).toBe(p[t + k]);
    }
  }
}

describe('GraphStore', () => {
  it('bulk loads with reallocation + memory growth and keeps invariants', () => {
    const ctx = setup();
    const r = rng();
    const N = 3000;
    ctx.store.setData({
      nodes: Array.from({ length: N }, (_, i) => ({ id: 'n' + i, color: '#ff8800' })),
      edges: [
        ...Array.from({ length: 6000 }, () => ({ source: 'n' + ((r() * N) | 0), target: 'n' + ((r() * N) | 0) })),
        { source: 'n5', target: 'n5' },
      ],
    });
    ctx.store.flush();
    expect(ctx.store.nodeCount).toBe(N);
    expect(ctx.renderer.getNodeColors()[0]).toBe(0xff8800ff);
    checkIntegrity(ctx);
  });

  it('survives random interleaved swap-and-pop removals and insertions', () => {
    const ctx = setup();
    const r = rng(3);
    ctx.store.setData({
      nodes: Array.from({ length: 500 }, (_, i) => ({ id: i })),
      edges: Array.from({ length: 1500 }, () => ({ source: (r() * 500) | 0, target: (r() * 500) | 0 })),
    });
    for (let k = 0; k < 600; k++) {
      if (r() < 0.5) ctx.store.removeNode(ctx.store.getNodeId((r() * ctx.store.nodeCount) | 0));
      else if (ctx.store.edgeCount) ctx.store.removeEdge(ctx.store.getEdgeId((r() * ctx.store.edgeCount) | 0));
      if (r() < 0.3) {
        ctx.store.addNode({ id: 'x' + k, x: k, y: -k, z: 1 });
        ctx.store.addEdge({ source: 'x' + k, target: ctx.store.getNodeId((r() * ctx.store.nodeCount) | 0) });
      }
    }
    checkIntegrity(ctx);
  });

  it('rebinds views transparently after WASM memory growth', () => {
    const ctx = setup();
    ctx.store.setData({ nodes: [{ id: 'a' }, { id: 'b' }], edges: [{ source: 'a', target: 'b' }] });
    const before = ctx.renderer.viewVersion;
    ctx.engine.alloc(4 << 20);
    ctx.store.setNodePosition(0, 1, 2, 3);
    expect(ctx.renderer.viewVersion).toBeGreaterThan(before);
    expect(Array.from(ctx.renderer.getEdgeVertices().subarray(0, 3))).toEqual([1, 2, 3]);
  });

  it('uploads only the dirty sub-range after a single-node move', () => {
    const ctx = setup();
    ctx.store.setData({
      nodes: Array.from({ length: 20 }, (_, i) => ({ id: i })),
      edges: [{ source: 10, target: 11 }, { source: 1, target: 2 }],
    });
    ctx.store.flush();
    ctx.engine.calls.commits.length = 0;
    ctx.store.setNodePosition(10, 5, 5, 5);
    ctx.store.flush();
    expect(ctx.engine.calls.commits.map((c) => c[0])).toEqual(['meshRange', 'lineRange']);
    expect(ctx.engine.calls.commits[0].slice(2)).toEqual([10, 1]);
    expect(ctx.engine.calls.commits[1].slice(2)).toEqual([0, 1]);
  });

  it('flattens in 2D and injects bounded deterministic Z jitter back in 3D', () => {
    const ctx = setup();
    ctx.store.setData({ nodes: Array.from({ length: 50 }, (_, i) => ({ id: i })) });
    ctx.store.setDimensionMode('2d');
    const p = ctx.store.getNodeTransforms();
    for (let i = 0; i < 50; i++) expect(p[i * 4 + 2]).toBe(0);
    ctx.store.setDimensionMode('3d');
    const z = Array.from({ length: 50 }, (_, i) => ctx.store.getNodeTransforms()[i * 4 + 2]);
    expect(z.some((v) => v !== 0)).toBe(true);
    expect(z.every((v) => v >= -5 && v <= 5)).toBe(true);
  });

  it('hides nodes at scale 0, collapses their edges and drives highlight colors from flags', () => {
    const ctx = setup();
    ctx.store.setData({ nodes: [{ id: 'a' }, { id: 'b', x: 50, y: 0, z: 0 }], edges: [{ source: 'a', target: 'b' }] });
    ctx.store.setNodeFlag(1, NodeFlag.HIDDEN, true);
    expect(ctx.store.getNodeTransforms()[7]).toBe(0);
    const v = ctx.renderer.getEdgeVertices();
    expect([v[3], v[4], v[5]]).toEqual([v[0], v[1], v[2]]);
    ctx.store.setNodeFlag(0, NodeFlag.HOVERED, true);
    expect(ctx.renderer.getNodeColors()[0]).toBe(0xffc857ff);
    ctx.store.setNodeFlag(0, NodeFlag.SELECTED, true);
    expect(ctx.renderer.getNodeColors()[0]).toBe(0xff5d73ff);
    ctx.store.setNodeFlag(1, NodeFlag.HIDDEN, false);
    expect(ctx.store.getNodeTransforms()[7]).toBe(4);
  });

  it('rejects duplicate ids and dangling edges', () => {
    const { store } = setup();
    store.addNode({ id: 'a' });
    expect(() => store.addNode({ id: 'a' })).toThrow(/duplicate/);
    expect(() => store.addEdge({ source: 'a', target: 'zzz' })).toThrow(/not found/);
  });
});

describe('EventBus', () => {
  it('supports once, unsubscribing during emit, and isolates throwing listeners', async () => {
    const bus = new EventBus<{ a: number }>();
    const log: string[] = [];
    const off = bus.on('a', (x) => {
      log.push('b' + x);
      off();
    });
    bus.once('a', (x) => log.push('o' + x));
    bus.on('a', () => {
      throw new Error('boom');
    });
    bus.on('a', (x) => log.push('c' + x));
    const errors: unknown[] = [];
    const onErr = (e: unknown) => errors.push(e);
    process.on('uncaughtException', onErr);
    bus.emit('a', 1);
    bus.emit('a', 2);
    await new Promise((r) => setTimeout(r, 0));
    process.off('uncaughtException', onErr);
    expect(log).toEqual(['b1', 'o1', 'c1', 'c2']);
    expect(bus.listenerCount('a')).toBe(2);
    expect(errors).toHaveLength(2);
  });
});
