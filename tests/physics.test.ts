import { describe, expect, it } from 'vitest';
import {
  GraphStore,
  ManyBodyForce,
  NodeFlag,
  O3DEngineAdapter,
  O3DRendererAdapter,
  PhysicsSimulator,
  SpatialTree,
  createDefaultForces,
  type DimensionMode,
} from '../src/index.js';
import { createMockEngine, rng } from './helpers/mockEngine.js';

function randomStore(N: number, E: number, mode: DimensionMode = '3d', seed = 11) {
  const renderer = new O3DRendererAdapter(new O3DEngineAdapter(createMockEngine()));
  const store = new GraphStore(renderer, {}, mode);
  const r = rng(seed);
  store.setData({
    nodes: Array.from({ length: N }, (_, i) => ({ id: i, mass: 1 + (i % 3) })),
    edges: Array.from({ length: E }, () => ({ source: (r() * N) | 0, target: (r() * N) | 0 })),
  });
  return store;
}

describe('SpatialTree', () => {
  for (const dims of [2, 3] as const) {
    it(`${dims}D: places every visible body exactly once inside its leaf and aggregates mass`, () => {
      const store = randomStore(2000, 0, dims === 2 ? '2d' : '3d');
      const p = store.getNodeTransforms();
      for (let i = 100; i < 150; i++) p.copyWithin(i * 4, 0, 3); // 51 coincident bodies
      store.setNodeFlag(5, NodeFlag.HIDDEN, true);
      const tree = new SpatialTree(dims);
      tree.build(store);

      const seen = new Uint8Array(2000);
      for (let c = 0; c < tree.cellCount; c++) {
        for (let j = tree.cellFirstBody[c]; j !== -1; j = tree.bodyNext[j]) {
          seen[j]++;
          const h = tree.cellHalfSize[c] * (1 + 1e-5) + 1e-6;
          for (let k = 0; k < dims; k++) expect(Math.abs(p[j * 4 + k] - tree.cellCenter[c * 3 + k])).toBeLessThanOrEqual(h);
        }
      }
      let mass = 0;
      for (let i = 0; i < 2000; i++) {
        expect(seen[i]).toBe(i === 5 ? 0 : 1);
        if (i !== 5) mass += store.nodeMasses[i];
      }
      expect(tree.cellMass[tree.root]).toBeCloseTo(mass, 2);
      expect(tree.bodyCount).toBe(1999);
      expect(tree.cellBodyCount[tree.root]).toBe(1999);
      // Every subtree is a contiguous range of bodyOrder.
      const order = tree.bodyOrder;
      for (let c = 0; c < tree.cellCount; c += 97) {
        const start = tree.cellOrderStart[c];
        for (let q = start; q < start + tree.cellBodyCount[c]; q++) {
          const o = order[q] * 4;
          for (let k = 0; k < dims; k++) {
            expect(Math.abs(p[o + k] - tree.cellCenter[c * 3 + k])).toBeLessThanOrEqual(tree.cellHalfSize[c] * (1 + 1e-5) + 1e-6);
          }
        }
      }
    });
  }
});

describe('ManyBodyForce', () => {
  it('Barnes-Hut (θ = 0.8) stays within 5% of the exact O(N²) field', () => {
    const store = randomStore(3000, 0);
    const tree = new SpatialTree(3);
    tree.build(store);
    const field = (theta: number) => {
      const forces = new Float32Array(3000 * 3);
      new ManyBodyForce({ theta }).apply({
        alpha: 1, dimensions: 3, nodeCount: 3000, edgeCount: 0,
        positions: store.getNodeTransforms(), velocities: store.nodeVelocities, forces, graph: store, tree,
      });
      return forces;
    };
    const exact = field(1e-9), approx = field(0.8);
    let num = 0, den = 0;
    for (let k = 0; k < exact.length; k++) {
      num += (exact[k] - approx[k]) ** 2;
      den += exact[k] ** 2;
    }
    expect(Math.sqrt(num / den)).toBeLessThan(0.05);
  });
});

describe('PhysicsSimulator', () => {
  function treeGraph(N = 500) {
    const renderer = new O3DRendererAdapter(new O3DEngineAdapter(createMockEngine()));
    const store = new GraphStore(renderer);
    const r = rng(5);
    store.setData({
      nodes: Array.from({ length: N }, (_, i) => ({ id: i })),
      edges: Array.from({ length: N - 1 }, (_, i) => ({ source: i + 1, target: (r() * (i + 1)) | 0 })),
    });
    const sim = new PhysicsSimulator(store);
    for (const f of createDefaultForces()) sim.registerForce(f);
    return { store, sim };
  }

  it('cools to rest on the d3 schedule: centered, finite, non-overlapping', () => {
    const { store, sim } = treeGraph();
    let ticks = 0;
    while (sim.tick()) ticks++;
    expect(ticks).toBeGreaterThan(250);
    expect(ticks).toBeLessThan(310);
    expect(sim.isActive).toBe(false);
    const p = store.getNodeTransforms();
    let cx = 0;
    for (let i = 0; i < store.nodeCount; i++) {
      for (let k = 0; k < 3; k++) expect(Number.isFinite(p[i * 4 + k])).toBe(true);
      cx += p[i * 4];
    }
    expect(Math.abs(cx / store.nodeCount)).toBeLessThan(1e-2);
    let overlaps = 0;
    for (let i = 0; i < store.nodeCount; i++)
      for (let j = i + 1; j < store.nodeCount; j++)
        if (Math.hypot(p[i * 4] - p[j * 4], p[i * 4 + 1] - p[j * 4 + 1], p[i * 4 + 2] - p[j * 4 + 2]) < 6.4) overlaps++;
    expect(overlaps).toBe(0);
  });

  it('keeps pinned nodes fixed while neighbors are pulled toward them', () => {
    const { store, sim } = treeGraph(100);
    while (sim.tick());
    const nb = store.edgeSources[store.nodeFirstHalfEdge[0] >> 1] === 0
      ? store.edgeTargets[store.nodeFirstHalfEdge[0] >> 1]
      : store.edgeSources[store.nodeFirstHalfEdge[0] >> 1];
    const before = store.getNodeTransforms()[nb * 4];
    store.setNodeFlag(0, NodeFlag.PINNED, true);
    store.setNodePosition(0, 300, 0, 0);
    sim.alphaTarget = 0.3;
    sim.reheat();
    for (let k = 0; k < 60; k++) sim.tick();
    expect(store.getNodeTransforms()[0]).toBe(300);
    expect(store.getNodeTransforms()[nb * 4]).toBeGreaterThan(before + 50);
  });

  it('holds Z = 0 in 2D and re-expands along Z after returning to 3D', () => {
    const { store, sim } = treeGraph(300);
    for (let k = 0; k < 100; k++) sim.tick();
    store.setDimensionMode('2d');
    sim.setDimensionMode('2d');
    expect(sim.tree.branching).toBe(4);
    for (let k = 0; k < 50; k++) sim.tick();
    for (let i = 0; i < store.nodeCount; i++) expect(store.getNodeTransforms()[i * 4 + 2]).toBe(0);
    store.setDimensionMode('3d');
    sim.setDimensionMode('3d');
    expect(sim.tree.branching).toBe(8);
    for (let k = 0; k < 150; k++) sim.tick();
    let zMax = 0;
    for (let i = 0; i < store.nodeCount; i++) zMax = Math.max(zMax, Math.abs(store.getNodeTransforms()[i * 4 + 2]));
    expect(zMax).toBeGreaterThan(10);
  });
});
