import { describe, expect, it } from 'vitest';
import { ContainForce, GraphStore, NodeFlag, O3DEngineAdapter, O3DRendererAdapter, PhysicsSimulator, createDefaultForces } from '../src/index.js';
import { createMockEngine, rng } from './helpers/mockEngine.js';

function layout(n: number, radius: number) {
  const store = new GraphStore(new O3DRendererAdapter(new O3DEngineAdapter(createMockEngine())), { defaultLinkDistance: 8, defaultNodeRadius: 2 });
  const r = rng(9);
  store.setData({
    nodes: Array.from({ length: n }, (_, i) => ({ id: i })),
    edges: Array.from({ length: n - 1 }, (_, i) => ({ source: i + 1, target: (r() * (i + 1)) | 0 })),
  });
  const sim = new PhysicsSimulator(store, { chargeStrength: -6 });
  for (const f of createDefaultForces({ chargeStrength: -6, collidePadding: 0.6 })) sim.registerForce(f);
  const contain = new ContainForce({ radius });
  sim.registerForce(contain);
  return { store, sim, contain };
}

const maxRadius = (store: GraphStore) => {
  const p = store.getNodeTransforms();
  let m = 0;
  for (let i = 0; i < store.nodeCount; i++) {
    if (store.hasFlag(i, NodeFlag.HIDDEN)) continue;
    m = Math.max(m, Math.hypot(p[i * 4], p[i * 4 + 1], p[i * 4 + 2]));
  }
  return m;
};

describe('ContainForce', () => {
  it.each([300, 2500])('keeps a %i-node layout at the target radius', (n) => {
    const { store, sim, contain } = layout(n, 120);
    while (sim.tick());
    const r = maxRadius(store);
    expect(r).toBeGreaterThan(120 * 0.85);
    expect(r).toBeLessThan(120 * 1.15);
    expect(contain.strength).toBeGreaterThan(0);
  });

  it('ignores pinned and hidden nodes far outside the radius', () => {
    const { store, sim } = layout(300, 120);
    store.addNode({ id: 'pinned', x: 5000, y: 0, z: 0, pinned: true });
    store.addNode({ id: 'hidden', x: 0, y: 5000, z: 0, hidden: true });
    while (sim.tick());
    expect(store.getNodeTransforms()[store.getNodeIndex('pinned') * 4]).toBe(5000);
    expect(maxRadius(store)).toBeGreaterThan(4000); // the pinned node stays where it is
  });
});
