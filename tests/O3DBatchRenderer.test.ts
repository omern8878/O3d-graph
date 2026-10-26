import { describe, expect, it } from 'vitest';
import { O3DBatchRenderer, O3DGraph, encodeColorLayout, packRGBA } from '../src/index.js';
import { FakeCanvas, cameraRay, createManualScheduler, project } from './helpers/mockEngine.js';
import { FakeInstancedMesh, FakeLineBatch, fakeLog as log } from './helpers/fakeO3D.js';

function fakeCamera() {
  const calls: string[] = [];
  return {
    calls,
    orbit: true,
    getForward: () => [0, 0, -1] as [number, number, number],
    unprojectNdc: (x: number, y: number) => {
      const r = cameraRay(x, y);
      return { origin: [r[0], r[1], r[2]] as [number, number, number], direction: [r[3], r[4], r[5]] as [number, number, number] };
    },
    getViewProjection: () => new Float32Array(16),
    setOrbitEnabled(e: boolean) { this.orbit = e; calls.push(`orbit:${e}`); },
    setMode: (m: '2d' | '3d') => void calls.push(`mode:${m}`),
    lockToXYPlane: (l: boolean) => void calls.push(`lock:${l}`),
  };
}

function setup(nodeCap = 4, edgeCap = 4) {
  const camera = fakeCamera();
  let changes = 0;
  const renderer = new O3DBatchRenderer({
    o3d: { InstancedMesh: FakeInstancedMesh, LineBatch: FakeLineBatch },
    device: {},
    nodeMesh: { boundingRadius: 1 },
    camera,
    initialNodeCapacity: nodeCap,
    initialEdgeCapacity: edgeCap,
    onChange: () => changes++,
  });
  return { renderer, camera, changes: () => changes };
}

describe('encodeColorLayout', () => {
  it('byte-swaps into o3d layout (red in the lowest byte) and is its own inverse', () => {
    const rgba = packRGBA(0x11, 0x22, 0x33, 0x44);
    expect(rgba).toBe(0x11223344);
    expect(encodeColorLayout(rgba, 'abgr')).toBe(0x44332211);
    expect(encodeColorLayout(encodeColorLayout(rgba, 'abgr'), 'abgr')).toBe(rgba);
    expect(encodeColorLayout(rgba, 'rgba')).toBe(rgba);
  });
});

describe('O3DBatchRenderer', () => {
  it('drives a full O3DGraph: colors in o3d layout, dirty ranges, growth, picking, drag', () => {
    const { renderer, camera, changes } = setup();
    const canvas = new FakeCanvas();
    const sched = createManualScheduler();
    const graph = new O3DGraph(renderer, { canvas: canvas as unknown as HTMLCanvasElement, node: { radius: 6 } }, { scheduler: sched });

    // 40 nodes / 39 edges from capacity 16 (nextCapacity floor): forces several reallocations.
    graph.setData({
      nodes: Array.from({ length: 40 }, (_, i) => ({ id: i, color: '#ff8800' })),
      edges: Array.from({ length: 39 }, (_, i) => ({ source: i + 1, target: i >> 1, color: 0x11223344 })),
    });
    expect(renderer.nodeCapacity).toBeGreaterThanOrEqual(40);
    expect(renderer.scene.meshes).toHaveLength(1);
    expect(log.destroyed).toBeGreaterThan(0);
    expect(renderer.getNodeColors()[0]).toBe(0xff0088ff); // 0xff8800ff byte-swapped
    expect(renderer.getEdgeColors()[0]).toBe(0x44332211);

    sched.step(2000);
    expect(changes()).toBeGreaterThan(0);
    // Per-frame commits upload transforms + colors only; rotations stay identity and are never resent.
    expect(log.dirty).toContain('mesh 0..39 transform+color');
    expect(renderer.scene.meshes[0].getCount()).toBe(40);
    expect(renderer.scene.lines[0].getCount()).toBe(39);

    // Edge vertices in the o3d line buffer track node transforms.
    const p = renderer.getNodeTransforms(), v = renderer.getEdgeVertices();
    expect([v[0], v[1], v[2]]).toEqual([p[4], p[5], p[6]]); // edge 0 source = node 1

    // Hover + drag through the o3d raycast and camera ray.
    const at = graph.getNodePosition(7)!;
    const ndc = project(at[0], at[1], at[2]);
    canvas.fire('pointermove', ndc);
    sched.step();
    expect(graph.interaction!.hoveredNode).toBe(graph.store.getNodeIndex(7));
    canvas.fire('pointerdown', ndc);
    expect(camera.orbit).toBe(false);
    const dest: [number, number] = [ndc[0] + 0.2, ndc[1]];
    for (let k = 0; k < 30; k++) {
      canvas.fire('pointermove', dest);
      sched.step();
    }
    const after = graph.getNodePosition(7)!;
    const [x, y] = project(after[0], after[1], after[2]);
    expect(Math.abs(x - dest[0])).toBeLessThan(1e-3);
    expect(Math.abs(y - dest[1])).toBeLessThan(1e-3);
    canvas.fire('pointerup', dest);
    expect(camera.orbit).toBe(true);

    graph.setDimensionMode('2d');
    expect(camera.calls).toContain('mode:2d');
    expect(camera.calls).toContain('lock:true');
    graph.dispose();
  });
});
