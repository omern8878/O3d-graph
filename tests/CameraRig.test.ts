import { describe, expect, it } from 'vitest';
import { CameraRig } from '../src/index.js';
import { FakeCanvas } from './helpers/mockEngine.js';
import { FakeCameraController } from './helpers/fakeO3D.js';

function setup(options = {}) {
  const camera = new FakeCameraController({ mode: '3d', position: { x: 0, y: 0, z: 600 } });
  const canvas = new FakeCanvas();
  const rig = new CameraRig(camera, canvas as unknown as HTMLCanvasElement, options);
  const settle = () => { for (let k = 0; k < 120; k++) rig.tick(16); };
  return { camera, canvas, rig, settle };
}
const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

describe('CameraRig', () => {
  it('fits a sphere, but never past the far plane', () => {
    const { camera, rig, settle } = setup();
    rig.fit([10, 20, 30], 100);
    settle();
    expect(camera.target).toEqual([10, 20, 30]);
    expect(dist(camera.eye, camera.target)).toBeCloseTo((100 / Math.tan(Math.PI / 8)) * 1.06, 0);

    rig.fit([0, 0, 0], 600); // would need ~1500 units of distance
    settle();
    expect(dist(camera.eye, camera.target) + 600).toBeLessThanOrEqual(1000);
  });

  it('far: Infinity removes the limit', () => {
    const { camera, rig, settle } = setup({ far: Infinity });
    rig.fit([0, 0, 0], 600);
    settle();
    expect(dist(camera.eye, camera.target)).toBeGreaterThan(1400);
  });

  it('glides to a followed node, then tracks it exactly as it moves', () => {
    const { camera, rig, settle } = setup();
    const node: [number, number, number] = [50, -20, 10];
    rig.followNode(() => [...node]);
    settle();
    expect(camera.target).toEqual(node);
    node[0] = 80; // the node moves (physics)
    rig.tick(16);
    expect(camera.target).toEqual([80, -20, 10]);
    expect(rig.isFollowing).toBe(true);
  });

  it('holds still while paused (node drag) and stops following when the node disappears', () => {
    const { camera, rig, settle } = setup();
    let node: [number, number, number] | null = [5, 5, 5];
    rig.followNode(() => node);
    settle();
    rig.followPaused = true;
    node = [100, 100, 100];
    const before = [...camera.eye];
    rig.tick(16);
    expect(camera.eye).toEqual(before);
    rig.followPaused = false;
    node = null;
    rig.tick(16);
    expect(rig.isFollowing).toBe(false);
  });

  it('zoom keeps the anchor point fixed and panning releases follow', () => {
    const { camera, canvas, rig, settle } = setup();
    rig.fit([0, 0, 0], 100);
    settle();
    const anchor: [number, number, number] = [40, 0, 0];
    rig.zoomAnchor = () => anchor;
    const before = dist(camera.eye, camera.target);
    const handlers = (canvas as unknown as { handlers: Record<string, Array<(e: unknown) => void>> }).handlers;
    for (const h of handlers.wheel ?? []) h({ deltaY: -100, clientX: 480, clientY: 400, ctrlKey: false, preventDefault() {} });
    const after = dist(camera.eye, camera.target);
    expect(after).toBeLessThan(before);
    // Scaling about the anchor: target moved toward it by the same ratio.
    expect(camera.target[0]).toBeCloseTo(40 + (0 - 40) * (after / before), 3);

    rig.followNode(() => [0, 0, 0]);
    canvas.fire('pointerdown', [0.5, 0.5]);
    for (const h of handlers.pointermove ?? []) h({ pointerId: 1, clientX: 700, clientY: 200, shiftKey: true, buttons: 1 });
    for (const h of handlers.pointermove ?? []) h({ pointerId: 1, clientX: 720, clientY: 200, shiftKey: true, buttons: 1 });
    expect(rig.isFollowing).toBe(false);
  });

  it('dispose removes every listener', () => {
    const { canvas, rig } = setup();
    expect(canvas.listenerCount()).toBeGreaterThan(0);
    rig.dispose();
    expect(canvas.listenerCount()).toBe(0);
  });
});
