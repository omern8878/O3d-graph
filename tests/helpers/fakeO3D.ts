import type { O3DModule } from '../../src/index.js';
import { cameraRay, viewProjection } from './mockEngine.js';

/**
 * Fakes mirroring @omernach/o3d's batch API semantics: fixed capacity, views
 * over a shared WebAssembly.Memory that re-bind after growth, inclusive dirty
 * ranges. Picking uses the fixed test camera from `mockEngine` (`cameraRay`).
 */
const memory = new WebAssembly.Memory({ initial: 1, maximum: 16384 });
let top = 64;
function alloc(bytes: number): number {
  const p = top;
  top += (bytes + 15) & ~15;
  const pages = Math.ceil(top / 65536) - memory.buffer.byteLength / 65536;
  if (pages > 0) memory.grow(pages);
  return p;
}

export const fakeLog = { dirty: [] as string[], destroyed: 0, renders: 0 };

export class FakeInstancedMesh {
  readonly capacity: number;
  private count = 0;
  private readonly t: number;
  private readonly c: number;
  constructor(_device: unknown, readonly mesh: { boundingRadius: number }, capacity: number) {
    this.capacity = capacity;
    this.t = alloc(capacity * 16);
    this.c = alloc(capacity * 4);
    const t = this.getTransforms();
    for (let i = 0; i < capacity; i++) t[i * 4 + 3] = 1;
  }
  getCount() { return this.count; }
  setCount(n: number) { return (this.count = Math.min(Math.max(0, n), this.capacity)); }
  getTransforms() { return new Float32Array(memory.buffer, this.t, this.capacity * 4); }
  getColors() { return new Uint32Array(memory.buffer, this.c, this.capacity); }
  markDirty(first: number, last = first, channels: readonly string[] = ['transform', 'rotation', 'color']) {
    fakeLog.dirty.push(`mesh ${first}..${last} ${channels.join('+')}`);
  }
  raycastNdc(_camera: unknown, nx: number, ny: number) {
    const s = this.getTransforms();
    const [ox, oy, oz, dx, dy, dz] = cameraRay(nx, ny);
    let best = -1, bestT = Infinity;
    for (let i = 0; i < this.count; i++) {
      const r = Math.abs(s[i * 4 + 3]) * this.mesh.boundingRadius;
      const lx = s[i * 4] - ox, ly = s[i * 4 + 1] - oy, lz = s[i * 4 + 2] - oz;
      const tc = lx * dx + ly * dy + lz * dz;
      const d2 = lx * lx + ly * ly + lz * lz - tc * tc;
      if (r <= 0 || d2 > r * r) continue;
      const t = tc - Math.sqrt(r * r - d2);
      if (t > 0 && t < bestT) { bestT = t; best = i; }
    }
    return best;
  }
  destroy() { fakeLog.destroyed++; }
}

export class FakeLineBatch {
  readonly capacity: number;
  private count = 0;
  private readonly v: number;
  private readonly c: number;
  constructor(_device: unknown, capacity: number) {
    this.capacity = capacity;
    this.v = alloc(capacity * 24);
    this.c = alloc(capacity * 8);
  }
  getCount() { return this.count; }
  setCount(n: number) { return (this.count = Math.min(Math.max(0, n), this.capacity)); }
  getVertices() { return new Float32Array(memory.buffer, this.v, this.capacity * 6); }
  getColors() { return new Uint32Array(memory.buffer, this.c, this.capacity * 2); }
  markDirty(first: number, last = first) { fakeLog.dirty.push(`lines ${first}..${last}`); }
  destroy() { fakeLog.destroyed++; }
}

/** CameraController stand-in that keeps eye/target state (forward, orbit) like the real one. */
export class FakeCameraController {
  eye: [number, number, number];
  target: [number, number, number];
  orthoHalfHeight = 100;
  mode: '2d' | '3d';
  orbitEnabled = true;
  locked = false;
  destroyed = false;
  constructor(config: { mode?: '2d' | '3d'; position?: { x: number; y: number; z: number }; target?: { x: number; y: number; z: number } } = {}) {
    this.mode = config.mode ?? '2d';
    const p = config.position ?? { x: 0, y: 0, z: 10 };
    const t = config.target ?? { x: 0, y: 0, z: 0 };
    this.eye = [p.x, p.y, p.z];
    this.target = [t.x, t.y, t.z];
  }
  getForward(): [number, number, number] {
    const d = [this.target[0] - this.eye[0], this.target[1] - this.eye[1], this.target[2] - this.eye[2]];
    const l = Math.hypot(d[0], d[1], d[2]) || 1;
    return [d[0] / l, d[1] / l, d[2] / l];
  }
  unprojectNdc(x: number, y: number) {
    const r = cameraRay(x, y);
    return { origin: [r[0], r[1], r[2]] as [number, number, number], direction: [r[3], r[4], r[5]] as [number, number, number] };
  }
  getViewProjection() { return viewProjection(); }
  isOrbitEnabled() { return this.orbitEnabled; }
  setOrbitEnabled(e: boolean) { this.orbitEnabled = e; }
  orbit(az: number, _el: number) {
    if (!this.orbitEnabled || this.locked) return;
    const dx = this.eye[0] - this.target[0], dz = this.eye[2] - this.target[2];
    const c = Math.cos(az), s = Math.sin(az);
    this.eye = [this.target[0] + dx * c - dz * s, this.eye[1], this.target[2] + dx * s + dz * c];
  }
  setMode(m: '2d' | '3d') { this.mode = m; }
  lockToXYPlane(l: boolean) { this.locked = l; }
  setTarget(x: number, y: number, z: number) { this.target = [x, y, z]; }
  setPosition(x: number, y: number, z: number) { this.eye = [x, y, z]; }
  setOrthoHalfHeight(h: number) { this.orthoHalfHeight = h; }
  resize() {}
  destroy() { this.destroyed = true; }
  getRawWasmCamera() { return {}; }
}

/** A stand-in for `import * as o3d from '@omernach/o3d'`. */
export function createFakeO3DModule() {
  const created = { cameras: [] as FakeCameraController[], meshes: 0, batchRenderers: 0, meshDestroyed: 0, batchDestroyed: 0 };
  const module = {
    initEngine: async () => ({}),
    GpuContext: {
      create: async ({ canvas }: { canvas: HTMLCanvasElement }) => ({ device: {}, canvas, resizeToDisplaySize() {} }),
    },
    BatchRenderer: class {
      constructor() { created.batchRenderers++; }
      render() { fakeLog.renders++; return { drawCalls: 2, instances: 0, segments: 0 }; }
      destroy() { created.batchDestroyed++; }
    },
    CameraController: class extends FakeCameraController {
      constructor(config?: ConstructorParameters<typeof FakeCameraController>[0]) {
        super(config);
        created.cameras.push(this);
      }
    },
    GpuMesh: {
      sphere: () => { created.meshes++; return { boundingRadius: 1, destroy() { created.meshDestroyed++; } }; },
      disc: () => { created.meshes++; return { boundingRadius: 1, destroy() { created.meshDestroyed++; } }; },
    },
    InstancedMesh: FakeInstancedMesh,
    LineBatch: FakeLineBatch,
  };
  return { o3d: module as unknown as O3DModule, created };
}
