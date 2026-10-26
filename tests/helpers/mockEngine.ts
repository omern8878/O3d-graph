import type { O3DWasmBindings } from '../../src/index.js';

/** Test camera: perspective, at (0, 0, CAM_Z) looking down −Z, 60° vertical FOV, aspect 1. */
export const CAM_Z = 800;
const TAN = Math.tan(Math.PI / 6);

export function cameraRay(nx: number, ny: number): number[] {
  const dx = nx * TAN, dy = ny * TAN, dz = -1;
  const l = Math.hypot(dx, dy, dz);
  return [0, 0, CAM_Z, dx / l, dy / l, dz / l];
}

/** Column-major view-projection matching `cameraRay`/`project` (WebGPU depth 0..1, near 1, far 5000). */
export function viewProjection(): Float32Array {
  const f = 1 / TAN;
  const near = 1, far = 5000;
  const a = far / (near - far);
  const b = (near * far) / (near - far);
  // view = translate(0, 0, -CAM_Z); clip.w = CAM_Z - z
  return new Float32Array([f, 0, 0, 0, 0, f, 0, 0, 0, 0, a, -1, 0, 0, b - CAM_Z * a, CAM_Z]);
}

export function project(x: number, y: number, z: number): [number, number] {
  const d = CAM_Z - z;
  return [x / d / TAN, y / d / TAN];
}

export interface MockEngine extends O3DWasmBindings {
  memory: WebAssembly.Memory;
  calls: { orbit: boolean[]; camera2D: boolean[]; commits: Array<[string, ...number[]]> };
  /** Simulates an engine-side allocation (may grow / detach WASM memory). */
  alloc(bytes: number): number;
}

/**
 * Bump-allocator implementation of the @omernach/o3d contract over a real
 * `WebAssembly.Memory`, with JS ray-sphere picking and a fixed camera.
 */
export function createMockEngine(): MockEngine {
  const memory = new WebAssembly.Memory({ initial: 1, maximum: 16384 });
  let top = 64;
  let nextHandle = 1;
  const meshes = new Map<number, { t: number; c: number }>();
  const lines = new Map<number, { v: number; c: number }>();
  const calls: MockEngine['calls'] = { orbit: [], camera2D: [], commits: [] };

  const alloc = (bytes: number): number => {
    const ptr = top;
    top += (bytes + 15) & ~15;
    const pages = Math.ceil(top / 65536) - memory.buffer.byteLength / 65536;
    if (pages > 0) memory.grow(pages);
    return ptr;
  };

  return {
    memory,
    calls,
    alloc,
    get_wasm_memory: () => memory,
    create_instanced_mesh: (_g, n) => {
      const h = nextHandle++;
      meshes.set(h, { t: alloc(n * 16), c: alloc(n * 4) });
      return h;
    },
    get_instance_transforms_ptr: (h) => meshes.get(h)!.t,
    get_instance_colors_ptr: (h) => meshes.get(h)!.c,
    update_instanced_mesh: (h, n) => void calls.commits.push(['mesh', h, n]),
    update_instance_sub_range: (h, s, n) => void calls.commits.push(['meshRange', h, s, n]),
    create_line_buffer: (n) => {
      const h = nextHandle++;
      lines.set(h, { v: alloc(n * 24), c: alloc(n * 8) });
      return h;
    },
    get_line_vertices_ptr: (h) => lines.get(h)!.v,
    get_line_colors_ptr: (h) => lines.get(h)!.c,
    update_line_buffer: (h, n) => void calls.commits.push(['line', h, n]),
    update_line_sub_range: (h, s, n) => void calls.commits.push(['lineRange', h, s, n]),
    raycast_spheres_ndc: (nx, ny, ptr, count) => {
      const s = new Float32Array(memory.buffer, ptr, count * 4);
      const [ox, oy, oz, dx, dy, dz] = cameraRay(nx, ny);
      let best = -1;
      let bestT = Infinity;
      for (let i = 0; i < count; i++) {
        const r = s[i * 4 + 3];
        if (r <= 0) continue;
        const lx = s[i * 4] - ox, ly = s[i * 4 + 1] - oy, lz = s[i * 4 + 2] - oz;
        const tc = lx * dx + ly * dy + lz * dz;
        const d2 = lx * lx + ly * ly + lz * lz - tc * tc;
        if (d2 > r * r) continue;
        const t = tc - Math.sqrt(r * r - d2);
        if (t > 0 && t < bestT) {
          bestT = t;
          best = i;
        }
      }
      return best;
    },
    unproject_ndc_ray: (nx, ny) => {
      alloc(64); // wasm-bindgen return values allocate inside WASM
      return new Float32Array(cameraRay(nx, ny));
    },
    get_camera_forward: () => new Float32Array([0, 0, -1]),
    get_camera_view_proj_matrix: () => viewProjection(),
    set_camera_orbit_enabled: (e) => void calls.orbit.push(e),
    set_camera_2d_mode: (e) => void calls.camera2D.push(e),
  };
}

type Handler = (e: unknown) => void;

/** Minimal canvas stand-in: pointer listeners, bounding rect, pointer capture, cursor style. */
export class FakeCanvas {
  readonly handlers: Record<string, Handler[]> = {};
  readonly style = { cursor: '' };
  readonly clientWidth = 800;
  readonly clientHeight = 800;
  private readonly captured = new Set<number>();

  addEventListener(type: string, fn: Handler): void {
    (this.handlers[type] ??= []).push(fn);
  }
  removeEventListener(type: string, fn: Handler): void {
    this.handlers[type] = (this.handlers[type] ?? []).filter((f) => f !== fn);
  }
  getBoundingClientRect() {
    return { left: 0, top: 0, width: 800, height: 800 };
  }
  setPointerCapture(id: number): void {
    this.captured.add(id);
  }
  releasePointerCapture(id: number): void {
    this.captured.delete(id);
  }
  hasPointerCapture(id: number): boolean {
    return this.captured.has(id);
  }
  listenerCount(): number {
    return Object.values(this.handlers).reduce((n, list) => n + list.length, 0);
  }

  /** Dispatches a pointer event at an NDC position. Returns the event (check `.stopped`). */
  fire(type: string, ndc: readonly [number, number] | number[]) {
    const e = {
      type,
      clientX: (ndc[0] + 1) * 400,
      clientY: (1 - ndc[1]) * 400,
      isPrimary: true,
      button: 0,
      pointerId: 1,
      stopped: false,
      stopPropagation() {
        this.stopped = true;
      },
      preventDefault() {},
    };
    for (const fn of this.handlers[type] ?? []) fn(e);
    return e;
  }
}

/** Manually stepped frame scheduler. */
export function createManualScheduler() {
  const queue: Array<(t: number) => void> = [];
  return {
    queue,
    request: (cb: (t: number) => void) => queue.push(cb),
    cancel: () => void (queue.length = 0),
    /** Runs up to `n` frames; returns how many actually ran. */
    step(n = 1): number {
      let k = 0;
      for (; k < n && queue.length > 0; k++) queue.shift()!(k);
      return k;
    },
  };
}

/** Deterministic PRNG for test data. */
export function rng(seed = 7) {
  let s = seed >>> 0;
  return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32;
}
