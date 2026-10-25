import type { CameraController } from '@omernach/o3d';

/** o3d's perspective camera: 45° vertical FOV, far plane fixed at 1000 units (no setter). */
const FAR = 1000;
const TAN_HALF_FOV = Math.tan(Math.PI / 8);
const MIN_DISTANCE = 3;
const MIN_ORTHO = 2;
const MAX_ORTHO = 4000;
const ORBIT_RAD_PER_PX = 0.007;
const TAP_SLOP_PX = 5;
/** Time constant of camera transitions (ms): ~95% of the way in 3τ. */
const EASE_MS = 120;

type Vec3 = [number, number, number];

/**
 * Camera gestures, framing and node-following on top of o3d's
 * `CameraController`.
 *
 * - Wheel / pinch zoom toward the point under the cursor; zoom-out stops
 *   before the far side of the graph would cross o3d's fixed far plane.
 * - `follow()` makes a node the orbit center: the camera glides to it and
 *   keeps it there while it moves, so orbiting and zooming always stay
 *   around it. Panning or fitting releases it.
 * - Transitions are eased per frame in `tick()`.
 *
 * The controller doesn't expose its eye position, so the rig tracks target,
 * distance and ortho height itself. Gestures start only when orbit is
 * enabled at `pointerdown`: o3d-graph handles presses in the capture phase
 * and disables orbit when one lands on a node, so dragging a node never
 * moves the camera.
 */
export class CameraRig {
  autoRotate = false;
  /** Set while the user drags a node: the camera holds completely still so the drag stays under the cursor. */
  followPaused = false;
  /** Called after any camera change: schedule a render. */
  onChange: () => void = () => {};
  /**
   * World point under the cursor (e.g. the node there), used as the zoom
   * anchor so it stays exactly under the cursor. Falls back to the plane
   * through the orbit center.
   */
  zoomAnchor: ((clientX: number, clientY: number) => Vec3 | null) | null = null;
  /** A press-and-release on empty space that didn't drag. */
  onBackgroundTap: () => void = () => {};

  private readonly camera: CameraController;
  private readonly canvas: HTMLCanvasElement;
  private mode: '2d' | '3d' = '3d';
  private target: Vec3 = [0, 0, 0];
  private distance = 600;
  private orthoHalfHeight = 280;
  private sceneRadius = 250;
  private sceneCenter: Vec3 = [0, 0, 0];

  // Eased transition goals (null = no transition on that channel).
  private goalTarget: Vec3 | null = null;
  private goalDistance: number | null = null;
  private goalOrtho: number | null = null;
  private follow: (() => Vec3 | null) | null = null;
  /** After the glide to a followed node completes, track it exactly every frame. */
  private followLocked = false;

  private readonly pointers = new Map<number, { x: number; y: number }>();
  private press: { x: number; y: number; moved: boolean } | null = null;
  private pinchDistance = 0;
  private interacting = false;

  constructor(camera: CameraController, canvas: HTMLCanvasElement) {
    this.camera = camera;
    this.canvas = canvas;
    canvas.addEventListener('pointerdown', this.onPointerDown);
    canvas.addEventListener('pointermove', this.onPointerMove);
    canvas.addEventListener('pointerup', this.onPointerUp);
    canvas.addEventListener('pointercancel', this.onPointerUp);
    canvas.addEventListener('wheel', this.onWheel, { passive: false });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    this.apply();
  }

  get isFollowing(): boolean {
    return this.follow !== null;
  }

  /** Keep in sync with the graph (the library already switches projection + XY lock). */
  setMode(mode: '2d' | '3d'): void {
    this.mode = mode;
    this.apply();
  }

  /**
   * Glides to frame a sphere of `radius` around `center` into the top
   * `viewFraction` of the canvas, raised by `liftPx`. Releases any followed node.
   */
  fit(center: Vec3, radius: number, viewFraction = 1, liftPx = 0): void {
    this.follow = null;
    const fraction = clamp(viewFraction, 0.3, 1);
    this.sceneRadius = radius;
    this.sceneCenter = [...center];
    const distance = this.clampDistance((radius / (TAN_HALF_FOV * fraction)) * 1.06, center);
    const ortho = clamp((radius / fraction) * 1.06, MIN_ORTHO, MAX_ORTHO);
    // Shift the target down so the graph appears raised by `liftPx`.
    const worldPerPx =
      (2 * (this.mode === '2d' ? ortho : distance * TAN_HALF_FOV)) / Math.max(1, this.canvas.clientHeight);
    const up = this.upVector();
    const target: Vec3 = [
      center[0] - up[0] * liftPx * worldPerPx,
      center[1] - up[1] * liftPx * worldPerPx,
      center[2] - up[2] * liftPx * worldPerPx,
    ];
    this.animateTo(target, distance, ortho);
  }

  /**
   * Makes the node returned by `position` the orbit center and keeps it
   * there while it moves. With `radius`, also zooms so a sphere of that size
   * around it fills the view.
   */
  followNode(position: () => Vec3 | null, radius?: number): void {
    this.follow = position;
    this.followLocked = false;
    const p = position();
    if (!p) return;
    const distance = radius === undefined ? this.distance : (radius / TAN_HALF_FOV) * 1.25;
    const ortho = radius === undefined ? this.orthoHalfHeight : radius * 1.25;
    this.animateTo(p, this.clampDistance(distance, p), clamp(ortho, MIN_ORTHO, MAX_ORTHO));
  }

  unfollow(): void {
    this.follow = null;
  }

  /** Advances easing, following and auto-rotation. Returns `true` if the camera moved. */
  tick(dtMs: number): boolean {
    if (this.followPaused) {
      this.followLocked = false; // glide back to a followed node after the drag
      return false; // transitions resume where they left off
    }
    let moved = false;
    if (this.autoRotate && this.mode === '3d' && !this.interacting) {
      this.camera.orbit(dtMs * 0.00012, 0);
      moved = true;
    }
    if (this.follow) {
      const p = this.follow();
      if (!p) {
        this.follow = null; // node removed
      } else if (this.followLocked) {
        if (p[0] !== this.target[0] || p[1] !== this.target[1] || p[2] !== this.target[2]) {
          this.target = p;
          moved = true;
        }
      } else {
        this.goalTarget = p;
      }
    }

    const k = 1 - Math.exp(-dtMs / EASE_MS);
    if (this.goalTarget) {
      let d2 = 0;
      for (let a = 0; a < 3; a++) {
        const delta = this.goalTarget[a] - this.target[a];
        this.target[a] += delta * k;
        d2 += delta * delta;
      }
      // Snap once visually done (0.1 units; 0.5 when locking onto a followed node).
      if (d2 < 0.01 || (this.follow && d2 < 0.25)) {
        this.target = [...this.goalTarget];
        this.goalTarget = null;
        if (this.follow) this.followLocked = true;
      }
      if (d2 > 0) moved = true; // a resting followed node gives d2 = 0: no redraw loop
    }
    if (this.goalDistance !== null) {
      const delta = this.goalDistance - this.distance;
      this.distance += delta * k;
      if (Math.abs(delta) < 0.01 + this.goalDistance * 0.002) {
        this.distance = this.goalDistance;
        this.goalDistance = null;
      }
      moved = true;
    }
    if (this.goalOrtho !== null) {
      const delta = this.goalOrtho - this.orthoHalfHeight;
      this.orthoHalfHeight += delta * k;
      if (Math.abs(delta) < 0.01 + this.goalOrtho * 0.002) {
        this.orthoHalfHeight = this.goalOrtho;
        this.goalOrtho = null;
      }
      moved = true;
    }
    if (moved) this.apply();
    return moved;
  }

  // ---------------------------------------------------------------------------

  private animateTo(target: Vec3, distance: number, ortho: number): void {
    this.goalTarget = target;
    this.goalDistance = distance;
    this.goalOrtho = ortho;
    this.onChange();
  }

  private apply(): void {
    // Read the view direction before moving the target, so it doesn't swing toward it.
    const f = this.camera.getForward();
    const len = Math.hypot(f[0], f[1], f[2]) || 1;
    const [tx, ty, tz] = this.target;
    this.camera.setTarget(tx, ty, tz);
    if (this.mode === '2d') {
      this.camera.setPosition(tx, ty, tz + this.distance);
      this.camera.setOrthoHalfHeight(this.orthoHalfHeight);
    } else {
      const d = this.distance / len;
      this.camera.setPosition(tx - f[0] * d, ty - f[1] * d, tz - f[2] * d);
    }
    this.onChange();
  }

  /** The far side of the graph (seen from `target`) must stay inside the far plane. */
  private clampDistance(d: number, target: Vec3 = this.target): number {
    const offCenter = Math.hypot(
      target[0] - this.sceneCenter[0],
      target[1] - this.sceneCenter[1],
      target[2] - this.sceneCenter[2],
    );
    const max = Math.min(FAR * 0.9, FAR * 0.97 - this.sceneRadius - offCenter);
    return clamp(d, MIN_DISTANCE, Math.max(MIN_DISTANCE, max));
  }

  /** Zooms by `factor` about the point under (clientX, clientY), or about the target when following. */
  private zoom(factor: number, clientX?: number, clientY?: number): void {
    this.goalDistance = null;
    this.goalOrtho = null;
    const before = this.mode === '2d' ? this.orthoHalfHeight : this.distance;
    let after: number;
    if (this.mode === '2d') {
      after = clamp(before * factor, MIN_ORTHO, MAX_ORTHO);
    } else {
      after = this.clampDistance(before * factor);
      if (factor > 1 && after < before * factor && !this.follow) {
        // At the far-plane limit: drift back toward the graph so zooming out still reveals all of it.
        for (let a = 0; a < 3; a++) this.target[a] += (this.sceneCenter[a] - this.target[a]) * 0.25;
        after = this.clampDistance(before * factor);
      }
    }
    if (after === before) {
      this.apply();
      return;
    }

    // Keep the world point under the cursor fixed: T' = P + (T - P) · ratio.
    if (!this.follow && clientX !== undefined && clientY !== undefined) {
      const p = this.zoomAnchor?.(clientX, clientY) ?? this.pointOnFocusPlane(clientX, clientY);
      if (p) {
        const ratio = after / before;
        for (let a = 0; a < 3; a++) this.target[a] = p[a] + (this.target[a] - p[a]) * ratio;
        if (this.goalTarget && !this.follow) this.goalTarget = null;
      }
    }
    if (this.mode === '2d') this.orthoHalfHeight = after;
    else this.distance = after;
    this.apply();
  }

  /** Where the cursor ray meets the plane through the target facing the camera. */
  private pointOnFocusPlane(clientX: number, clientY: number): Vec3 | null {
    const r = this.canvas.getBoundingClientRect();
    const ray = this.camera.unprojectNdc(((clientX - r.left) / r.width) * 2 - 1, 1 - ((clientY - r.top) / r.height) * 2);
    if (!ray) return null;
    const f = this.camera.getForward();
    const [ox, oy, oz] = ray.origin;
    const [dx, dy, dz] = ray.direction;
    const denom = dx * f[0] + dy * f[1] + dz * f[2];
    if (Math.abs(denom) < 1e-6) return null;
    const t = ((this.target[0] - ox) * f[0] + (this.target[1] - oy) * f[1] + (this.target[2] - oz) * f[2]) / denom;
    return [ox + dx * t, oy + dy * t, oz + dz * t];
  }

  private upVector(): Vec3 {
    if (this.mode === '2d') return [0, 1, 0];
    const f = this.camera.getForward();
    const right = normalize([-f[2], 0, f[0]]);
    return [right[1] * f[2] - right[2] * f[1], right[2] * f[0] - right[0] * f[2], right[0] * f[1] - right[1] * f[0]];
  }

  private pan(dx: number, dy: number): void {
    // Panning moves the view away from a followed node: let it go.
    this.follow = null;
    this.goalTarget = null;
    const h = Math.max(1, this.canvas.clientHeight);
    const f = this.camera.getForward();
    const right: Vec3 = this.mode === '2d' ? [1, 0, 0] : normalize([-f[2], 0, f[0]]);
    const up = this.upVector();
    const k = (2 * (this.mode === '2d' ? this.orthoHalfHeight : this.distance * TAN_HALF_FOV)) / h;
    for (let a = 0; a < 3; a++) this.target[a] += (-dx * right[a] + dy * up[a]) * k;
    if (this.mode === '3d') this.distance = this.clampDistance(this.distance);
    this.apply();
  }

  private readonly onPointerDown = (e: PointerEvent): void => {
    // A press on a node already disabled orbit (o3d-graph, capture phase).
    if (!this.camera.isOrbitEnabled()) return;
    this.canvas.setPointerCapture(e.pointerId);
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (this.pointers.size === 1) {
      this.press = { x: e.clientX, y: e.clientY, moved: false };
    } else {
      this.press = null;
      this.pinchDistance = this.currentPinch();
    }
    this.interacting = true;
  };

  private readonly onPointerMove = (e: PointerEvent): void => {
    const prev = this.pointers.get(e.pointerId);
    if (!prev) return;
    const dx = e.clientX - prev.x;
    const dy = e.clientY - prev.y;
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (this.pointers.size >= 2) {
      const d = this.currentPinch();
      if (this.pinchDistance > 0 && d > 0) {
        const [a, b] = [...this.pointers.values()];
        this.zoom(this.pinchDistance / d, (a.x + b.x) / 2, (a.y + b.y) / 2);
      }
      this.pinchDistance = d;
      return;
    }
    if (this.press && !this.press.moved) {
      if (Math.hypot(e.clientX - this.press.x, e.clientY - this.press.y) < TAP_SLOP_PX) return;
      this.press.moved = true;
    }
    if (this.mode === '2d' || e.shiftKey || (e.buttons & 2) !== 0) this.pan(dx, dy);
    else {
      this.camera.orbit(-dx * ORBIT_RAD_PER_PX, dy * ORBIT_RAD_PER_PX);
      this.onChange();
    }
  };

  private readonly onPointerUp = (e: PointerEvent): void => {
    if (!this.pointers.has(e.pointerId)) return;
    const tap = this.press !== null && !this.press.moved && this.pointers.size === 1 && e.type === 'pointerup';
    this.pointers.delete(e.pointerId);
    if (this.pointers.size === 0) {
      this.press = null;
      this.interacting = false;
    } else {
      this.pinchDistance = this.currentPinch();
    }
    if (tap) this.onBackgroundTap();
  };

  private readonly onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    // Trackpad pinch arrives as ctrl+wheel with small deltas: make it as responsive.
    const scale = e.ctrlKey ? 0.01 : 0.0015;
    this.zoom(Math.exp(clamp(e.deltaY, -200, 200) * scale), e.clientX, e.clientY);
  };

  private currentPinch(): number {
    const [a, b] = [...this.pointers.values()];
    return a && b ? Math.hypot(a.x - b.x, a.y - b.y) : 0;
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function normalize(v: Vec3): Vec3 {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
