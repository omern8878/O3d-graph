/** World-space point. */
export type Vec3 = [number, number, number];

/**
 * Camera operations the rig needs. `@omernach/o3d`'s `CameraController`
 * satisfies it as-is.
 */
export interface CameraRigCamera {
  getForward(): ArrayLike<number>;
  unprojectNdc(ndcX: number, ndcY: number): { origin: ArrayLike<number>; direction: ArrayLike<number> } | null;
  isOrbitEnabled(): boolean;
  orbit(deltaAzimuthRad: number, deltaElevationRad: number): void;
  setTarget(x: number, y: number, z: number): void;
  setPosition(x: number, y: number, z: number): void;
  setOrthoHalfHeight(halfHeight: number): void;
}

export interface CameraRigOptions {
  /** Vertical field of view of the perspective projection (radians). Default `π/4` (o3d). */
  fovY?: number;
  /**
   * Far clipping distance. Zoom-out is limited so the far side of the scene
   * stays inside it. Default `1000` (o3d 1.0's fixed far plane); `Infinity`
   * disables the limit.
   */
  far?: number;
  /** Closest zoom distance in 3D. Default `3`. */
  minDistance?: number;
  /** 2D zoom limits (orthographic half-height). Defaults `2` / `4000`. */
  minOrthoHalfHeight?: number;
  maxOrthoHalfHeight?: number;
  /** Orbit radians per dragged pixel. Default `0.007`. */
  orbitSpeed?: number;
  /** Zoom per wheel-delta unit. Default `0.0015`. */
  zoomSpeed?: number;
  /** Time constant of eased transitions (ms). Default `120`. */
  easeMs?: number;
  /** Auto-rotation speed (radians per ms). Default `0.00012`. */
  autoRotateSpeed?: number;
}

const TAP_SLOP_PX = 5;

/**
 * Mouse / touch camera control and framing for graph views.
 *
 * - Drag to orbit (3D) or pan (2D, or shift / right-drag in 3D); wheel,
 *   trackpad pinch or two-finger pinch to zoom toward the point under the
 *   cursor (`zoomAnchor` can pin it to the node there).
 * - `followNode()` makes a node the orbit center: the camera glides to it
 *   and then tracks it exactly while it moves, so orbit and zoom always stay
 *   around it. Panning or `fit()` releases it.
 * - `fit()` frames a sphere (with optional UI insets); zoom-out never pushes
 *   the scene past the far plane.
 * - Transitions are eased in `tick()`, which the render loop calls each frame.
 *
 * Gestures start only if orbit is enabled at `pointerdown`. o3d-graph's
 * InteractionController handles presses in the capture phase and disables
 * orbit when one lands on a node, so dragging a node never moves the camera.
 */
export class CameraRig {
  autoRotate = false;
  /** While true (e.g. a node is being dragged) the camera holds completely still. */
  followPaused = false;
  /** Called after any camera change — schedule a render. */
  onChange: () => void = () => {};
  /** A press-and-release on empty space that didn't drag. */
  onBackgroundTap: () => void = () => {};
  /**
   * World point under the cursor (e.g. the node there) used as the zoom
   * anchor, so it stays exactly under the cursor at any depth. Falls back to
   * the plane through the orbit center.
   */
  zoomAnchor: ((clientX: number, clientY: number) => Vec3 | null) | null = null;

  private readonly camera: CameraRigCamera;
  private readonly canvas: HTMLCanvasElement;
  private readonly tanHalfFov: number;
  private readonly far: number;
  private readonly minDistance: number;
  private readonly minOrtho: number;
  private readonly maxOrtho: number;
  private readonly orbitSpeed: number;
  private readonly zoomSpeed: number;
  private readonly easeMs: number;
  private readonly autoRotateSpeed: number;

  private mode: '2d' | '3d' = '3d';
  private target: Vec3 = [0, 0, 0];
  private distance = 600;
  private orthoHalfHeight = 280;
  private sceneRadius = 250;
  private sceneCenter: Vec3 = [0, 0, 0];

  // Eased transition goals (null = idle channel).
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
  private disposed = false;

  constructor(camera: CameraRigCamera, canvas: HTMLCanvasElement, options: CameraRigOptions = {}) {
    this.camera = camera;
    this.canvas = canvas;
    this.tanHalfFov = Math.tan((options.fovY ?? Math.PI / 4) / 2);
    this.far = options.far ?? 1000;
    this.minDistance = options.minDistance ?? 3;
    this.minOrtho = options.minOrthoHalfHeight ?? 2;
    this.maxOrtho = options.maxOrthoHalfHeight ?? 4000;
    this.orbitSpeed = options.orbitSpeed ?? 0.007;
    this.zoomSpeed = options.zoomSpeed ?? 0.0015;
    this.easeMs = options.easeMs ?? 120;
    this.autoRotateSpeed = options.autoRotateSpeed ?? 0.00012;

    canvas.addEventListener('pointerdown', this.onPointerDown);
    canvas.addEventListener('pointermove', this.onPointerMove);
    canvas.addEventListener('pointerup', this.onPointerUp);
    canvas.addEventListener('pointercancel', this.onPointerUp);
    canvas.addEventListener('wheel', this.onWheel, { passive: false });
    canvas.addEventListener('contextmenu', this.onContextMenu);
    this.apply();
  }

  get isFollowing(): boolean {
    return this.follow !== null;
  }

  /** `true` while the user is orbiting / panning / pinching. */
  get isInteracting(): boolean {
    return this.interacting;
  }

  /** `true` while a transition is in progress or the camera is tracking / rotating. */
  get isAnimating(): boolean {
    return (
      this.goalTarget !== null ||
      this.goalDistance !== null ||
      this.goalOrtho !== null ||
      (this.autoRotate && this.mode === '3d')
    );
  }

  /** Keep in sync with the graph's dimension mode (o3d-graph switches the projection itself). */
  setMode(mode: '2d' | '3d'): void {
    this.mode = mode;
    this.apply();
  }

  /**
   * Glides to frame a sphere of `radius` around `center`, releasing any
   * followed node. `insets` (CSS px) keep the framing clear of overlaid UI.
   */
  fit(center: Vec3, radius: number, insets: { top?: number; right?: number; bottom?: number; left?: number } = {}): void {
    this.follow = null;
    const w = Math.max(1, this.canvas.clientWidth || 0);
    const h = Math.max(1, this.canvas.clientHeight || 0);
    const top = insets.top ?? 0;
    const bottom = insets.bottom ?? 0;
    const left = insets.left ?? 0;
    const right = insets.right ?? 0;
    // Diameter must fit both the free height and the free width (portrait screens).
    const fraction = clamp(Math.min((h - top - bottom) / h, (w - left - right) / h), 0.2, 1);
    this.sceneRadius = radius;
    this.sceneCenter = [...center];
    const distance = this.clampDistance((radius / (this.tanHalfFov * fraction)) * 1.06, center);
    const ortho = clamp((radius / fraction) * 1.06, this.minOrtho, this.maxOrtho);

    // Offset the target so the scene is centered in the free area.
    const worldPerPx = (2 * (this.mode === '2d' ? ortho : distance * this.tanHalfFov)) / h;
    const up = this.upVector();
    const rightV = this.rightVector();
    const liftPx = (bottom - top) / 2;
    const shiftPx = (right - left) / 2;
    const target: Vec3 = [0, 0, 0];
    for (let a = 0; a < 3; a++) {
      target[a] = center[a] - up[a] * liftPx * worldPerPx + rightV[a] * shiftPx * worldPerPx;
    }
    this.animateTo(target, distance, ortho);
  }

  /**
   * Makes the point returned by `position` the orbit center and keeps it
   * there while it moves (return `null` to stop, e.g. when the node is
   * removed). With `radius`, also zooms so a sphere of that size fills the view.
   */
  followNode(position: () => Vec3 | null, radius?: number): void {
    this.follow = position;
    this.followLocked = false;
    const p = position();
    if (!p) return;
    const distance = radius === undefined ? this.distance : (radius / this.tanHalfFov) * 1.25;
    const ortho = radius === undefined ? this.orthoHalfHeight : radius * 1.25;
    this.animateTo(p, this.clampDistance(distance, p), clamp(ortho, this.minOrtho, this.maxOrtho));
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
      this.camera.orbit(dtMs * this.autoRotateSpeed, 0);
      moved = true;
    }
    if (this.follow) {
      const p = this.follow();
      if (!p) {
        this.follow = null;
      } else if (this.followLocked) {
        if (p[0] !== this.target[0] || p[1] !== this.target[1] || p[2] !== this.target[2]) {
          this.target = [p[0], p[1], p[2]];
          moved = true;
        }
      } else {
        this.goalTarget = p;
      }
    }

    const k = 1 - Math.exp(-dtMs / this.easeMs);
    if (this.goalTarget) {
      let d2 = 0;
      for (let a = 0; a < 3; a++) {
        const delta = this.goalTarget[a] - this.target[a];
        this.target[a] += delta * k;
        d2 += delta * delta;
      }
      // Snap once visually done (0.1 units; 0.5 when locking onto a followed node).
      if (d2 < 0.01 || (this.follow && d2 < 0.25)) {
        this.target = [this.goalTarget[0], this.goalTarget[1], this.goalTarget[2]];
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

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.canvas.removeEventListener('pointerdown', this.onPointerDown);
    this.canvas.removeEventListener('pointermove', this.onPointerMove);
    this.canvas.removeEventListener('pointerup', this.onPointerUp);
    this.canvas.removeEventListener('pointercancel', this.onPointerUp);
    this.canvas.removeEventListener('wheel', this.onWheel);
    this.canvas.removeEventListener('contextmenu', this.onContextMenu);
    this.pointers.clear();
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

  /** The far side of the scene, seen from `target`, must stay inside the far plane. */
  private clampDistance(d: number, target: Vec3 = this.target): number {
    if (!Number.isFinite(this.far)) return Math.max(this.minDistance, d);
    const offCenter = Math.hypot(
      target[0] - this.sceneCenter[0],
      target[1] - this.sceneCenter[1],
      target[2] - this.sceneCenter[2],
    );
    const max = Math.min(this.far * 0.9, this.far * 0.97 - this.sceneRadius - offCenter);
    return clamp(d, this.minDistance, Math.max(this.minDistance, max));
  }

  /** Zooms by `factor` about the point under (clientX, clientY), or about the target when following. */
  private zoom(factor: number, clientX?: number, clientY?: number): void {
    if (!(factor > 0) || !Number.isFinite(factor)) return; // malformed wheel / pinch input
    this.goalDistance = null;
    this.goalOrtho = null;
    const before = this.mode === '2d' ? this.orthoHalfHeight : this.distance;
    let after: number;
    if (this.mode === '2d') {
      after = clamp(before * factor, this.minOrtho, this.maxOrtho);
    } else {
      after = this.clampDistance(before * factor);
      if (factor > 1 && after < before * factor && !this.follow) {
        // At the far-plane limit: drift back toward the scene so zooming out still reveals all of it.
        for (let a = 0; a < 3; a++) this.target[a] += (this.sceneCenter[a] - this.target[a]) * 0.25;
        after = this.clampDistance(before * factor);
      }
    }
    if (after === before) {
      this.apply();
      return;
    }
    // Keep the anchor point fixed on screen: scale the view about it (T' = P + (T − P)·ratio).
    if (!this.follow && clientX !== undefined && clientY !== undefined) {
      const p = this.zoomAnchor?.(clientX, clientY) ?? this.pointOnFocusPlane(clientX, clientY);
      if (p) {
        const ratio = after / before;
        for (let a = 0; a < 3; a++) this.target[a] = p[a] + (this.target[a] - p[a]) * ratio;
        this.goalTarget = null;
      }
    }
    if (this.mode === '2d') this.orthoHalfHeight = after;
    else this.distance = after;
    this.apply();
  }

  /** Where the cursor ray meets the plane through the target facing the camera. */
  private pointOnFocusPlane(clientX: number, clientY: number): Vec3 | null {
    const r = this.canvas.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return null;
    const ray = this.camera.unprojectNdc(((clientX - r.left) / r.width) * 2 - 1, 1 - ((clientY - r.top) / r.height) * 2);
    if (!ray) return null;
    const f = this.camera.getForward();
    const ox = ray.origin[0], oy = ray.origin[1], oz = ray.origin[2];
    const dx = ray.direction[0], dy = ray.direction[1], dz = ray.direction[2];
    const denom = dx * f[0] + dy * f[1] + dz * f[2];
    if (Math.abs(denom) < 1e-6) return null;
    const t = ((this.target[0] - ox) * f[0] + (this.target[1] - oy) * f[1] + (this.target[2] - oz) * f[2]) / denom;
    return [ox + dx * t, oy + dy * t, oz + dz * t];
  }

  private rightVector(): Vec3 {
    if (this.mode === '2d') return [1, 0, 0];
    const f = this.camera.getForward();
    return normalize([-f[2], 0, f[0]]);
  }

  private upVector(): Vec3 {
    if (this.mode === '2d') return [0, 1, 0];
    const f = this.camera.getForward();
    const r = this.rightVector();
    return [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
  }

  private pan(dx: number, dy: number): void {
    // Panning moves the view away from a followed node: let it go.
    this.follow = null;
    this.goalTarget = null;
    const h = Math.max(1, this.canvas.clientHeight || 0);
    const right = this.rightVector();
    const up = this.upVector();
    const k = (2 * (this.mode === '2d' ? this.orthoHalfHeight : this.distance * this.tanHalfFov)) / h;
    for (let a = 0; a < 3; a++) this.target[a] += (-dx * right[a] + dy * up[a]) * k;
    if (this.mode === '3d') this.distance = this.clampDistance(this.distance);
    this.apply();
  }

  private readonly onPointerDown = (e: PointerEvent): void => {
    // A press on a node already disabled orbit (o3d-graph, capture phase).
    if (!this.camera.isOrbitEnabled()) return;
    try {
      this.canvas.setPointerCapture(e.pointerId);
    } catch {
      // Synthetic or already-released pointers can't be captured; gestures still work inside.
    }
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
      this.camera.orbit(-dx * this.orbitSpeed, dy * this.orbitSpeed);
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
    const scale = e.ctrlKey ? this.zoomSpeed * 6.5 : this.zoomSpeed;
    this.zoom(Math.exp(clamp(e.deltaY, -200, 200) * scale), e.clientX, e.clientY);
  };

  private readonly onContextMenu = (e: Event): void => {
    e.preventDefault();
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
