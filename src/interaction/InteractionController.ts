import {
  NODE_TRANSFORM_STRIDE,
  NO_INDEX,
  NodeFlag,
  type IGraphEventMap,
  type IGraphRenderer,
  type IGraphStore,
  type InteractionOptions,
  type IPhysicsEngine,
  type NodeDragPayload,
  type NodeEventPayload,
} from '../types/index.js';
import type { EventBus } from '../events/EventBus.js';

export type ResolvedInteractionOptions = Required<InteractionOptions>;

export function resolveInteractionOptions(options: InteractionOptions = {}): ResolvedInteractionOptions {
  return {
    enabled: options.enabled ?? true,
    hover: options.hover ?? true,
    drag: options.drag ?? true,
    pinOnDragEnd: options.pinOnDragEnd ?? false,
    dragAlphaTarget: options.dragAlphaTarget ?? 0.3,
    reheatOnDrag: options.reheatOnDrag ?? true,
    clickTolerancePx: options.clickTolerancePx ?? 4,
  };
}

export interface InteractionControllerDeps<TNodeMeta> {
  element: HTMLElement;
  renderer: IGraphRenderer;
  store: IGraphStore<TNodeMeta, unknown>;
  physics: IPhysicsEngine;
  events: EventBus<IGraphEventMap<TNodeMeta>>;
  options?: InteractionOptions;
  /** Wakes the render loop (it sleeps while the graph is idle). */
  requestFrame: () => void;
}

const enum Phase {
  Idle,
  /** Pointer down on a node, not yet past the click tolerance. */
  Pressed,
  Dragging,
}

/** Rays nearly parallel to the drag plane are ignored to avoid shooting to infinity. */
const PARALLEL_EPSILON = 1e-6;

/**
 * Pointer → graph interaction: hover picking, click, and 3D ray-plane dragging.
 *
 * Pointer events only record state; the expensive work (WASM sphere raycast,
 * ray unprojection) runs at most once per animation frame in `update()`, so a
 * 1000 Hz mouse costs the same as a 60 Hz one. `pointerdown` is handled
 * synchronously in the capture phase so camera orbit can be locked before the
 * engine's own handlers see the press.
 *
 * Dragging (node `i`):
 *  - press: pin `i`, lock orbit, record `P_i` and plane normal `N` = camera
 *    forward (or `+Z` through `z = 0` in 2D), plus the grab offset so the
 *    node doesn't snap its center to the cursor;
 *  - move: `t = ((P_i − O) · N) / (D · N)`, `P = O + tD + offset`, written
 *    straight into WASM memory with velocity zeroed and the k incident edges
 *    resynced in O(k);
 *  - release: restore the pin state (or keep it with `pinOnDragEnd`), restore
 *    `alphaTarget`, unlock orbit.
 * A press released within `clickTolerancePx` is a click, never a drag, and
 * never reheats the simulation.
 */
export class InteractionController<TNodeMeta = unknown> {
  readonly options: ResolvedInteractionOptions;

  private readonly el: HTMLElement;
  private readonly renderer: IGraphRenderer;
  private readonly store: IGraphStore<TNodeMeta, unknown>;
  private readonly physics: IPhysicsEngine;
  private readonly events: EventBus<IGraphEventMap<TNodeMeta>>;
  private readonly requestFrame: () => void;

  // Pending pointer state (written by DOM handlers, consumed by update()).
  private pointerInside = false;
  private moveDirty = false;
  private ndcX = 0;
  private ndcY = 0;
  private lastEvent: PointerEvent | null = null;

  // Hover
  private hoveredIndex = NO_INDEX;
  private hoveredId: string | number | null = null;
  private hoverTopology = -1;

  // Press / drag
  private phase = Phase.Idle;
  private pointerId = -1;
  private dragIndex = NO_INDEX;
  private dragId: string | number | null = null;
  private dragTopology = -1;
  private wasPinned = false;
  private savedAlphaTarget = 0;
  private pressClientX = 0;
  private pressClientY = 0;
  private readonly planePoint = new Float64Array(3);
  private readonly planeNormal = new Float64Array(3);
  private readonly grabOffset = new Float64Array(3);
  private readonly hit = new Float64Array(3);
  private readonly pos = new Float32Array(3);
  private savedCursor = '';

  // Pooled payloads (see IGraphEventMap docs).
  private readonly nodePayload: NodeEventPayload<TNodeMeta> = { id: 0, index: 0, meta: undefined, event: null };
  private readonly dragPayload: NodeDragPayload<TNodeMeta> = {
    id: 0, index: 0, meta: undefined, event: null, x: 0, y: 0, z: 0,
  };

  private disposed = false;

  constructor(deps: InteractionControllerDeps<TNodeMeta>) {
    this.el = deps.element;
    this.renderer = deps.renderer;
    this.store = deps.store;
    this.physics = deps.physics;
    this.events = deps.events;
    this.requestFrame = deps.requestFrame;
    this.options = resolveInteractionOptions(deps.options);
    this.savedCursor = this.el.style.cursor;

    this.el.addEventListener('pointerdown', this.onPointerDown, { capture: true });
    this.el.addEventListener('pointermove', this.onPointerMove, { passive: true });
    this.el.addEventListener('pointerup', this.onPointerUp);
    this.el.addEventListener('pointercancel', this.onPointerCancel);
    this.el.addEventListener('pointerleave', this.onPointerLeave, { passive: true });
    this.el.addEventListener('lostpointercapture', this.onPointerCancel);
  }

  get isDragging(): boolean {
    return this.phase === Phase.Dragging;
  }

  /** `true` while a press or drag is in progress (keeps the render loop awake). */
  get isBusy(): boolean {
    return this.phase !== Phase.Idle || this.moveDirty;
  }

  get hoveredNode(): number {
    return this.hoveredIndex;
  }

  /** Per-frame: resolves coalesced pointer movement. */
  update(): void {
    if (this.disposed) return;
    this.revalidateIndices();
    if (!this.moveDirty) return;
    this.moveDirty = false;

    if (this.phase === Phase.Dragging) {
      this.dragTo(this.ndcX, this.ndcY);
    } else if (this.phase === Phase.Idle && this.options.enabled && this.options.hover) {
      this.updateHover(this.pointerInside ? this.pick(this.ndcX, this.ndcY) : NO_INDEX);
    }
  }

  /** Cancels an active drag and clears hover (e.g. before the graph is replaced). */
  reset(): void {
    if (this.phase !== Phase.Idle) this.endPress(null, true);
    this.updateHover(NO_INDEX);
  }

  dispose(): void {
    if (this.disposed) return;
    this.reset();
    this.disposed = true;
    this.el.removeEventListener('pointerdown', this.onPointerDown, { capture: true });
    this.el.removeEventListener('pointermove', this.onPointerMove);
    this.el.removeEventListener('pointerup', this.onPointerUp);
    this.el.removeEventListener('pointercancel', this.onPointerCancel);
    this.el.removeEventListener('pointerleave', this.onPointerLeave);
    this.el.removeEventListener('lostpointercapture', this.onPointerCancel);
    this.el.style.cursor = this.savedCursor;
  }

  // ===========================================================================
  // DOM handlers (arrow properties: stable identity for add/removeEventListener)
  // ===========================================================================

  private readonly onPointerDown = (e: PointerEvent): void => {
    if (!e.isPrimary) return;
    // Touch/pen presses aren't necessarily preceded by a pointermove.
    this.pointerInside = true;
    if (!this.options.enabled || e.button !== 0 || this.phase !== Phase.Idle) return;
    this.revalidateIndices();
    this.setNdc(e);
    const i = this.pick(this.ndcX, this.ndcY);
    if (i === NO_INDEX) return;

    // Keep the engine's orbit controls from treating this press as a camera drag.
    e.stopPropagation();
    e.preventDefault();
    this.renderer.camera.setCameraOrbitEnabled(false);
    try {
      this.el.setPointerCapture(e.pointerId);
    } catch {
      // Synthetic events or detached elements can't capture; dragging still works while inside.
    }

    this.phase = Phase.Pressed;
    this.pointerId = e.pointerId;
    this.dragIndex = i;
    this.dragId = this.store.getNodeId(i);
    this.dragTopology = this.store.topologyVersion;
    this.pressClientX = e.clientX;
    this.pressClientY = e.clientY;
    this.lastEvent = e;
    this.wasPinned = this.store.hasFlag(i, NodeFlag.PINNED);
    if (this.options.drag) {
      this.store.setNodeFlag(i, NodeFlag.PINNED, true);
      this.setupDragPlane(i, this.ndcX, this.ndcY);
    }
    this.updateHover(i);
    this.requestFrame();
  };

  private readonly onPointerMove = (e: PointerEvent): void => {
    if (!e.isPrimary) return;
    this.pointerInside = true;
    this.setNdc(e);
    this.lastEvent = e;

    if (this.phase === Phase.Pressed && e.pointerId === this.pointerId && this.options.drag) {
      const dx = e.clientX - this.pressClientX;
      const dy = e.clientY - this.pressClientY;
      const tol = this.options.clickTolerancePx;
      if (dx * dx + dy * dy > tol * tol) this.beginDrag(e);
    }
    this.moveDirty = true;
    this.requestFrame();
  };

  private readonly onPointerUp = (e: PointerEvent): void => {
    if (this.phase === Phase.Idle || e.pointerId !== this.pointerId) return;
    this.setNdc(e);
    this.lastEvent = e;
    this.endPress(e, false);
  };

  private readonly onPointerCancel = (e: PointerEvent): void => {
    if (this.phase === Phase.Idle || e.pointerId !== this.pointerId) return;
    this.endPress(e, true);
  };

  private readonly onPointerLeave = (e: PointerEvent): void => {
    if (!e.isPrimary) return;
    this.pointerInside = false;
    if (this.phase === Phase.Idle) {
      this.moveDirty = true;
      this.requestFrame();
    }
  };

  // ===========================================================================
  // Picking & hover
  // ===========================================================================

  private setNdc(e: PointerEvent): void {
    const r = this.el.getBoundingClientRect();
    const w = r.width || 1;
    const h = r.height || 1;
    this.ndcX = ((e.clientX - r.left) / w) * 2 - 1;
    this.ndcY = 1 - ((e.clientY - r.top) / h) * 2;
  }

  private pick(ndcX: number, ndcY: number): number {
    const n = this.store.nodeCount;
    if (n === 0) return NO_INDEX;
    const i = this.renderer.raycastNodes(ndcX, ndcY, n);
    // Hidden nodes have scale 0 and shouldn't hit, but guard against engines that disagree.
    return i !== NO_INDEX && !this.store.hasFlag(i, NodeFlag.HIDDEN) ? i : NO_INDEX;
  }

  private updateHover(i: number): void {
    if (i === this.hoveredIndex) return;
    const prev = this.hoveredIndex;
    if (prev !== NO_INDEX && prev < this.store.nodeCount) {
      this.store.setNodeFlag(prev, NodeFlag.HOVERED, false);
      this.emitNode('node:leave', prev);
    }
    this.hoveredIndex = i;
    this.hoverTopology = this.store.topologyVersion;
    if (i !== NO_INDEX) {
      this.hoveredId = this.store.getNodeId(i);
      this.store.setNodeFlag(i, NodeFlag.HOVERED, true);
      this.emitNode('node:hover', i);
    } else {
      this.hoveredId = null;
    }
    this.syncCursor();
    this.requestFrame();
  }

  /** Swap-and-pop removals move indices; re-resolve held indices through their IDs. */
  private revalidateIndices(): void {
    const version = this.store.topologyVersion;
    if (this.hoveredIndex !== NO_INDEX && this.hoverTopology !== version) {
      const i = this.hoveredId === null ? NO_INDEX : this.store.getNodeIndex(this.hoveredId);
      this.hoveredIndex = i;
      this.hoverTopology = version;
      if (i === NO_INDEX) {
        this.hoveredId = null;
        this.syncCursor();
      }
    }
    if (this.phase !== Phase.Idle && this.dragTopology !== version) {
      const i = this.dragId === null ? NO_INDEX : this.store.getNodeIndex(this.dragId);
      this.dragTopology = version;
      if (i === NO_INDEX) {
        // Dragged node was removed: abandon the gesture without touching the store.
        this.dragIndex = NO_INDEX;
        this.endPress(null, true);
      } else {
        this.dragIndex = i;
      }
    }
  }

  // ===========================================================================
  // Dragging
  // ===========================================================================

  private setupDragPlane(i: number, ndcX: number, ndcY: number): void {
    const t = this.renderer.getNodeTransforms();
    const o = i * NODE_TRANSFORM_STRIDE;
    const P = this.planePoint;
    const N = this.planeNormal;
    P[0] = t[o];
    P[1] = t[o + 1];
    P[2] = t[o + 2];

    if (this.store.dimensionMode === '2d') {
      P[2] = 0;
      N[0] = 0;
      N[1] = 0;
      N[2] = 1;
    } else {
      const fwd = this.renderer.camera.getCameraForward();
      N[0] = fwd[0];
      N[1] = fwd[1];
      N[2] = fwd[2];
    }

    // Offset between the node center and where the cursor ray meets the plane,
    // so the node keeps its grab point instead of snapping its center.
    const G = this.grabOffset;
    if (this.intersectPlane(ndcX, ndcY)) {
      G[0] = P[0] - this.hit[0];
      G[1] = P[1] - this.hit[1];
      G[2] = P[2] - this.hit[2];
    } else {
      G[0] = G[1] = G[2] = 0;
    }
  }

  /** Ray-plane intersection into `this.hit`. Returns `false` for (near-)parallel rays. */
  private intersectPlane(ndcX: number, ndcY: number): boolean {
    const ray = this.renderer.camera.unprojectNdcRay(ndcX, ndcY);
    const ox = ray[0], oy = ray[1], oz = ray[2];
    const dx = ray[3], dy = ray[4], dz = ray[5];
    const N = this.planeNormal;
    const P = this.planePoint;
    const denom = dx * N[0] + dy * N[1] + dz * N[2];
    if (denom > -PARALLEL_EPSILON && denom < PARALLEL_EPSILON) return false;
    const t = ((P[0] - ox) * N[0] + (P[1] - oy) * N[1] + (P[2] - oz) * N[2]) / denom;
    if (!Number.isFinite(t)) return false;
    this.hit[0] = ox + t * dx;
    this.hit[1] = oy + t * dy;
    this.hit[2] = oz + t * dz;
    return true;
  }

  private beginDrag(e: PointerEvent): void {
    this.phase = Phase.Dragging;
    if (this.options.reheatOnDrag) {
      this.savedAlphaTarget = this.physics.alphaTarget;
      this.physics.alphaTarget = Math.max(this.savedAlphaTarget, this.options.dragAlphaTarget);
      this.physics.restart();
    }
    this.syncCursor();
    this.emitDrag('node:dragstart', this.dragIndex, e);
  }

  private dragTo(ndcX: number, ndcY: number): void {
    const i = this.dragIndex;
    if (i === NO_INDEX || !this.intersectPlane(ndcX, ndcY)) return;
    const G = this.grabOffset;
    const x = this.hit[0] + G[0];
    const y = this.hit[1] + G[1];
    const z = this.store.dimensionMode === '2d' ? 0 : this.hit[2] + G[2];
    // Writes into WASM memory, zeroes velocity, resyncs only the k incident edges.
    this.store.setNodePosition(i, x, y, z);
    this.emitDrag('node:drag', i, this.lastEvent);
  }

  private endPress(e: PointerEvent | null, cancelled: boolean): void {
    const i = this.dragIndex;
    const wasDragging = this.phase === Phase.Dragging;

    if (i !== NO_INDEX) {
      if (this.options.drag) {
        const keepPinned = this.wasPinned || (wasDragging && !cancelled && this.options.pinOnDragEnd);
        this.store.setNodeFlag(i, NodeFlag.PINNED, keepPinned);
      }
      if (wasDragging) this.emitDrag('node:dragend', i, e);
      else if (!cancelled) this.emitNode('node:click', i, e);
    }
    if (wasDragging && this.options.reheatOnDrag) this.physics.alphaTarget = this.savedAlphaTarget;

    if (this.pointerId !== -1 && this.el.hasPointerCapture?.(this.pointerId)) {
      try {
        this.el.releasePointerCapture(this.pointerId);
      } catch {
        // Already released.
      }
    }
    this.phase = Phase.Idle;
    this.pointerId = -1;
    this.dragIndex = NO_INDEX;
    this.dragId = null;
    this.renderer.camera.setCameraOrbitEnabled(true);
    this.syncCursor();
    this.moveDirty = true; // re-evaluate hover where the pointer was released
    this.requestFrame();
  }

  private syncCursor(): void {
    this.el.style.cursor =
      this.phase === Phase.Dragging ? 'grabbing' : this.hoveredIndex !== NO_INDEX ? 'pointer' : this.savedCursor;
  }

  // ===========================================================================
  // Events
  // ===========================================================================

  private emitNode(
    type: 'node:hover' | 'node:leave' | 'node:click',
    i: number,
    e: PointerEvent | null = this.lastEvent,
  ): void {
    if (!this.events.hasListeners(type)) return;
    const p = this.nodePayload;
    p.id = this.store.getNodeId(i);
    p.index = i;
    p.meta = this.store.getNodeMeta(i);
    p.event = e;
    this.events.emit(type, p);
  }

  private emitDrag(type: 'node:dragstart' | 'node:drag' | 'node:dragend', i: number, e: PointerEvent | null): void {
    if (!this.events.hasListeners(type)) return;
    const p = this.dragPayload;
    this.store.readNodePosition(i, this.pos);
    p.id = this.store.getNodeId(i);
    p.index = i;
    p.meta = this.store.getNodeMeta(i);
    p.event = e;
    p.x = this.pos[0];
    p.y = this.pos[1];
    p.z = this.pos[2];
    this.events.emit(type, p);
  }
}
