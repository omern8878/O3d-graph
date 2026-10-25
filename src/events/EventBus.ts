import type { Unsubscribe } from '../types/index.js';

export type Listener<T> = (payload: T) => void;

interface Subscription<T> {
  readonly fn: Listener<T>;
  readonly once: boolean;
}

interface Channel<T> {
  /** `null` marks a slot removed during emission; compacted afterward. */
  subs: Array<Subscription<T> | null>;
  /** Re-entrancy depth: > 0 while this channel is emitting. */
  depth: number;
  dirty: boolean;
}

/**
 * Strictly typed, zero-dependency event emitter.
 *
 * - `emit` performs no allocation: listeners are iterated in place, and
 *   removals during emission are tombstoned and compacted afterward.
 * - Listeners added during an emission are not invoked by that emission.
 * - A throwing listener does not prevent the remaining listeners from running;
 *   the error is rethrown asynchronously so it still surfaces in devtools.
 */
export class EventBus<TMap extends object> {
  private readonly channels = new Map<keyof TMap, Channel<unknown>>();

  on<K extends keyof TMap>(event: K, listener: Listener<TMap[K]>): Unsubscribe {
    return this.subscribe(event, listener, false);
  }

  once<K extends keyof TMap>(event: K, listener: Listener<TMap[K]>): Unsubscribe {
    return this.subscribe(event, listener, true);
  }

  off<K extends keyof TMap>(event: K, listener: Listener<TMap[K]>): boolean {
    const channel = this.channels.get(event);
    if (channel === undefined) return false;
    const subs = channel.subs;
    for (let i = 0; i < subs.length; i++) {
      const sub = subs[i];
      if (sub !== null && sub.fn === listener) {
        this.removeAt(channel, i);
        return true;
      }
    }
    return false;
  }

  /** Cheap guard so callers can skip building payloads nobody listens to. */
  hasListeners<K extends keyof TMap>(event: K): boolean {
    const channel = this.channels.get(event);
    if (channel === undefined) return false;
    const subs = channel.subs;
    for (let i = 0; i < subs.length; i++) {
      if (subs[i] !== null) return true;
    }
    return false;
  }

  listenerCount<K extends keyof TMap>(event: K): number {
    const channel = this.channels.get(event);
    if (channel === undefined) return 0;
    let n = 0;
    const subs = channel.subs;
    for (let i = 0; i < subs.length; i++) if (subs[i] !== null) n++;
    return n;
  }

  emit<K extends keyof TMap>(event: K, payload: TMap[K]): void {
    const channel = this.channels.get(event);
    if (channel === undefined) return;
    const subs = channel.subs;
    const len = subs.length;
    if (len === 0) return;

    channel.depth++;
    try {
      for (let i = 0; i < len; i++) {
        const sub = subs[i];
        if (sub === null) continue;
        if (sub.once) this.removeAt(channel, i);
        try {
          (sub.fn as Listener<TMap[K]>)(payload);
        } catch (err) {
          reportAsync(err);
        }
      }
    } finally {
      channel.depth--;
      if (channel.depth === 0 && channel.dirty) compact(channel);
    }
  }

  /** Removes all listeners for `event`, or for every event when omitted. */
  clear<K extends keyof TMap>(event?: K): void {
    if (event === undefined) {
      for (const channel of this.channels.values()) this.clearChannel(channel);
      return;
    }
    const channel = this.channels.get(event);
    if (channel !== undefined) this.clearChannel(channel);
  }

  private subscribe<K extends keyof TMap>(
    event: K,
    listener: Listener<TMap[K]>,
    once: boolean,
  ): Unsubscribe {
    if (typeof listener !== 'function') {
      throw new TypeError(`EventBus: listener for "${String(event)}" must be a function`);
    }
    let channel = this.channels.get(event);
    if (channel === undefined) {
      channel = { subs: [], depth: 0, dirty: false };
      this.channels.set(event, channel);
    }
    const sub: Subscription<TMap[K]> = { fn: listener, once };
    channel.subs.push(sub as Subscription<unknown>);

    const owner = channel;
    return () => {
      const subs = owner.subs;
      for (let i = 0; i < subs.length; i++) {
        if (subs[i] === (sub as Subscription<unknown>)) {
          this.removeAt(owner, i);
          return;
        }
      }
    };
  }

  private removeAt(channel: Channel<unknown>, index: number): void {
    if (channel.depth > 0) {
      channel.subs[index] = null;
      channel.dirty = true;
    } else {
      channel.subs.splice(index, 1);
    }
  }

  private clearChannel(channel: Channel<unknown>): void {
    if (channel.depth > 0) {
      channel.subs.fill(null);
      channel.dirty = true;
    } else {
      channel.subs.length = 0;
    }
  }
}

function compact(channel: Channel<unknown>): void {
  const subs = channel.subs;
  let w = 0;
  for (let r = 0; r < subs.length; r++) {
    const sub = subs[r];
    if (sub !== null) subs[w++] = sub;
  }
  subs.length = w;
  channel.dirty = false;
}

function reportAsync(err: unknown): void {
  queueMicrotask(() => {
    throw err;
  });
}
