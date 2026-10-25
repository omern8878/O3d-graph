/** Smallest power of two ≥ `max(required, current * 2, 16)`. */
export function nextCapacity(current: number, required: number): number {
  let cap = Math.max(16, current * 2, required);
  cap--;
  cap |= cap >>> 1;
  cap |= cap >>> 2;
  cap |= cap >>> 4;
  cap |= cap >>> 8;
  cap |= cap >>> 16;
  return cap + 1;
}

export function growFloat32(src: Float32Array, length: number): Float32Array {
  const out = new Float32Array(length);
  out.set(src.subarray(0, Math.min(src.length, length)));
  return out;
}

export function growUint32(src: Uint32Array, length: number): Uint32Array {
  const out = new Uint32Array(length);
  out.set(src.subarray(0, Math.min(src.length, length)));
  return out;
}

export function growUint8(src: Uint8Array, length: number): Uint8Array {
  const out = new Uint8Array(length);
  out.set(src.subarray(0, Math.min(src.length, length)));
  return out;
}

/** Grows an index buffer, filling new slots with `fill` (typically `NO_INDEX`). */
export function growInt32(src: Int32Array, length: number, fill: number): Int32Array {
  const out = new Int32Array(length);
  const n = Math.min(src.length, length);
  out.set(src.subarray(0, n));
  if (length > n) out.fill(fill, n);
  return out;
}
