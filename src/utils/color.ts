import type { ColorInput, ColorLayout } from '../types/index.js';

/** Packs 8-bit channels into `0xRRGGBBAA` (unsigned). */
export function packRGBA(r: number, g: number, b: number, a = 255): number {
  return (((r & 0xff) << 24) | ((g & 0xff) << 16) | ((b & 0xff) << 8) | (a & 0xff)) >>> 0;
}

/** Packs normalized `[0, 1]` channels into `0xRRGGBBAA`. */
export function packRGBAf(r: number, g: number, b: number, a = 1): number {
  return packRGBA(toByte(r), toByte(g), toByte(b), toByte(a));
}

export function unpackR(rgba: number): number {
  return (rgba >>> 24) & 0xff;
}
export function unpackG(rgba: number): number {
  return (rgba >>> 16) & 0xff;
}
export function unpackB(rgba: number): number {
  return (rgba >>> 8) & 0xff;
}
export function unpackA(rgba: number): number {
  return rgba & 0xff;
}

/** Replaces the alpha channel of a packed color. */
export function withAlpha(rgba: number, a: number): number {
  return ((rgba & 0xffffff00) | (a & 0xff)) >>> 0;
}

/**
 * Resolves a `ColorInput` into packed `0xRRGGBBAA`. Numbers are taken as-is;
 * strings must be CSS hex (`#rgb`, `#rgba`, `#rrggbb`, `#rrggbbaa`).
 * Invalid input resolves to `fallback`.
 */
export function parseColor(input: ColorInput | undefined, fallback: number): number {
  if (input === undefined) return fallback >>> 0;
  if (typeof input === 'number') {
    return Number.isFinite(input) ? input >>> 0 : fallback >>> 0;
  }
  const s = input.charCodeAt(0) === 35 /* # */ ? input.slice(1) : input;
  if (!/^[0-9a-fA-F]+$/.test(s)) return fallback >>> 0;
  switch (s.length) {
    case 3:
    case 4: {
      const r = parseInt(s[0] + s[0], 16);
      const g = parseInt(s[1] + s[1], 16);
      const b = parseInt(s[2] + s[2], 16);
      const a = s.length === 4 ? parseInt(s[3] + s[3], 16) : 255;
      return packRGBA(r, g, b, a);
    }
    case 6:
      return ((parseInt(s, 16) << 8) | 0xff) >>> 0;
    case 8:
      return parseInt(s, 16) >>> 0;
    default:
      return fallback >>> 0;
  }
}

/**
 * Converts public `0xRRGGBBAA` into `layout`. `'abgr'` is a byte swap
 * (red ends up in the lowest byte), so it is its own inverse.
 */
export function encodeColorLayout(rgba: number, layout: ColorLayout): number {
  if (layout === 'rgba') return rgba >>> 0;
  return (
    (((rgba & 0xff) << 24) | (((rgba >>> 8) & 0xff) << 16) | (((rgba >>> 16) & 0xff) << 8) | (rgba >>> 24)) >>> 0
  );
}

function toByte(v: number): number {
  return v <= 0 ? 0 : v >= 1 ? 255 : Math.round(v * 255);
}
