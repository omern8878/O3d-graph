const LCG_A = 1664525;
const LCG_C = 1013904223;
const INV_2_32 = 1 / 4294967296;

/**
 * Deterministic 32-bit linear congruential generator (Numerical Recipes
 * constants, same as d3-force). Allocation-free after construction.
 */
export class Lcg {
  private state: number;

  constructor(seed = 1) {
    this.state = seed >>> 0;
  }

  reseed(seed: number): void {
    this.state = seed >>> 0;
  }

  /** Uniform float in `[0, 1)`. */
  next(): number {
    this.state = (Math.imul(LCG_A, this.state) + LCG_C) >>> 0;
    return this.state * INV_2_32;
  }

  /** Uniform float in `[min, max)`. */
  range(min: number, max: number): number {
    return min + (max - min) * this.next();
  }
}

/** Tiny deterministic offset used to separate exactly coincident nodes. */
export function jiggle(rng: Lcg): number {
  return (rng.next() - 0.5) * 1e-6;
}
