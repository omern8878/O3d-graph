import type { IForce, PhysicsOptions } from '../../types/index.js';
import { resolvePhysicsOptions } from '../PhysicsSimulator.js';
import { CenterForce, RadialForce } from './CenterForce.js';
import { CollideForce } from './CollideForce.js';
import { LinkForce } from './LinkForce.js';
import { ManyBodyForce } from './ManyBodyForce.js';

export { CenterForce, RadialForce, CollideForce, LinkForce, ManyBodyForce };
export type { CenterForceOptions, RadialForceOptions } from './CenterForce.js';
export type { CollideForceOptions } from './CollideForce.js';
export type { LinkForceOptions } from './LinkForce.js';
export type { ManyBodyForceOptions } from './ManyBodyForce.js';

/**
 * Builds the standard force stack from `PhysicsOptions`, in application order:
 * `link` → `charge` → `radial` (if `radialRadius > 0`) → `collide` (if enabled) → `center`.
 *
 * Center runs last so it re-centers the final positions of the tick.
 */
export function createDefaultForces(options: PhysicsOptions = {}, seed = 1): IForce[] {
  const o = resolvePhysicsOptions(options);
  const forces: IForce[] = [
    new LinkForce({ iterations: o.linkIterations, seed }),
    new ManyBodyForce({
      strength: o.chargeStrength,
      theta: o.theta,
      distanceMin: o.chargeDistanceMin,
      distanceMax: o.chargeDistanceMax,
      seed: seed + 1,
    }),
  ];
  if (o.radialRadius > 0) {
    forces.push(new RadialForce({ radius: o.radialRadius, strength: o.radialStrength }));
  }
  if (o.collide) {
    forces.push(
      new CollideForce({
        strength: o.collideStrength,
        padding: o.collidePadding,
        iterations: o.collideIterations,
        seed: seed + 2,
      }),
    );
  }
  forces.push(new CenterForce({ strength: o.centerStrength }));
  return forces;
}
