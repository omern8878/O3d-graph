import type { EdgeInput, GraphData, NodeInput } from '../src/index.js';

export interface PersonMeta {
  name: string;
  team: number;
  role: string;
}

export interface LinkMeta {
  crossTeam: boolean;
}

export const TEAMS = [
  { name: 'Platform', color: 0x4f9dffff },
  { name: 'Graphics', color: 0xff5c8aff },
  { name: 'Data', color: 0x34d399ff },
  { name: 'Design', color: 0xfbbf24ff },
  { name: 'Infra', color: 0xa78bfaff },
  { name: 'Mobile', color: 0x22d3eeff },
  { name: 'Research', color: 0xf472b6ff },
  { name: 'Security', color: 0xfb923cff },
  { name: 'Growth', color: 0xa3e635ff },
  { name: 'Support', color: 0x93c5fdff },
] as const;

const PEER_RELATIONS = ['pairs with', 'reviews for', 'collaborates', 'mentors'];

const ROLES = ['Engineer', 'Engineer', 'Engineer', 'Senior Engineer', 'Designer', 'Analyst', 'Researcher', 'PM'];
const FIRST = ['Ada', 'Bo', 'Cyra', 'Dev', 'Eli', 'Fay', 'Gus', 'Hana', 'Ivo', 'Jun', 'Kai', 'Lena', 'Milo', 'Nia', 'Omar', 'Pia', 'Quinn', 'Rae', 'Sol', 'Tara', 'Uri', 'Vera', 'Wes', 'Xin', 'Yara', 'Zeke'];
const LAST = ['Archer', 'Brooks', 'Castillo', 'Dahl', 'Ellis', 'Frost', 'Gray', 'Hale', 'Ito', 'Jensen', 'Kerr', 'Lund', 'Moss', 'Nash', 'Okafor', 'Park', 'Quist', 'Reyes', 'Stone', 'Tran', 'Varga', 'Wolfe', 'Young', 'Zhou'];

/** Edge alpha in the normal view; focus mode raises incident edges and dims the rest. */
export const EDGE_ALPHA = 0x50;

export function withAlpha(rgba: number, alpha: number): number {
  return ((rgba & 0xffffff00) | (alpha & 0xff)) >>> 0;
}

function lcg(seed: number) {
  let s = seed >>> 0;
  return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32;
}

/**
 * Deterministic mock organisation: `count` people in 10 teams of uneven size.
 *
 * - Inside a team each new person reports to / works with 1–2 existing
 *   teammates picked by preferential attachment, so every team grows a few
 *   well-connected leads.
 * - Teams are bridged mostly through those leads (cross-team links attach
 *   to high-degree people), which keeps clusters visually distinct.
 * - Node radius grows with degree; edges fade from the source team's color
 *   to the target team's.
 */
export function createMockGraph(count = 1000, seed = 7): GraphData<PersonMeta, LinkMeta> {
  const rand = lcg(seed);
  const pick = <T,>(xs: readonly T[]) => xs[(rand() * xs.length) | 0];

  const teamOf = new Uint8Array(count);
  const degree = new Uint32Array(count);
  const members: number[][] = TEAMS.map(() => []);
  const endpoints: number[][] = TEAMS.map(() => []); // sampling from this = preferential attachment
  const pairs: Array<[number, number, boolean]> = [];
  const seen = new Set<number>();

  const link = (a: number, b: number, crossTeam: boolean) => {
    if (a === b) return;
    const key = a < b ? a * count + b : b * count + a;
    if (seen.has(key)) return;
    seen.add(key);
    pairs.push([a, b, crossTeam]);
    degree[a]++;
    degree[b]++;
    endpoints[teamOf[a]].push(a);
    endpoints[teamOf[b]].push(b);
  };

  for (let i = 0; i < count; i++) {
    const t = Math.min(TEAMS.length - 1, Math.floor(Math.pow(rand(), 1.25) * TEAMS.length));
    teamOf[i] = t;
    const team = members[t];
    if (team.length > 0) {
      const links = rand() < 0.3 ? 2 : 1;
      for (let k = 0; k < links; k++) {
        const pool = endpoints[t].length > 0 && rand() < 0.85 ? endpoints[t] : team;
        link(i, pick(pool), false);
      }
    }
    team.push(i);
  }

  // Bridges: ~5% of people link to a well-connected person in another team.
  for (let i = 0; i < count; i++) {
    if (rand() >= 0.05) continue;
    let other = (rand() * TEAMS.length) | 0;
    if (other === teamOf[i]) other = (other + 1) % TEAMS.length;
    if (endpoints[other].length > 0) link(i, pick(endpoints[other]), true);
  }

  const nodes: Array<NodeInput<PersonMeta>> = [];
  for (let i = 0; i < count; i++) {
    const t = teamOf[i];
    const name = `${pick(FIRST)} ${pick(LAST)}`;
    nodes.push({
      id: `p${i}`,
      label: name,
      radius: 1.3 + 0.8 * Math.sqrt(degree[i]),
      color: TEAMS[t].color,
      meta: {
        name,
        team: t,
        role: degree[i] >= 12 ? 'Team Lead' : degree[i] >= 7 ? 'Staff Engineer' : pick(ROLES),
      },
    });
  }

  const edges: Array<EdgeInput<LinkMeta>> = pairs.map(([a, b, crossTeam]) => ({
    source: `p${a}`,
    target: `p${b}`,
    // Newer person → better-connected teammate reads as a reporting line.
    label: crossTeam
      ? 'partners with'
      : degree[b] >= 7 && degree[b] > degree[a] * 2
        ? 'reports to'
        : PEER_RELATIONS[(a + b) % PEER_RELATIONS.length],
    color: withAlpha(TEAMS[teamOf[a]].color, EDGE_ALPHA),
    targetColor: withAlpha(TEAMS[teamOf[b]].color, EDGE_ALPHA),
    meta: { crossTeam },
  }));

  return { nodes, edges };
}
