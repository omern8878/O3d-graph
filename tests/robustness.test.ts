import { describe, expect, it } from 'vitest';
import { GraphStore, O3DEngineAdapter, O3DGraph, O3DRendererAdapter, parseColor } from '../src/index.js';
import { FakeCanvas, createManualScheduler, createMockEngine } from './helpers/mockEngine.js';

function store() {
  return new GraphStore(new O3DRendererAdapter(new O3DEngineAdapter(createMockEngine())));
}

describe('GraphStore.setData is atomic', () => {
  it.each([
    ['duplicate node id', { nodes: [{ id: 'a' }, { id: 'a' }] }, /duplicate node id "a"/],
    ['missing node id', { nodes: [{ id: 'a' }, {} as never] }, /nodes\[1\] has no id/],
    ['dangling edge', { nodes: [{ id: 'a' }], edges: [{ source: 'a', target: 'ghost' }] }, /target "ghost" is not a node/],
    ['duplicate edge id', { nodes: [{ id: 'a' }, { id: 'b' }], edges: [{ id: 1, source: 'a', target: 'b' }, { id: 1, source: 'b', target: 'a' }] }, /duplicate edge id "1"/],
  ])('rejects %s and keeps the current graph', (_name, data, message) => {
    const s = store();
    s.setData({ nodes: [{ id: 'keep', label: 'Keep' }, { id: 'me' }], edges: [{ source: 'keep', target: 'me' }] });
    expect(() => s.setData(data)).toThrow(message);
    expect(s.nodeCount).toBe(2);
    expect(s.edgeCount).toBe(1);
    expect(s.getNodeLabel(s.getNodeIndex('keep'))).toBe('Keep');
  });
});

describe('parseColor', () => {
  it.each([
    ['#f80', 0xff8800ff],
    ['#ff880080', 0xff880080],
    ['rgb(255, 136, 0)', 0xff8800ff],
    ['rgba(255, 136, 0, 0.5)', 0xff880080],
    ['rgb(100% 0% 50% / 50%)', 0xff008080],
    [0x11223344, 0x11223344],
  ])('%s → %s', (input, expected) => {
    expect(parseColor(input, 0)).toBe(expected >>> 0);
  });
  it('falls back on invalid input', () => {
    expect(parseColor('not-a-color', 0xabcdef12)).toBe(0xabcdef12);
    expect(parseColor(Number.NaN, 7)).toBe(7);
  });
});

describe('O3DGraph startup', () => {
  it('does not simulate (or schedule frames) before there is data', () => {
    const sched = createManualScheduler();
    const graph = new O3DGraph(createMockEngine(), { canvas: new FakeCanvas() as unknown as HTMLCanvasElement }, { scheduler: sched, interaction: false });
    expect(graph.physics.isActive).toBe(false);
    sched.step(5);
    expect(sched.queue).toHaveLength(0);
    expect(graph.physics.tickCount).toBe(0);
    graph.addNode({ id: 'first' });
    expect(graph.physics.isActive).toBe(true);
    expect(sched.queue).toHaveLength(1);
  });
});
