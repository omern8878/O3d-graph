import { describe, expect, it } from 'vitest';
import { LabelLayer, NodeFlag, O3DGraph, type LabelLayerOptions } from '../src/index.js';
import { FakeCanvas, createManualScheduler, createMockEngine } from './helpers/mockEngine.js';

/** Records text draws; measureText ≈ 0.55 em per character at the current font size. */
function fakeOverlay() {
  const drawn: Array<{ text: string; x: number; y: number; color: string }> = [];
  const ctx = {
    font: '', textAlign: 'left', textBaseline: 'middle', lineJoin: 'round', lineWidth: 1,
    fillStyle: '', strokeStyle: '',
    setTransform() {}, clearRect() {}, save() {}, restore() {}, translate() {}, rotate() {},
    beginPath() {}, moveTo() {}, lineTo() {}, arcTo() {}, closePath() {}, fill() {}, strokeText() {},
    fillText(text: string, x: number, y: number) { drawn.push({ text, x, y, color: String(this.fillStyle) }); },
    measureText(text: string) {
      const size = Number(/(\d+(?:\.\d+)?)px/.exec(this.font)?.[1] ?? 10);
      return { width: text.length * size * 0.55 };
    },
  };
  const canvas = { width: 0, height: 0, getContext: () => ctx, remove() {} } as unknown as HTMLCanvasElement;
  return { canvas, drawn };
}

/** Overlay whose context records every font it is asked to draw with. */
function fontProbe(fonts: string[]): HTMLCanvasElement {
  const overlay = fakeOverlay();
  const ctx = overlay.canvas.getContext('2d') as unknown as { fillText: (...a: unknown[]) => void; font: string };
  const fill = ctx.fillText.bind(ctx);
  ctx.fillText = (...a: unknown[]) => {
    fonts.push(ctx.font);
    fill(...a);
  };
  return overlay.canvas;
}

function setup(options: LabelLayerOptions = {}) {
  const canvas = new FakeCanvas();
  const graph = new O3DGraph(
    createMockEngine(),
    { canvas: canvas as unknown as HTMLCanvasElement, node: { radius: 6 } },
    { scheduler: createManualScheduler(), interaction: false },
  );
  const overlay = fakeOverlay();
  const labels = new LabelLayer(graph, { canvas: overlay.canvas, ...options });
  return { graph, labels, drawn: overlay.drawn };
}

describe('LabelLayer', () => {
  it('labels the focused node first, then its neighbors and connecting edges', () => {
    const { graph, labels, drawn } = setup({ maxNodeLabels: 3, maxEdgeLabels: 2, focusColor: 'FOCUS' });
    graph.setData({
      nodes: [
        { id: 'hub', x: 0, y: 0, z: 0, label: 'Hub' },
        { id: 'a', x: -200, y: 120, z: 0, label: 'Alpha' },
        { id: 'b', x: 200, y: -120, z: 0, label: 'Beta' },
        // Far bigger on screen, but unrelated to the focus.
        { id: 'big', x: 150, y: 250, z: 300, radius: 30, label: 'Big' },
      ],
      edges: [
        { source: 'hub', target: 'a', label: 'mentors' },
        { source: 'hub', target: 'b', label: 'pairs with' },
      ],
    });
    graph.select('hub');
    labels.render();

    const texts = drawn.map((d) => d.text);
    expect(texts[0]).toBe('Hub');
    expect(drawn[0].color).toBe('FOCUS');
    expect(texts).toEqual(expect.arrayContaining(['Hub', 'mentors', 'pairs with', 'Alpha', 'Beta']));
    expect(texts).not.toContain('Big'); // budget of 3 went to the focus + neighbors
    expect(labels.placedNodeCount).toBe(3);
    expect(labels.placedEdgeCount).toBe(2);
  });

  it('without focus, prefers nodes that are largest on screen and respects the budget', () => {
    const { graph, labels } = setup({ maxNodeLabels: 2, edgeLabels: 'none' });
    graph.setData({
      nodes: [
        { id: 'small', x: -250, y: 0, z: 0, radius: 4, label: 'Small' },
        { id: 'near', x: 0, y: 200, z: 400, radius: 6, label: 'Near' }, // closer to the camera
        { id: 'large', x: 250, y: 0, z: 0, radius: 20, label: 'Large' },
      ],
    });
    labels.render();
    const placed = Array.from(labels.placedNodes.subarray(0, labels.placedNodeCount)).map((i) => graph.store.getNodeId(i));
    expect(placed).toEqual(['large', 'near']);
  });

  it('centers labels on the ball and sizes them from its on-screen radius', () => {
    const { graph, labels, drawn } = setup({ edgeLabels: 'none' });
    graph.setData({ nodes: [{ id: 'n', x: 0, y: 0, z: 0, radius: 20, label: 'Center' }] });
    labels.render();
    expect(drawn[0].x).toBeCloseTo(400, 1); // canvas is 800 px wide; node at the origin
    expect(drawn[0].y).toBeCloseTo(400, 1);
    const fontAt = (z: number) => {
      graph.updateNode('n', { z });
      const fonts: string[] = [];
      const layer = new LabelLayer(graph, { canvas: fontProbe(fonts), edgeLabels: 'none' });
      layer.render();
      return Number(/(\d+)px/.exec(fonts.at(-1)!)![1]);
    };
    expect(fontAt(500)).toBeGreaterThan(fontAt(0)); // closer → bigger text
  });

  it('skips overlapping labels, hidden nodes and nodes behind the camera', () => {
    const { graph, labels, drawn } = setup({ edgeLabels: 'none' });
    graph.setData({
      nodes: [
        { id: 'a', x: 0, y: 0, z: 0, radius: 10, label: 'Same spot A' },
        { id: 'b', x: 0, y: 0, z: 0, radius: 9, label: 'Same spot B' },
        { id: 'hidden', x: 100, y: 0, z: 0, label: 'Hidden', hidden: true },
        { id: 'behind', x: 0, y: 0, z: 900, label: 'Behind' },
      ],
    });
    labels.render();
    const texts = drawn.map((d) => d.text);
    // Centered labels of coincident nodes overlap: only the larger node's survives.
    expect(texts).toEqual(['Same spot A']);

    // Side placement can use both the right and the left of the ball.
    const side = new LabelLayer(graph, { canvas: fakeOverlay().canvas, edgeLabels: 'none', nodeLabelPosition: 'right' });
    side.render();
    expect(side.placedNodeCount).toBe(2);
  });

  it('reflects label edits and supports custom text sources', () => {
    const { graph, labels, drawn } = setup({ edgeLabels: 'none' });
    graph.setData({ nodes: [{ id: 'n', x: 0, y: 0, z: 0, label: 'Before' }] });
    labels.render();
    expect(drawn.at(-1)?.text).toBe('Before');
    graph.setNodeLabel('n', 'After');
    labels.render();
    expect(drawn.at(-1)?.text).toBe('After');
    graph.setNodeLabel('n', undefined);
    drawn.length = 0;
    labels.render();
    expect(drawn).toHaveLength(0);

    const custom = new LabelLayer(graph, { canvas: fakeOverlay().canvas, nodeText: (i) => `#${i}` });
    custom.render();
    expect(custom.placedNodeCount).toBe(1);
  });

  it('keeps labels attached through swap-and-pop removals', () => {
    const { graph, labels, drawn } = setup({ edgeLabels: 'none' });
    graph.setData({
      nodes: [
        { id: 'x', x: -100, y: 0, z: 0, label: 'X' },
        { id: 'y', x: 0, y: 0, z: 0, label: 'Y' },
        { id: 'z', x: 100, y: 0, z: 0, label: 'Z' },
      ],
    });
    graph.removeNode('x'); // 'z' moves into slot 0
    expect(graph.store.getNodeLabel(graph.store.getNodeIndex('z'))).toBe('Z');
    labels.render();
    expect(drawn.map((d) => d.text).sort()).toEqual(['Y', 'Z']);
    expect(graph.store.hasFlag(0, NodeFlag.HIDDEN)).toBe(false);
  });
});
