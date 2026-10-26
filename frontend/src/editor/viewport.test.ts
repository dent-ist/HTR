import { describe, expect, it } from "vitest";
import {
  MAX_SCALE,
  MIN_SCALE,
  ensureVisible,
  fitWidth,
  hitTestLine,
  panBy,
  pointInPolygon,
  polygonBox,
  screenToWorld,
  wheelFactor,
  worldToScreen,
  zoomAt,
} from "./viewport";
import type { Point } from "../api/types";

const close = (a: number, b: number) => expect(a).toBeCloseTo(b, 6);

describe("viewport transforms", () => {
  it("round-trips world and screen", () => {
    const vp = { scale: 2.5, tx: -30, ty: 12 };
    const s = worldToScreen(vp, [100, 40]);
    expect(s).toEqual([220, 112]);
    const w = screenToWorld(vp, s);
    close(w[0], 100);
    close(w[1], 40);
  });

  it("fits width", () => {
    expect(fitWidth({ width: 2000, height: 3000 }, { width: 500, height: 400 })).toEqual({ scale: 0.25, tx: 0, ty: 0 });
  });

  it("zooms keeping the point under the cursor fixed", () => {
    const vp = { scale: 1, tx: 10, ty: 20 };
    const before = screenToWorld(vp, [300, 200]);
    const z = zoomAt(vp, 300, 200, 2);
    expect(z.scale).toBe(2);
    const after = screenToWorld(z, [300, 200]);
    close(after[0], before[0]);
    close(after[1], before[1]);
  });

  it("clamps zoom", () => {
    expect(zoomAt({ scale: 1, tx: 0, ty: 0 }, 0, 0, 1e9).scale).toBe(MAX_SCALE);
    expect(zoomAt({ scale: 1, tx: 0, ty: 0 }, 0, 0, 1e-9).scale).toBe(MIN_SCALE);
  });

  it("pans", () => {
    expect(panBy({ scale: 2, tx: 1, ty: 2 }, 5, -3)).toEqual({ scale: 2, tx: 6, ty: -1 });
  });

  it("wheel up zooms in, down zooms out, large deltas are bounded", () => {
    expect(wheelFactor(-100)).toBeGreaterThan(1);
    expect(wheelFactor(100)).toBeLessThan(1);
    expect(wheelFactor(-1e6)).toBe(wheelFactor(-200));
  });
});

describe("ensureVisible", () => {
  const view = { width: 400, height: 300 };
  const box = (minX: number, minY: number, maxX: number, maxY: number) => ({ minX, minY, maxX, maxY });

  it("returns the same viewport when already visible", () => {
    const vp = { scale: 1, tx: 0, ty: 0 };
    expect(ensureVisible(vp, box(50, 50, 150, 80), view)).toBe(vp);
  });

  it("pans minimally to reveal a box below the view", () => {
    const vp = { scale: 1, tx: 0, ty: 0 };
    const r = ensureVisible(vp, box(50, 400, 150, 440), view, 20);
    expect(r.tx).toBe(0);
    expect(r.ty).toBe(-(440 - (300 - 20))); // bottom edge lands at height - margin
    expect(worldToScreen(r, [0, 440])[1]).toBe(280);
  });

  it("pans to reveal a box above/left", () => {
    const vp = { scale: 2, tx: 0, ty: 0 };
    const r = ensureVisible(vp, box(-30, -10, 10, 10), view, 10);
    expect(worldToScreen(r, [-30, -10])).toEqual([10, 10]);
  });

  it("centres a box larger than the view", () => {
    const vp = { scale: 1, tx: 0, ty: 0 };
    const r = ensureVisible(vp, box(0, 0, 1000, 20), view);
    expect(r.tx).toBe(-300); // box centre (500) -> view centre (200)
  });
});

describe("hit testing", () => {
  const sq: Point[] = [
    [0, 0],
    [10, 0],
    [10, 10],
    [0, 10],
  ];
  it("point in polygon", () => {
    expect(pointInPolygon([5, 5], sq)).toBe(true);
    expect(pointInPolygon([11, 5], sq)).toBe(false);
    expect(pointInPolygon([-1, 5], sq)).toBe(false);
  });

  it("polygon box", () => {
    expect(polygonBox(sq)).toEqual({ minX: 0, minY: 0, maxX: 10, maxY: 10 });
  });

  it("picks the line with the nearest baseline where polygons overlap", () => {
    const a = {
      id: 1,
      polygon: [[0, 0], [100, 0], [100, 40], [0, 40]] as Point[],
      baseline: [[0, 15], [100, 15]] as Point[],
    };
    const b = {
      id: 2,
      polygon: [[0, 20], [100, 20], [100, 60], [0, 60]] as Point[],
      baseline: [[0, 45], [100, 45]] as Point[],
    };
    expect(hitTestLine([a, b], [50, 10])).toBe(1); // only in a
    expect(hitTestLine([a, b], [50, 50])).toBe(2); // only in b
    expect(hitTestLine([a, b], [50, 25])).toBe(1); // overlap: closer to a's baseline (10 vs 20)
    expect(hitTestLine([a, b], [50, 35])).toBe(2); // overlap: closer to b's baseline (10 vs 20)
    expect(hitTestLine([a, b], [500, 500])).toBeNull();
  });
});
