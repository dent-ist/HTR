import { describe, expect, it } from "vitest";
import type { Point } from "../api/types";
import { baselineYAt, canSplitAt, clampPoint, clampRect, clipPolygonToRect, overlapShare, rectFromPoints } from "./geometry";

const sq: Point[] = [
  [0, 0],
  [10, 0],
  [10, 10],
  [0, 10],
];

describe("geometry helpers", () => {
  it("builds normalised rects from any two corners", () => {
    expect(rectFromPoints([10, 20], [4, 5])).toEqual({ x: 4, y: 5, w: 6, h: 15 });
  });

  it("clamps points and rects to the page", () => {
    expect(clampPoint([-5, 120], { width: 100, height: 100 })).toEqual([0, 100]);
    expect(clampRect({ x: -10, y: 90, w: 50, h: 50 }, { width: 100, height: 100 })).toEqual({ x: 0, y: 90, w: 40, h: 10 });
  });

  it("clips a polygon to a rectangle", () => {
    const c = clipPolygonToRect(sq, { x: 5, y: 0, w: 20, h: 10 });
    expect(c.map((p) => p[0])).toSatisfy((xs: number[]) => Math.min(...xs) === 5 && Math.max(...xs) === 10);
    expect(clipPolygonToRect(sq, { x: 50, y: 50, w: 5, h: 5 })).toEqual([]);
  });

  it("computes the overlap share (server threshold is 0.5)", () => {
    expect(overlapShare(sq, { x: 0, y: 0, w: 5, h: 10 })).toBeCloseTo(0.5);
    expect(overlapShare(sq, { x: 0, y: 0, w: 4, h: 10 })).toBeCloseTo(0.4);
    expect(overlapShare(sq, { x: -5, y: -5, w: 100, h: 100 })).toBeCloseTo(1);
    expect(overlapShare(sq, { x: 20, y: 20, w: 5, h: 5 })).toBe(0);
    expect(overlapShare(sq, { x: 0, y: 0, w: 0, h: 5 })).toBe(0);
  });

  it("interpolates the baseline", () => {
    const b: Point[] = [
      [0, 10],
      [100, 20],
      [200, 20],
    ];
    expect(baselineYAt(b, 50)).toBeCloseTo(15);
    expect(baselineYAt(b, 150)).toBe(20);
    expect(baselineYAt(b, 250)).toBeNull();
  });

  it("only allows cuts strictly inside the baseline", () => {
    const b: Point[] = [
      [10, 5],
      [110, 5],
    ];
    expect(canSplitAt(b, 60)).toBe(true);
    expect(canSplitAt(b, 10)).toBe(false);
    expect(canSplitAt(b, 11)).toBe(true);
    expect(canSplitAt(b, 500)).toBe(false);
  });
});
