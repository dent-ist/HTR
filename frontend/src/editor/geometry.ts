// Pure client-side geometry for repair previews. The server does the authoritative work; these only draw ghosts
// (which lines a region would ignore, where a split would cut).
import type { Point, Rect } from "../api/types";
import { polygonArea } from "./viewport";

export type { Rect };

export const rectFromPoints = (a: Point, b: Point): Rect => ({
  x: Math.min(a[0], b[0]),
  y: Math.min(a[1], b[1]),
  w: Math.abs(a[0] - b[0]),
  h: Math.abs(a[1] - b[1]),
});

export function clampPoint(p: Point, size: { width: number; height: number }): Point {
  return [Math.min(size.width, Math.max(0, p[0])), Math.min(size.height, Math.max(0, p[1]))];
}

export function clampRect(r: Rect, size: { width: number; height: number }): Rect {
  const x0 = Math.min(size.width, Math.max(0, r.x));
  const y0 = Math.min(size.height, Math.max(0, r.y));
  const x1 = Math.min(size.width, Math.max(0, r.x + r.w));
  const y1 = Math.min(size.height, Math.max(0, r.y + r.h));
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

/** Sutherland-Hodgman clip of a polygon against an axis-aligned rectangle. */
export function clipPolygonToRect(poly: Point[], r: Rect): Point[] {
  const edges: [(p: Point) => boolean, (a: Point, b: Point) => Point][] = [
    [(p) => p[0] >= r.x, (a, b) => [r.x, a[1] + ((b[1] - a[1]) * (r.x - a[0])) / (b[0] - a[0])]],
    [(p) => p[0] <= r.x + r.w, (a, b) => [r.x + r.w, a[1] + ((b[1] - a[1]) * (r.x + r.w - a[0])) / (b[0] - a[0])]],
    [(p) => p[1] >= r.y, (a, b) => [a[0] + ((b[0] - a[0]) * (r.y - a[1])) / (b[1] - a[1]), r.y]],
    [(p) => p[1] <= r.y + r.h, (a, b) => [a[0] + ((b[0] - a[0]) * (r.y + r.h - a[1])) / (b[1] - a[1]), r.y + r.h]],
  ];
  let out = poly;
  for (const [inside, cross] of edges) {
    const input = out;
    out = [];
    for (let i = 0; i < input.length; i++) {
      const cur = input[i]!;
      const prev = input[(i + input.length - 1) % input.length]!;
      if (inside(cur)) {
        if (!inside(prev)) out.push(cross(prev, cur));
        out.push(cur);
      } else if (inside(prev)) {
        out.push(cross(prev, cur));
      }
    }
    if (out.length === 0) break;
  }
  return out;
}

/** Share (0..1) of the polygon's area inside the rectangle; the server uses >= 0.5 to ignore a line. */
export function overlapShare(poly: Point[], r: Rect): number {
  const area = polygonArea(poly);
  if (area <= 0 || r.w <= 0 || r.h <= 0) return 0;
  const clipped = clipPolygonToRect(poly, r);
  return clipped.length < 3 ? 0 : polygonArea(clipped) / area;
}

export const IGNORE_OVERLAP = 0.5;

/** Baseline y at x by linear interpolation; null outside the baseline's x range. */
export function baselineYAt(baseline: Point[], x: number): number | null {
  for (let i = 1; i < baseline.length; i++) {
    const a = baseline[i - 1]!;
    const b = baseline[i]!;
    const lo = Math.min(a[0], b[0]);
    const hi = Math.max(a[0], b[0]);
    if (x >= lo && x <= hi && a[0] !== b[0]) return a[1] + ((b[1] - a[1]) * (x - a[0])) / (b[0] - a[0]);
  }
  return null;
}

/** True when x is a valid cut: strictly inside the baseline, one pixel away from both ends (as the server requires). */
export function canSplitAt(baseline: Point[], x: number): boolean {
  const xs = baseline.map((p) => p[0]);
  return x >= Math.min(...xs) + 1 && x <= Math.max(...xs) - 1;
}
