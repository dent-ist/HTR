// Pure coordinate maths for the page canvas. "World" = working-copy image pixels (what the API uses);
// "screen" = CSS pixels inside the canvas element. screen = world * scale + (tx, ty).
import type { Point } from "../api/types";

export interface Viewport {
  scale: number;
  tx: number;
  ty: number;
}

export interface Size {
  width: number;
  height: number;
}

export interface Box {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export const MIN_SCALE = 0.02;
export const MAX_SCALE = 16;

export const clampScale = (s: number): number => Math.min(MAX_SCALE, Math.max(MIN_SCALE, s));

export function worldToScreen(vp: Viewport, p: Point): Point {
  return [p[0] * vp.scale + vp.tx, p[1] * vp.scale + vp.ty];
}

export function screenToWorld(vp: Viewport, p: Point): Point {
  return [(p[0] - vp.tx) / vp.scale, (p[1] - vp.ty) / vp.scale];
}

/** Scale so the image fills the view width, left edge at x=0, top of the image at y=0. */
export function fitWidth(image: Size, view: Size): Viewport {
  const scale = clampScale(view.width / Math.max(1, image.width));
  return { scale, tx: 0, ty: 0 };
}

/** Zoom by `factor` keeping the world point under screen point (cx, cy) fixed. */
export function zoomAt(vp: Viewport, cx: number, cy: number, factor: number): Viewport {
  const scale = clampScale(vp.scale * factor);
  const k = scale / vp.scale;
  return { scale, tx: cx - (cx - vp.tx) * k, ty: cy - (cy - vp.ty) * k };
}

export function panBy(vp: Viewport, dx: number, dy: number): Viewport {
  return { ...vp, tx: vp.tx + dx, ty: vp.ty + dy };
}

/** Wheel delta (deltaY, pixels) to a zoom factor; negative delta (scroll up) zooms in. */
export function wheelFactor(deltaY: number): number {
  return Math.exp(-Math.max(-200, Math.min(200, deltaY)) * 0.0015);
}

export function polygonBox(poly: Point[]): Box {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const [x, y] of poly) {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  return { minX, minY, maxX, maxY };
}

/**
 * Smallest pan that brings `box` (world coords) fully inside the view with `margin` screen pixels to spare.
 * Returns the same object when it is already visible, so callers can skip redraws. If the box is larger than
 * the view on an axis it is centred on that axis. The scale is never changed.
 */
export function ensureVisible(vp: Viewport, box: Box, view: Size, margin = 24): Viewport {
  const axis = (min: number, max: number, t: number, len: number): number => {
    const lo = min * vp.scale + t;
    const hi = max * vp.scale + t;
    if (hi - lo + 2 * margin > len) return t + (len / 2 - (lo + hi) / 2);
    if (lo < margin) return t + (margin - lo);
    if (hi > len - margin) return t - (hi - (len - margin));
    return t;
  };
  const tx = axis(box.minX, box.maxX, vp.tx, view.width);
  const ty = axis(box.minY, box.maxY, vp.ty, view.height);
  return tx === vp.tx && ty === vp.ty ? vp : { ...vp, tx, ty };
}

export function pointInPolygon(p: Point, poly: Point[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]!;
    const [xj, yj] = poly[j]!;
    if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export function polygonArea(poly: Point[]): number {
  let a = 0;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    a += poly[j]![0] * poly[i]![1] - poly[i]![0] * poly[j]![1];
  }
  return Math.abs(a) / 2;
}

function distToSegment(p: Point, a: Point, b: Point): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}

interface Hittable {
  id: number;
  polygon: Point[];
  baseline: Point[];
}

/**
 * Line under a world point. Polygons overlap on handwriting, so among all containing polygons pick the one whose
 * baseline is closest (ties: smaller area). If none contains the point, null.
 */
export function hitTestLine(lines: Hittable[], world: Point): number | null {
  let best: { id: number; d: number; area: number } | null = null;
  for (const l of lines) {
    if (!pointInPolygon(world, l.polygon)) continue;
    let d = Infinity;
    for (let i = 1; i < l.baseline.length; i++) d = Math.min(d, distToSegment(world, l.baseline[i - 1]!, l.baseline[i]!));
    const area = polygonArea(l.polygon);
    if (!best || d < best.d - 1e-9 || (Math.abs(d - best.d) <= 1e-9 && area < best.area)) best = { id: l.id, d, area };
  }
  return best ? best.id : null;
}
