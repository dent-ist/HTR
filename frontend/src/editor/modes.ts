// Pure state machine for the repair modes (select, merge, split, draw/redraw baseline, region ignore).
// It never calls the API: confirming a mode yields a Command that the editor executes.
import type { Line, Point } from "../api/types";
import { type Rect, canSplitAt, clampPoint, clampRect, rectFromPoints } from "./geometry";

export type ModeKind = "select" | "merge" | "split" | "baseline" | "region";

export type Mode =
  | { kind: "select" }
  | { kind: "merge"; picked: number[] }
  | { kind: "split"; preview: { lineId: number; x: number; y: number } | null }
  | { kind: "baseline"; points: Point[]; redrawId: number | null }
  | { kind: "region"; anchor: Point | null; rect: Rect | null };

export const SELECT: Mode = { kind: "select" };

export type Command =
  | { type: "merge"; ids: number[] }
  | { type: "split"; lineId: number; x: number; y: number }
  | { type: "addBaseline"; points: Point[] }
  | { type: "redrawBaseline"; lineId: number; points: Point[] }
  | { type: "ignoreRegion"; rect: Rect };

export type ModeEvent =
  | { type: "enter"; mode: "merge" | "split" | "baseline" | "redraw" | "region"; selectedId: number | null }
  | { type: "cancel" }
  | { type: "confirm" }
  | { type: "back" }
  | { type: "click"; world: Point; hitId: number | null; tol: number }
  | { type: "regionStart"; world: Point }
  | { type: "regionMove"; world: Point }
  | { type: "regionEnd"; world: Point };

export interface ModeContext {
  lines: Line[];
  size: { width: number; height: number };
}

export interface ModeResult {
  mode: Mode;
  command: Command | null;
  notice: string | null;
}

export const MIN_REGION = 4; // world px; smaller drags are treated as accidental

export const MODE_HELP: Record<ModeKind, string> = {
  select: "Select: click a line to select it.",
  merge: "Merge: click lines to add or remove them, Enter to merge, Esc to cancel.",
  split: "Split: click a point on a line to preview the cut, click it again or press Enter to cut, Esc to cancel.",
  baseline: "Draw baseline: click 2 or more points along the text, Backspace removes the last point, Enter finishes, Esc cancels.",
  region: "Ignore region: drag a rectangle; lines at least half inside it are ignored. Esc to cancel.",
};

export function modeHelp(m: Mode): string {
  if (m.kind === "baseline" && m.redrawId !== null) {
    return "Redraw baseline: click 2 or more points for the selected line, Backspace removes the last point, Enter finishes, Esc cancels.";
  }
  return MODE_HELP[m.kind];
}

const stay = (mode: Mode, notice: string | null = null): ModeResult => ({ mode, command: null, notice });
const done = (command: Command): ModeResult => ({ mode: SELECT, command, notice: null });

const dist = (a: Point, b: Point) => Math.hypot(a[0] - b[0], a[1] - b[1]);

export function modeReducer(mode: Mode, ev: ModeEvent, ctx: ModeContext): ModeResult {
  if (ev.type === "enter") {
    switch (ev.mode) {
      case "merge":
        return stay({ kind: "merge", picked: ev.selectedId === null ? [] : [ev.selectedId] });
      case "split":
        return stay({ kind: "split", preview: null });
      case "baseline":
        return stay({ kind: "baseline", points: [], redrawId: null });
      case "redraw":
        return ev.selectedId === null
          ? stay(mode, "Select a line first, then redraw its baseline.")
          : stay({ kind: "baseline", points: [], redrawId: ev.selectedId });
      case "region":
        return stay({ kind: "region", anchor: null, rect: null });
    }
  }
  if (ev.type === "cancel") return stay(SELECT);

  switch (mode.kind) {
    case "select":
      return stay(mode);

    case "merge": {
      if (ev.type === "click") {
        if (ev.hitId === null) return stay(mode);
        const picked = mode.picked.includes(ev.hitId) ? mode.picked.filter((i) => i !== ev.hitId) : [...mode.picked, ev.hitId];
        return stay({ kind: "merge", picked });
      }
      if (ev.type === "confirm") {
        return mode.picked.length >= 2 ? done({ type: "merge", ids: mode.picked }) : stay(mode, "Pick at least two lines to merge.");
      }
      return stay(mode);
    }

    case "split": {
      if (ev.type === "click") {
        const p = mode.preview;
        if (p && dist([p.x, p.y], ev.world) <= ev.tol) return done({ type: "split", lineId: p.lineId, x: p.x, y: p.y });
        if (ev.hitId === null) return stay({ kind: "split", preview: null }, "Click on a line to cut it.");
        const line = ctx.lines.find((l) => l.id === ev.hitId);
        const [x, y] = clampPoint(ev.world, ctx.size);
        const cx = Math.round(x);
        if (!line || !canSplitAt(line.baseline, cx)) return stay(mode, "Cut inside the line's baseline, not at its very ends.");
        return stay({ kind: "split", preview: { lineId: ev.hitId, x: cx, y: Math.round(y) } });
      }
      if (ev.type === "confirm") {
        return mode.preview
          ? done({ type: "split", lineId: mode.preview.lineId, x: mode.preview.x, y: mode.preview.y })
          : stay(mode, "Click a point on a line first.");
      }
      return stay(mode);
    }

    case "baseline": {
      if (ev.type === "click") {
        const [x, y] = clampPoint(ev.world, ctx.size);
        return stay({ ...mode, points: [...mode.points, [Math.round(x), Math.round(y)]] });
      }
      if (ev.type === "back") return stay({ ...mode, points: mode.points.slice(0, -1) });
      if (ev.type === "confirm") {
        if (mode.points.length < 2) return stay(mode, "A baseline needs at least 2 points.");
        return done(
          mode.redrawId === null
            ? { type: "addBaseline", points: mode.points }
            : { type: "redrawBaseline", lineId: mode.redrawId, points: mode.points },
        );
      }
      return stay(mode);
    }

    case "region": {
      if (ev.type === "regionStart") {
        const a = clampPoint(ev.world, ctx.size);
        return stay({ kind: "region", anchor: a, rect: { x: a[0], y: a[1], w: 0, h: 0 } });
      }
      if (ev.type === "regionMove" && mode.anchor) {
        return stay({ ...mode, rect: clampRect(rectFromPoints(mode.anchor, ev.world), ctx.size) });
      }
      if (ev.type === "regionEnd" && mode.anchor) {
        const rect = clampRect(rectFromPoints(mode.anchor, ev.world), ctx.size);
        if (rect.w < MIN_REGION || rect.h < MIN_REGION) return stay({ kind: "region", anchor: null, rect: null }, "Drag a larger rectangle.");
        return done({ type: "ignoreRegion", rect });
      }
      return stay(mode);
    }
  }
}
