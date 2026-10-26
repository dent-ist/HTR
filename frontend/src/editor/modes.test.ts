import { describe, expect, it } from "vitest";
import type { Line, Point } from "../api/types";
import { type Mode, type ModeContext, type ModeEvent, SELECT, modeReducer } from "./modes";

function line(id: number, x0 = 100, x1 = 500): Line {
  return {
    id,
    page_id: 1,
    index: id - 1,
    baseline: [
      [x0, 50 * id],
      [x1, 50 * id],
    ],
    polygon: [
      [x0, 50 * id - 15],
      [x1, 50 * id - 15],
      [x1, 50 * id + 10],
      [x0, 50 * id + 10],
    ],
    text: null,
    text_source: null,
    confidence: null,
    is_ground_truth: false,
    ignored: false,
    crop_url: "",
  };
}

const ctx: ModeContext = { lines: [line(1), line(2), line(3)], size: { width: 600, height: 400 } };
const step = (m: Mode, ev: ModeEvent) => modeReducer(m, ev, ctx);
const click = (world: Point, hitId: number | null, tol = 5): ModeEvent => ({ type: "click", world, hitId, tol });

describe("mode state machine", () => {
  it("Select ignores repair events and Esc always returns to Select", () => {
    expect(step(SELECT, { type: "confirm" }).mode).toBe(SELECT);
    for (const m of ["merge", "split", "baseline", "region"] as const) {
      const entered = step(SELECT, { type: "enter", mode: m, selectedId: 1 }).mode;
      expect(entered.kind).toBe(m);
      expect(step(entered, { type: "cancel" }).mode).toEqual(SELECT);
    }
  });

  describe("merge", () => {
    it("starts with the selected line, toggles lines, needs two to confirm", () => {
      let r = step(SELECT, { type: "enter", mode: "merge", selectedId: 2 });
      expect(r.mode).toEqual({ kind: "merge", picked: [2] });
      r = step(r.mode, { type: "confirm" });
      expect(r.command).toBeNull();
      expect(r.notice).toMatch(/at least two/i);
      r = step(r.mode, click([0, 0], 3));
      expect(r.mode).toEqual({ kind: "merge", picked: [2, 3] });
      r = step(r.mode, click([0, 0], 2)); // toggle off
      expect(r.mode).toEqual({ kind: "merge", picked: [3] });
      r = step(r.mode, click([0, 0], 1));
      r = step(r.mode, { type: "confirm" });
      expect(r.command).toEqual({ type: "merge", ids: [3, 1] });
      expect(r.mode).toEqual(SELECT);
    });
    it("clicking empty space changes nothing", () => {
      const m: Mode = { kind: "merge", picked: [1] };
      expect(step(m, click([5, 5], null)).mode).toBe(m);
    });
  });

  describe("split", () => {
    it("previews on first click, confirms with a click on the preview or Enter", () => {
      let r = step(SELECT, { type: "enter", mode: "split", selectedId: null });
      r = step(r.mode, click([300.4, 100.2], 2));
      expect(r.mode).toEqual({ kind: "split", preview: { lineId: 2, x: 300, y: 100 } });
      const moved = step(r.mode, click([350, 100], 2)); // elsewhere: moves the preview
      expect(moved.mode).toEqual({ kind: "split", preview: { lineId: 2, x: 350, y: 100 } });
      const byClick = step(r.mode, click([302, 101], 2, 5));
      expect(byClick.command).toEqual({ type: "split", lineId: 2, x: 300, y: 100 });
      expect(byClick.mode).toEqual(SELECT);
      const byEnter = step(r.mode, { type: "confirm" });
      expect(byEnter.command).toEqual({ type: "split", lineId: 2, x: 300, y: 100 });
    });
    it("rejects cuts at the ends of the baseline and clicks off lines", () => {
      const start: Mode = { kind: "split", preview: null };
      const end = step(start, click([100, 100], 2));
      expect(end.mode).toBe(start);
      expect(end.notice).toMatch(/baseline/);
      expect(step(start, click([10, 10], null)).notice).toMatch(/click on a line/i);
      expect(step(start, { type: "confirm" }).command).toBeNull();
    });
  });

  describe("baseline", () => {
    it("collects points, backspace removes the last, Enter needs two", () => {
      let r = step(SELECT, { type: "enter", mode: "baseline", selectedId: 1 });
      expect(r.mode).toEqual({ kind: "baseline", points: [], redrawId: null });
      r = step(r.mode, click([10.4, 20.6], null));
      expect(step(r.mode, { type: "confirm" }).notice).toMatch(/at least 2/);
      r = step(r.mode, click([700, -9], null)); // clamped to the page
      expect(r.mode).toEqual({
        kind: "baseline",
        points: [
          [10, 21],
          [600, 0],
        ],
        redrawId: null,
      });
      expect(step(r.mode, { type: "back" }).mode).toMatchObject({ points: [[10, 21]] });
      const fin = step(r.mode, { type: "confirm" });
      expect(fin.command).toEqual({
        type: "addBaseline",
        points: [
          [10, 21],
          [600, 0],
        ],
      });
      expect(fin.mode).toEqual(SELECT);
    });
    it("redraw targets the selected line and needs a selection", () => {
      expect(step(SELECT, { type: "enter", mode: "redraw", selectedId: null })).toMatchObject({ mode: SELECT, notice: expect.stringMatching(/select a line/i) });
      let r = step(SELECT, { type: "enter", mode: "redraw", selectedId: 3 });
      r = step(r.mode, click([1, 1], null));
      r = step(r.mode, click([50, 1], null));
      expect(step(r.mode, { type: "confirm" }).command).toMatchObject({ type: "redrawBaseline", lineId: 3 });
    });
  });

  describe("region", () => {
    it("drags a rectangle clamped to the page and emits ignoreRegion on release", () => {
      let r = step(SELECT, { type: "enter", mode: "region", selectedId: null });
      r = step(r.mode, { type: "regionStart", world: [100, 100] });
      r = step(r.mode, { type: "regionMove", world: [700, 50] });
      expect(r.mode).toMatchObject({ rect: { x: 100, y: 50, w: 500, h: 50 } });
      const end = step(r.mode, { type: "regionEnd", world: [300, 200] });
      expect(end.command).toEqual({ type: "ignoreRegion", rect: { x: 100, y: 100, w: 200, h: 100 } });
      expect(end.mode).toEqual(SELECT);
    });
    it("ignores accidental tiny drags", () => {
      let r = step(SELECT, { type: "enter", mode: "region", selectedId: null });
      r = step(r.mode, { type: "regionStart", world: [100, 100] });
      const end = step(r.mode, { type: "regionEnd", world: [101, 101] });
      expect(end.command).toBeNull();
      expect(end.mode).toEqual({ kind: "region", anchor: null, rect: null });
    });
  });
});
