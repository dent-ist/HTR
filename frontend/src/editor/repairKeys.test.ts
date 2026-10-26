import { describe, expect, it } from "vitest";
import type { Line } from "../api/types";
import { type KeyInfo, initialState, keyToAction, reducer } from "./transcription";

const k = (key: string, over: Partial<KeyInfo> = {}): KeyInfo => ({
  key,
  ctrlKey: false,
  metaKey: false,
  shiftKey: false,
  altKey: false,
  ...over,
});
const text = { inTextField: true };
const free = { inTextField: false };

describe("repair shortcuts", () => {
  it("Alt+letter works inside the text box (uses the physical key)", () => {
    const alt = (code: string, over: Partial<KeyInfo> = {}) => k("µ", { altKey: true, code, ...over });
    expect(keyToAction(alt("KeyM"), text)).toEqual({ kind: "repair", command: "mode:merge" });
    expect(keyToAction(alt("KeyJ"), text)).toEqual({ kind: "repair", command: "mergeNext" });
    expect(keyToAction(alt("KeyS"), text)).toEqual({ kind: "repair", command: "mode:split" });
    expect(keyToAction(alt("KeyB"), text)).toEqual({ kind: "repair", command: "mode:baseline" });
    expect(keyToAction(alt("KeyB", { shiftKey: true }), text)).toEqual({ kind: "repair", command: "mode:redraw" });
    expect(keyToAction(alt("KeyR"), text)).toEqual({ kind: "repair", command: "mode:region" });
    expect(keyToAction(alt("Delete"), text)).toEqual({ kind: "repair", command: "deleteLine" });
    expect(keyToAction(alt("Backspace"), text)).toEqual({ kind: "repair", command: "deleteLine" });
    expect(keyToAction(alt("ArrowUp"), text)).toEqual({ kind: "repair", command: "moveUp" });
    expect(keyToAction(alt("ArrowDown"), text)).toEqual({ kind: "repair", command: "moveDown" });
    expect(keyToAction(alt("KeyZ"), text)).toEqual({ kind: "repair", command: "undo" });
  });

  it("plain letters are typing in the text box but shortcuts elsewhere", () => {
    for (const key of ["m", "s", "b", "r", "j", "Delete"]) expect(keyToAction(k(key), text)).toBeNull();
    expect(keyToAction(k("m"), free)).toEqual({ kind: "repair", command: "mode:merge" });
    expect(keyToAction(k("B", { shiftKey: true }), free)).toEqual({ kind: "repair", command: "mode:redraw" });
    expect(keyToAction(k("Delete"), free)).toEqual({ kind: "repair", command: "deleteLine" });
  });

  it("Ctrl+Z is the text undo while the box has a draft, otherwise a repair undo", () => {
    const z = k("z", { ctrlKey: true });
    expect(keyToAction(z, { inTextField: true, hasDraft: true })).toBeNull();
    expect(keyToAction(z, { inTextField: true, hasDraft: false })).toEqual({ kind: "repair", command: "undo" });
    expect(keyToAction(z, free)).toEqual({ kind: "repair", command: "undo" });
    expect(keyToAction(k("z", { ctrlKey: true, shiftKey: true }), free)).toBeNull();
  });
});

function ln(id: number, over: Partial<Line> = {}): Line {
  return {
    id,
    page_id: 1,
    index: id - 1,
    baseline: [
      [0, 1],
      [9, 1],
    ],
    polygon: [
      [0, 0],
      [9, 0],
      [9, 2],
    ],
    text: null,
    text_source: null,
    confidence: null,
    is_ground_truth: false,
    ignored: false,
    crop_url: "",
    ...over,
  };
}

describe("replaceLines (after a repair operation)", () => {
  const start = reducer(initialState, { type: "loaded", lines: [ln(1), ln(2), ln(3)], selectId: 2 });

  it("keeps drafts of surviving lines and drops those of removed lines", () => {
    let s = reducer(start, { type: "edit", text: "typing on 2" });
    s = reducer(s, { type: "select", id: 3 });
    s = reducer(s, { type: "edit", text: "typing on 3" });
    s = reducer(s, { type: "replaceLines", lines: [ln(1), ln(3), ln(9)], selectId: 9 });
    expect(s.drafts).toEqual({ 3: "typing on 3" });
    expect(s.selectedId).toBe(9);
  });

  it("drops a draft that now equals the saved text and keeps a valid selection", () => {
    let s = reducer(start, { type: "edit", text: "same" });
    s = reducer(s, { type: "replaceLines", lines: [ln(1), ln(2, { text: "same", text_source: "manual" })] });
    expect(s.drafts).toEqual({});
    expect(s.selectedId).toBe(2);
  });

  it("falls back to the first open line when the selection disappeared", () => {
    const s = reducer(start, { type: "replaceLines", lines: [ln(1, { is_ground_truth: true, text: "a" }), ln(5)] });
    expect(s.selectedId).toBe(5);
  });
});
