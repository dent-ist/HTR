import { describe, expect, it } from "vitest";
import type { Line } from "../api/types";
import {
  type EditorAction,
  type EditorState,
  type KeyInfo,
  currentText,
  hasUnsaved,
  initialSelection,
  initialState,
  keyToAction,
  reducer,
  selectedLine,
} from "./transcription";

function mk(id: number, over: Partial<Line> = {}): Line {
  return {
    id,
    page_id: 1,
    index: id - 1,
    baseline: [
      [0, 10 * id],
      [50, 10 * id],
    ],
    polygon: [
      [0, 10 * id - 5],
      [50, 10 * id - 5],
      [50, 10 * id + 5],
    ],
    text: null,
    text_source: null,
    confidence: null,
    is_ground_truth: false,
    ignored: false,
    crop_url: `/lines/${id}/crop.png`,
    ...over,
  };
}

const run = (s: EditorState, ...actions: EditorAction[]) => actions.reduce(reducer, s);
const load = (lines: Line[], selectId?: number | null) => run(initialState, { type: "loaded", lines, selectId });

describe("initial selection", () => {
  it("prefers the first active non-ground-truth line", () => {
    const lines = [mk(1, { is_ground_truth: true, text: "a" }), mk(2, { ignored: true }), mk(3), mk(4)];
    expect(initialSelection(lines)).toBe(3);
    expect(load(lines).selectedId).toBe(3);
  });
  it("falls back to first active, then first, then null", () => {
    expect(initialSelection([mk(1, { is_ground_truth: true, text: "a" }), mk(2, { is_ground_truth: true, text: "b" })])).toBe(1);
    expect(initialSelection([mk(1, { ignored: true }), mk(2, { ignored: true })])).toBe(1);
    expect(initialSelection([])).toBeNull();
  });
});

describe("navigation", () => {
  const s0 = load([mk(1), mk(2), mk(3)], 1);
  it("moves with clamping and without saving", () => {
    let s = run(s0, { type: "edit", text: "hello" }, { type: "move", delta: 1 });
    expect(s.selectedId).toBe(2);
    expect(s.outbox).toHaveLength(0);
    expect(s.drafts[1]).toBe("hello"); // typed text kept
    s = run(s, { type: "move", delta: -1 });
    expect(currentText(s)).toBe("hello");
    s = run(s, { type: "move", delta: -1 });
    expect(s.selectedId).toBe(1);
    s = run(s, { type: "move", delta: 1 }, { type: "move", delta: 1 }, { type: "move", delta: 1 });
    expect(s.selectedId).toBe(3);
  });
  it("Tab visits ignored lines too", () => {
    const s = run(load([mk(1), mk(2, { ignored: true }), mk(3)], 1), { type: "move", delta: 1 });
    expect(s.selectedId).toBe(2);
  });
  it("select ignores unknown ids", () => {
    expect(run(s0, { type: "select", id: 99 }).selectedId).toBe(1);
    expect(run(s0, { type: "select", id: 3 }).selectedId).toBe(3);
  });
});

describe("edit and revert", () => {
  it("tracks drafts only while they differ from saved text", () => {
    let s = load([mk(1, { text: "abc" })], 1);
    s = run(s, { type: "edit", text: "abcd" });
    expect(hasUnsaved(s)).toBe(true);
    s = run(s, { type: "edit", text: "abc" });
    expect(hasUnsaved(s)).toBe(false);
  });
  it("Esc reverts the current line only", () => {
    let s = load([mk(1, { text: "one" }), mk(2, { text: "two" })], 1);
    s = run(s, { type: "edit", text: "x" }, { type: "move", delta: 1 }, { type: "edit", text: "y" }, { type: "revert" });
    expect(currentText(s)).toBe("two");
    expect(s.drafts[1]).toBe("x");
  });
});

describe("commit (Enter)", () => {
  it("saves optimistically, marks ground truth and advances to the next active line", () => {
    const s0 = load([mk(1), mk(2, { ignored: true }), mk(3)], 1);
    const s = run(s0, { type: "edit", text: "first line" }, { type: "commit", advance: true });
    expect(s.selectedId).toBe(3); // skipped the ignored line 2
    expect(s.lines[0]).toMatchObject({ text: "first line", is_ground_truth: true, text_source: "manual" });
    expect(s.drafts).toEqual({});
    expect(s.outbox).toHaveLength(1);
    expect(s.outbox[0]).toMatchObject({ lineId: 1, kind: "save", patch: { text: "first line", is_ground_truth: true } });
    expect(hasUnsaved(s)).toBe(true); // until the server confirms
  });

  it("stays and reports a notice on the last active line", () => {
    const s = run(load([mk(1), mk(2, { ignored: true })], 1), { type: "edit", text: "x" }, { type: "commit", advance: true });
    expect(s.selectedId).toBe(1);
    expect(s.notice).toMatch(/last active line/i);
    expect(s.lines[0]!.is_ground_truth).toBe(true);
  });

  it("refuses empty text and keeps the selection", () => {
    const s = run(load([mk(1), mk(2)], 1), { type: "edit", text: "   " }, { type: "commit", advance: true });
    expect(s.selectedId).toBe(1);
    expect(s.outbox).toHaveLength(0);
    expect(s.errors[1]).toMatch(/empty/i);
    expect(s.drafts[1]).toBe("   ");
  });

  it("does not resend an unchanged ground-truth line, just advances", () => {
    const s = run(load([mk(1, { text: "a", is_ground_truth: true }), mk(2)], 1), { type: "commit", advance: true });
    expect(s.selectedId).toBe(2);
    expect(s.outbox).toHaveLength(0);
  });

  it("confirms existing non-ground-truth text without editing", () => {
    const s = run(load([mk(1, { text: "a", text_source: "manual" })], 1), { type: "commit", advance: false });
    expect(s.outbox[0]!.patch).toEqual({ text: "a", is_ground_truth: true });
  });

  it("only advances on an ignored line", () => {
    const s = run(load([mk(1, { ignored: true }), mk(2)], 1), { type: "commit", advance: true });
    expect(s.selectedId).toBe(2);
    expect(s.outbox).toHaveLength(0);
  });
});

describe("request lifecycle", () => {
  const start = () => run(load([mk(1), mk(2)], 1), { type: "edit", text: "typed" }, { type: "commit", advance: true });

  it("dispatched moves outbox to inflight; saveOk applies the server line and clears unsaved", () => {
    let s = start();
    const reqId = s.outbox[0]!.reqId;
    s = run(s, { type: "dispatched", reqId });
    expect(s.outbox).toHaveLength(0);
    expect(s.inflight[reqId]).toBeDefined();
    const server = mk(1, { text: "typed", text_source: "manual", is_ground_truth: true });
    s = run(s, { type: "saveOk", reqId, line: server });
    expect(s.lines[0]).toBe(server);
    expect(hasUnsaved(s)).toBe(false);
  });

  it("saveFail rolls back, keeps the typed text and shows the error", () => {
    let s = start();
    const reqId = s.outbox[0]!.reqId;
    s = run(s, { type: "dispatched", reqId }, { type: "saveFail", reqId, message: "HTTP 500" });
    expect(s.lines[0]).toMatchObject({ text: null, is_ground_truth: false });
    expect(s.drafts[1]).toBe("typed");
    expect(s.errors[1]).toBe("HTTP 500");
    expect(hasUnsaved(s)).toBe(true);
    expect(s.selectedId).toBe(2); // the user keeps moving
    // go back: the typed text is there, retry resubmits it
    s = run(s, { type: "select", id: 1 });
    expect(currentText(s)).toBe("typed");
    s = run(s, { type: "commit", advance: false });
    expect(s.outbox[0]).toMatchObject({ lineId: 1, patch: { text: "typed", is_ground_truth: true } });
    expect(s.errors[1]).toBeUndefined();
  });

  it("ignores stale responses when a newer request exists for the line", () => {
    let s = start();
    const first = s.outbox[0]!.reqId;
    s = run(s, { type: "dispatched", reqId: first }, { type: "select", id: 1 }, { type: "edit", text: "typed again" }, { type: "commit", advance: false });
    const second = s.outbox[0]!.reqId;
    s = run(s, { type: "dispatched", reqId: second });
    s = run(s, { type: "saveFail", reqId: first, message: "late failure" });
    expect(s.errors[1]).toBeUndefined();
    expect(s.lines[0]!.text).toBe("typed again");
    s = run(s, { type: "saveOk", reqId: second, line: mk(1, { text: "typed again", is_ground_truth: true }) });
    expect(hasUnsaved(s)).toBe(false);
  });

  it("ignore toggles optimistically and rolls back on failure", () => {
    let s = run(load([mk(1), mk(2)], 1), { type: "toggleIgnore" });
    expect(selectedLine(s)!.ignored).toBe(true);
    expect(s.outbox[0]).toMatchObject({ kind: "ignore", patch: { ignored: true } });
    const reqId = s.outbox[0]!.reqId;
    s = run(s, { type: "dispatched", reqId }, { type: "saveFail", reqId, message: "nope" });
    expect(selectedLine(s)!.ignored).toBe(false);
    expect(s.errors[1]).toBe("nope");
    expect(s.drafts).toEqual({}); // nothing typed, nothing to preserve
  });

  it("dismissError clears the flag", () => {
    let s = run(load([mk(1)], 1), { type: "edit", text: " " }, { type: "commit", advance: false });
    expect(s.errors[1]).toBeDefined();
    s = run(s, { type: "dismissError", lineId: 1 });
    expect(s.errors[1]).toBeUndefined();
  });
});

describe("keyToAction", () => {
  const k = (key: string, over: Partial<KeyInfo> = {}): KeyInfo => ({
    key,
    ctrlKey: false,
    metaKey: false,
    shiftKey: false,
    altKey: false,
    ...over,
  });
  const text = { inTextField: true };

  it("maps the transcription keys", () => {
    expect(keyToAction(k("Enter"), text)).toEqual({ kind: "editor", action: { type: "commit", advance: true } });
    expect(keyToAction(k("Tab"), text)).toEqual({ kind: "editor", action: { type: "move", delta: 1 } });
    expect(keyToAction(k("Tab", { shiftKey: true }), text)).toEqual({ kind: "editor", action: { type: "move", delta: -1 } });
    expect(keyToAction(k("Escape"), text)).toEqual({ kind: "editor", action: { type: "revert" } });
    expect(keyToAction(k("i", { ctrlKey: true }), text)).toEqual({ kind: "editor", action: { type: "toggleIgnore" } });
    expect(keyToAction(k("I", { ctrlKey: true }), text)).toEqual({ kind: "editor", action: { type: "toggleIgnore" } });
  });

  it("leaves normal typing alone", () => {
    expect(keyToAction(k("a"), text)).toBeNull();
    expect(keyToAction(k("?", { shiftKey: true }), text)).toBeNull(); // typing a question mark in the text box
    expect(keyToAction(k("Enter", { shiftKey: true }), text)).toBeNull();
    expect(keyToAction(k("Enter", { ctrlKey: true }), text)).toBeNull();
    expect(keyToAction(k("Enter", { isComposing: true }), text)).toBeNull();
  });

  it("help: ? only outside the text field, Ctrl+/ and F1 anywhere", () => {
    expect(keyToAction(k("?", { shiftKey: true }), { inTextField: false })).toEqual({ kind: "ui", command: "help" });
    expect(keyToAction(k("/", { ctrlKey: true }), text)).toEqual({ kind: "ui", command: "help" });
    expect(keyToAction(k("F1"), text)).toEqual({ kind: "ui", command: "help" });
  });

  it("Esc closes the overlay first", () => {
    expect(keyToAction(k("Escape"), { inTextField: true, overlayOpen: true })).toEqual({ kind: "ui", command: "closeOverlay" });
  });

  it("page navigation and fit", () => {
    expect(keyToAction(k("PageDown", { altKey: true }), text)).toEqual({ kind: "ui", command: "nextPage" });
    expect(keyToAction(k("PageUp", { altKey: true }), text)).toEqual({ kind: "ui", command: "prevPage" });
    expect(keyToAction(k("0", { altKey: true }), text)).toEqual({ kind: "ui", command: "fitWidth" });
  });
});
