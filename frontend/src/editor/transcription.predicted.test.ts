import { describe, expect, it } from "vitest";
import type { Line } from "../api/types";
import { currentText, hasUnsaved, initialState, reducer, type EditorState } from "./transcription";

const pred = (id: number, text: string, conf = 0.8): Line => ({
  id,
  page_id: 1,
  index: id - 1,
  baseline: [],
  polygon: [],
  text,
  text_source: "predicted",
  confidence: conf,
  is_ground_truth: false,
  ignored: false,
  crop_url: "",
});

const loaded = (): EditorState => reducer(initialState, { type: "loaded", lines: [pred(1, "hello wxrld"), pred(2, "second")] });

describe("predicted lines in the editor", () => {
  it("prefill the text box with the prediction without counting as an unsaved edit", () => {
    const s = loaded();
    expect(s.selectedId).toBe(1);
    expect(currentText(s)).toBe("hello wxrld");
    expect(hasUnsaved(s)).toBe(false); // the navigation guard stays quiet
    expect(s.drafts).toEqual({});
  });

  it("Enter accepts the prediction unchanged: one PATCH with the predicted text as ground truth", () => {
    const s = reducer(loaded(), { type: "commit", advance: true });
    expect(s.outbox).toHaveLength(1);
    expect(s.outbox[0]!.patch).toEqual({ text: "hello wxrld", is_ground_truth: true });
    expect(s.lines[0]).toMatchObject({ is_ground_truth: true, text_source: "manual", confidence: null });
    expect(s.selectedId).toBe(2);
  });

  it("editing makes a draft (now unsaved) and Enter sends the corrected text", () => {
    let s = reducer(loaded(), { type: "edit", text: "hello world" });
    expect(hasUnsaved(s)).toBe(true);
    s = reducer(s, { type: "commit", advance: false });
    expect(s.outbox[0]!.patch).toEqual({ text: "hello world", is_ground_truth: true });
  });

  it("Esc restores the prediction and clears the unsaved state", () => {
    let s = reducer(loaded(), { type: "edit", text: "oops" });
    s = reducer(s, { type: "revert" });
    expect(currentText(s)).toBe("hello wxrld");
    expect(hasUnsaved(s)).toBe(false);
  });

  it("typing the prediction back exactly removes the draft again", () => {
    let s = reducer(loaded(), { type: "edit", text: "x" });
    s = reducer(s, { type: "edit", text: "hello wxrld" });
    expect(hasUnsaved(s)).toBe(false);
  });

  it("an unsaved correction survives moving to another line and back", () => {
    let s = reducer(loaded(), { type: "edit", text: "fixed" });
    s = reducer(s, { type: "move", delta: 1 });
    expect(currentText(s)).toBe("second");
    s = reducer(s, { type: "move", delta: -1 });
    expect(currentText(s)).toBe("fixed");
    expect(hasUnsaved(s)).toBe(true);
  });

  it("ignoring a prediction sends only the ignored flag", () => {
    const s = reducer(loaded(), { type: "toggleIgnore" });
    expect(s.outbox[0]!.patch).toEqual({ ignored: true });
  });
});
