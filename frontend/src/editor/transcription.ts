// Pure state machine for the keyboard-first transcription flow. No React, no network: side effects are expressed as
// an "outbox" of PATCH requests that the hook (useTranscription) sends, then reports back with saveOk / saveFail.
// Repair modes (merge, split, redraw baseline) will add their own actions and request kinds here later.
import type { Line, LinePatch } from "../api/types";

export interface SaveRequest {
  reqId: number;
  lineId: number;
  kind: "save" | "ignore";
  patch: LinePatch;
  /** The line as it was before the optimistic update, restored if the request fails. */
  prev: Line;
}

export interface EditorState {
  lines: Line[];
  selectedId: number | null;
  /** Unsaved typed text per line id; only present while it differs from the saved text. Survives navigation. */
  drafts: Record<number, string>;
  /** Last failed save per line id (message shown to the user; the typed text stays in `drafts`). */
  errors: Record<number, string>;
  /** Requests waiting to be sent. */
  outbox: SaveRequest[];
  /** Requests sent and not yet answered, by reqId. */
  inflight: Record<number, SaveRequest>;
  /** Newest reqId per line; stale responses are ignored. */
  latest: Record<number, number>;
  nextReqId: number;
  notice: string | null;
}

export type EditorAction =
  | { type: "loaded"; lines: Line[]; selectId?: number | null }
  | { type: "select"; id: number }
  | { type: "move"; delta: 1 | -1 }
  | { type: "edit"; text: string }
  | { type: "revert" }
  | { type: "commit"; advance: boolean; lineId?: number }
  | { type: "toggleIgnore" }
  | { type: "dispatched"; reqId: number }
  | { type: "saveOk"; reqId: number; line: Line }
  | { type: "saveFail"; reqId: number; message: string }
  | { type: "dismissError"; lineId: number };

export const initialState: EditorState = {
  lines: [],
  selectedId: null,
  drafts: {},
  errors: {},
  outbox: [],
  inflight: {},
  latest: {},
  nextReqId: 1,
  notice: null,
};

const savedText = (l: Line): string => l.text ?? "";

export function selectedLine(s: EditorState): Line | null {
  return s.lines.find((l) => l.id === s.selectedId) ?? null;
}

/** What the text input shows for the selected line: the draft if any, else the saved text. */
export function currentText(s: EditorState): string {
  const l = selectedLine(s);
  if (!l) return "";
  return s.drafts[l.id] ?? savedText(l);
}

/** True while anything would be lost by leaving the page: unsent drafts, failed saves, requests still in flight. */
export function hasUnsaved(s: EditorState): boolean {
  return (
    Object.keys(s.drafts).length > 0 ||
    Object.keys(s.errors).length > 0 ||
    s.outbox.length > 0 ||
    Object.keys(s.inflight).length > 0
  );
}

/** Where to start: first active line that is not ground truth yet, else the first active one, else the first. */
export function initialSelection(lines: Line[]): number | null {
  const active = lines.filter((l) => !l.ignored);
  return (active.find((l) => !l.is_ground_truth) ?? active[0] ?? lines[0])?.id ?? null;
}

function indexOfSelected(s: EditorState): number {
  return s.lines.findIndex((l) => l.id === s.selectedId);
}

function nextActiveId(s: EditorState, from: number): number | null {
  for (let i = from + 1; i < s.lines.length; i++) if (!s.lines[i]!.ignored) return s.lines[i]!.id;
  return null;
}

function replaceLine(lines: Line[], id: number, f: (l: Line) => Line): Line[] {
  return lines.map((l) => (l.id === id ? f(l) : l));
}

function without<T>(rec: Record<number, T>, key: number): Record<number, T> {
  const { [key]: _drop, ...rest } = rec;
  void _drop;
  return rest;
}

function enqueue(s: EditorState, req: Omit<SaveRequest, "reqId">): EditorState {
  const reqId = s.nextReqId;
  return {
    ...s,
    nextReqId: reqId + 1,
    outbox: [...s.outbox, { ...req, reqId }],
    latest: { ...s.latest, [req.lineId]: reqId },
  };
}

function commit(s: EditorState, advance: boolean, lineId?: number): EditorState {
  const idx = lineId === undefined ? indexOfSelected(s) : s.lines.findIndex((l) => l.id === lineId);
  const line = s.lines[idx];
  if (!line) return s;
  const goNext = (st: EditorState): EditorState => {
    if (!advance) return st;
    const next = nextActiveId(st, idx);
    return next === null ? { ...st, notice: "Last active line on this page." } : { ...st, selectedId: next };
  };
  s = { ...s, notice: null };
  if (line.ignored) return goNext(s); // nothing to label on an ignored line
  const text = s.drafts[line.id] ?? savedText(line);
  if (text.trim() === "") {
    return {
      ...s,
      errors: { ...s.errors, [line.id]: "Cannot save an empty line as ground truth. Type text, or use Ctrl+I to ignore it." },
    };
  }
  if (line.is_ground_truth && text === savedText(line) && !s.errors[line.id]) return goNext(s); // already saved
  const optimistic: Line = { ...line, text, text_source: "manual", confidence: null, is_ground_truth: true };
  let next = enqueue(s, { lineId: line.id, kind: "save", patch: { text, is_ground_truth: true }, prev: line });
  next = {
    ...next,
    lines: replaceLine(next.lines, line.id, () => optimistic),
    drafts: without(next.drafts, line.id),
    errors: without(next.errors, line.id),
  };
  return goNext(next);
}

export function reducer(s: EditorState, a: EditorAction): EditorState {
  switch (a.type) {
    case "loaded":
      return {
        ...initialState,
        lines: a.lines,
        selectedId: a.selectId === undefined ? initialSelection(a.lines) : a.selectId,
      };
    case "select":
      return s.lines.some((l) => l.id === a.id) ? { ...s, selectedId: a.id, notice: null } : s;
    case "move": {
      const i = indexOfSelected(s);
      if (i < 0) return s;
      const j = Math.min(s.lines.length - 1, Math.max(0, i + a.delta));
      return { ...s, selectedId: s.lines[j]!.id, notice: null };
    }
    case "edit": {
      const l = selectedLine(s);
      if (!l) return s;
      const drafts =
        a.text === savedText(l) ? without(s.drafts, l.id) : { ...s.drafts, [l.id]: a.text };
      return { ...s, drafts, notice: null };
    }
    case "revert": {
      const l = selectedLine(s);
      if (!l) return s;
      return { ...s, drafts: without(s.drafts, l.id), errors: without(s.errors, l.id), notice: null };
    }
    case "commit":
      return commit(s, a.advance, a.lineId);
    case "toggleIgnore": {
      const l = selectedLine(s);
      if (!l) return s;
      const ignored = !l.ignored;
      const next = enqueue(s, { lineId: l.id, kind: "ignore", patch: { ignored }, prev: l });
      return { ...next, lines: replaceLine(next.lines, l.id, (x) => ({ ...x, ignored })), notice: null };
    }
    case "dispatched": {
      const req = s.outbox.find((r) => r.reqId === a.reqId);
      if (!req) return s;
      return {
        ...s,
        outbox: s.outbox.filter((r) => r.reqId !== a.reqId),
        inflight: { ...s.inflight, [a.reqId]: req },
      };
    }
    case "saveOk": {
      const req = s.inflight[a.reqId];
      if (!req) return s;
      const inflight = without(s.inflight, a.reqId);
      if (s.latest[req.lineId] !== a.reqId) return { ...s, inflight }; // a newer request owns this line now
      return {
        ...s,
        inflight,
        lines: replaceLine(s.lines, req.lineId, () => a.line),
        errors: without(s.errors, req.lineId),
      };
    }
    case "saveFail": {
      const req = s.inflight[a.reqId];
      if (!req) return s;
      const inflight = without(s.inflight, a.reqId);
      if (s.latest[req.lineId] !== a.reqId) return { ...s, inflight };
      const errors = { ...s.errors, [req.lineId]: a.message };
      // Roll the optimistic change back, but keep what the user typed so nothing is lost.
      const drafts =
        req.kind === "save" && req.patch.text !== undefined && req.patch.text !== savedText(req.prev)
          ? { ...s.drafts, [req.lineId]: req.patch.text }
          : s.drafts;
      return { ...s, inflight, errors, drafts, lines: replaceLine(s.lines, req.lineId, () => req.prev) };
    }
    case "dismissError":
      return { ...s, errors: without(s.errors, a.lineId) };
  }
}

// ---- keyboard mapping ---------------------------------------------------------------------------------------

export interface KeyInfo {
  key: string;
  ctrlKey: boolean;
  metaKey: boolean;
  shiftKey: boolean;
  altKey: boolean;
  isComposing?: boolean;
}

export type UiCommand = "help" | "nextPage" | "prevPage" | "fitWidth" | "closeOverlay";

export type KeyResult =
  | { kind: "editor"; action: EditorAction }
  | { kind: "ui"; command: UiCommand }
  | null;

/** The documented shortcuts; also rendered by the help overlay so docs and behaviour cannot drift. */
export const SHORTCUTS: { keys: string; what: string }[] = [
  { keys: "Enter", what: "Save text, mark as ground truth, go to the next active line" },
  { keys: "Tab / Shift+Tab", what: "Next / previous line (any line, without saving; typed text is kept)" },
  { keys: "Esc", what: "Revert unsaved edits on the current line (closes this help if open)" },
  { keys: "Ctrl+I", what: "Ignore / un-ignore the current line" },
  { keys: "Alt+PageDown / Alt+PageUp", what: "Next / previous page" },
  { keys: "Alt+0", what: "Fit page to width" },
  { keys: "Ctrl+/ , F1, or ? outside the text box", what: "Show / hide this help" },
  { keys: "Mouse wheel", what: "Zoom at the cursor" },
  { keys: "Drag", what: "Pan the page" },
  { keys: "Click a line", what: "Select it" },
];

export function keyToAction(e: KeyInfo, ctx: { inTextField: boolean; overlayOpen?: boolean }): KeyResult {
  if (e.isComposing) return null; // IME in progress: Enter confirms the composition, not the line
  const mod = e.ctrlKey || e.metaKey;
  const plain = !mod && !e.altKey;
  const key = e.key;
  if (key === "Escape") {
    return ctx.overlayOpen ? { kind: "ui", command: "closeOverlay" } : { kind: "editor", action: { type: "revert" } };
  }
  if (plain && !e.shiftKey && key === "Enter") return { kind: "editor", action: { type: "commit", advance: true } };
  if (plain && key === "Tab") return { kind: "editor", action: { type: "move", delta: e.shiftKey ? -1 : 1 } };
  if (mod && !e.altKey && !e.shiftKey && key.toLowerCase() === "i") return { kind: "editor", action: { type: "toggleIgnore" } };
  if (e.altKey && !mod && key === "PageDown") return { kind: "ui", command: "nextPage" };
  if (e.altKey && !mod && key === "PageUp") return { kind: "ui", command: "prevPage" };
  if (e.altKey && !mod && key === "0") return { kind: "ui", command: "fitWidth" };
  if (mod && key === "/") return { kind: "ui", command: "help" };
  if (key === "F1") return { kind: "ui", command: "help" };
  if (!ctx.inTextField && plain && key === "?") return { kind: "ui", command: "help" };
  return null;
}
