import { useCallback, useRef, useState } from "react";
import { ApiError, type ApiClient } from "../api/client";
import type { Line } from "../api/types";
import { type Command, type Mode, type ModeEvent, SELECT, modeReducer } from "./modes";
import { type EditorAction, type EditorState, initialSelection, selectedLine } from "./transcription";

const MAX_UNDO = 50;

export interface UndoEntry {
  label: string;
  /** The page's lines before the operation (server truth, saves flushed). */
  before: Line[];
}

export interface Message {
  kind: "error" | "info";
  text: string;
}

/** Text of a line that was cut in two, offered back to the user (both halves start empty). */
export interface SplitText {
  text: string;
  leftId: number;
  rightId: number;
}

interface Options {
  api: ApiClient;
  pageId: number;
  state: EditorState;
  dispatch: (a: EditorAction) => void;
  size: { width: number; height: number };
  confirm?: (message: string) => boolean;
}

const meanY = (l: Pick<Line, "baseline">) => l.baseline.reduce((s, p) => s + p[1], 0) / Math.max(1, l.baseline.length);
const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** Runs repair commands against the API, keeps the undo stack, and owns the current mode. */
export function useRepair({ api, pageId, state, dispatch, size, confirm = (m) => window.confirm(m) }: Options) {
  const [mode, setModeState] = useState<Mode>(SELECT);
  const [undo, setUndo] = useState<UndoEntry[]>([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<Message | null>(null);
  const [splitText, setSplitText] = useState<SplitText | null>(null);

  const modeRef = useRef(mode);
  const stateRef = useRef(state);
  stateRef.current = state;
  const busyRef = useRef(false);
  const undoRef = useRef(undo);
  undoRef.current = undo;

  const setMode = useCallback((m: Mode) => {
    modeRef.current = m;
    setModeState(m);
  }, []);
  const info = (text: string) => setMessage({ kind: "info", text });
  const fail = (text: string) => setMessage({ kind: "error", text });

  const fetchLines = useCallback(async () => (await api.getPage(pageId)).lines, [api, pageId]);

  /** Common wrapper: one operation at a time, never while saves are in flight; pushes the undo snapshot. */
  const run = useCallback(
    async (label: string, op: () => Promise<{ lines: Line[]; selectId?: number | null; note?: string; noUndo?: boolean }>) => {
      if (busyRef.current) return;
      const s = stateRef.current;
      if (s.outbox.length > 0 || Object.keys(s.inflight).length > 0) {
        info("Still saving; try again in a moment.");
        return;
      }
      busyRef.current = true;
      setBusy(true);
      setMessage(null);
      const before = s.lines;
      try {
        const r = await op();
        dispatch({ type: "replaceLines", lines: r.lines, selectId: r.selectId });
        if (!r.noUndo) setUndo([...undoRef.current, { label, before }].slice(-MAX_UNDO));
        info(r.note ?? `${label}. Ctrl+Z undoes it.`);
      } catch (e) {
        fail(`${label} failed: ${errText(e)}`);
      } finally {
        busyRef.current = false;
        setBusy(false);
      }
    },
    [dispatch],
  );

  const draftOn = (ids: number[]): number | null => {
    const s = stateRef.current;
    const i = s.lines.findIndex((l) => ids.includes(l.id) && s.drafts[l.id] !== undefined);
    return i < 0 ? null : i + 1;
  };

  const merge = useCallback(
    async (ids: number[]) => {
      const s = stateRef.current;
      const d = draftOn(ids);
      if (d !== null) return fail(`Line ${d} has unsaved text. Press Enter to save it or Esc to revert it, then merge.`);
      const withText = s.lines.filter((l) => ids.includes(l.id) && (l.text || l.is_ground_truth));
      if (withText.length > 0 && !confirm(`Merging combines the text of ${withText.length} transcribed line(s) into one line and resets its ground-truth flag, so it must be checked again. Continue? (Ctrl+Z undoes it)`)) return;
      await run("Merged lines", async () => {
        const r = await api.mergeLines(ids);
        return { lines: r.page.lines, selectId: r.line.id };
      });
    },
    [api, confirm, run],
  );

  const split = useCallback(
    async (lineId: number, x: number, y: number) => {
      const line = stateRef.current.lines.find((l) => l.id === lineId);
      if (!line) return;
      const d = draftOn([lineId]);
      if (d !== null) return fail(`Line ${d} has unsaved text. Press Enter to save it or Esc to revert it, then split.`);
      if ((line.text || line.is_ground_truth) && !confirm("Splitting clears this line's text and ground-truth flag (the old text stays available to paste into either half). Continue? (Ctrl+Z undoes it)")) return;
      await run("Split line", async () => {
        const r = await api.splitLine(lineId, x, y);
        setSplitText(r.old_text ? { text: r.old_text, leftId: r.lines[0].id, rightId: r.lines[1].id } : null);
        return { lines: r.page.lines, selectId: r.lines[0].id };
      });
    },
    [api, confirm, run],
  );

  const addBaseline = useCallback(
    async (points: [number, number][]) => {
      await run("Added a line", async () => {
        const existing = stateRef.current.lines;
        const created = await api.addLine(pageId, points);
        // The server appends; put the new line where it belongs in top-to-bottom reading order.
        const at = existing.filter((l) => meanY(l) < meanY(created)).length;
        if (at < existing.length) await api.patchLine(created.id, { index: at });
        return { lines: await fetchLines(), selectId: created.id };
      });
    },
    [api, fetchLines, pageId, run],
  );

  const redrawBaseline = useCallback(
    async (lineId: number, points: [number, number][]) => {
      await run("Redrew baseline", async () => {
        await api.redrawBaseline(lineId, points);
        return { lines: await fetchLines(), selectId: lineId };
      });
    },
    [api, fetchLines, run],
  );

  const ignoreRegion = useCallback(
    async (rect: { x: number; y: number; w: number; h: number }) => {
      await run("Ignored region", async () => {
        const r = await api.ignoreRegion(pageId, rect);
        const sel = stateRef.current.selectedId;
        let selectId: number | null | undefined = sel;
        if (sel !== null && r.affected.includes(sel)) {
          const after = r.page.lines.filter((l) => !l.ignored && l.index > (r.page.lines.find((x) => x.id === sel)?.index ?? 0));
          selectId = after[0]?.id ?? r.page.lines.find((l) => !l.ignored)?.id ?? sel;
        }
        if (r.affected.length === 0) {
          return { lines: r.page.lines, selectId, noUndo: true, note: "No line had at least half of its area inside the region." };
        }
        return { lines: r.page.lines, selectId, note: `Ignored ${r.affected.length} line(s). Ctrl+Z undoes it.` };
      });
    },
    [api, pageId, run],
  );

  const deleteSelected = useCallback(async () => {
    const s = stateRef.current;
    const line = selectedLine(s);
    if (!line) return;
    if (s.drafts[line.id] !== undefined && !confirm("This line has unsaved typed text that Ctrl+Z cannot bring back. Delete it anyway?")) return;
    await run("Deleted line", async () => {
      await api.deleteLine(line.id);
      const lines = await fetchLines();
      const idx = s.lines.findIndex((l) => l.id === line.id);
      return { lines, selectId: (lines[Math.min(idx, lines.length - 1)] ?? null)?.id ?? null };
    });
  }, [api, confirm, fetchLines, run]);

  const move = useCallback(
    async (delta: 1 | -1) => {
      const s = stateRef.current;
      const ids = s.lines.map((l) => l.id);
      const i = ids.findIndex((id) => id === s.selectedId);
      const j = i + delta;
      if (i < 0 || j < 0 || j >= ids.length) return;
      [ids[i], ids[j]] = [ids[j]!, ids[i]!];
      await run(delta < 0 ? "Moved line up" : "Moved line down", async () => {
        const page = await api.reorder(pageId, ids);
        return { lines: page.lines, selectId: s.selectedId };
      });
    },
    [api, pageId, run],
  );

  const mergeNext = useCallback(async () => {
    const s = stateRef.current;
    const i = s.lines.findIndex((l) => l.id === s.selectedId);
    const next = s.lines[i + 1];
    if (i < 0 || !next) return info("There is no next line to merge with.");
    await merge([s.lines[i]!.id, next.id]);
  }, [merge]);

  const undoLast = useCallback(async () => {
    const entry = undoRef.current[undoRef.current.length - 1];
    if (!entry) return info("Nothing to undo.");
    await run("Undo", async () => {
      let page;
      try {
        page = await api.restore(pageId, entry.before);
      } catch (e) {
        if (e instanceof ApiError && e.status === 409) {
          if (!confirm(`${e.message}\n\nUndo anyway and discard that text?`)) throw new Error("undo cancelled; nothing changed", { cause: e });
          page = await api.restore(pageId, entry.before, true);
        } else {
          throw e;
        }
      }
      setUndo(undoRef.current.slice(0, -1));
      const sel = stateRef.current.selectedId;
      return {
        lines: page.lines,
        selectId: page.lines.some((l) => l.id === sel) ? sel : initialSelection(page.lines),
        noUndo: true,
        note: `Undid: ${entry.label.toLowerCase()}.`,
      };
    });
  }, [api, confirm, pageId, run]);

  const execute = useCallback(
    (c: Command) => {
      switch (c.type) {
        case "merge":
          return merge(c.ids);
        case "split":
          return split(c.lineId, c.x, c.y);
        case "addBaseline":
          return addBaseline(c.points);
        case "redrawBaseline":
          return redrawBaseline(c.lineId, c.points);
        case "ignoreRegion":
          return ignoreRegion(c.rect);
      }
    },
    [addBaseline, ignoreRegion, merge, redrawBaseline, split],
  );

  /** Feed a mode event through the state machine and run any command it produces. */
  const modeEvent = useCallback(
    (ev: ModeEvent) => {
      if (busyRef.current && ev.type !== "cancel") return;
      const res = modeReducer(modeRef.current, ev, { lines: stateRef.current.lines, size });
      setMode(res.mode);
      if (res.notice) info(res.notice);
      if (res.command) void execute(res.command);
    },
    [execute, setMode, size],
  );

  return {
    mode,
    busy,
    message,
    splitText,
    undoDepth: undo.length,
    undoLabel: undo[undo.length - 1]?.label ?? null,
    modeEvent,
    enter: (kind: "merge" | "split" | "baseline" | "redraw" | "region") =>
      modeEvent({ type: "enter", mode: kind, selectedId: stateRef.current.selectedId }),
    cancel: () => modeEvent({ type: "cancel" }),
    mergeNext,
    deleteSelected,
    move,
    undoLast,
    dismissMessage: () => setMessage(null),
    dismissSplitText: () => setSplitText(null),
  };
}
