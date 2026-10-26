import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ApiClient } from "../api/client";
import type { Page, PageDetail } from "../api/types";
import { navigate, pageHash, REVIEW_HASH, setNavigationGuard } from "../router";
import { HelpOverlay } from "./HelpOverlay";
import { PageCanvas, type CanvasEvent, type PageCanvasHandle } from "./PageCanvas";
import { type ModeKind, modeHelp } from "./modes";
import { currentText, hasUnsaved, keyToAction, selectedLine } from "./transcription";
import { useRepair } from "./useRepair";
import { ConfidenceBadge, ConfidenceLegend } from "./ConfidenceBadge";
import { isPrediction } from "./confidence";
import { useTranscription } from "./useTranscription";

const UNSAVED_MSG = "You have unsaved or unsent line edits on this page. Leave anyway?";

const MODE_LABEL: Record<ModeKind, string> = {
  select: "Select",
  merge: "Merge",
  split: "Split",
  baseline: "Draw baseline",
  region: "Ignore region",
};

export function PageEditor({ api, pageId }: { api: ApiClient; pageId: number }) {
  const [page, setPage] = useState<PageDetail | null>(null);
  const [siblings, setSiblings] = useState<Page[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const [state, dispatch] = useTranscription(api);
  const size = useMemo(() => ({ width: page?.width ?? 0, height: page?.height ?? 0 }), [page?.width, page?.height]);
  const repair = useRepair({ api, pageId, state, dispatch, size });
  const { mode } = repair;
  const canvasRef = useRef<PageCanvasHandle>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let live = true;
    setPage(null);
    setLoadError(null);
    api
      .getPage(pageId)
      .then((p) => {
        if (!live) return;
        setPage(p);
        dispatch({ type: "loaded", lines: p.lines });
        api
          .listPages({ document_id: p.document_id })
          .then((ps) => live && setSiblings(ps))
          .catch(() => undefined); // page navigation is optional
      })
      .catch((e: unknown) => live && setLoadError(e instanceof Error ? e.message : String(e)));
    return () => {
      live = false;
    };
  }, [api, pageId, dispatch]);

  const unsaved = hasUnsaved(state);
  const unsavedRef = useRef(unsaved);
  unsavedRef.current = unsaved;

  // Warn before leaving: in-app navigation (guard) and closing/reloading the tab.
  useEffect(() => {
    setNavigationGuard(() => !unsavedRef.current || window.confirm(UNSAVED_MSG));
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      if (unsavedRef.current) {
        e.preventDefault();
        e.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => {
      setNavigationGuard(null);
      window.removeEventListener("beforeunload", onBeforeUnload);
    };
  }, []);

  const line = selectedLine(state);
  const text = currentText(state);

  // Keyboard focus: the text box in Select mode; the editor root while a repair mode is active (so Enter / Esc /
  // Backspace belong to the mode and typing cannot leak into a line).
  const modeKindRef = useRef(mode.kind);
  modeKindRef.current = mode.kind;
  const focusTarget = useCallback(() => {
    (modeKindRef.current === "select" ? inputRef.current : rootRef.current)?.focus();
  }, []);
  useEffect(() => {
    if (!helpOpen) focusTarget();
  }, [line?.id, helpOpen, page?.id, mode.kind, focusTarget]);

  const idx = siblings.findIndex((p) => p.id === pageId);
  const prevPage = idx > 0 ? siblings[idx - 1] : undefined;
  const nextPage = idx >= 0 ? siblings[idx + 1] : undefined;

  const onCanvasEvent = (e: CanvasEvent) => {
    if (mode.kind === "select") {
      if (e.type === "click" && e.hitId !== null) dispatch({ type: "select", id: e.hitId });
      return;
    }
    repair.modeEvent(e);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    const target = e.target as HTMLElement;
    const inTextField = target.tagName === "INPUT" || target.tagName === "TEXTAREA";
    // Buttons keep their native Enter/Tab/Space behaviour so the toolbar stays keyboard-accessible.
    if (target.tagName === "BUTTON" && (e.key === "Enter" || e.key === "Tab" || e.key === " ")) return;
    const noMods = !e.ctrlKey && !e.metaKey && !e.altKey;

    if (!helpOpen && mode.kind !== "select") {
      if (e.key === "Escape") {
        e.preventDefault();
        return repair.cancel();
      }
      if (noMods && e.key === "Enter") {
        e.preventDefault();
        return repair.modeEvent({ type: "confirm" });
      }
      if (noMods && e.key === "Backspace" && mode.kind === "baseline") {
        e.preventDefault();
        return repair.modeEvent({ type: "back" });
      }
    }

    const res = keyToAction(
      {
        key: e.key,
        code: e.code,
        ctrlKey: e.ctrlKey,
        metaKey: e.metaKey,
        shiftKey: e.shiftKey,
        altKey: e.altKey,
        isComposing: e.nativeEvent.isComposing,
      },
      { inTextField, overlayOpen: helpOpen, hasDraft: line ? state.drafts[line.id] !== undefined : false },
    );
    if (!res) return;
    if (helpOpen && res.kind !== "ui") return; // overlay swallows editing keys
    e.preventDefault();
    if (res.kind === "editor") {
      if (mode.kind === "select") dispatch(res.action); // in a mode these keys do nothing
      return;
    }
    if (res.kind === "repair") {
      switch (res.command) {
        case "mode:merge":
          return repair.enter("merge");
        case "mode:split":
          return repair.enter("split");
        case "mode:baseline":
          return repair.enter("baseline");
        case "mode:redraw":
          return repair.enter("redraw");
        case "mode:region":
          return repair.enter("region");
        case "mergeNext":
          return void repair.mergeNext();
        case "deleteLine":
          return void repair.deleteSelected();
        case "moveUp":
          return void repair.move(-1);
        case "moveDown":
          return void repair.move(1);
        case "undo":
          return void repair.undoLast();
      }
      return;
    }
    switch (res.command) {
      case "help":
        return setHelpOpen((v) => !v);
      case "closeOverlay":
        return setHelpOpen(false);
      case "fitWidth":
        return canvasRef.current?.fitWidth();
      case "nextPage":
        return nextPage && navigate(pageHash(nextPage.id));
      case "prevPage":
        return prevPage && navigate(pageHash(prevPage.id));
    }
  };

  const failedIds = useMemo(() => Object.keys(state.errors).map(Number), [state.errors]);
  const select = useCallback((id: number) => dispatch({ type: "select", id }), [dispatch]);

  if (loadError) {
    return (
      <main className="editor-state">
        <p role="alert" className="error">
          Could not load page {pageId}: {loadError}
        </p>
        <button onClick={() => navigate("#/")}>Back to library</button>
      </main>
    );
  }
  if (!page) return <main className="editor-state">Loading page...</main>;

  const lineNo = line ? state.lines.indexOf(line) + 1 : 0;
  const active = state.lines.filter((l) => !l.ignored);
  const gt = active.filter((l) => l.is_ground_truth).length;
  const lineIdx = line ? state.lines.indexOf(line) : -1;
  const refocus = () => setTimeout(focusTarget, 0);
  const predicted = line ? isPrediction(line) : false;

  return (
    <div className="editor" onKeyDown={onKeyDown} ref={rootRef} tabIndex={-1}>
      <header className="editor-bar">
        <button onClick={() => navigate("#/")}>&larr; Library</button>
        <button onClick={() => navigate(REVIEW_HASH)} title="Review the least certain predictions first">
          Review
        </button>
        <h1 title={page.original_filename}>{page.original_filename}</h1>
        <span className="muted" data-testid="progress">
          {gt}/{active.length} lines transcribed
          {state.lines.length - active.length > 0 ? `, ${state.lines.length - active.length} ignored` : ""}
        </span>
        <span className={`save-state ${unsaved ? "pending" : ""}`} role="status" data-testid="save-state">
          {failedIds.length ? "Save failed" : unsaved ? "Unsaved changes" : "All changes saved"}
        </span>
        <span className="spacer" />
        <button disabled={!prevPage} onClick={() => prevPage && navigate(pageHash(prevPage.id))} title="Alt+PageUp">
          &lsaquo; Prev page
        </button>
        <button disabled={!nextPage} onClick={() => nextPage && navigate(pageHash(nextPage.id))} title="Alt+PageDown">
          Next page &rsaquo;
        </button>
        <button onClick={() => canvasRef.current?.fitWidth()} title="Alt+0">
          Fit width
        </button>
        <button onClick={() => setHelpOpen(true)} aria-label="Keyboard shortcuts" title="Ctrl+/ or F1">
          ?
        </button>
      </header>

      <div className="toolbar" role="toolbar" aria-label="Segmentation repair tools" onClickCapture={refocus}>
        <button aria-pressed={mode.kind === "select"} onClick={repair.cancel} title="Esc">
          Select
        </button>
        <button aria-pressed={mode.kind === "merge"} onClick={() => repair.enter("merge")} title="Alt+M">
          Merge
        </button>
        <button onClick={() => void repair.mergeNext()} disabled={lineIdx < 0 || lineIdx >= state.lines.length - 1} title="Alt+J">
          Merge with next
        </button>
        <button aria-pressed={mode.kind === "split"} onClick={() => repair.enter("split")} title="Alt+S">
          Split
        </button>
        <button aria-pressed={mode.kind === "baseline" && !hasRedraw(mode)} onClick={() => repair.enter("baseline")} title="Alt+B">
          Draw baseline
        </button>
        <button aria-pressed={mode.kind === "baseline" && hasRedraw(mode)} onClick={() => repair.enter("redraw")} disabled={!line} title="Alt+Shift+B">
          Redraw baseline
        </button>
        <button aria-pressed={mode.kind === "region"} onClick={() => repair.enter("region")} title="Alt+R">
          Ignore region
        </button>
        <span className="sep" />
        <button onClick={() => void repair.deleteSelected()} disabled={!line} title="Alt+Delete">
          Delete line
        </button>
        <button onClick={() => void repair.move(-1)} disabled={lineIdx <= 0} title="Alt+Up" aria-label="Move line up">
          Move up
        </button>
        <button onClick={() => void repair.move(1)} disabled={lineIdx < 0 || lineIdx >= state.lines.length - 1} title="Alt+Down" aria-label="Move line down">
          Move down
        </button>
        <button onClick={() => void repair.undoLast()} disabled={repair.undoDepth === 0} title="Ctrl+Z">
          Undo{repair.undoLabel ? `: ${repair.undoLabel}` : ""}
        </button>
      </div>
      <div className={`mode-banner mode-${mode.kind}`} role="status" data-testid="mode-banner" data-mode={mode.kind}>
        <strong>{MODE_LABEL[mode.kind]}{mode.kind === "baseline" && hasRedraw(mode) ? " (redraw)" : ""}</strong>
        <span>{modeHelp(mode)}</span>
        {mode.kind !== "select" && (
          <button onClick={repair.cancel} aria-label="Cancel mode">
            Cancel (Esc)
          </button>
        )}
      </div>

      <div className="editor-main" onClick={() => mode.kind === "select" && inputRef.current?.focus()}>
        <PageCanvas
          ref={canvasRef}
          imageUrl={api.url(page.image_url)}
          imageSize={{ width: page.width, height: page.height }}
          lines={state.lines}
          selectedId={state.selectedId}
          mode={mode}
          onEvent={onCanvasEvent}
        />
        <nav className="line-list" aria-label="Lines">
          <ol>
            {state.lines.map((l, i) => (
              <li key={l.id}>
                <button
                  className={[
                    l.id === state.selectedId ? "sel" : "",
                    l.ignored ? "ign" : "",
                    l.is_ground_truth ? "gt" : "",
                    state.errors[l.id] ? "err" : "",
                  ].join(" ")}
                  onClick={() => (mode.kind === "merge" ? repair.modeEvent({ type: "click", world: [0, 0], hitId: l.id, tol: 0 }) : select(l.id))}
                  aria-current={l.id === state.selectedId ? "true" : undefined}
                  tabIndex={-1}
                  onMouseDown={(e) => e.preventDefault()} // keep keyboard focus where it is (Enter must not re-click)
                >
                  <span className="n">{i + 1}</span>
                  <span className="t">{state.drafts[l.id] ?? l.text ?? ""}</span>
                  <span className="flag">
                    {l.ignored ? (
                      "ignored"
                    ) : l.is_ground_truth ? (
                      "GT"
                    ) : isPrediction(l) ? (
                      <ConfidenceBadge confidence={l.confidence!} />
                    ) : (
                      ""
                    )}
                  </span>
                </button>
              </li>
            ))}
          </ol>
          <ConfidenceLegend />
        </nav>
      </div>

      <section className="dock" aria-label="Current line">
        {state.lines.length === 0 ? (
          <p className="muted">
            This page has no lines yet. {page.status === "unsegmented" ? "Run Segment from the library first, or draw baselines (Alt+B)." : "Draw a baseline (Alt+B) to add one."}
          </p>
        ) : (
          line && (
            <>
              <div className="dock-head">
                <strong>
                  Line {lineNo} of {state.lines.length}
                </strong>
                {line.ignored && <span className="badge ign">ignored</span>}
                {line.is_ground_truth && !line.ignored && <span className="badge gt">ground truth</span>}
                {predicted && (
                  <span className="badge pred" data-testid="predicted-badge">
                    predicted <ConfidenceBadge confidence={line.confidence!} />
                  </span>
                )}
                <span className="spacer" />
                <button onClick={() => dispatch({ type: "toggleIgnore" })} title="Ctrl+I">
                  {line.ignored ? "Un-ignore" : "Ignore"} (Ctrl+I)
                </button>
              </div>
              <div className="crop-box">
                <img src={api.url(line.crop_url)} alt={`Crop of line ${lineNo}`} data-testid="line-crop" key={`${line.id}-${line.polygon.length}-${line.polygon[0]?.join(",")}-${line.baseline.join(",")}`} />
              </div>
              {predicted && (
                <p className="pred-label" id="pred-label">
                  Predicted text, not yet checked: Enter accepts it as ground truth, edit then Enter to correct it, Esc restores the prediction.
                </p>
              )}
              <input
                ref={inputRef}
                aria-describedby={predicted ? "pred-label" : undefined}
                className="line-input"
                type="text"
                value={text}
                readOnly={line.ignored}
                aria-label={`Transcription of line ${lineNo}`}
                placeholder={line.ignored ? "Line is ignored (Ctrl+I to restore)" : "Type what the crop says, Enter to save"}
                autoComplete="off"
                spellCheck={false}
                onChange={(e) => dispatch({ type: "edit", text: e.target.value })}
              />
              <p className="hint">
                Enter save/accept + next &middot; Tab/Shift+Tab move &middot; Esc revert &middot; Ctrl+I ignore &middot; Alt+M/S/B/R repair &middot; Ctrl+Z undo &middot; Ctrl+/ help
              </p>
            </>
          )
        )}
        {state.notice && (
          <p role="status" className="notice">
            {state.notice}
          </p>
        )}
        {repair.message && (
          <div role={repair.message.kind === "error" ? "alert" : "status"} className={`banner ${repair.message.kind === "error" ? "error" : "info"}`} data-testid="repair-message">
            <span>{repair.message.text}</span>
            <button onClick={repair.dismissMessage}>Dismiss</button>
          </div>
        )}
        {repair.splitText && (
          <div className="banner info" data-testid="split-text">
            <span>
              The cut line had the text &ldquo;{repair.splitText.text}&rdquo;. Both halves are empty now.
            </span>
            <button
              onClick={() => {
                dispatch({ type: "select", id: repair.splitText!.leftId });
                dispatch({ type: "edit", text: repair.splitText!.text });
                repair.dismissSplitText();
              }}
            >
              Put in left half
            </button>
            <button
              onClick={() => {
                dispatch({ type: "select", id: repair.splitText!.rightId });
                dispatch({ type: "edit", text: repair.splitText!.text });
                repair.dismissSplitText();
              }}
            >
              Put in right half
            </button>
            <button onClick={repair.dismissSplitText}>Dismiss</button>
          </div>
        )}
        {failedIds.map((id) => {
          const i = state.lines.findIndex((l) => l.id === id);
          return (
            <div key={id} role="alert" className="error banner" data-testid="save-error">
              <span>
                Line {i + 1}: {state.errors[id]} {state.drafts[id] !== undefined ? "Your text is kept." : ""}
              </span>
              <button onClick={() => dispatch({ type: "select", id })}>Go to line</button>
              {state.drafts[id] !== undefined && state.drafts[id]!.trim() !== "" && (
                <button onClick={() => dispatch({ type: "commit", advance: false, lineId: id })}>Retry save</button>
              )}
              <button onClick={() => dispatch({ type: "dismissError", lineId: id })}>Dismiss</button>
            </div>
          );
        })}
      </section>
      {helpOpen && <HelpOverlay onClose={() => setHelpOpen(false)} />}
    </div>
  );
}

function hasRedraw(m: { kind: string; redrawId?: number | null }): boolean {
  return m.kind === "baseline" && m.redrawId !== null && m.redrawId !== undefined;
}
