import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ApiClient } from "../api/client";
import type { Page, PageDetail } from "../api/types";
import { navigate, pageHash, setNavigationGuard } from "../router";
import { HelpOverlay } from "./HelpOverlay";
import { PageCanvas, type PageCanvasHandle } from "./PageCanvas";
import { currentText, hasUnsaved, keyToAction, selectedLine } from "./transcription";
import { useTranscription } from "./useTranscription";

const UNSAVED_MSG = "You have unsaved or unsent line edits on this page. Leave anyway?";

export function PageEditor({ api, pageId }: { api: ApiClient; pageId: number }) {
  const [page, setPage] = useState<PageDetail | null>(null);
  const [siblings, setSiblings] = useState<Page[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const [state, dispatch] = useTranscription(api);
  const canvasRef = useRef<PageCanvasHandle>(null);
  const inputRef = useRef<HTMLInputElement>(null);

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

  // Keep keyboard focus in the text box whenever the selected line changes.
  useEffect(() => {
    if (!helpOpen) inputRef.current?.focus();
  }, [line?.id, helpOpen, page?.id]);

  const idx = siblings.findIndex((p) => p.id === pageId);
  const prevPage = idx > 0 ? siblings[idx - 1] : undefined;
  const nextPage = idx >= 0 ? siblings[idx + 1] : undefined;

  const onKeyDown = (e: React.KeyboardEvent) => {
    const target = e.target as HTMLElement;
    const inTextField = target.tagName === "INPUT" || target.tagName === "TEXTAREA";
    // Buttons keep their native Enter/Tab/Space behaviour so the toolbar stays keyboard-accessible.
    if (target.tagName === "BUTTON" && (e.key === "Enter" || e.key === "Tab" || e.key === " ")) return;
    const res = keyToAction(
      { key: e.key, ctrlKey: e.ctrlKey, metaKey: e.metaKey, shiftKey: e.shiftKey, altKey: e.altKey, isComposing: e.nativeEvent.isComposing },
      { inTextField, overlayOpen: helpOpen },
    );
    if (!res) return;
    if (helpOpen && res.kind === "editor") return; // overlay swallows editing keys
    e.preventDefault();
    if (res.kind === "editor") return dispatch(res.action);
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

  return (
    <div className="editor" onKeyDown={onKeyDown}>
      <header className="editor-bar">
        <button onClick={() => navigate("#/")}>&larr; Library</button>
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

      <div className="editor-main" onClick={() => inputRef.current?.focus()}>
        <PageCanvas
          ref={canvasRef}
          imageUrl={api.url(page.image_url)}
          imageSize={{ width: page.width, height: page.height }}
          lines={state.lines}
          selectedId={state.selectedId}
          onSelect={select}
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
                  onClick={() => select(l.id)}
                  aria-current={l.id === state.selectedId ? "true" : undefined}
                  tabIndex={-1}
                >
                  <span className="n">{i + 1}</span>
                  <span className="t">{state.drafts[l.id] ?? l.text ?? ""}</span>
                  <span className="flag">{l.ignored ? "ignored" : l.is_ground_truth ? "GT" : ""}</span>
                </button>
              </li>
            ))}
          </ol>
        </nav>
      </div>

      <section className="dock" aria-label="Current line">
        {state.lines.length === 0 ? (
          <p className="muted">
            This page has no lines yet. {page.status === "unsegmented" ? "Run Segment from the library first." : ""}
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
                <span className="spacer" />
                <button onClick={() => dispatch({ type: "toggleIgnore" })} title="Ctrl+I">
                  {line.ignored ? "Un-ignore" : "Ignore"} (Ctrl+I)
                </button>
              </div>
              <div className="crop-box">
                <img src={api.url(line.crop_url)} alt={`Crop of line ${lineNo}`} data-testid="line-crop" />
              </div>
              <input
                ref={inputRef}
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
                Enter save + next &middot; Tab/Shift+Tab move &middot; Esc revert &middot; Ctrl+I ignore &middot; Ctrl+/ help
              </p>
            </>
          )
        )}
        {state.notice && (
          <p role="status" className="notice">
            {state.notice}
          </p>
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
