import { useCallback, useEffect, useRef, useState } from "react";
import type { ApiClient } from "../api/client";
import type { ReviewItem } from "../api/types";
import { ConfidenceBadge, ConfidenceLegend } from "../editor/ConfidenceBadge";
import { pageHash } from "../router";
import { RetrainNotice } from "../training/RetrainNotice";

const errText = (e: unknown): string => (e instanceof Error ? e.message : String(e));

interface Session {
  accepted: number;
  corrected: number;
  ignored: number;
}

/**
 * Review queue: the least certain predictions first, one at a time, keyboard first.
 * Enter accepts (or saves your correction) and loads the next line, Tab skips (the line returns after "Reset
 * skipped"), Ctrl+I ignores, Esc restores the prediction. Saving marks the line ground truth, and the server records
 * whether it was accepted unchanged or corrected (that is the correction-effort measure on the Training page).
 */
export function Review({ api }: { api: ApiClient }) {
  const [item, setItem] = useState<ReviewItem | null>(null);
  const [remaining, setRemaining] = useState<number | null>(null);
  const [text, setText] = useState("");
  const [skipped, setSkipped] = useState<number[]>([]);
  const [session, setSession] = useState<Session>({ accepted: 0, corrected: 0, ignored: 0 });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const load = useCallback(
    async (exclude: number[]) => {
      try {
        const res = await api.reviewNext(exclude, 1);
        const next = res.items[0] ?? null;
        setItem(next);
        setRemaining(res.remaining);
        setText(next?.predicted_text ?? "");
        setError(null);
      } catch (e) {
        setError(errText(e));
      }
    },
    [api],
  );

  useEffect(() => {
    void load([]);
  }, [load]);

  useEffect(() => {
    inputRef.current?.focus();
  }, [item?.line.id]);

  const run = async (f: () => Promise<void>) => {
    if (busy) return;
    setBusy(true);
    try {
      await f();
    } catch (e) {
      setError(errText(e));
    } finally {
      setBusy(false);
    }
  };

  const accept = () =>
    item &&
    run(async () => {
      if (text.trim() === "") {
        setError("Cannot save an empty line as ground truth. Type text, or press Ctrl+I to ignore the line.");
        return;
      }
      await api.patchLine(item.line.id, { text, is_ground_truth: true });
      const same = text === item.predicted_text;
      setSession((s) => ({ ...s, accepted: s.accepted + (same ? 1 : 0), corrected: s.corrected + (same ? 0 : 1) }));
      await load(skipped);
    });

  const skip = () =>
    item &&
    run(async () => {
      const next = [...skipped, item.line.id];
      setSkipped(next);
      await load(next);
    });

  const ignore = () =>
    item &&
    run(async () => {
      await api.patchLine(item.line.id, { ignored: true });
      setSession((s) => ({ ...s, ignored: s.ignored + 1 }));
      await load(skipped);
    });

  const resetSkipped = () => {
    setSkipped([]);
    void load([]);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.nativeEvent.isComposing) return;
    const mod = e.ctrlKey || e.metaKey;
    if (e.key === "Enter" && !mod && !e.altKey && !e.shiftKey) {
      e.preventDefault();
      void accept();
    } else if (e.key === "Tab" && !mod && !e.altKey && !e.shiftKey) {
      e.preventDefault();
      void skip();
    } else if (mod && !e.altKey && !e.shiftKey && e.key.toLowerCase() === "i") {
      e.preventDefault();
      void ignore();
    } else if (e.key === "Escape" && item) {
      e.preventDefault();
      setText(item.predicted_text);
    }
  };

  const done = session.accepted + session.corrected + session.ignored;
  const edited = item !== null && text !== item.predicted_text;

  return (
    <main className="review" onKeyDown={onKeyDown}>
      <header className="training-head">
        <h1>Review</h1>
        <span className="actions">
          <a className="button" href="#/">
            Library
          </a>
          <a className="button" href="#/training">
            Training
          </a>
        </span>
      </header>
      <RetrainNotice api={api} />
      {error && (
        <p role="alert" className="error banner">
          <span>{error}</span>
          <button onClick={() => void load(skipped)}>Retry</button>
        </p>
      )}
      <p className="muted" data-testid="review-progress" role="status">
        {remaining === null ? "Loading..." : `${remaining} ${remaining === 1 ? "line" : "lines"} left, least certain first.`} This session: {done} reviewed
        ({session.accepted} accepted, {session.corrected} corrected, {session.ignored} ignored).
      </p>

      {item === null && remaining === 0 && (
        <section className="panel" data-testid="review-empty">
          <p>
            {skipped.length > 0
              ? `Nothing left except ${skipped.length} skipped ${skipped.length === 1 ? "line" : "lines"}.`
              : "No predictions are waiting for review."}
          </p>
          {skipped.length > 0 ? (
            <button onClick={resetSkipped}>Reset skipped</button>
          ) : (
            <p className="muted">Run a round on the Training page to predict the lines you have not transcribed.</p>
          )}
        </section>
      )}

      {item && (
        <section className="panel review-card" aria-label="Line to review">
          <div className="dock-head">
            <strong>Predicted by {item.model_id === null ? "a model" : `model ${item.model_id}`}</strong>
            <ConfidenceBadge confidence={item.confidence} />
            <span className="spacer" />
            <a href={pageHash(item.page_id)} data-testid="open-in-editor">
              Open page in editor
            </a>
          </div>
          <div className="crop-box review-crop">
            <img src={api.url(item.crop_url)} alt="Crop of the line to review" data-testid="review-crop" key={item.line.id} />
          </div>
          <label className="pred-label" htmlFor="review-input">
            {edited ? "Your correction" : "Predicted text (check it against the crop)"}
          </label>
          <input
            id="review-input"
            ref={inputRef}
            className="line-input"
            type="text"
            value={text}
            disabled={busy}
            onChange={(e) => setText(e.target.value)}
            aria-label="Text of the line (prefilled with the prediction)"
            autoComplete="off"
            spellCheck={false}
          />
          <p className="hint" data-testid="review-hint">
            Enter {edited ? "save correction" : "accept"} + next &middot; Tab skip &middot; Ctrl+I ignore &middot; Esc restore the prediction
          </p>
          {skipped.length > 0 && (
            <p className="muted small">
              {skipped.length} skipped this session. <button onClick={resetSkipped}>Reset skipped</button>
            </p>
          )}
        </section>
      )}
      <ConfidenceLegend />
    </main>
  );
}
