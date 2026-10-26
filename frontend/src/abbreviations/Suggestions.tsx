import { useCallback, useEffect, useRef, useState } from "react";
import type { ApiClient } from "../api/client";
import type { Job, Suggestion, SuggestionList } from "../api/types";
import { jobIsActive, jobProblem } from "../library/format";

const errText = (e: unknown): string => (e instanceof Error ? e.message : String(e));

interface Draft {
  id: number;
  written: string;
  expansion: string;
}

/**
 * Optional local-LLM abbreviation suggestions. Off unless the server was started with HTR_LLM_ENABLED=1. A suggestion
 * is only a proposal: it never changes a transcription, and it becomes a dictionary entry only when the person
 * accepts it (as proposed, or after editing). Rejected proposals are remembered and not suggested again.
 */
export function Suggestions({ api, onAccepted, pollMs = 1000 }: { api: ApiClient; onAccepted?: () => void; pollMs?: number }) {
  const [list, setList] = useState<SuggestionList | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [startError, setStartError] = useState<string | null>(null);
  const [rowError, setRowError] = useState<string | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const jobRef = useRef<Job | null>(null);
  jobRef.current = job;

  const refresh = useCallback(async () => {
    try {
      setList(await api.listSuggestions("pending"));
      setLoadError(null);
    } catch (e) {
      setLoadError(errText(e));
    }
  }, [api]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Follow a running suggestion job; show the new suggestions as they are stored.
  const active = job !== null && jobIsActive(job);
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(async () => {
      const current = jobRef.current;
      if (!current) return;
      try {
        const fresh = await api.getJob(current.id);
        setJob(fresh);
        void refresh();
      } catch (e) {
        setStartError(errText(e));
      }
    }, pollMs);
    return () => clearInterval(timer);
  }, [active, api, refresh, pollMs]);

  const start = async () => {
    setStartError(null);
    try {
      setJob(await api.generateSuggestions());
    } catch (e) {
      setStartError(errText(e));
    }
  };

  const cancel = async () => {
    if (!job) return;
    try {
      setJob(await api.cancelJob(job.id));
    } catch (e) {
      setStartError(errText(e));
    }
  };

  const decide = async (action: () => Promise<unknown>, accepted: boolean) => {
    setRowError(null);
    try {
      await action();
      setDraft(null);
      await refresh();
      if (accepted) onAccepted?.();
    } catch (e) {
      setRowError(errText(e));
    }
  };

  const result = (job?.result ?? {}) as { tokens?: number; asked?: number; suggestions_added?: number };
  const items: Suggestion[] = list?.items ?? [];

  return (
    <section className="panel" aria-labelledby="sugg-heading">
      <h2 id="sugg-heading">
        Suggestions from a local language model{items.length > 0 ? ` (${items.length})` : ""}
      </h2>
      <p className="muted small">
        Optional and off by default. A model running on this computer (for example Ollama) can propose what an
        unfamiliar abbreviation stands for, using only the abbreviation and the line it appears in. Suggestions never
        change your transcriptions; one becomes a dictionary entry only when you accept it.
      </p>
      {loadError && (
        <p role="alert" className="error banner">
          <span>{loadError}</span>
          <button onClick={() => void refresh()}>Retry</button>
        </p>
      )}
      {list && !list.enabled && (
        <p className="muted" data-testid="llm-off">
          Local-LLM suggestions are switched off. To use them, install a local model server and start the app with{" "}
          <code>HTR_LLM_ENABLED=1</code>; the README explains how.
        </p>
      )}
      {list?.enabled && (
        <p>
          <button onClick={() => void start()} disabled={active}>
            Suggest abbreviations
          </button>{" "}
          <span className="muted small">
            Model: {list.backend?.model} at {list.backend?.url}. Only text you wrote or confirmed is used.
          </span>
        </p>
      )}
      {startError && (
        <p role="alert" className="error" data-testid="suggest-error">
          {startError}
        </p>
      )}
      {job && (
        <div className={`job ${job.status}`} role="status" data-testid="suggest-status">
          {job.status === "queued" && "Waiting for its turn..."}
          {job.status === "running" && `Asking the model... ${result.asked ?? 0} of ${result.tokens ?? "?"} abbreviations`}
          {job.status === "succeeded" &&
            `Done: asked about ${result.asked ?? 0} abbreviation(s), ${result.suggestions_added ?? 0} new suggestion(s).`}
          {job.status === "cancelled" && `Stopped after ${result.asked ?? 0} abbreviation(s); what was answered is kept below.`}
          {job.status === "failed" && <span className="error">Suggestions failed: {jobProblem(job)}</span>}
          {active && <progress value={job.progress} max={1} aria-label="Suggestion progress" />}{" "}
          {active && (
            <button onClick={() => void cancel()} disabled={job.cancel_requested}>
              {job.cancel_requested ? "Stopping..." : "Cancel"}
            </button>
          )}
        </div>
      )}
      {rowError && (
        <p role="alert" className="error banner">
          <span>{rowError}</span>
          <button onClick={() => setRowError(null)}>Dismiss</button>
        </p>
      )}
      {list && items.length === 0 && list.enabled && !active && <p className="muted">No suggestions waiting for a decision.</p>}
      {items.length > 0 && (
        <table className="abbr-table">
          <caption className="sr-only">Suggested abbreviations</caption>
          <thead>
            <tr>
              <th scope="col">Written</th>
              <th scope="col">Suggested expansion</th>
              <th scope="col">Actions</th>
            </tr>
          </thead>
          <tbody>
            {items.map((s) =>
              draft?.id === s.id ? (
                <tr key={s.id} className="editing">
                  <td colSpan={3}>
                    <form
                      className="abbr-form"
                      aria-label={`Edit suggestion ${s.written}`}
                      onSubmit={(e) => {
                        e.preventDefault();
                        void decide(() => api.acceptSuggestion(s.id, { written: draft.written, expansion: draft.expansion }), true);
                      }}
                    >
                      <label>
                        Written form
                        <input value={draft.written} onChange={(e) => setDraft({ ...draft, written: e.target.value })} />
                      </label>
                      <label>
                        Expansion
                        <input value={draft.expansion} onChange={(e) => setDraft({ ...draft, expansion: e.target.value })} autoFocus />
                      </label>
                      <button type="submit" disabled={!draft.written.trim() || !draft.expansion.trim()}>
                        Accept as edited
                      </button>
                      <button type="button" onClick={() => setDraft(null)}>
                        Cancel
                      </button>
                    </form>
                  </td>
                </tr>
              ) : (
                <tr key={s.id} className="suggestion-row" data-testid={`suggestion-${s.id}`}>
                  <th scope="row">
                    <code>{s.written}</code>
                  </th>
                  <td>
                    {s.expansion}
                    {s.rationale && <span className="why">{s.rationale}</span>}
                    {s.context && <span className="why ctx">From: &ldquo;{s.context}&rdquo;</span>}
                  </td>
                  <td className="row-actions">
                    <button onClick={() => void decide(() => api.acceptSuggestion(s.id), true)} aria-label={`Accept ${s.written} = ${s.expansion}`}>
                      Accept
                    </button>
                    <button onClick={() => setDraft({ id: s.id, written: s.written, expansion: s.expansion })} aria-label={`Edit suggestion ${s.written} = ${s.expansion}`}>
                      Edit
                    </button>
                    <button onClick={() => void decide(() => api.rejectSuggestion(s.id), false)} aria-label={`Reject ${s.written} = ${s.expansion}`}>
                      Reject
                    </button>
                  </td>
                </tr>
              ),
            )}
          </tbody>
        </table>
      )}
    </section>
  );
}
