import { Fragment, useEffect, useState } from "react";
import type { ApiClient } from "../api/client";
import type { ExpandSpan, Page, PageDetail, PageText } from "../api/types";
import { pageHash, textHash } from "../router";

const errText = (e: unknown): string => (e instanceof Error ? e.message : String(e));

/** The raw text with the matched abbreviations highlighted (`expanded` and `ambiguous` spans refer to raw offsets). */
export function HighlightedRaw({ raw, spans }: { raw: string; spans: ExpandSpan[] }) {
  const parts: React.ReactNode[] = [];
  let pos = 0;
  for (const s of [...spans].sort((a, b) => a.start - b.start)) {
    if (s.status === "unmatched" || s.start < pos) continue;
    if (s.start > pos) parts.push(raw.slice(pos, s.start));
    parts.push(
      <mark key={s.start} className={s.status} title={s.status === "ambiguous" ? `Ambiguous: ${s.candidates.join(" / ")}` : `= ${s.expansion}`}>
        {raw.slice(s.start, s.end)}
      </mark>,
    );
    pos = s.end;
  }
  if (pos < raw.length) parts.push(raw.slice(pos));
  return (
    <>
      {parts.map((p, i) => (
        <Fragment key={i}>{p}</Fragment>
      ))}
    </>
  );
}

/**
 * Read-only text of one page, raw and expanded side by side. The raw column is what was written (and what is stored);
 * the expanded column is computed from the abbreviation dictionary on demand. Ambiguous abbreviations are not expanded.
 */
export function TextView({ api, pageId }: { api: ApiClient; pageId: number }) {
  const [page, setPage] = useState<PageDetail | null>(null);
  const [siblings, setSiblings] = useState<Page[]>([]);
  const [text, setText] = useState<PageText | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mark, setMark] = useState(true);
  const [predicted, setPredicted] = useState(true);

  useEffect(() => {
    let live = true;
    api
      .getPage(pageId)
      .then((p) => {
        if (!live) return;
        setPage(p);
        api
          .listPages({ document_id: p.document_id })
          .then((ps) => live && setSiblings(ps))
          .catch(() => undefined);
      })
      .catch((e: unknown) => live && setError(errText(e)));
    return () => {
      live = false;
    };
  }, [api, pageId]);

  useEffect(() => {
    let live = true;
    setError(null);
    api
      .pageText(pageId, { ambiguous: mark ? "mark" : "raw", include_predicted: predicted })
      .then((t) => live && setText(t))
      .catch((e: unknown) => live && setError(errText(e)));
    return () => {
      live = false;
    };
  }, [api, pageId, mark, predicted]);

  const idx = siblings.findIndex((p) => p.id === pageId);
  const prev = idx > 0 ? siblings[idx - 1] : undefined;
  const next = idx >= 0 ? siblings[idx + 1] : undefined;
  const ambiguousLines = text?.lines.filter((l) => l.spans.some((s) => s.status === "ambiguous")).length ?? 0;

  return (
    <main className="training text-view">
      <header className="training-head">
        <h1>Text{page ? `: ${page.original_filename}` : ""}</h1>
        <span className="actions">
          <a className="button" href={pageHash(pageId)}>
            Back to editor
          </a>
          {prev ? (
            <a className="button" href={textHash(prev.id)}>
              &lsaquo; Prev page
            </a>
          ) : (
            <span className="button disabled" aria-disabled="true">
              &lsaquo; Prev page
            </span>
          )}
          {next ? (
            <a className="button" href={textHash(next.id)}>
              Next page &rsaquo;
            </a>
          ) : (
            <span className="button disabled" aria-disabled="true">
              Next page &rsaquo;
            </span>
          )}
          <a className="button" href="#/abbreviations">
            Abbreviations
          </a>
        </span>
      </header>
      <div className="abbr-tools">
        <label className="check">
          <input type="checkbox" checked={mark} onChange={(e) => setMark(e.target.checked)} /> Mark ambiguous abbreviations <code>[?]</code>
        </label>
        <label className="check">
          <input type="checkbox" checked={predicted} onChange={(e) => setPredicted(e.target.checked)} /> Show predicted lines
        </label>
      </div>
      {error && (
        <p role="alert" className="error banner">
          <span>{error}</span>
        </p>
      )}
      {!text && !error && <p className="muted">Loading...</p>}
      {text && text.lines.length === 0 && <p className="muted">This page has no transcribed text yet (ignored and empty lines are left out).</p>}
      {text && text.lines.length > 0 && (
        <table className="text-table">
          <caption className="sr-only">Raw and expanded text of the page</caption>
          <thead>
            <tr>
              <th scope="col">Line</th>
              <th scope="col">As written</th>
              <th scope="col">Expanded</th>
            </tr>
          </thead>
          <tbody>
            {text.lines.map((l) => (
              <tr key={l.line_id} data-testid={`text-line-${l.line_id}`}>
                <th scope="row">
                  <a href={pageHash(pageId, l.line_id)} title="Open this line in the editor">
                    {l.index + 1}
                  </a>
                </th>
                <td className="raw">
                  <HighlightedRaw raw={l.raw} spans={l.spans} />
                  {l.text_source === "predicted" && <span className="badge pred">predicted</span>}
                </td>
                <td className="exp">{l.expanded}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {ambiguousLines > 0 && (
        <p className="muted small" role="note">
          {ambiguousLines} {ambiguousLines === 1 ? "line has" : "lines have"} an ambiguous abbreviation; it is left as written
          {mark ? " and marked [?]" : ""}. Resolve it by editing the dictionary on the Abbreviations page.
        </p>
      )}
    </main>
  );
}
