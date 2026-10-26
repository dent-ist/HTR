import { useCallback, useEffect, useMemo, useState } from "react";
import type { ApiClient } from "../api/client";
import type { Abbreviation, ExpandResult, OccurrencePage } from "../api/types";
import { pageHash } from "../router";

const errText = (e: unknown): string => (e instanceof Error ? e.message : String(e));
const PAGE = 25;

type Filter = "all" | "ambiguous";
type Sort = "written" | "count";

/** Client-side search over written form, expansion and notes (case-insensitive). */
export function filterAbbreviations(items: Abbreviation[], q: string, filter: Filter, sort: Sort): Abbreviation[] {
  const needle = q.trim().toLowerCase();
  const out = items.filter(
    (a) =>
      (filter === "all" || a.ambiguous) &&
      (!needle || [a.written, a.expansion, a.notes ?? ""].some((s) => s.toLowerCase().includes(needle))),
  );
  out.sort((a, b) =>
    sort === "count" && a.count !== b.count
      ? b.count - a.count
      : a.written.toLowerCase().localeCompare(b.written.toLowerCase()) || a.expansion.localeCompare(b.expansion),
  );
  return out;
}

interface Edit {
  id: number;
  written: string;
  expansion: string;
  notes: string;
  case_sensitive: boolean;
}

/**
 * The personal abbreviation dictionary: written form -> expansion. Recognition output stays as written; the dictionary
 * only drives the expanded view and the expanded export. Rows that share a written form are ambiguous and are never
 * expanded automatically.
 */
export function Abbreviations({ api }: { api: ApiClient }) {
  const [items, setItems] = useState<Abbreviation[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [sort, setSort] = useState<Sort>("written");
  const [written, setWritten] = useState("");
  const [expansion, setExpansion] = useState("");
  const [notes, setNotes] = useState("");
  const [caseSensitive, setCaseSensitive] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [rowError, setRowError] = useState<string | null>(null);
  const [edit, setEdit] = useState<Edit | null>(null);
  const [openId, setOpenId] = useState<number | null>(null);
  const [occ, setOcc] = useState<OccurrencePage | null>(null);
  const [occError, setOccError] = useState<string | null>(null);
  const [previewText, setPreviewText] = useState("pt c/o SOB");
  const [preview, setPreview] = useState<ExpandResult | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setItems(await api.listAbbreviations());
      setLoadError(null);
    } catch (e) {
      setLoadError(errText(e));
    }
  }, [api]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const shown = useMemo(() => (items ? filterAbbreviations(items, q, filter, sort) : []), [items, q, filter, sort]);
  const ambiguousCount = items?.filter((a) => a.ambiguous).length ?? 0;

  const add = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);
    try {
      await api.createAbbreviation({ written, expansion, notes: notes.trim() || null, case_sensitive: caseSensitive });
      setWritten("");
      setExpansion("");
      setNotes("");
      setCaseSensitive(false);
      await refresh();
    } catch (err) {
      setFormError(errText(err));
    }
  };

  const saveEdit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!edit) return;
    setRowError(null);
    try {
      await api.patchAbbreviation(edit.id, {
        written: edit.written,
        expansion: edit.expansion,
        notes: edit.notes.trim() || null,
        case_sensitive: edit.case_sensitive,
      });
      setEdit(null);
      await refresh();
      if (openId === edit.id) await loadOccurrences(edit.id, 0, true);
    } catch (err) {
      setRowError(errText(err));
    }
  };

  const remove = async (a: Abbreviation) => {
    if (!window.confirm(`Delete "${a.written}" = "${a.expansion}"? Text already written is not changed.`)) return;
    setRowError(null);
    try {
      await api.deleteAbbreviation(a.id);
      if (openId === a.id) setOpenId(null);
      await refresh();
    } catch (err) {
      setRowError(errText(err));
    }
  };

  const loadOccurrences = async (id: number, offset: number, replace: boolean) => {
    setOccError(null);
    try {
      const page = await api.abbreviationOccurrences(id, PAGE, offset);
      setOcc((cur) => (replace || !cur ? page : { ...page, offset: 0, items: [...cur.items, ...page.items] }));
    } catch (err) {
      setOccError(errText(err));
    }
  };

  const toggleOccurrences = (a: Abbreviation) => {
    if (openId === a.id) {
      setOpenId(null);
      return;
    }
    setOpenId(a.id);
    setOcc(null);
    void loadOccurrences(a.id, 0, true);
  };

  const runPreview = async () => {
    setPreviewError(null);
    try {
      setPreview(await api.expandText(previewText, "mark"));
    } catch (err) {
      setPreviewError(errText(err));
    }
  };

  return (
    <main className="training abbreviations">
      <header className="training-head">
        <h1>Abbreviations</h1>
        <span className="actions">
          <a className="button" href="#/">
            Library
          </a>
          <a className="button" href="#/export">
            Export
          </a>
        </span>
      </header>
      <p className="muted small">
        Recognition keeps your text exactly as written. This dictionary maps written forms to expansions for the
        expanded view and the expanded export. A written form with several expansions is ambiguous and is never
        expanded automatically.
      </p>

      {loadError && (
        <p role="alert" className="error banner">
          <span>{loadError}</span>
          <button onClick={() => void refresh()}>Retry</button>
        </p>
      )}

      <section className="panel" aria-labelledby="add-heading">
        <h2 id="add-heading">Add an abbreviation</h2>
        <form className="abbr-form" onSubmit={(e) => void add(e)}>
          <label>
            Written form
            <input value={written} onChange={(e) => setWritten(e.target.value)} placeholder="e.g. c/o" maxLength={64} />
          </label>
          <label>
            Expansion
            <input value={expansion} onChange={(e) => setExpansion(e.target.value)} placeholder="e.g. complains of" maxLength={500} />
          </label>
          <label>
            Notes (optional)
            <input value={notes} onChange={(e) => setNotes(e.target.value)} maxLength={500} />
          </label>
          <label className="check">
            <input type="checkbox" checked={caseSensitive} onChange={(e) => setCaseSensitive(e.target.checked)} /> Case sensitive
          </label>
          <button type="submit" disabled={!written.trim() || !expansion.trim()}>
            Add
          </button>
        </form>
        {formError && (
          <p role="alert" className="error" data-testid="abbr-form-error">
            {formError}
          </p>
        )}
      </section>

      <section className="panel" aria-labelledby="dict-heading">
        <h2 id="dict-heading">
          Dictionary{items ? ` (${items.length})` : ""}
          {ambiguousCount > 0 && (
            <span className="badge amb" data-testid="ambiguous-total">
              {ambiguousCount} ambiguous
            </span>
          )}
        </h2>
        <div className="abbr-tools">
          <label>
            Search
            <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="written form, expansion or notes" />
          </label>
          <label>
            Show
            <select value={filter} onChange={(e) => setFilter(e.target.value as Filter)}>
              <option value="all">All</option>
              <option value="ambiguous">Ambiguous only</option>
            </select>
          </label>
          <label>
            Sort by
            <select value={sort} onChange={(e) => setSort(e.target.value as Sort)}>
              <option value="written">Written form</option>
              <option value="count">Occurrences</option>
            </select>
          </label>
        </div>
        {rowError && (
          <p role="alert" className="error banner">
            <span>{rowError}</span>
            <button onClick={() => setRowError(null)}>Dismiss</button>
          </p>
        )}
        {items === null && !loadError && <p className="muted">Loading...</p>}
        {items?.length === 0 && <p className="muted">No abbreviations yet. Add the first one above.</p>}
        {items && items.length > 0 && shown.length === 0 && <p className="muted">Nothing matches this search.</p>}
        {shown.length > 0 && (
          <table className="abbr-table">
            <caption className="sr-only">Abbreviation dictionary</caption>
            <thead>
              <tr>
                <th scope="col">Written</th>
                <th scope="col">Expansion</th>
                <th scope="col">Notes</th>
                <th scope="col">Occurrences</th>
                <th scope="col">Actions</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((a) =>
                edit?.id === a.id ? (
                  <tr key={a.id} className="editing">
                    <td colSpan={5}>
                      <form className="abbr-form" onSubmit={(e) => void saveEdit(e)} aria-label={`Edit ${a.written}`}>
                        <label>
                          Written form
                          <input value={edit.written} onChange={(e) => setEdit({ ...edit, written: e.target.value })} autoFocus />
                        </label>
                        <label>
                          Expansion
                          <input value={edit.expansion} onChange={(e) => setEdit({ ...edit, expansion: e.target.value })} />
                        </label>
                        <label>
                          Notes
                          <input value={edit.notes} onChange={(e) => setEdit({ ...edit, notes: e.target.value })} />
                        </label>
                        <label className="check">
                          <input
                            type="checkbox"
                            checked={edit.case_sensitive}
                            onChange={(e) => setEdit({ ...edit, case_sensitive: e.target.checked })}
                          />{" "}
                          Case sensitive
                        </label>
                        <button type="submit" disabled={!edit.written.trim() || !edit.expansion.trim()}>
                          Save
                        </button>
                        <button type="button" onClick={() => setEdit(null)}>
                          Cancel
                        </button>
                      </form>
                    </td>
                  </tr>
                ) : (
                  <AbbrRow
                    key={a.id}
                    a={a}
                    open={openId === a.id}
                    occ={openId === a.id ? occ : null}
                    occError={openId === a.id ? occError : null}
                    onEdit={() =>
                      setEdit({ id: a.id, written: a.written, expansion: a.expansion, notes: a.notes ?? "", case_sensitive: a.case_sensitive })
                    }
                    onDelete={() => void remove(a)}
                    onToggle={() => toggleOccurrences(a)}
                    onMore={() => occ && void loadOccurrences(a.id, occ.items.length, false)}
                  />
                ),
              )}
            </tbody>
          </table>
        )}
      </section>

      <section className="panel" aria-labelledby="try-heading">
        <h2 id="try-heading">Try the dictionary</h2>
        <label className="block">
          Text as written
          <textarea value={previewText} onChange={(e) => setPreviewText(e.target.value)} rows={2} maxLength={20000} />
        </label>
        <button onClick={() => void runPreview()}>Preview expansion</button>
        {previewError && (
          <p role="alert" className="error">
            {previewError}
          </p>
        )}
        {preview && (
          <p className="preview-out" aria-label="Expanded preview" data-testid="preview-out">
            {preview.text}
          </p>
        )}
        {preview?.spans.some((s) => s.status === "ambiguous") && (
          <p className="muted small">[?] marks an ambiguous abbreviation that was left as written.</p>
        )}
      </section>
    </main>
  );
}

function AbbrRow({
  a,
  open,
  occ,
  occError,
  onEdit,
  onDelete,
  onToggle,
  onMore,
}: {
  a: Abbreviation;
  open: boolean;
  occ: OccurrencePage | null;
  occError: string | null;
  onEdit: () => void;
  onDelete: () => void;
  onToggle: () => void;
  onMore: () => void;
}) {
  return (
    <>
      <tr data-testid={`abbr-${a.id}`}>
        <th scope="row">
          <code>{a.written}</code>
          {a.case_sensitive && (
            <span className="badge" title="Matches only this exact spelling">
              Aa
            </span>
          )}
        </th>
        <td>
          {a.expansion}
          {a.ambiguous && (
            <span
              className="badge amb"
              title={`Competing expansions: ${a.candidates.map((c) => c.expansion).join(" / ")}`}
              data-testid={`ambiguous-${a.id}`}
            >
              ambiguous ({a.candidates.length})
            </span>
          )}
        </td>
        <td className="muted">{a.notes}</td>
        <td>
          <button
            onClick={onToggle}
            aria-expanded={open}
            aria-label={`${a.count} ${a.count === 1 ? "occurrence" : "occurrences"} of ${a.written}`}
            disabled={a.count === 0}
          >
            {a.count}
          </button>
        </td>
        <td className="row-actions">
          <button onClick={onEdit} aria-label={`Edit ${a.written} = ${a.expansion}`}>
            Edit
          </button>
          <button onClick={onDelete} aria-label={`Delete ${a.written} = ${a.expansion}`}>
            Delete
          </button>
        </td>
      </tr>
      {open && (
        <tr className="occurrences">
          <td colSpan={5}>
            {occError && (
              <p role="alert" className="error">
                {occError}
              </p>
            )}
            {!occ && !occError && <p className="muted">Loading...</p>}
            {occ && (
              <>
                <p className="muted small">
                  {occ.total} {occ.total === 1 ? "occurrence" : "occurrences"} in lines that are not ignored (predicted lines included).
                </p>
                <ul className="occ-list" aria-label={`Occurrences of ${a.written}`}>
                  {occ.items.map((o) => (
                    <li key={`${o.line_id}-${o.start}`}>
                      <a href={pageHash(o.page_id, o.line_id)}>
                        {o.document_name} / {o.page_filename}, line {o.line_index + 1}
                      </a>
                      {": "}
                      <span className="occ-text">
                        {o.text.slice(0, o.start)}
                        <mark>{o.text.slice(o.start, o.end)}</mark>
                        {o.text.slice(o.end)}
                      </span>
                      {o.text_source === "predicted" && <span className="badge pred">predicted</span>}
                      {o.ambiguous && <span className="badge amb">ambiguous</span>}
                    </li>
                  ))}
                </ul>
                {occ.items.length < occ.total && <button onClick={onMore}>Show more</button>}
              </>
            )}
          </td>
        </tr>
      )}
    </>
  );
}
