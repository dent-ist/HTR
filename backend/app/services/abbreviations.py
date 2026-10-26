"""The personal abbreviation dictionary: CRUD, occurrence counts, and expansion of line / page text.

The matching rules live in app/abbreviations.py (pure); this module connects them to the `abbreviations` table.
Expansion always runs on the stored (diplomatic) text; expanded text is computed on demand and never stored.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .. import abbreviations as engine
from ..abbreviations import Dictionary, Entry
from ..db import now_iso, write_tx
from ..errors import Conflict, Invalid, NotFound

PREDICTED_MARKER = "[predicted] "  # leading marker for predicted (not yet reviewed) lines in text output


def _entry(r: sqlite3.Row) -> Entry:
    return Entry(r["id"], r["written"], r["expansion"], bool(r["case_sensitive"]))


def entries(conn: sqlite3.Connection) -> list[Entry]:
    return [_entry(r) for r in conn.execute("SELECT * FROM abbreviations ORDER BY id")]


def dictionary(conn: sqlite3.Connection) -> Dictionary:
    return Dictionary(entries(conn))


def get_row(conn: sqlite3.Connection, abbreviation_id: int) -> sqlite3.Row:
    r = conn.execute("SELECT * FROM abbreviations WHERE id = ?", (abbreviation_id,)).fetchone()
    if r is None:
        raise NotFound(f"abbreviation {abbreviation_id} not found")
    return r


def _candidates(r: sqlite3.Row, everyone: list[Entry]) -> list[dict[str, Any]]:
    """The other rows that can match the same token with a different expansion (empty when unambiguous)."""
    me = _entry(r)
    return [
        {"id": o.id, "written": o.written, "expansion": o.expansion, "case_sensitive": o.case_sensitive}
        for o in everyone
        if o.id != me.id and engine.conflicts(me, o) and o.expansion != me.expansion
    ]


def _view(r: sqlite3.Row, everyone: list[Entry], count: int) -> dict[str, Any]:
    cands = _candidates(r, everyone)
    return {
        "id": r["id"],
        "written": r["written"],
        "expansion": r["expansion"],
        "notes": r["notes"],
        "case_sensitive": bool(r["case_sensitive"]),
        "count": count,
        "ambiguous": bool(cands),
        # every expansion that competes for this written form, this row's own first
        "candidates": (
            [{"id": r["id"], "expansion": r["expansion"]}]
            + [{"id": c["id"], "expansion": c["expansion"]} for c in cands]
            if cands
            else []
        ),
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    }


def count_occurrences(conn: sqlite3.Connection, d: Dictionary | None = None) -> dict[int, int]:
    """Entry id -> number of tokens it applies to in the text of all non-ignored lines. An ambiguous token counts for
    each of its candidates."""
    d = d or dictionary(conn)
    counts: dict[int, int] = {}
    if not d.entries:
        return counts
    for r in conn.execute("SELECT text FROM lines WHERE ignored = 0 AND text <> ''"):
        for m in engine.scan(r["text"], d):
            for e in m.entries:
                counts[e.id] = counts.get(e.id, 0) + 1
    return counts


def refresh_counts(conn: sqlite3.Connection) -> dict[int, int]:
    """Recompute the occurrence counts and store those that changed (the `count` column is a cache for readers
    that do not go through the API)."""
    counts = count_occurrences(conn)
    stale = [
        (counts.get(r["id"], 0), r["id"])
        for r in conn.execute("SELECT id, count FROM abbreviations")
        if r["count"] != counts.get(r["id"], 0)
    ]
    if stale:
        with write_tx(conn):
            conn.executemany("UPDATE abbreviations SET count = ? WHERE id = ?", stale)
    return counts


def list_abbreviations(
    conn: sqlite3.Connection, q: str | None = None, ambiguous: bool | None = None, sort: str = "written"
) -> list[dict]:
    everyone = entries(conn)
    counts = refresh_counts(conn)
    out = [_view(r, everyone, counts.get(r["id"], 0)) for r in conn.execute("SELECT * FROM abbreviations")]
    if q:
        needle = q.strip().casefold()
        out = [
            a
            for a in out
            if needle in a["written"].casefold()
            or needle in a["expansion"].casefold()
            or needle in (a["notes"] or "").casefold()
        ]
    if ambiguous is not None:
        out = [a for a in out if a["ambiguous"] == ambiguous]
    if sort == "count":
        out.sort(key=lambda a: (-a["count"], a["written"].casefold(), a["id"]))
    else:
        out.sort(key=lambda a: (a["written"].casefold(), a["written"], a["expansion"].casefold(), a["id"]))
    return out


def get_abbreviation(conn: sqlite3.Connection, abbreviation_id: int) -> dict:
    r = get_row(conn, abbreviation_id)
    counts = count_occurrences(conn)
    return _view(r, entries(conn), counts.get(abbreviation_id, 0))


def _clean(written: str, expansion: str, notes: str | None) -> tuple[str, str, str | None]:
    try:
        w = engine.validate_written(written)
        x = engine.validate_expansion(expansion)
    except ValueError as e:
        raise Invalid(str(e)) from None
    if w.casefold() == x.casefold():
        raise Invalid("the expansion is the same as the written form")
    n = (notes or "").strip()
    if len(n) > 500:
        raise Invalid("notes are longer than 500 characters")
    return w, x, n or None


def create(
    conn: sqlite3.Connection, written: str, expansion: str, notes: str | None = None, case_sensitive: bool = False
) -> dict:
    w, x, n = _clean(written, expansion, notes)
    now = now_iso()
    try:
        with write_tx(conn):
            cur = conn.execute(
                "INSERT INTO abbreviations (written, expansion, notes, case_sensitive, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (w, x, n, int(case_sensitive), now, now),
            )
    except sqlite3.IntegrityError:
        dup = conn.execute("SELECT id FROM abbreviations WHERE written = ? AND expansion = ?", (w, x)).fetchone()
        raise Conflict(f"'{w}' = '{x}' is already in the dictionary (id {dup['id'] if dup else '?'})") from None
    return get_abbreviation(conn, cur.lastrowid)


def update(conn: sqlite3.Connection, abbreviation_id: int, changes: dict[str, Any]) -> dict:
    r = get_row(conn, abbreviation_id)
    w, x, n = _clean(
        changes.get("written", r["written"]),
        changes.get("expansion", r["expansion"]),
        changes["notes"] if "notes" in changes else r["notes"],
    )
    cs = int(changes["case_sensitive"]) if "case_sensitive" in changes else r["case_sensitive"]
    try:
        with write_tx(conn):
            conn.execute(
                "UPDATE abbreviations SET written = ?, expansion = ?, notes = ?, case_sensitive = ?, updated_at = ? "
                "WHERE id = ?",
                (w, x, n, cs, now_iso(), abbreviation_id),
            )
    except sqlite3.IntegrityError:
        raise Conflict(f"'{w}' = '{x}' is already in the dictionary") from None
    return get_abbreviation(conn, abbreviation_id)


def delete(conn: sqlite3.Connection, abbreviation_id: int) -> None:
    get_row(conn, abbreviation_id)
    with write_tx(conn):
        conn.execute("DELETE FROM abbreviations WHERE id = ?", (abbreviation_id,))


def occurrences(conn: sqlite3.Connection, abbreviation_id: int, limit: int = 50, offset: int = 0) -> dict:
    """Where the entry applies: one item per matching token in non-ignored lines, in document / page / reading
    order, paginated. `ambiguous` is true when other expansions compete for the token."""
    get_row(conn, abbreviation_id)
    d = dictionary(conn)
    items: list[dict[str, Any]] = []
    rows = conn.execute(
        'SELECT l.id, l.page_id, l."index", l.text, l.text_source, p.document_id, p.original_filename, d.name '
        "FROM lines l JOIN pages p ON p.id = l.page_id JOIN documents d ON d.id = p.document_id "
        "WHERE l.ignored = 0 AND l.text <> '' ORDER BY p.document_id, p.id, l.\"index\", l.id"
    )
    for r in rows:
        for m in engine.scan(r["text"], d):
            if any(e.id == abbreviation_id for e in m.entries):
                items.append(
                    {
                        "line_id": r["id"],
                        "page_id": r["page_id"],
                        "document_id": r["document_id"],
                        "document_name": r["name"],
                        "page_filename": r["original_filename"],
                        "line_index": r["index"],
                        "text": r["text"],
                        "text_source": r["text_source"],
                        "start": m.start,
                        "end": m.end,
                        "ambiguous": len({e.expansion for e in m.entries}) > 1,
                    }
                )
    return {"total": len(items), "limit": limit, "offset": offset, "items": items[offset : offset + limit]}


# ---- expansion of text -----------------------------------------------------------------------------------------


def expand_text(
    conn: sqlite3.Connection,
    text: str,
    ambiguous: engine.Ambiguous = "raw",
    d: Dictionary | None = None,
    include_unmatched: bool = True,
) -> dict:
    res = engine.expand(text, d or dictionary(conn), ambiguous)
    spans = [s.as_dict() for s in res.spans if include_unmatched or s.status != "unmatched"]
    return {"text": res.text, "spans": spans}


def page_text_lines(conn: sqlite3.Connection, page_id: int, include_predicted: bool = True) -> list[sqlite3.Row]:
    """The lines that carry text for reading / export: not ignored, non-empty, in reading order. Predicted lines
    only when `include_predicted`."""
    rows = conn.execute(
        "SELECT * FROM lines WHERE page_id = ? AND ignored = 0 AND text <> '' ORDER BY \"index\", id", (page_id,)
    ).fetchall()
    return [r for r in rows if include_predicted or r["text_source"] != "predicted"]


def render_lines(rows: list[sqlite3.Row], d: Dictionary | None, ambiguous: engine.Ambiguous = "raw") -> list[dict]:
    """Per line: raw text, expanded text (raw when `d` is None) and spans; the predicted marker is added by the
    caller through `line_output`."""
    out = []
    for r in rows:
        if d is None:
            expanded, spans = r["text"], []
        else:
            res = engine.expand(r["text"], d, ambiguous)
            expanded, spans = res.text, [s.as_dict() for s in res.spans if s.status != "unmatched"]
        out.append(
            {
                "line_id": r["id"],
                "index": r["index"],
                "raw": r["text"],
                "expanded": expanded,
                "text_source": r["text_source"],
                "is_ground_truth": bool(r["is_ground_truth"]),
                "confidence": r["confidence"],
                "spans": spans,
            }
        )
    return out


def line_output(item: dict, expanded: bool, mark_predicted: bool = True) -> str:
    text = item["expanded"] if expanded else item["raw"]
    return (PREDICTED_MARKER + text) if mark_predicted and item["text_source"] == "predicted" else text


def page_text(
    conn: sqlite3.Connection,
    page_id: int,
    expanded: bool = False,
    ambiguous: engine.Ambiguous = "raw",
    include_predicted: bool = True,
    d: Dictionary | None = None,
) -> dict:
    page = conn.execute("SELECT id, document_id, original_filename FROM pages WHERE id = ?", (page_id,)).fetchone()
    if page is None:
        raise NotFound(f"page {page_id} not found")
    d = d or dictionary(conn)
    lines = render_lines(page_text_lines(conn, page_id, include_predicted), d, ambiguous)
    return {
        "page_id": page_id,
        "document_id": page["document_id"],
        "expanded": expanded,
        "ambiguous": ambiguous,
        "include_predicted": include_predicted,
        "lines": lines,
        "text": "\n".join(line_output(x, expanded) for x in lines),
    }


def line_view(conn: sqlite3.Connection, line: dict, expanded: bool, ambiguous: engine.Ambiguous = "raw") -> dict:
    """The line object, plus `expanded_text` and `spans` when `expanded`."""
    if not expanded:
        return line
    res = engine.expand(line["text"] or "", dictionary(conn), ambiguous)
    return {**line, "expanded_text": res.text, "spans": [s.as_dict() for s in res.spans if s.status != "unmatched"]}
