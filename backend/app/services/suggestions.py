"""Reviewable abbreviation suggestions from a local LLM (optional; see app/abbreviation_llm.py).

A suggestion is a (written, expansion) proposal with the line it came from. It lives in the `suggestions` table until
the user accepts it (which creates a normal dictionary entry), edits and accepts it, or rejects it (remembered, so the
same pair is never proposed again). Suggestions never change a transcription.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from dataclasses import dataclass

from .. import abbreviations as engine
from ..abbreviation_llm import MAX_CONTEXT_LINES, Candidate
from ..db import now_iso, write_tx
from ..errors import Conflict, Invalid, NotFound
from . import abbreviations as abbr_svc

DEFAULT_MAX_TOKENS = 25
MAX_TOKENS_LIMIT = 100

# Short everyday words that are not abbreviations (so they are not sent to the model as candidates).
_COMMON = frozenset(
    "a an the of to in on at is it he she we i me my us do go no so up by or if as be am are was were and for but not "
    "you had has his her its our out who can may did one two six ten new old all any now how why yes got get let put "
    "set say saw see too via per off yet nor was him how".split()
)
_ALPHA = re.compile(r"[^\W\d_]", re.UNICODE)


@dataclass(frozen=True)
class TokenContext:
    token: str  # the core of the token (surrounding punctuation stripped)
    count: int
    line_id: int
    line: str
    before: list[str]
    after: list[str]


def _core(token: str) -> str:
    """The token without wrapping punctuation (keeps `c/o` and `s.o.b.`; drops the sentence period of `SOB.`)."""
    n = len(token)
    lead = 0
    while lead < n and token[lead] in engine.LEADING_PUNCT:
        lead += 1
    trail = 0
    while trail < n - lead and token[n - 1 - trail] in engine.TRAILING_PUNCT:
        trail += 1
    core = token[lead : n - trail]
    if trail and token[n - trail] == "." and "." in core:
        core += "."  # a dotted abbreviation such as s.o.b. keeps its last period
    return core or token


def looks_like_abbreviation(core: str) -> bool:
    """Cheap shape test (there is no word list): contains `/` or an inner `.`, is a short ALL-CAPS token, or is a very
    short lowercase token that is not an everyday word."""
    if not core or len(core) > engine.MAX_WRITTEN:
        return False
    letters = _ALPHA.findall(core)
    if not letters:
        return False
    if "/" in core or "." in core.rstrip("."):
        return True
    if len(core) <= 6 and len(letters) >= 2 and core.upper() == core and any(c.isalpha() for c in core):
        return True
    return len(core) <= 3 and core.isalpha() and core.casefold() not in _COMMON


def _known(conn: sqlite3.Connection) -> tuple[set[str], set[tuple[str, str]]]:
    """(casefolded written forms that already have any suggestion, (written, expansion) pairs rejected)."""
    seen, rejected = set(), set()
    for r in conn.execute("SELECT written, expansion, status FROM suggestions"):
        seen.add(r["written"].casefold())
        if r["status"] == "rejected":
            rejected.add((r["written"].casefold(), r["expansion"].casefold()))
    return seen, rejected


def rejected_pairs(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    return _known(conn)[1]


def find_candidates(
    conn: sqlite3.Connection, limit: int = DEFAULT_MAX_TOKENS, tokens: list[str] | None = None, context_lines: int = 1
) -> list[TokenContext]:
    """Tokens worth asking the model about, most frequent first, each with the first line it appears in.

    Only text the user wrote or verified is looked at (`manual` source or ground truth), never raw predictions. A token
    qualifies when it is not an unambiguous dictionary entry, looks like an abbreviation, and has no suggestion yet
    (pending, accepted or rejected). Explicit `tokens` skip the shape test and the "already suggested" test, but a
    rejected (written, expansion) pair is still never proposed again (the caller filters the model's answer).
    """
    context_lines = max(0, min(int(context_lines), MAX_CONTEXT_LINES))
    d = abbr_svc.dictionary(conn)
    seen, _ = _known(conn)
    wanted = {t.casefold() for t in tokens} if tokens else None
    counts: Counter[str] = Counter()
    first: dict[str, tuple[int, int, str]] = {}  # key -> (line id, page id, display token)
    page_lines: dict[int, list[sqlite3.Row]] = {}
    rows = conn.execute(
        'SELECT l.id, l.page_id, l."index", l.text FROM lines l JOIN pages p ON p.id = l.page_id '
        "WHERE l.ignored = 0 AND l.text <> '' AND (l.text_source = 'manual' OR l.is_ground_truth = 1) "
        'ORDER BY p.document_id, p.id, l."index", l.id'
    ).fetchall()
    for r in rows:
        for m in engine._TOKEN.finditer(r["text"]):
            raw = m.group()
            core = _core(raw)
            key = core.casefold()
            if wanted is not None:
                if key not in wanted:
                    continue
            else:
                if key in seen or not looks_like_abbreviation(core):
                    continue
            hit = d.match_token(raw)
            if hit is not None and len({e.expansion for e in hit[2]}) == 1:
                continue  # an unambiguous dictionary entry: nothing to suggest
            counts[key] += 1
            first.setdefault(key, (r["id"], r["page_id"], core))
    ordered = [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    if wanted is not None:
        ordered = [k for k in ordered if k in wanted]
    out: list[TokenContext] = []
    for key in ordered[: max(1, min(limit, MAX_TOKENS_LIMIT))]:
        line_id, page_id, display = first[key]
        if page_id not in page_lines:
            page_lines[page_id] = conn.execute(
                "SELECT id, text FROM lines WHERE page_id = ? AND ignored = 0 AND text <> '' "
                "AND (text_source = 'manual' OR is_ground_truth = 1) ORDER BY \"index\", id",
                (page_id,),
            ).fetchall()
        lines = page_lines[page_id]
        i = next(k for k, ln in enumerate(lines) if ln["id"] == line_id)
        out.append(
            TokenContext(
                token=display,
                count=counts[key],
                line_id=line_id,
                line=lines[i]["text"],
                before=[ln["text"] for ln in lines[max(0, i - context_lines) : i]],
                after=[ln["text"] for ln in lines[i + 1 : i + 1 + context_lines]],
            )
        )
    return out


def store(conn: sqlite3.Connection, ctx: TokenContext, candidates: list[Candidate], model: str) -> int:
    """Save new suggestions for a token; pairs that exist (pending, accepted or rejected) or that are already
    dictionary entries are skipped. Returns how many rows were added."""
    _, rejected = _known(conn)
    existing = {(e.written.casefold(), e.expansion.casefold()) for e in abbr_svc.entries(conn)}
    added = 0
    with write_tx(conn):
        for c in candidates:
            pair = (ctx.token.casefold(), c.expansion.casefold())
            if pair in rejected or pair in existing:
                continue
            cur = conn.execute(
                "INSERT OR IGNORE INTO suggestions (written, expansion, rationale, line_id, context, model, status, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
                (ctx.token, c.expansion, c.rationale, ctx.line_id, ctx.line, model, now_iso()),
            )
            added += cur.rowcount
    return added


# ---- review ----------------------------------------------------------------------------------------------------


def _view(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "written": r["written"],
        "expansion": r["expansion"],
        "rationale": r["rationale"],
        "line_id": r["line_id"],
        "context": r["context"],
        "model": r["model"],
        "status": r["status"],
        "abbreviation_id": r["abbreviation_id"],
        "created_at": r["created_at"],
        "resolved_at": r["resolved_at"],
    }


def list_suggestions(conn: sqlite3.Connection, status: str = "pending") -> list[dict]:
    where, args = ("", ()) if status == "all" else (" WHERE status = ?", (status,))
    rows = conn.execute(f"SELECT * FROM suggestions{where} ORDER BY written COLLATE NOCASE, id", args)
    return [_view(r) for r in rows]


def get_row(conn: sqlite3.Connection, suggestion_id: int) -> sqlite3.Row:
    r = conn.execute("SELECT * FROM suggestions WHERE id = ?", (suggestion_id,)).fetchone()
    if r is None:
        raise NotFound(f"suggestion {suggestion_id} not found")
    return r


def accept(conn: sqlite3.Connection, suggestion_id: int, edits: dict | None = None) -> dict:
    """Create the dictionary entry (optionally with the user's edited written form / expansion / notes) and mark the
    suggestion accepted. If the dictionary already has exactly that pair, it is linked instead of duplicated."""
    r = get_row(conn, suggestion_id)
    if r["status"] != "pending":
        raise Conflict(f"suggestion {suggestion_id} was already {r['status']}", "suggestion_resolved")
    edits = edits or {}
    written = edits.get("written") or r["written"]
    expansion = edits.get("expansion") or r["expansion"]
    try:
        entry = abbr_svc.create(conn, written, expansion, edits.get("notes"), bool(edits.get("case_sensitive", False)))
    except Conflict:
        w, x, _ = abbr_svc._clean(written, expansion, None)
        row = conn.execute("SELECT id FROM abbreviations WHERE written = ? AND expansion = ?", (w, x)).fetchone()
        if row is None:
            raise
        entry = abbr_svc.get_abbreviation(conn, row["id"])
    with write_tx(conn):
        conn.execute(
            "UPDATE suggestions SET status = 'accepted', abbreviation_id = ?, resolved_at = ? WHERE id = ?",
            (entry["id"], now_iso(), suggestion_id),
        )
    return {"suggestion": _view(get_row(conn, suggestion_id)), "abbreviation": entry}


def reject(conn: sqlite3.Connection, suggestion_id: int) -> dict:
    r = get_row(conn, suggestion_id)
    if r["status"] != "pending":
        raise Conflict(f"suggestion {suggestion_id} was already {r['status']}", "suggestion_resolved")
    with write_tx(conn):
        conn.execute(
            "UPDATE suggestions SET status = 'rejected', resolved_at = ? WHERE id = ?", (now_iso(), suggestion_id)
        )
    return _view(get_row(conn, suggestion_id))


def validate_tokens(tokens: list[str] | None) -> list[str] | None:
    if tokens is None:
        return None
    out = []
    for t in tokens:
        try:
            out.append(engine.validate_written(t))
        except ValueError as e:
            raise Invalid(f"token '{t[:20]}': {e}") from None
    return out
