"""Segmentation repair operations: merge, split, ignore region, reorder, restore (undo).

Each runs inside sync.page_transaction (DB change + status + PAGE-XML rewrite, all or nothing) and keeps
lines.index contiguous (0..n-1 in reading order).
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from typing import Any

from ..config import Settings
from ..errors import Conflict, Invalid, NotFound
from ..pipeline.segment import Line
from . import geometry, review, sync
from . import lines as lines_svc
from . import pages as pages_svc
from .rows import json_dumps_points, json_loads

MAX_TEXT = 2000
IGNORE_OVERLAP = 0.5


def page_detail(conn: sqlite3.Connection, page_id: int) -> dict:
    return {**pages_svc.get_page(conn, page_id), "lines": lines_svc.list_lines(conn, page_id)}


def _pts(raw) -> list[tuple[int, int]]:
    return [(int(p[0]), int(p[1])) for p in raw]


def _page_size(conn: sqlite3.Connection, page_id: int) -> tuple[int, int]:
    return lines_svc._page_size(conn, page_id)


# ---- merge ---------------------------------------------------------------------------------------------------


def merge_lines(conn: sqlite3.Connection, settings: Settings, line_ids: list[int]) -> dict:
    """Merge lines of one page into a new line at the first one's reading-order position.

    Baseline: all baseline points left to right. Polygon: union (convex hull if disjoint), clipped to the page.
    Text: non-empty texts joined by a space in reading order. Ground truth is reset (the combined text has not been
    verified); the result is never ignored. The inputs are deleted.
    """
    if len(set(line_ids)) != len(line_ids):
        raise Invalid("line_ids must not repeat")
    if len(line_ids) < 2:
        raise Invalid("merge needs at least two lines")
    rows = [lines_svc.get_line_row(conn, i) for i in line_ids]
    page_ids = {r["page_id"] for r in rows}
    if len(page_ids) != 1:
        raise Invalid("all lines must be on the same page")
    page_id = page_ids.pop()
    size = _page_size(conn, page_id)
    rows.sort(key=lambda r: (r["index"], r["id"]))
    texts = [r["text"] for r in rows if r["text"]]
    text = " ".join(texts)
    if len(text) > MAX_TEXT:
        raise Invalid(f"merged text would be longer than {MAX_TEXT} characters")
    baseline = geometry.merge_baselines([_pts(json_loads(r["baseline"])) for r in rows])
    polygon = geometry.merge_polygons([_pts(json_loads(r["polygon"])) for r in rows], size)
    with sync.page_transaction(conn, settings, page_id):
        for lid in line_ids:  # a line may have vanished since the read above
            if conn.execute("SELECT 1 FROM lines WHERE id = ?", (lid,)).fetchone() is None:
                raise NotFound(f"line {lid} not found")
        ids = lines_svc._ordered_ids(conn, page_id)
        position = min(ids.index(r["id"]) for r in rows)
        new_id = sync.insert_line(
            conn,
            page_id,
            Line(
                id="",
                baseline=baseline,
                polygon=polygon,
                index=len(ids),
                text=text,
                text_source="manual" if text else None,
            ),
        )
        ids = [i for i in ids if i not in set(line_ids)]
        ids.insert(position, new_id)
        conn.executemany("DELETE FROM lines WHERE id = ?", [(i,) for i in line_ids])
        lines_svc._renumber(conn, ids)
    return {"line": lines_svc.get_line(conn, new_id), "removed_ids": list(line_ids), "page": page_detail(conn, page_id)}


# ---- split ---------------------------------------------------------------------------------------------------


def split_line(conn: sqlite3.Connection, settings: Settings, line_id: int, x: float, y: float) -> dict:
    """Cut a line at x. The left half keeps the id; the right half is new and follows it in reading order.

    Text is cleared on both halves (the old text is returned as `old_text`); ground truth is reset. A prediction on the
    line is dropped with its text (`predicted_by_model_id` / `predicted_text` are cleared: the halves are new lines
    nobody predicted, and no review event is written). Undo (restore) gives all of it back.
    """
    cur = lines_svc.get_line_row(conn, line_id)
    page_id = cur["page_id"]
    size = _page_size(conn, page_id)
    lines_svc.check_points([(x, y)], "split point", 1, size)
    (lb, lp), (rb, rp) = geometry.split_line(
        _pts(json_loads(cur["baseline"])), _pts(json_loads(cur["polygon"])), x, size
    )
    old_text = cur["text"] or ""
    with sync.page_transaction(conn, settings, page_id):
        if conn.execute("SELECT 1 FROM lines WHERE id = ?", (line_id,)).fetchone() is None:
            raise NotFound(f"line {line_id} not found")
        conn.execute(
            "UPDATE lines SET baseline = ?, polygon = ?, text = '', text_source = NULL, confidence = NULL, "
            "is_ground_truth = 0, predicted_by_model_id = NULL, predicted_text = NULL WHERE id = ?",
            (json_dumps_points(lb), json_dumps_points(lp), line_id),
        )
        ids = lines_svc._ordered_ids(conn, page_id)
        new_id = sync.insert_line(
            conn,
            page_id,
            Line(id="", baseline=rb, polygon=rp, index=len(ids), ignored=bool(cur["ignored"])),
        )
        ids.insert(ids.index(line_id) + 1, new_id)
        lines_svc._renumber(conn, ids)
    return {
        "lines": [lines_svc.get_line(conn, line_id), lines_svc.get_line(conn, new_id)],
        "old_text": old_text,
        "page": page_detail(conn, page_id),
    }


# ---- ignore region -------------------------------------------------------------------------------------------


def ignore_region(conn: sqlite3.Connection, settings: Settings, page_id: int, body: dict[str, Any]) -> dict:
    """Mark every active line with >= 50% of its polygon area inside the region as ignored. For each predicted line
    this writes an `ignored` review event (same transaction), like ignoring the line through PATCH."""
    size = _page_size(conn, page_id)
    if body.get("polygon") is not None:
        region = geometry.region_from_points(lines_svc.check_points(body["polygon"], "region", 3, size))
    else:
        x, y, w, h = (float(body[k]) for k in ("x", "y", "w", "h"))
        lines_svc.check_points([(x, y), (x + w, y + h)], "region", 2, size)
        region = geometry.region_from_rect(x, y, w, h)
    if region.is_empty or region.area <= 0:
        raise Invalid("region is empty")
    affected: list[int] = []
    with sync.page_transaction(conn, settings, page_id):
        for r in conn.execute("SELECT * FROM lines WHERE page_id = ? AND ignored = 0", (page_id,)).fetchall():
            if geometry.region_overlap_share(_pts(json_loads(r["polygon"])), region) >= IGNORE_OVERLAP:
                affected.append(r["id"])
                # Ignoring a predicted line is a review decision, exactly as if it had been ignored one by one.
                review.record_if_reviewed(conn, r, None, None, 1)
        conn.executemany("UPDATE lines SET ignored = 1 WHERE id = ?", [(i,) for i in affected])
    return {"affected": affected, "page": page_detail(conn, page_id)}


# ---- reorder -------------------------------------------------------------------------------------------------


def reorder(conn: sqlite3.Connection, settings: Settings, page_id: int, line_ids: list[int]) -> dict:
    pages_svc.get_page_row(conn, page_id)
    with sync.page_transaction(conn, settings, page_id):
        current = lines_svc._ordered_ids(conn, page_id)
        if len(set(line_ids)) != len(line_ids) or sorted(line_ids) != sorted(current):
            raise Invalid("line_ids must list every line of the page exactly once")
        lines_svc._renumber(conn, line_ids)
    return page_detail(conn, page_id)


# ---- restore (undo) ------------------------------------------------------------------------------------------


def restore(conn: sqlite3.Connection, settings: Settings, page_id: int, snapshot: list[dict], force: bool) -> dict:
    """Replace the page's lines with `snapshot` (in that order), keeping ids where possible.

    Prediction state is given back exactly: each snapshot line's `predicted_by_model_id` and `predicted_text` are
    restored (also on lines that are re-created because the operation deleted them), and an `ignored` review event that
    retired the same prediction is withdrawn. Merge and split themselves do not carry predictions over (a merged or
    split line is a new, unpredicted line with its text rules as documented), so undoing them is what brings the
    prediction back.

    Safety: unless force=true this refuses (409) if the page currently holds non-empty text that the snapshot would
    not give back (compared word by word: the words of every current line must be found in the snapshot's texts), so
    an undo can never silently drop typed work.
    """
    pages_svc.get_page_row(conn, page_id)
    size = _page_size(conn, page_id)
    clean: list[dict] = []
    for i, ln in enumerate(snapshot):
        if ln["is_ground_truth"] and not ln["text"]:
            raise Invalid(f"snapshot line {i}: a ground-truth line needs non-empty text")
        clean.append(
            {
                **ln,
                "baseline": lines_svc.check_points(ln["baseline"], "baseline", 2, size),
                "polygon": lines_svc.check_points(ln["polygon"], "polygon", 3, size),
            }
        )
    with sync.page_transaction(conn, settings, page_id):
        current = conn.execute("SELECT id, text FROM lines WHERE page_id = ?", (page_id,)).fetchall()
        # A current text is covered when all its words can be found in the snapshot's texts (so undoing a merge,
        # whose text is the join of the originals, is fine; typed-in new words are not covered).
        pool = Counter(w for s in clean for w in s["text"].split())
        lost: list[str] = []
        for r in current:
            need = Counter((r["text"] or "").split())
            if need - pool:
                lost.append(r["text"])
            else:
                pool -= need
        if lost and not force:
            shown = "; ".join(f'"{t}"' for t in lost[:5])
            raise Conflict(f"restoring would discard text of {len(lost)} line(s): {shown}")
        mine = {r["id"] for r in current}
        keep = {s["id"] for s in clean if s.get("id") in mine}
        for lid in mine - keep:
            conn.execute("DELETE FROM lines WHERE id = ?", (lid,))
        known_models = {r[0] for r in conn.execute("SELECT id FROM models")}
        ids: list[int] = []
        for s in clean:
            sid = s.get("id")
            model_id = s.get("predicted_by_model_id")
            if model_id not in known_models:
                model_id = None  # a model that no longer exists cannot be given back
            predicted_text = None
            if model_id:
                given = s.get("predicted_text")
                predicted_text = given if given is not None else s["text"]
            values = (
                json_dumps_points(s["baseline"]),
                json_dumps_points(s["polygon"]),
                s["text"],
                s["text_source"] if s["text"] else None,
                s["confidence"],
                model_id,
                predicted_text,
                int(s["is_ground_truth"]),
                int(s["ignored"]),
            )
            if sid in keep:
                conn.execute(
                    "UPDATE lines SET baseline=?, polygon=?, text=?, text_source=?, confidence=?, "
                    "predicted_by_model_id=?, predicted_text=?, is_ground_truth=?, ignored=? WHERE id=?",
                    (*values, sid),
                )
                ids.append(sid)
            else:
                taken = sid is not None and conn.execute("SELECT 1 FROM lines WHERE id = ?", (sid,)).fetchone()
                cols = (
                    'page_id, "index", baseline, polygon, text, text_source, confidence, predicted_by_model_id, '
                    "predicted_text, is_ground_truth, ignored"
                )
                if sid is not None and not taken:
                    cur = conn.execute(
                        f"INSERT INTO lines (id, {cols}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (sid, page_id, len(ids), *values),
                    )
                else:
                    cur = conn.execute(
                        f"INSERT INTO lines ({cols}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (page_id, len(ids), *values),
                    )
                ids.append(int(cur.lastrowid))
            if model_id:
                # The prediction is back in force, so the 'ignored' review decision that retired it (an
                # ignore-region that is being undone) is withdrawn: one prediction yields at most one event.
                conn.execute(
                    "DELETE FROM review_events WHERE line_id = ? AND model_id = ? AND predicted_text = ? "
                    "AND action = 'ignored'",
                    (ids[-1], model_id, predicted_text),
                )
        # indices of kept rows may collide transiently; renumber in two passes is unnecessary (no UNIQUE), just set
        lines_svc._renumber(conn, ids)
    return page_detail(conn, page_id)
