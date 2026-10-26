"""Line CRUD. Every mutation goes through sync.page_transaction (DB change + PAGE-XML rewrite, all or nothing)."""

from __future__ import annotations

import sqlite3
from typing import Any

from ..config import Settings
from ..errors import Conflict, Invalid, NotFound
from ..pipeline.segment import Line
from . import geometry, review, sync
from .rows import json_dumps_points, json_loads, line_dict

PATCHABLE = {"text", "polygon", "baseline", "ignored", "is_ground_truth", "index", "regenerate_polygon"}


def get_line_row(conn: sqlite3.Connection, line_id: int) -> sqlite3.Row:
    r = conn.execute("SELECT * FROM lines WHERE id = ?", (line_id,)).fetchone()
    if r is None:
        raise NotFound(f"line {line_id} not found")
    return r


def get_line(conn: sqlite3.Connection, line_id: int) -> dict:
    return line_dict(get_line_row(conn, line_id))


def list_lines(conn: sqlite3.Connection, page_id: int) -> list[dict]:
    return [
        line_dict(r) for r in conn.execute('SELECT * FROM lines WHERE page_id = ? ORDER BY "index", id', (page_id,))
    ]


def _page_size(conn: sqlite3.Connection, page_id: int) -> tuple[int, int]:
    r = conn.execute("SELECT width, height FROM pages WHERE id = ?", (page_id,)).fetchone()
    if r is None:
        raise NotFound(f"page {page_id} not found")
    return r["width"], r["height"]


def check_points(points: Any, name: str, minimum: int, size: tuple[int, int]) -> list[tuple[int, int]]:
    """Validate a point list against the page (working-copy pixels) and round it to integers."""
    if not isinstance(points, list | tuple) or len(points) < minimum:
        raise Invalid(f"{name} needs at least {minimum} points")
    w, h = size
    out = []
    for p in points:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError) as e:
            raise Invalid(f"{name}: each point must be [x, y]") from e
        if not (0 <= x <= w and 0 <= y <= h):  # also rejects NaN
            raise Invalid(f"{name}: point ({x:g}, {y:g}) is outside the page ({w}x{h})")
        out.append((round(x), round(y)))
    return out


def generate_polygon(
    conn: sqlite3.Connection, page_id: int, baseline: list[tuple[int, int]], exclude_line_id: int | None = None
) -> list[tuple[int, int]]:
    """Polygon for a baseline using the page's median ascent/descent (fallback: the page's measured line height)."""
    page = conn.execute("SELECT width, height, line_height_px FROM pages WHERE id = ?", (page_id,)).fetchone()
    if page is None:
        raise NotFound(f"page {page_id} not found")
    size = (page["width"], page["height"])
    others = [
        (json_loads(r["baseline"]), json_loads(r["polygon"]))
        for r in conn.execute("SELECT id, baseline, polygon FROM lines WHERE page_id = ?", (page_id,))
        if r["id"] != exclude_line_id
    ]
    asc, desc = geometry.typical_extent(
        [([tuple(p) for p in b], [tuple(p) for p in pg]) for b, pg in others], size, page["line_height_px"]
    )
    return geometry.polygon_from_baseline(baseline, asc, desc, size)


def _renumber(conn: sqlite3.Connection, ordered_ids: list[int]) -> None:
    for i, lid in enumerate(ordered_ids):
        conn.execute('UPDATE lines SET "index" = ? WHERE id = ? AND "index" != ?', (i, lid, i))


def _ordered_ids(conn: sqlite3.Connection, page_id: int) -> list[int]:
    return [r[0] for r in conn.execute('SELECT id FROM lines WHERE page_id = ? ORDER BY "index", id', (page_id,))]


def update_line(conn: sqlite3.Connection, settings: Settings, line_id: int, patch: dict[str, Any]) -> dict:
    """Apply a partial update. Keys: text, polygon, baseline, ignored, is_ground_truth, index.

    * Setting text always sets text_source='manual' and clears the prediction confidence; empty text clears
      text_source.
    * is_ground_truth=true requires non-empty text (after the patch).
    * `index` moves the line to that reading-order position; the other lines shift, all in one transaction.
    """
    unknown = set(patch) - PATCHABLE
    if unknown:
        raise Invalid(f"cannot patch: {', '.join(sorted(unknown))}")
    cur = get_line_row(conn, line_id)
    page_id = cur["page_id"]
    size = _page_size(conn, page_id)
    sets: dict[str, Any] = {}
    if "text" in patch:
        text = patch["text"]
        sets.update(text=text, text_source="manual" if text else None, confidence=None)
    regenerate = bool(patch.get("regenerate_polygon"))
    if regenerate and "polygon" in patch:
        raise Invalid("give either polygon or regenerate_polygon, not both")
    if "polygon" in patch:
        sets["polygon"] = json_dumps_points(check_points(patch["polygon"], "polygon", 3, size))
    if "baseline" in patch:
        sets["baseline"] = json_dumps_points(check_points(patch["baseline"], "baseline", 2, size))
    if regenerate:
        base = (
            geometry.orient_left_to_right(check_points(patch["baseline"], "baseline", 2, size))
            if "baseline" in patch
            else [tuple(p) for p in json_loads(cur["baseline"])]
        )
        sets["baseline"] = json_dumps_points(base)
        sets["polygon"] = json_dumps_points(generate_polygon(conn, page_id, base, exclude_line_id=line_id))
    if "ignored" in patch:
        sets["ignored"] = int(bool(patch["ignored"]))
    if "is_ground_truth" in patch:
        sets["is_ground_truth"] = int(bool(patch["is_ground_truth"]))
    final_text = sets.get("text", cur["text"])
    final_gt = sets.get("is_ground_truth", cur["is_ground_truth"])
    if final_gt and not final_text:
        raise Invalid("a ground-truth line needs non-empty text")
    with sync.page_transaction(conn, settings, page_id):
        # the line may have been deleted since the read above
        if conn.execute("SELECT 1 FROM lines WHERE id = ?", (line_id,)).fetchone() is None:
            raise NotFound(f"line {line_id} not found")
        review.record_if_reviewed(
            conn, get_line_row(conn, line_id), sets.get("text"), sets.get("is_ground_truth"), sets.get("ignored")
        )
        if sets:
            cols = ", ".join(f"{k} = ?" for k in sets)
            conn.execute(f"UPDATE lines SET {cols} WHERE id = ?", (*sets.values(), line_id))
        if "index" in patch:
            ids = _ordered_ids(conn, page_id)
            new = patch["index"]
            if not isinstance(new, int) or isinstance(new, bool) or not 0 <= new < len(ids):
                raise Invalid(f"index must be an integer between 0 and {len(ids) - 1}")
            ids.remove(line_id)
            ids.insert(new, line_id)
            _renumber(conn, ids)
    return get_line(conn, line_id)


def delete_line(conn: sqlite3.Connection, settings: Settings, line_id: int) -> None:
    page_id = get_line_row(conn, line_id)["page_id"]
    with sync.page_transaction(conn, settings, page_id):
        conn.execute("DELETE FROM lines WHERE id = ?", (line_id,))
        _renumber(conn, _ordered_ids(conn, page_id))


def add_line(conn: sqlite3.Connection, settings: Settings, page_id: int, data: dict[str, Any]) -> dict:
    """Add a line to a segmented page. Keys: baseline (required), polygon (generated from the baseline when absent),
    text, ignored, index (default: append)."""
    page = conn.execute("SELECT segmented_at FROM pages WHERE id = ?", (page_id,)).fetchone()
    if page is None:
        raise NotFound(f"page {page_id} not found")
    if page["segmented_at"] is None:
        raise Conflict("page has not been segmented yet")
    size = _page_size(conn, page_id)
    baseline = check_points(data.get("baseline"), "baseline", 2, size)
    polygon = None
    if data.get("polygon") is not None:
        polygon = check_points(data["polygon"], "polygon", 3, size)
    else:
        baseline = geometry.orient_left_to_right(baseline)
        polygon = generate_polygon(conn, page_id, baseline)
    text = data.get("text") or ""
    with sync.page_transaction(conn, settings, page_id):
        ids = _ordered_ids(conn, page_id)
        index = data.get("index")
        if index is None:
            index = len(ids)
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index <= len(ids):
            raise Invalid(f"index must be an integer between 0 and {len(ids)}")
        ln = Line(
            id="",
            baseline=baseline,
            polygon=polygon,
            index=len(ids),
            text=text,
            text_source="manual" if text else None,
            ignored=bool(data.get("ignored", False)),
        )
        new_id = sync.insert_line(conn, page_id, ln)
        ids.insert(index, new_id)
        _renumber(conn, ids)
    return get_line(conn, new_id)
