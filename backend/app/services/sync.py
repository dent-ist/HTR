"""Keeping the page's PAGE-XML in sync with the database.

Every operation that changes a page's lines runs inside `page_transaction`:

    BEGIN IMMEDIATE -> mutate rows -> refresh page status -> render XML to <page>.xml.tmp and rename it over
    <page>.xml (atomic) -> COMMIT

The rename happens just before COMMIT, while SQLite's write lock is held, so nobody else can write the same page
in between. If the mutation, the XML write or the rename fails, the transaction is rolled back and the old XML
file is untouched (only a stray .tmp may remain, which is removed). The one window left is COMMIT itself failing
after the rename (disk full, I/O error): the transaction is rolled back and the XML is regenerated from the
now-unchanged database, best effort.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..config import Settings
from ..errors import Invalid, NotFound
from ..pipeline import pagexml
from ..pipeline.segment import Line, PageSegmentation
from . import pages as pages_svc
from .rows import json_dumps_points, json_loads

log = logging.getLogger(__name__)


def page_segmentation(conn: sqlite3.Connection, page_id: int) -> tuple[PageSegmentation, str]:
    """The page as the PAGE-XML writer wants it, plus the working-copy file name (PAGE imageFilename)."""
    page = conn.execute("SELECT width, height, working_path FROM pages WHERE id = ?", (page_id,)).fetchone()
    if page is None:
        raise NotFound(f"page {page_id} not found")
    seg = PageSegmentation(width=page["width"], height=page["height"])
    for r in conn.execute('SELECT * FROM lines WHERE page_id = ? ORDER BY "index", id', (page_id,)):
        seg.lines.append(
            Line(
                id=f"l{r['id']}",
                baseline=[tuple(p) for p in json_loads(r["baseline"])],
                polygon=[tuple(p) for p in json_loads(r["polygon"])],
                index=r["index"],
                text=r["text"],
                text_source=r["text_source"],
                confidence=r["confidence"],
                is_ground_truth=bool(r["is_ground_truth"]),
                ignored=bool(r["ignored"]),
            )
        )
    return seg, Path(page["working_path"]).name


def xml_file(conn: sqlite3.Connection, settings: Settings, page_id: int) -> Path:
    row = conn.execute("SELECT xml_path FROM pages WHERE id = ?", (page_id,)).fetchone()
    if row is None:
        raise NotFound(f"page {page_id} not found")
    return settings.data_dir / row["xml_path"]


def write_page_xml(conn: sqlite3.Connection, settings: Settings, page_id: int) -> Path:
    """Render the page's lines from the database into its PAGE-XML (atomic: temp file, then rename)."""
    seg, image_filename = page_segmentation(conn, page_id)
    path = xml_file(conn, settings, page_id)
    try:
        pagexml.write_pagexml(seg, image_filename, path)
    except BaseException:
        path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
        raise
    return path


@contextmanager
def page_transaction(conn: sqlite3.Connection, settings: Settings, page_id: int) -> Iterator[sqlite3.Connection]:
    """One logical operation: DB change + status + XML rewrite, all or nothing (see module docstring)."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        pages_svc.refresh_status(conn, page_id)
        write_page_xml(conn, settings, page_id)
    except BaseException:
        conn.rollback()
        raise
    try:
        conn.commit()
    except BaseException:
        conn.rollback()
        try:
            write_page_xml(conn, settings, page_id)
        except Exception:  # noqa: BLE001
            log.exception("could not restore PAGE-XML for page %s after a failed commit", page_id)
        raise


def insert_line(conn: sqlite3.Connection, page_id: int, ln: Line) -> int:
    cur = conn.execute(
        'INSERT INTO lines (page_id, "index", baseline, polygon, text, text_source, confidence, is_ground_truth, '
        "ignored) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            page_id,
            ln.index,
            json_dumps_points(ln.baseline),
            json_dumps_points(ln.polygon),
            ln.text,
            ln.text_source if ln.text else None,
            ln.confidence,
            int(ln.is_ground_truth),
            int(ln.ignored),
        ),
    )
    return int(cur.lastrowid)


def replace_lines(conn: sqlite3.Connection, page_id: int, lines: list[Line]) -> None:
    """Delete the page's lines and insert `lines` (their `index` is renumbered 0..n-1 in the order given)."""
    conn.execute("DELETE FROM lines WHERE page_id = ?", (page_id,))
    for i, ln in enumerate(lines):
        ln.index = i
        insert_line(conn, page_id, ln)


def rebuild_lines_from_xml(
    conn: sqlite3.Connection, settings: Settings, page_id: int, xml_path: Path | None = None
) -> int:
    """Import path: replace the page's DB lines with those in a PAGE-XML file (default: the page's own XML).

    The file's coordinates must be in this page's working-copy pixels (imageWidth/imageHeight must match).
    Restores text, text_source, confidence, ground-truth and ignored flags (see pipeline/pagexml.py). The page's
    canonical XML is then rewritten from the database. Line ids are new. Returns the number of lines.
    """
    page = conn.execute("SELECT width, height FROM pages WHERE id = ?", (page_id,)).fetchone()
    if page is None:
        raise NotFound(f"page {page_id} not found")
    path = Path(xml_path) if xml_path else xml_file(conn, settings, page_id)
    try:
        seg, _ = pagexml.read_pagexml(path)
    except (OSError, ValueError, pagexml.ET.ParseError) as e:
        raise Invalid(f"cannot read PAGE-XML {path.name}: {e}") from e
    if (seg.width, seg.height) != (page["width"], page["height"]):
        raise Invalid(
            f"PAGE-XML is {seg.width}x{seg.height} but the page working copy is {page['width']}x{page['height']}"
        )
    with page_transaction(conn, settings, page_id):
        replace_lines(conn, page_id, seg.lines)
        conn.execute(
            "UPDATE pages SET segmented_at = COALESCE(segmented_at, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')) "
            "WHERE id = ?",
            (page_id,),
        )
    return len(seg.lines)
