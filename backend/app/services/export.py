"""Plain-text and PAGE-XML export of the transcription (not the project bundle, see bundle.py)."""

from __future__ import annotations

import io
import re
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import BinaryIO

from .. import abbreviations as engine
from ..errors import Invalid, NotFound
from ..pipeline import pagexml
from . import abbreviations as abbr_svc
from . import sync


@dataclass(frozen=True)
class PageRef:
    id: int
    document_id: int
    document_name: str
    number: int  # 1-based position of the page within its document (by import order)
    original_filename: str
    xml_path: str
    segmented: bool


def select_pages(conn: sqlite3.Connection, document_id: int | None, page_ids: list[int] | None) -> list[PageRef]:
    """The pages to export, in document / import order. Unknown ids are 404, not silently skipped."""
    if (
        document_id is not None
        and conn.execute("SELECT 1 FROM documents WHERE id = ?", (document_id,)).fetchone() is None
    ):
        raise NotFound(f"document {document_id} not found")
    rows = conn.execute(
        "SELECT p.id, p.document_id, d.name, p.original_filename, p.xml_path, p.segmented_at, "
        "ROW_NUMBER() OVER (PARTITION BY p.document_id ORDER BY p.id) AS n "
        "FROM pages p JOIN documents d ON d.id = p.document_id ORDER BY p.document_id, p.id"
    ).fetchall()
    chosen = []
    for r in rows:
        if document_id is not None and r["document_id"] != document_id:
            continue
        if page_ids is not None and r["id"] not in page_ids:
            continue
        chosen.append(
            PageRef(
                r["id"],
                r["document_id"],
                r["name"],
                r["n"],
                r["original_filename"],
                r["xml_path"],
                bool(r["segmented_at"]),
            )
        )
    if page_ids is not None:
        missing = sorted(set(page_ids) - {p.id for p in chosen})
        if missing:
            raise NotFound(f"page {missing[0]} not found" + (" in that document" if document_id is not None else ""))
    return chosen


def parse_ids(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    try:
        return [int(x) for x in value.split(",") if x.strip()]
    except ValueError:
        raise Invalid("page_ids must be a comma-separated list of page ids") from None


def page_header(p: PageRef) -> str:
    return f"=== {p.document_name} / page {p.number} ({p.original_filename}) ==="


def export_text(
    conn: sqlite3.Connection,
    pages: list[PageRef],
    expanded: bool,
    include_predicted: bool,
    ambiguous: engine.Ambiguous,
) -> str:
    """UTF-8 text: per page a header line, then its lines in reading order; pages are separated by a blank line.
    Ignored and empty lines are left out; predicted lines only when `include_predicted`, then each starts with
    `[predicted] `. Pages without any text are left out. Ambiguous abbreviations stay as written (`raw`) or get a
    `[?]` marker (`mark`); both only matter when `expanded`."""
    d = abbr_svc.dictionary(conn) if expanded else None
    blocks = []
    for p in pages:
        rows = abbr_svc.page_text_lines(conn, p.id, include_predicted)
        if not rows:
            continue
        items = abbr_svc.render_lines(rows, d, ambiguous)
        blocks.append(page_header(p) + "\n" + "\n".join(abbr_svc.line_output(x, expanded) for x in items))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _xml_name(p: PageRef) -> str:
    stem = PurePosixPath(p.xml_path).name
    return stem if stem.endswith(".xml") else stem + ".xml"


def pagexml_bytes(conn: sqlite3.Connection, page_id: int, include_predicted: bool) -> bytes:
    """The page's PAGE-XML rendered from the database (always current), exactly as the sync layer writes it. Without
    `include_predicted` the text and confidence of predicted lines are left out (their geometry stays)."""
    seg, image_filename = sync.page_segmentation(conn, page_id)
    if not include_predicted:
        for ln in seg.lines:
            if ln.text_source == "predicted":
                ln.text, ln.text_source, ln.confidence = "", None, None
    buf = io.BytesIO()
    pagexml.build_pagexml(seg, image_filename).write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue()


def export_pagexml(conn: sqlite3.Connection, pages: list[PageRef], include_predicted: bool) -> BinaryIO:
    """A zip with one PAGE-XML file per segmented page (named like the project's own `pagexml/` files), rewound and
    ready to read. Spooled to disk when large."""
    out = tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024)  # noqa: SIM115  closed by the response
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in pages:
            if p.segmented:
                zf.writestr(_xml_name(p), pagexml_bytes(conn, p.id, include_predicted))
    out.seek(0)
    return out


def slug(name: str, fallback: str = "project") -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")
    return s[:60] or fallback
