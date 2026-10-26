"""Documents and pages: upload/import (reusing the M1 pipeline), listing, status."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import BinaryIO

from PIL import Image, UnidentifiedImageError

from ..config import Settings
from ..db import now_iso, write_tx
from ..errors import Invalid, NotFound, TooLarge, UnsupportedMedia
from ..pipeline import workflow
from .rows import page_dict

# extension -> PIL format names accepted for it. The content is checked, not just the name.
ALLOWED = {".png": {"PNG"}, ".jpg": {"JPEG"}, ".jpeg": {"JPEG"}, ".tif": {"TIFF"}, ".tiff": {"TIFF"}}

PAGE_SELECT = (
    "SELECT p.*, COUNT(l.id) AS line_count, COALESCE(SUM(l.is_ground_truth), 0) AS ground_truth_count, "
    "COALESCE(SUM(l.ignored), 0) AS ignored_count FROM pages p LEFT JOIN lines l ON l.page_id = p.id"
)


# ---- documents -------------------------------------------------------------------------------------------------


def create_document(conn: sqlite3.Connection, name: str) -> dict:
    with write_tx(conn):
        cur = conn.execute("INSERT INTO documents (name, imported_at) VALUES (?, ?)", (name, now_iso()))
    return get_document(conn, cur.lastrowid)


def get_document(conn: sqlite3.Connection, document_id: int) -> dict:
    r = conn.execute(
        "SELECT d.*, (SELECT COUNT(*) FROM pages WHERE document_id = d.id) AS page_count FROM documents d "
        "WHERE d.id = ?",
        (document_id,),
    ).fetchone()
    if r is None:
        raise NotFound(f"document {document_id} not found")
    return dict(r)


def list_documents(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT d.*, (SELECT COUNT(*) FROM pages WHERE document_id = d.id) AS page_count FROM documents d ORDER BY d.id"
    )
    return [dict(r) for r in rows]


# ---- pages -----------------------------------------------------------------------------------------------------


def get_page_row(conn: sqlite3.Connection, page_id: int) -> sqlite3.Row:
    r = conn.execute(f"{PAGE_SELECT} WHERE p.id = ? GROUP BY p.id", (page_id,)).fetchone()
    if r is None:
        raise NotFound(f"page {page_id} not found")
    return r


def get_page(conn: sqlite3.Connection, page_id: int) -> dict:
    return page_dict(get_page_row(conn, page_id))


def list_pages(conn: sqlite3.Connection, document_id: int | None = None, status: str | None = None) -> list[dict]:
    where, args = [], []
    if document_id is not None:
        where.append("p.document_id = ?")
        args.append(document_id)
    if status is not None:
        where.append("p.status = ?")
        args.append(status)
    sql = PAGE_SELECT + (" WHERE " + " AND ".join(where) if where else "") + " GROUP BY p.id ORDER BY p.id"
    return [page_dict(r) for r in conn.execute(sql, args)]


def compute_status(segmented: bool, lines: list[sqlite3.Row]) -> str:
    """unsegmented | segmented | partially_transcribed | done.

    Ignored lines are out of scope. A line counts as worked on when it is ground truth or has manual text
    (predictions alone do not count). The page is done when it has at least one active line and all are ground truth.
    """
    if not segmented:
        return "unsegmented"
    active = [r for r in lines if not r["ignored"]]
    if active and all(r["is_ground_truth"] for r in active):
        return "done"
    if any(r["is_ground_truth"] or r["text_source"] == "manual" for r in active):
        return "partially_transcribed"
    return "segmented"


def refresh_status(conn: sqlite3.Connection, page_id: int) -> str:
    """Recompute and store pages.status (call inside the transaction that changed the lines)."""
    seg = conn.execute("SELECT segmented_at FROM pages WHERE id = ?", (page_id,)).fetchone()
    if seg is None:
        raise NotFound(f"page {page_id} not found")
    lines = conn.execute("SELECT ignored, is_ground_truth, text_source FROM lines WHERE page_id = ?", (page_id,))
    status = compute_status(seg["segmented_at"] is not None, lines.fetchall())
    conn.execute("UPDATE pages SET status = ? WHERE id = ?", (status, page_id))
    return status


def _clean_filename(filename: str | None) -> str:
    name = Path((filename or "").replace("\\", "/")).name.strip()
    if not name or name in (".", ".."):
        raise Invalid("upload has no file name")
    return name


def _receive(stream: BinaryIO, dest: Path, limit: int) -> str:
    """Copy the upload to `dest`, hashing as we go; refuse anything over `limit` bytes. Returns the SHA-256."""
    h = hashlib.sha256()
    size = 0
    with open(dest, "wb") as out:
        while chunk := stream.read(1 << 20):
            size += len(chunk)
            if size > limit:
                raise TooLarge(f"file exceeds the {limit // (1024 * 1024)} MB upload limit")
            h.update(chunk)
            out.write(chunk)
    if size == 0:
        raise Invalid("uploaded file is empty")
    return h.hexdigest()


def _check_image(path: Path, suffix: str) -> None:
    try:
        with Image.open(path) as im:
            fmt = im.format
            im.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as e:
        raise UnsupportedMedia(f"not a readable image: {e}") from e
    if fmt not in ALLOWED[suffix]:
        raise UnsupportedMedia(f"file extension {suffix} does not match the image content ({fmt})")


def import_upload(
    conn: sqlite3.Connection, settings: Settings, stream: BinaryIO, filename: str | None, document_id: int
) -> tuple[dict, bool]:
    """Import one uploaded scan. Returns (page, created); created=False when identical content already exists.

    Reuses the M1 pipeline (workflow.register_original / derive_working): the original is stored byte-for-byte
    under data/originals, the working copy (grayscale, contrast-normalized, deskewed, possibly rescaled) under
    data/working, and the import warnings and measurements are recorded on the page.
    """
    get_document(conn, document_id)  # 404 early
    name = _clean_filename(filename)
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED:
        raise UnsupportedMedia(f"unsupported file type {suffix or '(none)'}; use PNG, JPEG or TIFF")
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = settings.tmp_dir / f"upload_{now_iso().replace(':', '')}_{id(stream):x}{suffix}"
    ws = settings.workspace
    created_files: list[Path] = []
    try:
        digest = _receive(stream, tmp, settings.max_upload_bytes)
        existing = conn.execute("SELECT id FROM pages WHERE sha256 = ?", (digest,)).fetchone()
        if existing:
            return get_page(conn, existing["id"]), False
        _check_image(tmp, suffix)
        try:
            rec = workflow.register_original(tmp, ws, digest, source_name=name)
        except ValueError as e:
            raise UnsupportedMedia(f"not a readable image: {e}") from e
        created_files.append(ws.root / rec["original"])
        info = workflow.derive_working(ws, rec, settings.target_line_height, True)
        created_files.append(ws.root / rec["working"])
        try:
            with write_tx(conn):
                cur = conn.execute(
                    "INSERT INTO pages (document_id, image_path, xml_path, width, height, working_path, "
                    "original_filename, sha256, original_width, original_height, dpi_x, dpi_y, line_height_px, "
                    "scale, skew_deg, deskewed, warnings, preprocess, imported_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        document_id,
                        rec["original"],
                        f"pagexml/{rec['id']}.xml",
                        info.width,
                        info.height,
                        rec["working"],
                        name,
                        digest,
                        rec["width"],
                        rec["height"],
                        rec["dpi"][0],
                        rec["dpi"][1],
                        rec["line_height_px"],
                        info.scale,
                        info.skew_deg,
                        int(info.deskewed),
                        json.dumps(rec["warnings"]),
                        json.dumps(rec["preprocess"]),
                        now_iso(),
                    ),
                )
        except sqlite3.IntegrityError:
            # Lost a race with an identical concurrent upload: the files are the winner's, keep them.
            existing = conn.execute("SELECT id FROM pages WHERE sha256 = ?", (digest,)).fetchone()
            if existing is None:
                raise
            created_files.clear()
            return get_page(conn, existing["id"]), False
        created_files.clear()
        return get_page(conn, cur.lastrowid), True
    finally:
        tmp.unlink(missing_ok=True)
        for f in created_files:  # import failed half way: leave nothing behind
            f.unlink(missing_ok=True)
