"""Import of a pre-cropped word image as a one-line page (used only by `scripts/rxhand_import.py`, not by the API).

A page here is one image that already is a single text line (a handwritten word). The normal import would measure a
"text line height", rescale to a target height, deskew and run the segmenter; all of that is meaningless for a
128 px crop, so this path skips it:

* the original file is stored byte-for-byte (`data/originals`), as in the normal import;
* the working copy is `pipeline.wordcrop.apply_word_canvas(original)` (grayscale + horizontal stretch, see there);
* the page gets exactly one line whose polygon is the whole working image and whose baseline runs across it at
  three quarters of the height, with the given text as manual ground truth;
* the page counts as segmented, and its PAGE-XML is written through the normal transactional sync.

The page's `pages.preprocess` records `{"mode": "word_canvas", "key": ..., "stretch_x": ...}`; the train and predict
jobs read it (see `worker/ml_jobs.py`). Everything downstream (line splits, training export, prediction, review) is the
unchanged page pipeline.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..config import Settings
from ..db import now_iso, write_tx
from ..errors import Invalid
from ..pipeline import workflow
from ..pipeline.segment import Line
from ..pipeline.wordcrop import WordCanvas, write_word_working
from . import pages as pages_svc
from . import sync


def import_word_image(
    conn: sqlite3.Connection,
    settings: Settings,
    document_id: int,
    src: Path,
    text: str,
    canvas: WordCanvas,
    ground_truth: bool = True,
) -> tuple[dict, bool]:
    """Import one word image with its transcription. Returns (page, created); created=False when a page with
    identical bytes exists (the existing page is returned untouched)."""
    if not text.strip():
        raise Invalid("a word image needs a non-blank transcription")
    pages_svc.get_document(conn, document_id)
    ws = settings.workspace
    digest = workflow.sha256_file(src)
    existing = conn.execute("SELECT id FROM pages WHERE sha256 = ?", (digest,)).fetchone()
    if existing:
        return pages_svc.get_page(conn, existing["id"]), False
    rec = workflow.register_original(src, ws, digest, source_name=src.name)  # ValueError if unreadable
    created = [ws.root / rec["original"]]
    try:
        work = ws.working / f"{rec['id']}.png"
        w, h = write_word_working(ws.root / rec["original"], work, canvas)
        created.append(work)
        rec["working"] = str(work.relative_to(ws.root))
        preprocess = {**canvas.to_dict(), "width": w, "height": h, "source": "presegmented word image"}
        with write_tx(conn):
            cur = conn.execute(
                "INSERT INTO pages (document_id, image_path, xml_path, width, height, working_path, "
                "original_filename, sha256, original_width, original_height, dpi_x, dpi_y, line_height_px, "
                "scale, skew_deg, deskewed, warnings, preprocess, imported_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 1.0, 0.0, 0, '[]', ?, ?)",
                (
                    document_id,
                    rec["original"],
                    f"pagexml/{rec['id']}.xml",
                    w,
                    h,
                    rec["working"],
                    src.name,
                    digest,
                    rec["width"],
                    rec["height"],
                    rec["dpi"][0],
                    rec["dpi"][1],
                    json.dumps(preprocess),
                    now_iso(),
                ),
            )
        page_id = int(cur.lastrowid)
        (settings.data_dir / "pagexml").mkdir(parents=True, exist_ok=True)
        base_y = round(0.75 * (h - 1))
        line = Line(
            id="l0",
            baseline=[(0, base_y), (w - 1, base_y)],
            polygon=[(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)],
            index=0,
            text=text,
            text_source="manual",
            is_ground_truth=ground_truth,
        )
        try:
            with sync.page_transaction(conn, settings, page_id):
                sync.insert_line(conn, page_id, line)
                conn.execute("UPDATE pages SET segmented_at = ? WHERE id = ?", (now_iso(), page_id))
        except BaseException:
            with write_tx(conn):
                conn.execute("DELETE FROM pages WHERE id = ?", (page_id,))
            (settings.data_dir / f"pagexml/{rec['id']}.xml").unlink(missing_ok=True)
            raise
        created.clear()
        return pages_svc.get_page(conn, page_id), True
    finally:
        for f in created:
            f.unlink(missing_ok=True)
