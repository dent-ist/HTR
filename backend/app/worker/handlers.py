"""Job handlers. A handler takes a JobContext and returns a JSON-able result dict; raising fails the job."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from PIL import Image

from ..config import Settings
from ..errors import Conflict, NotFound
from ..pipeline import segment
from ..pipeline.segment import Line
from ..services import sync
from . import queue


@dataclass
class JobContext:
    job: dict
    conn: sqlite3.Connection
    settings: Settings
    notices: list[str] = field(default_factory=list)  # visible, non-fatal remarks (e.g. "running on the CPU")

    @property
    def payload(self) -> dict:
        return self.job["payload"]

    def progress(self, fraction: float) -> None:
        queue.set_progress(self.conn, self.job["id"], fraction)

    def info(self, info: dict) -> None:
        """Publish live status (shown as the running job's `result`)."""
        queue.set_info(self.conn, self.job["id"], info)

    def cancel_requested(self) -> bool:
        return queue.cancel_requested(self.conn, self.job["id"])

    def check_cancelled(self, result: dict | None = None) -> None:
        """Cooperative cancellation point: ends the job as cancelled if the user asked for it."""
        if self.cancel_requested():
            raise queue.JobCancelled(result)


Handler = Callable[[JobContext], dict | None]


def _clamp_points(points, w: int, h: int):
    return [(min(max(x, 0), w), min(max(y, 0), h)) for x, y in points]


def segment_page(ctx: JobContext) -> dict:
    """Run Kraken baseline segmentation on a page's working copy and replace its lines (and PAGE-XML).

    payload: {"page_id": int, "force": bool}. Without force, a page that already has manually transcribed or
    ground-truth lines is refused, so a re-segmentation can never silently destroy typed work.
    """
    page_id = ctx.payload["page_id"]
    force = bool(ctx.payload.get("force"))
    conn = ctx.conn
    page = conn.execute("SELECT * FROM pages WHERE id = ?", (page_id,)).fetchone()
    if page is None:
        raise NotFound(f"page {page_id} not found")
    guard_resegment(conn, page_id, force)
    ctx.progress(0.05)
    t0 = time.time()
    with Image.open(ctx.settings.data_dir / page["working_path"]) as im:
        im.load()
        ctx.progress(0.1)
        seg = segment.segment_image(im, ctx.settings.device)
    ctx.progress(0.8)
    if seg.device == "cpu" and ctx.settings.device != "cpu":
        ctx.notices.append(
            "No usable GPU (CUDA) was found, so segmentation ran on the CPU (slower). "
            "Run `make check` to see why the GPU is not visible."
        )
    w, h = page["width"], page["height"]
    lines = [
        Line(id="", baseline=_clamp_points(ln.baseline, w, h), polygon=_clamp_points(ln.polygon, w, h))
        for ln in seg.lines
    ]
    with sync.page_transaction(conn, ctx.settings, page_id):
        guard_resegment(conn, page_id, force)  # re-check under the write lock
        sync.replace_lines(conn, page_id, lines)
        conn.execute("UPDATE pages SET segmented_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?", (page_id,))
    return {"page_id": page_id, "lines": len(lines), "device": seg.device, "seconds": round(time.time() - t0, 2)}


def guard_resegment(conn: sqlite3.Connection, page_id: int, force: bool) -> None:
    if force:
        return
    n = conn.execute(
        "SELECT COUNT(*) FROM lines WHERE page_id = ? AND (is_ground_truth = 1 OR text_source = 'manual')", (page_id,)
    ).fetchone()[0]
    if n:
        raise Conflict(f"page {page_id} has {n} transcribed line(s); re-segmenting would discard them (use force)")


def train_job(ctx: JobContext) -> dict | None:
    from .ml_jobs import train_model  # imported lazily: pulls in the recognition stack

    return train_model(ctx)


def predict_job(ctx: JobContext) -> dict | None:
    from .ml_jobs import predict_lines

    return predict_lines(ctx)


HANDLERS: dict[str, Handler] = {
    "segment": segment_page,
    "predict": predict_job,
    "train": train_job,
}
