"""HTTP routes. Thin: validation and status codes here, behaviour in app.services and app.worker."""

from __future__ import annotations

import io
import json
import sqlite3
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from ..config import Settings
from ..errors import Conflict, NotFound
from ..services import lines as lines_svc
from ..services import models as models_svc
from ..services import pages as pages_svc
from ..services import repair as repair_svc
from ..services import training as training_svc
from ..worker import queue
from .deps import get_conn, get_settings
from .schemas import (
    DocumentCreate,
    IgnoreRegionRequest,
    LineCreate,
    LinePatch,
    MergeRequest,
    PredictRequest,
    ReorderRequest,
    RestoreRequest,
    SegmentRequest,
    SplitRequest,
    TrainRequest,
)

router = APIRouter()
Conn = Annotated[sqlite3.Connection, Depends(get_conn)]
Cfg = Annotated[Settings, Depends(get_settings)]

PageStatus = Literal["unsegmented", "segmented", "partially_transcribed", "done"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


# ---- documents -------------------------------------------------------------------------------------------------


@router.post("/documents", status_code=201, tags=["documents"])
def create_document(body: DocumentCreate, conn: Conn):
    return pages_svc.create_document(conn, body.name)


@router.get("/documents", tags=["documents"])
def list_documents(conn: Conn):
    return pages_svc.list_documents(conn)


# ---- pages -----------------------------------------------------------------------------------------------------


@router.post("/pages", tags=["pages"])
def upload_page(
    request: Request,
    response: Response,
    conn: Conn,
    settings: Cfg,
    file: Annotated[UploadFile, File(description="PNG, JPEG or TIFF scan")],
    document_id: Annotated[int, Form()],
):
    """Import one scan into a document. 201 for a new page; 200 with the existing page if identical content
    was imported before."""
    page, created = pages_svc.import_upload(conn, settings, file.file, file.filename, document_id)
    response.status_code = 201 if created else 200
    return {**page, "duplicate": not created}


@router.get("/pages", tags=["pages"])
def list_pages(conn: Conn, document_id: int | None = None, status: PageStatus | None = None):
    return pages_svc.list_pages(conn, document_id, status)


@router.get("/pages/{page_id}", tags=["pages"])
def get_page(page_id: int, conn: Conn):
    return {**pages_svc.get_page(conn, page_id), "lines": lines_svc.list_lines(conn, page_id)}


@router.get("/pages/{page_id}/original", tags=["pages"])
def page_original(page_id: int, conn: Conn, settings: Cfg):
    row = pages_svc.get_page_row(conn, page_id)
    return _file(settings, row["image_path"])


@router.get("/pages/{page_id}/working", tags=["pages"])
def page_working(page_id: int, conn: Conn, settings: Cfg):
    """The working copy: the image all line coordinates refer to."""
    row = pages_svc.get_page_row(conn, page_id)
    return _file(settings, row["working_path"], "image/png")


def _file(settings: Settings, rel: str, media_type: str | None = None) -> FileResponse:
    path = settings.data_dir / rel
    if not path.is_file():
        raise NotFound(f"image file missing on disk: {rel}")
    return FileResponse(path, media_type=media_type)


@router.post("/pages/{page_id}/segment", status_code=202, tags=["pages"])
def segment_page(page_id: int, conn: Conn, request: Request, body: SegmentRequest | None = None):
    """Enqueue a segmentation job. Re-segmenting replaces the page's lines; refused (409) when the page has
    transcribed lines unless force=true."""
    from ..worker.handlers import guard_resegment

    force = body.force if body else False
    pages_svc.get_page_row(conn, page_id)
    active = queue.find_active(conn, "segment", "page_id", page_id)
    if active:
        raise Conflict(f"page {page_id} already has a segmentation job ({active['status']}): job {active['id']}")
    guard_resegment(conn, page_id, force)
    job = queue.enqueue(conn, "segment", {"page_id": page_id, "force": force})
    request.app.state.worker.wake()
    return job


@router.post("/pages/{page_id}/lines", status_code=201, tags=["lines"])
def add_line(page_id: int, body: LineCreate, conn: Conn, settings: Cfg):
    return lines_svc.add_line(conn, settings, page_id, body.model_dump())


@router.post("/pages/{page_id}/ignore-region", tags=["repair"])
def ignore_region(page_id: int, body: IgnoreRegionRequest, conn: Conn, settings: Cfg):
    """Ignore every active line with at least half of its polygon inside the rectangle or polygon."""
    return repair_svc.ignore_region(conn, settings, page_id, body.model_dump())


@router.post("/pages/{page_id}/reorder", tags=["repair"])
def reorder_lines(page_id: int, body: ReorderRequest, conn: Conn, settings: Cfg):
    """Set the full reading order in one transaction."""
    return repair_svc.reorder(conn, settings, page_id, body.line_ids)


@router.post("/pages/{page_id}/restore", tags=["repair"])
def restore_lines(page_id: int, body: RestoreRequest, conn: Conn, settings: Cfg):
    """Replace the page's lines with a snapshot (undo); 409 if that would discard typed text and force is false."""
    return repair_svc.restore(conn, settings, page_id, [ln.model_dump() for ln in body.lines], body.force)


# ---- lines -----------------------------------------------------------------------------------------------------


@router.post("/lines/merge", tags=["repair"])
def merge_lines(body: MergeRequest, conn: Conn, settings: Cfg):
    return repair_svc.merge_lines(conn, settings, body.line_ids)


@router.post("/lines/{line_id}/split", tags=["repair"])
def split_line(line_id: int, body: SplitRequest, conn: Conn, settings: Cfg):
    return repair_svc.split_line(conn, settings, line_id, body.x, body.y)


@router.patch("/lines/{line_id}", tags=["lines"])
def patch_line(line_id: int, body: LinePatch, conn: Conn, settings: Cfg):
    return lines_svc.update_line(conn, settings, line_id, body.changes())


@router.delete("/lines/{line_id}", status_code=204, tags=["lines"])
def delete_line(line_id: int, conn: Conn, settings: Cfg):
    lines_svc.delete_line(conn, settings, line_id)
    return Response(status_code=204)


@router.get("/lines/{line_id}/crop.png", tags=["lines"])
def line_crop(line_id: int, conn: Conn, settings: Cfg):
    """The working copy cropped to the bounding box of the line's polygon."""
    line = lines_svc.get_line_row(conn, line_id)
    page = pages_svc.get_page_row(conn, line["page_id"])
    xs, ys = zip(*json.loads(line["polygon"]), strict=True)
    path = settings.data_dir / page["working_path"]
    if not path.is_file():
        raise NotFound("working copy missing on disk")
    with Image.open(path) as im:
        left, top = max(0, min(xs)), max(0, min(ys))
        right, bottom = min(im.width, max(xs) + 1), min(im.height, max(ys) + 1)
        crop = im.crop((left, top, max(right, left + 1), max(bottom, top + 1)))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


# ---- jobs ------------------------------------------------------------------------------------------------------


@router.get("/jobs", tags=["jobs"])
def list_jobs(conn: Conn, status: JobStatus | None = None, type: str | None = None, limit: int = 100):
    return queue.list_jobs(conn, status, type, max(1, min(limit, 500)))


@router.get("/jobs/{job_id}", tags=["jobs"])
def get_job(job_id: int, conn: Conn):
    return queue.get_job(conn, job_id)


@router.post("/jobs/{job_id}/cancel", tags=["jobs"])
def cancel_job(job_id: int, conn: Conn):
    """Cancel a queued job at once, or ask a running train/predict job to stop (it ends as `cancelled` at its next
    checkpoint; `cancel_requested` is true meanwhile). 409 for a running segmentation or a finished job."""
    return queue.cancel(conn, job_id)


# ---- training, prediction, models ------------------------------------------------------------------------------


def _reject_if_active(conn: sqlite3.Connection, job_type: str) -> None:
    for status in ("queued", "running"):
        for j in queue.list_jobs(conn, status=status, job_type=job_type, limit=1):
            raise Conflict(f"a {job_type} job is already {j['status']} (job {j['id']})")


@router.post("/jobs/train", status_code=202, tags=["training"])
def start_training(request: Request, conn: Conn, settings: Cfg, body: TrainRequest | None = None):
    """Enqueue a fine-tune job. 422 when there is nothing to train on or no start model exists; a short ground
    truth count is only a `warnings` entry in the response, never a refusal."""
    body = body or TrainRequest()
    models_svc.resolve_model(conn, settings, body.base_model_id)  # 404 / 422 / 409 now rather than in the job
    warnings = training_svc.check_can_train(conn)
    _reject_if_active(conn, "train")
    payload = body.model_dump(exclude_none=True)
    payload["warnings"] = warnings
    job = queue.enqueue(conn, "train", payload)
    request.app.state.worker.wake()
    return {**job, "warnings": warnings}


@router.post("/jobs/predict", status_code=202, tags=["training"])
def start_prediction(request: Request, conn: Conn, settings: Cfg, body: PredictRequest | None = None):
    """Enqueue a prediction job over the lines that are not ground truth, not ignored and have no manual text."""
    body = body or PredictRequest()
    row = models_svc.resolve_model(conn, settings, body.model_id)
    for page_id in body.page_ids or []:
        pages_svc.get_page_row(conn, page_id)
    _reject_if_active(conn, "predict")
    warnings = []
    if row["kind"] == "base":
        warnings.append(
            "This is the pretrained base model, not yet adapted to your handwriting; predictions will be rough. "
            "Fine-tune on your own ground-truth lines first for better results."
        )
    job = queue.enqueue(conn, "predict", body.model_dump(exclude_none=True))
    request.app.state.worker.wake()
    return {**job, "warnings": warnings}


@router.get("/models", tags=["models"])
def list_models(conn: Conn, settings: Cfg):
    models_svc.ensure_registry(conn, settings)
    return models_svc.list_models(conn)


@router.get("/models/{model_id}", tags=["models"])
def get_model(model_id: int, conn: Conn):
    return models_svc.get_model(conn, model_id)


@router.post("/models/{model_id}/activate", tags=["models"])
def activate_model(model_id: int, conn: Conn):
    """Make this the model predict and the next fine-tune use by default."""
    return models_svc.activate(conn, model_id)


@router.get("/training/summary", tags=["training"])
def training_summary(conn: Conn, settings: Cfg):
    """Ground-truth counts against the recommended minimums, with non-blocking warnings."""
    models_svc.ensure_registry(conn, settings)
    return training_svc.summary(conn)
