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
from ..services import pages as pages_svc
from ..worker import queue
from .deps import get_conn, get_settings
from .schemas import DocumentCreate, LineCreate, LinePatch, SegmentRequest

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


# ---- lines -----------------------------------------------------------------------------------------------------


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
    """Cancel a queued job (409 if it already started)."""
    return queue.cancel(conn, job_id)
