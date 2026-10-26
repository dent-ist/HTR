"""Text / PAGE-XML export and the project bundle."""

from __future__ import annotations

import sqlite3
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask

from ..config import Settings
from ..db import connect
from ..errors import Conflict, Invalid
from ..services import bundle as bundle_svc
from ..services import export as export_svc
from ..worker.runner import Worker
from .deps import get_conn, get_settings

router = APIRouter()
Conn = Annotated[sqlite3.Connection, Depends(get_conn)]
Cfg = Annotated[Settings, Depends(get_settings)]


def _chunks(f, size: int = 1 << 20):
    while chunk := f.read(size):
        yield chunk


@router.get("/export", tags=["export"])
def export(
    conn: Conn,
    format: Literal["txt", "pagexml"] = "txt",
    expanded: bool = False,
    ambiguous: Literal["raw", "mark"] = "raw",
    document_id: int | None = None,
    page_ids: str | None = None,
    include_predicted: bool = False,
):
    """Plain text (UTF-8, `.txt`) or PAGE-XML (zip). Ignored and empty lines are never exported; predicted lines only
    with `include_predicted` (text lines are then prefixed `[predicted] `). `expanded` and `ambiguous` apply to text."""
    if format == "pagexml" and expanded:
        raise Invalid("expanded applies to text export only; PAGE-XML always carries the text as written")
    pages = export_svc.select_pages(conn, document_id, export_svc.parse_ids(page_ids))
    scope = export_svc.slug(pages[0].document_name) if document_id is not None and pages else "all"
    if format == "txt":
        text = export_svc.export_text(conn, pages, expanded, include_predicted, ambiguous)
        name = f"htr-{scope}-{'expanded' if expanded else 'raw'}.txt"
        return Response(
            text.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )
    out = export_svc.export_pagexml(conn, pages, include_predicted)
    return StreamingResponse(
        _chunks(out),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="htr-{scope}-pagexml.zip"'},
        background=BackgroundTask(out.close),
    )


@router.get("/bundle/export", tags=["bundle"])
def bundle_export(settings: Cfg):
    """Stream the whole project as a `.htrbundle` zip. 409 while a segment, train or predict job is queued or running
    (checked before the first byte)."""
    plan = bundle_svc.prepare_export(settings)
    stamp = plan.created_at[:19].replace("-", "").replace(":", "").replace("T", "-")
    return StreamingResponse(
        bundle_svc.iter_bundle(plan),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="htr-project-{stamp}.htrbundle"'},
        background=BackgroundTask(plan.close),
    )


@router.post("/bundle/import", tags=["bundle"])
def bundle_import(request: Request, settings: Cfg, file: Annotated[UploadFile, File(description=".htrbundle")]):
    """Import a bundle into an EMPTY project (409 otherwise). The upload is spooled to disk, verified completely in a
    staging directory, then swapped in; on any failure the project is untouched. Other requests get 503 meanwhile."""
    gate = request.app.state.gate
    worker: Worker = request.app.state.worker
    gate.begin()
    stopped = False
    try:
        conn = connect(settings.db_path)
        try:
            bundle_svc.require_no_active_jobs(conn)
        finally:
            conn.close()
        if not worker.stop(timeout=60):
            raise Conflict("the background worker is still busy; try again when it is idle")
        stopped = True
        return bundle_svc.import_bundle(settings, file.file)
    finally:
        try:
            file.file.close()
        finally:
            if stopped:
                worker.start()
            gate.end()
