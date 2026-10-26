"""Optional local-LLM abbreviation suggestions: list, generate (a job), accept / reject. Off unless HTR_LLM_ENABLED."""

from __future__ import annotations

import sqlite3
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request

from ..config import Settings
from ..errors import Conflict
from ..services import suggestions as svc
from ..worker import queue
from .deps import get_conn, get_settings
from .schemas import SuggestionAccept, SuggestRequest

router = APIRouter(prefix="/abbreviations/suggestions", tags=["suggestions"])
Conn = Annotated[sqlite3.Connection, Depends(get_conn)]
Cfg = Annotated[Settings, Depends(get_settings)]


def _backend(settings: Settings) -> dict | None:
    if not settings.llm_enabled:
        return None
    return {"api": settings.llm_api, "url": settings.llm_url, "model": settings.llm_model}


@router.get("")
def list_suggestions(conn: Conn, settings: Cfg, status: Literal["pending", "accepted", "rejected", "all"] = "pending"):
    """Stored suggestions (default: those waiting for a decision) and whether the feature is switched on. Works while
    the feature is off, so earlier suggestions stay reviewable."""
    return {"enabled": settings.llm_enabled, "backend": _backend(settings), "items": svc.list_suggestions(conn, status)}


@router.post("/generate", status_code=202)
def generate(request: Request, conn: Conn, settings: Cfg, body: SuggestRequest | None = None):
    """Enqueue a suggestion job (one at a time; cancellable with POST /jobs/{id}/cancel). 409 while the feature is off
    or a suggestion job is already queued or running. Never changes a transcription."""
    body = body or SuggestRequest()
    if not settings.llm_enabled:
        raise Conflict(
            "Local-LLM suggestions are turned off.",
            "llm_disabled",
            "Set HTR_LLM_ENABLED=1 and start a local model server (for example Ollama) before starting the app; see "
            "the README.",
        )
    for status in ("queued", "running"):
        for job in queue.list_jobs(conn, status=status, job_type="suggest", limit=1):
            raise Conflict(f"a suggestion job is already {job['status']} (job {job['id']})", "job_active")
    payload = body.model_dump(exclude_none=True)
    if "tokens" in payload:
        payload["tokens"] = svc.validate_tokens(payload["tokens"])
    job = queue.enqueue(conn, "suggest", payload)
    request.app.state.worker.wake()
    return job


@router.post("/{suggestion_id}/accept")
def accept(suggestion_id: int, conn: Conn, body: SuggestionAccept | None = None):
    """Create the dictionary entry (optionally with edited written form / expansion / notes) and mark the suggestion
    accepted. -> {suggestion, abbreviation}"""
    return svc.accept(conn, suggestion_id, body.model_dump(exclude_none=True) if body else None)


@router.post("/{suggestion_id}/reject")
def reject(suggestion_id: int, conn: Conn):
    """Reject it; the (written, expansion) pair is remembered and never suggested again."""
    return svc.reject(conn, suggestion_id)
