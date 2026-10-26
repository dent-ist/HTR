"""Abbreviation dictionary, expansion preview and raw / expanded text views."""

from __future__ import annotations

import sqlite3
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Response

from ..services import abbreviations as svc
from ..services import lines as lines_svc
from .deps import get_conn
from .schemas import AbbreviationCreate, AbbreviationPatch, ExpandRequest

router = APIRouter()
Conn = Annotated[sqlite3.Connection, Depends(get_conn)]
Ambiguous = Literal["raw", "mark"]


@router.get("/abbreviations", tags=["abbreviations"])
def list_abbreviations(
    conn: Conn,
    q: str | None = None,
    ambiguous: bool | None = None,
    sort: Literal["written", "count"] = "written",
):
    """The dictionary with live occurrence counts and ambiguity (several rows that can match the same token)."""
    return svc.list_abbreviations(conn, q, ambiguous, sort)


@router.post("/abbreviations", status_code=201, tags=["abbreviations"])
def create_abbreviation(body: AbbreviationCreate, conn: Conn):
    return svc.create(conn, body.written, body.expansion, body.notes, body.case_sensitive)


@router.get("/abbreviations/{abbreviation_id}", tags=["abbreviations"])
def get_abbreviation(abbreviation_id: int, conn: Conn):
    return svc.get_abbreviation(conn, abbreviation_id)


@router.patch("/abbreviations/{abbreviation_id}", tags=["abbreviations"])
def patch_abbreviation(abbreviation_id: int, body: AbbreviationPatch, conn: Conn):
    return svc.update(conn, abbreviation_id, body.changes())


@router.delete("/abbreviations/{abbreviation_id}", status_code=204, tags=["abbreviations"])
def delete_abbreviation(abbreviation_id: int, conn: Conn):
    svc.delete(conn, abbreviation_id)
    return Response(status_code=204)


@router.get("/abbreviations/{abbreviation_id}/occurrences", tags=["abbreviations"])
def abbreviation_occurrences(
    abbreviation_id: int,
    conn: Conn,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Where the entry applies (non-ignored lines), paginated, in document / page / reading order."""
    return svc.occurrences(conn, abbreviation_id, limit, offset)


@router.post("/expand", tags=["abbreviations"])
def expand_preview(body: ExpandRequest, conn: Conn):
    """Expand arbitrary text with the dictionary (a preview; nothing is stored). All tokens come back as spans,
    including the unmatched ones."""
    return svc.expand_text(conn, body.text, body.ambiguous)


@router.get("/lines/{line_id}", tags=["lines"])
def get_line(line_id: int, conn: Conn, expanded: bool = False, ambiguous: Ambiguous = "raw"):
    """One line; with `expanded=true` also `expanded_text` and `spans`."""
    return svc.line_view(conn, lines_svc.get_line(conn, line_id), expanded, ambiguous)


@router.get("/pages/{page_id}/text", tags=["pages"])
def page_text(
    page_id: int,
    conn: Conn,
    expanded: bool = False,
    ambiguous: Ambiguous = "raw",
    include_predicted: bool = True,
):
    """The page's text lines (not ignored, not empty), each with raw and expanded text, plus the joined `text`
    (expanded or raw). Predicted lines are included by default here (marked `[predicted] ` in `text`)."""
    return svc.page_text(conn, page_id, expanded, ambiguous, include_predicted)
