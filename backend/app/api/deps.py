from __future__ import annotations

import sqlite3
from collections.abc import Iterator

from fastapi import Request

from ..config import Settings
from ..db import connect


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_conn(request: Request) -> Iterator[sqlite3.Connection]:
    conn = connect(request.app.state.settings.db_path)
    try:
        yield conn
    finally:
        conn.close()
