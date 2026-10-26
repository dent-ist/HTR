from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def connect(path: Path | str) -> sqlite3.Connection:
    """Open a connection: foreign keys on, WAL, rows addressable by name.

    Autocommit mode (isolation_level=None): transactions are explicit (`write_tx`), so there is never an
    implicit transaction hiding behind a statement. One connection per request / worker; a connection is never
    used by two threads at once, hence check_same_thread=False (FastAPI may hop threads between a dependency and
    its endpoint).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, isolation_level=None, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


@contextmanager
def write_tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE ... COMMIT, ROLLBACK on any exception."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.rollback()
        raise
    try:
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
