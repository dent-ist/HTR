"""Versioned schema migrations.

Each migration is a (version, name, SQL script). Applied versions are recorded in `schema_version`; `migrate`
applies the pending ones in order, each in its own transaction (a failing migration leaves the database at the
previous version). To change the schema, append a new migration; never edit an applied one.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .connection import now_iso


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str


_V1 = """
CREATE TABLE documents (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    imported_at TEXT NOT NULL
);

-- Paths are relative to the data directory. width/height are the WORKING COPY size: all line coordinates and
-- the PAGE-XML are in working-copy pixels. image_path is the untouched original.
CREATE TABLE pages (
    id                INTEGER PRIMARY KEY,
    document_id       INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    image_path        TEXT NOT NULL,
    xml_path          TEXT NOT NULL,       -- the file exists only once the page has been segmented
    width             INTEGER NOT NULL,
    height            INTEGER NOT NULL,
    status            TEXT NOT NULL DEFAULT 'unsegmented'
                      CHECK (status IN ('unsegmented', 'segmented', 'partially_transcribed', 'done')),
    working_path      TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    sha256            TEXT NOT NULL UNIQUE,  -- content hash of the original; dedupes re-uploads
    original_width    INTEGER NOT NULL,
    original_height   INTEGER NOT NULL,
    dpi_x             REAL,
    dpi_y             REAL,
    line_height_px    REAL,                -- measured on the normalized, deskewed image
    scale             REAL NOT NULL DEFAULT 1.0,    -- working copy / original scale factor
    skew_deg          REAL NOT NULL DEFAULT 0.0,
    deskewed          INTEGER NOT NULL DEFAULT 0 CHECK (deskewed IN (0, 1)),
    warnings          TEXT NOT NULL DEFAULT '[]',   -- json list: import warnings
    preprocess        TEXT NOT NULL DEFAULT '{}',   -- json: full PreprocessInfo
    imported_at       TEXT NOT NULL,
    segmented_at      TEXT
);
CREATE INDEX idx_pages_document ON pages(document_id);

CREATE TABLE lines (
    id              INTEGER PRIMARY KEY,
    page_id         INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    "index"         INTEGER NOT NULL,      -- reading order, 0-based, contiguous per page (kept by the service layer)
    baseline        TEXT NOT NULL,         -- json [[x, y], ...]
    polygon         TEXT NOT NULL,         -- json [[x, y], ...]
    text            TEXT NOT NULL DEFAULT '',
    text_source     TEXT CHECK (text_source IN ('manual', 'predicted')),  -- NULL while there is no text
    confidence      REAL,
    is_ground_truth INTEGER NOT NULL DEFAULT 0 CHECK (is_ground_truth IN (0, 1)),
    ignored         INTEGER NOT NULL DEFAULT 0 CHECK (ignored IN (0, 1))
);
CREATE INDEX idx_lines_page ON lines(page_id, "index");

CREATE TABLE models (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL,
    base_model  TEXT,
    trained_at  TEXT,
    train_lines INTEGER,
    cer         REAL
);

CREATE TABLE jobs (
    id          INTEGER PRIMARY KEY,
    type        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued'
                CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    payload     TEXT NOT NULL DEFAULT '{}',
    progress    REAL NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 1),
    error       TEXT,
    result      TEXT,
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT
);
CREATE INDEX idx_jobs_status ON jobs(status, id);

CREATE TABLE abbreviations (
    id        INTEGER PRIMARY KEY,
    written   TEXT NOT NULL,
    expansion TEXT NOT NULL,
    notes     TEXT,
    count     INTEGER NOT NULL DEFAULT 0
);
"""

MIGRATIONS: list[Migration] = [Migration(1, "initial schema", _V1)]
LATEST_VERSION = MIGRATIONS[-1].version


def current_version(conn: sqlite3.Connection) -> int:
    has = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'").fetchone()
    if not has:
        return 0
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return row[0] or 0


def migrate(conn: sqlite3.Connection, migrations: list[Migration] | None = None) -> int:
    """Apply pending migrations; returns the resulting schema version."""
    migrations = MIGRATIONS if migrations is None else migrations
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version ("
        "version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)"
    )
    current = current_version(conn)
    latest = max((m.version for m in migrations), default=0)
    if current > latest:
        raise RuntimeError(f"database schema version {current} is newer than this app supports ({latest})")
    for m in sorted(migrations, key=lambda m: m.version):
        if m.version <= current:
            continue
        name = m.name.replace("'", "''")
        # executescript commits any open transaction first, so the BEGIN/COMMIT live inside the script.
        script = (
            "BEGIN IMMEDIATE;\n"
            f"{m.sql}\n"
            "INSERT INTO schema_version (version, name, applied_at) "
            f"VALUES ({int(m.version)}, '{name}', '{now_iso()}');\n"
            "COMMIT;"
        )
        try:
            conn.executescript(script)
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
    return current_version(conn)
