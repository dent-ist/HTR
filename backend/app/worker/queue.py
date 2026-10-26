"""SQLite-backed job queue (the `jobs` table). All functions take a connection and manage their own transaction."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ..db import now_iso, write_tx
from ..errors import Conflict, NotFound

INTERRUPTED = "interrupted: the server stopped while this job was running"


def job_dict(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "type": r["type"],
        "status": r["status"],
        "payload": json.loads(r["payload"]),
        "progress": r["progress"],
        "error": r["error"],
        "result": json.loads(r["result"]) if r["result"] else None,
        "created_at": r["created_at"],
        "started_at": r["started_at"],
        "finished_at": r["finished_at"],
    }


def enqueue(conn: sqlite3.Connection, job_type: str, payload: dict[str, Any] | None = None) -> dict:
    with write_tx(conn):
        cur = conn.execute(
            "INSERT INTO jobs (type, payload, created_at) VALUES (?, ?, ?)",
            (job_type, json.dumps(payload or {}), now_iso()),
        )
    return get_job(conn, cur.lastrowid)


def get_job(conn: sqlite3.Connection, job_id: int) -> dict:
    r = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if r is None:
        raise NotFound(f"job {job_id} not found")
    return job_dict(r)


def list_jobs(
    conn: sqlite3.Connection, status: str | None = None, job_type: str | None = None, limit: int = 100
) -> list[dict]:
    where, args = [], []
    if status:
        where.append("status = ?")
        args.append(status)
    if job_type:
        where.append("type = ?")
        args.append(job_type)
    sql = "SELECT * FROM jobs" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id DESC LIMIT ?"
    return [job_dict(r) for r in conn.execute(sql, (*args, limit))]


def find_active(conn: sqlite3.Connection, job_type: str, key: str, value: Any) -> dict | None:
    """A queued or running job of `job_type` whose payload has payload[key] == value."""
    for r in conn.execute("SELECT * FROM jobs WHERE type = ? AND status IN ('queued', 'running')", (job_type,)):
        if json.loads(r["payload"]).get(key) == value:
            return job_dict(r)
    return None


def claim_next(conn: sqlite3.Connection) -> dict | None:
    """Atomically take the oldest queued job and mark it running. One worker, so one job at a time."""
    with write_tx(conn):
        r = conn.execute("SELECT * FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1").fetchone()
        if r is None:
            return None
        conn.execute(
            "UPDATE jobs SET status = 'running', started_at = ?, progress = 0 WHERE id = ?", (now_iso(), r["id"])
        )
    return get_job(conn, r["id"])


def set_progress(conn: sqlite3.Connection, job_id: int, progress: float) -> None:
    p = min(1.0, max(0.0, float(progress)))
    conn.execute("UPDATE jobs SET progress = ? WHERE id = ? AND status = 'running'", (p, job_id))


def finish(conn: sqlite3.Connection, job_id: int, result: dict | None = None) -> None:
    with write_tx(conn):
        conn.execute(
            "UPDATE jobs SET status = 'succeeded', progress = 1, result = ?, finished_at = ? WHERE id = ?",
            (json.dumps(result) if result is not None else None, now_iso(), job_id),
        )


def fail(conn: sqlite3.Connection, job_id: int, error: str) -> None:
    with write_tx(conn):
        conn.execute(
            "UPDATE jobs SET status = 'failed', error = ?, finished_at = ? WHERE id = ?", (error, now_iso(), job_id)
        )


def cancel(conn: sqlite3.Connection, job_id: int) -> dict:
    """Cancel a job that has not started. A running job cannot be interrupted (it holds the GPU)."""
    with write_tx(conn):
        r = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if r is None:
            raise NotFound(f"job {job_id} not found")
        if r["status"] != "queued":
            raise Conflict(f"job {job_id} is {r['status']}; only queued jobs can be cancelled")
        conn.execute("UPDATE jobs SET status = 'cancelled', finished_at = ? WHERE id = ?", (now_iso(), job_id))
    return get_job(conn, job_id)


def fail_interrupted(conn: sqlite3.Connection) -> int:
    """Startup recovery: jobs still 'running' belong to a process that died. Mark them failed. Returns the count."""
    with write_tx(conn):
        cur = conn.execute(
            "UPDATE jobs SET status = 'failed', error = ?, finished_at = ? WHERE status = 'running'",
            (INTERRUPTED, now_iso()),
        )
    return cur.rowcount
