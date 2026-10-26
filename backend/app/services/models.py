"""Model registry: the `models` table (pretrained base models found on disk, plus fine-tuned models from train jobs)
and the single *active* model that predict and the next fine-tune use by default."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from ..config import Settings
from ..db import now_iso, write_tx
from ..errors import Conflict, Invalid, NotFound

log = logging.getLogger(__name__)

BASE_DIR = "models/base"  # relative to the data directory; filled by `make base-model`


def model_dict(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "name": r["name"],
        "kind": r["kind"],
        "engine": r["engine"],
        "path": r["path"],
        "base_model": r["base_model"],
        "parent_model_id": r["parent_model_id"],
        "trained_at": r["trained_at"],
        "train_lines": r["train_lines"],
        "eval_lines": r["eval_lines"],
        "cer": r["cer"],
        "wer": r["wer"],
        "base_cer": r["base_cer"],
        "base_wer": r["base_wer"],
        "hyperparameters": json.loads(r["hyperparameters"] or "{}"),
        "split_info": json.loads(r["split_info"] or "{}"),
        "metrics": json.loads(r["metrics"] or "{}"),
        "job_id": r["job_id"],
        "license": r["license"],
        "is_active": bool(r["is_active"]),
    }


def get_model_row(conn: sqlite3.Connection, model_id: int) -> sqlite3.Row:
    r = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    if r is None:
        raise NotFound(f"model {model_id} not found")
    return r


def get_model(conn: sqlite3.Connection, model_id: int) -> dict:
    return model_dict(get_model_row(conn, model_id))


def list_models(conn: sqlite3.Connection) -> list[dict]:
    """Newest first (base models, which have no training date, sort by id)."""
    rows = conn.execute("SELECT * FROM models ORDER BY COALESCE(trained_at, '') DESC, id DESC")
    return [model_dict(r) for r in rows]


def active_row(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM models WHERE is_active = 1").fetchone()


def model_file(settings: Settings, row: sqlite3.Row) -> Path:
    return settings.data_dir / row["path"]


def resolve_model(conn: sqlite3.Connection, settings: Settings, model_id: int | None) -> sqlite3.Row:
    """The requested model, or the active one. Raises Invalid (422) when there is none and Conflict (409) when its
    file is gone, with a message that says what to do."""
    ensure_registry(conn, settings)
    if model_id is not None:
        row = get_model_row(conn, model_id)
    else:
        row = active_row(conn)
        if row is None:
            raise Invalid(
                "no recognition model is available: run `make base-model` once to download the pretrained base "
                "model, or train one"
            )
    if not model_file(settings, row).is_file():
        raise Conflict(f"the file of model {row['id']} is missing on disk ({row['path']})")
    return row


def activate(conn: sqlite3.Connection, model_id: int) -> dict:
    get_model_row(conn, model_id)
    with write_tx(conn):
        conn.execute("UPDATE models SET is_active = 0 WHERE is_active = 1")
        conn.execute("UPDATE models SET is_active = 1 WHERE id = ?", (model_id,))
    return get_model(conn, model_id)


def register_base_models(conn: sqlite3.Connection, settings: Settings) -> int:
    """Add a `kind='base'` row for every pretrained model under data/models/base/<name>/BASE_MODEL.json that is not
    registered yet; if no model is active afterwards, the first base model becomes active. Returns rows added."""
    root = settings.data_dir / BASE_DIR
    added = 0
    if root.is_dir():
        for meta_file in sorted(root.glob("*/BASE_MODEL.json")):
            try:
                meta = json.loads(meta_file.read_text())
                rel = (meta_file.parent / meta["file"]).relative_to(settings.data_dir).as_posix()
            except (OSError, ValueError, KeyError):
                log.warning("ignoring unreadable base model record %s", meta_file)
                continue
            if not (settings.data_dir / rel).is_file():
                continue
            with write_tx(conn):
                cur = conn.execute(
                    "INSERT OR IGNORE INTO models (path, name, kind, engine, license, metrics, hyperparameters) "
                    "VALUES (?, ?, 'base', 'kraken', ?, ?, ?)",
                    (
                        rel,
                        meta.get("name") or meta_file.parent.name,
                        meta.get("license"),
                        json.dumps({"reported_cer_percent": meta.get("reported_cer_percent")}),
                        json.dumps({"doi": meta.get("doi"), "sha256": meta.get("sha256"), "title": meta.get("title")}),
                    ),
                )
                added += cur.rowcount
    if added and active_row(conn) is None:
        first = conn.execute("SELECT id FROM models WHERE kind = 'base' ORDER BY id LIMIT 1").fetchone()
        if first:
            activate(conn, first["id"])
    return added


def ensure_registry(conn: sqlite3.Connection, settings: Settings) -> None:
    """Cheap and idempotent: pick up a base model that was downloaded while the server was running."""
    register_base_models(conn, settings)


def create_finetuned(conn: sqlite3.Connection, fields: dict, lines: list[tuple[int, str]]) -> int:
    """Insert a fine-tuned model row plus its (line_id, split) membership; returns the id. Caller owns the tx."""
    cols = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    cur = conn.execute(
        f"INSERT INTO models ({cols}, kind, trained_at) VALUES ({marks}, 'finetuned', ?)", (*fields.values(), now_iso())
    )
    model_id = int(cur.lastrowid)
    conn.executemany(
        "INSERT INTO model_lines (model_id, line_id, split) VALUES (?, ?, ?)", [(model_id, lid, s) for lid, s in lines]
    )
    return model_id
