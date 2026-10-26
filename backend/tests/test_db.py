import sqlite3

import pytest

from app.config import Settings
from app.db import LATEST_VERSION, connect, migrate, now_iso
from app.db.migrations import Migration

EXPECTED_TABLES = {"documents", "pages", "lines", "models", "jobs", "abbreviations", "schema_version"}


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "x" / "htr.db")
    yield c
    c.close()


def tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def columns(conn, table):
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def test_migrate_creates_schema_and_records_version(conn):
    assert migrate(conn) == LATEST_VERSION == 6
    assert EXPECTED_TABLES <= tables(conn)
    rows = conn.execute("SELECT version, name FROM schema_version ORDER BY version").fetchall()
    assert [(r["version"], r["name"]) for r in rows] == [
        (1, "initial schema"),
        (2, "models registry and splits"),
        (3, "review events"),
        (4, "abbreviation dictionary"),
        (5, "job error codes and hints"),
        (6, "abbreviation suggestions"),
    ]


def test_v2_adds_model_registry_columns_and_split_tables(conn):
    migrate(conn)
    assert columns(conn, "models") >= {
        "id", "path", "base_model", "trained_at", "train_lines", "cer",  # from v1
        "eval_lines", "wer", "engine", "hyperparameters", "split_info", "is_active", "kind", "parent_model_id",
        "base_cer", "base_wer", "metrics", "job_id", "license", "name",
    }  # fmt: skip
    assert {"cancel_requested"} <= columns(conn, "jobs")
    assert {"line_id", "split", "assigned_at"} <= columns(conn, "line_splits")
    assert {"model_id", "line_id", "split"} <= columns(conn, "model_lines")


def test_v5_and_v6_add_job_error_details_and_the_suggestions_table(conn):
    migrate(conn)
    assert {"error_code", "error_hint"} <= columns(conn, "jobs")
    assert columns(conn, "suggestions") >= {
        "id", "written", "expansion", "rationale", "line_id", "context", "model", "status", "abbreviation_id",
        "created_at", "resolved_at",
    }  # fmt: skip
    conn.execute("INSERT INTO suggestions (written, expansion, created_at) VALUES ('c/o', 'complains of', 't')")
    with pytest.raises(sqlite3.IntegrityError):  # one row per (written, expansion): a rejection is remembered
        conn.execute("INSERT INTO suggestions (written, expansion, created_at) VALUES ('c/o', 'complains of', 't')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO suggestions (written, expansion, status, created_at) VALUES ('a', 'b', 'x', 't')")


def test_only_one_model_can_be_active(conn):
    migrate(conn)
    conn.execute("INSERT INTO models (path, is_active) VALUES ('a', 1)")
    conn.execute("INSERT INTO models (path, is_active) VALUES ('b', 0)")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE models SET is_active = 1 WHERE path = 'b'")


def test_v2_migrates_an_existing_v1_database_in_place(conn):
    from app.db.migrations import MIGRATIONS

    migrate(conn, MIGRATIONS[:1])
    conn.execute("INSERT INTO models (path, base_model, train_lines, cer) VALUES ('old.mlmodel', 'x', 5, 0.2)")
    assert migrate(conn) == LATEST_VERSION
    row = conn.execute("SELECT * FROM models").fetchone()
    assert (row["path"], row["cer"], row["engine"], row["kind"], row["is_active"]) == (
        "old.mlmodel",
        0.2,
        "kraken",
        "finetuned",
        0,
    )


def test_schema_matches_plan_data_model(conn):
    migrate(conn)
    assert {"id", "name", "imported_at"} <= columns(conn, "documents")
    assert {"id", "document_id", "image_path", "xml_path", "width", "height", "status"} <= columns(conn, "pages")
    assert {"original_filename", "sha256", "scale", "skew_deg", "warnings"} <= columns(conn, "pages")
    assert columns(conn, "lines") >= {
        "id", "page_id", "index", "baseline", "polygon", "text", "text_source", "confidence",
        "is_ground_truth", "ignored",
    }  # fmt: skip
    assert columns(conn, "models") >= {"id", "path", "base_model", "trained_at", "train_lines", "cer"}
    assert columns(conn, "jobs") >= {"id", "type", "status", "payload", "progress", "error", "created_at"}
    assert columns(conn, "abbreviations") >= {"id", "written", "expansion", "notes", "count"}


def test_migrate_is_idempotent(conn):
    migrate(conn)
    conn.execute("INSERT INTO documents (name, imported_at) VALUES ('d', ?)", (now_iso(),))
    assert migrate(conn) == LATEST_VERSION
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == LATEST_VERSION


def test_pragmas_wal_and_foreign_keys(conn):
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_later_migration_applies_on_top(conn):
    migrate(conn)
    extra = Migration(LATEST_VERSION + 1, "add note", "ALTER TABLE documents ADD COLUMN note TEXT;")
    from app.db.migrations import MIGRATIONS

    assert migrate(conn, [*MIGRATIONS, extra]) == LATEST_VERSION + 1
    assert "note" in columns(conn, "documents")


def test_failing_migration_rolls_back_and_keeps_old_version(conn):
    from app.db.migrations import MIGRATIONS

    migrate(conn)
    bad = Migration(LATEST_VERSION + 1, "broken", "CREATE TABLE half (id INTEGER); INSERT INTO nonexistent VALUES (1);")
    with pytest.raises(sqlite3.OperationalError):
        migrate(conn, [*MIGRATIONS, bad])
    assert "half" not in tables(conn)
    assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == LATEST_VERSION
    assert not conn.in_transaction


def test_database_newer_than_app_is_refused(conn):
    migrate(conn)
    conn.execute("INSERT INTO schema_version VALUES (99, 'future', ?)", (now_iso(),))
    with pytest.raises(RuntimeError, match="newer"):
        migrate(conn)


def test_foreign_keys_enforced_and_cascade(conn):
    migrate(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO pages (document_id, image_path, xml_path, width, height, working_path, original_filename, "
            "sha256, original_width, original_height, imported_at) "
            "VALUES (999, 'a', 'b', 1, 1, 'c', 'd', 'e', 1, 1, 'f')"
        )
    conn.execute("INSERT INTO documents (name, imported_at) VALUES ('d', 'now')")
    conn.execute(
        "INSERT INTO pages (document_id, image_path, xml_path, width, height, working_path, original_filename, "
        "sha256, original_width, original_height, imported_at) VALUES (1, 'a', 'b', 1, 1, 'c', 'd', 'e', 1, 1, 'f')"
    )
    conn.execute('INSERT INTO lines (page_id, "index", baseline, polygon) VALUES (1, 0, "[]", "[]")')
    conn.execute("DELETE FROM documents WHERE id = 1")
    assert conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM lines").fetchone()[0] == 0


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE pages SET status = 'bogus'",
        "UPDATE lines SET text_source = 'guessed'",
        "UPDATE lines SET ignored = 2",
        "UPDATE jobs SET status = 'paused'",
        "UPDATE jobs SET progress = 1.5",
    ],
)
def test_check_constraints(conn, sql):
    migrate(conn)
    conn.execute("INSERT INTO documents (name, imported_at) VALUES ('d', 'now')")
    conn.execute(
        "INSERT INTO pages (document_id, image_path, xml_path, width, height, working_path, original_filename, "
        "sha256, original_width, original_height, imported_at) VALUES (1, 'a', 'b', 1, 1, 'c', 'd', 'e', 1, 1, 'f')"
    )
    conn.execute('INSERT INTO lines (page_id, "index", baseline, polygon) VALUES (1, 0, "[]", "[]")')
    conn.execute("INSERT INTO jobs (type, created_at) VALUES ('segment', 'now')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(sql)


def test_sha256_is_unique(conn):
    migrate(conn)
    conn.execute("INSERT INTO documents (name, imported_at) VALUES ('d', 'now')")
    ins = (
        "INSERT INTO pages (document_id, image_path, xml_path, width, height, working_path, original_filename, "
        "sha256, original_width, original_height, imported_at) VALUES (1, 'a', 'b', 1, 1, 'c', 'd', 'same', 1, 1, 'f')"
    )
    conn.execute(ins)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(ins)


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HTR_DATA_DIR", str(tmp_path / "d"))
    s = Settings.from_env()
    assert s.data_dir == tmp_path / "d" and s.db_path == tmp_path / "d" / "htr.db"
    monkeypatch.setenv("HTR_DB_PATH", str(tmp_path / "other.db"))
    assert Settings.from_env().db_path == tmp_path / "other.db"
    monkeypatch.delenv("HTR_DATA_DIR")
    monkeypatch.delenv("HTR_DB_PATH")
    assert Settings.from_env().db_path.name == "htr.db"
