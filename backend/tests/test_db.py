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
    assert migrate(conn) == LATEST_VERSION == 1
    assert EXPECTED_TABLES <= tables(conn)
    row = conn.execute("SELECT version, name FROM schema_version").fetchone()
    assert (row["version"], row["name"]) == (1, "initial schema")


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
    assert migrate(conn) == 1
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1


def test_pragmas_wal_and_foreign_keys(conn):
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_later_migration_applies_on_top(conn):
    migrate(conn)
    extra = Migration(2, "add note", "ALTER TABLE documents ADD COLUMN note TEXT;")
    from app.db.migrations import MIGRATIONS

    assert migrate(conn, [*MIGRATIONS, extra]) == 2
    assert "note" in columns(conn, "documents")


def test_failing_migration_rolls_back_and_keeps_old_version(conn):
    from app.db.migrations import MIGRATIONS

    migrate(conn)
    bad = Migration(2, "broken", "CREATE TABLE half (id INTEGER); INSERT INTO nonexistent VALUES (1);")
    with pytest.raises(sqlite3.OperationalError):
        migrate(conn, [*MIGRATIONS, bad])
    assert "half" not in tables(conn)
    assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 1
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
