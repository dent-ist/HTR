"""Project bundle: export, import, round trip, and every way an import must be refused (see services/bundle.py)."""

from __future__ import annotations

import io
import json
import os
import shutil
import sqlite3
import stat
import zipfile

import pytest
from fastapi.testclient import TestClient

from app.db import LATEST_VERSION, connect
from app.db.migrations import MIGRATIONS, migrate
from app.errors import Conflict
from app.main import create_app
from app.services import bundle as bundle_svc
from tests import bundle_helpers as H
from tests.test_ml_jobs import project_template  # noqa: F401  (session fixture)


@pytest.fixture(scope="module")
def golden(tmp_path_factory, project_template):  # noqa: F811
    return H.make_golden(tmp_path_factory, project_template)


@pytest.fixture
def fresh(tmp_path, fake_ml):
    """A running app on a brand-new empty data directory (fake ML engines patched in)."""
    settings = H.fresh_settings(tmp_path)
    with TestClient(create_app(settings)) as c:
        yield c, settings


# ---- the round trip ----------------------------------------------------------------------------------------------


def test_manifest_and_layout(golden):
    entries = H.read_entries(golden.bundle)
    manifest = json.loads(entries["manifest.json"])
    assert manifest["format"] == "htr-bundle" and manifest["bundle_format_version"] == 1
    assert manifest["schema_version"] == LATEST_VERSION and manifest["app_version"] and manifest["created_at"]
    assert manifest["counts"]["pages"] == 10 and manifest["counts"]["finetuned_models"] == 1
    assert manifest["counts"]["abbreviations"] == 4 and manifest["counts"]["review_events"] == 3
    assert manifest["counts"]["files"] == len(manifest["files"]) == len(entries) - 1
    listed = {f["path"]: f for f in manifest["files"]}
    assert set(listed) == set(entries) - {"manifest.json"}
    import hashlib

    for path, f in listed.items():
        assert f["size"] == len(entries[path]) and f["sha256"] == hashlib.sha256(entries[path]).hexdigest()
        assert not path.startswith("/") and ".." not in path.split("/") and "\\" not in path
    tops = {p.split("/")[0] for p in listed}
    assert tops == {"db", "originals", "working", "pagexml", "models"}
    assert "db/htr.sqlite" in listed and "models/base/test-base/base.mlmodel" in listed
    assert "models/base/test-base/BASE_MODEL.json" in listed
    assert f"models/{golden.texts['model_id']}.mlmodel" in listed
    # scans are byte-identical to what the API serves as the original
    for row in golden.dump["db"]["pages"]:
        assert entries[row["image_path"]] == golden.texts["original"][row["id"]]
    # no absolute path or personal location anywhere in the manifest
    assert "/home/" not in json.dumps(manifest)


def test_bundle_database_is_a_consistent_standalone_snapshot(golden, tmp_path):
    entries = H.read_entries(golden.bundle)
    db = tmp_path / "snap.sqlite"
    db.write_bytes(entries["db/htr.sqlite"])
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert conn.execute("SELECT COUNT(*) FROM lines").fetchone()[0] == len(golden.dump["db"]["lines"])
    assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == LATEST_VERSION
    conn.close()


def test_round_trip_into_a_fresh_data_directory(golden, fresh):
    client, settings = fresh
    assert H.db_dump(settings)["pages"] == []
    r = H.import_via_api(client, golden.bundle)
    assert r.status_code == 200, r.text
    summary = r.json()
    assert summary["imported"] is True and summary["counts"]["pages"] == 10 and summary["migrated_from"] is None

    # every table row, including ids, text, flags, geometry and order, plus job history and review events
    got = H.db_dump(settings)
    for table, rows in golden.dump["db"].items():
        assert got[table] == rows, table
    # scans, working copies, PAGE-XML and model files byte for byte (the base model came out of the bundle)
    assert H.file_dump(settings) == golden.files
    assert "models/base/test-base/base.mlmodel" in H.file_dump(settings)
    assert H.xml_in_sync(settings) == []

    # the API shows the same project
    pages = {p["id"]: client.get(f"/pages/{p['id']}").json() for p in client.get("/pages").json()}
    assert pages == golden.texts["pages"]
    for pid in pages:
        assert client.get(f"/pages/{pid}/original").content == golden.texts["original"][pid]
    for lid, crop in golden.texts["crops"].items():
        assert client.get(f"/lines/{lid}/crop.png").content == crop
    models = client.get("/models").json()
    assert models == golden.texts["models"]
    assert [m["id"] for m in models if m["is_active"]] == [golden.texts["model_id"]]
    assert client.get("/abbreviations").json() == golden.texts["abbreviations"]
    assert client.get("/review/stats").json() == golden.texts["review_stats"]
    assert client.get("/jobs?limit=500").json() == golden.texts["jobs"]
    ambiguous = [a for a in client.get("/abbreviations").json() if a["ambiguous"]]
    assert len(ambiguous) == 2
    # ground truth / ignored / predicted flags
    ignored = client.get(f"/lines/{golden.texts['ignored_gt']}").json()
    assert ignored["ignored"] is True
    # the expanded text export is identical
    assert client.get("/export", params={"expanded": "true", "ambiguous": "mark"}).text == golden.texts["export_text"]
    assert "patient complains of SOB[?]" in golden.texts["export_text"]


def test_imported_project_keeps_working_predict_and_edit(golden, fresh):
    client, settings = fresh
    assert H.import_via_api(client, golden.bundle).status_code == 200
    # the worker was restarted after the import: a predict job runs with the imported (active) model
    done = H.start_job(client, "/jobs/predict", {})
    assert done["result"]["model_id"] == golden.texts["model_id"] and done["result"]["predicted"] >= 1
    # edits keep PAGE-XML in sync, new abbreviations work, a second export/import cycle is stable
    lid = golden.texts["open_ids"][5]
    assert client.patch(f"/lines/{lid}", json={"text": "pt after import", "is_ground_truth": True}).status_code == 200
    assert client.post("/abbreviations", json={"written": "xyz", "expansion": "after"}).status_code == 201
    assert H.xml_in_sync(settings) == []
    assert client.get("/bundle/export").status_code == 200


def test_cli_round_trip(golden, tmp_path, fake_ml):
    import htr

    src = tmp_path / "src"
    shutil.copytree(golden.settings.data_dir, src)
    out = tmp_path / "out" / "project.htrbundle"
    out.parent.mkdir()
    assert htr.main(["--data", str(src), "bundle", "export", str(out)]) == 0
    assert out.is_file() and zipfile.is_zipfile(out)
    assert htr.main(["--data", str(src), "bundle", "export", str(out)]) == 1  # never overwrites
    dest = tmp_path / "dest"
    assert htr.main(["--data", str(dest), "bundle", "import", str(out)]) == 0
    from app.config import Settings

    s = Settings.for_dir(dest)
    assert H.db_dump(s)["lines"] == golden.dump["db"]["lines"]
    assert H.file_dump(s) == golden.files
    assert htr.main(["--data", str(dest), "bundle", "import", str(out)]) == 1  # now it is not empty
    assert htr.main(["--data", str(dest), "bundle", "import", str(tmp_path / "nope.htrbundle")]) == 2


def test_cli_import_refuses_while_a_server_holds_the_data_directory(golden, tmp_path):
    import htr

    from app.projectlock import try_lock

    out = tmp_path / "p.htrbundle"
    out.write_bytes(golden.bundle)
    dest = tmp_path / "dest"
    lock = try_lock(dest)
    assert lock is not None
    try:
        assert htr.main(["--data", str(dest), "bundle", "import", str(out)]) == 1
    finally:
        lock.close()
    assert htr.main(["--data", str(dest), "bundle", "import", str(out)]) == 0


# ---- exporting ---------------------------------------------------------------------------------------------------


def test_export_headers_and_streaming(golden, tmp_path):
    settings = H.fresh_settings(tmp_path, "stream")
    shutil.copytree(golden.settings.data_dir, settings.data_dir)
    # a large "scan" so that streaming is visible: the response arrives in chunks well below the file size
    big = settings.data_dir / "originals" / "padding.bin"
    big.write_bytes(os.urandom(6 << 20))
    conn = connect(settings.db_path)
    conn.execute("UPDATE pages SET image_path = 'originals/padding.bin' WHERE id = (SELECT MIN(id) FROM pages)")
    conn.close()
    plan = bundle_svc.prepare_export(settings)
    sizes = [len(c) for c in bundle_svc.iter_bundle(plan)]
    assert len(sizes) > 4 and max(sizes) <= 2 << 20
    assert not list(settings.tmp_dir.glob("snapshot-*"))  # the snapshot is cleaned up


def test_export_http_headers(golden, fresh):
    client, _ = fresh
    r = client.get("/bundle/export")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    cd = r.headers["content-disposition"]
    assert cd.startswith('attachment; filename="htr-project-') and cd.endswith('.htrbundle"')
    assert zipfile.ZipFile(io.BytesIO(r.content)).testzip() is None  # an empty project exports too


def test_export_refused_while_a_job_is_queued_or_running(golden, tmp_path, fake_ml):
    settings = H.fresh_settings(tmp_path, "busy")
    shutil.copytree(golden.settings.data_dir, settings.data_dir)
    with TestClient(create_app(settings)) as c:
        conn = connect(settings.db_path)
        conn.execute("INSERT INTO jobs (type, status, created_at) VALUES ('train', 'running', 'x')")
        r = c.get("/bundle/export")
        assert r.status_code == 409 and "train job is running" in r.json()["detail"]
        conn.execute("UPDATE jobs SET status = 'cancelled' WHERE status = 'running'")
        assert c.get("/bundle/export").status_code == 200
        conn.close()


def test_a_queued_job_also_blocks_export(tmp_path):
    settings = H.fresh_settings(tmp_path, "queued")
    conn = connect(settings.db_path)
    migrate(conn)
    conn.execute("INSERT INTO jobs (type, status, created_at) VALUES ('segment', 'queued', 'x')")
    conn.close()
    with pytest.raises(Conflict, match="segment job is queued"):
        bundle_svc.prepare_export(settings)


def test_snapshot_is_consistent_with_wal_commits(tmp_path):
    """Commits that still sit in the -wal file are in the bundle's snapshot (a plain file copy would miss them)."""
    settings = H.fresh_settings(tmp_path, "wal")
    conn = connect(settings.db_path)
    migrate(conn)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("PRAGMA wal_autocheckpoint = 0")
    conn.execute("INSERT INTO documents (name, imported_at) VALUES ('in the wal', 'now')")
    assert (settings.db_path.parent / "htr.db-wal").stat().st_size > 0
    shutil.copyfile(settings.db_path, tmp_path / "raw.db")  # the naive copy
    naive = sqlite3.connect(tmp_path / "raw.db")
    assert naive.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    naive.close()
    snap = tmp_path / "snap.sqlite"
    bundle_svc._snapshot_db(settings, snap)
    s = sqlite3.connect(snap)
    assert s.execute("SELECT name FROM documents").fetchall() == [("in the wal",)]
    s.close()
    conn.close()


# ---- refusing non-empty projects ---------------------------------------------------------------------------------


def test_import_into_a_non_empty_project_is_refused_and_changes_nothing(golden, client, api):
    api.import_page()
    before = H.all_files(client.app.state.settings), H.db_dump(client.app.state.settings)
    r = H.import_via_api(client, golden.bundle)
    assert r.status_code == 409
    assert "not empty" in r.json()["detail"] and "1 documents" in r.json()["detail"]
    assert (H.all_files(client.app.state.settings), H.db_dump(client.app.state.settings)) == before
    assert client.post("/documents", json={"name": "still works"}).status_code == 201


def test_a_single_abbreviation_also_makes_the_project_non_empty(golden, fresh):
    client, _ = fresh
    client.post("/abbreviations", json={"written": "pt", "expansion": "patient"})
    r = H.import_via_api(client, golden.bundle)
    assert r.status_code == 409 and "abbreviations" in r.json()["detail"]


def test_base_model_rows_created_at_startup_do_not_count(golden, tmp_path, fake_ml):
    settings = H.fresh_settings(tmp_path, "withbase")
    H.fake_base(settings.data_dir)
    with TestClient(create_app(settings)) as c:
        assert [m["kind"] for m in c.get("/models").json()] == ["base"]
        r = H.import_via_api(c, golden.bundle)
        assert r.status_code == 200, r.text
        assert len(c.get("/models").json()) == len(golden.texts["models"])
        assert [m["id"] for m in c.get("/models").json() if m["is_active"]] == [golden.texts["model_id"]]
    assert H.file_dump(settings) == golden.files


def test_a_different_base_model_file_in_the_way_is_a_conflict(golden, tmp_path, fake_ml):
    settings = H.fresh_settings(tmp_path, "otherbase")
    H.fake_base(settings.data_dir)
    (settings.data_dir / "models/base/test-base/base.mlmodel").write_bytes(b"someone else's model")
    with TestClient(create_app(settings)) as c:
        before = H.all_files(settings)
        r = H.import_via_api(c, golden.bundle)
        assert r.status_code == 409 and "different file" in r.json()["detail"]
        assert H.all_files(settings) == before and H.db_dump(settings)["pages"] == []


def test_import_refused_while_a_job_is_running(golden, fresh):
    client, settings = fresh
    conn = connect(settings.db_path)
    conn.execute("INSERT INTO jobs (type, status, created_at) VALUES ('predict', 'running', 'x')")
    conn.close()
    r = H.import_via_api(client, golden.bundle)
    assert r.status_code == 409 and "predict job is running" in r.json()["detail"]
    assert client.get("/health").status_code == 200  # maintenance mode was left again
    assert client.get("/pages").status_code == 200


# ---- damaged, tampered and hostile bundles -----------------------------------------------------------------------


def refused(client, settings, data: bytes, status: int, needle: str):
    before = H.all_files(settings), H.db_dump(settings)
    r = H.import_via_api(client, data)
    assert r.status_code == status, r.text
    assert needle in r.json()["detail"], r.json()["detail"]
    assert (H.all_files(settings), H.db_dump(settings)) == before  # exactly as it was
    assert not list(settings.tmp_dir.glob("import-*")) and not list(settings.data_dir.glob(".incoming-*"))
    assert client.get("/pages").status_code == 200  # server is out of maintenance mode


def test_corrupted_file_checksum(golden, fresh):
    client, settings = fresh
    entries = H.read_entries(golden.bundle)
    victim = next(p for p in entries if p.startswith("originals/"))
    data = bytearray(entries[victim])
    data[len(data) // 2] ^= 0xFF
    entries[victim] = bytes(data)
    refused(client, settings, H.write_zip(entries), 422, "checksum mismatch")


def test_corrupted_database_entry(golden, fresh):
    client, settings = fresh
    entries = H.read_entries(golden.bundle)
    entries["db/htr.sqlite"] = entries["db/htr.sqlite"][:-512]
    refused(client, settings, H.write_zip(entries), 422, "db/htr.sqlite")


def test_tampered_manifest_without_refreshing_its_checksum(golden, fresh):
    client, settings = fresh
    entries = H.read_entries(golden.bundle)
    tampered = H.with_manifest(entries, lambda m: m["counts"].__setitem__("pages", 99), refresh_checksum=False)
    refused(client, settings, H.write_zip(tampered), 422, "manifest does not match its own checksum")


def test_manifest_checksum_of_a_file_changed_and_refreshed(golden, fresh):
    client, settings = fresh
    entries = H.read_entries(golden.bundle)

    def lie(m):
        f = next(x for x in m["files"] if x["path"].startswith("working/"))
        f["sha256"] = "0" * 64

    refused(client, settings, H.write_zip(H.with_manifest(entries, lie)), 422, "checksum mismatch")


def test_manifest_counts_must_match_the_database(golden, fresh):
    client, settings = fresh
    entries = H.read_entries(golden.bundle)
    tampered = H.with_manifest(entries, lambda m: m["counts"].__setitem__("lines", 1))
    refused(client, settings, H.write_zip(tampered), 422, "manifest says 1")


def test_manifest_schema_version_must_match_the_database(golden, fresh):
    client, settings = fresh
    entries = H.read_entries(golden.bundle)
    tampered = H.with_manifest(entries, lambda m: m.__setitem__("schema_version", LATEST_VERSION - 1))
    refused(client, settings, H.write_zip(tampered), 422, "database is at schema")


def test_newer_schema_version_is_refused_with_a_clear_message(golden, fresh):
    client, settings = fresh
    entries = H.read_entries(golden.bundle)
    tampered = H.with_manifest(entries, lambda m: m.__setitem__("schema_version", LATEST_VERSION + 1))
    refused(client, settings, H.write_zip(tampered), 422, "made by a newer version of the app")


def test_newer_bundle_format_is_refused(golden, fresh):
    client, settings = fresh
    entries = H.read_entries(golden.bundle)
    tampered = H.with_manifest(entries, lambda m: m.__setitem__("bundle_format_version", 2))
    refused(client, settings, H.write_zip(tampered), 422, "newer than this app understands")


@pytest.mark.parametrize(
    "name", ["../evil.txt", "/abs/evil.txt", "originals/../../evil.txt", "originals\\..\\evil.txt", "C:/evil.txt"]
)
def test_zip_slip_and_absolute_entries(golden, fresh, name, tmp_path):
    client, settings = fresh
    entries = H.read_entries(golden.bundle)
    entries[name] = b"owned"
    refused(client, settings, H.write_zip(entries), 422, "unsafe" if "evil" in name else "")
    # also when the (re-checksummed) manifest vouches for the entry
    import hashlib

    def vouch(m):
        m["files"].append({"path": name, "size": 5, "sha256": hashlib.sha256(b"owned").hexdigest()})

    refused(client, settings, H.write_zip(H.with_manifest(entries, vouch)), 422, "unsafe")
    assert not (settings.data_dir.parent / "evil.txt").exists() and not (tmp_path / "evil.txt").exists()


def test_unexpected_top_level_name_and_unlisted_entry(golden, fresh):
    client, settings = fresh
    entries = H.read_entries(golden.bundle)
    entries["secrets/readme.txt"] = b"x"
    refused(client, settings, H.write_zip(entries), 422, "unexpected path")
    entries = H.read_entries(golden.bundle)
    entries["originals/extra.png"] = b"x"
    refused(client, settings, H.write_zip(entries), 422, "does not list")


def test_symlink_entries_are_rejected(golden, fresh):
    client, settings = fresh
    entries = H.read_entries(golden.bundle)
    link = zipfile.ZipInfo("originals/link.png", date_time=(2026, 1, 1, 0, 0, 0))
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    entries["originals/link.png"] = b"/etc/passwd"
    import hashlib

    def vouch(m):
        m["files"].append(
            {"path": "originals/link.png", "size": 11, "sha256": hashlib.sha256(b"/etc/passwd").hexdigest()}
        )

    refused(client, settings, H.write_zip(H.with_manifest(entries, vouch), {"originals/link.png": link}), 422, "link")


def test_missing_entry_listed_in_the_manifest(golden, fresh):
    client, settings = fresh
    entries = H.read_entries(golden.bundle)
    victim = next(p for p in entries if p.startswith("working/"))
    del entries[victim]
    refused(client, settings, H.write_zip(entries), 422, "is missing")


def test_database_paths_must_stay_inside_the_bundle(golden, fresh, tmp_path):
    client, settings = fresh
    entries = H.read_entries(golden.bundle)
    db = tmp_path / "evil.sqlite"
    db.write_bytes(entries["db/htr.sqlite"])
    conn = sqlite3.connect(db)
    conn.execute("UPDATE pages SET working_path = '/etc/passwd' WHERE id = (SELECT MIN(id) FROM pages)")
    conn.commit()
    conn.close()
    import hashlib

    entries["db/htr.sqlite"] = db.read_bytes()

    def fix(m):
        f = next(x for x in m["files"] if x["path"] == "db/htr.sqlite")
        f["size"], f["sha256"] = len(entries["db/htr.sqlite"]), hashlib.sha256(entries["db/htr.sqlite"]).hexdigest()

    refused(client, settings, H.write_zip(H.with_manifest(entries, fix)), 422, "unsafe path in the database")


def test_database_with_triggers_is_rejected(golden, fresh, tmp_path):
    client, settings = fresh
    entries = H.read_entries(golden.bundle)
    db = tmp_path / "t.sqlite"
    db.write_bytes(entries["db/htr.sqlite"])
    conn = sqlite3.connect(db)
    conn.execute("CREATE TRIGGER evil AFTER INSERT ON documents BEGIN DELETE FROM lines; END")
    conn.commit()
    conn.close()
    import hashlib

    entries["db/htr.sqlite"] = db.read_bytes()

    def fix(m):
        f = next(x for x in m["files"] if x["path"] == "db/htr.sqlite")
        f["size"], f["sha256"] = len(entries["db/htr.sqlite"]), hashlib.sha256(entries["db/htr.sqlite"]).hexdigest()

    refused(client, settings, H.write_zip(H.with_manifest(entries, fix)), 422, "unexpected objects")


@pytest.mark.parametrize("cut", [0.0, 0.05, 0.5, 0.97])
def test_truncated_zip(golden, fresh, cut):
    client, settings = fresh
    data = golden.bundle[: int(len(golden.bundle) * cut)] if cut else b"PK"
    refused(client, settings, data, 422, "not a readable project bundle")


def test_not_a_zip_and_wrong_kind_of_zip(golden, fresh):
    client, settings = fresh
    refused(client, settings, b"hello world, not a zip", 422, "not a readable project bundle")
    refused(client, settings, H.write_zip({"a.txt": b"x"}), 422, "manifest.json is missing")
    refused(client, settings, H.write_zip({"manifest.json": b"{}"}), 422, "not a project bundle")


# ---- failure in the middle of the import leaves the project untouched ---------------------------------------------


@pytest.mark.parametrize(
    "point", ["validated", "prepared", "placed:0", "placed:7", "before-db", "db-moved", "db-swapped"]
)
def test_a_failure_during_import_leaves_the_project_as_it_was(golden, fresh, monkeypatch, point):
    client, settings = fresh
    before = H.all_files(settings), H.db_dump(settings)

    def boom(p):
        if p == point:
            raise OSError(f"injected failure at {p}")

    monkeypatch.setattr(bundle_svc, "_fault", boom)
    with pytest.raises(OSError, match="injected"):
        # TestClient re-raises server errors; a real client would get a 500
        H.import_via_api(client, golden.bundle)
    monkeypatch.undo()
    assert (H.all_files(settings), H.db_dump(settings)) == before
    assert not list(settings.tmp_dir.glob("import-*")) and not list(settings.data_dir.glob(".incoming-*"))
    assert not list(settings.data_dir.glob(".pre-import-*"))
    assert not (settings.data_dir / "originals").exists() or not any((settings.data_dir / "originals").iterdir())
    # the app is fully usable, and a retry now succeeds
    assert client.get("/pages").json() == []
    assert H.import_via_api(client, golden.bundle).status_code == 200
    assert H.file_dump(settings) == golden.files


def test_failure_while_extracting_leaves_nothing_behind(golden, fresh, monkeypatch):
    client, settings = fresh
    before = H.all_files(settings), H.db_dump(settings)
    calls = []

    def boom(p):
        calls.append(p)
        if p.startswith("extracted:") and len(calls) == 6:
            raise OSError("disk full")

    monkeypatch.setattr(bundle_svc, "_fault", boom)
    with pytest.raises(OSError, match="disk full"):
        H.import_via_api(client, golden.bundle)
    monkeypatch.undo()
    assert (H.all_files(settings), H.db_dump(settings)) == before and not list(settings.tmp_dir.glob("import-*"))


# ---- older bundles are migrated ----------------------------------------------------------------------------------


def test_older_schema_bundle_is_migrated_on_import(tmp_path, fake_ml):
    """A project whose database is at schema 3 (before M6) exports and imports into a current project, with the
    restored database brought up to date by the normal migration runner."""
    old = H.fresh_settings(tmp_path, "old")
    old.data_dir.mkdir(parents=True)
    conn = connect(old.db_path)
    migrate(conn, MIGRATIONS[:3])
    for rel, data in (("originals/a_1.png", b"orig"), ("working/a_1.png", b"work"), ("pagexml/a_1.xml", b"<xml/>")):
        (old.data_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (old.data_dir / rel).write_bytes(data)
    conn.execute("INSERT INTO documents (id, name, imported_at) VALUES (1, 'old notes', 't')")
    conn.execute(
        "INSERT INTO pages (id, document_id, image_path, xml_path, width, height, working_path, original_filename, "
        "sha256, original_width, original_height, imported_at, segmented_at, status) "
        "VALUES (1, 1, 'originals/a_1.png', 'pagexml/a_1.xml', 10, 10, 'working/a_1.png', 'a.png', 'abc', 10, 10, 't', "
        "'t', 'segmented')"
    )
    conn.execute("INSERT INTO abbreviations (written, expansion, count) VALUES ('pt', 'patient', 0)")
    conn.close()
    out = tmp_path / "old.htrbundle"
    # the pagexml of a segmented page is rendered from the database, so give it a line
    conn = connect(old.db_path)
    conn.execute(
        "INSERT INTO lines (page_id, \"index\", baseline, polygon, text) VALUES (1, 0, '[[1,5],[9,5]]', "
        "'[[1,1],[9,1],[9,9],[1,9]]', 'pt seen')"
    )
    conn.close()
    info = bundle_svc.export_to_path(old, out)
    assert info["schema_version"] == 3
    assert json.loads(H.read_entries(out.read_bytes())["manifest.json"])["schema_version"] == 3

    settings = H.fresh_settings(tmp_path, "new")
    with TestClient(create_app(settings)) as c:
        r = H.import_via_api(c, out.read_bytes())
        assert r.status_code == 200, r.text
        assert r.json()["migrated_from"] == 3 and r.json()["schema_version"] == LATEST_VERSION
        ab = c.get("/abbreviations").json()
        assert [(a["written"], a["expansion"], a["count"], a["case_sensitive"]) for a in ab] == [
            ("pt", "patient", 1, False)
        ]
        assert c.get("/pages/1").json()["lines"][0]["text"] == "pt seen"
        assert c.get("/review/stats").status_code == 200
    conn = sqlite3.connect(settings.db_path)
    assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == LATEST_VERSION
    conn.close()


# ---- maintenance mode --------------------------------------------------------------------------------------------


def test_gate_refuses_new_requests_and_waits_for_running_ones():
    import threading
    import time

    from app.errors import Conflict
    from app.maintenance import Gate

    g = Gate()
    assert g.try_enter()  # the "import request" itself
    g.begin(timeout=1)
    assert g.try_enter() is False and g.try_enter(exempt=True) is True
    g.leave()
    with pytest.raises(Conflict, match="already running"):
        g.begin()
    g.end()
    assert g.try_enter()
    g.leave()
    g.leave()

    # an in-flight request that does not finish makes begin() give up, and maintenance is left again
    g2 = Gate()
    g2.try_enter()  # caller
    g2.try_enter()  # somebody else, still running
    with pytest.raises(Conflict, match="still running"):
        g2.begin(timeout=0.1)
    assert g2.maintenance is False
    t = threading.Timer(0.1, g2.leave)
    t.start()
    t0 = time.time()
    g2.begin(timeout=2)
    assert time.time() - t0 < 1.5
    g2.end()


def test_requests_during_import_get_503(golden, fresh, monkeypatch):
    client, settings = fresh
    seen = {}

    def peek(point):
        if point == "prepared":
            seen["during"] = client.app.state.gate.maintenance
            r = client.get("/pages")
            seen["status"], seen["detail"] = r.status_code, r.json()["detail"]
            seen["health"] = client.get("/health").status_code

    monkeypatch.setattr(bundle_svc, "_fault", peek)
    assert H.import_via_api(client, golden.bundle).status_code == 200
    assert seen["during"] is True and client.app.state.gate.maintenance is False
    assert seen["status"] == 503 and "being imported" in seen["detail"] and seen["health"] == 200
    assert client.get("/pages").status_code == 200
