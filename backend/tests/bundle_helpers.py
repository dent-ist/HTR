"""Shared helpers for the project-bundle tests: build a project through the API, snapshot everything that must survive
a round trip, and tamper with bundle zips."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sqlite3
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services import bundle as bundle_svc
from app.services import export as export_svc
from tests.conftest import fake_blla_factory, wait_job
from tests.ml_fakes import REF, FakeRecognizer, FakeTrainer

TABLES = [
    "documents",
    "pages",
    "lines",
    "models",
    "jobs",
    "abbreviations",
    "review_events",
    "model_lines",
    "line_splits",
]
FILE_DIRS = ("originals", "working", "pagexml", "models")


def fake_base(data_dir: Path) -> None:
    """The pretrained base model `make base-model` would download, as a fake."""
    base = data_dir / "models" / "base" / "test-base"
    base.mkdir(parents=True)
    (base / "base.mlmodel").write_bytes(b"fake base model")
    (base / "BASE_MODEL.json").write_text(
        json.dumps({"name": "test-base", "doi": "10.0/test", "license": "Apache-2.0", "file": "base.mlmodel"})
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def db_dump(settings: Settings) -> dict[str, list[dict]]:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        out = {t: [dict(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY 1, 2")] for t in TABLES}
        out["schema_version"] = [r["version"] for r in conn.execute("SELECT version FROM schema_version ORDER BY 1")]
        return out
    finally:
        conn.close()


def file_dump(settings: Settings) -> dict[str, str]:
    """relative path -> sha256 of every project file (scans, working copies, PAGE-XML, models)."""
    out = {}
    for d in FILE_DIRS:
        root = settings.data_dir / d
        for p in sorted(root.rglob("*")) if root.is_dir() else []:
            if p.is_file():
                out[p.relative_to(settings.data_dir).as_posix()] = sha(p)
    return out


def all_files(settings: Settings) -> dict[str, tuple[int, str]]:
    """Every file under the data directory except the lock file and the database (compare that with db_dump), for
    'untouched' checks."""
    out = {}
    for p in sorted(settings.data_dir.rglob("*")):
        if p.is_file() and not p.name.endswith(("-wal", "-shm")) and p.name not in (".htr-server.lock", "htr.db"):
            out[p.relative_to(settings.data_dir).as_posix()] = (p.stat().st_size, sha(p))
    return out


@dataclass
class Golden:
    bundle: bytes
    dump: dict
    files: dict
    settings: Settings
    texts: dict  # extra facts used by assertions


def start_job(client, path, body=None, expect="succeeded"):
    r = client.post(path, json=body or {})
    assert r.status_code == 202, r.text
    done = wait_job(client, r.json()["id"], timeout=60)
    assert done["status"] == expect, done
    return done


def build_project(client, settings) -> dict:
    """Work through the real API on a template project (24 ground-truth lines on 8 pages, 2 open pages): abbreviations,
    a train job (fake trainer) that registers a model, a predict job, review actions, ignores and a split."""
    facts: dict = {}
    for written, expansion in (
        ("pt", "patient"),
        ("c/o", "complains of"),
        ("SOB", "shortness of breath"),
        ("SOB", "son of a bitch"),
    ):
        assert client.post("/abbreviations", json={"written": written, "expansion": expansion}).status_code == 201
    pages = sorted(client.get("/pages").json(), key=lambda p: p["id"])
    detail = [client.get(f"/pages/{p['id']}").json() for p in pages]
    gt_lines = [ln for p in detail[:8] for ln in p["lines"]]
    open_lines = [ln for p in detail[8:] for ln in p["lines"]]
    # text with abbreviations on ground-truth lines; one ground-truth line ignored
    client.patch(f"/lines/{gt_lines[0]['id']}", json={"text": "pt c/o SOB"})
    client.patch(f"/lines/{gt_lines[1]['id']}", json={"text": "Pt, w/ pt seen"})
    client.patch(f"/lines/{gt_lines[2]['id']}", json={"ignored": True})
    facts["ignored_gt"] = gt_lines[2]["id"]
    # round 1: train (fake trainer) -> a fine-tuned, active model
    train = start_job(client, "/jobs/train", {"epochs": 2, "batch_size": 2})
    facts["model_id"] = train["result"]["model_id"]
    # predict the open lines, then review: accept, correct, ignore; leave two predicted lines open
    start_job(client, "/jobs/predict", {})
    ids = [ln["id"] for ln in open_lines]
    assert client.patch(f"/lines/{ids[0]}", json={"is_ground_truth": True}).status_code == 200  # accepted
    assert client.patch(f"/lines/{ids[1]}", json={"text": "hello sunshine", "is_ground_truth": True}).status_code == 200
    assert client.patch(f"/lines/{ids[2]}", json={"ignored": True}).status_code == 200
    facts["open_ids"] = ids
    # a segmentation repair on one of the open pages: split a still-open line
    r = client.get(f"/lines/{ids[4]}").json()
    xs = [p[0] for p in r["baseline"]]
    mid = (min(xs) + max(xs)) // 2
    y = r["baseline"][0][1]
    assert client.post(f"/lines/{ids[4]}/split", json={"x": mid, "y": y}).status_code == 200
    # the pages' PAGE-XML must be in sync with all of that before we export
    return facts


def fresh_settings(tmp_path: Path, name: str = "fresh") -> Settings:
    return Settings.for_dir(tmp_path / name, worker_poll_seconds=0.02, device="cpu")


def snapshot_all(settings: Settings) -> dict:
    return {"db": db_dump(settings), "files": file_dump(settings)}


def xml_in_sync(settings: Settings) -> list[str]:
    """Pages whose PAGE-XML file differs from what the database would render (should be empty)."""
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    bad = []
    try:
        for p in conn.execute("SELECT id, xml_path, segmented_at FROM pages"):
            if not p["segmented_at"]:
                continue
            want = export_svc.pagexml_bytes(conn, p["id"], include_predicted=True)
            got = (settings.data_dir / p["xml_path"]).read_bytes()
            if want != got:
                bad.append(p["xml_path"])
    finally:
        conn.close()
    return bad


# ---- zip surgery -------------------------------------------------------------------------------------------------


def read_entries(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return {n: z.read(n) for n in z.namelist()}


def write_zip(entries: dict[str, bytes], infos: dict[str, zipfile.ZipInfo] | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in entries.items():
            if infos and name in infos:
                z.writestr(infos[name], data)
            else:
                zi = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                zi.external_attr = 0o100644 << 16
                z.writestr(zi, data)
    return buf.getvalue()


def with_manifest(entries: dict[str, bytes], mutate, refresh_checksum: bool = True) -> dict[str, bytes]:
    """Copy of the entries with the manifest changed by `mutate(manifest)` (and its own checksum recomputed unless
    `refresh_checksum` is false)."""
    manifest = json.loads(entries["manifest.json"])
    mutate(manifest)
    if refresh_checksum:
        manifest["manifest_sha256"] = bundle_svc.manifest_checksum(manifest)
    out = dict(entries)
    out["manifest.json"] = json.dumps(manifest).encode()
    return out


def import_via_api(client, data: bytes):
    return client.post("/bundle/import", files={"file": ("project.htrbundle", data)})


def make_golden(tmp_path_factory, project_template) -> Golden:
    """Build the reference project once (module scoped fixtures call this): returns its bundle bytes and a dump."""
    mp = pytest.MonkeyPatch()
    from app.pipeline import segment
    from app.worker import ml_jobs

    FakeRecognizer.instances.clear()
    FakeRecognizer.on_predict = None
    FakeTrainer.reset()
    mp.setattr(segment, "run_blla", fake_blla_factory(n_lines=3))
    mp.setattr(ml_jobs, "load_recognizer", lambda engine, path, device="auto", batch_size=4: FakeRecognizer(path))
    mp.setattr(ml_jobs, "make_trainer", lambda engine: FakeTrainer())
    base = tmp_path_factory.mktemp("golden")
    settings = Settings.for_dir(base / "data", worker_poll_seconds=0.02, device="cpu")
    shutil.copytree(project_template, settings.data_dir)
    fake_base(settings.data_dir)
    try:
        with TestClient(create_app(settings)) as c:
            facts = build_project(c, settings)
            r = c.get("/bundle/export")
            assert r.status_code == 200, r.text
            bundle = r.content
            facts["export_text"] = c.get("/export", params={"expanded": "true", "ambiguous": "mark"}).text
            facts["review_stats"] = c.get("/review/stats").json()
            facts["models"] = c.get("/models").json()
            facts["jobs"] = c.get("/jobs?limit=500").json()
            facts["abbreviations"] = c.get("/abbreviations").json()
            facts["pages"] = {p["id"]: c.get(f"/pages/{p['id']}").json() for p in c.get("/pages").json()}
            facts["original"] = {pid: c.get(f"/pages/{pid}/original").content for pid in facts["pages"]}
            facts["crops"] = {
                ln["id"]: c.get(ln["crop_url"]).content for p in facts["pages"].values() for ln in p["lines"][:1]
            }
    finally:
        mp.undo()
    return Golden(bundle, snapshot_all(settings), file_dump(settings), settings, facts)


__all__ = ["REF", "Golden", "make_golden"]
