"""Structured rotating log in data/logs, and the promise that it never holds the user's content."""

import json
import logging
import stat

from app import logging_setup
from tests.conftest import wait_job

SECRET_TEXT = "ZEBRA-SECRET-123 quarterly biopsy"
SECRET_ABBR = "xyzabbr"
SECRET_EXPANSION = "SECRETEXPANSION-nephrology"
SECRET_FILE = "patient_johndoe_scan.png"
SECRET_QUERY = "QUERYSECRET"


def read_logs(settings) -> str:
    files = sorted(settings.log_dir.glob("htr.log*"))
    assert files, "no log file was written"
    return "\n".join(f.read_text(encoding="utf-8") for f in files)


def test_log_file_is_structured_json_lines(client, settings):
    client.get("/pages/12345")  # a 404 is logged as a request failure
    logs = read_logs(settings)
    entries = [json.loads(line) for line in logs.splitlines()]
    assert entries and all({"ts", "level", "logger", "event"} <= e.keys() for e in entries)
    events = {e["event"] for e in entries}
    assert "startup" in events and "request" in events
    req = next(e for e in entries if e["event"] == "request")
    assert req["method"] == "GET" and req["path"] == "/pages/12345" and req["status"] == 404 and "ms" in req
    assert stat.S_IMODE((settings.log_dir / "htr.log").stat().st_mode) == 0o600


def test_no_transcribed_text_filenames_or_abbreviations_in_the_logs(client, api, settings, fake_ml, monkeypatch):
    doc = api.new_document("Secret notebook name")
    page = api.import_page(doc["id"], name=SECRET_FILE)
    seg = client.post(f"/pages/{page['id']}/segment").json()
    wait_job(client, seg["id"])
    lines = client.get(f"/pages/{page['id']}").json()["lines"]
    # edits: manual text, ground truth, a correction, ignore, merge, split, restore (undo)
    client.patch(f"/lines/{lines[0]['id']}", json={"text": SECRET_TEXT, "is_ground_truth": True})
    client.patch(f"/lines/{lines[1]['id']}", json={"text": f"{SECRET_ABBR} {SECRET_TEXT}"})
    client.patch(f"/lines/{lines[1]['id']}", json={"text": f"{SECRET_TEXT} again", "is_ground_truth": True})
    client.post("/abbreviations", json={"written": SECRET_ABBR, "expansion": SECRET_EXPANSION, "notes": SECRET_TEXT})
    client.get("/abbreviations", params={"q": SECRET_QUERY})
    client.post("/expand", json={"text": f"{SECRET_ABBR} {SECRET_TEXT}"})
    client.get("/export", params={"expanded": "true", "include_predicted": "true"})
    client.get(f"/pages/{page['id']}/text", params={"expanded": "true"})
    # predictions (the fake model reads every line as "hello wxrld")
    pred = wait_job(client, client.post("/jobs/predict").json()["id"])
    assert pred["status"] == "succeeded"
    after = client.get(f"/pages/{page['id']}").json()["lines"]
    assert any(ln["text_source"] == "predicted" for ln in after)
    client.post("/lines/merge", json={"line_ids": [lines[1]["id"], lines[2]["id"]]})
    client.post(f"/pages/{page['id']}/ignore-region", json={"x": 0, "y": 0, "w": 5000, "h": 5000})
    # failures: a 500 whose exception text contains user text, and a failing job that does too
    from app.services import abbreviations as abbr_svc

    def leaky(conn, *a, **k):
        raise RuntimeError(f"leaked {SECRET_TEXT} {SECRET_EXPANSION}")

    monkeypatch.setattr(abbr_svc, "list_abbreviations", leaky)
    assert client.get("/abbreviations", params={"q": SECRET_QUERY}).status_code == 500

    from app.pipeline import segment

    monkeypatch.setattr(segment, "run_blla", lambda img, device: (_ for _ in ()).throw(ValueError(SECRET_TEXT)))
    failed = wait_job(client, client.post(f"/pages/{page['id']}/segment", json={"force": True}).json()["id"])
    assert failed["status"] == "failed" and SECRET_TEXT in failed["error"]  # the user sees it, the log must not

    logs = read_logs(settings)
    for secret in (
        SECRET_TEXT,
        "ZEBRA",
        SECRET_ABBR,
        SECRET_EXPANSION,
        SECRET_FILE,
        "johndoe",
        "Secret notebook",
        SECRET_QUERY,
        "hello wxrld",
        "hello world",
        "quarterly",
    ):
        assert secret not in logs, f"{secret!r} leaked into the log"
    # but the useful, content-free facts are there
    entries = [json.loads(line) for line in logs.splitlines()]
    kinds = {(e["event"], e.get("job_type")) for e in entries}
    assert ("job finished", "predict") in kinds and ("job failed", "segment") in kinds
    boom = next(e for e in entries if e["event"] == "request failed")
    assert boom["code"] == "internal_error" and boom["error"].startswith("RuntimeError at ")
    failed_job = next(e for e in entries if e["event"] == "job failed")
    assert failed_job["error"].startswith("ValueError") and failed_job["job_id"] == failed["id"]


def test_exceptions_are_described_without_their_message():
    try:
        raise ValueError("secret words")
    except ValueError as e:
        text = logging_setup.describe_exception(e)
    assert text.startswith("ValueError") and "secret" not in text


def test_formatter_never_prints_exception_messages(tmp_path):
    handler = logging_setup.setup_logging(tmp_path / "logs")
    log = logging.getLogger("app.test")
    try:
        raise KeyError("private-note")
    except KeyError:
        log.error("oops", exc_info=True)
    handler.flush()
    logging_setup.teardown_logging()
    out = (tmp_path / "logs" / "htr.log").read_text()
    assert "private-note" not in out and "Traceback" not in out and "KeyError" in out


def test_logs_rotate(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_setup, "MAX_BYTES", 2000)
    monkeypatch.setattr(logging_setup, "BACKUPS", 2)
    handler = logging_setup.setup_logging(tmp_path / "logs")
    assert handler.maxBytes == 2000 and handler.backupCount == 2
    log = logging.getLogger("app.rotation")
    for i in range(400):
        logging_setup.log_event(log, logging.INFO, "tick", n=i)
    logging_setup.teardown_logging()
    files = sorted(p.name for p in (tmp_path / "logs").iterdir())
    assert files == ["htr.log", "htr.log.1", "htr.log.2"]  # bounded: current + 2 backups
    assert all(p.stat().st_size <= 2200 for p in (tmp_path / "logs").iterdir())


def test_unwritable_log_directory_does_not_stop_the_app(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert logging_setup.setup_logging(blocker / "logs") is None


def test_handler_is_removed_when_the_app_stops(settings, fake_blla):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(settings)):
        assert any(getattr(h, "_htr_file_handler", False) for h in logging.getLogger("app").handlers)
    assert not any(getattr(h, "_htr_file_handler", False) for h in logging.getLogger("app").handlers)
