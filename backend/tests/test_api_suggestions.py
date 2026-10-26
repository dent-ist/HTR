"""Optional local-LLM abbreviation suggestions through the API: off by default, reviewable, never applied on their own.
The model is a fake server on 127.0.0.1; real-LLM behaviour is not tested here."""

import http.client
import json
import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import connect
from app.main import create_app
from tests.conftest import fake_blla_factory, wait_job
from tests.fake_llm import FakeLlm

TEXTS = ["pt c/o SOB since mon", "BP 120/80 f/u prn", "walked to the market today"]


@pytest.fixture
def fake_llm():
    with FakeLlm() as srv:
        yield srv


def build(settings, monkeypatch, texts=TEXTS):
    from app.pipeline import segment

    monkeypatch.setattr(segment, "run_blla", fake_blla_factory(n_lines=len(texts)))
    c = TestClient(create_app(settings))
    c.__enter__()
    return c


@pytest.fixture
def enabled(tmp_path, fake_llm, monkeypatch, scan_bytes):
    settings = Settings.for_dir(
        tmp_path / "data", worker_poll_seconds=0.02, device="cpu", llm_enabled=True, llm_url=fake_llm.url,
        llm_model="test-model",
    )  # fmt: skip
    c = build(settings, monkeypatch)
    doc = c.post("/documents", json={"name": "Private notebook"}).json()
    page = c.post(
        "/pages", data={"document_id": str(doc["id"])}, files={"file": ("secret_scan.png", scan_bytes("PNG", 1))}
    ).json()
    wait_job(c, c.post(f"/pages/{page['id']}/segment").json()["id"])
    lines = c.get(f"/pages/{page['id']}").json()["lines"]
    for ln, text in zip(lines, TEXTS, strict=True):
        c.patch(f"/lines/{ln['id']}", json={"text": text})
    c.page_id = page["id"]
    yield c
    c.__exit__(None, None, None)


def generate(c, **body):
    r = c.post("/abbreviations/suggestions/generate", json=body)
    assert r.status_code == 202, r.text
    return wait_job(c, r.json()["id"])


def page_snapshot(c):
    return json.dumps(c.get(f"/pages/{c.page_id}").json()["lines"], sort_keys=True)


def pending(c, status="pending"):
    return c.get(f"/abbreviations/suggestions?status={status}").json()["items"]


# ---- generating --------------------------------------------------------------------------------------------------


def test_generate_asks_about_abbreviation_like_tokens_and_stores_reviewable_suggestions(enabled, fake_llm):
    before = page_snapshot(enabled)
    job = generate(enabled)
    assert job["status"] == "succeeded", job["error"]
    assert job["result"]["model"] == "test-model" and job["result"]["asked"] == job["result"]["tokens"]
    asked = sorted(p.split('"')[1] for p in fake_llm.prompts)
    assert asked == sorted(["pt", "c/o", "SOB", "mon", "BP", "f/u", "prn"])  # not "walked", "120/80", "since", ...
    got = pending(enabled)
    assert len(got) == 2 * len(asked)
    first = next(s for s in got if s["written"] == "c/o")
    assert first["expansion"] in ("c/o expanded", "c/o alternative") and first["rationale"]
    assert first["context"] == "pt c/o SOB since mon" and first["model"] == "test-model"
    assert first["status"] == "pending" and first["abbreviation_id"] is None
    assert page_snapshot(enabled) == before  # nothing in the transcriptions changed
    assert enabled.get("/abbreviations").json() == []  # and nothing entered the dictionary on its own


def test_prompts_hold_only_the_token_its_line_and_capped_neighbours(enabled, fake_llm):
    generate(enabled)
    known = set(TEXTS)
    for prompt in fake_llm.prompts:
        for ln in prompt.splitlines()[1:]:
            label, _, text = ln.partition(": ")
            assert label in ("Line", "Line before", "Line after") and text in known, ln
        assert prompt.splitlines()[0].startswith("Abbreviation: ")
        assert len(prompt) < 400
    # the request carries nothing but the prompt: no page, document or file names, no ids
    wire = json.dumps([r["body"] for r in fake_llm.requests if r["body"]])
    for secret in ("Private notebook", "secret_scan", str(enabled.page_id) + '"'):
        assert secret not in wire
    c_o = next(p for p in fake_llm.prompts if 'Abbreviation: "c/o"' in p)
    assert c_o == 'Abbreviation: "c/o"\nLine: pt c/o SOB since mon\nLine after: BP 120/80 f/u prn'
    by_token = {p.split('"')[1]: p for p in fake_llm.prompts}
    assert "Line before" in by_token["BP"] and "Line after" in by_token["BP"]  # a middle line has both neighbours


def test_context_lines_can_be_turned_off_per_request(enabled, fake_llm):
    generate(enabled, tokens=["c/o"], context_lines=0)
    assert fake_llm.prompts == ['Abbreviation: "c/o"\nLine: pt c/o SOB since mon']


def test_predictions_are_never_sent(enabled, fake_llm):
    c = connect(enabled.app.state.settings.db_path)
    c.execute("UPDATE lines SET text_source = 'predicted', text = 'pq/r zz', is_ground_truth = 0 WHERE \"index\" = 2")
    c.close()
    generate(enabled)
    assert not any("pq/r" in p for p in fake_llm.prompts)  # only text the user wrote or verified leaves the database


def test_existing_dictionary_entries_are_not_asked_about_again(enabled, fake_llm):
    enabled.post("/abbreviations", json={"written": "c/o", "expansion": "complains of"})
    enabled.post("/abbreviations", json={"written": "BP", "expansion": "blood pressure"})
    enabled.post("/abbreviations", json={"written": "BP", "expansion": "bypass"})  # ambiguous: still worth asking
    generate(enabled)
    asked = {p.split('"')[1] for p in fake_llm.prompts}
    assert "c/o" not in asked and "BP" in asked


def test_second_run_only_asks_about_new_tokens(enabled, fake_llm):
    generate(enabled)
    n = len(fake_llm.prompts)
    again = generate(enabled)
    assert again["result"]["tokens"] == 0 and len(fake_llm.prompts) == n
    # explicit tokens are asked again, but a pending pair is not duplicated
    generate(enabled, tokens=["c/o"])
    assert len(fake_llm.prompts) == n + 1 and len([s for s in pending(enabled) if s["written"] == "c/o"]) == 2


def test_max_tokens_limits_the_run(enabled, fake_llm):
    job = generate(enabled, max_tokens=2)
    assert job["result"]["tokens"] == 2 == len(fake_llm.prompts)


# ---- review: accept, edit, reject --------------------------------------------------------------------------------


def test_accept_creates_a_dictionary_entry_and_changes_no_text(enabled):
    generate(enabled, tokens=["c/o"])
    before = page_snapshot(enabled)
    s = next(s for s in pending(enabled) if s["expansion"] == "c/o expanded")
    r = enabled.post(f"/abbreviations/suggestions/{s['id']}/accept")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["suggestion"]["status"] == "accepted"
    assert body["suggestion"]["abbreviation_id"] == body["abbreviation"]["id"]
    entries = enabled.get("/abbreviations").json()
    assert [(e["written"], e["expansion"]) for e in entries] == [("c/o", "c/o expanded")]
    assert entries[0]["count"] == 1
    assert page_snapshot(enabled) == before  # accepting only adds to the dictionary
    first_line = enabled.get(f"/pages/{enabled.page_id}").json()["lines"][0]["id"]
    assert enabled.get(f"/lines/{first_line}?expanded=true").json()["expanded_text"].startswith("pt c/o expanded")
    assert enabled.post(f"/abbreviations/suggestions/{s['id']}/accept").status_code == 409  # decided once
    assert enabled.post(f"/abbreviations/suggestions/{s['id']}/reject").status_code == 409
    assert s["id"] not in [x["id"] for x in pending(enabled)]
    assert [x["id"] for x in pending(enabled, "accepted")] == [s["id"]]


def test_edit_then_accept(enabled):
    generate(enabled, tokens=["prn"])
    s = pending(enabled)[0]
    r = enabled.post(
        f"/abbreviations/suggestions/{s['id']}/accept",
        json={"expansion": "as needed", "notes": "from the LLM, edited", "case_sensitive": True},
    )
    assert r.status_code == 200, r.text
    entry = r.json()["abbreviation"]
    assert (entry["written"], entry["expansion"], entry["notes"], entry["case_sensitive"]) == (
        "prn", "as needed", "from the LLM, edited", True
    )  # fmt: skip
    bad = pending(enabled)[0]
    assert enabled.post(f"/abbreviations/suggestions/{bad['id']}/accept", json={"expansion": "prn"}).status_code == 422
    assert pending(enabled)[0]["status"] == "pending"  # a refused accept leaves the suggestion open


def test_accepting_a_pair_that_is_already_in_the_dictionary_links_it(enabled):
    generate(enabled, tokens=["prn"])
    s = pending(enabled)[0]
    existing = enabled.post("/abbreviations", json={"written": "prn", "expansion": s["expansion"]}).json()
    r = enabled.post(f"/abbreviations/suggestions/{s['id']}/accept")
    assert r.status_code == 200 and r.json()["abbreviation"]["id"] == existing["id"]
    assert len(enabled.get("/abbreviations").json()) == 1


def test_reject_is_remembered_and_not_suggested_again(enabled, fake_llm):
    generate(enabled, tokens=["c/o"])
    s = next(s for s in pending(enabled) if s["expansion"] == "c/o expanded")
    r = enabled.post(f"/abbreviations/suggestions/{s['id']}/reject")
    assert r.status_code == 200 and r.json()["status"] == "rejected" and r.json()["resolved_at"]
    assert s["id"] not in [x["id"] for x in pending(enabled)]
    assert [x["id"] for x in pending(enabled, "rejected")] == [s["id"]]
    generate(enabled, tokens=["c/o"])  # asked again on purpose; the model repeats itself
    now = pending(enabled, "all")
    assert [x["expansion"] for x in now if x["written"] == "c/o" and x["status"] == "rejected"] == ["c/o expanded"]
    assert [x["expansion"] for x in now if x["written"] == "c/o" and x["status"] == "pending"] == ["c/o alternative"]
    # automatic discovery does not return to a token that already has suggestions
    n = len(fake_llm.prompts)
    generate(enabled)
    assert not any('"c/o"' in p for p in fake_llm.prompts[n:])
    assert enabled.get("/abbreviations").json() == []


def test_unknown_suggestion_ids_and_bodies(enabled):
    assert enabled.post("/abbreviations/suggestions/999/accept").status_code == 404
    assert enabled.post("/abbreviations/suggestions/999/reject").status_code == 404
    assert enabled.post("/abbreviations/suggestions/generate", json={"tokens": ["a b"]}).status_code == 422
    assert enabled.post("/abbreviations/suggestions/generate", json={"max_tokens": 0}).status_code == 422
    assert enabled.post("/abbreviations/suggestions/generate", json={"context_lines": 5}).status_code == 422
    assert enabled.post("/abbreviations/suggestions/generate", json={"surprise": 1}).status_code == 422
    assert enabled.get("/abbreviations/suggestions?status=weird").status_code == 422
    assert enabled.get("/abbreviations/1").status_code == 404  # the {id} route still works next to /suggestions


# ---- the job -----------------------------------------------------------------------------------------------------


def test_only_one_suggestion_job_at_a_time_and_it_can_be_cancelled(enabled, fake_llm):
    fake_llm.delay = 0.4
    r = enabled.post("/abbreviations/suggestions/generate", json={})
    assert r.status_code == 202 and r.json()["type"] == "suggest"
    job_id = r.json()["id"]
    second = enabled.post("/abbreviations/suggestions/generate", json={})
    assert second.status_code == 409 and second.json()["code"] == "job_active"
    deadline = time.time() + 10
    while not fake_llm.requests and time.time() < deadline:
        time.sleep(0.02)
    cancel = enabled.post(f"/jobs/{job_id}/cancel")
    assert cancel.status_code == 200 and cancel.json()["cancel_requested"] is True
    done = wait_job(enabled, job_id)
    assert done["status"] == "cancelled" and done["result"]["asked"] < done["result"]["tokens"]
    stored = {s["written"] for s in pending(enabled, "all")}
    assert len(stored) == done["result"]["asked"]  # what was answered before the cancel is kept
    assert enabled.post("/abbreviations/suggestions/generate", json={"tokens": ["c/o"]}).status_code == 202


def test_unreachable_model_fails_the_job_with_a_hint_and_keeps_earlier_results(enabled, fake_llm):
    generate(enabled, tokens=["c/o"])
    stored = len(pending(enabled))
    fake_llm.server.shutdown()
    fake_llm.server.server_close()
    job = generate(enabled, tokens=["prn"])
    assert job["status"] == "failed" and job["error_code"] == "llm_unreachable"
    assert "ollama serve" in job["error_hint"] and "Traceback" not in job["error"]
    assert len(pending(enabled)) == stored


def test_garbage_answers_produce_no_suggestions_but_do_not_fail(enabled, fake_llm):
    fake_llm.reply = lambda prompt: "I am a small model and I rambled."
    job = generate(enabled, tokens=["c/o", "prn"])
    assert job["status"] == "succeeded" and job["result"]["asked"] == 2 and job["result"]["suggestions_added"] == 0
    assert pending(enabled) == []


def test_missing_model_gets_a_pull_hint(enabled, fake_llm):
    fake_llm.status = 404
    job = generate(enabled, tokens=["prn"])
    assert job["status"] == "failed" and job["error_code"] == "llm_model_missing" and "ollama pull" in job["error_hint"]


# ---- off by default ----------------------------------------------------------------------------------------------


@pytest.fixture
def disabled(settings, monkeypatch, scan_bytes):
    assert settings.llm_enabled is False  # the default
    c = build(settings, monkeypatch)
    yield c
    c.__exit__(None, None, None)


def test_disabled_by_default_generate_is_refused_with_instructions(disabled):
    r = disabled.post("/abbreviations/suggestions/generate", json={})
    assert r.status_code == 409 and r.json()["code"] == "llm_disabled"
    assert "HTR_LLM_ENABLED" in r.json()["hint"]
    listing = disabled.get("/abbreviations/suggestions").json()
    assert listing == {"enabled": False, "backend": None, "items": []}
    assert disabled.get("/jobs").json() == []  # no job was even queued


def test_disabled_means_no_connection_attempts_at_all(disabled, monkeypatch, scan_bytes):
    attempts = []
    monkeypatch.setattr(http.client.HTTPConnection, "connect", lambda self: attempts.append(self.host))
    monkeypatch.setattr(http.client.HTTPConnection, "request", lambda self, *a, **k: attempts.append("request"))
    doc = disabled.post("/documents", json={"name": "d"}).json()
    page = disabled.post("/pages", data={"document_id": str(doc["id"])}, files={"file": ("a.png", scan_bytes())}).json()
    wait_job(disabled, disabled.post(f"/pages/{page['id']}/segment").json()["id"])
    for ln in disabled.get(f"/pages/{page['id']}").json()["lines"]:
        disabled.patch(f"/lines/{ln['id']}", json={"text": "pt c/o SOB"})
    disabled.post("/abbreviations/suggestions/generate", json={"tokens": ["c/o"]})
    disabled.get("/abbreviations/suggestions")
    disabled.get("/abbreviations")
    disabled.get("/health/details")
    # even a suggest job that reaches the worker by another route (a stale queue entry) must not connect
    from app.worker import queue

    conn = connect(disabled.app.state.settings.db_path)
    job = queue.enqueue(conn, "suggest", {"tokens": ["c/o"]})
    conn.close()
    disabled.app.state.worker.wake()
    done = wait_job(disabled, job["id"])
    assert done["status"] == "failed" and done["error_code"] == "llm_disabled"
    assert attempts == []


def test_old_suggestions_stay_reviewable_while_the_feature_is_off(tmp_path, monkeypatch):
    settings = Settings.for_dir(tmp_path / "d", worker_poll_seconds=0.02, device="cpu")
    c = build(settings, monkeypatch)
    try:
        conn = connect(settings.db_path)
        conn.execute("INSERT INTO suggestions (written, expansion, created_at) VALUES ('prn', 'as needed', 't')")
        conn.close()
        assert [s["expansion"] for s in pending(c)] == ["as needed"]
        r = c.post(f"/abbreviations/suggestions/{pending(c)[0]['id']}/accept")
        assert r.status_code == 200 and c.get("/abbreviations").json()[0]["expansion"] == "as needed"
    finally:
        c.__exit__(None, None, None)


def test_suggestions_travel_in_a_project_bundle(enabled, tmp_path, monkeypatch, fake_llm):
    generate(enabled, tokens=["c/o"])
    n = len(pending(enabled))
    bundle = enabled.get("/bundle/export")
    assert bundle.status_code == 200
    other = Settings.for_dir(tmp_path / "other", worker_poll_seconds=0.02, device="cpu")
    c = build(other, monkeypatch)
    try:
        r = c.post("/bundle/import", files={"file": ("p.htrbundle", bundle.content)})
        assert r.status_code == 200, r.text
        assert len(pending(c)) == n and r.json()["counts"]["suggestions"] == n
    finally:
        c.__exit__(None, None, None)
