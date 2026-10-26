"""Train and predict jobs, the model registry and the API around them (fake engines; see tests/ml_fakes.py)."""

import threading

import pytest

from app.db import connect
from app.pipeline import pagexml
from app.pipeline.recognize import (
    KrakenRecognizer,
    OutOfMemory,
    Prediction,
    RecognitionError,
    line_confidence,
)
from app.services import training as T
from app.worker import ml_jobs
from tests.conftest import wait_job
from tests.ml_fakes import REF, FakeRecognizer, FakeTrainer
from tests.test_training import label, make_lines


@pytest.fixture(scope="session")
def project_template(tmp_path_factory):
    """A data directory with 10 imported, mock-segmented pages (3 lines each): the first 8 pages fully labeled as
    ground truth (REF), the last 2 untranscribed. Built once; every test that wants a project gets a copy, because
    importing a page costs about a second."""
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app
    from app.pipeline import segment
    from tests.conftest import _scan_bytes, fake_blla_factory

    data = tmp_path_factory.mktemp("template") / "data"
    settings = Settings.for_dir(data, worker_poll_seconds=0.02, device="cpu")
    mp = pytest.MonkeyPatch()
    mp.setattr(segment, "run_blla", fake_blla_factory(n_lines=3))
    try:
        with TestClient(create_app(settings)) as c:
            doc = c.post("/documents", json={"name": "notes"}).json()
            for i in range(10):
                up = c.post(
                    "/pages",
                    data={"document_id": str(doc["id"])},
                    files={"file": ("page.png", _scan_bytes("PNG", i + 1, 700, 900))},
                )
                assert up.status_code == 201, up.text
                pid = up.json()["id"]
                assert wait_job(c, c.post(f"/pages/{pid}/segment").json()["id"])["status"] == "succeeded"
                if i < 8:
                    for ln in c.get(f"/pages/{pid}").json()["lines"]:
                        label(c, ln["id"])
    finally:
        mp.undo()
    conn = connect(settings.db_path)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    return data


@pytest.fixture
def client(request, settings, fake_blla, project_template):
    """Like the shared `client`, but tests that ask for `project` start from a copy of the template project."""
    import shutil

    from fastapi.testclient import TestClient

    from app.main import create_app

    fake_blla(n_lines=3)
    if "project" in request.fixturenames:
        shutil.copytree(project_template, settings.data_dir, dirs_exist_ok=True)
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def project(client, fake_ml):
    """24 ground-truth lines (REF) on 8 pages plus 2 pages of 6 untranscribed lines."""
    pages = sorted(client.get("/pages").json(), key=lambda p: p["id"])
    detail = [client.get(f"/pages/{p['id']}").json() for p in pages]
    return {
        "gt": [ln for p in detail[:8] for ln in p["lines"]],
        "open": [ln for p in detail[8:] for ln in p["lines"]],
        "open_pages": detail[8:],
    }


def start_train(client, **body):
    r = client.post("/jobs/train", json=body)
    assert r.status_code == 202, r.text
    return r.json()


def db(client):
    return connect(client.app.state.settings.db_path)


# ---- train job ---------------------------------------------------------------------------------------------------


def test_train_job_registers_a_model_with_metrics_and_provenance(client, project, settings):
    job = start_train(client, epochs=3, batch_size=2, learning_rate=0.0005)
    assert job["type"] == "train" and job["warnings"]  # 24 lines < 100: warned, not refused
    done = wait_job(client, job["id"])
    assert done["status"] == "succeeded", done["error"]
    r = done["result"]
    assert r["phase"] == "done" and r["epochs_run"] == 3 and len(r["history"]) == 3
    assert r["base_cer"] == pytest.approx(1 / 11) and r["cer"] == 0.0 and r["wer"] == 0.0  # fake base: 1 edit / 11
    assert r["activated"] is True and r["held_out_lines"] >= 1
    assert r["history"][0]["val_accuracy"] is not None and r["history"][0]["train_loss"] is not None

    m = client.get(f"/models/{r['model_id']}").json()
    assert m["kind"] == "finetuned" and m["engine"] == "kraken" and m["is_active"] is True
    assert m["path"] == f"models/{m['id']}.mlmodel" and (settings.data_dir / m["path"]).is_file()
    assert not list((settings.data_dir / "models").glob("pending-*"))
    assert m["base_model"] == "test-base" and m["parent_model_id"] == 1
    assert m["train_lines"] + m["eval_lines"] == 24 and m["eval_lines"] == r["held_out_lines"]
    assert m["cer"] == 0.0 and m["wer"] == 0.0 and m["base_cer"] == pytest.approx(1 / 11) and m["trained_at"]
    assert m["hyperparameters"]["epochs"] == 3 and m["hyperparameters"]["batch_size"] == 2
    assert m["hyperparameters"]["learning_rate"] == 0.0005
    assert m["split_info"]["rule"] == "htr-split-v1" and m["split_info"]["held_out_lines"] == m["eval_lines"]
    assert (settings.data_dir / m["split_info"]["run_dir"] / "manifest.json").is_file()
    # the trainer got what was asked for, and the base model was only the start point
    assert FakeTrainer.last_request.batch_size == 2 and FakeTrainer.last_request.epochs == 3
    assert FakeTrainer.last_request.start_model.name == "base.mlmodel"
    # exactly one model is active; membership of every ground-truth line is recorded
    conn = db(client)
    assert [r_["id"] for r_ in conn.execute("SELECT id FROM models WHERE is_active = 1")] == [m["id"]]
    members = conn.execute("SELECT split, COUNT(*) FROM model_lines WHERE model_id = ? GROUP BY split", (m["id"],))
    assert {r_[0]: r_[1] for r_ in members} == {"train": m["train_lines"], "eval": m["eval_lines"]}


def test_progress_and_live_info_while_training(client, project):
    seen = []
    orig = FakeTrainer.train

    def watching(self, request, observer=None):
        out = orig(self, request, observer)
        job = client.get("/jobs").json()[0]
        seen.append((job["status"], job["progress"], job["result"]["phase"], job["result"]["history"]))
        return out

    FakeTrainer.train = watching
    try:
        done = wait_job(client, start_train(client, epochs=2)["id"])
    finally:
        FakeTrainer.train = orig
    assert done["status"] == "succeeded"
    status, progress, phase, history = seen[0]
    assert status == "running" and 0.1 < progress < 1 and phase == "training" and len(history) == 2


def test_second_round_starts_from_the_active_model_and_keeps_the_split(client, project, api):
    first = wait_job(client, start_train(client, epochs=1)["id"])["result"]
    conn = db(client)
    held1 = {r[0] for r in conn.execute("SELECT line_id FROM line_splits WHERE split = 'eval'")}
    for ln in project["open"][:6]:
        label(client, ln["id"])
    s = client.get("/training/summary").json()
    assert (
        s["ground_truth_lines"] == 30
        and s["new_since_last_round"] == 6
        and s["last_round"]["model_id"] == first["model_id"]
    )
    assert any("Only 6 new" in w for w in s["warnings"])  # retrain minimum is 50
    second = wait_job(client, start_train(client, epochs=1)["id"])["result"]
    assert FakeTrainer.last_request.start_model.name == f"{first['model_id']}.mlmodel"
    m2 = client.get(f"/models/{second['model_id']}").json()
    assert m2["parent_model_id"] == first["model_id"] and m2["base_model"] == "fine-tune 1"
    held2 = {r[0] for r in db(client).execute("SELECT line_id FROM line_splits WHERE split = 'eval'")}
    assert held1 <= held2  # held-out lines never move out
    ev1 = {
        r[0]
        for r in conn.execute(
            "SELECT line_id FROM model_lines WHERE model_id = ? AND split = 'eval'", (first["model_id"],)
        )
    }
    tr2 = {
        r[0]
        for r in conn.execute(
            "SELECT line_id FROM model_lines WHERE model_id = ? AND split = 'train'", (second["model_id"],)
        )
    }
    assert not ev1 & tr2  # a line held out in round 1 is never trained on later
    assert client.get("/training/summary").json()["new_since_last_round"] == 0


def test_explicit_base_model_id(client, project):
    done = wait_job(client, start_train(client, epochs=1)["id"])["result"]
    done2 = wait_job(client, start_train(client, epochs=1, base_model_id=1)["id"])["result"]  # back to the base model
    assert client.get(f"/models/{done2['model_id']}").json()["parent_model_id"] == 1
    assert done["model_id"] != done2["model_id"]


def test_worse_model_is_kept_but_not_activated(client, project, monkeypatch):
    # the fine-tuned fake is *worse* than its start model: the start model reads perfectly, the new one does not
    monkeypatch.setattr(FakeRecognizer, "text", lambda self: REF if "base" in str(self.path.parent) else "hxllo wxrld")
    done = wait_job(client, start_train(client, epochs=1)["id"])
    r = done["result"]
    assert r["activated"] is False and r["cer"] > r["base_cer"] == 0.0
    assert any("worse" in w for w in r["warnings"])
    assert client.get(f"/models/{r['model_id']}").json()["is_active"] is False
    assert [m["name"] for m in client.get("/models").json() if m["is_active"]] == ["test-base"]


def test_training_below_eight_lines_runs_without_a_cer(api, client, fake_ml):
    for ln in make_lines(api, 2)[:5]:
        label(client, ln["id"])
    job = start_train(client, epochs=1)
    assert any("fewer than 8" in w for w in job["warnings"])
    r = wait_job(client, job["id"])["result"]
    assert r["held_out_lines"] == 0 and r["cer"] is None and r["base_cer"] is None and r["activated"] is True
    m = client.get(f"/models/{r['model_id']}").json()
    assert m["eval_lines"] == 0 and m["cer"] is None and m["train_lines"] == 5


def test_cancel_running_training(client, project, settings):
    FakeTrainer.block_until_cancelled = True
    job = start_train(client, epochs=5)
    assert FakeTrainer.started.wait(10)
    r = client.post(f"/jobs/{job['id']}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "running" and r.json()["cancel_requested"] is True
    done = wait_job(client, job["id"])
    assert done["status"] == "cancelled" and done["finished_at"]
    assert client.get("/models").json()[0]["kind"] == "base" and len(client.get("/models").json()) == 1
    assert not list((settings.data_dir / "models").glob("pending-*"))
    # the worker is free again
    FakeTrainer.block_until_cancelled = False
    assert wait_job(client, start_train(client, epochs=1)["id"])["status"] == "succeeded"


def test_cancel_queued_training_and_cancel_of_finished_job(client, project):
    client.app.state.worker.stop()
    job = start_train(client, epochs=1)
    assert client.post(f"/jobs/{job['id']}/cancel").json()["status"] == "cancelled"
    assert client.post(f"/jobs/{job['id']}/cancel").status_code == 409


def test_out_of_memory_fails_the_job_with_advice_and_the_server_lives(client, project):
    FakeTrainer.oom = True
    done = wait_job(client, start_train(client, epochs=2, batch_size=8)["id"])
    assert done["status"] == "failed"
    assert "out of memory" in done["error"] and "smaller batch" in done["error"]
    assert client.get("/health").status_code == 200
    FakeTrainer.oom = False
    assert wait_job(client, start_train(client, epochs=1)["id"])["status"] == "succeeded"


def test_kraken_trainer_turns_cuda_oom_into_advice(monkeypatch, tmp_path):
    from app.pipeline.recognize import TrainRequest
    from app.pipeline.train import KrakenTrainer

    def boom(self, req, obs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(KrakenTrainer, "_train", boom)
    req = TrainRequest(tmp_path / "a", tmp_path / "b", [tmp_path], [tmp_path], tmp_path, batch_size=8)
    with pytest.raises(OutOfMemory, match="smaller batch size.*4"):
        KrakenTrainer().train(req)
    monkeypatch.setattr(KrakenTrainer, "_train", lambda self, r, o: (_ for _ in ()).throw(ValueError("other")))
    with pytest.raises(ValueError, match="other"):  # anything else is not disguised
        KrakenTrainer().train(req)


# ---- API validation ----------------------------------------------------------------------------------------------


def test_train_api_errors(client, fake_ml, api):
    r = client.post("/jobs/train", json={})
    assert r.status_code == 422 and "no usable ground-truth" in r.json()["detail"]
    for body in (
        {"epochs": 0},
        {"epochs": 1000},
        {"batch_size": 0},
        {"learning_rate": -1},
        {"bogus": 1},
        {"device": "tpu"},
    ):
        assert client.post("/jobs/train", json=body).status_code == 422, body
    assert client.post("/jobs/train", json={"base_model_id": 99}).status_code == 404


def test_train_without_any_model_is_a_clear_422(client, api):
    for ln in make_lines(api, 3)[:5]:
        label(client, ln["id"])
    r = client.post("/jobs/train", json={})
    assert r.status_code == 422 and "make base-model" in r.json()["detail"]


def test_missing_model_file_is_a_409(client, project, settings):
    assert client.get("/models").json()  # registered
    (settings.data_dir / "models" / "base" / "test-base" / "base.mlmodel").unlink()
    r = client.post("/jobs/train", json={})
    assert r.status_code == 409 and "missing on disk" in r.json()["detail"]


def test_only_one_training_job_at_a_time(client, project):
    FakeTrainer.block_until_cancelled = True
    job = start_train(client, epochs=3)
    assert FakeTrainer.started.wait(10)
    r = client.post("/jobs/train", json={})
    assert r.status_code == 409 and str(job["id"]) in r.json()["detail"]
    client.post(f"/jobs/{job['id']}/cancel")
    wait_job(client, job["id"])


def test_cancelling_a_running_segment_job_is_still_refused(client, fake_ml):
    from app.worker import queue

    client.app.state.worker.stop()
    conn = db(client)
    jid = queue.enqueue(conn, "segment", {"page_id": 1})["id"]
    queue.claim_next(conn)
    assert client.post(f"/jobs/{jid}/cancel").status_code == 409


# ---- models ------------------------------------------------------------------------------------------------------


def test_model_registry_endpoints(client, project):
    models = client.get("/models").json()
    assert [(m["id"], m["kind"], m["is_active"], m["license"]) for m in models] == [(1, "base", True, "Apache-2.0")]
    r = wait_job(client, start_train(client, epochs=1)["id"])["result"]
    models = client.get("/models").json()
    assert [m["id"] for m in models] == [r["model_id"], 1]  # newest first
    assert client.get("/models/1").json()["name"] == "test-base"
    assert client.get("/models/99").status_code == 404
    act = client.post("/models/1/activate")
    assert act.status_code == 200 and act.json()["is_active"] is True
    assert [m["id"] for m in client.get("/models").json() if m["is_active"]] == [1]
    assert client.post("/models/99/activate").status_code == 404


def test_base_model_picked_up_after_startup(client, settings):
    assert client.get("/models").json() == []
    import json as _json

    base = settings.data_dir / "models" / "base" / "late"
    base.mkdir(parents=True)
    (base / "m.mlmodel").write_bytes(b"x")
    (base / "BASE_MODEL.json").write_text(_json.dumps({"name": "late", "file": "m.mlmodel", "license": "MIT"}))
    models = client.get("/models").json()
    assert [m["name"] for m in models] == ["late"] and models[0]["is_active"]


# ---- predict job -------------------------------------------------------------------------------------------------


def line_map(client, pages):
    out = {}
    for p in pages:
        for ln in client.get(f"/pages/{p['id']}").json()["lines"]:
            out[ln["id"]] = ln
    return out


def all_pages(client):
    return client.get("/pages").json()


def test_predict_writes_text_and_confidence_only_where_allowed(client, project):
    ignored, typed = project["open"][0], project["open"][1]
    client.patch(f"/lines/{ignored['id']}", json={"ignored": True})
    client.patch(f"/lines/{typed['id']}", json={"text": "typed by hand"})  # manual text, not ground truth
    before = line_map(client, all_pages(client))

    job = client.post("/jobs/predict", json={}).json()
    assert any("base model" in w for w in job["warnings"])  # active model is the pretrained one
    done = wait_job(client, job["id"])
    assert done["status"] == "succeeded", done["error"]
    r = done["result"]
    assert r["lines"] == 4 and r["predicted"] == 4 and r["mean_confidence"] == 0.9 and r["model"] == "test-base"

    after = line_map(client, all_pages(client))
    for lid, ln in before.items():
        a = after[lid]
        if ln["is_ground_truth"]:
            assert (
                a["text"] == REF and a["text_source"] == "manual" and a["confidence"] is None and a["is_ground_truth"]
            )
        elif lid == ignored["id"]:
            assert a["text"] == "" and a["ignored"]
        elif lid == typed["id"]:
            assert a["text"] == "typed by hand" and a["text_source"] == "manual"
        else:
            assert a["text"] == "hello wxrld" and a["text_source"] == "predicted" and a["confidence"] == 0.9
            assert not a["is_ground_truth"]
    # the PAGE-XML on disk matches the database
    page = project["open_pages"][0]
    seg, _ = pagexml.read_pagexml(settings_xml(client, page["id"]))
    by_id = {int(ln.id[1:]): ln for ln in seg.lines}
    for lid, a in line_map(client, [page]).items():
        assert by_id[lid].text == a["text"] and by_id[lid].text_source == a["text_source"]
        assert by_id[lid].confidence == a["confidence"]


def settings_xml(client, page_id):
    conn = db(client)
    rel = conn.execute("SELECT xml_path FROM pages WHERE id = ?", (page_id,)).fetchone()[0]
    return client.app.state.settings.data_dir / rel


def test_predict_page_filter_model_choice_and_repeat(client, project):
    a, b = project["open_pages"]
    done = wait_job(client, client.post("/jobs/predict", json={"page_ids": [a["id"]]}).json()["id"])
    assert done["result"]["lines"] == 3
    lm = line_map(client, [a, b])
    assert all(ln["text_source"] == "predicted" for ln in lm.values() if ln["page_id"] == a["id"])
    assert all(ln["text"] == "" for ln in lm.values() if ln["page_id"] == b["id"])
    # a second run with the fine-tuned model replaces earlier predictions (they are not user work)
    tr = wait_job(client, start_train(client, epochs=1)["id"])["result"]
    done = wait_job(client, client.post("/jobs/predict", json={"model_id": tr["model_id"]}).json()["id"])
    assert done["result"]["lines"] == 6 and done["result"]["model_id"] == tr["model_id"]
    assert {ln["text"] for ln in line_map(client, [a, b]).values() if not ln["is_ground_truth"]} == {REF}


def test_predict_does_not_clobber_a_line_the_user_changed_meanwhile(client, project):
    target = project["open"][0]

    def user_edits_during_prediction(rec, n):
        c = connect(client.app.state.settings.db_path)
        c.execute(
            "UPDATE lines SET text = 'mine', text_source = 'manual', is_ground_truth = 1 WHERE id = ?", (target["id"],)
        )
        c.close()

    FakeRecognizer.on_predict = user_edits_during_prediction
    done = wait_job(client, client.post("/jobs/predict", json={}).json()["id"])
    FakeRecognizer.on_predict = None
    assert done["status"] == "succeeded" and done["result"]["changed_meanwhile"] >= 1
    ln = client.get(f"/pages/{target['page_id']}").json()["lines"]
    mine = next(x for x in ln if x["id"] == target["id"])
    assert mine["text"] == "mine" and mine["is_ground_truth"] and mine["text_source"] == "manual"


def test_predict_progress_cancel_and_resume(client, project):
    gate = threading.Event()
    reached = threading.Event()
    calls = []

    def slow(rec, n):
        calls.append(n)
        if len(calls) == 2:  # second page: wait for the cancel
            reached.set()
            gate.wait(10)

    FakeRecognizer.on_predict = slow
    job = client.post("/jobs/predict", json={}).json()
    assert reached.wait(10)
    mid = client.get(f"/jobs/{job['id']}").json()
    assert mid["status"] == "running" and 0 < mid["progress"] < 1 and mid["result"]["predicted"] == 3
    client.post(f"/jobs/{job['id']}/cancel")
    gate.set()
    done = wait_job(client, job["id"])
    assert done["status"] == "cancelled" and done["result"]["phase"] == "cancelled"
    lm = line_map(client, project["open_pages"])
    assert sum(ln["text_source"] == "predicted" for ln in lm.values()) == 3  # first page kept, second untouched
    gt_ids = {ln["id"] for ln in project["gt"]}
    gt_now = line_map(client, [{"id": pid} for pid in {ln["page_id"] for ln in project["gt"]}])
    assert all(gt_now[i]["text"] == REF and gt_now[i]["text_source"] == "manual" for i in gt_ids)
    FakeRecognizer.on_predict = None


def test_predict_with_no_candidate_lines_succeeds(api, client, fake_ml):
    for ln in api.segmented_page()["lines"]:
        label(client, ln["id"])
    done = wait_job(client, client.post("/jobs/predict", json={}).json()["id"])
    assert done["status"] == "succeeded" and done["result"]["lines"] == 0


def test_predict_api_errors(client, fake_ml, api):
    assert client.post("/jobs/predict", json={"page_ids": [99]}).status_code == 404
    assert client.post("/jobs/predict", json={"model_id": 99}).status_code == 404
    assert client.post("/jobs/predict", json={"batch_size": 0}).status_code == 422
    assert client.post("/jobs/predict", json={"bogus": 1}).status_code == 422


def test_predict_without_a_model_is_a_clear_422(client):
    r = client.post("/jobs/predict", json={})
    assert r.status_code == 422 and "make base-model" in r.json()["detail"]


def test_only_one_prediction_job_at_a_time(client, project):
    client.app.state.worker.stop()
    first = client.post("/jobs/predict", json={})
    assert first.status_code == 202
    assert client.post("/jobs/predict", json={}).status_code == 409


def test_predict_survives_a_model_that_cannot_be_loaded(client, project, monkeypatch):
    def broken(engine, path, device="auto", batch_size=4):
        raise ValueError("corrupt weights")

    monkeypatch.setattr(ml_jobs, "load_recognizer", broken)
    done = wait_job(client, client.post("/jobs/predict", json={}).json()["id"])
    assert done["status"] == "failed" and "could not load model" in done["error"] and "corrupt weights" in done["error"]


def test_predict_oom_becomes_a_failed_job_not_a_crash(client, project, monkeypatch):
    def oom(rec, n):
        raise OutOfMemory("out of memory while reading a line")

    FakeRecognizer.on_predict = oom
    done = wait_job(client, client.post("/jobs/predict", json={}).json()["id"])
    FakeRecognizer.on_predict = None
    assert done["status"] == "failed" and "out of memory" in done["error"]
    assert client.get("/health").status_code == 200


# ---- confidence, text hygiene, recognizer OOM fallback ------------------------------------------------------------


def test_line_confidence_is_a_geometric_mean_in_unit_interval():
    assert line_confidence([]) == 0.0
    assert line_confidence([1.0, 1.0]) == pytest.approx(1.0)
    assert line_confidence([0.5, 0.5]) == pytest.approx(0.5)
    assert line_confidence([1.0, 0.25]) == pytest.approx(0.5)
    assert line_confidence([0.9] * 9 + [0.5]) < sum([0.9] * 9 + [0.5]) / 10  # below the arithmetic mean
    for bad in ([0.0], [-3.0], [7.0], [0.0, 1.0]):
        assert 0.0 <= line_confidence(bad) <= 1.0


def test_clean_prediction_removes_control_characters_and_limits_length():
    assert ml_jobs.clean_prediction("a\tb\x00c\nd") == "abcd"
    assert ml_jobs.clean_prediction("é ü") == "é ü"
    assert len(ml_jobs.clean_prediction("x" * 5000)) == 2000


class _Net:
    def __init__(self):
        self.moved = []


def make_recognizer(batch_size, device="cuda:0"):
    """A KrakenRecognizer without a network: only the batching / OOM logic is exercised."""
    import torch

    rec = object.__new__(KrakenRecognizer)
    rec.batch_size, rec.device, rec.fell_back_to_cpu = batch_size, device, False
    rec._transforms = lambda im: torch.rand(1, 120, 40 + im.width) + 0.1
    rec._to = lambda dev: setattr(rec, "device", dev)
    return rec


def test_recognizer_halves_the_batch_on_oom_then_falls_back_to_cpu():
    from PIL import Image

    rec = make_recognizer(4)
    seen = []

    def run(batch):
        seen.append((rec.device, len(batch)))
        if rec.device.startswith("cuda") and len(batch) > 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")
        if rec.device.startswith("cuda"):
            raise RuntimeError("CUDA out of memory.")
        return [Prediction("ok", 0.8) for _ in batch]

    rec._run = run
    imgs = [Image.new("L", (30 + 10 * i, 20)) for i in range(5)]
    out = rec.predict(imgs)
    assert [p.text for p in out] == ["ok"] * 5
    assert rec.fell_back_to_cpu and rec.device == "cpu"
    assert seen[0] == ("cuda:0", 4) and ("cuda:0", 2) in seen and ("cuda:0", 1) in seen and seen[-1][0] == "cpu"


def test_recognizer_reraises_other_errors_and_keeps_order():
    from PIL import Image

    rec = make_recognizer(2, device="cpu")
    widths = []

    def run(batch):
        widths.append([t.shape[2] for t in batch])
        return [Prediction(str(t.shape[2]), 0.5) for t in batch]

    rec._run = run
    imgs = [Image.new("L", (w, 20)) for w in (10, 50, 30)]
    out = rec.predict(imgs)
    assert [p.text for p in out] == ["50", "90", "70"]  # original order (width + 40 from the fake transform)
    assert widths[0][0] >= widths[0][-1]  # widest first inside a batch
    rec._run = lambda batch: (_ for _ in ()).throw(ValueError("shape mismatch"))
    with pytest.raises(ValueError, match="shape mismatch"):
        rec.predict(imgs)


def test_recognizer_predict_can_be_cancelled_between_batches():
    from PIL import Image

    from app.pipeline.recognize import Cancelled

    rec = make_recognizer(1, device="cpu")
    n = []
    rec._run = lambda batch: n.append(1) or [Prediction("x", 1.0)]
    with pytest.raises(Cancelled):
        rec.predict([Image.new("L", (10, 10))] * 3, should_stop=lambda: len(n) >= 1)
    assert len(n) == 1


def test_unknown_engine_is_a_recognition_error():
    from app.pipeline.recognize import load_recognizer, make_trainer

    with pytest.raises(RecognitionError):
        load_recognizer("trocr", "x")
    with pytest.raises(RecognitionError):
        make_trainer("trocr")


def test_training_summary_matches_job_payload_warnings(client, project):
    job = start_train(client, epochs=1)
    assert job["warnings"] == client.get("/training/summary").json()["warnings"]
    assert wait_job(client, job["id"])["status"] == "succeeded"
    assert T.RECOMMENDED_FIRST_ROUND == 100 and T.RECOMMENDED_PER_RETRAIN == 50
