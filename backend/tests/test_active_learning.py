"""M5: review queue, review events, correction-effort stats, chained rounds, retrain suggestion."""

import pytest

from app.db import connect
from tests.conftest import wait_job
from tests.ml_fakes import REF, FakeRecognizer, FakeTrainer
from tests.test_ml_jobs import client, project, project_template, start_train  # noqa: F401  (fixtures)

PRED = "hello wxrld"  # what the fake base model reads (1 edit away from REF)


def db(client):
    return connect(client.app.state.settings.db_path)


def predict(client, **body):
    job = client.post("/jobs/predict", json=body).json()
    done = wait_job(client, job["id"])
    assert done["status"] == "succeeded", done["error"]
    return done


def set_conf(client, project, confs):
    c = db(client)
    for ln, conf in zip(project["open"], confs, strict=True):
        c.execute("UPDATE lines SET confidence = ? WHERE id = ?", (conf, ln["id"]))
    c.close()


def events(client):
    c = db(client)
    rows = [dict(r) for r in c.execute("SELECT * FROM review_events ORDER BY id")]
    c.close()
    return rows


# ---- queue -------------------------------------------------------------------------------------------------------


def test_queue_is_lowest_confidence_first_with_ties_by_id(client, project):
    predict(client)
    set_conf(client, project, [0.5, 0.3, 0.9, 0.3, 0.7, 0.1])
    ids = [ln["id"] for ln in project["open"]]
    got = client.get("/review/next?limit=6").json()
    assert [i["line"]["id"] for i in got["items"]] == [
        ids[5],
        ids[1],
        ids[3],
        ids[0],
        ids[4],
        ids[2],
    ]  # tie 0.3: lower id first
    assert got["remaining"] == 6
    first = got["items"][0]
    assert first["page_id"] == project["open"][5]["page_id"] and first["crop_url"].endswith("/crop.png")
    assert first["predicted_text"] == PRED and first["confidence"] == 0.1 and first["model_id"] == 1


def test_exclude_limit_and_remaining(client, project):
    predict(client)
    set_conf(client, project, [0.5, 0.3, 0.9, 0.3, 0.7, 0.1])
    ids = [ln["id"] for ln in project["open"]]
    r = client.get(f"/review/next?exclude={ids[5]},{ids[1]}&limit=2").json()
    assert [i["line"]["id"] for i in r["items"]] == [ids[3], ids[0]] and r["remaining"] == 4
    assert client.get("/review/next").json()["items"][0]["line"]["id"] == ids[5]  # default limit 1
    assert client.get("/review/next?exclude=a,b").status_code == 422
    assert client.get("/review/next?exclude=" + ",".join(map(str, ids))).json() == {"items": [], "remaining": 0}


def test_queue_only_holds_predicted_open_lines(client, project):
    assert client.get("/review/next").json() == {"items": [], "remaining": 0}  # nothing predicted yet
    predict(client)
    ids = [ln["id"] for ln in project["open"]]
    client.patch(f"/lines/{ids[0]}", json={"ignored": True})
    client.patch(f"/lines/{ids[1]}", json={"text": "typed", "is_ground_truth": True})
    client.patch(f"/lines/{ids[2]}", json={"text": "typed, not gt"})  # manual text: no longer a prediction
    queue_ids = {i["line"]["id"] for i in client.get("/review/next?limit=50").json()["items"]}
    assert queue_ids == set(ids[3:])


def test_queue_overview_histogram(client, project):
    assert client.get("/review/queue").json()["total"] == 0
    predict(client)
    set_conf(client, project, [0.05, 0.12, 0.19, 0.5, 0.95, 1.0])
    q = client.get("/review/queue?lowest=2").json()
    assert q["total"] == 6 and len(q["buckets"]) == 10
    counts = [b["count"] for b in q["buckets"]]
    assert counts == [1, 2, 0, 0, 0, 1, 0, 0, 0, 2]  # 1.0 lands in the last bucket
    assert q["buckets"][0]["min"] == 0 and q["buckets"][9]["max"] == 1
    assert [i["confidence"] for i in q["lowest"]] == [0.05, 0.12]
    assert q["mean_confidence"] == pytest.approx(sum([0.05, 0.12, 0.19, 0.5, 0.95, 1.0]) / 6)


# ---- review events -----------------------------------------------------------------------------------------------


def test_accept_correct_and_ignore_are_recorded(client, project):
    predict(client)
    a, b, c, d = (ln["id"] for ln in project["open"][:4])
    assert client.patch(f"/lines/{a}", json={"text": PRED, "is_ground_truth": True}).status_code == 200
    assert client.patch(f"/lines/{b}", json={"text": REF, "is_ground_truth": True}).status_code == 200
    assert client.patch(f"/lines/{c}", json={"ignored": True}).status_code == 200
    ev = {e["line_id"]: e for e in events(client)}
    assert set(ev) == {a, b, c}
    assert (ev[a]["action"], ev[a]["edit_distance"], ev[a]["final_text"]) == ("accepted", 0, PRED)
    assert (ev[b]["action"], ev[b]["edit_distance"], ev[b]["final_text"]) == ("corrected", 1, REF)
    assert (ev[c]["action"], ev[c]["edit_distance"], ev[c]["final_text"]) == ("ignored", None, None)
    for e in ev.values():
        assert e["model_id"] == 1 and e["predicted_text"] == PRED and e["predicted_chars"] == 11
        assert e["confidence"] == pytest.approx(0.9) and e["reviewed_at"]
    # only reviewed predictions produced events; an untouched prediction did not
    assert d not in ev


def test_one_event_per_prediction_and_no_event_for_other_lines(client, project):
    predict(client)
    a = project["open"][0]["id"]
    client.patch(f"/lines/{a}", json={"text": PRED, "is_ground_truth": True})
    client.patch(f"/lines/{a}", json={"text": "later edit"})  # editing ground truth again is not a review
    client.patch(f"/lines/{a}", json={"is_ground_truth": False})
    client.patch(f"/lines/{a}", json={"is_ground_truth": True})
    assert len(events(client)) == 1
    gt = project["gt"][0]["id"]  # a line that never was a prediction
    client.patch(f"/lines/{gt}", json={"text": "changed"})
    client.patch(f"/lines/{gt}", json={"ignored": True})
    assert len(events(client)) == 1


def test_typing_over_a_prediction_then_marking_ground_truth_counts_as_corrected(client, project):
    predict(client)
    a = project["open"][0]["id"]
    client.patch(f"/lines/{a}", json={"text": "hello wxrld!"})  # saved as manual text first (not ground truth)
    assert events(client) == []
    client.patch(f"/lines/{a}", json={"is_ground_truth": True})
    (e,) = events(client)
    assert e["action"] == "corrected" and e["predicted_text"] == PRED and e["final_text"] == "hello wxrld!"
    assert e["edit_distance"] == 1


def test_event_is_written_in_the_same_transaction_as_the_line_change(client, project, monkeypatch):
    from app.services import sync

    predict(client)
    a = project["open"][0]["id"]

    def boom(*a_, **k):
        raise OSError("disk full")

    monkeypatch.setattr(sync, "write_page_xml", boom)
    failed = client.patch(f"/lines/{a}", json={"text": REF, "is_ground_truth": True})
    assert failed.status_code == 500 and failed.json()["code"] == "internal_error"  # a friendly error, no stack trace
    assert "disk full" not in failed.text and "Traceback" not in failed.text
    monkeypatch.undo()
    assert events(client) == []  # rolled back together with the line
    ln = next(x for x in client.get(f"/pages/{project['open'][0]['page_id']}").json()["lines"] if x["id"] == a)
    assert ln["text_source"] == "predicted" and not ln["is_ground_truth"] and ln["predicted_by_model_id"] == 1
    # and it works again afterwards
    assert client.patch(f"/lines/{a}", json={"text": REF, "is_ground_truth": True}).status_code == 200
    assert len(events(client)) == 1


def test_page_xml_stays_valid_with_predictions_and_reviews(client, project, settings):
    from app.pipeline import pagexml

    predict(client)
    a, b = project["open"][0], project["open"][1]
    client.patch(f"/lines/{a['id']}", json={"text": REF, "is_ground_truth": True})
    client.patch(f"/lines/{b['id']}", json={"ignored": True})
    for pg in project["open_pages"]:
        rel = db(client).execute("SELECT xml_path FROM pages WHERE id = ?", (pg["id"],)).fetchone()[0]
        seg, _ = pagexml.read_pagexml(settings.data_dir / rel)
        by = {int(x.id[1:]): x for x in seg.lines}
        if a["id"] in by:
            assert by[a["id"]].text == REF and by[a["id"]].is_ground_truth and by[a["id"]].text_source == "manual"
        if b["id"] in by:
            assert by[b["id"]].ignored
        others = [x for i, x in by.items() if i not in (a["id"], b["id"])]
        assert all(x.text == PRED and x.text_source == "predicted" and x.confidence == 0.9 for x in others)


def test_repredicting_overwrites_only_older_predictions(client, project):
    predict(client)  # base model
    a, b = project["open"][0]["id"], project["open"][1]["id"]
    client.patch(f"/lines/{a}", json={"text": "mine", "is_ground_truth": True})
    client.patch(f"/lines/{b}", json={"text": "typed, not gt"})
    tr = wait_job(client, start_train(client, epochs=1)["id"])["result"]
    predict(client, model_id=tr["model_id"])
    lines = {}
    for pg in project["open_pages"]:
        lines.update({x["id"]: x for x in client.get(f"/pages/{pg['id']}").json()["lines"]})
    assert lines[a]["text"] == "mine" and lines[a]["text_source"] == "manual"
    assert lines[b]["text"] == "typed, not gt"
    others = [x for i, x in lines.items() if i not in (a, b)]
    assert all(x["text"] == REF and x["predicted_by_model_id"] == tr["model_id"] for x in others)


# ---- stats -------------------------------------------------------------------------------------------------------


def test_stats_per_round_with_model_cer(client, project):
    assert client.get("/review/stats").json() == {"rounds": [], "total_reviewed": 0, "accepted_rate": None}
    predict(client)  # round 1: the base model, which reads PRED
    o = [ln["id"] for ln in project["open"]]
    client.patch(f"/lines/{o[0]}", json={"text": PRED, "is_ground_truth": True})  # accepted
    client.patch(f"/lines/{o[1]}", json={"text": REF, "is_ground_truth": True})  # corrected, 1 edit
    client.patch(f"/lines/{o[2]}", json={"text": "xhello wxrld xx", "is_ground_truth": True})  # corrected, 4 edits
    client.patch(f"/lines/{o[3]}", json={"ignored": True})
    tr = wait_job(client, start_train(client, epochs=1)["id"])["result"]  # model 2: reads REF
    predict(client)
    client.patch(f"/lines/{o[4]}", json={"text": REF, "is_ground_truth": True})  # accepted
    s = client.get("/review/stats").json()
    r1, r2 = s["rounds"]
    assert (r1["round"], r1["model_id"], r1["kind"]) == (1, 1, "base")
    assert (r1["reviewed"], r1["accepted"], r1["corrected"], r1["ignored"]) == (4, 1, 2, 1)
    assert r1["accepted_rate"] == pytest.approx(1 / 3)
    assert r1["mean_edits_per_line"] == pytest.approx(5 / 3) and r1["mean_edits_per_char"] == pytest.approx(5 / 33)
    assert r1["cer"] is None
    assert (r2["round"], r2["model_id"], r2["accepted"], r2["accepted_rate"]) == (2, tr["model_id"], 1, 1.0)
    assert r2["mean_edits_per_line"] == 0.0 and r2["cer"] == tr["cer"] == 0.0
    assert s["total_reviewed"] == 5 and s["accepted_rate"] == pytest.approx(2 / 4)


# ---- rounds ------------------------------------------------------------------------------------------------------


def test_round_trains_then_predicts_with_the_new_model(client, project):
    r = client.post("/jobs/round", json={"epochs": 1})
    assert r.status_code == 202 and r.json()["round"] is True and r.json()["type"] == "train"
    done = wait_job(client, r.json()["id"])
    assert done["status"] == "succeeded" and done["result"]["activated"] is True
    nxt = done["result"]["next_job_id"]
    pred = wait_job(client, nxt)
    assert pred["type"] == "predict" and pred["status"] == "succeeded"
    assert pred["payload"] == {"model_id": done["result"]["model_id"], "round_job_id": r.json()["id"]}
    assert pred["result"]["predicted"] == 6 and pred["result"]["model_id"] == done["result"]["model_id"]
    lines = [x for pg in project["open_pages"] for x in client.get(f"/pages/{pg['id']}").json()["lines"]]
    assert {x["text"] for x in lines} == {REF} and {x["predicted_by_model_id"] for x in lines} == {
        done["result"]["model_id"]
    }


def test_failed_round_step_leaves_a_clear_status_and_no_prediction_job(client, project):
    FakeTrainer.oom = True
    r = client.post("/jobs/round", json={"epochs": 1, "batch_size": 8}).json()
    done = wait_job(client, r["id"])
    assert done["status"] == "failed" and "smaller batch" in done["error"]
    assert client.get("/jobs?type=predict").json() == []
    FakeTrainer.oom = False
    # the predict step failing leaves the (successful) training intact
    ok = wait_job(client, client.post("/jobs/round", json={"epochs": 1}).json()["id"])
    assert ok["status"] == "succeeded"
    pj = client.get("/jobs?type=predict").json()[0]
    assert wait_job(client, pj["id"])["status"] == "succeeded"


def test_predict_step_failure_is_reported_on_the_predict_job(client, project, monkeypatch):
    from app.worker import ml_jobs

    real = ml_jobs.load_recognizer

    def flaky(engine, path, device="auto", batch_size=4):
        if path.name.endswith("base.mlmodel"):
            return real(engine, path, device, batch_size)  # the "before" scoring works
        if "pending" in path.name:
            return real(engine, path, device, batch_size)  # the "after" scoring works
        raise ValueError("cannot load the new model")

    monkeypatch.setattr(ml_jobs, "load_recognizer", flaky)
    train = wait_job(client, client.post("/jobs/round", json={"epochs": 1}).json()["id"])
    assert train["status"] == "succeeded"
    pred = wait_job(client, train["result"]["next_job_id"])
    assert pred["status"] == "failed" and "could not load model" in pred["error"]
    assert client.get(f"/models/{train['result']['model_id']}").status_code == 200  # the trained model is kept


def test_cancelling_the_training_step_ends_the_round(client, project):
    FakeTrainer.block_until_cancelled = True
    r = client.post("/jobs/round", json={"epochs": 3}).json()
    assert FakeTrainer.started.wait(10)
    client.post(f"/jobs/{r['id']}/cancel")
    assert wait_job(client, r["id"])["status"] == "cancelled"
    assert client.get("/jobs?type=predict").json() == []


def test_cancelling_the_predict_step(client, project):
    import threading

    gate, reached = threading.Event(), threading.Event()

    def hold_new_model(rec, n):
        if rec.path.name[0].isdigit():  # models/<id>.mlmodel: the predict step (scoring uses pending-*.mlmodel)
            reached.set()
            gate.wait(10)

    FakeRecognizer.on_predict = hold_new_model
    try:
        train = wait_job(client, client.post("/jobs/round", json={"epochs": 1}).json()["id"])
        pid = train["result"]["next_job_id"]
        assert reached.wait(10)
        client.post(f"/jobs/{pid}/cancel")
        gate.set()
        assert wait_job(client, pid)["status"] == "cancelled"
    finally:
        gate.set()
        FakeRecognizer.on_predict = None
    assert client.get(f"/models/{train['result']['model_id']}").json()["is_active"] is True  # training stands


def test_round_validation_errors(client, fake_ml):
    assert client.post("/jobs/round", json={}).status_code == 422  # nothing to train on
    assert client.post("/jobs/round", json={"epochs": 0}).status_code == 422


def test_second_round_while_training_is_a_409(client, project):
    FakeTrainer.block_until_cancelled = True
    r = client.post("/jobs/round", json={"epochs": 3}).json()
    assert FakeTrainer.started.wait(10)
    assert client.post("/jobs/round", json={}).status_code == 409
    client.post(f"/jobs/{r['id']}/cancel")
    wait_job(client, r["id"])


# ---- retrain suggestion ------------------------------------------------------------------------------------------


def add_gt(client, page_id, n):
    c = db(client)
    for i in range(n):
        c.execute(
            'INSERT INTO lines (page_id, "index", baseline, polygon, text, text_source, is_ground_truth) '
            "VALUES (?, ?, '[[0,0],[9,0]]', '[[0,0],[9,0],[9,9]]', 'x', 'manual', 1)",
            (page_id, 100 + i),
        )
    c.close()


def test_retrain_suggested_thresholds(client, project):
    page = project["open_pages"][0]["id"]
    s = client.get("/training/summary").json()
    assert s["retrain_suggested"] is False and s["ground_truth_lines"] == 24  # first round: needs about 100
    add_gt(client, page, 75)  # 99
    assert client.get("/training/summary").json()["retrain_suggested"] is False
    add_gt(client, page, 1)  # 100
    s = client.get("/training/summary").json()
    assert s["retrain_suggested"] is True and "100" in s["retrain_reason"]
    wait_job(client, start_train(client, epochs=1)["id"])
    s = client.get("/training/summary").json()
    assert s["new_since_last_round"] == 0 and s["retrain_suggested"] is False
    add_gt(client, page, 49)
    s = client.get("/training/summary").json()
    assert s["new_since_last_round"] == 49 and s["retrain_suggested"] is False
    add_gt(client, page, 1)
    s = client.get("/training/summary").json()
    assert s["new_since_last_round"] == 50 and s["retrain_suggested"] is True
    # suggested is advice only: training is still never blocked
    assert client.post("/jobs/train", json={"epochs": 1}).status_code == 202
