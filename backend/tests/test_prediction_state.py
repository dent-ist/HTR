"""M7: predictions survive the repair operations that should give them back (undo / restore), merge and split follow
their documented rules, and ignore-region records review events like ignoring lines one by one."""

from app.db import connect
from tests.test_active_learning import PRED, events, predict
from tests.test_ml_jobs import client, project, project_template  # noqa: F401  (fixtures)

SNAPSHOT_KEYS = (
    "id",
    "baseline",
    "polygon",
    "text",
    "text_source",
    "confidence",
    "predicted_by_model_id",
    "predicted_text",
    "is_ground_truth",
    "ignored",
)


def snapshot(lines):
    return [{k: ln[k] for k in SNAPSHOT_KEYS} for ln in lines]


def page_lines(client, page_id):
    return client.get(f"/pages/{page_id}").json()["lines"]


def prediction_state(line):
    return {k: line[k] for k in ("text", "text_source", "confidence", "predicted_by_model_id", "predicted_text")}


def db_row(client, line_id):
    c = connect(client.app.state.settings.db_path)
    try:
        return dict(c.execute("SELECT * FROM lines WHERE id = ?", (line_id,)).fetchone())
    finally:
        c.close()


def predicted_page(client, project):
    predict(client)
    page = project["open_pages"][0]
    lines = page_lines(client, page["id"])
    assert all(ln["predicted_by_model_id"] and ln["predicted_text"] == PRED for ln in lines)
    return page["id"], lines


def restore(client, page_id, lines, force=False):
    r = client.post(f"/pages/{page_id}/restore", json={"lines": snapshot(lines), "force": force})
    assert r.status_code == 200, r.text
    return r.json()["lines"]


def test_line_objects_expose_the_prediction_snapshot(client, project):
    _, lines = predicted_page(client, project)
    assert lines[0]["text_source"] == "predicted" and lines[0]["predicted_text"] == PRED
    client.patch(f"/lines/{lines[0]['id']}", json={"text": "typed"})  # editing keeps the snapshot to compare against
    after = client.get(f"/lines/{lines[0]['id']}").json()
    assert after["text"] == "typed" and after["predicted_text"] == PRED and after["predicted_by_model_id"]


def test_merge_makes_an_unpredicted_line_and_undo_gives_the_predictions_back(client, project):
    page_id, before = predicted_page(client, project)
    ids = [ln["id"] for ln in before[:2]]
    merged = client.post("/lines/merge", json={"line_ids": ids}).json()["line"]
    assert merged["predicted_by_model_id"] is None and merged["predicted_text"] is None  # new line, nobody predicted it
    assert merged["text"] == f"{PRED} {PRED}" and merged["text_source"] == "manual"
    restored = restore(client, page_id, before)
    assert [ln["id"] for ln in restored] == [ln["id"] for ln in before]
    for old, new in zip(before, restored, strict=True):
        assert prediction_state(new) == prediction_state(old)
    # the restored prediction is real: reviewing it produces exactly one event, measured against the old prediction
    client.patch(f"/lines/{ids[0]}", json={"text": "hello world", "is_ground_truth": True})
    ev = [e for e in events(client) if e["line_id"] == ids[0]]
    assert len(ev) == 1 and ev[0]["action"] == "corrected" and ev[0]["predicted_text"] == PRED


def test_split_clears_the_prediction_on_the_left_half_and_undo_restores_it(client, project):
    page_id, before = predicted_page(client, project)
    target = before[1]
    xs = [p[0] for p in target["baseline"]]
    mid_x = (min(xs) + max(xs)) / 2
    r = client.post(f"/lines/{target['id']}/split", json={"x": mid_x, "y": target["baseline"][0][1]})
    assert r.status_code == 200, r.text
    left, right = r.json()["lines"]
    for half in (left, right):
        assert half["text"] == "" and half["text_source"] is None and half["confidence"] is None
        assert half["predicted_by_model_id"] is None and half["predicted_text"] is None
    row = db_row(client, left["id"])
    assert row["predicted_by_model_id"] is None and row["predicted_text"] is None
    assert events(client) == []  # dropping a prediction by splitting is not a review decision
    restored = restore(client, page_id, before, force=True)
    assert [prediction_state(ln) for ln in restored] == [prediction_state(ln) for ln in before]
    assert len(restored) == len(before)


def test_delete_then_undo_reinserts_the_line_with_its_prediction(client, project):
    page_id, before = predicted_page(client, project)
    victim = before[2]
    assert client.delete(f"/lines/{victim['id']}").status_code == 204
    restored = restore(client, page_id, before)
    back = next(ln for ln in restored if ln["id"] == victim["id"])  # same id: it was free
    assert prediction_state(back) == prediction_state(victim)
    assert db_row(client, back["id"])["predicted_text"] == PRED


def test_restore_ignores_a_model_that_no_longer_exists(client, project):
    page_id, before = predicted_page(client, project)
    snap = snapshot(before)
    snap[0]["predicted_by_model_id"] = 9999
    r = client.post(f"/pages/{page_id}/restore", json={"lines": snap, "force": True})
    assert r.status_code == 200, r.text
    first = r.json()["lines"][0]
    assert first["predicted_by_model_id"] is None and first["predicted_text"] is None
    assert first["text"] == PRED  # the text itself is untouched


def test_ignore_region_records_ignored_events_for_predicted_lines(client, project):
    page_id, lines = predicted_page(client, project)
    # make one line ground truth first: it is not a prediction any more and must not get an event
    gt = lines[0]["id"]
    client.patch(f"/lines/{gt}", json={"text": PRED, "is_ground_truth": True})
    assert [e["action"] for e in events(client)] == ["accepted"]
    page = client.get(f"/pages/{page_id}").json()
    everything = {"x": 0, "y": 0, "w": page["width"], "h": page["height"]}
    r = client.post(f"/pages/{page_id}/ignore-region", json=everything)
    assert r.status_code == 200, r.text
    assert set(r.json()["affected"]) == {ln["id"] for ln in lines}  # all three lines, the ground-truth one too
    ev = [e for e in events(client) if e["action"] == "ignored"]
    assert {e["line_id"] for e in ev} == {ln["id"] for ln in lines[1:]}  # only the predicted ones
    for e in ev:
        assert e["model_id"] and e["predicted_text"] == PRED and e["final_text"] is None and e["edit_distance"] is None
        assert db_row(client, e["line_id"])["predicted_by_model_id"] is None  # one prediction, one event
    client.post(f"/pages/{page_id}/ignore-region", json=everything)  # again: nothing new to ignore
    assert len([e for e in events(client) if e["action"] == "ignored"]) == len(ev)


def test_undoing_an_ignore_region_withdraws_its_events_and_rearms_the_predictions(client, project):
    page_id, before = predicted_page(client, project)
    page = client.get(f"/pages/{page_id}").json()
    everything = {"x": 0, "y": 0, "w": page["width"], "h": page["height"]}
    client.post(f"/pages/{page_id}/ignore-region", json=everything)
    assert len(events(client)) == 3
    restored = restore(client, page_id, before)
    assert [prediction_state(ln) for ln in restored] == [prediction_state(ln) for ln in before]
    assert all(not ln["ignored"] for ln in restored)
    assert events(client) == []  # the decisions the undone operation made are gone
    # and the lines can be reviewed again, once
    client.patch(f"/lines/{before[0]['id']}", json={"text": PRED, "is_ground_truth": True})
    assert [e["action"] for e in events(client)] == ["accepted"]
