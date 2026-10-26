"""Segmentation repair endpoints: merge, split, baseline-only add, redraw, ignore region, reorder, restore."""

import pytest

from app.db import connect
from app.pipeline import pagexml
from app.services import repair as repair_svc
from app.services import sync


@pytest.fixture
def page(api):
    return api.segmented_page()


def get_page(client, pid):
    return client.get(f"/pages/{pid}").json()


def set_text(client, lid, text, gt=False):
    r = client.patch(f"/lines/{lid}", json={"text": text, **({"is_ground_truth": True} if gt else {})})
    assert r.status_code == 200, r.text
    return r.json()


def assert_contiguous(page):
    assert [x["index"] for x in page["lines"]] == list(range(len(page["lines"])))


def xml_matches_db(settings, pid, client):
    conn = connect(settings.db_path)
    try:
        seg, _ = pagexml.read_pagexml(sync.xml_file(conn, settings, pid))
    finally:
        conn.close()
    lines = get_page(client, pid)["lines"]
    assert [ln.text for ln in seg.lines] == [x["text"] or "" for x in lines]
    assert [ln.baseline for ln in seg.lines] == [[tuple(p) for p in x["baseline"]] for x in lines]
    assert [ln.ignored for ln in seg.lines] == [x["ignored"] for x in lines]


# ---- merge ----------------------------------------------------------------------------------------------------


def test_merge_joins_text_resets_ground_truth_and_renumbers(client, page, settings):
    a, b, c = (x["id"] for x in page["lines"])
    set_text(client, a, "hello", gt=True)
    set_text(client, b, "world", gt=True)
    r = client.post("/lines/merge", json={"line_ids": [b, a]})  # order of ids does not matter: reading order wins
    assert r.status_code == 200, r.text
    body = r.json()
    new = body["line"]
    assert new["text"] == "hello world" and new["text_source"] == "manual"
    assert new["is_ground_truth"] is False and new["ignored"] is False
    assert new["id"] not in (a, b, c) and sorted(body["removed_ids"]) == sorted([a, b])
    xs = [p[0] for p in new["baseline"]]
    assert xs == sorted(xs)
    w, h = page["width"], page["height"]
    assert all(0 <= x <= w and 0 <= y <= h for x, y in new["polygon"])
    pg = body["page"]
    assert [x["id"] for x in pg["lines"]] == [new["id"], c]
    assert_contiguous(pg)
    xml_matches_db(settings, page["id"], client)


def test_merge_non_adjacent_lands_at_first_position(client, page):
    a, b, c = (x["id"] for x in page["lines"])
    body = client.post("/lines/merge", json={"line_ids": [c, a]}).json()
    assert [x["id"] for x in body["page"]["lines"]] == [body["line"]["id"], b]
    assert body["line"]["text"] == "" and body["line"]["text_source"] is None


def test_merge_ignores_empty_texts_when_joining(client, page):
    a, b, _ = (x["id"] for x in page["lines"])
    set_text(client, b, "only")
    assert client.post("/lines/merge", json={"line_ids": [a, b]}).json()["line"]["text"] == "only"


def test_merge_validation(client, api, page):
    a, b, _ = (x["id"] for x in page["lines"])
    assert client.post("/lines/merge", json={"line_ids": [a]}).status_code == 422
    assert client.post("/lines/merge", json={"line_ids": [a, a]}).status_code == 422
    assert client.post("/lines/merge", json={"line_ids": [a, 99999]}).status_code == 404
    other = api.segmented_page()
    assert client.post("/lines/merge", json={"line_ids": [a, other["lines"][0]["id"]]}).status_code == 422
    set_text(client, a, "x" * 1500)
    set_text(client, b, "y" * 1500)
    assert client.post("/lines/merge", json={"line_ids": [a, b]}).status_code == 422
    assert len(get_page(client, page["id"])["lines"]) == 3  # nothing changed


# ---- split ----------------------------------------------------------------------------------------------------


def test_split_makes_two_lines_and_returns_old_text(client, page, settings):
    first = page["lines"][0]
    set_text(client, first["id"], "left right", gt=True)
    x0, x1 = first["baseline"][0][0], first["baseline"][-1][0]
    cx = (x0 + x1) // 2
    y = first["baseline"][0][1]
    r = client.post(f"/lines/{first['id']}/split", json={"x": cx, "y": y})
    assert r.status_code == 200, r.text
    body = r.json()
    left, right = body["lines"]
    assert left["id"] == first["id"] and right["id"] != first["id"]
    assert body["old_text"] == "left right"
    for half in (left, right):
        assert half["text"] == "" and half["text_source"] is None and half["is_ground_truth"] is False
    assert left["baseline"][-1][0] == cx == right["baseline"][0][0]
    assert max(p[0] for p in left["polygon"]) <= cx <= min(p[0] for p in right["polygon"])
    pg = body["page"]
    assert [x["id"] for x in pg["lines"]][:2] == [left["id"], right["id"]] and len(pg["lines"]) == 4
    assert_contiguous(pg)
    xml_matches_db(settings, page["id"], client)


def test_split_without_text_returns_empty_old_text_and_inherits_ignored(client, page):
    ln = page["lines"][1]
    client.patch(f"/lines/{ln['id']}", json={"ignored": True})
    body = client.post(f"/lines/{ln['id']}/split", json={"x": 300, "y": ln["baseline"][0][1]}).json()
    assert body["old_text"] == ""
    assert [x["ignored"] for x in body["lines"]] == [True, True]


@pytest.mark.parametrize("x", [0, 3, 10**6])
def test_split_outside_baseline_is_rejected(client, page, x):
    ln = page["lines"][0]
    r = client.post(f"/lines/{ln['id']}/split", json={"x": x, "y": 10})
    assert r.status_code == 422
    assert len(get_page(client, page["id"])["lines"]) == 3


def test_split_unknown_line_and_bad_point(client, page):
    assert client.post("/lines/9999/split", json={"x": 100, "y": 10}).status_code == 404
    ln = page["lines"][0]
    assert client.post(f"/lines/{ln['id']}/split", json={"x": 300, "y": -5}).status_code == 422
    assert client.post(f"/lines/{ln['id']}/split", json={"x": 300}).status_code == 422


# ---- baseline-only add / redraw ---------------------------------------------------------------------------------


def test_add_line_from_baseline_only_uses_median_extent(client, page, settings):
    # the fake segmenter's polygons reach 15 px above and 10 px below the baseline
    y = 600
    r = client.post(f"/pages/{page['id']}/lines", json={"baseline": [[100, y], [400, y + 4]]})
    assert r.status_code == 201, r.text
    ln = r.json()
    ys = [p[1] for p in ln["polygon"]]
    assert min(ys) == y - 15 and max(ys) == y + 4 + 10
    assert min(p[0] for p in ln["polygon"]) == 100 and max(p[0] for p in ln["polygon"]) == 400
    assert ln["index"] == 3
    xml_matches_db(settings, page["id"], client)


def test_baseline_only_orients_and_clamps_to_page(client, page):
    r = client.post(f"/pages/{page['id']}/lines", json={"baseline": [[300, 5], [100, 5]]})
    assert r.status_code == 201
    ln = r.json()
    assert ln["baseline"][0][0] == 100  # stored left to right
    assert min(p[1] for p in ln["polygon"]) == 0  # clamped at the page top


def test_baseline_only_fallback_uses_page_line_height(client, api):
    page = api.segmented_page()
    for x in page["lines"]:
        client.delete(f"/lines/{x['id']}")
    lh = client.get(f"/pages/{page['id']}").json()["line_height_px"]
    ln = client.post(f"/pages/{page['id']}/lines", json={"baseline": [[100, 500], [300, 500]]}).json()
    ys = [p[1] for p in ln["polygon"]]
    assert abs(min(ys) - (500 - 0.7 * lh)) <= 1
    assert abs(max(ys) - (500 + 0.3 * lh)) <= 1


@pytest.mark.parametrize("baseline", [[[1, 1]], [], [[1, 1], [99999, 1]], [[100, 100], [100, 100]]])
def test_baseline_only_validation(client, page, baseline):
    r = client.post(f"/pages/{page['id']}/lines", json={"baseline": baseline})
    assert r.status_code == 422
    assert len(get_page(client, page["id"])["lines"]) == 3


def test_redraw_baseline_regenerates_polygon(client, page, settings):
    ln = page["lines"][0]
    new_base = [[80, 200], [300, 210], [500, 205]]
    r = client.patch(f"/lines/{ln['id']}", json={"baseline": new_base, "regenerate_polygon": True})
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["baseline"] == new_base
    ys = [p[1] for p in got["polygon"]]
    assert min(ys) < 200 < max(ys) and min(ys) >= 200 - 16
    xml_matches_db(settings, page["id"], client)


def test_regenerate_conflicts_with_polygon(client, page):
    ln = page["lines"][0]
    r = client.patch(f"/lines/{ln['id']}", json={"regenerate_polygon": True, "polygon": [[1, 1], [9, 1], [9, 5]]})
    assert r.status_code == 422


def test_regenerate_without_new_baseline_uses_current(client, page):
    ln = page["lines"][0]
    client.patch(f"/lines/{ln['id']}", json={"polygon": [[60, 60], [70, 60], [70, 70]]})
    got = client.patch(f"/lines/{ln['id']}", json={"regenerate_polygon": True}).json()
    assert len(got["polygon"]) >= 4 and got["baseline"] == ln["baseline"]


# ---- ignore region -------------------------------------------------------------------------------------------


def test_ignore_region_rectangle(client, page, settings):
    l1, l2, l3 = page["lines"]
    y = l2["baseline"][0][1]
    box = {"x": 0, "y": y - 40, "w": page["width"], "h": 80}
    r = client.post(f"/pages/{page['id']}/ignore-region", json=box)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["affected"] == [l2["id"]]
    flags = {x["id"]: x["ignored"] for x in body["page"]["lines"]}
    assert flags == {l1["id"]: False, l2["id"]: True, l3["id"]: False}
    assert body["page"]["ignored_count"] == 1
    xml_matches_db(settings, page["id"], client)
    # idempotent: already-ignored lines are not reported again
    assert client.post(f"/pages/{page['id']}/ignore-region", json=box).json()["affected"] == []


def test_ignore_region_needs_half_the_area(client, page):
    l1 = page["lines"][0]
    xs = [p[0] for p in l1["polygon"]]
    ys = [p[1] for p in l1["polygon"]]
    width = max(xs) - min(xs)
    under = {"x": min(xs), "y": min(ys) - 5, "w": width * 0.4, "h": 200}
    assert client.post(f"/pages/{page['id']}/ignore-region", json=under).json()["affected"] == []
    over = {"x": min(xs), "y": min(ys) - 5, "w": width * 0.6, "h": 200}
    assert client.post(f"/pages/{page['id']}/ignore-region", json=over).json()["affected"] == [l1["id"]]


def test_ignore_region_polygon_and_validation(client, page):
    pid = page["id"]
    l3 = page["lines"][2]
    cy = l3["baseline"][0][1]
    quad = [[0, cy - 100], [page["width"], cy - 100], [page["width"], cy + 100], [0, cy + 100]]
    assert client.post(f"/pages/{pid}/ignore-region", json={"polygon": quad}).json()["affected"] == [l3["id"]]
    for bad in (
        {},
        {"x": 1, "y": 1, "w": 0, "h": 5},
        {"x": 1, "y": 1, "w": 5},
        {"x": 1, "y": 1, "w": 5, "h": 5, "polygon": quad},
        {"x": -5, "y": 1, "w": 10, "h": 5},
        {"x": 1, "y": 1, "w": 10**6, "h": 5},
        {"polygon": [[1, 1], [5, 5]]},
    ):
        assert client.post(f"/pages/{pid}/ignore-region", json=bad).status_code == 422, bad
    assert client.post("/pages/9999/ignore-region", json={"x": 1, "y": 1, "w": 5, "h": 5}).status_code == 404


# ---- reorder --------------------------------------------------------------------------------------------------


def test_reorder_full_permutation(client, page, settings):
    ids = [x["id"] for x in page["lines"]]
    r = client.post(f"/pages/{page['id']}/reorder", json={"line_ids": [ids[2], ids[0], ids[1]]})
    assert r.status_code == 200
    assert [x["id"] for x in r.json()["lines"]] == [ids[2], ids[0], ids[1]]
    assert_contiguous(r.json())
    xml_matches_db(settings, page["id"], client)


@pytest.mark.parametrize(
    "pick",
    [lambda i: i[:2], lambda i: [*i, i[0]], lambda i: [i[0], i[0], i[1]], lambda i: [*i[:2], 999]],
)
def test_reorder_rejects_non_permutations(client, page, pick):
    ids = [x["id"] for x in page["lines"]]
    assert client.post(f"/pages/{page['id']}/reorder", json={"line_ids": pick(ids)}).status_code == 422
    assert [x["id"] for x in get_page(client, page["id"])["lines"]] == ids


# ---- restore (undo) -------------------------------------------------------------------------------------------


def snapshot(page):
    keys = ("id", "baseline", "polygon", "text", "text_source", "confidence", "is_ground_truth", "ignored")
    return [{k: x[k] for k in keys} for x in page["lines"]]


def test_restore_undoes_merge_keeping_ids_and_text(client, page, settings):
    a, b, _ = (x["id"] for x in page["lines"])
    set_text(client, a, "one", gt=True)
    set_text(client, b, "two", gt=True)
    before = get_page(client, page["id"])
    client.post("/lines/merge", json={"line_ids": [a, b]})
    r = client.post(f"/pages/{page['id']}/restore", json={"lines": snapshot(before)})
    assert r.status_code == 200, r.text
    after = r.json()
    assert [x["id"] for x in after["lines"]] == [x["id"] for x in before["lines"]]  # ids recreated
    keys = ("text", "is_ground_truth", "baseline", "polygon", "index")
    for x, y in zip(after["lines"], before["lines"], strict=True):
        assert {k: x[k] for k in keys} == {k: y[k] for k in keys}
    assert after["status"] == before["status"]
    xml_matches_db(settings, page["id"], client)


def test_restore_refuses_to_drop_typed_text_unless_forced(client, page):
    a, b, _ = (x["id"] for x in page["lines"])
    before = get_page(client, page["id"])  # no text yet
    merged = client.post("/lines/merge", json={"line_ids": [a, b]}).json()["line"]
    set_text(client, merged["id"], "typed after the merge", gt=True)
    r = client.post(f"/pages/{page['id']}/restore", json={"lines": snapshot(before)})
    assert r.status_code == 409 and "typed after the merge" in r.json()["detail"]
    cur = get_page(client, page["id"])
    assert cur["lines"][0]["text"] == "typed after the merge" and len(cur["lines"]) == 2  # untouched
    forced = client.post(f"/pages/{page['id']}/restore", json={"lines": snapshot(before), "force": True})
    assert forced.status_code == 200 and len(forced.json()["lines"]) == 3


def test_restore_undoes_delete_and_split(client, page):
    ids = [x["id"] for x in page["lines"]]
    set_text(client, ids[1], "keep me", gt=True)
    before = get_page(client, page["id"])
    client.delete(f"/lines/{ids[1]}")
    r = client.post(f"/pages/{page['id']}/restore", json={"lines": snapshot(before)})
    assert r.status_code == 200
    restored = r.json()["lines"][1]
    assert restored["id"] == ids[1] and restored["text"] == "keep me" and restored["is_ground_truth"]
    # split clears text, so undoing it must give the text back
    client.post(f"/lines/{restored['id']}/split", json={"x": 300, "y": restored["baseline"][0][1]})
    r = client.post(f"/pages/{page['id']}/restore", json={"lines": snapshot(before)})
    assert r.status_code == 200 and r.json()["lines"][1]["text"] == "keep me"
    assert len(r.json()["lines"]) == 3


def test_restore_validation_and_rollback(client, page):
    snap = snapshot(get_page(client, page["id"]))
    snap[0]["is_ground_truth"] = True  # no text
    assert client.post(f"/pages/{page['id']}/restore", json={"lines": snap}).status_code == 422
    snap = snapshot(get_page(client, page["id"]))
    snap[0]["polygon"] = [[1, 1], [99999, 5], [5, 5]]
    assert client.post(f"/pages/{page['id']}/restore", json={"lines": snap}).status_code == 422
    assert client.post("/pages/9999/restore", json={"lines": []}).status_code == 404
    assert len(get_page(client, page["id"])["lines"]) == 3


def test_restore_does_not_steal_ids_from_other_pages(client, api, page):
    other = api.segmented_page()
    snap = snapshot(get_page(client, page["id"]))
    snap[0]["id"] = other["lines"][0]["id"]  # belongs to another page: must not be stolen
    r = client.post(f"/pages/{page['id']}/restore", json={"lines": snap})
    assert r.status_code == 200
    assert get_page(client, other["id"])["lines"][0]["id"] == other["lines"][0]["id"]
    assert len(get_page(client, other["id"])["lines"]) == 3


# ---- XML sync / rollback on failure ----------------------------------------------------------------------------


def test_repair_ops_roll_back_when_the_xml_write_fails(settings, page, monkeypatch):
    conn = connect(settings.db_path)
    try:
        pid = page["id"]
        ids = [x["id"] for x in page["lines"]]
        path = sync.xml_file(conn, settings, pid)
        before_xml = path.read_bytes()
        query = 'SELECT id, "index", text FROM lines WHERE page_id = ? ORDER BY "index"'
        before = [tuple(r) for r in conn.execute(query, (pid,))]

        def broken(seg, name, p):
            raise OSError("disk full")

        monkeypatch.setattr(pagexml, "write_pagexml", broken)
        ops = [
            lambda: repair_svc.merge_lines(conn, settings, ids[:2]),
            lambda: repair_svc.split_line(conn, settings, ids[0], 300, 10),
            lambda: repair_svc.ignore_region(conn, settings, pid, {"x": 0, "y": 0, "w": 600, "h": 800}),
            lambda: repair_svc.reorder(conn, settings, pid, ids[::-1]),
            lambda: repair_svc.restore(conn, settings, pid, [], True),
        ]
        for op in ops:
            with pytest.raises(OSError):
                op()
            assert not conn.in_transaction
        assert path.read_bytes() == before_xml
        assert [tuple(r) for r in conn.execute(query, (pid,))] == before
    finally:
        conn.close()


# ---- active_ground_truth_count --------------------------------------------------------------------------------


def test_active_ground_truth_count_excludes_ignored(client, page):
    a, b, _ = (x["id"] for x in page["lines"])
    set_text(client, a, "x", gt=True)
    set_text(client, b, "y", gt=True)
    client.patch(f"/lines/{b}", json={"ignored": True})
    detail = get_page(client, page["id"])
    assert detail["ground_truth_count"] == 2 and detail["active_ground_truth_count"] == 1
    listed = next(p for p in client.get("/pages").json() if p["id"] == page["id"])
    assert listed["ground_truth_count"] == 2 and listed["active_ground_truth_count"] == 1
