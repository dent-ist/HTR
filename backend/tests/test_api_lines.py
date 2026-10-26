import pytest


@pytest.fixture
def page(api):
    return api.segmented_page()


def patch(client, line_id, **body):
    return client.patch(f"/lines/{line_id}", json=body)


def get_page(client, page_id):
    return client.get(f"/pages/{page_id}").json()


def test_patch_text_sets_manual_source(client, page):
    lid = page["lines"][0]["id"]
    r = patch(client, lid, text="pt c/o SOB")
    assert r.status_code == 200
    ln = r.json()
    assert ln["text"] == "pt c/o SOB" and ln["text_source"] == "manual" and ln["confidence"] is None
    assert get_page(client, page["id"])["lines"][0]["text"] == "pt c/o SOB"


def test_text_is_stored_diplomatically_untouched(client, page):
    lid = page["lines"][0]["id"]
    text = '  A&B <x> "q" café  '
    assert patch(client, lid, text=text).json()["text"] == text


def test_manual_edit_clears_prediction_confidence(client, page, settings):
    from app.db import connect

    lid = page["lines"][0]["id"]
    conn = connect(settings.db_path)
    conn.execute("UPDATE lines SET text='pred', text_source='predicted', confidence=0.42 WHERE id=?", (lid,))
    conn.close()
    got = get_page(client, page["id"])["lines"][0]
    assert got["text_source"] == "predicted" and got["confidence"] == 0.42
    ln = patch(client, lid, text="pred").json()  # accepting the prediction as typed text
    assert ln["text_source"] == "manual" and ln["confidence"] is None


def test_clearing_text_clears_source(client, page):
    lid = page["lines"][0]["id"]
    patch(client, lid, text="x")
    ln = patch(client, lid, text="").json()
    assert ln["text"] == "" and ln["text_source"] is None


def test_ground_truth_and_ignored_flags(client, page):
    lid = page["lines"][0]["id"]
    assert patch(client, lid, is_ground_truth=True).status_code == 422  # no text yet
    ln = patch(client, lid, text="hello", is_ground_truth=True).json()
    assert ln["is_ground_truth"] is True
    assert patch(client, lid, text="").status_code == 422  # would leave a GT line empty
    assert patch(client, lid, text="", is_ground_truth=False).status_code == 200
    assert patch(client, lid, ignored=True).json()["ignored"] is True
    assert patch(client, lid, ignored=False).json()["ignored"] is False


def test_geometry_update(client, page):
    lid = page["lines"][0]["id"]
    poly = [[10, 10], [200.4, 10], [200.6, 40], [10, 40]]
    base = [[10, 30], [200, 30]]
    ln = patch(client, lid, polygon=poly, baseline=base).json()
    assert ln["polygon"] == [[10, 10], [200, 10], [201, 40], [10, 40]]  # rounded to integer pixels
    assert ln["baseline"] == base


@pytest.mark.parametrize(
    "body",
    [
        {"text": "a\nb"},
        {"text": "tab\there"},
        {"text": "x" * 2001},
        {"text": None},
        {"ignored": None},
        {"polygon": [[0, 0], [10, 10]]},  # needs 3+ points
        {"polygon": [[0, 0], [10, 10], [5, 99999]]},  # outside the page
        {"polygon": [[0, 0], [10, 10], [-1, 5]]},
        {"polygon": [[0, 0], [10, 10], ["a", 5]]},
        {"polygon": [[0, 0], [10, 10], [1]]},
        {"polygon": "nope"},
        {"baseline": [[5, 5]]},  # needs 2+ points
        {"baseline": [[5, 5], [5, 99999]]},
        {"index": -1},
        {"index": 3},  # out of range for 3 lines
        {"index": "first"},
        {"unknown": 1},
        {"confidence": 0.5},  # not patchable
        {"is_ground_truth": "maybe"},
    ],
)
def test_patch_validation(client, page, body):
    lid = page["lines"][0]["id"]
    before = get_page(client, page["id"])
    assert patch(client, lid, **body).status_code == 422
    assert get_page(client, page["id"]) == before  # nothing changed


def test_patch_nan_rejected(client, page):
    lid = page["lines"][0]["id"]
    r = client.patch(
        f"/lines/{lid}", content='{"polygon": [[0,0],[1,1],[NaN,2]]}', headers={"content-type": "application/json"}
    )
    assert r.status_code == 422


def test_patch_unknown_line(client):
    assert patch(client, 424242, text="x").status_code == 404
    assert client.delete("/lines/424242").status_code == 404


def test_reorder_via_index(client, page):
    ids = [x["id"] for x in page["lines"]]
    patch(client, ids[2], index=0)
    after = get_page(client, page["id"])["lines"]
    assert [x["id"] for x in after] == [ids[2], ids[0], ids[1]]
    assert [x["index"] for x in after] == [0, 1, 2]
    patch(client, ids[2], index=2)
    assert [x["id"] for x in get_page(client, page["id"])["lines"]] == ids
    patch(client, ids[0], index=1)
    assert [x["id"] for x in get_page(client, page["id"])["lines"]] == [ids[1], ids[0], ids[2]]


def test_delete_line_renumbers(client, page):
    ids = [x["id"] for x in page["lines"]]
    assert client.delete(f"/lines/{ids[0]}").status_code == 204
    after = get_page(client, page["id"])["lines"]
    assert [x["id"] for x in after] == ids[1:] and [x["index"] for x in after] == [0, 1]
    assert client.get(f"/lines/{ids[0]}/crop.png").status_code == 404


def test_add_line(client, page):
    body = {"baseline": [[20, 500], [300, 500]], "polygon": [[20, 480], [300, 480], [300, 520], [20, 520]]}
    r = client.post(f"/pages/{page['id']}/lines", json=body)
    assert r.status_code == 201
    new = r.json()
    assert new["index"] == 3 and new["text"] == "" and new["page_id"] == page["id"]
    r2 = client.post(f"/pages/{page['id']}/lines", json={**body, "index": 0, "text": "first"})
    assert r2.json()["index"] == 0 and r2.json()["text_source"] == "manual"
    lines = get_page(client, page["id"])["lines"]
    assert [x["index"] for x in lines] == [0, 1, 2, 3, 4] and lines[0]["id"] == r2.json()["id"]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"baseline": [[1, 1]], "polygon": [[1, 1], [5, 1], [5, 5]]},
        {"baseline": [[1, 1], [2, 2]], "polygon": [[1, 1], [5, 1]]},
        {"baseline": [[1, 1], [2, 2]], "polygon": [[1, 1], [5, 1], [99999, 5]]},
        {"baseline": [[1, 1], [2, 2]], "polygon": [[1, 1], [5, 1], [5, 5]], "index": 9},
        {"baseline": [[1, 1], [2, 2]], "polygon": [[1, 1], [5, 1], [5, 5]], "text": "a\nb"},
    ],
)
def test_add_line_validation(client, page, body):
    assert client.post(f"/pages/{page['id']}/lines", json=body).status_code == 422
    assert len(get_page(client, page["id"])["lines"]) == 3


def test_add_line_errors(api, client):
    body = {"baseline": [[1, 1], [2, 2]], "polygon": [[1, 1], [5, 1], [5, 5]]}
    assert client.post("/pages/9999/lines", json=body).status_code == 404
    unseg = api.import_page()
    assert client.post(f"/pages/{unseg['id']}/lines", json=body).status_code == 409


def test_status_transitions(client, page):
    ids = [x["id"] for x in page["lines"]]
    pid = page["id"]

    def status():
        return get_page(client, pid)["status"]

    assert status() == "segmented"
    patch(client, ids[0], text="one")
    assert status() == "partially_transcribed"  # manual text counts
    patch(client, ids[0], text="")
    assert status() == "segmented"
    for i in ids:
        patch(client, i, text=f"t{i}", is_ground_truth=True)
    assert status() == "done"
    client.post(
        f"/pages/{pid}/lines",
        json={"baseline": [[1, 1], [9, 1]], "polygon": [[1, 1], [9, 1], [9, 5]]},
    )
    assert status() == "partially_transcribed"  # a new untranscribed line
    new_id = get_page(client, pid)["lines"][-1]["id"]
    patch(client, new_id, ignored=True)
    assert status() == "done"  # ignored lines do not block completion
    for i in ids:
        client.delete(f"/lines/{i}")
    assert status() == "segmented"  # only an ignored line is left: nothing to do, not done
