"""Abbreviation dictionary API: CRUD, ambiguity, counts, occurrences, expansion views."""

from __future__ import annotations


def add(client, written, expansion, **kw):
    r = client.post("/abbreviations", json={"written": written, "expansion": expansion, **kw})
    assert r.status_code == 201, r.text
    return r.json()


def label(client, page, texts, **flags):
    """Set the text of the page's lines in reading order; returns the lines."""
    out = []
    for ln, text in zip(page["lines"], texts, strict=False):
        r = client.patch(f"/lines/{ln['id']}", json={"text": text, **flags})
        assert r.status_code == 200, r.text
        out.append(r.json())
    return out


def test_crud_roundtrip(client):
    a = add(client, "  pt ", "patient", notes="common", case_sensitive=False)
    assert a["written"] == "pt" and a["expansion"] == "patient" and a["notes"] == "common"
    assert a["count"] == 0 and a["ambiguous"] is False and a["candidates"] == []
    assert a["created_at"] and a["updated_at"]
    assert client.get(f"/abbreviations/{a['id']}").json()["id"] == a["id"]
    r = client.patch(f"/abbreviations/{a['id']}", json={"expansion": "the patient", "notes": None})
    assert r.status_code == 200 and r.json()["expansion"] == "the patient" and r.json()["notes"] is None
    assert [x["id"] for x in client.get("/abbreviations").json()] == [a["id"]]
    assert client.delete(f"/abbreviations/{a['id']}").status_code == 204
    assert client.get("/abbreviations").json() == []
    assert client.get(f"/abbreviations/{a['id']}").status_code == 404
    assert client.delete(f"/abbreviations/{a['id']}").status_code == 404
    assert client.patch(f"/abbreviations/{a['id']}", json={"notes": "x"}).status_code == 404


def test_validation(client):
    for body in (
        {"written": "", "expansion": "x"},
        {"written": "a b", "expansion": "x"},
        {"written": "(", "expansion": "x"},
        {"written": "pt", "expansion": ""},
        {"written": "pt", "expansion": "pt"},  # a no-op expansion
        {"written": "pt", "expansion": "x", "bogus": 1},
        {"written": "x" * 65, "expansion": "y"},
    ):
        assert client.post("/abbreviations", json=body).status_code == 422, body
    a = add(client, "pt", "patient")
    assert client.patch(f"/abbreviations/{a['id']}", json={}).status_code == 422
    assert client.patch(f"/abbreviations/{a['id']}", json={"written": None}).status_code == 422


def test_unique_written_expansion_pairs(client):
    a = add(client, "pt", "patient")
    r = client.post("/abbreviations", json={"written": "pt", "expansion": "patient"})
    assert r.status_code == 409 and str(a["id"]) in r.json()["detail"]
    b = add(client, "pt", "physical therapy")  # same written form, other expansion: allowed
    r = client.patch(f"/abbreviations/{b['id']}", json={"expansion": "patient"})
    assert r.status_code == 409


def test_ambiguity_is_flagged_with_candidates(client):
    a = add(client, "SOB", "shortness of breath")
    assert a["ambiguous"] is False
    b = add(client, "SOB", "son of a bitch")
    got = {x["id"]: x for x in client.get("/abbreviations").json()}
    for me, other in ((a, b), (b, a)):
        x = got[me["id"]]
        assert x["ambiguous"] is True
        assert [c["expansion"] for c in x["candidates"]] == [me["expansion"], other["expansion"]]
    only = client.get("/abbreviations", params={"ambiguous": "true"}).json()
    assert {x["id"] for x in only} == {a["id"], b["id"]}
    assert client.get("/abbreviations", params={"ambiguous": "false"}).json() == []
    client.delete(f"/abbreviations/{b['id']}")
    assert client.get(f"/abbreviations/{a['id']}").json()["ambiguous"] is False


def test_case_sensitive_rows_do_not_conflict_when_spelled_differently(client):
    a = add(client, "US", "ultrasound", case_sensitive=True)
    b = add(client, "us", "ours", case_sensitive=True)
    assert not client.get(f"/abbreviations/{a['id']}").json()["ambiguous"]
    c = add(client, "Us", "united states")  # case-insensitive: overlaps both
    assert client.get(f"/abbreviations/{c['id']}").json()["ambiguous"]
    assert client.get(f"/abbreviations/{b['id']}").json()["ambiguous"]


def test_search_filter_and_sort(client):
    add(client, "pt", "patient", notes="clinic")
    add(client, "bid", "twice daily")
    add(client, "c/o", "complains of")
    assert [x["written"] for x in client.get("/abbreviations").json()] == ["bid", "c/o", "pt"]
    assert [x["written"] for x in client.get("/abbreviations", params={"q": "TWICE"}).json()] == ["bid"]
    assert [x["written"] for x in client.get("/abbreviations", params={"q": "clinic"}).json()] == ["pt"]


def test_counts_and_occurrences(client, api):
    page = api.segmented_page()
    pt = add(client, "pt", "patient")
    sob = add(client, "SOB", "shortness of breath")
    add(client, "SOB", "son of a bitch")
    label(client, page, ["pt c/o SOB", "Pt, pt seen", "no abbreviations here"])
    listing = {x["written"] + x["expansion"]: x for x in client.get("/abbreviations").json()}
    assert listing["ptpatient"]["count"] == 3
    assert listing["SOBshortness of breath"]["count"] == 1  # an ambiguous token counts for each candidate
    assert listing["SOBson of a bitch"]["count"] == 1
    # the cache column is refreshed by listing
    import sqlite3

    db = sqlite3.connect(client.app.state.settings.db_path)
    assert db.execute("SELECT count FROM abbreviations WHERE id = ?", (pt["id"],)).fetchone()[0] == 3
    db.close()

    occ = client.get(f"/abbreviations/{pt['id']}/occurrences").json()
    assert occ["total"] == 3
    first = occ["items"][0]
    assert first["page_id"] == page["id"] and first["line_index"] == 0 and first["text"] == "pt c/o SOB"
    assert (first["start"], first["end"]) == (0, 2) and first["ambiguous"] is False
    assert first["page_filename"] and first["document_name"] and first["text_source"] == "manual"
    assert [i["line_index"] for i in occ["items"]] == [0, 1, 1]
    # pagination
    p2 = client.get(f"/abbreviations/{pt['id']}/occurrences", params={"limit": 1, "offset": 1}).json()
    assert p2["total"] == 3 and len(p2["items"]) == 1 and p2["items"][0]["line_index"] == 1
    assert client.get(f"/abbreviations/{pt['id']}/occurrences", params={"limit": 0}).status_code == 422
    # ambiguous occurrence flagged
    o = client.get(f"/abbreviations/{sob['id']}/occurrences").json()["items"][0]
    assert o["ambiguous"] is True


def test_ignored_lines_are_not_counted(client, api):
    page = api.segmented_page()
    pt = add(client, "pt", "patient")
    lines = label(client, page, ["pt pt", "pt", "x"])
    assert client.get(f"/abbreviations/{pt['id']}").json()["count"] == 3
    client.patch(f"/lines/{lines[0]['id']}", json={"ignored": True})
    assert client.get(f"/abbreviations/{pt['id']}").json()["count"] == 1
    assert client.get(f"/abbreviations/{pt['id']}/occurrences").json()["total"] == 1


def test_counts_follow_edits(client, api):
    page = api.segmented_page()
    pt = add(client, "pt", "patient")
    lines = label(client, page, ["pt"])
    assert client.get("/abbreviations").json()[0]["count"] == 1
    client.patch(f"/lines/{lines[0]['id']}", json={"text": "patient"})
    assert client.get("/abbreviations").json()[0]["count"] == 0
    assert pt["id"]


def test_expand_preview_endpoint(client):
    add(client, "pt", "patient")
    add(client, "SOB", "shortness of breath")
    add(client, "SOB", "son of a bitch")
    r = client.post("/expand", json={"text": "Pt c/o SOB."})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "Patient c/o SOB."
    statuses = [s["status"] for s in body["spans"]]
    assert statuses == ["expanded", "unmatched", "ambiguous"]
    amb = body["spans"][2]
    assert amb["candidates"] == ["shortness of breath", "son of a bitch"] and amb["expansion"] is None
    assert client.post("/expand", json={"text": "SOB", "ambiguous": "mark"}).json()["text"] == "SOB[?]"
    assert client.post("/expand", json={"text": "x", "ambiguous": "nope"}).status_code == 422
    assert client.post("/expand", json={"text": "x" * 20001}).status_code == 422


def test_line_expanded_view(client, api):
    page = api.segmented_page()
    add(client, "pt", "patient")
    [ln] = label(client, page, ["pt c/o"])[:1]
    plain = client.get(f"/lines/{ln['id']}").json()
    assert plain["text"] == "pt c/o" and "expanded_text" not in plain
    exp = client.get(f"/lines/{ln['id']}", params={"expanded": "true"}).json()
    assert exp["text"] == "pt c/o" and exp["expanded_text"] == "patient c/o"
    assert [s["status"] for s in exp["spans"]] == ["expanded"]
    assert client.get("/lines/99999").status_code == 404


def test_page_text_raw_and_expanded(client, api):
    page = api.segmented_page()
    add(client, "pt", "patient")
    add(client, "SOB", "shortness of breath")
    add(client, "SOB", "son of a bitch")
    lines = label(client, page, ["pt c/o SOB", "Pt ok", "ignored pt"])
    client.patch(f"/lines/{lines[2]['id']}", json={"ignored": True})
    raw = client.get(f"/pages/{page['id']}/text").json()
    assert raw["text"] == "pt c/o SOB\nPt ok" and raw["expanded"] is False
    assert [x["raw"] for x in raw["lines"]] == ["pt c/o SOB", "Pt ok"]
    assert raw["lines"][0]["expanded"] == "patient c/o SOB"  # always provided for side-by-side views
    exp = client.get(f"/pages/{page['id']}/text", params={"expanded": "true", "ambiguous": "mark"}).json()
    assert exp["text"] == "patient c/o SOB[?]\nPatient ok"
    assert [s["status"] for s in exp["lines"][0]["spans"]] == ["expanded", "ambiguous"]
    assert client.get("/pages/99999/text").status_code == 404


def test_page_text_marks_and_filters_predicted_lines(client, api):
    page = api.segmented_page()
    add(client, "pt", "patient")
    first = page["lines"][0]
    # a predicted line is created through the database the way the predict job does
    import sqlite3

    db = sqlite3.connect(client.app.state.settings.db_path)
    db.execute(
        "UPDATE lines SET text = 'pt?', text_source = 'predicted', confidence = 0.5 WHERE id = ?", (first["id"],)
    )
    db.commit()
    db.close()
    both = client.get(f"/pages/{page['id']}/text", params={"expanded": "true"}).json()
    assert both["text"] == "[predicted] patient?"
    only = client.get(f"/pages/{page['id']}/text", params={"include_predicted": "false"}).json()
    assert only["text"] == "" and only["lines"] == []
