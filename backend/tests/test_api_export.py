"""Text and PAGE-XML export."""

from __future__ import annotations

import io
import sqlite3
import zipfile

import pytest

from app.pipeline import pagexml
from tests.test_api_abbreviations import add, label


@pytest.fixture
def project(client, api):
    """Two pages in 'Field notes' (page 1 labelled, page 2 partly) and one page in another document."""
    doc = api.new_document("Field notes")
    p1 = api.segmented_page(document_id=doc["id"], name="a.png")
    p2 = api.segmented_page(document_id=doc["id"], name="b.png")
    other = api.segmented_page(name="c.png")
    add(client, "pt", "patient")
    add(client, "c/o", "complains of")
    add(client, "SOB", "shortness of breath")
    add(client, "SOB", "son of a bitch")
    l1 = label(client, p1, ["pt c/o SOB", "Pt seen, w/ pt", "scribble"], is_ground_truth=True)
    client.patch(f"/lines/{l1[2]['id']}", json={"ignored": True})
    label(client, p2, ["second page pt"])  # lines 2 and 3 stay empty
    label(client, other, ["other doc"])
    return {"doc": doc, "p1": p1, "p2": p2, "other": other, "l1": l1}


def set_predicted(client, line_id, text, conf=0.6):
    db = sqlite3.connect(client.app.state.settings.db_path)
    db.execute(
        "UPDATE lines SET text = ?, text_source = 'predicted', confidence = ?, is_ground_truth = 0 WHERE id = ?",
        (text, conf, line_id),
    )
    db.commit()
    db.close()


def test_raw_text_export(client, project):
    r = client.get("/export", params={"document_id": project["doc"]["id"]})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain") and "utf-8" in r.headers["content-type"]
    assert r.headers["content-disposition"] == 'attachment; filename="htr-Field-notes-raw.txt"'
    assert r.text == (
        "=== Field notes / page 1 (a.png) ===\n"
        "pt c/o SOB\n"
        "Pt seen, w/ pt\n"
        "\n"
        "=== Field notes / page 2 (b.png) ===\n"
        "second page pt\n"
    )


def test_expanded_text_export_with_ambiguous_options(client, project):
    params = {"document_id": project["doc"]["id"], "expanded": "true"}
    raw_amb = client.get("/export", params=params).text
    assert "patient complains of SOB\n" in raw_amb  # ambiguous left as written by default
    assert "Patient seen, w/ patient" in raw_amb and "second page patient" in raw_amb
    marked = client.get("/export", params={**params, "ambiguous": "mark"}).text
    assert "patient complains of SOB[?]\n" in marked
    r = client.get("/export", params=params)
    assert r.headers["content-disposition"].endswith('-expanded.txt"')


def test_export_scopes(client, project):
    everything = client.get("/export").text
    assert "other doc" in everything and "Field notes / page 2" in everything
    assert client.get("/export").headers["content-disposition"] == 'attachment; filename="htr-all-raw.txt"'
    one = client.get("/export", params={"page_ids": str(project["p2"]["id"])}).text
    assert "second page pt" in one and "pt c/o SOB" not in one and "other doc" not in one
    assert client.get("/export", params={"document_id": 999}).status_code == 404
    assert client.get("/export", params={"page_ids": "999"}).status_code == 404
    assert client.get("/export", params={"page_ids": "x"}).status_code == 422
    # a page outside the requested document is reported, not silently skipped
    assert (
        client.get(
            "/export", params={"document_id": project["doc"]["id"], "page_ids": str(project["other"]["id"])}
        ).status_code
        == 404
    )


def test_predicted_lines_are_excluded_by_default_and_marked_when_included(client, project):
    p2_lines = client.get(f"/pages/{project['p2']['id']}").json()["lines"]
    set_predicted(client, p2_lines[1]["id"], "predicted words pt")
    doc = project["doc"]["id"]
    default = client.get("/export", params={"document_id": doc}).text
    assert "predicted words" not in default
    incl = client.get("/export", params={"document_id": doc, "include_predicted": "true", "expanded": "true"}).text
    assert "[predicted] predicted words patient\n" in incl
    assert "second page patient\n[predicted]" in incl  # reading order kept


def test_pages_without_text_are_left_out_and_empty_export_is_empty(client, api):
    api.segmented_page()
    r = client.get("/export")
    assert r.status_code == 200 and r.text == ""


def test_export_is_utf8(client, api):
    page = api.segmented_page()
    label(client, page, ["café über — ß"])
    r = client.get("/export")
    assert r.content.decode("utf-8").splitlines()[1] == "café über — ß"


def test_export_has_no_personal_paths(client, project):
    text = client.get("/export").text
    assert "/home" not in text and "\\" not in text
    disp = client.get("/export").headers["content-disposition"]
    assert "/" not in disp.split("filename=")[1].strip('"')


def test_pagexml_export_is_a_zip_of_valid_page_xml(client, project, tmp_path):
    set_predicted(client, client.get(f"/pages/{project['p2']['id']}").json()["lines"][1]["id"], "guess", 0.42)
    r = client.get("/export", params={"format": "pagexml", "include_predicted": "true"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert r.headers["content-disposition"] == 'attachment; filename="htr-all-pagexml.zip"'
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert len(names) == 3 and all(n.endswith(".xml") and "/" not in n for n in names)
    # equal to the project's own PAGE-XML files (same writer, same database)
    settings = client.app.state.settings
    for page in (project["p1"], project["other"]):  # p2 got a prediction behind the sync layer's back
        row = client.get(f"/pages/{page['id']}").json()
        db = sqlite3.connect(settings.db_path)
        xml_path = db.execute("SELECT xml_path FROM pages WHERE id = ?", (page["id"],)).fetchone()[0]
        db.close()
        assert zf.read(xml_path.split("/")[-1]) == (settings.data_dir / xml_path).read_bytes()
        f = tmp_path / "x.xml"
        f.write_bytes(zf.read(xml_path.split("/")[-1]))
        seg, _ = pagexml.read_pagexml(f)
        assert [ln.text for ln in seg.lines] == [ln["text"] for ln in row["lines"]]
        assert [ln.ignored for ln in seg.lines] == [ln["ignored"] for ln in row["lines"]]
        assert [ln.is_ground_truth for ln in seg.lines] == [ln["is_ground_truth"] for ln in row["lines"]]
    p2 = tmp_path / "p2.xml"
    p2.write_bytes(zf.read([n for n in names if n.startswith(project["p2"]["original_filename"].split(".")[0])][0]))
    seg, _ = pagexml.read_pagexml(p2)
    assert seg.lines[1].text == "guess" and seg.lines[1].text_source == "predicted" and seg.lines[1].confidence == 0.42


def test_pagexml_export_drops_predictions_by_default(client, project, tmp_path):
    lid = client.get(f"/pages/{project['p2']['id']}").json()["lines"][1]["id"]
    set_predicted(client, lid, "guess")
    r = client.get("/export", params={"format": "pagexml", "document_id": project["doc"]["id"]})
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert len(zf.namelist()) == 2
    assert b"guess" not in b"".join(zf.read(n) for n in zf.namelist())
    assert b"pt c/o SOB" in b"".join(zf.read(n) for n in zf.namelist())  # ground truth stays


def test_pagexml_export_skips_unsegmented_pages_and_rejects_expanded(client, api):
    api.import_page()
    r = client.get("/export", params={"format": "pagexml"})
    assert r.status_code == 200 and zipfile.ZipFile(io.BytesIO(r.content)).namelist() == []
    assert client.get("/export", params={"format": "pagexml", "expanded": "true"}).status_code == 422
    assert client.get("/export", params={"format": "docx"}).status_code == 422
