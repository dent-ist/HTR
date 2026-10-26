"""PAGE-XML <-> database consistency: sync on every mutation, rollback on failure, round trip."""

import sqlite3
import xml.etree.ElementTree as ET

import pytest

from app.db import connect
from app.errors import Invalid
from app.pipeline import pagexml
from app.pipeline.segment import Line
from app.services import lines as lines_svc
from app.services import sync

FIELDS = ("index", "baseline", "polygon", "text", "text_source", "confidence", "is_ground_truth", "ignored")


@pytest.fixture
def page(api):
    return api.segmented_page()


@pytest.fixture
def conn(settings, page):
    c = connect(settings.db_path)
    yield c
    c.close()


def xml_path(settings, page_id, conn):
    return sync.xml_file(conn, settings, page_id)


def xml_lines(path):
    seg, name = pagexml.read_pagexml(path)
    return seg.lines


def db_lines(conn, page_id):
    rows = conn.execute('SELECT * FROM lines WHERE page_id = ? ORDER BY "index"', (page_id,)).fetchall()
    return [{f: r[f] for f in FIELDS} for r in rows]


def test_segmentation_writes_pagexml_with_all_lines(settings, conn, page):
    path = xml_path(settings, page["id"], conn)
    assert path.exists() and not path.with_suffix(".xml.tmp").exists()
    seg, image_filename = pagexml.read_pagexml(path)
    assert len(seg.lines) == 3 and (seg.width, seg.height) == (page["width"], page["height"])
    assert image_filename.endswith(".png")
    assert [ln.polygon for ln in seg.lines] == [[tuple(p) for p in x["polygon"]] for x in page["lines"]]


def test_every_mutation_rewrites_the_xml(client, settings, conn, page):
    pid, ids = page["id"], [x["id"] for x in page["lines"]]
    path = xml_path(settings, pid, conn)

    client.patch(f"/lines/{ids[0]}", json={"text": "hello & <bye>", "is_ground_truth": True})
    first = xml_lines(path)[0]
    assert first.text == "hello & <bye>" and first.is_ground_truth and first.text_source == "manual"

    client.patch(f"/lines/{ids[1]}", json={"ignored": True})
    assert xml_lines(path)[1].ignored is True

    client.patch(f"/lines/{ids[2]}", json={"polygon": [[1, 1], [50, 1], [50, 20], [1, 20]]})
    assert xml_lines(path)[2].polygon == [(1, 1), (50, 1), (50, 20), (1, 20)]

    client.patch(f"/lines/{ids[2]}", json={"index": 0})
    assert xml_lines(path)[0].polygon == [(1, 1), (50, 1), (50, 20), (1, 20)]

    client.delete(f"/lines/{ids[0]}")
    assert len(xml_lines(path)) == 2

    client.post(f"/pages/{pid}/lines", json={"baseline": [[1, 1], [9, 1]], "polygon": [[1, 1], [9, 1], [9, 5]]})
    assert len(xml_lines(path)) == 3
    assert not path.with_suffix(".xml.tmp").exists()


def test_xml_uses_documented_custom_attributes(client, settings, conn, page):
    ids = [x["id"] for x in page["lines"]]
    client.patch(f"/lines/{ids[0]}", json={"text": "abc", "is_ground_truth": True})
    client.patch(f"/lines/{ids[1]}", json={"ignored": True})
    root = ET.parse(xml_path(settings, page["id"], conn)).getroot()
    ns = {"p": pagexml.NS}
    tls = root.findall(".//p:TextLine", ns)
    assert tls[0].get("custom") == "readingOrder {index:0;} htr {ground_truth:true; text_source:manual;}"
    assert tls[0].find("p:TextEquiv/p:Unicode", ns).text == "abc"
    assert tls[1].get("custom") == "readingOrder {index:1;} htr {ignored:true;}"
    assert tls[1].find("p:TextEquiv", ns) is None
    assert tls[2].get("custom") == "readingOrder {index:2;}"


def test_failed_xml_write_rolls_back_db_and_keeps_old_xml(settings, conn, page, monkeypatch):
    pid, lid = page["id"], page["lines"][0]["id"]
    path = xml_path(settings, pid, conn)
    before_xml, before_db = path.read_bytes(), db_lines(conn, pid)

    def broken_write(seg, image_filename, p):
        p.with_suffix(p.suffix + ".tmp").write_text("<half written")
        raise OSError("disk full")

    monkeypatch.setattr(pagexml, "write_pagexml", broken_write)
    with pytest.raises(OSError, match="disk full"):
        lines_svc.update_line(conn, settings, lid, {"text": "must not stick"})
    with pytest.raises(OSError):
        lines_svc.delete_line(conn, settings, lid)
    with pytest.raises(OSError):
        lines_svc.add_line(conn, settings, pid, {"baseline": [[1, 1], [9, 1]], "polygon": [[1, 1], [9, 1], [9, 5]]})
    assert path.read_bytes() == before_xml
    assert not path.with_suffix(".xml.tmp").exists()
    assert not conn.in_transaction
    assert db_lines(conn, pid) == before_db
    assert conn.execute("SELECT status FROM pages WHERE id = ?", (pid,)).fetchone()[0] == "segmented"


def test_failed_db_change_leaves_xml_untouched(settings, conn, page):
    pid, lid = page["id"], page["lines"][0]["id"]
    path = xml_path(settings, pid, conn)
    before = path.read_bytes()
    with pytest.raises(Invalid):  # validation happens before anything is written
        lines_svc.update_line(conn, settings, lid, {"index": 99})
    assert path.read_bytes() == before
    # a failure in the middle of the transaction (after some rows changed) also rolls back
    with pytest.raises(Invalid):
        lines_svc.update_line(conn, settings, lid, {"text": "partly applied", "index": 99})
    assert path.read_bytes() == before
    assert db_lines(conn, pid)[0]["text"] == ""


def test_failed_commit_rolls_back_and_restores_xml(settings, conn, page):
    """COMMIT can fail after the XML was already renamed into place; the XML must end up matching the DB."""
    pid, lid = page["id"], page["lines"][0]["id"]
    path = xml_path(settings, pid, conn)
    before_xml = path.read_bytes()
    with pytest.raises(sqlite3.IntegrityError):
        with sync.page_transaction(conn, settings, pid):
            conn.execute("UPDATE lines SET text = 'will roll back', text_source = 'manual' WHERE id = ?", (lid,))
            conn.execute("PRAGMA defer_foreign_keys = ON")
            conn.execute('INSERT INTO lines (page_id, "index", baseline, polygon) VALUES (99999, 0, "[]", "[]")')
    assert db_lines(conn, pid)[0]["text"] == ""
    assert path.read_bytes() == before_xml  # regenerated from the unchanged database


def test_concurrent_connections_see_consistent_state(settings, conn, page):
    other = connect(settings.db_path)
    lines_svc.update_line(conn, settings, page["lines"][0]["id"], {"text": "visible"})
    assert other.execute("SELECT text FROM lines WHERE id = ?", (page["lines"][0]["id"],)).fetchone()[0] == "visible"
    other.close()


# ---- round trip --------------------------------------------------------------------------------------------------


def make_varied_page(conn, settings, pid):
    """Lines with every kind of state: manual+GT, predicted with confidence, ignored, empty, awkward text."""
    ids = [r[0] for r in conn.execute('SELECT id FROM lines WHERE page_id = ? ORDER BY "index"', (pid,))]
    lines_svc.update_line(conn, settings, ids[0], {"text": ' lead & trail <"q"> café ✓ ', "is_ground_truth": True})
    lines_svc.update_line(conn, settings, ids[1], {"ignored": True, "text": "crossed out"})
    with sync.page_transaction(conn, settings, pid):
        conn.execute(
            "UPDATE lines SET text='predicted text', text_source='predicted', confidence=0.8125 WHERE id=?", (ids[2],)
        )
    lines_svc.add_line(
        conn,
        settings,
        pid,
        {"baseline": [[5, 5], [80, 9]], "polygon": [[5, 0], [80, 0], [80, 20], [5, 20]], "ignored": True},
    )


def test_round_trip_db_to_xml_to_db(settings, conn, page):
    pid = page["id"]
    make_varied_page(conn, settings, pid)
    before = db_lines(conn, pid)
    assert len(before) == 4
    assert {(r["text_source"], r["is_ground_truth"], r["ignored"]) for r in before} >= {
        ("manual", 1, 0),
        ("manual", 0, 1),
        ("predicted", 0, 0),
        (None, 0, 1),
    }
    n = sync.rebuild_lines_from_xml(conn, settings, pid)
    assert n == 4
    assert db_lines(conn, pid) == before
    # and it is stable: another cycle changes nothing
    sync.rebuild_lines_from_xml(conn, settings, pid)
    assert db_lines(conn, pid) == before


def test_rebuild_from_foreign_xml_file(settings, conn, page, tmp_path):
    pid = page["id"]
    make_varied_page(conn, settings, pid)
    before = db_lines(conn, pid)
    saved = tmp_path / "exported.xml"
    saved.write_bytes(xml_path(settings, pid, conn).read_bytes())
    for lid in [r[0] for r in conn.execute("SELECT id FROM lines WHERE page_id = ?", (pid,))]:
        lines_svc.delete_line(conn, settings, lid)
    assert db_lines(conn, pid) == []
    assert sync.rebuild_lines_from_xml(conn, settings, pid, saved) == 4
    assert db_lines(conn, pid) == before
    assert conn.execute("SELECT status FROM pages WHERE id = ?", (pid,)).fetchone()[0] == "partially_transcribed"


def test_rebuild_rejects_bad_xml(settings, conn, page, tmp_path):
    pid = page["id"]
    before = db_lines(conn, pid)
    bad = tmp_path / "bad.xml"
    bad.write_text("<not xml")
    with pytest.raises(Invalid, match="cannot read"):
        sync.rebuild_lines_from_xml(conn, settings, pid, bad)
    with pytest.raises(Invalid, match="cannot read"):
        sync.rebuild_lines_from_xml(conn, settings, pid, tmp_path / "missing.xml")
    wrong = tmp_path / "wrong.xml"
    seg = pagexml.PageSegmentation(
        width=10, height=10, lines=[Line(id="l1", baseline=[(0, 0), (5, 5)], polygon=[(0, 0), (5, 0), (5, 5)])]
    )
    pagexml.write_pagexml(seg, "x.png", wrong)
    with pytest.raises(Invalid, match="working copy"):
        sync.rebuild_lines_from_xml(conn, settings, pid, wrong)
    assert db_lines(conn, pid) == before  # nothing changed


def test_m1_style_xml_without_state_still_reads(tmp_path):
    seg = pagexml.PageSegmentation(
        width=100, height=100, lines=[Line(id="l001", baseline=[(1, 1), (9, 9)], polygon=[(0, 0), (9, 0), (9, 9)])]
    )
    p = tmp_path / "m1.xml"
    pagexml.write_pagexml(seg, "w.png", p)
    back, _ = pagexml.read_pagexml(p)
    ln = back.lines[0]
    assert (ln.text, ln.text_source, ln.confidence, ln.is_ground_truth, ln.ignored) == ("", None, None, False, False)
