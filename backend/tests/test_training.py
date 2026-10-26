"""Training data: usable lines, the fixed held-out split, the summary and warnings, and the per-run export."""

import json
import xml.etree.ElementTree as ET

import pytest

from app.db import connect
from app.pipeline import pagexml
from app.services import training as T
from tests.ml_fakes import REF


def label(client, line_id, text=REF, gt=True, **extra):
    r = client.patch(f"/lines/{line_id}", json={"text": text, "is_ground_truth": gt, **extra})
    assert r.status_code == 200, r.text
    return r.json()


def make_lines(api, n_pages):
    """n_pages segmented pages with 3 lines each (mocked segmenter); returns all line dicts."""
    return [ln for _ in range(n_pages) for ln in api.segmented_page()["lines"]]


@pytest.fixture
def conn(client, settings):
    c = connect(settings.db_path)
    yield c
    c.close()


# ---- the split rule ----------------------------------------------------------------------------------------------


def test_split_hash_is_stable_and_spread():
    assert [T.split_hash(i) for i in (1, 2, 3)] == [T.split_hash(i) for i in (1, 2, 3)]
    assert T.split_hash(1) != T.split_hash(2)
    held = sum(T.natural_split(i) == "eval" for i in range(1, 5001))
    assert 0.09 < held / 5000 < 0.15  # ~12% of lines are held out


def test_usable_lines_exclude_ignored_blank_and_non_ground_truth(api, client, conn):
    lines = make_lines(api, 2)  # 6 lines
    label(client, lines[0]["id"])
    label(client, lines[1]["id"], ignored=True)  # ground truth but ignored
    label(client, lines[2]["id"], gt=False)  # manual text only
    client.patch(f"/lines/{lines[3]['id']}", json={"text": "   "})  # whitespace only, not ground truth
    assert [r["id"] for r in T.usable_lines(conn)] == [lines[0]["id"]]
    client.patch(f"/lines/{lines[4]['id']}", json={"text": REF})
    client.patch(f"/lines/{lines[4]['id']}", json={"is_ground_truth": True})
    assert {r["id"] for r in T.usable_lines(conn)} == {lines[0]["id"], lines[4]["id"]}


def test_no_lines_are_held_out_below_eight_and_nothing_is_persisted(api, client, conn):
    lines = make_lines(api, 2)
    for ln in lines[:5]:
        label(client, ln["id"])
    ids = [r["id"] for r in T.usable_lines(conn)]
    assert set(T.assign_splits(conn, ids).values()) == {"train"}
    assert conn.execute("SELECT COUNT(*) FROM line_splits").fetchone()[0] == 0


def test_at_least_one_held_out_line_from_eight_lines_on(api, client, conn):
    lines = make_lines(api, 3)  # 9 lines
    for ln in lines[:8]:
        label(client, ln["id"])
    ids = [r["id"] for r in T.usable_lines(conn)]
    assert len(ids) == 8
    splits = T.assign_splits(conn, ids)
    assert "eval" in splits.values() and "train" in splits.values()


@pytest.mark.parametrize("start", range(1, 60, 7))
def test_held_out_guarantee_for_any_set_of_ids(conn, start):
    """Unit-level: eight consecutive line ids always yield >= 1 held-out line (promotion of the lowest hash)."""
    conn.execute("INSERT INTO documents (name, imported_at) VALUES ('d', 'now')")
    conn.execute(
        "INSERT INTO pages (document_id, image_path, xml_path, width, height, working_path, original_filename, "
        "sha256, original_width, original_height, imported_at) VALUES (1, 'a', 'b', 1, 1, 'c', 'd', 'e', 1, 1, 'f')"
    )
    ids = list(range(start, start + 8))
    for i in ids:
        conn.execute('INSERT INTO lines (id, page_id, "index", baseline, polygon) VALUES (?, 1, ?, "[]", "[]")', (i, i))
    splits = T.assign_splits(conn, ids)
    assert list(splits.values()).count("eval") >= 1
    natural = [i for i in ids if T.natural_split(i) == "eval"]
    if natural:  # no promotion needed: exactly the natural assignment
        assert {i for i, s in splits.items() if s == "eval"} == set(natural)
    else:  # promoted: the single lowest hash
        assert [i for i, s in splits.items() if s == "eval"] == [min(ids, key=lambda i: (T.split_hash(i), i))]


def test_a_line_never_changes_side_across_rounds(api, client, conn):
    lines = make_lines(api, 8)  # 24 lines
    first = lines[:12]
    for ln in first:
        label(client, ln["id"])
    ids1 = [r["id"] for r in T.usable_lines(conn)]
    s1 = T.assign_splits(conn, ids1)
    for ln in lines[12:]:
        label(client, ln["id"])
    ids2 = [r["id"] for r in T.usable_lines(conn)]
    s2 = T.assign_splits(conn, ids2)
    assert len(s2) == 24
    assert all(s2[i] == s1[i] for i in ids1)  # nothing moved
    # a third call changes nothing either, and un-flagging then re-flagging a line keeps its side
    victim = ids1[0]
    side = s2[victim]
    client.patch(f"/lines/{victim}", json={"is_ground_truth": False})
    client.patch(f"/lines/{victim}", json={"is_ground_truth": True})
    assert T.assign_splits(conn, [r["id"] for r in T.usable_lines(conn)])[victim] == side


def test_first_round_with_few_lines_does_not_pin_them_to_train(api, client, conn):
    """Below 8 lines nothing is stored, so those lines can still be held out once the project has enough."""
    lines = make_lines(api, 4)
    for ln in lines[:5]:
        label(client, ln["id"])
    T.assign_splits(conn, [r["id"] for r in T.usable_lines(conn)])
    assert conn.execute("SELECT COUNT(*) FROM line_splits").fetchone()[0] == 0
    for ln in lines[5:12]:
        label(client, ln["id"])
    ids = [r["id"] for r in T.usable_lines(conn)]
    splits = T.assign_splits(conn, ids)
    assert conn.execute("SELECT COUNT(*) FROM line_splits").fetchone()[0] == 12 == len(splits)
    assert "eval" in splits.values()


def test_validation_subset_is_deterministic_and_never_the_held_out_set():
    train = list(range(100, 140))
    v1, is_train = T.choose_validation(train)
    assert v1 == T.choose_validation(train)[0] and not is_train and 1 <= len(v1) < len(train)
    small, is_train = T.choose_validation([1, 2, 3])
    assert small == [1, 2, 3] and is_train  # fewer than 10 lines: validate on the training lines


# ---- summary and warnings ----------------------------------------------------------------------------------------


def test_summary_empty_project(client, fake_ml):
    s = client.get("/training/summary").json()
    assert s["ground_truth_lines"] == 0 and s["recommended"] == 100 and s["last_round"] is None
    assert any("100" in w for w in s["warnings"]) and any("held out" in w for w in s["warnings"])
    assert s["active_model"]["name"] == "test-base"


def test_summary_counts_and_first_round_warning(api, client, fake_ml):
    lines = make_lines(api, 4)
    for ln in lines[:10]:
        label(client, ln["id"])
    s = client.get("/training/summary").json()
    assert s["ground_truth_lines"] == 10 and s["new_since_last_round"] == 10
    assert s["predictable_lines"] == 2  # the two lines that are neither ground truth nor ignored nor typed
    assert any("Only 10 ground-truth lines" in w for w in s["warnings"])
    assert s["projected_held_out_lines"] >= 1


def test_no_warning_once_the_recommended_number_is_reached(api, client, fake_ml, conn):
    page = api.segmented_page()  # one real page; the other ground-truth lines are inserted directly
    for i in range(101):
        conn.execute(
            'INSERT INTO lines (page_id, "index", baseline, polygon, text, text_source, is_ground_truth) '
            "VALUES (?, ?, '[[0,0],[9,0]]', '[[0,0],[9,0],[9,9]]', 'x', 'manual', 1)",
            (page["id"], 10 + i),
        )
    s = client.get("/training/summary").json()
    assert s["ground_truth_lines"] == 101 and s["warnings"] == []
    conn.execute("UPDATE lines SET is_ground_truth = 0 WHERE id % 2 = 0")
    assert any("Only" in w for w in client.get("/training/summary").json()["warnings"])


# ---- export ------------------------------------------------------------------------------------------------------


def test_export_run_writes_xml_crops_and_manifest(api, client, settings, conn):
    lines = make_lines(api, 6)  # 18 lines
    texts = {}
    for i, ln in enumerate(lines[:14]):
        texts[ln["id"]] = f"line {i} ünïcode"
        label(client, ln["id"], texts[ln["id"]])
    label(client, lines[14]["id"], "ignored one", ignored=True)
    client.patch(f"/lines/{lines[15]['id']}", json={"text": "typed, not ground truth"})
    ids = [r["id"] for r in T.usable_lines(conn)]
    assert set(ids) == set(texts)
    splits = T.assign_splits(conn, ids)
    train_ids = [i for i in ids if splits[i] == "train"]
    val_ids, val_is_train = T.choose_validation(train_ids)
    exp = T.export_run(conn, settings, "run-7", splits, set(val_ids), val_is_train, {"job_id": 7})

    assert exp.run_dir == settings.data_dir / "training" / "run-7"
    manifest = json.loads((exp.run_dir / "manifest.json").read_text())
    n_eval = sum(s == "eval" for s in splits.values())
    assert manifest["job_id"] == 7 and manifest["counts"]["held_out"] == n_eval >= 1
    assert manifest["counts"]["train"] == len(train_ids) and manifest["split_rule"]["version"] == "htr-split-v1"
    assert {e["line_id"] for e in manifest["lines"]} == set(ids)

    # PAGE-XML: train files hold exactly the train lines with their exact text; no ignored / untranscribed lines
    seen = {"train": {}, "eval": {}}
    for part, files in (("train", exp.train_files), ("eval", exp.eval_files)):
        for f in files:
            seg, image = pagexml.read_pagexml(f)
            assert image.startswith("/") and image.endswith(".png")  # absolute working-copy path
            for ln in seg.lines:
                seen[part][int(ln.id[1:])] = ln.text
    assert set(seen["train"]) == set(train_ids) and set(seen["eval"]) == {i for i in ids if splits[i] == "eval"}
    assert all(texts[i] == t for part in seen.values() for i, t in part.items())
    assert not set(seen["train"]) & set(seen["eval"])
    # crops + transcriptions on disk
    for lid in ids:
        part = "eval" if splits[lid] == "eval" else "train"
        stem = exp.run_dir / "lines" / part / f"line_{lid}"
        assert stem.with_suffix(".png").is_file()
        assert stem.with_suffix(".gt.txt").read_text(encoding="utf-8") == texts[lid]
    assert len(exp.eval_samples) == n_eval
    # kraken can parse what we wrote
    from kraken.lib.xml import XMLPage

    f = exp.train_files[0]
    n_lines = len(ET.parse(f).getroot().findall(".//{*}TextLine"))
    assert len(XMLPage(f).to_container().lines) == n_lines


def test_export_is_reproducible_for_the_same_lines(api, client, settings, conn):
    lines = make_lines(api, 4)
    for ln in lines[:10]:
        label(client, ln["id"])
    ids = [r["id"] for r in T.usable_lines(conn)]
    splits = T.assign_splits(conn, ids)
    val, flag = T.choose_validation([i for i in ids if splits[i] == "train"])
    a = T.export_run(conn, settings, "run-a", splits, set(val), flag)
    b = T.export_run(conn, settings, "run-b", splits, set(val), flag)
    assert [p.name for p in a.train_files] == [p.name for p in b.train_files]
    assert (a.run_dir / "train" / a.train_files[0].name).read_text().replace("run-a", "") == (
        b.run_dir / "train" / b.train_files[0].name
    ).read_text().replace("run-b", "")
