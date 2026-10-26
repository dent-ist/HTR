"""Word-level experiment tooling: held-out split design (no leaks), bootstrap CER interval, the import tool."""

import json
import random
import sqlite3

import numpy as np
import pytest
from PIL import Image
from rxhand_import import ImportRefused, import_dataset, read_labels

from app.pipeline.evaluate import bootstrap_cer_ci, evaluate, exact_match
from app.services import wordsplits as WS


def synthetic_items(n_words=300, n_samples=2000, seed=5):
    """(id, label): a skewed word distribution with case variants and many singletons, like a real word dataset."""
    rng = random.Random(seed)
    words = []
    for i in range(n_words):
        w = "".join(rng.choice("abcdefghijklmnop") for _ in range(rng.randint(3, 9))) + str(i)
        words += [w, w.capitalize()] if i % 3 == 0 else [w]
    weights = [1 / (1 + k) ** 0.9 for k in range(len(words))]
    labels = rng.choices(words, weights, k=n_samples)
    return [(i + 1, lab) for i, lab in enumerate(labels)]


# ---- splits ------------------------------------------------------------------------------------------------------


def test_splits_partition_and_do_not_leak():
    items = synthetic_items()
    labels = dict(items)
    sp = WS.build_word_splits(items, val_size=100)
    assert WS.check_no_leak(sp, labels) == []
    parts = sp.part_of()
    assert set(parts) == set(labels)  # every sample in exactly one part
    assert sum(len(getattr(sp, n)) for n in ("train", "val", "seen", "unseen")) == len(items)
    assert len(sp.unseen) > 0 and len(sp.seen) > 0 and len(sp.val) == 100


def test_unseen_words_never_appear_in_training_even_as_case_variants():
    items = synthetic_items()
    labels = dict(items)
    sp = WS.build_word_splits(items)
    unseen_keys = {WS.label_key(labels[i]) for i in sp.unseen}
    for part in ("train", "val", "seen"):
        assert not unseen_keys & {WS.label_key(labels[i]) for i in getattr(sp, part)}
    # every sample of an unseen label is held out together
    all_with_key = [i for i, lab in items if WS.label_key(lab) in unseen_keys]
    assert sorted(all_with_key) == sorted(sp.unseen)


def test_seen_words_have_their_label_in_training():
    items = synthetic_items()
    labels = dict(items)
    sp = WS.build_word_splits(items)
    train_keys = {WS.label_key(labels[i]) for i in sp.train}
    assert all(WS.label_key(labels[i]) in train_keys for i in sp.seen)
    assert all(i not in set(sp.train) for i in sp.seen)


def test_splits_are_deterministic_and_independent_of_input_order():
    items = synthetic_items()
    a = WS.build_word_splits(items)
    b = WS.build_word_splits(list(reversed(items)))
    assert (a.train, a.val, a.seen, a.unseen) == (b.train, b.val, b.seen, b.unseen)


def test_nested_training_subsets_never_contain_held_out_samples():
    items = synthetic_items()
    labels = dict(items)
    sp = WS.build_word_splits(items)
    held = set(sp.seen) | set(sp.unseen) | set(sp.val)
    previous: list[int] = []
    for n in (100, 250, 500, len(sp.train)):
        sub = sp.train[:n]
        assert not held & set(sub) and sub[: len(previous)] == previous  # nested
        assert WS.check_no_leak(sp, labels, train_n=n) == []
        previous = sub
    assert 0 < WS.seen_coverage(sp, labels, 100) < WS.seen_coverage(sp, labels, len(sp.train)) <= 1


def test_check_no_leak_detects_a_leak():
    items = synthetic_items()
    labels = dict(items)
    sp = WS.build_word_splits(items)
    sp.train.append(sp.unseen[0])  # simulate a leak
    problems = WS.check_no_leak(sp, labels)
    assert any("share samples" in p for p in problems) and any("unseen-words label" in p for p in problems)


def test_app_split_marks_both_held_out_sets_as_eval():
    items = synthetic_items()
    sp = WS.build_word_splits(items)
    app = sp.app_split()
    assert {app[i] for i in sp.seen} == {"eval"} == {app[i] for i in sp.unseen}
    assert {app[i] for i in sp.train + sp.val} == {"train"}


def test_label_cleaning_and_key():
    assert WS.clean_label("  Napa \t") == "Napa" and WS.clean_label("Ace Power") == "Ace Power"
    assert WS.label_key("Ace  Power") == WS.label_key("ace power ") == "ace power"


# ---- bootstrap CER ------------------------------------------------------------------------------------------------


def test_bootstrap_ci_brackets_the_point_estimate_and_is_deterministic():
    pairs = [("hello", "hello")] * 80 + [("world", "wxrld")] * 15 + [("test", "")] * 5
    point = evaluate(pairs).cer
    lo, hi = bootstrap_cer_ci(pairs, n_boot=500)
    assert lo <= point <= hi and lo < hi
    assert bootstrap_cer_ci(pairs, n_boot=500) == (lo, hi)
    assert bootstrap_cer_ci(pairs, n_boot=500, seed=2) != (lo, hi)


def test_bootstrap_ci_narrows_with_more_samples():
    rng = random.Random(1)

    def make(n):
        return [("abcdefgh", "abcdefgh" if rng.random() > 0.3 else "abcdxfgh") for _ in range(n)]

    lo1, hi1 = bootstrap_cer_ci(make(40), n_boot=500)
    lo2, hi2 = bootstrap_cer_ci(make(1600), n_boot=500)
    assert (hi2 - lo2) < (hi1 - lo1) / 3


def test_bootstrap_ci_edge_cases_and_known_value():
    assert bootstrap_cer_ci([("abc", "abc")] * 10, n_boot=100) == (0.0, 0.0)
    assert bootstrap_cer_ci([], n_boot=10) is None and bootstrap_cer_ci([("", "x")], n_boot=10) is None
    # all samples equally wrong: every resample has the same CER, so the interval collapses on it
    lo, hi = bootstrap_cer_ci([("abcd", "abxd")] * 20, n_boot=100)
    assert lo == hi == pytest.approx(0.25)
    assert exact_match([("a", "a"), ("b", "c")]) == 0.5 and exact_match([]) is None


# ---- the import tool -----------------------------------------------------------------------------------------------


def make_dataset(root, labels):
    (root / "data").mkdir(parents=True)
    rng = np.random.default_rng(0)
    with open(root / "labels.csv", "w", encoding="utf-8-sig", newline="") as f:
        f.write("Images,Text,,\r\n")
        for i, lab in enumerate(labels, 1):
            name = f"P{i:04d}.jpg"
            arr = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
            Image.fromarray(arr).convert("RGB").save(root / "data" / name, quality=90, dpi=(240, 240))
            f.write(f"{name},{lab},,\r\n")
    return root / "data", root / "labels.csv"


LABELS = [w for w in ["Tablet", "tablet", "Napa", "Dexter", "Omep", "Inj", "Algin", "Zinc"] for _ in range(14)]


def test_read_labels_cleans_and_reports(tmp_path):
    images, csv_path = make_dataset(tmp_path, ["Tablet", "Napa  ", "???", "?icro", "Ok"])
    rows, rep = read_labels(csv_path)
    assert [t for _, t in rows] == ["Tablet", "Napa", "Ok"]
    assert rep["stripped"] == 1 and len(rep["dropped_question_mark"]) == 2 and rep["rows"] == 5


def test_import_dataset_builds_an_isolated_leak_free_project(tmp_path):
    images, csv_path = make_dataset(tmp_path / "src", LABELS)
    out = tmp_path / "proj"
    meta = import_dataset(images, csv_path, out, 3.0, val_size=10, log=lambda *_: None)
    c = meta["counts"]
    assert c["train"] + c["val"] + c["seen"] + c["unseen"] == len(LABELS)
    assert (out / "htr.db").exists() and (out / "experiment" / "meta.json").exists()
    db = sqlite3.connect(out / "htr.db")
    assert db.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == len(LABELS)
    n_gt = db.execute("SELECT COUNT(*) FROM lines WHERE is_ground_truth = 1 AND text_source = 'manual'").fetchone()[0]
    assert n_gt == len(LABELS)
    # originals are byte-identical
    first = sorted(images.iterdir())[0]
    assert (next((out / "originals").glob("P0001_*.jpg"))).read_bytes() == first.read_bytes()
    # line_splits: held-out sets are 'eval' and match the metadata
    splits = dict(db.execute("SELECT line_id, split FROM line_splits"))
    ent = {e["line_id"]: e for e in json.loads((out / "experiment" / "meta.json").read_text())["entries"]}
    assert all((splits[i] == "eval") == (e["part"] in ("seen", "unseen")) for i, e in ent.items())
    held_keys = {e["key"] for e in ent.values() if e["part"] == "unseen"}
    assert not held_keys & {e["key"] for e in ent.values() if e["part"] in ("train", "val", "seen")}
    # the working copy is the stretched canvas
    w = Image.open(next((out / "working").glob("P0001_*.png")))
    assert w.size == (96, 32)


def test_import_refuses_the_app_data_dir_and_existing_projects(tmp_path, monkeypatch):
    from app.config import REPO_ROOT

    images, csv_path = make_dataset(tmp_path / "src", LABELS[:20])
    with pytest.raises(ImportRefused):
        import_dataset(images, csv_path, REPO_ROOT / "data", 3.0)
    monkeypatch.setenv("HTR_DATA_DIR", str(tmp_path / "mine"))
    with pytest.raises(ImportRefused):
        import_dataset(images, csv_path, tmp_path / "mine", 3.0)
    out = tmp_path / "proj"
    import_dataset(images, csv_path, out, 3.0, val_size=2, log=lambda *_: None)
    with pytest.raises(ImportRefused):
        import_dataset(images, csv_path, out, 3.0)
