"""The word-crop path: the word canvas, the presegmented import, and the train / predict preprocessing contract."""

import numpy as np
import pytest
from PIL import Image

from app.db import connect
from app.errors import Invalid
from app.pipeline import pagexml, wordcrop
from app.pipeline.recognize import extract_line_image
from app.services import presegmented, training
from tests.conftest import wait_job
from tests.ml_fakes import FakeRecognizer
from tests.test_ml_jobs import client, project, project_template, start_train  # noqa: F401


def word_file(tmp_path, seed: int, size=(40, 40), name=None):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    p = tmp_path / (name or f"w{seed}.jpg")
    Image.fromarray(arr).save(p, quality=90)
    return p


# ---- the canvas --------------------------------------------------------------------------------------------------


def test_canvas_stretches_horizontally_and_is_deterministic(tmp_path):
    src = Image.open(word_file(tmp_path, 1, (30, 20)))
    c = wordcrop.WordCanvas(3)
    a, b = wordcrop.apply_word_canvas(src, c), wordcrop.apply_word_canvas(src, c)
    assert a.mode == "L" and a.size == (90, 20)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert wordcrop.apply_word_canvas(src, wordcrop.WordCanvas(1)).size == (30, 20)


def test_canvas_key_and_validation():
    assert wordcrop.WordCanvas(3).key == "word_canvas:x3" and wordcrop.WordCanvas(2.5).key == "word_canvas:x2.5"
    with pytest.raises(ValueError):
        wordcrop.WordCanvas(0.5)
    with pytest.raises(ValueError):
        wordcrop.WordCanvas(20)


def test_preprocess_keys_and_compatibility():
    assert wordcrop.key_of_preprocess({}) == wordcrop.PAGE_KEY
    assert wordcrop.key_of_preprocess(wordcrop.WordCanvas(3).to_dict()) == "word_canvas:x3"
    assert wordcrop.compatible(None, "page") and wordcrop.compatible("word_canvas:x3", "word_canvas:x3")
    assert not wordcrop.compatible("word_canvas:x3", "page") and not wordcrop.compatible("page", "word_canvas:x3")


# ---- the presegmented import ---------------------------------------------------------------------------------------


@pytest.fixture
def conn_settings(client):
    s = client.app.state.settings
    return connect(s.db_path), s


def test_import_word_image_makes_a_one_line_ground_truth_page(conn_settings, tmp_path):
    conn, s = conn_settings
    doc = conn.execute("INSERT INTO documents (name, imported_at) VALUES ('w', 'now')").lastrowid
    src = word_file(tmp_path, 7, (40, 30))
    page, created = presegmented.import_word_image(conn, s, doc, src, "Napa", wordcrop.WordCanvas(3))
    assert created and (page["width"], page["height"]) == (120, 30) and page["segmented_at"]
    assert page["scale"] == 1.0 and page["skew_deg"] == 0.0 and not page["deskewed"]
    original = s.data_dir / conn.execute("SELECT image_path FROM pages WHERE id = ?", (page["id"],)).fetchone()[0]
    assert original.read_bytes() == src.read_bytes()  # byte-identical
    assert conn.execute("SELECT preprocess FROM pages WHERE id = ?", (page["id"],)).fetchone()[0].count("word_canvas")
    lines = conn.execute("SELECT * FROM lines WHERE page_id = ?", (page["id"],)).fetchall()
    assert len(lines) == 1 and lines[0]["text"] == "Napa" and lines[0]["is_ground_truth"] == 1
    assert lines[0]["text_source"] == "manual"
    seg, _ = pagexml.read_pagexml(s.data_dir / f"pagexml/{original.stem}.xml")
    assert [ln.text for ln in seg.lines] == ["Napa"] and seg.lines[0].is_ground_truth
    # the crop Kraken's extractor cuts is the whole working copy: no rescale, no clipping
    work = Image.open(s.data_dir / conn.execute("SELECT working_path FROM pages").fetchone()[0])
    import json

    crop = extract_line_image(work, json.loads(lines[0]["baseline"]), json.loads(lines[0]["polygon"]))
    assert crop.size == work.size and np.array_equal(np.asarray(crop.convert("L")), np.asarray(work))


def test_import_is_idempotent_and_rejects_blank_text(conn_settings, tmp_path):
    conn, s = conn_settings
    doc = conn.execute("INSERT INTO documents (name, imported_at) VALUES ('w', 'now')").lastrowid
    src = word_file(tmp_path, 8)
    c = wordcrop.WordCanvas(2)
    p1, new1 = presegmented.import_word_image(conn, s, doc, src, "a", c)
    p2, new2 = presegmented.import_word_image(conn, s, doc, src, "a", c)
    assert new1 and not new2 and p1["id"] == p2["id"]
    with pytest.raises(Invalid):
        presegmented.import_word_image(conn, s, doc, word_file(tmp_path, 9), "  ", c)
    assert conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 1


def test_unreadable_file_leaves_nothing_behind(conn_settings, tmp_path):
    conn, s = conn_settings
    doc = conn.execute("INSERT INTO documents (name, imported_at) VALUES ('w', 'now')").lastrowid
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        presegmented.import_word_image(conn, s, doc, bad, "x", wordcrop.WordCanvas(3))
    assert conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 0
    assert not list((s.data_dir / "originals").glob("*"))


# ---- train / predict see the same pixels, and a model is never used with the wrong preprocessing ---------------------


def import_words(conn, s, tmp_path, n, canvas, gt=True, first_seed=100):
    """gt=False pages are left untranscribed (what the predict job picks up)."""
    doc = conn.execute("INSERT INTO documents (name, imported_at) VALUES ('w', 'now')").lastrowid
    out = []
    for k in range(n):
        page, _ = presegmented.import_word_image(
            conn, s, doc, word_file(tmp_path, first_seed + k), f"word{k % 5}", canvas, ground_truth=gt
        )
        out.append(page["id"])
        if not gt:
            conn.execute("UPDATE lines SET text = '', text_source = NULL WHERE page_id = ?", (page["id"],))
    return out


def test_training_export_and_prediction_read_identical_crops(client, fake_ml, conn_settings, tmp_path):
    conn, s = conn_settings
    import_words(conn, s, tmp_path, 14, wordcrop.WordCanvas(3))
    open_page = import_words(conn, s, tmp_path, 1, wordcrop.WordCanvas(3), gt=False, first_seed=500)[0]
    seen = []
    FakeRecognizer.on_predict = None
    orig = FakeRecognizer.predict

    def spy(self, line_images, should_stop=None):
        seen.extend(im.copy() for im in line_images)
        return orig(self, line_images, should_stop)

    FakeRecognizer.predict = spy
    try:
        done = wait_job(client, start_train(client, epochs=1)["id"])
        assert done["status"] == "succeeded", done["error"]
        seen.clear()
        job = client.post("/jobs/predict", json={"model_id": done["result"]["model_id"]}).json()
        assert wait_job(client, job["id"])["status"] == "succeeded"
    finally:
        FakeRecognizer.predict = orig
    predicted = np.asarray(seen[0].convert("L"))
    work = Image.open(
        s.data_dir / conn.execute("SELECT working_path FROM pages WHERE id = ?", (open_page,)).fetchone()[0]
    )
    assert np.array_equal(predicted, np.asarray(work))  # prediction crop == working copy
    # a training export crop of any page is also exactly its working copy
    run = s.data_dir / done["result"]["run_dir"]
    for png in list((run / "lines" / "train").glob("*.png"))[:3]:
        lid = int(png.stem.split("_")[1])
        row = conn.execute(
            "SELECT pages.working_path FROM lines JOIN pages ON pages.id = lines.page_id WHERE lines.id = ?", (lid,)
        ).fetchone()
        assert np.array_equal(np.asarray(Image.open(png).convert("L")), np.asarray(Image.open(s.data_dir / row[0])))
    model = client.get(f"/models/{done['result']['model_id']}").json()
    assert model["hyperparameters"]["line_preprocess"] == "word_canvas:x3"


def test_model_is_refused_on_pages_prepared_differently(client, fake_ml, conn_settings, tmp_path):
    conn, s = conn_settings
    import_words(conn, s, tmp_path, 14, wordcrop.WordCanvas(3))
    done = wait_job(client, start_train(client, epochs=1)["id"])
    assert done["status"] == "succeeded"
    mid = done["result"]["model_id"]
    import_words(conn, s, tmp_path, 1, wordcrop.WordCanvas(2), gt=False, first_seed=900)  # a x2 page to predict
    job = wait_job(client, client.post("/jobs/predict", json={"model_id": mid}).json()["id"])
    assert job["status"] == "failed" and "word_canvas:x3" in job["error"] and "word_canvas:x2" in job["error"]
    # the pretrained base model has no such constraint
    ok = wait_job(client, client.post("/jobs/predict", json={"model_id": 1}).json()["id"])
    assert ok["status"] == "succeeded"
    # further training on the x2 page together with x3 pages is refused too
    conn.execute("UPDATE lines SET is_ground_truth = 1 WHERE text != ''")
    bad = wait_job(client, start_train(client, epochs=1)["id"])
    assert bad["status"] == "failed" and "different ways" in bad["error"]


def test_ordinary_pages_keep_the_page_key(client, fake_ml, project):
    """Existing behaviour: a fine-tune on segmented scans records the default key and predicts as before."""
    done = wait_job(client, start_train(client, epochs=1)["id"])
    assert done["status"] == "succeeded"
    m = client.get(f"/models/{done['result']['model_id']}").json()
    assert m["hyperparameters"]["line_preprocess"] == "page"
    assert training.page_preprocess_keys(connect(client.app.state.settings.db_path), [1]) == {1: "page"}
