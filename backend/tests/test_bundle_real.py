"""Opt-in round trip with the REAL Kraken engine (`make test-train`; needs `make base-model` and `make fonts`):
a tiny real fine-tune and prediction, a bundle export, an import into a fresh data directory that has no base model of
its own, and a real prediction there with the imported (fine-tuned) model.

Like test_train_real.py this proves wiring, not accuracy (ten synthetic lines, two epochs).
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from make_synthetic_pages import available_fonts, render_page

from app.config import REPO_ROOT, Settings
from app.main import create_app
from tests import bundle_helpers as H
from tests.conftest import fake_blla_factory, wait_job

pytestmark = pytest.mark.train

BASE = REPO_ROOT / "data" / "models" / "base"


@pytest.mark.skipif(not list(BASE.glob("*/BASE_MODEL.json")), reason="run `make base-model` first")
def test_real_model_survives_a_bundle_round_trip_and_predicts(tmp_path, monkeypatch):
    from app.pipeline import segment

    monkeypatch.setattr(segment, "run_blla", fake_blla_factory(n_lines=0))  # lines come from the rendered truth
    src = Settings.for_dir(tmp_path / "src", worker_poll_seconds=0.05, device="auto")
    (src.data_dir / "models").mkdir(parents=True)
    (src.data_dir / "models" / "base").symlink_to(BASE)
    rng = np.random.default_rng(5)
    texts: dict[int, str] = {}
    with TestClient(create_app(src)) as c:
        doc = c.post("/documents", json={"name": "real"}).json()
        for p in range(2):
            truth: list[dict] = []
            page = render_page(rng, available_fonts()[p % len(available_fonts())], font_size=44, n_lines=6,
                               margin_note=False, truth=truth)  # fmt: skip
            buf = io.BytesIO()
            page.image.save(buf, format="PNG")
            pg = c.post(
                "/pages", data={"document_id": str(doc["id"])}, files={"file": ("p.png", buf.getvalue())}
            ).json()
            assert wait_job(c, c.post(f"/pages/{pg['id']}/segment").json()["id"])["status"] == "succeeded"
            k = pg["scale"]
            for t in truth:
                x0, y0, x1, y1 = (v * k for v in t["bbox"])
                h = y1 - y0
                base_y = y0 + 0.75 * h
                r = c.post(
                    f"/pages/{pg['id']}/lines",
                    json={
                        "baseline": [[x0, base_y], [x1, base_y]],
                        "polygon": [
                            [x0 - 4, y0 - 0.2 * h],
                            [x1 + 4, y0 - 0.2 * h],
                            [x1 + 4, y1 + 0.25 * h],
                            [x0 - 4, y1 + 0.25 * h],
                        ],
                    },
                )  # fmt: skip
                assert r.status_code == 201, r.text
                texts[r.json()["id"]] = t["text"]
        ids = sorted(texts)
        for lid in ids[:10]:
            assert c.patch(f"/lines/{lid}", json={"text": texts[lid], "is_ground_truth": True}).status_code == 200
        assert c.post("/abbreviations", json={"written": "pt", "expansion": "patient"}).status_code == 201
        train = wait_job(c, c.post("/jobs/train", json={"epochs": 2, "batch_size": 2}).json()["id"], timeout=300)
        assert train["status"] == "succeeded", train["error"]
        model_id = train["result"]["model_id"]
        pred = wait_job(c, c.post("/jobs/predict", json={"model_id": model_id}).json()["id"], timeout=300)
        assert pred["status"] == "succeeded", pred["error"]
        predicted = {
            ln["id"]: ln["text"]
            for p in c.get("/pages").json()
            for ln in c.get(f"/pages/{p['id']}").json()["lines"]
            if ln["text_source"] == "predicted"
        }
        assert len(predicted) == len(ids) - 10
        bundle = c.get("/bundle/export")
        assert bundle.status_code == 200, bundle.text
        models = c.get("/models").json()
    reference = H.snapshot_all(src)

    # a fresh project that has no base model installed: everything comes out of the bundle
    dst = Settings.for_dir(tmp_path / "dst", worker_poll_seconds=0.05, device="auto")
    with TestClient(create_app(dst)) as c:
        assert c.get("/models").json() == []
        r = c.post("/bundle/import", files={"file": ("p.htrbundle", bundle.content)})
        assert r.status_code == 200, r.text
        got = H.snapshot_all(dst)
        assert got["files"].keys() == reference["files"].keys()
        assert got["files"] == reference["files"]  # scans, working copies, PAGE-XML, base and fine-tuned model files
        for table, rows in reference["db"].items():
            assert got["db"][table] == rows, table
        assert c.get("/models").json() == models
        assert H.xml_in_sync(dst) == []
        # the imported fine-tuned model predicts with the real engine: same lines, same texts as before the export
        again = wait_job(c, c.post("/jobs/predict", json={"model_id": model_id}).json()["id"], timeout=300)
        assert again["status"] == "succeeded", again["error"]
        assert again["result"]["model_id"] == model_id and again["result"]["predicted"] == len(predicted)
        after = {
            ln["id"]: ln["text"]
            for p in c.get("/pages").json()
            for ln in c.get(f"/pages/{p['id']}").json()["lines"]
            if ln["text_source"] == "predicted"
        }
        assert after == predicted
