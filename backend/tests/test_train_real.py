"""Opt-in end-to-end check with the real Kraken engine (`make test-train`, needs `make base-model` and `make fonts`):
a tiny fine-tune plus a prediction, through the API and the job queue, on a handful of synthetic lines.

This proves the wiring (export -> Kraken training -> .mlmodel -> held-out evaluation -> predict -> PAGE-XML), not
accuracy: ten synthetic lines and two epochs teach a model nothing meaningful.
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from make_synthetic_pages import available_fonts, render_page

from app.config import REPO_ROOT, Settings
from app.main import create_app
from app.pipeline import pagexml
from app.pipeline.evaluate import evaluate
from tests.conftest import fake_blla_factory, wait_job

pytestmark = pytest.mark.train

BASE = REPO_ROOT / "data" / "models" / "base"


@pytest.mark.skipif(not list(BASE.glob("*/BASE_MODEL.json")), reason="run `make base-model` first")
def test_tiny_real_fine_tune_and_predict(tmp_path, monkeypatch):
    from app.pipeline import segment

    monkeypatch.setattr(segment, "run_blla", fake_blla_factory(n_lines=0))  # lines come from the rendered truth
    settings = Settings.for_dir(tmp_path / "data", worker_poll_seconds=0.05, device="auto")
    (settings.data_dir / "models").mkdir(parents=True)
    (settings.data_dir / "models" / "base").symlink_to(BASE)
    rng = np.random.default_rng(3)
    texts: dict[int, str] = {}  # line id -> true text
    with TestClient(create_app(settings)) as c:
        doc = c.post("/documents", json={"name": "tiny"}).json()
        for p in range(2):
            truth: list[dict] = []
            page = render_page(rng, available_fonts()[p % len(available_fonts())], font_size=44, n_lines=6,
                               margin_note=False, truth=truth)  # fmt: skip
            buf = io.BytesIO()
            page.image.save(buf, format="PNG")
            files = {"file": ("p.png", buf.getvalue())}
            pg = c.post("/pages", data={"document_id": str(doc["id"])}, files=files).json()
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
        labeled, unlabeled = ids[:10], ids[10:]
        for lid in labeled:
            assert c.patch(f"/lines/{lid}", json={"text": texts[lid], "is_ground_truth": True}).status_code == 200

        job = c.post("/jobs/train", json={"epochs": 2, "batch_size": 2}).json()
        assert job["warnings"]  # 10 lines is far below the recommended 100
        done = wait_job(c, job["id"], timeout=300)
        assert done["status"] == "succeeded", done["error"]
        r = done["result"]
        print(
            f"\ntrain: {r['train_lines']} train / {r['held_out_lines']} held out, device {r['device']}, "
            f"{r['training_seconds']}s, peak VRAM {r['peak_vram_mb']} MB, "
            f"CER before {r['base_cer']} -> after {r['cer']}"
        )
        assert r["epochs_run"] == 2 and r["cer"] is not None and r["base_cer"] is not None
        model = c.get(f"/models/{r['model_id']}").json()
        assert (settings.data_dir / model["path"]).is_file() and model["path"].endswith(".mlmodel")
        assert model["is_active"] or r["cer"] > r["base_cer"]

        pdone = wait_job(c, c.post("/jobs/predict", json={"model_id": r["model_id"]}).json()["id"], timeout=300)
        assert pdone["status"] == "succeeded", pdone["error"]
        assert pdone["result"]["lines"] == len(unlabeled)
        pairs = []
        for page in c.get("/pages").json():
            for ln in c.get(f"/pages/{page['id']}").json()["lines"]:
                if ln["is_ground_truth"]:
                    assert ln["text"] == texts[ln["id"]] and ln["text_source"] == "manual"  # untouched
                else:
                    assert ln["text_source"] == "predicted" and 0.0 <= ln["confidence"] <= 1.0
                    pairs.append((texts[ln["id"]], ln["text"]))
        print(f"predicted {len(pairs)} unlabeled lines, CER {evaluate(pairs).cer:.3f}; sample: {pairs[0]!r}")
    predicted_in_xml = 0
    for xml in (settings.data_dir / "pagexml").glob("*.xml"):
        seg, _ = pagexml.read_pagexml(xml)
        predicted_in_xml += sum(ln.text_source == "predicted" for ln in seg.lines)
    assert predicted_in_xml == len(unlabeled)  # the PAGE-XML on disk carries the predictions too
