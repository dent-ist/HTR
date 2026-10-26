"""M2 done-when check with the real Kraken model (no mocks):
import a synthetic page through the API, segment it via the job queue, return its lines.

Run with `make test-kraken` (add `-s` to see the printed line table).
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from make_synthetic_pages import available_fonts, render_page
from PIL import Image

from app.config import Settings
from app.main import create_app
from app.pipeline import pagexml
from tests.conftest import wait_job

pytestmark = pytest.mark.kraken


def test_import_segment_and_return_lines_with_real_kraken(tmp_path):
    synth = render_page(np.random.default_rng(1), available_fonts()[0], font_size=44, n_lines=8, margin_note=True)
    buf = io.BytesIO()
    synth.image.save(buf, format="PNG")
    settings = Settings.for_dir(tmp_path / "data", device="auto", worker_poll_seconds=0.05)

    with TestClient(create_app(settings)) as client:
        doc = client.post("/documents", json={"name": "synthetic"}).json()
        up = client.post(
            "/pages", data={"document_id": str(doc["id"])}, files={"file": ("synthetic.png", buf.getvalue())}
        )
        assert up.status_code == 201, up.text
        page = up.json()
        sizes = f"{page['original_width']}x{page['original_height']} -> working {page['width']}x{page['height']}"
        print(f"\nimported page {page['id']}: {sizes}, status={page['status']}, warnings={page['warnings']}")

        job = client.post(f"/pages/{page['id']}/segment").json()
        print(f"enqueued job {job['id']} ({job['type']}, {job['status']})")
        done = wait_job(client, job["id"], timeout=300)
        print(f"job {done['id']}: {done['status']} progress={done['progress']} result={done['result']}")
        assert done["status"] == "succeeded", done["error"]

        got = client.get(f"/pages/{page['id']}").json()
        lines = got["lines"]
        print(f"page status={got['status']}, {len(lines)} lines (synthetic page has {synth.total_lines}):")
        for ln in lines:
            xs = [p[0] for p in ln["polygon"]]
            ys = [p[1] for p in ln["polygon"]]
            bbox = f"({min(xs)},{min(ys)})-({max(xs)},{max(ys)})"
            npts = f"baseline_pts={len(ln['baseline'])} polygon_pts={len(ln['polygon'])}"
            print(f"  #{ln['index']:<2} id={ln['id']:<3} {npts} bbox={bbox} text={ln['text']!r}")
        assert 0.7 * synth.total_lines <= len(lines) <= 1.3 * synth.total_lines
        assert got["status"] == "segmented" and all(len(x["polygon"]) >= 3 for x in lines)

        crop = client.get(lines[0]["crop_url"])
        assert crop.status_code == 200 and Image.open(io.BytesIO(crop.content)).height > 5

        # transcribe one line; the PAGE-XML on disk follows and Kraken's own reader accepts it
        assert (
            client.patch(f"/lines/{lines[0]['id']}", json={"text": "hello world", "is_ground_truth": True}).status_code
            == 200
        )
        xml = settings.data_dir / "pagexml" / next((settings.data_dir / "pagexml").iterdir()).name
        seg, _ = pagexml.read_pagexml(xml)
        assert seg.lines[0].text == "hello world" and seg.lines[0].is_ground_truth
        from kraken.lib.xml import XMLPage

        assert len(XMLPage(xml).to_container().lines) == len(lines)
        print(f"PAGE-XML: {xml.name} ({len(seg.lines)} lines; line 0 text={seg.lines[0].text!r}, ground truth)")
