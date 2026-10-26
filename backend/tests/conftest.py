import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def fake_blla_factory(n_lines=3, fail_on=None):
    """Replacement for segment.run_blla: n horizontal lines, optionally raising for images of a given width."""

    def fake(img, device):
        if fail_on is not None and img.width == fail_on:
            raise ValueError("simulated segmentation failure")
        h = img.height // (n_lines + 1)
        lines = []
        for i in range(n_lines):
            y = h * (i + 1)
            lines.append(
                SimpleNamespace(
                    baseline=[(50, y), (img.width - 50, y)],
                    boundary=[(50, y - 15), (img.width - 50, y - 15), (img.width - 50, y + 10), (50, y + 10)],
                )
            )
        return SimpleNamespace(lines=lines)

    return fake


@pytest.fixture
def fake_blla(monkeypatch):
    from app.pipeline import segment

    def install(**kw):
        monkeypatch.setattr(segment, "run_blla", fake_blla_factory(**kw))

    return install


# ---- M2: API fixtures ---------------------------------------------------------------------------------------------

import io  # noqa: E402
import time  # noqa: E402
from functools import lru_cache  # noqa: E402

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@lru_cache(maxsize=32)
def _scan_bytes(fmt: str, seed: int, width: int, height: int) -> bytes:
    from make_synthetic_pages import available_fonts, render_page

    page = render_page(
        np.random.default_rng(seed), available_fonts()[0], width=width, height=height, font_size=26, n_lines=6
    )
    buf = io.BytesIO()
    page.image.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def scan_bytes():
    """scan_bytes(fmt="PNG", seed=0, width=700, height=900) -> bytes of a synthetic page (cached across tests)."""

    def make(fmt="PNG", seed=0, width=700, height=900):
        return _scan_bytes(fmt, seed, width, height)

    return make


@pytest.fixture
def settings(tmp_path):
    from app.config import Settings

    return Settings.for_dir(tmp_path / "data", worker_poll_seconds=0.02, device="cpu")


@pytest.fixture
def client(settings, fake_blla):
    """TestClient with the app lifespan running (migrated temp DB + real worker thread); segmentation is mocked."""
    from app.main import create_app

    fake_blla(n_lines=3)
    with TestClient(create_app(settings)) as c:
        yield c


def wait_job(client, job_id, timeout=30.0, until=("succeeded", "failed", "cancelled")):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in until:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {job}")


@pytest.fixture
def api(client, scan_bytes):
    """Helpers bound to the client: new_document(), import_page(), segmented_page()."""

    class Api:
        def __init__(self):
            self.client = client
            self.n = 0

        def new_document(self, name="notes"):
            r = client.post("/documents", json={"name": name})
            assert r.status_code == 201, r.text
            return r.json()

        def import_page(self, document_id=None, name="page.png", data=None, fmt="PNG"):
            if document_id is None:
                document_id = self.new_document()["id"]
            if data is None:
                self.n += 1
                data = scan_bytes(fmt, seed=self.n)
            r = client.post("/pages", data={"document_id": str(document_id)}, files={"file": (name, data)})
            assert r.status_code in (200, 201), r.text
            return r.json()

        def segmented_page(self, **kw):
            page = self.import_page(**kw)
            job = client.post(f"/pages/{page['id']}/segment").json()
            assert wait_job(client, job["id"])["status"] == "succeeded"
            return client.get(f"/pages/{page['id']}").json()

    return Api()


# ---- M4: training / prediction with fake engines ------------------------------------------------------------------


@pytest.fixture
def fake_ml(monkeypatch, settings):
    """Replace the Kraken recognizer and trainer in the job handlers by fast fakes (tests/ml_fakes.py) and put a
    fake pretrained base model on disk, as `make base-model` would."""
    import json

    from app.worker import ml_jobs
    from tests import ml_fakes

    ml_fakes.FakeRecognizer.instances.clear()
    ml_fakes.FakeRecognizer.on_predict = None
    ml_fakes.FakeTrainer.reset()
    monkeypatch.setattr(
        ml_jobs, "load_recognizer", lambda engine, path, device="auto", batch_size=4: ml_fakes.FakeRecognizer(path)
    )
    monkeypatch.setattr(ml_jobs, "make_trainer", lambda engine: ml_fakes.FakeTrainer())
    base = settings.data_dir / "models" / "base" / "test-base"
    base.mkdir(parents=True)
    (base / "base.mlmodel").write_bytes(b"fake base model")
    (base / "BASE_MODEL.json").write_text(
        json.dumps({"name": "test-base", "doi": "10.0/test", "license": "Apache-2.0", "file": "base.mlmodel"})
    )
    return ml_fakes
