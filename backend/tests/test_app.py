from pathlib import Path

from fastapi.testclient import TestClient

from app import main

MAKEFILE = Path(__file__).resolve().parents[2] / "Makefile"


def test_health():
    r = TestClient(main.app).get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_default_host_is_loopback():
    assert main.DEFAULT_HOST == "127.0.0.1"


def test_makefile_serve_binds_loopback_only():
    text = MAKEFILE.read_text()
    assert "--host 127.0.0.1" in text
    assert "0.0.0.0" not in text
