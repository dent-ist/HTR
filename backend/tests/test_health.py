"""Startup self-checks behind GET /health/details."""

import shutil
from collections import namedtuple

import pytest
from fastapi.testclient import TestClient

from app import health
from app.config import Settings
from app.db import connect, migrate
from app.main import create_app
from tests.fake_llm import FakeLlm

GPU_OK = {"torch": "2.9.1", "cuda_build": "12.6", "available": True, "name": "NVIDIA GeForce GTX 1050 Ti",
          "memory_mb": 4096, "capability": "sm_61", "arch_supported": True}  # fmt: skip
GPU_NONE = {"torch": "2.9.1", "cuda_build": "12.6", "available": False}
Usage = namedtuple("Usage", "total used free")


@pytest.fixture
def gpu(monkeypatch):
    def set_(info):
        monkeypatch.setattr(health, "probe_gpu", lambda refresh=False: info)

    set_(GPU_OK)
    return set_


def checks(client):
    r = client.get("/health/details")
    assert r.status_code == 200, r.text
    body = r.json()
    return body, {c["id"]: c for c in body["checks"]}


def test_details_shape_and_healthy_state(client, gpu, fake_ml, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(10**12, 0, 80 * 1024**3))  # the temp dir may be small
    body, by_id = checks(client)
    assert body["status"] == "ok" and body["version"]
    assert set(by_id) == {"database", "data_dir", "disk", "gpu", "base_model", "worker"}
    for c in by_id.values():
        assert c["status"] == "ok" and c["message"] and c["label"], c
        assert c["hint"] is None
    assert "Up to date" in by_id["database"]["message"] and "free" in by_id["disk"]["message"]
    assert client.get("/health").json() == {"status": "ok"}  # the cheap probe is unchanged


def test_a_visible_gpu_is_described(settings, gpu):
    c = health.check_gpu(Settings.for_dir(settings.data_dir, device="auto"))
    assert c["status"] == "ok" and "GTX 1050 Ti" in c["message"] and "4096 MB" in c["message"]


def test_no_gpu_is_a_warning_with_a_hint_not_an_error(client, gpu, fake_ml):
    gpu(GPU_NONE)
    settings = Settings.for_dir(client.app.state.settings.data_dir, device="auto")  # the fixture runs with cpu
    c = health.check_gpu(settings)
    assert c["status"] == "warn" and "CPU" in c["message"] and "make check" in c["hint"]
    assert c["hint"] and "WSL2" in c["hint"]


def test_cpu_mode_is_not_reported_as_a_problem(client, gpu, fake_ml):
    gpu(GPU_NONE)
    _, by_id = checks(client)  # the fixture's settings use device="cpu"
    assert by_id["gpu"]["status"] == "ok" and "CPU mode" in by_id["gpu"]["message"]


def test_gpu_without_kernels_for_its_architecture_is_flagged(settings):
    health._GPU_CACHE = {**GPU_OK, "arch_supported": False, "capability": "sm_61"}
    try:
        c = health.check_gpu(Settings.for_dir(settings.data_dir, device="auto"))
    finally:
        health._GPU_CACHE = None
    assert c["status"] == "warn" and "sm_61" in c["message"] and "12.6" in c["hint"]


@pytest.mark.parametrize(
    ("capability", "arch_list", "expected"),
    [
        # a GTX 1050 Ti on the cu126 wheels
        ("sm_61", ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"], True),
        ("sm_61", ["sm_61"], True),
        ("sm_75", ["sm_70"], True),  # a 7.0 binary runs on 7.5
        ("sm_61", ["sm_70", "sm_75"], False),  # newer-only builds dropped Pascal
        ("sm_61", ["sm_50"], False),  # a different major version does not run
        ("sm_86", ["sm_80", "compute_80"], True),
        ("sm_61", ["compute_50"], True),  # PTX runs on newer GPUs
        ("sm_61", ["compute_70"], False),
        ("sm_100", ["sm_90"], False),
        ("weird", [], True),  # unknown capability format: no false alarm
    ],
)
def test_architecture_compatibility_rule(capability, arch_list, expected):
    assert health.arch_supported(capability, arch_list) is expected


def test_a_broken_torch_does_not_break_the_health_page(settings, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *a, **k)

    health._GPU_CACHE = None
    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        info = health.probe_gpu(refresh=True)
    finally:
        monkeypatch.undo()
        health._GPU_CACHE = None
    assert info["available"] is False and info["error"] == "ImportError"


def test_missing_base_model_is_a_warning(client, gpu):
    body, by_id = checks(client)  # no fake_ml: no base model on disk
    assert by_id["base_model"]["status"] == "warn" and "make base-model" in by_id["base_model"]["hint"]
    assert body["status"] == "warning"


def test_active_model_file_gone_is_an_error(fake_ml, client, gpu, settings):
    (settings.data_dir / "models" / "base" / "test-base" / "base.mlmodel").unlink()
    body, by_id = checks(client)
    assert by_id["base_model"]["status"] == "error" and "missing" in by_id["base_model"]["message"]
    assert body["status"] == "error"


@pytest.mark.parametrize(
    ("free_gb", "status"),
    [(80, "ok"), (4.9, "warn"), (1, "warn"), (0.4, "error"), (0.01, "error")],
)
def test_free_disk_thresholds(settings, monkeypatch, free_gb, status):
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(10**12, 0, int(free_gb * 1024**3)))
    c = health.check_disk(settings)
    assert c["status"] == status and f"{free_gb:.1f} GB" in c["message"]
    assert (c["hint"] is None) == (status == "ok")


def test_unwritable_data_dir_is_an_error(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    c = health.check_data_dir(Settings.for_dir(blocker / "data"))
    assert c["status"] == "error" and "HTR_DATA_DIR" in c["hint"]


def test_database_not_migrated_is_an_error(tmp_path):
    conn = connect(tmp_path / "x.db")
    c = health.check_database(conn)
    assert c["status"] == "error" and "schema version 0" in c["message"] and "Restart" in c["hint"]
    migrate(conn)
    assert health.check_database(conn)["status"] == "ok"


def test_dead_worker_is_an_error(client, gpu, fake_ml):
    assert client.app.state.worker.stop()
    body, by_id = checks(client)
    assert by_id["worker"]["status"] == "error" and "make dev" in by_id["worker"]["hint"]
    assert body["status"] == "error"


def test_llm_check_appears_only_when_enabled(client, gpu, fake_ml, tmp_path):
    _, by_id = checks(client)
    assert "llm" not in by_id
    with FakeLlm() as llm:
        on = Settings.for_dir(tmp_path / "d2", device="cpu", llm_enabled=True, llm_url=llm.url)
        with TestClient(create_app(on)) as c:
            _, by_id = checks(c)
            assert by_id["llm"]["status"] == "ok" and on.llm_model in by_id["llm"]["message"]
        assert llm.requests and llm.requests[0]["path"] == "/api/tags" and llm.requests[0]["body"] is None
    dead = Settings.for_dir(tmp_path / "d3", device="cpu", llm_enabled=True, llm_url="http://127.0.0.1:9")
    with TestClient(create_app(dead)) as c:
        _, by_id = checks(c)
        assert by_id["llm"]["status"] == "warn" and "ollama serve" in by_id["llm"]["hint"]


def test_details_is_not_available_with_a_foreign_host_header(client):
    assert client.get("/health/details", headers={"Host": "evil.example"}).status_code == 421
