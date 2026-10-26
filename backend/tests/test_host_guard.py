"""DNS-rebinding protection: only requests that name this machine in their Host header are answered."""

import pytest

from app.main import create_app
from app.web import host_allowed

GOOD = ["127.0.0.1", "127.0.0.1:8000", "localhost", "localhost:5173", "LOCALHOST:8000", "[::1]", "[::1]:8000"]
BAD = [
    "evil.example",
    "evil.example:8000",
    "127.0.0.1.evil.example",
    "localhost.evil.example",
    "evil.example:127.0.0.1",
    "0.0.0.0",
    "0.0.0.0:8000",
    "192.168.1.20:8000",
    "10.0.0.1",
    "[::2]",
    "localhost:",
    "localhost:99999x",
    "localhost.",
    "127.0.0.1@evil.example",
    "user@localhost",
    "",
    "localhost evil",
    "127.1",
]


@pytest.mark.parametrize("host", GOOD)
def test_loopback_hosts_are_answered(client, host):
    r = client.get("/health", headers={"Host": host})
    assert r.status_code == 200 and r.json() == {"status": "ok"}


@pytest.mark.parametrize("host", BAD)
def test_other_hosts_are_refused(client, host):
    r = client.get("/health", headers={"Host": host})
    assert r.status_code in (400, 421), host
    body = r.json()
    assert body["code"] == "host_not_allowed" and body["hint"] and body["message"] == body["detail"]
    assert "traceback" not in r.text.lower()


def test_rebinding_attack_shape_is_refused_for_every_kind_of_request(client):
    """A page on attacker.example that rebinds its name to 127.0.0.1 sends `Host: attacker.example`."""
    evil = {"Host": "attacker.example:8000", "Origin": "http://attacker.example:8000"}
    assert client.get("/pages", headers=evil).status_code == 421
    assert client.get("/documents", headers=evil).status_code == 421
    assert client.post("/documents", json={"name": "x"}, headers=evil).status_code == 421
    assert client.get("/bundle/export", headers=evil).status_code == 421
    assert client.get("/openapi.json", headers=evil).status_code == 421
    pre = client.options(
        "/lines/1",
        headers={**evil, "Access-Control-Request-Method": "PATCH", "Access-Control-Request-Headers": "content-type"},
    )
    assert pre.status_code == 421 and "access-control-allow-origin" not in pre.headers
    assert client.get("/documents").json() == []  # nothing was created by the refused POST


def test_cors_allowlist_stays_exact_and_works_with_a_loopback_host(client):
    ok = client.get("/health", headers={"Origin": "http://127.0.0.1:5173", "Host": "127.0.0.1:8000"})
    assert ok.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    other = client.get("/health", headers={"Origin": "http://evil.example", "Host": "127.0.0.1:8000"})
    assert other.status_code == 200 and "access-control-allow-origin" not in other.headers


def test_two_host_headers_are_refused(client):
    r = client.get("/health", headers=[("Host", "127.0.0.1"), ("Host", "evil.example")])
    assert r.status_code in (400, 421)


def test_missing_host_header_is_refused(settings, fake_blla):
    """HTTP/1.0 clients may omit Host; the app does not guess."""
    import asyncio

    app = create_app(settings)
    sent = []

    async def run():
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(m):
            sent.append(m)

        scope = {"type": "http", "method": "GET", "path": "/health", "headers": [], "query_string": b""}
        await app(scope, receive, send)

    asyncio.run(run())
    assert sent[0]["status"] == 400


def test_host_pattern_unit():
    assert all(host_allowed(h) for h in GOOD)
    assert not any(host_allowed(h) for h in BAD) and not host_allowed(None)


def test_server_binds_loopback_only():
    from app import main

    assert main.DEFAULT_HOST == "127.0.0.1"
    from pathlib import Path

    makefile = (Path(__file__).resolve().parents[2] / "Makefile").read_text()
    serve = [ln for ln in makefile.splitlines() if "uvicorn" in ln]
    assert serve and all("--host 127.0.0.1" in ln for ln in serve) and "0.0.0.0" not in makefile


def test_a_real_server_refuses_a_forged_host_header(tmp_path):
    """End to end over a real socket (uvicorn on an ephemeral loopback port), not only through the TestClient."""
    import socket
    import threading
    import time

    import httpx
    import uvicorn

    from app.config import Settings

    settings = Settings.for_dir(tmp_path / "data", worker_poll_seconds=0.05, device="cpu")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(settings), host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started
        base = f"http://127.0.0.1:{port}"
        assert httpx.get(f"{base}/health", timeout=5).status_code == 200
        forged = httpx.get(f"{base}/documents", headers={"Host": "attacker.example"}, timeout=5)
        assert forged.status_code == 421 and forged.json()["code"] == "host_not_allowed"
    finally:
        server.should_exit = True
        t.join(10)
