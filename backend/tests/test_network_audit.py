"""The static no-outbound-calls audit (scripts/audit_network.py): the repository passes it, and it really catches
network code and external URLs."""

import audit_network as audit
import pytest


def test_repository_is_clean():
    problems = []
    for root, allow in (
        (audit.REPO / "backend" / "app", audit.APP_ALLOW),
        (audit.REPO / "scripts", audit.SCRIPT_ALLOW),
        (audit.REPO / "frontend" / "src", {}),
    ):
        for p in audit.iter_files(root):
            problems += audit.audit_file(p, allow)
    assert problems == []


def test_only_the_llm_client_may_use_network_code_in_the_app():
    assert set(audit.APP_ALLOW) == {"backend/app/abbreviation_llm.py"}
    assert all(k.startswith("scripts/") for k in audit.SCRIPT_ALLOW)


def scan(tmp_path, monkeypatch, name, text):
    f = tmp_path / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    monkeypatch.setattr(audit, "REPO", tmp_path)
    return audit.audit_file(f, {})


@pytest.mark.parametrize(
    "code",
    [
        "import requests\n",
        "import httpx\n",
        "from urllib.request import urlopen\n",
        "import urllib.request\n",
        "import http.client\n",
        "import socket\ns = socket.create_connection(('x', 1))\n",
        "import aiohttp\n",
        "import smtplib\n",
        "x = requests.get('http://localhost/')\n",
    ],
)
def test_network_code_is_found(tmp_path, monkeypatch, code):
    assert scan(tmp_path, monkeypatch, "app/x.py", code)


@pytest.mark.parametrize(
    "text",
    [
        'URL = "https://example.net/api"\n',
        'const u = "https://cdn.jsdelivr.net/npm/foo";\n',
        "@import url(https://fonts.googleapis.com/css?family=X);\n",
        'fetch("http://192.168.1.10:8000/x")\n',
        'ws = "wss://relay.example.org"\n',
    ],
)
def test_external_urls_are_found(tmp_path, monkeypatch, text):
    assert scan(tmp_path, monkeypatch, "src/x.ts", text)


@pytest.mark.parametrize(
    "text",
    [
        'const api = "http://127.0.0.1:8000";\n',
        'ORIGIN = "http://localhost:5173"\n',
        'xmlns="http://www.w3.org/2001/XMLSchema-instance"\n',
        'xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"\n',
        "see https://react.dev/errors/418 for details\n",
        'host = "http://[::1]:8000"\n',
    ],
)
def test_loopback_namespaces_and_known_strings_are_accepted(tmp_path, monkeypatch, text):
    assert scan(tmp_path, monkeypatch, "src/x.ts", text) == []


def test_built_frontend_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "REPO", tmp_path)
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text('<script type="module" src="/assets/app.js"></script><link href="/assets/a.css">')
    (dist / "assets" / "app.js").write_text('fetch("http://127.0.0.1:8000/x"); const e="https://react.dev/errors/1";')
    (dist / "assets" / "a.css").write_text("body{background:url(data:image/png;base64,AA)}")
    assert audit.audit_dist(dist) == []
    (dist / "index.html").write_text('<script src="https://cdn.example.com/x.js"></script>')
    (dist / "assets" / "app.js").write_text('fetch("https://tracker.example.net/ping")')
    (dist / "assets" / "a.css").write_text("@font-face{src:url(https://fonts.gstatic.com/x.woff2)}")
    found = "\n".join(audit.audit_dist(dist))
    assert "cdn.example.com" in found and "tracker.example.net" in found and "fonts.gstatic.com" in found
