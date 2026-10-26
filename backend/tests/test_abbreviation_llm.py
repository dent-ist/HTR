"""The local-LLM client: loopback-only, no redirects, bounded prompts, tolerant parsing. Uses a fake Ollama / llama.cpp
server on 127.0.0.1 (tests/fake_llm.py). Real model behaviour is NOT tested here (no model is installed)."""

import http.client
import json

import pytest

from app import abbreviation_llm as llm
from app.config import Settings
from tests import netguard
from tests.fake_llm import FakeLlm

# ---- loopback only -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "host", "port", "path"),
    [
        ("http://127.0.0.1:11434", "127.0.0.1", 11434, ""),
        ("http://localhost:11434/", "127.0.0.1", 11434, ""),  # localhost is mapped, never resolved
        ("http://LOCALHOST:8080/v1", "127.0.0.1", 8080, "/v1"),
        ("http://127.0.0.5:9000", "127.0.0.5", 9000, ""),
        ("http://[::1]:8080", "::1", 8080, ""),
        ("https://127.0.0.1:8443", "127.0.0.1", 8443, ""),
        ("http://127.0.0.1", "127.0.0.1", 80, ""),
    ],
)
def test_loopback_urls_are_accepted(url, host, port, path):
    ep = llm.parse_loopback_url(url)
    assert (ep.host, ep.port, ep.base_path) == (host, port, path)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "http://example.com:11434",
        "https://api.openai.com/v1",
        "http://192.168.1.5:11434",
        "http://10.0.0.2:11434",
        "http://172.16.0.9",
        "http://8.8.8.8",
        "http://0.0.0.0:11434",
        "http://localhost.evil.com:11434",
        "http://127.0.0.1.evil.com",
        "http://evil.com/127.0.0.1",
        "http://[2001:db8::1]:11434",
        "http://user:pw@127.0.0.1:11434",
        "ftp://127.0.0.1",
        "file:///etc/passwd",
        "127.0.0.1:11434",
        "",
        "http://",
        "http://127.0.0.1:99999",
    ],
)
def test_non_loopback_or_odd_urls_are_rejected(url):
    with pytest.raises(llm.LlmConfigError):
        llm.parse_loopback_url(url)
    with pytest.raises(llm.LlmConfigError):
        llm.LlmClient(url, "m")
    with pytest.raises(llm.LlmConfigError):
        Settings.for_dir("/nonexistent", llm_url=url)  # refused at configuration time, even while the feature is off


def test_environment_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("HTR_DATA_DIR", str(tmp_path))
    s = Settings.from_env()
    assert s.llm_enabled is False and s.llm_url == "http://127.0.0.1:11434"  # off by default
    monkeypatch.setenv("HTR_LLM_ENABLED", "1")
    monkeypatch.setenv("HTR_LLM_URL", "http://localhost:8080")
    monkeypatch.setenv("HTR_LLM_API", "openai")
    monkeypatch.setenv("HTR_LLM_MODEL", "tiny")
    s = Settings.from_env()
    assert (s.llm_enabled, s.llm_url, s.llm_api, s.llm_model) == (True, "http://localhost:8080", "openai", "tiny")
    monkeypatch.setenv("HTR_LLM_URL", "http://llm.example.org:8080")
    with pytest.raises(llm.LlmConfigError, match="not a loopback address"):
        Settings.from_env()
    monkeypatch.setenv("HTR_LLM_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("HTR_LLM_API", "gpt")
    with pytest.raises(llm.LlmConfigError):
        Settings.from_env()


def test_a_tampered_endpoint_is_refused_before_any_connection():
    client = llm.LlmClient("http://127.0.0.1:11434", "m")
    client.endpoint = llm.Endpoint("http", "8.8.8.8", 11434, "")
    with pytest.raises(llm.LlmConfigError):
        client.chat("hi")
    assert netguard.violations == []  # nothing was even attempted


def test_client_has_no_third_party_http_dependency():
    import inspect

    src = inspect.getsource(llm)
    assert "import requests" not in src and "httpx" not in src and "urllib.request" not in src


# ---- prompts: only what is needed --------------------------------------------------------------------------------


def test_prompt_holds_only_the_token_and_its_line():
    p = llm.build_prompt("c/o", "pt c/o SOB since monday")
    assert p == 'Abbreviation: "c/o"\nLine: pt c/o SOB since monday'


def test_prompt_neighbours_are_optional_limited_and_capped():
    long = "word " * 200
    p = llm.build_prompt("x", long, before=["a1", "a2", "a3"], after=["b1", "", "b3"])
    lines = p.splitlines()
    assert lines[0] == 'Abbreviation: "x"' and lines[1].startswith("Line: ")
    assert [ln for ln in lines if ln.startswith("Line before")] == ["Line before: a1", "Line before: a2"]  # max 2
    assert [ln for ln in lines if ln.startswith("Line after")] == ["Line after: b1", "Line after: b3"]  # blanks skipped
    assert all(len(ln) <= llm.MAX_LINE_CHARS + len("Line before: ") for ln in lines)
    assert len(llm.build_prompt("t" * 500, "l").splitlines()[0]) <= len('Abbreviation: ""') + llm.MAX_TOKEN_CHARS


def test_prompt_collapses_whitespace_and_never_adds_other_fields():
    p = llm.build_prompt("tok", "a\n\n b\tc")
    assert p == 'Abbreviation: "tok"\nLine: a b c'


# ---- answer parsing ----------------------------------------------------------------------------------------------


def test_parse_candidates():
    good = json.dumps(
        {"candidates": [{"expansion": "complains of", "rationale": "typical in notes"}, {"expansion": "care of"}]}
    )
    got = llm.parse_candidates(good, "c/o")
    assert [(c.expansion, c.rationale) for c in got] == [("complains of", "typical in notes"), ("care of", "")]
    fenced = "Sure! ```json\n" + good + "\n``` hope that helps"
    assert len(llm.parse_candidates(fenced, "c/o")) == 2


@pytest.mark.parametrize(
    "content",
    ["", "no json here", "{", "[1,2,3]", '{"candidates": "x"}', '{"candidates": [1, null, {"expansion": 5}]}',
     '{"candidates": [{"expansion": ""}]}', '{"candidates": [{"expansion": "c/o"}]}'],
)  # fmt: skip
def test_unusable_answers_yield_no_candidates_and_no_error(content):
    assert llm.parse_candidates(content, "c/o") == []


def test_candidates_are_deduplicated_cleaned_and_limited():
    raw = {
        "candidates": [
            {"expansion": "alpha", "rationale": "r\x00\x07x"},
            {"expansion": "ALPHA"},
            {"expansion": "be\ntta  gamma"},
            {"expansion": "d" * 300},
            {"expansion": "delta"},
            {"expansion": "epsilon"},
        ]
    }
    got = llm.parse_candidates(json.dumps(raw), "abc")
    assert [c.expansion for c in got] == ["alpha", "be tta gamma", "delta"]  # dup and over-long dropped, max 3
    assert "\x00" not in got[0].rationale


# ---- against a fake server ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("api", ["ollama", "openai"])
def test_suggest_round_trip(api):
    with FakeLlm(api) as srv:
        client = llm.LlmClient(srv.url, "tiny-model", api)
        got = client.suggest("c/o", "pt c/o SOB", before=["earlier line"], after=["later line"])
        assert [c.expansion for c in got] == ["c/o expanded", "c/o alternative"]
        req = srv.requests[-1]
        assert req["method"] == "POST" and req["path"] == ("/api/chat" if api == "ollama" else "/v1/chat/completions")
        body = req["body"]
        assert body["model"] == "tiny-model" and body["stream"] is False
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        assert body["messages"][1]["content"] == (
            'Abbreviation: "c/o"\nLine: pt c/o SOB\nLine before: earlier line\nLine after: later line'
        )
        if api == "ollama":
            assert body["format"] == "json"
        client.check()
        assert srv.requests[-1]["method"] == "GET"
        assert srv.requests[-1]["path"] == ("/api/tags" if api == "ollama" else "/v1/models")


def test_base_path_is_kept():
    with FakeLlm("openai") as srv:
        # a server mounted under /v1 (llama-server style base URL): the request goes to /v1 + path
        client = llm.LlmClient(srv.url + "/", "m", "openai")
        client.chat("hello")
        assert srv.requests[-1]["path"] == "/v1/chat/completions"


def test_server_down_is_a_clear_error():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(llm.LlmError) as e:
        llm.LlmClient(f"http://127.0.0.1:{port}", "m").chat("x")
    assert e.value.code == "llm_unreachable" and "ollama serve" in e.value.hint


def test_http_errors_and_missing_model():
    with FakeLlm("ollama", status=500) as srv, pytest.raises(llm.LlmError) as e:
        llm.LlmClient(srv.url, "m").chat("x")
    assert e.value.code == "llm_server_error"
    with FakeLlm("ollama", status=404) as srv, pytest.raises(llm.LlmError) as e:
        llm.LlmClient(srv.url, "ghost:1b").chat("x")
    assert e.value.code == "llm_model_missing" and "ollama pull ghost:1b" in e.value.hint


def test_timeout_is_reported():
    with FakeLlm("ollama", delay=1.0) as srv, pytest.raises(llm.LlmError) as e:
        llm.LlmClient(srv.url, "m", timeout=0.2).chat("x")
    assert e.value.code == "llm_timeout"


def test_redirects_are_not_followed():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    hits = []

    class Redirect(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            hits.append(self.path)
            self.send_response(307)
            self.send_header("Location", "http://203.0.113.5/steal")
            self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(llm.LlmError) as e:
            llm.LlmClient(f"http://127.0.0.1:{srv.server_address[1]}", "m").chat("x")
    finally:
        srv.shutdown()
        srv.server_close()
    assert e.value.code == "llm_redirect" and hits == ["/api/chat"] and netguard.violations == []


def test_non_json_and_empty_answers():
    with FakeLlm("ollama", reply=lambda p: "just prose, no json") as srv:
        assert llm.LlmClient(srv.url, "m").suggest("x", "line x") == []
    # a server that answers 200 with a body that is not JSON at all
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Html(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<html>oops</html>")

    s = ThreadingHTTPServer(("127.0.0.1", 0), Html)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        with pytest.raises(llm.LlmError) as e:
            llm.LlmClient(f"http://127.0.0.1:{s.server_address[1]}", "m").chat("x")
    finally:
        s.shutdown()
        s.server_close()
    assert e.value.code == "llm_bad_response"


def test_connection_class_is_only_ever_used_against_loopback(monkeypatch):
    """Whatever the settings say, the host handed to http.client is a literal loopback IP."""
    seen = []
    real = http.client.HTTPConnection.__init__

    def spy(self, host, *a, **k):
        seen.append(host)
        real(self, host, *a, **k)

    monkeypatch.setattr(http.client.HTTPConnection, "__init__", spy)
    with FakeLlm() as srv:
        llm.LlmClient(srv.url.replace("127.0.0.1", "localhost"), "m").chat("x")
    assert seen == ["127.0.0.1"]
