"""Optional local-LLM abbreviation suggestions (plan 5.6, "Later"). OFF by default.

Privacy contract (tested in tests/test_abbreviation_llm.py):

* This module talks to ONE kind of server only: an LLM server on this machine (Ollama, or a llama.cpp / OpenAI
  compatible server). The configured URL must name a loopback address (`127.0.0.0/8`, `::1` or `localhost`); anything
  else is refused when the settings are created and again before every request.
* Hostnames are never resolved through DNS: `localhost` is mapped to `127.0.0.1` here, and every other host must
  already be a literal loopback IP.
* Redirects are never followed (a loopback server cannot send the request somewhere else).
* Nothing is sent unless the feature is enabled and the user starts a suggestion job.
* The prompt contains only the abbreviation token and its own line (optionally one or two neighbouring lines, each
  capped in length). No page, document or file names, no ids, no other text.
* The model's answer is only ever turned into a `suggestions` row that the user reviews. It is never applied to a
  transcription.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import re
import ssl
from dataclasses import dataclass
from urllib.parse import urlsplit

from .abbreviations import validate_expansion

DEFAULT_URL = "http://127.0.0.1:11434"
APIS = ("ollama", "openai")  # "openai" = OpenAI-compatible chat endpoint, e.g. llama.cpp's `llama-server`
MAX_LINE_CHARS = 300  # each context line is cut to this length before it is sent
MAX_TOKEN_CHARS = 64
MAX_CONTEXT_LINES = 2  # neighbouring lines on each side, at most
MAX_CANDIDATES = 3
MAX_EXPANSION_CHARS = 120
MAX_RATIONALE_CHARS = 200
MAX_RESPONSE_BYTES = 1 << 20


class LlmConfigError(ValueError):
    """The LLM settings are not acceptable (for example a URL that is not on this machine)."""


class LlmError(Exception):
    """Talking to the local LLM server failed; the message is meant for the user."""

    def __init__(self, message: str, code: str = "llm_failed", hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.hint = hint


@dataclass(frozen=True)
class Endpoint:
    scheme: str
    host: str  # always a literal loopback IP
    port: int
    base_path: str


def parse_loopback_url(url: str) -> Endpoint:
    """Validate `url` and return where to connect. Raises LlmConfigError unless it is an http(s) URL whose host is a
    loopback address (`localhost` is accepted and becomes 127.0.0.1)."""
    try:
        parts = urlsplit(url.strip())
        host = parts.hostname
        port = parts.port
    except ValueError as e:
        raise LlmConfigError(f"not a valid URL: {e}") from None
    if parts.scheme not in ("http", "https"):
        raise LlmConfigError("the LLM URL must start with http:// or https://")
    if parts.username or parts.password:
        raise LlmConfigError("the LLM URL must not contain credentials")
    if not host:
        raise LlmConfigError("the LLM URL has no host")
    if host.lower() == "localhost":
        ip = "127.0.0.1"
    else:
        try:
            addr = ipaddress.ip_address(host)
        except ValueError:
            raise LlmConfigError(
                f"the LLM URL host '{host}' is not a loopback address; only 127.0.0.1, ::1 or localhost are allowed "
                "(no data may leave this machine)"
            ) from None
        if not addr.is_loopback:
            raise LlmConfigError(
                f"the LLM URL host '{host}' is not a loopback address; only 127.0.0.1, ::1 or localhost are allowed "
                "(no data may leave this machine)"
            )
        ip = str(addr)
    return Endpoint(parts.scheme, ip, port or (443 if parts.scheme == "https" else 80), parts.path.rstrip("/"))


# ---- prompt ----------------------------------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You help read a person's handwritten notes. The notes contain personal abbreviations. Given one abbreviation "
    "and the line it appears in, propose what it most likely stands for. Answer with JSON only, in exactly this "
    'shape: {"candidates": [{"expansion": "...", "rationale": "..."}]}. Give at most 3 candidates, the most likely '
    "first. An expansion is the short text that replaces the abbreviation; the rationale is one short sentence. If "
    'you cannot tell, answer {"candidates": []}.'
)


def _cap(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_prompt(token: str, line: str, before: list[str] | None = None, after: list[str] | None = None) -> str:
    """The user message: the token, its line and (optionally) neighbouring lines, each capped. Nothing else."""
    parts = [f'Abbreviation: "{_cap(token, MAX_TOKEN_CHARS)}"']
    for label, lines in (("Line before", before or []), ("Line after", after or [])):
        for text in [t for t in lines if t.strip()][:MAX_CONTEXT_LINES]:
            parts.append(f"{label}: {_cap(text, MAX_LINE_CHARS)}")
    parts.insert(1, f"Line: {_cap(line, MAX_LINE_CHARS)}")
    return "\n".join(parts)


@dataclass(frozen=True)
class Candidate:
    expansion: str
    rationale: str


_CTRL = re.compile(r"[\x00-\x1f\x7f]")


def parse_candidates(content: str, written: str) -> list[Candidate]:
    """Turn the model's reply into validated candidates. Tolerates code fences and prose around the JSON; anything
    that does not parse yields no candidates (never an error: small models are sloppy)."""
    text = content.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return []
    raw = data.get("candidates") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[Candidate] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("expansion"), str):
            continue
        try:
            expansion = validate_expansion(_CTRL.sub(" ", item["expansion"]))
        except ValueError:
            continue
        if len(expansion) > MAX_EXPANSION_CHARS or expansion.casefold() == written.casefold():
            continue
        if expansion.casefold() in seen:
            continue
        seen.add(expansion.casefold())
        rationale = item.get("rationale")
        rationale = _cap(_CTRL.sub(" ", rationale), MAX_RATIONALE_CHARS) if isinstance(rationale, str) else ""
        out.append(Candidate(expansion, rationale))
        if len(out) == MAX_CANDIDATES:
            break
    return out


# ---- client ----------------------------------------------------------------------------------------------------


class LlmClient:
    """Minimal client for a loopback LLM server (stdlib only; no third-party HTTP library, no redirects)."""

    def __init__(self, url: str, model: str, api: str = "ollama", timeout: float = 120.0):
        if api not in APIS:
            raise LlmConfigError(f"unknown LLM API '{api}'; use one of: {', '.join(APIS)}")
        if not model.strip():
            raise LlmConfigError("no LLM model name is configured")
        self.endpoint = parse_loopback_url(url)
        self.model = model.strip()
        self.api = api
        self.timeout = timeout

    # -- transport --

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        ep = parse_loopback_url(self._url())  # re-checked before every request
        conn: http.client.HTTPConnection
        if ep.scheme == "https":
            conn = http.client.HTTPSConnection(
                ep.host, ep.port, timeout=self.timeout, context=ssl.create_default_context()
            )
        else:
            conn = http.client.HTTPConnection(ep.host, ep.port, timeout=self.timeout)
        payload = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        try:
            conn.request(method, ep.base_path + path, body=payload, headers=headers)
            resp = conn.getresponse()
            data = resp.read(MAX_RESPONSE_BYTES + 1)
        except TimeoutError:
            raise LlmError(
                "The local LLM server did not answer in time.",
                "llm_timeout",
                "Small models are slow on a CPU; try a smaller model or raise HTR_LLM_TIMEOUT.",
            ) from None
        except OSError:
            raise LlmError(
                f"Cannot reach the local LLM server at {ep.host}:{ep.port}.",
                "llm_unreachable",
                "Start it (for Ollama: `ollama serve`) or check HTR_LLM_URL. Nothing is sent anywhere else.",
            ) from None
        finally:
            conn.close()
        if len(data) > MAX_RESPONSE_BYTES:
            raise LlmError("The LLM server sent an unexpectedly large answer.", "llm_bad_response")
        if 300 <= resp.status < 400:
            raise LlmError("The LLM server tried to redirect the request; redirects are not followed.", "llm_redirect")
        if resp.status == 404 and self.api == "ollama":
            raise LlmError(
                f"The LLM server does not know the model '{self.model}'.",
                "llm_model_missing",
                f"Download it once with `ollama pull {self.model}` (setup step; the app itself never downloads).",
            )
        if resp.status >= 400:
            raise LlmError(
                f"The LLM server answered with an error (HTTP {resp.status}).",
                "llm_server_error",
                "Check the LLM server's own log.",
            )
        try:
            parsed = json.loads(data)
        except ValueError:
            raise LlmError("The LLM server sent an answer that is not JSON.", "llm_bad_response") from None
        return parsed if isinstance(parsed, dict) else {}

    def _url(self) -> str:
        e = self.endpoint
        host = f"[{e.host}]" if ":" in e.host else e.host
        return f"{e.scheme}://{host}:{e.port}{e.base_path}"

    # -- operations --

    def check(self) -> None:
        """Raises LlmError unless the server answers (used by the health check; sends no user data)."""
        self._request("GET", "/api/tags" if self.api == "ollama" else "/v1/models")

    def chat(self, user_prompt: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
        if self.api == "ollama":
            reply = self._request(
                "POST",
                "/api/chat",
                {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.2, "num_predict": 300},
                },
            )
            content = (reply.get("message") or {}).get("content")
        else:
            reply = self._request(
                "POST",
                "/v1/chat/completions",
                {"model": self.model, "messages": messages, "temperature": 0.2, "max_tokens": 300, "stream": False},
            )
            choices = reply.get("choices") or [{}]
            content = ((choices[0] or {}).get("message") or {}).get("content")
        if not isinstance(content, str):
            raise LlmError("The LLM server answer had no text.", "llm_bad_response")
        return content

    def suggest(
        self, token: str, line: str, before: list[str] | None = None, after: list[str] | None = None
    ) -> list[Candidate]:
        return parse_candidates(self.chat(build_prompt(token, line, before, after)), token)
