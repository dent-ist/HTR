#!/usr/bin/env python3
"""Static audit: the app must not be able to make outbound network calls.

Scans the application code (backend/app), the frontend source and the built frontend (frontend/dist) and fails if it
finds

* an import or use of an HTTP/network client library (requests, httpx, urllib.request, aiohttp, http.client, ftplib,
  smtplib, websockets, socket.create_connection, ...) outside the explicit allowlist below;
* a URL that is not loopback, not an XML namespace / schema identifier, and not on the short list of known harmless
  strings;
* in the built frontend: any `fetch(`/XMLHttpRequest/WebSocket/EventSource/sendBeacon target other than the configured
  API base (those are checked by the URL rule above), plus external `<script src>`, `<link href>` or CSS `url(...)`.

Also scans the dev scripts, where only the files listed in SCRIPT_ALLOW may use network code (setup-time model
download, loopback-only developer clients).

Run `python scripts/audit_network.py [--dist]` (`make verify-offline` runs it after building the frontend).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

NETWORK_CODE = re.compile(
    r"^\s*(?:import|from)\s+(requests|httpx|urllib\.request|urllib3|aiohttp|http\.client|httplib2|ftplib|smtplib|"
    r"telnetlib|websockets?|xmlrpc\.client|paramiko|boto3|socketserver)\b"
    r"|\b(socket\.create_connection|urlopen|urllib\.request\.|http\.client\.|requests\.(get|post|put|delete|head)\()",
    re.M,
)
URL = re.compile(r"""(?:https?|wss?|ftp)://(?:\[[0-9a-f:.]+\])?[^\s'"`<>)\]},\\]*""", re.I)
LOOPBACK = re.compile(r"^(?:https?|wss?)://(?:127\.\d+\.\d+\.\d+|localhost|\[::1\]|0\.0\.0\.0)(?::\d+)?(?:/|$)", re.I)
# XML / schema identifiers and similar: they name things, nothing is fetched from them.
NAMESPACE = re.compile(
    r"^https?://(?:www\.w3\.org/|schema\.primaresearch\.org/|www\.primaresearch\.org/|purl\.org/|"
    r"xmlns\.com/|www\.opengis\.net/)",
    re.I,
)
# Text that appears in the built React bundle or in comments/docs and is never fetched.
HARMLESS = re.compile(
    r"^https?://(?:react\.dev/|reactjs\.org/|github\.com/facebook/react|json-schema\.org/|"
    r"example\.(?:com|org)|evil\.example)",
    re.I,
)

# Files that may use network code, and why. Everything else must be clean.
APP_ALLOW = {
    "backend/app/abbreviation_llm.py": "optional local-LLM client; refuses any non-loopback URL (see its tests)",
}
SCRIPT_ALLOW = {
    "scripts/get_base_model.py": "SETUP ONLY: one-time download of the pretrained base model (make base-model)",
    "scripts/make_synthetic_project.py": "developer client for a local dev server (--api defaults to 127.0.0.1)",
    "scripts/make_synthetic_rounds.py": "developer client for a local dev server (--api defaults to 127.0.0.1)",
    "scripts/offline_smoke.py": "talks only to the server it starts on 127.0.0.1",
    "scripts/audit_network.py": "this file names the patterns it looks for",
}
SKIP_DIRS = {"__pycache__", "node_modules", ".venv", "tests", "test-results", "e2e"}
TEXT_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".mjs", ".css", ".html", ".json", ".svg"}


def iter_files(root: Path):
    if not root.is_dir():
        return
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in TEXT_SUFFIXES and not (SKIP_DIRS & set(p.relative_to(root).parts)):
            if ".test." in p.name:
                continue
            yield p


def rel(p: Path) -> str:
    return p.relative_to(REPO).as_posix()


def audit_file(p: Path, allow: dict[str, str]) -> list[str]:
    text = p.read_text(errors="replace")
    name = rel(p)
    problems = []
    if p.suffix == ".py" and name not in allow:
        for m in NETWORK_CODE.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            problems.append(f"{name}:{line}: network code `{m.group().strip()}`")
    for m in URL.finditer(text):
        url = m.group().rstrip(".,;:")
        if LOOPBACK.match(url + "/") or NAMESPACE.match(url) or HARMLESS.match(url):
            continue
        if name in allow and p.suffix == ".py" and not allow.get(name, "").startswith("SETUP"):
            continue  # allowlisted network-capable file: URLs in it are covered by its own tests
        line = text.count("\n", 0, m.start()) + 1
        problems.append(f"{name}:{line}: URL {url}")
    return problems


def audit_dist(dist: Path) -> list[str]:
    problems = []
    index = dist / "index.html"
    if not index.is_file():
        return [f"{rel(dist)} has no index.html: run `npm run build` in frontend/ first"]
    html = index.read_text()
    for tag in re.findall(r"<(?:script|link|img|iframe)\b[^>]*>", html):
        for m in re.finditer(r"(?:src|href)=[\"']([^\"']+)[\"']", tag):
            ref = m.group(1)
            if re.match(r"^(?:[a-z]+:)?//", ref, re.I):
                problems.append(f"{rel(index)}: external reference {ref}")
    for p in iter_files(dist):
        text = p.read_text(errors="replace")
        for m in URL.finditer(text):
            url = m.group().rstrip(".,;:")
            if LOOPBACK.match(url + "/") or NAMESPACE.match(url) or HARMLESS.match(url):
                continue
            problems.append(f"{rel(p)}: URL {url[:100]}")
        for m in re.finditer(r"url\(\s*[\"']?(?:https?:)?//", text):
            problems.append(f"{rel(p)}: external CSS url() near offset {m.start()}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dist", action="store_true", help="also audit the built frontend (frontend/dist)")
    args = ap.parse_args()

    problems: list[str] = []
    scanned = 0
    for root, allow in (
        (REPO / "backend" / "app", APP_ALLOW),
        (REPO / "scripts", SCRIPT_ALLOW),
        (REPO / "frontend" / "src", {}),
    ):
        for p in iter_files(root):
            scanned += 1
            problems += audit_file(p, allow)
    idx = REPO / "frontend" / "index.html"
    if idx.is_file():
        scanned += 1
        problems += [x for x in audit_dist_html(idx)]
    if args.dist:
        dist = REPO / "frontend" / "dist"
        scanned += sum(1 for _ in iter_files(dist))
        problems += audit_dist(dist)
    print(f"network audit: scanned {scanned} files")
    for path, why in {**APP_ALLOW, **SCRIPT_ALLOW}.items():
        print(f"  allowed: {path}: {why}")
    if problems:
        print("FAIL:")
        for x in problems:
            print("  " + x)
        return 1
    print("OK: no outbound network code or external URLs outside the allowlist")
    return 0


def audit_dist_html(path: Path) -> list[str]:
    html = path.read_text()
    out = []
    for tag in re.findall(r"<(?:script|link|img|iframe)\b[^>]*>", html):
        for m in re.finditer(r"(?:src|href)=[\"']([^\"']+)[\"']", tag):
            if re.match(r"^(?:[a-z]+:)?//", m.group(1), re.I):
                out.append(f"{rel(path)}: external reference {m.group(1)}")
    return out


if __name__ == "__main__":
    sys.exit(main())
