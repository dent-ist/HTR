"""ASGI middleware and response helpers shared by the whole API.

* `HostGuardMiddleware` - answers only requests whose Host header names this machine (127.0.0.1, localhost, [::1],
  with an optional port). A web page on another site that rebinds its DNS name to 127.0.0.1 sends its own name in
  Host, so it is refused (421). CORS alone does not stop that; see the privacy notes in the README.
* `ErrorGuardMiddleware` - the one place that logs requests (method, path, status, duration: never the query string,
  headers or bodies) and turns an unexpected exception into the standard JSON error instead of a stack trace. It sits
  inside CORS so even those responses carry the CORS headers the browser needs to show the message.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from .failures import classify
from .logging_setup import describe_exception, log_event

log = logging.getLogger(__name__)

HOST_PATTERN = re.compile(r"^(?:127\.0\.0\.1|localhost|\[::1\])(?::\d{1,5})?$", re.IGNORECASE)
QUIET_PATHS = ("/health", "/jobs")  # polled by the UI; only failures on these are logged


def error_body(status: int, message: str, code: str, hint: str | None = None, detail: Any = None) -> dict[str, Any]:
    """The standard error body (see docs/api.md, "Errors")."""
    return {"detail": message if detail is None else detail, "code": code, "message": message, "hint": hint}


async def send_json(send, status: int, body: dict[str, Any], extra_headers: list[tuple[bytes, bytes]] | None = None):
    data = json.dumps(body).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": headers + (extra_headers or [])})
    await send({"type": "http.response.body", "body": data})


def host_allowed(host: str | None) -> bool:
    return bool(host and HOST_PATTERN.match(host.strip()))


class HostGuardMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        hosts = [v.decode("latin-1") for k, v in scope.get("headers", []) if k == b"host"]
        if len(hosts) == 1 and host_allowed(hosts[0]):
            await self.app(scope, receive, send)
            return
        log_event(log, logging.WARNING, "request refused", reason="host header", path=scope.get("path", ""))
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        missing = len(hosts) != 1
        await send_json(
            send,
            400 if missing else 421,
            error_body(
                400 if missing else 421,
                "This request was refused: it did not name this computer in its Host header.",
                "host_not_allowed",
                "The app only answers at http://127.0.0.1:8000 or http://localhost:8000 (and the same with the "
                "frontend's port). This protects your scans from web pages that try to reach the app through DNS "
                "tricks.",
            ),
        )


class ErrorGuardMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False
        status = 0
        t0 = time.monotonic()

        async def wrapped_send(message):
            nonlocal started, status
            if message["type"] == "http.response.start":
                started = True
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, wrapped_send)
        except Exception as e:  # noqa: BLE001  anything the routes did not handle
            failure = classify(e)
            log_event(
                log,
                logging.ERROR,
                "request failed",
                method=scope["method"],
                path=scope["path"],
                code=failure.code,
                error=describe_exception(e),
            )
            if started:
                raise
            status = failure.status
            await send_json(
                send,
                failure.status,
                error_body(failure.status, failure.message, failure.code, failure.hint),
                [(b"retry-after", b"3")] if failure.status == 503 else None,
            )
            return
        if status >= 400 or (scope["method"] != "GET" and not scope["path"].startswith(QUIET_PATHS)):
            log_event(
                log,
                logging.WARNING if status >= 400 else logging.INFO,
                "request",
                method=scope["method"],
                path=scope["path"],
                status=status,
                ms=round((time.monotonic() - t0) * 1000),
            )
