"""Exclusive maintenance mode, used while a project bundle is imported.

The database file is replaced during an import, so no other request may be running or start. `Gate.begin()` turns on
maintenance mode (new requests are answered 503 by `MaintenanceMiddleware`, except /health), waits until every request
that was already running has finished (the import request itself is one of them), and refuses with 409 if they do
not finish in time or another import is already running.
"""

from __future__ import annotations

import json
import threading

from .errors import Conflict

BUSY_MESSAGE = "The project is being imported; try again in a moment."


class Gate:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._in_flight = 0
        self.maintenance = False

    def try_enter(self, exempt: bool = False) -> bool:
        """Count a starting request; False (and not counted) during maintenance unless `exempt`."""
        with self._cond:
            if self.maintenance and not exempt:
                return False
            self._in_flight += 1
            return True

    def leave(self) -> None:
        with self._cond:
            self._in_flight -= 1
            self._cond.notify_all()

    def begin(self, timeout: float = 15.0) -> None:
        """Enter maintenance mode. The caller's own request counts as one in-flight request."""
        with self._cond:
            if self.maintenance:
                raise Conflict("another project import is already running")
            self.maintenance = True
            if not self._cond.wait_for(lambda: self._in_flight <= 1, timeout):
                self.maintenance = False
                raise Conflict(
                    "other requests are still running (for example a bundle download); wait for them to finish and "
                    "try again. Nothing was changed"
                )

    def end(self) -> None:
        with self._cond:
            self.maintenance = False


class MaintenanceMiddleware:
    """Pure ASGI middleware: counts running HTTP requests and rejects new ones during maintenance."""

    def __init__(self, app, gate: Gate):
        self.app = app
        self.gate = gate

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if not self.gate.try_enter(exempt=scope.get("path") == "/health"):
            body = json.dumps({"detail": BUSY_MESSAGE}).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [(b"content-type", b"application/json"), (b"retry-after", b"5")],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        try:
            await self.app(scope, receive, send)
        finally:
            self.gate.leave()
