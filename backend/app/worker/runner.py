"""In-process worker: one thread, one job at a time (the GPU is shared with nothing else)."""

from __future__ import annotations

import logging
import threading

from ..config import Settings
from ..db import connect
from . import queue
from .handlers import HANDLERS, Handler, JobContext

log = logging.getLogger(__name__)


class Worker:
    def __init__(self, settings: Settings, handlers: dict[str, Handler] | None = None):
        self.settings = settings
        self.handlers = HANDLERS if handlers is None else handlers
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- lifecycle ---------------------------------------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="htr-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> bool:
        """Ask the loop to stop and wait. A job in progress is allowed to finish; returns False if it did not
        within `timeout` (the daemon thread is then abandoned and the job is recovered as failed on next start)."""
        self._stop.set()
        self._wake.set()
        t = self._thread
        if t is None:
            return True
        t.join(timeout)
        return not t.is_alive()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def wake(self) -> None:
        """A job was enqueued: do not wait for the next poll."""
        self._wake.set()

    # ---- execution ---------------------------------------------------------------------------------------------

    def _loop(self) -> None:
        conn = connect(self.settings.db_path)
        try:
            while not self._stop.is_set():
                try:
                    ran = self.run_once(conn)
                except Exception:  # noqa: BLE001  the loop must survive anything (e.g. a locked database)
                    log.exception("worker loop error")
                    ran = False
                if not ran:
                    self._wake.wait(self.settings.worker_poll_seconds)
                    self._wake.clear()
        finally:
            conn.close()

    def run_once(self, conn=None) -> bool:
        """Claim and run the oldest queued job. Returns False if there was none. (Also used directly by tests.)"""
        own = conn is None
        conn = conn or connect(self.settings.db_path)
        try:
            job = queue.claim_next(conn)
            if job is None:
                return False
            handler = self.handlers.get(job["type"])
            try:
                if handler is None:
                    raise NotImplementedError(f"no handler for job type '{job['type']}'")
                result = handler(JobContext(job, conn, self.settings))
            except Exception as e:  # noqa: BLE001  any handler error becomes the job's error message
                if conn.in_transaction:
                    conn.rollback()
                log.warning("job %s (%s) failed: %s", job["id"], job["type"], e)
                queue.fail(conn, job["id"], _describe(e))
            else:
                queue.finish(conn, job["id"], result)
            return True
        finally:
            if own:
                conn.close()


def _describe(e: Exception) -> str:
    msg = str(getattr(e, "detail", None) or e)
    return msg if isinstance(e, NotImplementedError) or hasattr(e, "detail") else f"{type(e).__name__}: {msg}"
