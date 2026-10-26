"""Structured, rotating log file under <data>/logs that never contains the user's content.

What is logged: timestamps, levels, event names, numeric ids (page, line, job, model), counts, durations, status
codes and exception *class names with code locations*. What is never logged: transcribed text, predictions,
abbreviation dictionary content, scan file names (only ids), query strings, request or response bodies, exception
messages (they may quote text) and local variables. Call sites must follow the same rule; `describe_exception` is the
safe way to log an error. tests/test_logging.py checks this end to end.

Format: one JSON object per line: {"ts", "level", "logger", "event", ...fields}.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_NAME = "htr.log"
MAX_BYTES = 1_000_000
BACKUPS = 5
_HANDLER_ATTR = "_htr_file_handler"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            entry.update({k: v for k, v in fields.items() if k not in entry})
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = describe_exception(record.exc_info[1])  # never the message
        return json.dumps(entry, ensure_ascii=True, separators=(",", ":"))


def describe_exception(exc: BaseException) -> str:
    """`ValueError at services/lines.py:123 in update_line` : class and innermost app frame, but not the message
    (which may quote user text) and not the values of any variables."""
    where = ""
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        path = Path(frame.filename)
        if "app" in path.parts:
            tail = path.parts[len(path.parts) - 1 - path.parts[::-1].index("app") :]
            where = f" at {'/'.join(tail[1:])}:{frame.lineno} in {frame.name}"
            break
    return f"{type(exc).__name__}{where}"


def log_event(logger: logging.Logger, level: int, event: str, **fields) -> None:
    """Log a short event name plus structured fields (numbers and identifiers only!)."""
    logger.log(level, event, extra={"fields": fields})


def setup_logging(log_dir: Path) -> RotatingFileHandler | None:
    """Attach the rotating JSON file handler to the `app` logger (once per directory). Returns the handler, or None
    when the directory cannot be written (the app then just runs without a log file)."""
    root = logging.getLogger("app")
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        if getattr(h, _HANDLER_ATTR, False):
            root.removeHandler(h)
            h.close()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(log_dir / LOG_NAME, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8")
    except OSError:
        logging.getLogger(__name__).warning("cannot write the log file in %s; continuing without it", log_dir)
        return None
    handler.setFormatter(JsonFormatter())
    setattr(handler, _HANDLER_ATTR, True)
    try:
        os.chmod(log_dir / LOG_NAME, 0o600)
    except OSError:
        pass
    root.addHandler(handler)
    return handler


def teardown_logging() -> None:
    root = logging.getLogger("app")
    for h in list(root.handlers):
        if getattr(h, _HANDLER_ATTR, False):
            root.removeHandler(h)
            h.close()
