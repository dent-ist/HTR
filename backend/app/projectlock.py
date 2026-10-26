"""An advisory lock that says "a server is running on this data directory".

The server holds it for its whole life; the command-line `bundle import` takes it too, so the two can never replace the
database file under each other. (`bundle export` needs no lock: it only reads, through the SQLite backup API.)
"""

from __future__ import annotations

from pathlib import Path
from typing import IO

LOCK_NAME = ".htr-server.lock"


def try_lock(data_dir: Path) -> IO[str] | None:
    """The open lock file (keep it referenced; closing it releases the lock), or None if somebody else holds it or
    the platform has no flock."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover  (Windows)
        return None
    data_dir.mkdir(parents=True, exist_ok=True)
    f = open(data_dir / LOCK_NAME, "a+")  # noqa: SIM115  the handle is the lock
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f
