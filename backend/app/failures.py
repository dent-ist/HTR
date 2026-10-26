"""Turn any exception into something a person can act on: an HTTP status, a plain-language message, a stable code and
a hint. Used by the API's exception handlers and by the job runner, so a failure looks the same wherever it happens.

Technical details (exception class and code location, never the message or any text) go to data/logs/htr.log; the
user never sees a stack trace.
"""

from __future__ import annotations

import errno
import sqlite3
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from .errors import ApiError

LOG_HINT = "Technical details were written to data/logs/htr.log."


@dataclass(frozen=True)
class Failure:
    status: int
    message: str
    code: str
    hint: str | None = None


def is_cuda_oom(e: BaseException) -> bool:
    text = str(e).lower()
    return type(e).__name__ in ("OutOfMemory", "OutOfMemoryError") or (
        "out of memory" in text and ("cuda" in text or "cublas" in text or "cudnn" in text)
    )


def classify(e: BaseException, *, job: bool = False) -> Failure:
    """`job=True` keeps the exception text for unclassified errors (it becomes the job's visible error)."""
    if isinstance(e, ApiError):
        return Failure(e.status_code, e.detail, e.code, e.hint)
    if is_cuda_oom(e):
        return Failure(
            500,
            "The graphics card ran out of memory.",
            "cuda_oom",
            "Lower the batch size (Training page, or HTR_TRAIN_BATCH_SIZE / HTR_PREDICT_BATCH_SIZE), close other "
            "programs that use the GPU, or set the device to cpu.",
        )
    if isinstance(e, sqlite3.OperationalError | sqlite3.DatabaseError):
        text = str(e).lower()
        if "locked" in text or "busy" in text:
            return Failure(
                503,
                "The project database is busy.",
                "db_locked",
                "Another task is writing to it. Wait a few seconds and try again; if it keeps happening, make sure "
                "only one copy of the app is running.",
            )
        if "full" in text:
            return Failure(507, "The disk is full, so nothing more can be saved.", "disk_full", _disk_hint())
        if "malformed" in text or "not a database" in text or "corrupt" in text:
            return Failure(
                500,
                "The project database looks damaged.",
                "db_corrupt",
                "Stop the app and restore your last project bundle (Export page, or `scripts/htr.py bundle import`). "
                "Do not delete the data folder before you have a copy.",
            )
        if "no such table" in text or "no such column" in text:
            return Failure(
                500,
                "The project database has not been set up completely.",
                "db_not_migrated",
                "Restart the app so it can finish updating the database.",
            )
        return Failure(500, "The project database reported an error.", "db_error", LOG_HINT)
    if isinstance(e, OSError) and e.errno == errno.ENOSPC:
        return Failure(507, "The disk is full, so nothing more can be saved.", "disk_full", _disk_hint())
    if isinstance(e, UnidentifiedImageError | Image.DecompressionBombError) or (
        isinstance(e, OSError) and ("truncated" in str(e) or "cannot identify image" in str(e))
    ):
        return Failure(
            422,
            "The image file cannot be read; it may be damaged or not an image.",
            "image_unreadable",
            "Re-scan or re-export the page and import it again.",
        )
    if isinstance(e, FileNotFoundError):
        return Failure(
            404,
            "A file this needs is missing on disk.",
            "file_missing",
            "If you moved or deleted files inside the data folder, restore them from a project bundle; for a single "
            "page, delete it and import the scan again.",
        )
    if isinstance(e, PermissionError):
        return Failure(
            500,
            "The app is not allowed to write to its data folder.",
            "permission_denied",
            "Check the permissions of the data folder (HTR_DATA_DIR) and of data/logs.",
        )
    if isinstance(e, MemoryError):
        return Failure(
            500,
            "The computer ran out of memory.",
            "out_of_memory",
            "Close other programs, or import smaller scans (see HTR_MAX_IMAGE_MP).",
        )
    name = type(e).__name__
    message = f"{name}: {e}" if job and str(e) else f"Unexpected error ({name})."
    return Failure(500, message, "internal_error" if not job else "job_failed", LOG_HINT)


def _disk_hint() -> str:
    return (
        "Free some space on the drive that holds the data folder, for example by deleting project bundles you "
        "exported earlier."
    )
