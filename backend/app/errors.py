"""Errors raised by the service layer; the API turns them into HTTP responses.

Every error that reaches a user has the same shape (see docs/api.md, "Errors"):

    {"detail": "<what went wrong, in plain words>", "code": "<stable machine-readable id>",
     "message": "<same text as detail; for 422 a readable summary>", "hint": "<what to do about it>" | null}

`code` is stable and safe to switch on; `message` and `hint` are for people. Stack traces never appear in a response.
"""

from __future__ import annotations


class ApiError(Exception):
    status_code = 400
    default_code = "bad_request"
    default_hint: str | None = None

    def __init__(self, detail: str, code: str | None = None, hint: str | None = None):
        super().__init__(detail)
        self.detail = detail
        self.code = code or self.default_code
        self.hint = hint if hint is not None else self.default_hint


class NotFound(ApiError):
    status_code = 404
    default_code = "not_found"


class Conflict(ApiError):
    status_code = 409
    default_code = "conflict"


class Invalid(ApiError):
    status_code = 422
    default_code = "invalid_input"


class UnsupportedMedia(ApiError):
    status_code = 415
    default_code = "unsupported_media"


class TooLarge(ApiError):
    status_code = 413
    default_code = "too_large"


class Unavailable(ApiError):
    """A temporary condition (database busy, project being imported); trying again shortly usually works."""

    status_code = 503
    default_code = "unavailable"


class InsufficientStorage(ApiError):
    status_code = 507
    default_code = "disk_full"


class JobError(ApiError):
    """A job failed for a reason the user can act on; `detail` (and `hint`) are shown to them as the job's error."""

    status_code = 500
    default_code = "job_failed"
