"""Errors raised by the service layer; the API turns them into HTTP responses."""


class ApiError(Exception):
    status_code = 400

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class NotFound(ApiError):
    status_code = 404


class Conflict(ApiError):
    status_code = 409


class Invalid(ApiError):
    status_code = 422


class UnsupportedMedia(ApiError):
    status_code = 415


class TooLarge(ApiError):
    status_code = 413


class JobError(ApiError):
    """A job failed for a reason the user can act on; `detail` is shown to them verbatim as the job's error."""

    status_code = 500
