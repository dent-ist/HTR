"""FastAPI entrypoint: REST API, SQLite database, in-process job worker."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import health as health_checks
from .api.abbreviations_api import router as abbreviations_router
from .api.export_api import router as export_router
from .api.routes import router
from .api.suggestions_api import router as suggestions_router
from .config import APP_VERSION, CORS_ORIGINS, Settings
from .db import connect, migrate
from .errors import ApiError
from .logging_setup import log_event, setup_logging, teardown_logging
from .maintenance import Gate, MaintenanceMiddleware
from .projectlock import try_lock
from .services import models as models_svc
from .web import ErrorGuardMiddleware, HostGuardMiddleware, error_body
from .worker import queue
from .worker.runner import Worker

log = logging.getLogger(__name__)

# The server must only ever listen on loopback (see Makefile `serve`).
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

_HTTP_CODES = {404: "not_found", 405: "method_not_allowed", 413: "too_large", 415: "unsupported_media"}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(settings.log_dir)
        app.state.lock = try_lock(settings.data_dir)  # keeps `scripts/htr.py bundle import` out while we run
        if app.state.lock is None:
            log_event(log, logging.WARNING, "another process holds the data directory lock (second server running?)")
        conn = connect(settings.db_path)
        try:
            migrate(conn)
            models_svc.register_base_models(conn, settings)  # pretrained models downloaded by `make base-model`
            recovered = queue.fail_interrupted(conn)  # jobs left 'running' by a crash will never finish
            if recovered:
                log_event(log, logging.WARNING, "marked interrupted jobs as failed", count=recovered)
        finally:
            conn.close()
        worker = Worker(settings)
        app.state.worker = worker
        worker.start()
        conn = connect(settings.db_path)
        try:
            health_checks.log_startup(conn, settings, worker)
        finally:
            conn.close()
        try:
            yield
        finally:
            if not worker.stop():
                log_event(log, logging.WARNING, "worker still busy at shutdown; its job is marked failed on next start")
            if app.state.lock is not None:
                app.state.lock.close()
            teardown_logging()

    app = FastAPI(title="HTR", version=APP_VERSION, lifespan=lifespan)
    app.state.settings = settings
    app.state.gate = Gate()
    # Middleware order: the last one added is the outermost. Request path: host check -> CORS -> error guard / request
    # log -> maintenance gate -> routes.
    app.add_middleware(MaintenanceMiddleware, gate=app.state.gate)
    app.add_middleware(ErrorGuardMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )
    app.add_middleware(HostGuardMiddleware)  # DNS-rebinding protection; also covers CORS preflights

    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(error_body(exc.status_code, exc.detail, exc.code, exc.hint), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Drop the echoed input: it can hold values (NaN) that JSON cannot encode, and the client knows what it sent.
        detail = [{"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
        parts = []
        for e in detail:
            where = ".".join(str(p) for p in (e["loc"] or []) if p not in ("body", "query", "path"))
            parts.append(f"{where}: {e['msg']}" if where else str(e["msg"]))
        message = "; ".join(parts) or "The request is not valid."
        body = error_body(422, message, "validation_error", "Check the values you entered and try again.", detail)
        return JSONResponse(body, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, "http_error")
        message = str(exc.detail)
        hint = "Check the address; the interactive API docs are at /docs." if exc.status_code == 404 else None
        return JSONResponse(
            error_body(exc.status_code, message, code, hint), status_code=exc.status_code, headers=exc.headers
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/details", tags=["health"])
    def health_details(request: Request) -> dict:
        """Startup self-checks: database, data folder, free disk, GPU/CUDA, base model, worker (and the local LLM
        when enabled). `status` is ok, warning or error; every non-ok check carries a hint."""
        conn = connect(settings.db_path)
        try:
            return health_checks.run_checks(conn, settings, getattr(request.app.state, "worker", None))
        finally:
            conn.close()

    app.include_router(router)
    app.include_router(suggestions_router)  # before abbreviations_router: /abbreviations/suggestions vs /{id}
    app.include_router(abbreviations_router)
    app.include_router(export_router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)
