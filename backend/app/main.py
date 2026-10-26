"""FastAPI entrypoint: REST API, SQLite database, in-process job worker."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.abbreviations_api import router as abbreviations_router
from .api.export_api import router as export_router
from .api.routes import router
from .config import APP_VERSION, CORS_ORIGINS, Settings
from .db import connect, migrate
from .errors import ApiError
from .maintenance import Gate, MaintenanceMiddleware
from .projectlock import try_lock
from .services import models as models_svc
from .worker import queue
from .worker.runner import Worker

log = logging.getLogger(__name__)

# The server must only ever listen on loopback (see Makefile `serve`).
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        app.state.lock = try_lock(settings.data_dir)  # keeps `scripts/htr.py bundle import` out while we run
        if app.state.lock is None:
            log.warning("another process already holds the lock on %s (is a second server running?)", settings.data_dir)
        conn = connect(settings.db_path)
        try:
            migrate(conn)
            models_svc.register_base_models(conn, settings)  # pretrained models downloaded by `make base-model`
            recovered = queue.fail_interrupted(conn)  # jobs left 'running' by a crash will never finish
            if recovered:
                log.warning("marked %d interrupted job(s) as failed", recovered)
        finally:
            conn.close()
        worker = Worker(settings)
        app.state.worker = worker
        worker.start()
        try:
            yield
        finally:
            if not worker.stop():
                log.warning("worker still busy at shutdown; its job will be marked failed on next start")
            if app.state.lock is not None:
                app.state.lock.close()

    app = FastAPI(title="HTR", version=APP_VERSION, lifespan=lifespan)
    app.state.settings = settings
    app.state.gate = Gate()
    app.add_middleware(MaintenanceMiddleware, gate=app.state.gate)  # added first = inside CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )

    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Drop the echoed input: it can hold values (NaN) that JSON cannot encode, and the client knows what it sent.
        detail = [{"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
        return JSONResponse({"detail": detail}, status_code=422)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(router)
    app.include_router(abbreviations_router)
    app.include_router(export_router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)
