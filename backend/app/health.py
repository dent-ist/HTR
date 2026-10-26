"""Startup self-checks, surfaced by GET /health/details and logged at startup.

Each check answers one question a user would otherwise have to debug from a stack trace: is the database up to date,
can the app write to its data folder, is there room left on the disk, is the GPU usable, is a base model installed, is
the worker thread alive, (when enabled) does the local LLM answer. Every check returns `status` ok | warn | error, a
plain `message` and, when something is off, a `hint` that says what to do.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from .abbreviation_llm import LlmClient, LlmConfigError, LlmError
from .config import APP_VERSION, Settings
from .db import LATEST_VERSION
from .db.migrations import current_version
from .logging_setup import log_event
from .services import models as models_svc

log = logging.getLogger(__name__)

_GPU_LOCK = threading.Lock()
_GPU_CACHE: dict[str, Any] | None = None


def probe_gpu(refresh: bool = False) -> dict[str, Any]:
    """What PyTorch can see (cached: importing torch takes a few seconds). Never raises."""
    global _GPU_CACHE
    with _GPU_LOCK:
        if _GPU_CACHE is not None and not refresh:
            return _GPU_CACHE
        info: dict[str, Any] = {"torch": None, "cuda_build": None, "available": False}
        try:
            import torch

            info["torch"] = torch.__version__
            info["cuda_build"] = torch.version.cuda
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                capability = f"sm_{props.major}{props.minor}"
                info.update(
                    available=True,
                    name=props.name,
                    memory_mb=int(props.total_memory / 1024 / 1024),
                    capability=capability,
                    arch_supported=capability in torch.cuda.get_arch_list(),
                )
        except Exception as e:  # noqa: BLE001  a broken torch install must not break the health page
            info["error"] = type(e).__name__
        _GPU_CACHE = info
        return info


def _check(id_: str, label: str, status: str, message: str, hint: str | None = None) -> dict[str, Any]:
    return {"id": id_, "label": label, "status": status, "message": message, "hint": hint}


def check_database(conn: sqlite3.Connection) -> dict:
    try:
        version = current_version(conn)
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
    except sqlite3.Error as e:
        text = str(e).lower()
        if "locked" in text or "busy" in text:
            return _check("database", "Database", "warn", "The database is busy right now.", "Try again in a moment.")
        return _check(
            "database",
            "Database",
            "error",
            "The project database cannot be read.",
            "Stop the app and restore your last project bundle; do not delete the data folder first.",
        )
    if version != LATEST_VERSION:
        return _check(
            "database",
            "Database",
            "error",
            f"The database is at schema version {version}, the app expects {LATEST_VERSION}.",
            "Restart the app so it can finish updating the database.",
        )
    return _check("database", "Database", "ok", f"Up to date (schema version {version}).")


def check_data_dir(settings: Settings) -> dict:
    d = settings.data_dir
    probe = d / f".write-test-{uuid.uuid4().hex[:8]}"
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok")
        probe.unlink()
    except OSError:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        return _check(
            "data_dir",
            "Data folder",
            "error",
            "The app cannot write to its data folder.",
            "Check the permissions of the folder set by HTR_DATA_DIR (default: the data/ folder of the project) and "
            "that the disk is not full or read-only.",
        )
    return _check("data_dir", "Data folder", "ok", "Writable.")


def check_disk(settings: Settings) -> dict:
    try:
        where = settings.data_dir if settings.data_dir.exists() else Path(settings.data_dir.anchor)
        free = shutil.disk_usage(where).free
    except OSError:
        return _check("disk", "Disk space", "warn", "Could not measure the free disk space.")
    gb = free / 1024**3
    msg = f"{gb:.1f} GB free on the drive that holds the data folder."
    hint = "Free some space (for example delete project bundles you exported earlier) before importing or training."
    if free < settings.disk_error_bytes:
        return _check("disk", "Disk space", "error", msg + " Saving may fail.", hint)
    if free < settings.disk_warn_bytes:
        return _check("disk", "Disk space", "warn", msg + " That is getting low.", hint)
    return _check("disk", "Disk space", "ok", msg)


def check_gpu(settings: Settings) -> dict:
    if settings.device == "cpu":
        return _check("gpu", "GPU", "ok", "CPU mode is selected (HTR_DEVICE=cpu); training and prediction are slow.")
    g = probe_gpu()
    if g.get("available"):
        text = f"{g['name']}, {g['memory_mb']} MB, CUDA {g['cuda_build']}, PyTorch {g['torch']}."
        if not g.get("arch_supported", True):
            return _check(
                "gpu",
                "GPU",
                "warn",
                text + f" This PyTorch build has no kernels for {g['capability']}.",
                "Use the pinned CUDA 12.6 wheels (`make setup`); newer CUDA builds dropped older GPUs such as the "
                "GTX 1050 Ti.",
            )
        return _check("gpu", "GPU", "ok", text)
    if g.get("error"):
        return _check("gpu", "GPU", "warn", "PyTorch could not be loaded to look for a GPU.", "Run `make check`.")
    return _check(
        "gpu",
        "GPU",
        "warn",
        "No usable GPU (CUDA) was found. Everything still works, on the CPU, but training and prediction are much "
        "slower.",
        "Run `make check`. Under WSL2 you need the NVIDIA Windows driver; no CUDA toolkit is installed inside Linux.",
    )


def check_base_model(conn: sqlite3.Connection, settings: Settings) -> dict:
    try:
        models_svc.ensure_registry(conn, settings)
        row = models_svc.active_row(conn)
    except sqlite3.Error:
        return _check("base_model", "Recognition model", "warn", "Could not read the model registry.")
    if row is None:
        return _check(
            "base_model",
            "Recognition model",
            "warn",
            "No recognition model is installed, so training and prediction cannot start.",
            "Run `make base-model` once (it downloads the pretrained model; the app itself never goes online).",
        )
    if not models_svc.model_file(settings, row).is_file():
        return _check(
            "base_model",
            "Recognition model",
            "error",
            f"The file of the active model is missing ({row['path']}).",
            "Restore it from a project bundle, activate another model on the Training page, or run `make base-model`.",
        )
    return _check("base_model", "Recognition model", "ok", f"Active model: {row['name'] or row['path']}.")


def check_worker(worker: Any) -> dict:
    if worker is not None and worker.running:
        return _check("worker", "Job worker", "ok", "Running.")
    return _check(
        "worker",
        "Job worker",
        "error",
        "The background job worker is not running, so segmentation, training and prediction will not start.",
        "Restart the app (`make dev`).",
    )


def check_llm(settings: Settings) -> dict | None:
    if not settings.llm_enabled:
        return None
    try:
        LlmClient(settings.llm_url, settings.llm_model, settings.llm_api, min(settings.llm_timeout, 3.0)).check()
    except LlmConfigError as e:
        return _check("llm", "Local LLM", "error", str(e))
    except LlmError as e:
        return _check("llm", "Local LLM", "warn", e.message, e.hint)
    return _check("llm", "Local LLM", "ok", f"Answering on this computer ({settings.llm_model}).")


def run_checks(conn: sqlite3.Connection, settings: Settings, worker: Any, gpu: bool = True) -> dict[str, Any]:
    checks = [check_database(conn), check_data_dir(settings), check_disk(settings)]
    if gpu:
        checks.append(check_gpu(settings))
    checks += [check_base_model(conn, settings), check_worker(worker)]
    llm = check_llm(settings)
    if llm:
        checks.append(llm)
    statuses = {c["status"] for c in checks}
    overall = "error" if "error" in statuses else "warning" if "warn" in statuses else "ok"
    return {"status": overall, "version": APP_VERSION, "checks": checks}


def log_startup(conn: sqlite3.Connection, settings: Settings, worker: Any) -> None:
    """Write the cheap checks to the log at startup (the GPU probe is left to the first /health/details call)."""
    result = run_checks(conn, settings, worker, gpu=False)
    log_event(log, logging.INFO, "startup", version=APP_VERSION, status=result["status"], device=settings.device)
    for c in result["checks"]:
        if c["status"] != "ok":
            log_event(log, logging.WARNING, "startup check", check=c["id"], status=c["status"])
