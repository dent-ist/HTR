"""Runtime settings. Everything is overridable through environment variables so tests can use a temp directory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .abbreviation_llm import APIS as LLM_APIS
from .abbreviation_llm import DEFAULT_URL as LLM_DEFAULT_URL
from .abbreviation_llm import LlmConfigError, parse_loopback_url
from .pipeline.preprocess import DEFAULT_TARGET_LINE_HEIGHT
from .pipeline.workflow import Workspace

APP_VERSION = "0.3.0"
REPO_ROOT = Path(__file__).resolve().parents[2]

# Vite dev server origins only (plan section 4: nothing else may call the API from a browser).
CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

# Host header values the API answers to (DNS-rebinding protection; the server also only binds 127.0.0.1).
ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]")

GB = 1024**3


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_path: Path
    max_upload_bytes: int = 100 * 1024 * 1024
    # A decoded scan is held in memory several times over while it is processed; 80 megapixels is about A3 at 600 DPI.
    max_image_pixels: int = 80_000_000
    device: str = "auto"  # kraken device: auto | cpu | cuda:0
    worker_poll_seconds: float = 0.25
    target_line_height: float = DEFAULT_TARGET_LINE_HEIGHT
    # Training / prediction defaults, sized for a 4 GB GPU (see README "Hardware"). Requests may override batch size.
    train_batch_size: int = 4
    train_workers: int = 0  # dataloader worker processes; 0 = load in the training process (safest on 16 GB RAM)
    predict_batch_size: int = 4
    # Startup self-check thresholds for the free space on the data volume.
    disk_warn_bytes: int = 5 * GB
    disk_error_bytes: int = GB // 2
    # Optional local-LLM abbreviation suggestions: OFF unless enabled; the URL must be a loopback address.
    llm_enabled: bool = False
    llm_url: str = LLM_DEFAULT_URL
    llm_api: str = "ollama"  # ollama | openai (OpenAI-compatible chat endpoint, e.g. llama.cpp's llama-server)
    llm_model: str = "llama3.2:3b"
    llm_timeout: float = 120.0
    llm_context_lines: int = 1  # neighbouring lines (each side) sent along with the line that holds the abbreviation

    def __post_init__(self) -> None:
        parse_loopback_url(self.llm_url)  # refuses a non-loopback URL even while the feature is off
        if self.llm_api not in LLM_APIS:
            raise LlmConfigError(f"HTR_LLM_API must be one of: {', '.join(LLM_APIS)}")
        if not 0 <= self.llm_context_lines <= 2:
            raise LlmConfigError("HTR_LLM_CONTEXT_LINES must be 0, 1 or 2")

    @classmethod
    def for_dir(cls, data_dir: Path | str, **kw) -> Settings:
        data_dir = Path(data_dir)
        return cls(data_dir=data_dir, db_path=data_dir / "htr.db", **kw)

    @classmethod
    def from_env(cls) -> Settings:
        """HTR_DATA_DIR (default <repo>/data), HTR_DB_PATH (default <data>/htr.db), HTR_MAX_UPLOAD_MB,
        HTR_MAX_IMAGE_MP, HTR_DEVICE (auto | cpu | cuda:0), HTR_TRAIN_BATCH_SIZE, HTR_TRAIN_WORKERS,
        HTR_PREDICT_BATCH_SIZE, and the optional local LLM: HTR_LLM_ENABLED (default off), HTR_LLM_URL (loopback
        only), HTR_LLM_API (ollama | openai), HTR_LLM_MODEL, HTR_LLM_TIMEOUT, HTR_LLM_CONTEXT_LINES."""
        data_dir = Path(os.environ.get("HTR_DATA_DIR") or REPO_ROOT / "data")
        db_path = Path(os.environ.get("HTR_DB_PATH") or data_dir / "htr.db")
        return cls(
            data_dir=data_dir,
            db_path=db_path,
            max_upload_bytes=int(float(os.environ.get("HTR_MAX_UPLOAD_MB", "100")) * 1024 * 1024),
            max_image_pixels=int(float(os.environ.get("HTR_MAX_IMAGE_MP", "80")) * 1_000_000),
            device=os.environ.get("HTR_DEVICE", "auto"),
            train_batch_size=int(os.environ.get("HTR_TRAIN_BATCH_SIZE", "4")),
            train_workers=int(os.environ.get("HTR_TRAIN_WORKERS", "0")),
            predict_batch_size=int(os.environ.get("HTR_PREDICT_BATCH_SIZE", "4")),
            llm_enabled=_flag("HTR_LLM_ENABLED"),
            llm_url=os.environ.get("HTR_LLM_URL") or LLM_DEFAULT_URL,
            llm_api=os.environ.get("HTR_LLM_API", "ollama"),
            llm_model=os.environ.get("HTR_LLM_MODEL", "llama3.2:3b"),
            llm_timeout=float(os.environ.get("HTR_LLM_TIMEOUT", "120")),
            llm_context_lines=int(os.environ.get("HTR_LLM_CONTEXT_LINES", "1")),
        )

    @property
    def workspace(self) -> Workspace:
        return Workspace(self.data_dir)

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"
