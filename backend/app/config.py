"""Runtime settings. Everything is overridable through environment variables so tests can use a temp directory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .pipeline.preprocess import DEFAULT_TARGET_LINE_HEIGHT
from .pipeline.workflow import Workspace

REPO_ROOT = Path(__file__).resolve().parents[2]

# Vite dev server origins only (plan section 4: nothing else may call the API from a browser).
CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_path: Path
    max_upload_bytes: int = 100 * 1024 * 1024
    device: str = "auto"  # kraken device: auto | cpu | cuda:0
    worker_poll_seconds: float = 0.25
    target_line_height: float = DEFAULT_TARGET_LINE_HEIGHT
    # Training / prediction defaults, sized for a 4 GB GPU (see README "Hardware"). Requests may override batch size.
    train_batch_size: int = 4
    train_workers: int = 0  # dataloader worker processes; 0 = load in the training process (safest on 16 GB RAM)
    predict_batch_size: int = 4

    @classmethod
    def for_dir(cls, data_dir: Path | str, **kw) -> Settings:
        data_dir = Path(data_dir)
        return cls(data_dir=data_dir, db_path=data_dir / "htr.db", **kw)

    @classmethod
    def from_env(cls) -> Settings:
        """HTR_DATA_DIR (default <repo>/data), HTR_DB_PATH (default <data>/htr.db), HTR_MAX_UPLOAD_MB, HTR_DEVICE
        (auto | cpu | cuda:0), HTR_TRAIN_BATCH_SIZE, HTR_TRAIN_WORKERS, HTR_PREDICT_BATCH_SIZE."""
        data_dir = Path(os.environ.get("HTR_DATA_DIR") or REPO_ROOT / "data")
        db_path = Path(os.environ.get("HTR_DB_PATH") or data_dir / "htr.db")
        return cls(
            data_dir=data_dir,
            db_path=db_path,
            max_upload_bytes=int(float(os.environ.get("HTR_MAX_UPLOAD_MB", "100")) * 1024 * 1024),
            device=os.environ.get("HTR_DEVICE", "auto"),
            train_batch_size=int(os.environ.get("HTR_TRAIN_BATCH_SIZE", "4")),
            train_workers=int(os.environ.get("HTR_TRAIN_WORKERS", "0")),
            predict_batch_size=int(os.environ.get("HTR_PREDICT_BATCH_SIZE", "4")),
        )

    @property
    def workspace(self) -> Workspace:
        return Workspace(self.data_dir)

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"
