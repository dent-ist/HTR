"""Text recognition behind an engine-agnostic interface, plus the Kraken implementation.

`Recognizer.predict(line_images)` maps line images to `(text, confidence)`; `Trainer.train(...)` (see
pipeline/train.py for the Kraken one) fine-tunes a model from PAGE-XML ground truth. Only the Kraken engine exists
today; a TrOCR engine would implement the same two classes and be registered in `ENGINES`. The engine name is
stored on every `models` row.

Line images are produced by `extract_line_images`, which is *the same extraction Kraken uses for its training
samples* (polygon-masked, baseline-straightened crop of the page working copy), so a model sees at prediction time
what it saw in training.

Confidence of a line (documented choice): the **geometric mean of its per-character confidences**, where a
character's confidence is the highest softmax probability the network assigned to that character over the time
steps it occupies (Kraken's `ocr_record.confidences`). Always in [0, 1]. Geometric rather than arithmetic mean
because one unsure character should pull the score down noticeably (a line of nine certain characters and one
coin-flip scores 0.93, not 0.95), and it is length-normalized so long lines are not penalized just for being
long. Greedy per-character maxima are optimistic, so read the value as a *ranking* for review (lowest first), not as
a calibrated probability of the line being right. An empty prediction has confidence 0.0.
"""

from __future__ import annotations

import gc
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

Point = tuple[int, int]


# ---- errors ----------------------------------------------------------------------------------------------------


class RecognitionError(Exception):
    """A recognition or training problem that should reach the user as a plain message."""


class OutOfMemory(RecognitionError):
    """The GPU ran out of memory."""


class Cancelled(Exception):
    """Raised by `Recognizer.predict` / `Trainer.train` when the caller asked them to stop."""


def is_oom(e: BaseException) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or "cublas_status_alloc_failed" in msg or "cudnn_status_not_enough_workspace" in msg


def free_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ---- interfaces ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    text: str
    confidence: float  # [0, 1]

    def __iter__(self):  # allows `text, conf = prediction`
        yield self.text
        yield self.confidence


class Recognizer(ABC):
    """A loaded model that reads line images."""

    engine: str

    @abstractmethod
    def predict(
        self, line_images: Sequence[Image.Image], should_stop: Callable[[], bool] | None = None
    ) -> list[Prediction]:
        """One `Prediction` per image, in order. `should_stop` is polled between batches; if it returns True the
        call raises `Cancelled`."""

    def close(self) -> None:  # noqa: B027  optional hook
        """Release GPU memory."""


@dataclass
class TrainRequest:
    start_model: Path  # model to fine-tune
    out_path: Path  # where the best model is written (.mlmodel)
    train_files: list[Path]  # PAGE-XML files
    val_files: list[Path]  # PAGE-XML files used for early stopping and best-epoch selection (not the held-out set)
    work_dir: Path  # scratch space for checkpoints
    epochs: int = 15
    batch_size: int = 4
    learning_rate: float = 1e-4
    patience: int = 5  # early stopping: epochs without validation improvement
    device: str = "auto"
    num_workers: int = 0
    augment: bool = True
    seed: int = 1234


@dataclass
class EpochInfo:
    epoch: int  # 1-based
    epochs: int
    train_loss: float | None
    val_accuracy: float | None  # 1 - CER on the validation lines (Kraken's own metric)
    seconds: float

    def as_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "epochs": self.epochs,
            "train_loss": self.train_loss,
            "val_accuracy": self.val_accuracy,
            "seconds": round(self.seconds, 2),
        }


class TrainObserver:
    """Callbacks a trainer reports to; the default implementation ignores everything."""

    def on_batch(self, epoch: int, epochs: int, batch: int, n_batches: int, loss: float | None) -> None: ...

    def on_epoch(self, info: EpochInfo) -> None: ...

    def cancelled(self) -> bool:
        return False


@dataclass
class TrainOutcome:
    model_path: Path
    best_epoch: int
    best_val_accuracy: float | None
    epochs_run: int
    stopped_early: bool
    history: list[EpochInfo] = field(default_factory=list)
    seconds: float = 0.0
    device: str = "cpu"
    peak_vram_mb: float | None = None
    alphabet_added: int = 0


class Trainer(ABC):
    engine: str

    @abstractmethod
    def train(self, request: TrainRequest, observer: TrainObserver | None = None) -> TrainOutcome:
        """Fine-tune and write the best model to `request.out_path`. Raises `OutOfMemory` (GPU) or
        `Cancelled`, `RecognitionError` for anything else the user can act on."""


# ---- line images -----------------------------------------------------------------------------------------------


def extract_line_image(page: Image.Image, baseline: Sequence[Point], polygon: Sequence[Point]) -> Image.Image | None:
    """The recognition crop of one line: polygon-masked and straightened along its baseline (Kraken's extractor).
    Returns None when the geometry is unusable (baseline shorter than 5 px, degenerate polygon)."""
    from kraken.containers import BaselineLine, Segmentation
    from kraken.lib.segmentation import extract_polygons

    w, h = page.size
    clamp = lambda pts: [(min(max(int(x), 0), w - 1), min(max(int(y), 0), h - 1)) for x, y in pts]  # noqa: E731
    seg = Segmentation(
        type="baselines",
        imagename="page",
        text_direction="horizontal-lr",
        script_detection=False,
        lines=[BaselineLine(id="l", baseline=clamp(baseline), boundary=clamp(polygon))],
        regions={},
        line_orders=[],
    )
    try:
        im, _ = next(extract_polygons(page, seg))
    except (ValueError, StopIteration, IndexError, TypeError):
        return None
    if im.width < 2 or im.height < 2:
        return None
    return im


# ---- Kraken ----------------------------------------------------------------------------------------------------


def torch_device(device: str) -> str:
    """'auto' -> cuda:0 when available else cpu; otherwise the given torch device string."""
    if device != "auto":
        return device
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def line_confidence(char_confidences: Sequence[float]) -> float:
    """Geometric mean of per-character confidences, clamped to [0, 1]; 0.0 for an empty line."""
    if not char_confidences:
        return 0.0
    logs = 0.0
    for c in char_confidences:
        logs += math.log(min(max(float(c), 1e-9), 1.0))
    return min(1.0, max(0.0, math.exp(logs / len(char_confidences))))


class KrakenRecognizer(Recognizer):
    engine = "kraken"

    def __init__(self, model_path: Path | str, device: str = "auto", batch_size: int = 4):
        import torch
        from kraken.lib.dataset import ImageInputTransforms
        from kraken.models import load_models

        models = [m for m in load_models(str(model_path)) if "recognition" in (m.model_type or [])]
        if not models:
            raise RecognitionError(f"{model_path} contains no recognition model")
        self.net = models[0]
        self.path = Path(model_path)
        self.batch_size = max(1, int(batch_size))
        self.device = torch_device(device)
        self.fell_back_to_cpu = False
        _, channels, height, width = self.net.input
        self._transforms = ImageInputTransforms(1, height, width, channels, (16, 0), valid_norm=False)
        self.net.nn.eval()
        self._to(self.device)
        self._torch = torch

    def _to(self, device: str) -> None:
        self.net.nn.to(device)
        self.device = device

    def close(self) -> None:
        self.net = None
        free_cuda()

    def predict(
        self, line_images: Sequence[Image.Image], should_stop: Callable[[], bool] | None = None
    ) -> list[Prediction]:
        import torch

        n = len(line_images)
        out: list[Prediction | None] = [None] * n
        tensors: dict[int, torch.Tensor] = {}
        for i, im in enumerate(line_images):
            try:
                t = self._transforms(im)
                if t.max() == t.min():  # blank crop
                    out[i] = Prediction("", 0.0)
                else:
                    tensors[i] = t
            except Exception:  # noqa: BLE001  a crop the transforms cannot handle reads as empty
                out[i] = Prediction("", 0.0)
        # similar widths together keeps padding (and memory) small
        order = sorted(tensors, key=lambda i: tensors[i].shape[2], reverse=True)
        pos = 0
        size = self.batch_size
        while pos < len(order):
            if should_stop and should_stop():
                raise Cancelled()
            chunk = order[pos : pos + size]
            try:
                preds = self._run([tensors[i] for i in chunk])
            except Exception as e:  # noqa: BLE001
                if not is_oom(e):
                    raise
                free_cuda()
                if size > 1:
                    size = max(1, size // 2)
                    log.warning("CUDA out of memory in prediction; batch size reduced to %d", size)
                    continue
                if self.device.startswith("cuda"):
                    log.warning("CUDA out of memory with batch size 1; continuing on the CPU")
                    self._to("cpu")
                    self.fell_back_to_cpu = True
                    continue
                raise OutOfMemory("out of memory while reading a line (try a narrower line or more RAM)") from e
            for i, p in zip(chunk, preds, strict=True):
                out[i] = p
            pos += len(chunk)
        return [p if p is not None else Prediction("", 0.0) for p in out]

    def _run(self, batch: list) -> list[Prediction]:
        import torch
        import torch.nn.functional as F
        from kraken.lib.ctc_decoder import greedy_decoder

        width = max(t.shape[2] for t in batch)
        seqs = torch.stack([F.pad(t, (0, width - t.shape[2])) for t in batch]).to(self.device)
        lens = torch.LongTensor([t.shape[2] for t in batch])
        with torch.inference_mode():
            logits, olens = self.net.nn(seqs, lens)
            probs = logits.softmax(1).squeeze(2).float().cpu()
        result = []
        for locs in greedy_decoder(probs, olens.cpu()):
            decoded = self.net.codec.decode(locs)
            text = "".join(c[0] for c in decoded)
            result.append(Prediction(text, line_confidence([c[3] for c in decoded])))
        return result


# ---- registry --------------------------------------------------------------------------------------------------

ENGINES = ("kraken",)


def load_recognizer(engine: str, model_path: Path | str, device: str = "auto", batch_size: int = 4) -> Recognizer:
    if engine == "kraken":
        return KrakenRecognizer(model_path, device, batch_size)
    raise RecognitionError(f"unknown recognition engine '{engine}' (available: {', '.join(ENGINES)})")


def make_trainer(engine: str) -> Trainer:
    if engine == "kraken":
        from .train import KrakenTrainer

        return KrakenTrainer()
    raise RecognitionError(f"unknown recognition engine '{engine}' (available: {', '.join(ENGINES)})")
