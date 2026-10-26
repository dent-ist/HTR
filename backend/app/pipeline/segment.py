"""Kraken baseline segmentation (`blla`) -> baselines + polygons.

The default segmentation model (blla.mlmodel) ships inside the kraken wheel; no download is needed.
Kraken is imported lazily so importing this module is cheap and unit tests can mock `run_blla`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

Point = tuple[int, int]


@dataclass
class Line:
    id: str
    baseline: list[Point]
    polygon: list[Point]
    index: int = 0  # reading order, 0-based
    # Transcription state (M2). Kraken segmentation leaves these at their defaults.
    text: str = ""
    text_source: str | None = None  # "manual" | "predicted" | None (no text)
    confidence: float | None = None
    is_ground_truth: bool = False
    ignored: bool = False


@dataclass
class PageSegmentation:
    width: int
    height: int
    lines: list[Line] = field(default_factory=list)
    device: str = "cpu"


_MODEL = None


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def _load_model():
    global _MODEL
    if _MODEL is None:
        from importlib import resources

        from kraken.lib import vgsl

        _MODEL = vgsl.TorchVGSLModel.load_model(resources.files("kraken").joinpath("blla.mlmodel"))
    return _MODEL


def run_blla(img: Image.Image, device: str):
    """Call kraken's blla on a PIL image; returns a kraken Segmentation. Separate function so tests can mock it."""
    from kraken import blla

    return blla.segment(img.convert("RGB"), model=_load_model(), device=device)


def _pts(seq) -> list[Point]:
    return [(int(round(x)), int(round(y))) for x, y in seq]


def segment_image(img: Image.Image, device: str = "auto") -> PageSegmentation:
    """Segment one page image. Falls back to CPU if the GPU runs out of memory."""
    dev = resolve_device(device)
    try:
        seg = run_blla(img, dev)
    except RuntimeError as e:
        if dev.startswith("cuda") and "out of memory" in str(e).lower():
            log.warning("CUDA OOM, retrying on CPU")
            _free_cuda()
            dev = "cpu"
            seg = run_blla(img, dev)
        else:
            raise
    lines = []
    for i, ln in enumerate(seg.lines or []):
        base = _pts(ln.baseline or [])
        poly = _pts(ln.boundary or [])
        if len(base) < 2 or len(poly) < 3:
            continue  # degenerate line record
        lines.append(Line(id=f"l{i + 1:03d}", baseline=base, polygon=poly))
    # kraken returns lines in its reading order; keep it and number them
    for i, ln in enumerate(lines):
        ln.index = i
        ln.id = f"l{i + 1:03d}"
    return PageSegmentation(width=img.width, height=img.height, lines=lines, device=dev)


def segment_file(path: Path, device: str = "auto") -> PageSegmentation:
    with Image.open(path) as im:
        im.load()
        return segment_image(im, device)


def _free_cuda() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
