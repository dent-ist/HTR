"""Preprocessing for pre-cropped *word* images (not page scans): the "word canvas".

Why this exists. Kraken rescales every line crop to the height its recognition model was trained at (120 px for the
Rev City base model) and keeps the aspect ratio. The network's convolutions downsample the width by 8, so it emits
one CTC time step per 8 px of the rescaled line, and a normal text line gives roughly 7 time steps per character.
A square 128x128 word crop (dataset RxHand) becomes 120x120 (+ 16 px padding on each side) = **19 time steps in
total** for a word of up to 18 characters, i.e. about 2.3 steps per character (and fewer than the characters plus
the blanks CTC needs between repeated letters for the longest labels). The square crops are also clearly squeezed
horizontally. Stretching the crop horizontally before it reaches Kraken restores a normal step density.

The fix is **one function, `apply_word_canvas`**, applied once, when a word image is imported
(`services.presegmented.import_word_image`): it writes the *working copy* of the page, which is the image both the
training export and the predict job crop their lines from. Training and prediction therefore always see the same
pixels, and the original file stays byte-identical. The setting is recorded on the page (`pages.preprocess`), on
every fine-tuned model (`hyperparameters.line_preprocess`) and checked before a model is used
(`compatible`), so a model is never run on pages prepared differently from the pages it was trained on.

This applies only to the word-crop import path. Page scans keep the line-height normalization of
`pipeline/preprocess.py`; their key is `PAGE_KEY`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

PAGE_KEY = "page"  # the default pipeline: pages normalized by line height, lines cut by the segmenter
MODE = "word_canvas"
MAX_STRETCH = 8.0


@dataclass(frozen=True)
class WordCanvas:
    """Horizontal stretch factor applied to a word image (1.0 = grayscale conversion only)."""

    stretch_x: float = 3.0

    def __post_init__(self) -> None:
        if not 1.0 <= self.stretch_x <= MAX_STRETCH:
            raise ValueError(f"stretch_x must be between 1 and {MAX_STRETCH:g}")

    @property
    def key(self) -> str:
        return f"{MODE}:x{self.stretch_x:g}"

    def size_for(self, width: int, height: int) -> tuple[int, int]:
        return max(1, round(width * self.stretch_x)), height

    def to_dict(self) -> dict:
        return {"mode": MODE, "key": self.key, **asdict(self)}


def apply_word_canvas(img: Image.Image, canvas: WordCanvas) -> Image.Image:
    """The single preprocessing function of the word-crop path: grayscale, then horizontal bicubic stretch.
    Deterministic (same input, same output), no contrast normalization, no deskew, no rescale of the height."""
    gray = img.convert("L")
    size = canvas.size_for(*gray.size)
    return gray if size == gray.size else gray.resize(size, Image.BICUBIC)


def write_word_working(src: Path, dest: Path, canvas: WordCanvas) -> tuple[int, int]:
    """Read `src`, apply the canvas, save a PNG at `dest`; returns the working size."""
    with Image.open(src) as im:
        im.load()
        out = apply_word_canvas(im, canvas)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.save(dest)
    return out.size


def key_of_preprocess(preprocess_json: dict | None) -> str:
    """The line-preprocessing key of a page, from its stored `pages.preprocess` record."""
    if preprocess_json and preprocess_json.get("mode") == MODE and preprocess_json.get("key"):
        return str(preprocess_json["key"])
    return PAGE_KEY


def compatible(model_key: str | None, page_key: str) -> bool:
    """May a model with `model_key` read pages prepared as `page_key`? `None` means no constraint (a pretrained base
    model, which was not trained on this project's pages)."""
    return model_key is None or model_key == page_key
