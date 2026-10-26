"""Working-copy preprocessing (plan section 5.1). Originals are never modified; this only derives a new image.

Steps: grayscale -> background/contrast normalization -> light deskew (small angles only) -> measure line height ->
rescale toward a target line height only when the measured height is far outside the usable range.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .measure import MAX_LINE_HEIGHT_PX, MIN_LINE_HEIGHT_PX, binarize, estimate_line_height, to_gray

DEFAULT_TARGET_LINE_HEIGHT = 40.0
MAX_SKEW_DEG = 5.0  # larger angles are not corrected (likely a real rotation / odd page, flagged instead)
MIN_SKEW_DEG = 0.3  # smaller angles are not worth a resample
MAX_SCALE = 4.0  # never scale by more than this factor either way


@dataclass
class PreprocessInfo:
    skew_deg: float = 0.0
    deskewed: bool = False
    line_height_before: float | None = None  # measured on the normalized, deskewed image (rescale decision)
    line_pitch_before: float | None = None
    n_lines: int = 0
    line_height_after: float | None = None
    scale: float = 1.0
    rescaled: bool = False
    target_line_height: float = DEFAULT_TARGET_LINE_HEIGHT
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_contrast(gray: np.ndarray) -> np.ndarray:
    """Flatten uneven illumination, then stretch contrast to 0..255 (float array)."""
    h, w = gray.shape
    f = 8
    small = np.asarray(Image.fromarray(gray.astype(np.float32), mode="F").resize((max(w // f, 1), max(h // f, 1))))
    # grey closing removes thin dark features (ink, rules) and leaves the paper brightness
    bg = ndimage.grey_closing(small, size=(15, 15))
    bg = ndimage.gaussian_filter(bg, 3)
    bg = np.asarray(Image.fromarray(bg.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR))
    flat = np.clip(gray / np.maximum(bg, 1.0), 0, 1.2) * 255.0
    lo, hi = np.percentile(flat, [0.5, 99.5])
    if hi - lo < 10:  # blank / near uniform: leave alone
        return np.clip(flat, 0, 255)
    return np.clip((flat - lo) * (255.0 / (hi - lo)), 0, 255)


def estimate_skew(gray: np.ndarray, max_deg: float = MAX_SKEW_DEG) -> float:
    """Skew angle in degrees (positive = text rises to the right) via projection-profile sharpness."""
    scale = max(1, -(-gray.shape[1] // 700))
    small = np.asarray(Image.fromarray(gray.astype(np.float32), mode="F").reduce(scale)) if scale > 1 else gray
    ink = binarize(small).astype(np.float32)
    if ink.sum() < 50:
        return 0.0

    def score(angle: float) -> float:
        rot = ndimage.rotate(ink, -angle, reshape=False, order=1)
        prof = rot.sum(axis=1)
        return float(np.sum(np.diff(prof) ** 2))

    coarse = np.arange(-max_deg, max_deg + 1e-9, 0.5)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.5, best + 0.5 + 1e-9, 0.1)
    best = max(fine, key=score)
    return float(np.clip(best, -max_deg, max_deg))


def rotate_gray(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate counter-clockwise by angle_deg, same canvas, filling with the paper (bright) value."""
    fill = float(np.percentile(gray, 95))
    img = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), "L")
    return np.asarray(img.rotate(angle_deg, resample=Image.BICUBIC, fillcolor=int(fill)), dtype=np.float64)


def scale_factor(
    line_height: float | None,
    target: float,
    low: float = MIN_LINE_HEIGHT_PX,
    high: float = MAX_LINE_HEIGHT_PX,
) -> float:
    """1.0 unless line_height is outside [low, high]; then target/line_height (clamped to MAX_SCALE)."""
    if not line_height or low <= line_height <= high:
        return 1.0
    return float(np.clip(target / line_height, 1 / MAX_SCALE, MAX_SCALE))


def preprocess_array(
    gray: np.ndarray, target_line_height: float = DEFAULT_TARGET_LINE_HEIGHT, deskew: bool = True
) -> tuple[Image.Image, PreprocessInfo]:
    info = PreprocessInfo(target_line_height=target_line_height)
    g = normalize_contrast(gray)
    if deskew:
        angle = estimate_skew(g)
        info.skew_deg = round(angle, 2)
        if MIN_SKEW_DEG <= abs(angle) <= MAX_SKEW_DEG:
            g = rotate_gray(g, -angle)
            info.deskewed = True
    lh, pitch, n_lines = estimate_line_height(g)  # after deskew: band thickness is skew-sensitive
    info.line_height_before = None if lh is None else round(lh, 1)
    info.line_pitch_before = None if pitch is None else round(pitch, 1)
    info.n_lines = n_lines
    factor = scale_factor(lh, target_line_height)
    img = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8), "L")
    if abs(factor - 1.0) > 1e-6:
        size = (max(1, round(img.width * factor)), max(1, round(img.height * factor)))
        img = img.resize(size, Image.LANCZOS if factor < 1 else Image.BICUBIC)
        info.scale = round(factor, 4)
        info.rescaled = True
    info.line_height_after = None if lh is None else round(lh * factor, 1)
    info.width, info.height = img.size
    return img, info


def preprocess_file(
    src: Path, dst: Path, target_line_height: float = DEFAULT_TARGET_LINE_HEIGHT, deskew: bool = True
) -> PreprocessInfo:
    """Read src (never written), write the working copy as PNG to dst."""
    with Image.open(src) as im:
        gray = to_gray(im)
    img, info = preprocess_array(gray, target_line_height, deskew)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format="PNG")
    return info
