#!/usr/bin/env python3
"""Characterize a folder of scans: pixel size, DPI tag, text-line height, warnings.

Usage: python scripts/inspect_images.py FOLDER [-r] [--json]

DPI tags are reported but never trusted; the estimated text-line height (in
pixels) is what matters for the recognition models (plan section 5.1).

Line-height method: binarize (Otsu), erase long horizontal ink runs (paper rules,
underlines), take the horizontal projection profile, smooth it, and find text
bands (runs above a fraction of the typical peak). `line_height_px` is the median
band thickness; `line_pitch_px` is the line-to-line spacing from the profile's
autocorrelation. If bands touch (thickness >= pitch) the pitch is reported instead.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
ANALYSIS_WIDTH = 1600  # downscale wider images for analysis; result is scaled back

MIN_LINE_HEIGHT_PX = 20  # below this, thin strokes are hard to resolve
MAX_LINE_HEIGHT_PX = 150  # far above usable range; working copy gets downscaled
LOW_DPI = 150
MIN_SIDE_PX = 800
MAX_SIDE_PX = 10000
ASPECT_RANGE = (0.5, 2.0)  # width/height of a page-like image
PLACEHOLDER_DPIS = {72, 96}


@dataclass
class ImageReport:
    path: str
    width: int = 0
    height: int = 0
    dpi_x: float | None = None
    dpi_y: float | None = None
    line_height_px: float | None = None
    line_pitch_px: float | None = None
    n_lines: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def to_gray(img: Image.Image) -> np.ndarray:
    """Grayscale float array in 0..255, white background for transparency."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    if img.mode.startswith("I") or img.mode == "F":  # 16/32-bit
        arr = np.asarray(img, dtype=np.float64)
        lo, hi = arr.min(), arr.max()
        return (arr - lo) * (255.0 / (hi - lo)) if hi > lo else np.full(arr.shape, 255.0)
    return np.asarray(img.convert("L"), dtype=np.float64)


def otsu_threshold(gray: np.ndarray) -> float:
    hist, edges = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(hist * centers)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (w0 * w1) * ((m0 / w0) - ((mt - m0) / w1)) ** 2
    return float(centers[np.nanargmax(between)]) if np.isfinite(between).any() else 128.0


def binarize(gray: np.ndarray) -> np.ndarray:
    """Boolean ink mask (True = dark). Assumes dark ink on a light page."""
    if gray.max() - gray.min() < 30:  # blank / near-uniform page
        return np.zeros(gray.shape, dtype=bool)
    return gray < otsu_threshold(gray)


def remove_horizontal_rules(ink: np.ndarray, min_frac: float = 0.25) -> np.ndarray:
    """Erase ink runs longer than min_frac of the width (ruled lines, underlines)."""
    width = ink.shape[1]
    length = max(int(width * min_frac), 3)
    rules = ndimage.binary_opening(ink, structure=np.ones((1, length), dtype=bool))
    if not rules.any():
        return ink
    rules = ndimage.binary_dilation(rules, structure=np.ones((3, 1), dtype=bool))
    return ink & ~rules


def find_bands(profile: np.ndarray) -> list[tuple[int, int]]:
    """Runs of rows (start, end exclusive) where the smoothed profile is high."""
    if profile.max() <= 0:
        return []
    sigma = max(len(profile) / 800, 1.0)
    smooth = ndimage.gaussian_filter1d(profile, sigma)
    peak = np.percentile(smooth[smooth > 0], 90) if (smooth > 0).any() else 0
    if peak <= 0:
        return []
    above = smooth > 0.3 * peak
    labels, n = ndimage.label(above)
    bands = []
    for i in range(1, n + 1):
        rows = np.flatnonzero(labels == i)
        bands.append((int(rows[0]), int(rows[-1]) + 1))
    min_h = max(3, int(len(profile) * 0.004))
    return [b for b in bands if b[1] - b[0] >= min_h]


def estimate_pitch(profile: np.ndarray, min_pitch: int = 4) -> float | None:
    """Line spacing from the first strong autocorrelation peak of the profile."""
    p = profile - profile.mean()
    if np.allclose(p, 0) or len(p) < 4 * min_pitch:
        return None
    ac = np.correlate(p, p, mode="full")[len(p) - 1 :]
    ac = ac / ac[0]
    sigma = max(len(p) / 800, 1.0)
    ac = ndimage.gaussian_filter1d(ac, sigma)
    max_lag = len(p) // 2
    peaks = [
        lag for lag in range(min_pitch, max_lag) if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > 0.15
    ]
    if not peaks:
        return None
    best = max(ac[lag] for lag in peaks)
    # first peak that is nearly as strong as the strongest (avoids harmonics)
    return float(next(lag for lag in peaks if ac[lag] >= 0.8 * best))


def estimate_line_height(gray: np.ndarray) -> tuple[float | None, float | None, int]:
    """Return (line_height_px, line_pitch_px, n_lines) in full-resolution pixels."""
    w = gray.shape[1]
    scale = max(1, -(-w // ANALYSIS_WIDTH))
    if scale > 1:
        img = Image.fromarray(gray.astype(np.float32), mode="F").reduce(scale)
        gray = np.asarray(img, dtype=np.float64)
    ink = remove_horizontal_rules(binarize(gray))
    profile = ink.sum(axis=1).astype(np.float64)
    bands = find_bands(profile)
    if not bands:
        return None, None, 0
    thickness = [e - s for s, e in bands]
    height = float(statistics.median(thickness))
    pitch = estimate_pitch(profile) if len(bands) >= 3 else None
    if pitch is None and len(bands) >= 2:
        starts = [s for s, _ in bands]
        pitch = float(statistics.median(np.diff(starts)))
    if pitch is not None and height >= pitch:
        height = pitch  # bands touch; spacing is the better line-height proxy
    return height * scale, (pitch * scale if pitch else None), len(bands)


def read_dpi(img: Image.Image) -> tuple[float | None, float | None]:
    dpi = img.info.get("dpi")
    if not dpi:
        return None, None
    try:
        x, y = float(dpi[0]), float(dpi[1])
    except (TypeError, ValueError, IndexError):
        return None, None
    if x <= 1 or y <= 1:  # 0, or 1 = TIFF "no resolution unit" default: not a real DPI
        return None, None
    return round(x, 2), round(y, 2)


def inspect_image(path: Path) -> ImageReport:
    rep = ImageReport(path=str(path))
    try:
        with Image.open(path) as img:  # first frame only for multi-page TIFF
            rep.width, rep.height = img.size
            rep.dpi_x, rep.dpi_y = read_dpi(img)
            gray = to_gray(img)
    except Exception as e:  # noqa: BLE001
        rep.error = f"{type(e).__name__}: {e}"
        return rep
    rep.line_height_px, rep.line_pitch_px, rep.n_lines = estimate_line_height(gray)
    rep.warnings = image_warnings(rep)
    return rep


def image_warnings(rep: ImageReport) -> list[str]:
    w: list[str] = []
    short, long_ = sorted((rep.width, rep.height))
    if short < MIN_SIDE_PX:
        w.append(f"small image ({rep.width}x{rep.height}px)")
    if long_ > MAX_SIDE_PX:
        w.append(f"very large image ({rep.width}x{rep.height}px)")
    aspect = rep.width / rep.height if rep.height else 0
    if not ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]:
        w.append(f"unusual aspect ratio {aspect:.2f}")
    lh = rep.line_height_px
    if rep.dpi_x is None:
        w.append("missing DPI metadata")
    else:
        if abs(rep.dpi_x - rep.dpi_y) / max(rep.dpi_x, rep.dpi_y) > 0.02:
            w.append(f"contradictory DPI: x={rep.dpi_x:g} vs y={rep.dpi_y:g}")
        if rep.dpi_x < LOW_DPI:
            w.append(f"low DPI tag ({rep.dpi_x:g})")
        if round(rep.dpi_x) in PLACEHOLDER_DPIS and long_ > 1500:
            w.append(f"DPI tag {rep.dpi_x:g} looks like a placeholder for a {long_}px image")
        if rep.dpi_x >= 250 and lh is not None and lh < MIN_LINE_HEIGHT_PX:
            w.append("contradictory metadata: high DPI tag but tiny text")
    if lh is None:
        w.append("no text lines detected (blank page, or too faint)")
    else:
        if lh < MIN_LINE_HEIGHT_PX:
            w.append(f"low resolution: line height {lh:.0f}px < {MIN_LINE_HEIGHT_PX}px")
        if lh > MAX_LINE_HEIGHT_PX:
            w.append(f"line height {lh:.0f}px > {MAX_LINE_HEIGHT_PX}px (would be downscaled)")
    return w


def find_images(folder: Path, recursive: bool) -> list[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in EXTENSIONS)


def summarize(reports: list[ImageReport]) -> dict:
    ok = [r for r in reports if r.error is None]
    heights = [r.line_height_px for r in ok if r.line_height_px]
    summary: dict = {
        "images": len(reports),
        "unreadable": len(reports) - len(ok),
        "with_dpi_tag": sum(r.dpi_x is not None for r in ok),
        "with_warnings": sum(bool(r.warnings) for r in ok),
    }
    if ok:
        summary["width_range"] = [min(r.width for r in ok), max(r.width for r in ok)]
        summary["height_range"] = [min(r.height for r in ok), max(r.height for r in ok)]
    if heights:
        med = statistics.median(heights)
        summary["line_height_px"] = {
            "min": round(min(heights), 1),
            "median": round(med, 1),
            "max": round(max(heights), 1),
        }
        summary["line_height_outliers"] = [
            Path(r.path).name for r in ok if r.line_height_px and not 0.6 * med <= r.line_height_px <= 1.6 * med
        ]
    return summary


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.0f}"


def print_report(reports: list[ImageReport], summary: dict) -> None:
    for r in reports:
        name = Path(r.path).name
        if r.error:
            print(f"{name}\n  ERROR: {r.error}")
            continue
        dpi = "none" if r.dpi_x is None else f"{r.dpi_x:g}x{r.dpi_y:g}"
        print(name)
        print(
            f"  size {r.width}x{r.height}px | dpi tag {dpi} | line height "
            f"{_fmt(r.line_height_px)}px | pitch {_fmt(r.line_pitch_px)}px | lines {r.n_lines}"
        )
        for msg in r.warnings:
            print(f"  WARNING: {msg}")
    print("\n== Summary ==")
    print(f"images: {summary['images']} ({summary['unreadable']} unreadable)")
    print(f"with DPI tag: {summary['with_dpi_tag']}; with warnings: {summary['with_warnings']}")
    if "width_range" in summary:
        print(f"width px: {summary['width_range']}, height px: {summary['height_range']}")
    if "line_height_px" in summary:
        lh = summary["line_height_px"]
        print(f"line height px: min {lh['min']} / median {lh['median']} / max {lh['max']}")
        if summary["line_height_outliers"]:
            print(f"line height outliers: {', '.join(summary['line_height_outliers'])}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect a folder of scanned images.")
    ap.add_argument("folder", type=Path)
    ap.add_argument("-r", "--recursive", action="store_true")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)
    if not args.folder.is_dir():
        print(f"error: {args.folder} is not a directory", file=sys.stderr)
        return 2
    paths = find_images(args.folder, args.recursive)
    if not paths:
        print(f"error: no PNG/JPG/TIFF images in {args.folder}", file=sys.stderr)
        return 2
    reports = [inspect_image(p) for p in paths]
    summary = summarize(reports)
    if args.json:
        print(json.dumps({"images": [asdict(r) for r in reports], "summary": summary}, indent=2))
    else:
        print_report(reports, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
