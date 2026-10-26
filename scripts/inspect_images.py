#!/usr/bin/env python3
"""Characterize a folder of scans: pixel size, DPI tag, text-line height, warnings.

Usage: python scripts/inspect_images.py FOLDER [-r] [--json]

DPI tags are reported but never trusted; the estimated text-line height (in
pixels) is what matters for the recognition models (plan section 5.1).
The measurement code lives in backend/app/pipeline/measure.py (see its docstring
for the line-height method).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.pipeline.measure import (  # noqa: E402
    EXTENSIONS,
    ImageReport,
    estimate_line_height,
    find_images,
    image_warnings,
    inspect_image,
    summarize,
)

__all__ = ["EXTENSIONS", "ImageReport", "estimate_line_height", "find_images", "image_warnings", "inspect_image"]


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
