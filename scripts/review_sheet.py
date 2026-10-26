#!/usr/bin/env python3
"""Segmentation review (plan 5.2 / M1 go-no-go).

  review_sheet.py sample [-n 20] [--seed 0] [--out data/review_sheet.csv]
  review_sheet.py score  [SHEET.csv]

`sample` picks N segmented pages (the ones with the most import warnings first, the rest at random) and writes a
CSV with columns page, detected_lines, correct_lines, missed_lines, notes. Open data/overlays/<page>.png for each
page; fill correct_lines (detected lines that are right) and missed_lines (real lines with no detected line).
Both are required on every row (0 is fine, blank is an error).
`score` reads the filled sheet and prints share = correct / (detected + missed) and the plan's decision:
> 90% proceed; 70-90% prioritize repair tools; < 70% tune the segmenter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from app.pipeline import review  # noqa: E402
from app.pipeline.workflow import Workspace  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=REPO / "data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("-n", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, default=None)
    c = sub.add_parser("score")
    c.add_argument("sheet", type=Path, nargs="?", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "sample":
        ws = Workspace(args.data)
        manifest = ws.load()
        pages = review.sample_pages(manifest, args.n, args.seed)
        if not pages:
            print("error: no segmented pages (run import and segment first)", file=sys.stderr)
            return 2
        out = args.out or args.data / "review_sheet.csv"
        review.write_sheet(manifest, pages, out)
        print(f"wrote {out} with {len(pages)} pages; overlays in {ws.overlays}")
        for p in pages:
            flag = f"  [{len(manifest[p]['warnings'])} warning(s)]" if manifest[p]["warnings"] else ""
            print(f"  {p}: {manifest[p]['segmentation']['lines']} detected{flag}")
        return 0

    sheet = args.sheet or args.data / "review_sheet.csv"
    try:
        sc = review.score_sheet(sheet)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"pages scored: {sc.pages}")
    print(f"detected lines: {sc.detected}; correct: {sc.correct}; missed: {sc.missed}")
    print(f"share correctly segmented: {sc.share:.1f}%")
    print(f"decision: {sc.decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
