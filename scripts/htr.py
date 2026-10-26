#!/usr/bin/env python3
"""M1 CLI: import scans, segment them with Kraken, and run the segmentation review.

  htr.py import FOLDER [-r] [--target-line-height 40] [--no-deskew]
  htr.py segment [--device auto|cpu|cuda:0] [--force] [--page ID ...]

All output goes under --data (default: <repo>/data, gitignored). Both commands are re-runnable:
`import` skips files whose content hash is already imported; `segment` skips pages already done and retries
failed ones, and a failing page never aborts the batch (exit status 1 if any page failed).
Review sheet: scripts/review_sheet.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from app.pipeline.preprocess import DEFAULT_TARGET_LINE_HEIGHT  # noqa: E402
from app.pipeline.workflow import Workspace, import_folder, segment_pages  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=REPO / "data", help="data directory (default: ./data)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import", help="copy originals untouched, write working copies and an import report")
    imp.add_argument("folder", type=Path)
    imp.add_argument("-r", "--recursive", action="store_true")
    imp.add_argument(
        "--target-line-height",
        type=float,
        default=DEFAULT_TARGET_LINE_HEIGHT,
        help="px; working copies far outside the usable range are rescaled toward this (default %(default)s)",
    )
    imp.add_argument("--no-deskew", action="store_true")
    seg = sub.add_parser("segment", help="run Kraken blla on working copies; write PAGE-XML and overlay PNGs")
    seg.add_argument("--device", default="auto", help="auto (default), cpu or cuda:0")
    seg.add_argument("--force", action="store_true", help="re-segment pages that are already done")
    seg.add_argument("--page", action="append", help="only this page id (repeatable)")
    args = ap.parse_args(argv)
    ws = Workspace(args.data)

    if args.cmd == "import":
        if not args.folder.is_dir():
            print(f"error: {args.folder} is not a directory", file=sys.stderr)
            return 2
        stats = import_folder(args.folder, ws, args.target_line_height, args.recursive, not args.no_deskew)
        print(f"import: {stats}\nreport: {ws.root / 'import_report.md'}")
        return 1 if stats["failed"] else 0

    if not ws.load():
        print("error: nothing imported yet (run `import` first)", file=sys.stderr)
        return 2
    stats = segment_pages(ws, args.device, args.force, args.page)
    print(f"segment: {stats}\nPAGE-XML: {ws.pagexml}\noverlays: {ws.overlays}")
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
