#!/usr/bin/env python3
"""M1 CLI: import scans, segment them with Kraken, and run the segmentation review.

  htr.py import FOLDER [-r] [--target-line-height 40] [--no-deskew]
  htr.py segment [--device auto|cpu|cuda:0] [--force] [--page ID ...]

All output goes under --data (default: <repo>/data, gitignored). Both commands are re-runnable:
`import` skips files whose content hash is already imported; `segment` skips pages already done and retries
failed ones, and a failing page never aborts the batch (exit status 1 if any page failed).
Review sheet: scripts/review_sheet.py.

  htr.py bundle export PATH     write the whole project (scans, PAGE-XML, database, models) to a .htrbundle file
  htr.py bundle import PATH     restore a bundle into an EMPTY project (stop the server first)
Bundles contain personal data: they are written only where you point them, nothing is uploaded. Both commands work on
the --data directory without the server.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from app.config import Settings  # noqa: E402
from app.errors import ApiError  # noqa: E402
from app.pipeline.preprocess import DEFAULT_TARGET_LINE_HEIGHT  # noqa: E402
from app.pipeline.workflow import Workspace, import_folder, segment_pages  # noqa: E402


def _bundle(args) -> int:
    from app.projectlock import try_lock
    from app.services import bundle

    settings = Settings.for_dir(args.data)
    try:
        if args.bundle_cmd == "export":
            info = bundle.export_to_path(settings, args.path)
            c = info["counts"]
            print(
                f"bundle: {info['path']} ({info['bytes'] / 1e6:.1f} MB): {c['documents']} documents, "
                f"{c['pages']} pages, {c['lines']} lines, {c['finetuned_models']} fine-tuned models, "
                f"{c['abbreviations']} abbreviations"
            )
            for missing in info["missing_files"]:
                print(f"warning: {missing} was missing on disk and is not in the bundle", file=sys.stderr)
            print("The bundle contains personal data. Keep it only where you chose to put it.")
            return 0
        if not args.path.is_file():
            print(f"error: {args.path} is not a file", file=sys.stderr)
            return 2
        lock = try_lock(settings.data_dir)
        if lock is None:
            print(
                "error: a server is using this data directory; stop it first (or import from the Export page)",
                file=sys.stderr,
            )
            return 1
        try:
            with open(args.path, "rb") as f:
                info = bundle.import_bundle(settings, f)
        finally:
            lock.close()
        c = info["counts"]
        print(
            f"imported: {c['documents']} documents, {c['pages']} pages, {c['lines']} lines, "
            f"{c['finetuned_models']} fine-tuned models, {c['abbreviations']} abbreviations "
            f"(schema {info['schema_version']}"
            + (f", migrated from {info['migrated_from']}" if info["migrated_from"] else "")
            + ")"
        )
        return 0
    except ApiError as e:
        print(f"error: {e.detail}", file=sys.stderr)
        return 1


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
    bun = sub.add_parser("bundle", help="export or import a whole project as one .htrbundle file")
    bsub = bun.add_subparsers(dest="bundle_cmd", required=True)
    bsub.add_parser("export", help="write a bundle (refuses while a job runs or if the file exists)").add_argument(
        "path", type=Path
    )
    bsub.add_parser("import", help="restore a bundle into an empty project (never merges)").add_argument(
        "path", type=Path
    )
    args = ap.parse_args(argv)
    if args.cmd == "bundle":
        return _bundle(args)
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
