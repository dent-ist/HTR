#!/usr/bin/env python3
"""Import a word-image dataset (RxHand layout) into an isolated HTR project for recognizer experiments (dev tool).

Each image becomes its own one-line page through the presegmented import path (`services/presegmented.py`): no
segmentation, no line-height rescale, no deskew; the original JPG is stored byte-identical, the working copy is the
word canvas (`pipeline/wordcrop.py`, horizontal stretch), and the label is manual ground truth. The fixed splits
(train / validation / seen-words set A / unseen-words set B, see `services/wordsplits.py`) are written to the
project's `line_splits` table (A and B are 'eval') and, in full, to `<data-dir>/experiment/meta.json`.

    uv run --project backend python scripts/rxhand_import.py --data-dir data/experiments/rxhand/x3 --stretch 3

Input layout: `--images DIR` with the image files, `--csv FILE` with columns `Images,Text` (utf-8-sig). Defaults point
at the copy under `data/external/rxhand/` (gitignored; the dataset licence is unknown, never commit or share it).

The project directory is created for this experiment only. The tool refuses the app's own data directory (and
anything that already holds an `htr.db` of a different stretch), so a real project can never be touched.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from app.config import Settings  # noqa: E402
from app.db import connect, migrate, now_iso, write_tx  # noqa: E402
from app.pipeline.wordcrop import WordCanvas  # noqa: E402
from app.services import pages as pages_svc  # noqa: E402
from app.services import presegmented  # noqa: E402
from app.services import wordsplits as WS  # noqa: E402

DEFAULT_ROOT = REPO / "data" / "external" / "rxhand"


class ImportRefused(Exception):
    pass


def read_labels(csv_path: Path) -> tuple[list[tuple[str, str]], dict]:
    """[(file name, cleaned label)] in file-name order, plus a report of what was cleaned or dropped.

    Cleaning: outer whitespace is stripped; rows whose label contains '?' (unreadable placeholders such as '???',
    or a '?' standing for a character the annotator could not read) are dropped. Case is kept as written."""
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = [(r["Images"].strip(), r["Text"]) for r in csv.DictReader(f)]
    report = {"rows": len(rows), "stripped": 0, "dropped_question_mark": [], "empty": 0}
    out = []
    for name, text in rows:
        clean = WS.clean_label(text)
        if clean != text:
            report["stripped"] += 1
        if not clean:
            report["empty"] += 1
            continue
        if "?" in clean:
            report["dropped_question_mark"].append(name)
            continue
        out.append((name, clean))
    out.sort()
    report["kept"] = len(out)
    return out, report


def guard_data_dir(data_dir: Path, stretch: float) -> None:
    data_dir = data_dir.resolve()
    forbidden = {(REPO / "data").resolve(), Path(Settings.from_env().data_dir).resolve()}
    if data_dir in forbidden:
        raise ImportRefused(f"{data_dir} is the application's data directory; use a directory for this experiment")
    meta = data_dir / "experiment" / "meta.json"
    if (data_dir / "htr.db").exists():
        old = json.loads(meta.read_text()) if meta.exists() else None
        if old is None:
            raise ImportRefused(f"{data_dir} already holds an HTR project that is not an experiment project")
        raise ImportRefused(
            f"{data_dir} already holds an imported experiment (stretch {old['stretch_x']}); delete it first"
        )


def import_dataset(
    images: Path,
    csv_path: Path,
    data_dir: Path,
    stretch: float,
    limit: int | None = None,
    val_size: int = WS.DEFAULT_VAL,
    log=print,
) -> dict:
    guard_data_dir(data_dir, stretch)
    labels, report = read_labels(csv_path)
    if limit:
        labels = labels[:limit]
    missing = [n for n, _ in labels if not (images / n).is_file()]
    if missing:
        raise ImportRefused(f"{len(missing)} listed images are missing, e.g. {missing[:3]}")
    canvas = WordCanvas(stretch)
    settings = Settings.for_dir(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(settings.db_path)
    migrate(conn)
    doc = pages_svc.create_document(conn, "word images")
    entries = []
    duplicates = 0
    for k, (name, label) in enumerate(labels, 1):
        page, created = presegmented.import_word_image(conn, settings, doc["id"], images / name, label, canvas)
        if not created:
            duplicates += 1
            continue
        line = conn.execute("SELECT id FROM lines WHERE page_id = ?", (page["id"],)).fetchone()
        entries.append({"line_id": line["id"], "page_id": page["id"], "file": name, "label": label,
                        "dpi": page["dpi"][0]})  # fmt: skip
        if k % 500 == 0:
            log(f"imported {k}/{len(labels)}")
    splits = WS.build_word_splits([(e["line_id"], e["label"]) for e in entries], val_size)
    by_id = {e["line_id"]: e["label"] for e in entries}
    problems = WS.check_no_leak(splits, by_id)
    if problems:
        raise ImportRefused("split leak: " + "; ".join(problems))
    part = splits.part_of()
    rank = {lid: n for n, lid in enumerate(splits.train)}
    app = splits.app_split()
    with write_tx(conn):
        conn.executemany(
            "INSERT INTO line_splits (line_id, split, assigned_at) VALUES (?, ?, ?)",
            [(lid, sp, now_iso()) for lid, sp in sorted(app.items())],
        )
    for e in entries:
        e["part"] = part[e["line_id"]]
        e["train_rank"] = rank.get(e["line_id"])
        e["key"] = WS.label_key(e["label"])
    meta = {
        "created_at": now_iso(),
        "stretch_x": stretch,
        "preprocess_key": canvas.key,
        "source_rows": report,
        "duplicates_skipped": duplicates,
        "split_rules": {
            "unseen": WS.UNSEEN_RULE,
            "seen": "htr-split-v1 (services.training.natural_split on the line id)",
            "validation": f"{T_VAL} lowest hash, {val_size} samples",
            "subset_order": WS.SUBSET_RULE,
        },
        "counts": {
            "train": len(splits.train),
            "val": len(splits.val),
            "seen": len(splits.seen),
            "unseen": len(splits.unseen),
            "demoted_to_train": splits.demoted,
            "unseen_labels": len({e["key"] for e in entries if e["part"] == "unseen"}),
            "labels": len({e["key"] for e in entries}),
        },
        "entries": entries,
    }
    (data_dir / "experiment").mkdir(exist_ok=True)
    (data_dir / "experiment" / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    base = REPO / "data" / "models" / "base"
    if base.is_dir() and not (data_dir / "models" / "base").exists():
        (data_dir / "models").mkdir(exist_ok=True)
        (data_dir / "models" / "base").symlink_to(base)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    return meta


T_VAL = "htr-val-v1"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--images", type=Path, default=DEFAULT_ROOT / "data")
    ap.add_argument("--csv", type=Path, default=DEFAULT_ROOT / "Prescription_Labels.csv")
    ap.add_argument("--data-dir", type=Path, required=True, help="experiment project directory (created)")
    ap.add_argument("--stretch", type=float, default=3.0, help="horizontal stretch of the word canvas (1 = none)")
    ap.add_argument("--limit", type=int, default=None, help="import only the first N images (testing)")
    ap.add_argument("--val-size", type=int, default=WS.DEFAULT_VAL)
    args = ap.parse_args(argv)
    try:
        meta = import_dataset(args.images, args.csv, args.data_dir, args.stretch, args.limit, args.val_size)
    except ImportRefused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    c = meta["counts"]
    print(f"imported {len(meta['entries'])} word images into {args.data_dir}")
    print(
        f"  train {c['train']}, validation {c['val']}, A (seen words) {c['seen']}, "
        f"B (unseen words) {c['unseen']} ({c['unseen_labels']} labels), "
        f"{c['demoted_to_train']} A candidates returned to training"
    )
    src = meta["source_rows"]
    print(
        f"  dropped: {len(src['dropped_question_mark'])} with '?'; {src['stripped']} labels had outer whitespace "
        f"stripped; {meta['duplicates_skipped']} duplicate files"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
