"""Training data: which lines count, the fixed train / held-out split, the per-run export, and the summary the UI
shows before a run.

**Usable ground truth** = a line with `is_ground_truth = 1`, `ignored = 0` and non-blank text, on a segmented page.

**The held-out split** (plan 5.4: fixed, so CER is comparable across rounds). Each usable line has a stable hash
`h = int(sha256("htr-split-v1:<line id>")[:8], 16) % 10000`; a line is held out when `h < 1200` (12% in
expectation). The first training run that has at least 8 usable lines writes the assignment of every line it sees
to `line_splits`, and an assigned line never changes sides afterwards, in any later round. If that first
assignment leaves no held-out line at all (likely with few lines: 0.88^8 = 36%), the not-yet-assigned line with the
lowest hash is held out, so there is at least one held-out line whenever there are 8 or more usable lines. Below 8
usable lines nothing is assigned and nothing is held out (a run then reports no CER).

Held-out lines are never trained on and never used for choosing the best epoch. The best epoch and early stopping use
a separate *validation* subset of the training pool (hash "htr-val-v1", ~10%, at least one line once the pool has 10
lines; with fewer lines the validation set is the training set itself, which is stated in the run's manifest).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ..config import Settings
from ..db import now_iso, write_tx
from ..errors import Invalid
from ..pipeline import pagexml
from ..pipeline.recognize import extract_line_image
from ..pipeline.segment import Line, PageSegmentation

log = logging.getLogger(__name__)

SPLIT_RULE_VERSION = "htr-split-v1"
VAL_RULE_VERSION = "htr-val-v1"
EVAL_PERMILLE_X10 = 1200  # of 10000 -> 12%
VAL_THRESHOLD = 1000  # of 10000 -> 10%
MIN_LINES_FOR_HELD_OUT = 8
MIN_LINES_FOR_VALIDATION_SPLIT = 10
RECOMMENDED_FIRST_ROUND = 100
RECOMMENDED_PER_RETRAIN = 50

USABLE_SQL = (
    "lines.is_ground_truth = 1 AND lines.ignored = 0 AND TRIM(lines.text) != '' AND pages.segmented_at IS NOT NULL"
)


def _hash(prefix: str, line_id: int) -> int:
    return int(hashlib.sha256(f"{prefix}:{line_id}".encode()).hexdigest()[:8], 16) % 10000


def split_hash(line_id: int) -> int:
    return _hash(SPLIT_RULE_VERSION, line_id)


def val_hash(line_id: int) -> int:
    return _hash(VAL_RULE_VERSION, line_id)


def natural_split(line_id: int) -> str:
    return "eval" if split_hash(line_id) < EVAL_PERMILLE_X10 else "train"


def usable_lines(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT lines.id, lines.page_id, lines.text, lines.baseline, lines.polygon FROM lines "
        f'JOIN pages ON pages.id = lines.page_id WHERE {USABLE_SQL} ORDER BY lines.page_id, lines."index"'
    ).fetchall()


def page_preprocess_keys(conn: sqlite3.Connection, page_ids) -> dict[int, str]:
    """{page id: line-preprocessing key} (`pipeline.wordcrop`): "page" for ordinary pages."""
    from ..pipeline.wordcrop import key_of_preprocess

    ids = sorted(set(page_ids))
    out: dict[int, str] = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        q = f"SELECT id, preprocess FROM pages WHERE id IN ({','.join('?' for _ in chunk)})"
        for r in conn.execute(q, chunk):
            out[r["id"]] = key_of_preprocess(json.loads(r["preprocess"] or "{}"))
    return out


def assigned_splits(conn: sqlite3.Connection) -> dict[int, str]:
    return {r["line_id"]: r["split"] for r in conn.execute("SELECT line_id, split FROM line_splits")}


def assign_splits(conn: sqlite3.Connection, line_ids: list[int]) -> dict[int, str]:
    """Return the split of every given usable line, persisting new assignments (rule in the module docstring)."""
    current = assigned_splits(conn)
    result = {lid: current[lid] for lid in line_ids if lid in current}
    fresh = [lid for lid in line_ids if lid not in current]
    if len(line_ids) < MIN_LINES_FOR_HELD_OUT:
        # too few lines to hold any out; leave everything unassigned so the first real split sees all of them
        return {**result, **{lid: "train" for lid in fresh}}
    new = {lid: natural_split(lid) for lid in fresh}
    if fresh and "eval" not in {*result.values(), *new.values()}:
        promote = min(fresh, key=lambda lid: (split_hash(lid), lid))
        new[promote] = "eval"
    if new:
        now = now_iso()
        with write_tx(conn):
            conn.executemany(
                "INSERT OR IGNORE INTO line_splits (line_id, split, assigned_at) VALUES (?, ?, ?)",
                [(lid, s, now) for lid, s in new.items()],
            )
    return {**result, **new}


def choose_validation(train_ids: list[int]) -> tuple[list[int], bool]:
    """(validation ids, validation_is_training_set). Deterministic, see the module docstring."""
    if len(train_ids) < MIN_LINES_FOR_VALIDATION_SPLIT:
        return list(train_ids), True
    val = [lid for lid in train_ids if val_hash(lid) < VAL_THRESHOLD]
    if not val:
        val = [min(train_ids, key=lambda lid: (val_hash(lid), lid))]
    return val, False


# ---- summary ---------------------------------------------------------------------------------------------------


def last_finetuned(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM models WHERE kind = 'finetuned' ORDER BY trained_at DESC, id DESC LIMIT 1"
    ).fetchone()


def training_warnings(usable: int, new_since_last: int, has_finetuned: bool, held_out_possible: bool) -> list[str]:
    out = []
    if not has_finetuned and usable < RECOMMENDED_FIRST_ROUND:
        out.append(
            f"Only {usable} ground-truth lines; about {RECOMMENDED_FIRST_ROUND} are recommended for the first "
            "round. You can train anyway, but expect a weak model and a noisy accuracy estimate."
        )
    if has_finetuned and new_since_last < RECOMMENDED_PER_RETRAIN:
        out.append(
            f"Only {new_since_last} new ground-truth lines since the last training round; about "
            f"{RECOMMENDED_PER_RETRAIN} are recommended before retraining."
        )
    if usable < MIN_LINES_FOR_HELD_OUT:
        out.append(
            f"With fewer than {MIN_LINES_FOR_HELD_OUT} ground-truth lines no lines are held out, so no CER or WER "
            "will be measured for this run."
        )
    elif not held_out_possible:
        out.append("No held-out lines are available for this run; accuracy cannot be measured.")
    return out


def summary(conn: sqlite3.Connection) -> dict:
    lines = usable_lines(conn)
    ids = [r["id"] for r in lines]
    current = assigned_splits(conn)
    eval_assigned = sum(1 for lid in ids if current.get(lid) == "eval")
    unassigned = sum(1 for lid in ids if lid not in current)
    last = last_finetuned(conn)
    if last is not None:
        seen = {r["line_id"] for r in conn.execute("SELECT line_id FROM model_lines WHERE model_id = ?", (last["id"],))}
        new_since = sum(1 for lid in ids if lid not in seen)
    else:
        new_since = len(ids)
    # would a run right now have held-out lines?
    projected_eval = eval_assigned
    if len(ids) >= MIN_LINES_FOR_HELD_OUT:
        projected_eval += sum(1 for lid in ids if lid not in current and natural_split(lid) == "eval")
        if projected_eval == 0 and unassigned:
            projected_eval = 1
    predictable = conn.execute(
        "SELECT COUNT(*) FROM lines JOIN pages ON pages.id = lines.page_id WHERE lines.is_ground_truth = 0 "
        "AND lines.ignored = 0 AND (lines.text_source IS NULL OR lines.text_source = 'predicted') "
        "AND pages.segmented_at IS NOT NULL"
    ).fetchone()[0]
    active = conn.execute("SELECT id, name, kind, cer FROM models WHERE is_active = 1").fetchone()
    if last is not None:
        suggested = new_since >= RECOMMENDED_PER_RETRAIN
        reason = f"{new_since} new ground-truth lines since the last round (about {RECOMMENDED_PER_RETRAIN} suggested)"
    else:
        suggested = len(ids) >= RECOMMENDED_FIRST_ROUND
        reason = f"{len(ids)} ground-truth lines (about {RECOMMENDED_FIRST_ROUND} suggested for a first round)"
    return {
        "ground_truth_lines": len(ids),
        "held_out_lines": eval_assigned,
        "unassigned_lines": unassigned,
        "projected_held_out_lines": projected_eval,
        "train_pool_lines": len(ids) - eval_assigned,
        "new_since_last_round": new_since,
        "last_round": (
            {"model_id": last["id"], "trained_at": last["trained_at"], "train_lines": last["train_lines"]}
            if last
            else None
        ),
        "recommended_first_round": RECOMMENDED_FIRST_ROUND,
        "recommended_per_retrain": RECOMMENDED_PER_RETRAIN,
        "recommended": RECOMMENDED_PER_RETRAIN if last else RECOMMENDED_FIRST_ROUND,
        "predictable_lines": predictable,
        "active_model": dict(active) if active else None,
        "warnings": training_warnings(len(ids), new_since, last is not None, projected_eval > 0),
        "retrain_suggested": suggested,
        "retrain_reason": reason,
    }


def check_can_train(conn: sqlite3.Connection) -> list[str]:
    """Hard checks for POST /jobs/train; returns the non-blocking warnings."""
    s = summary(conn)
    if s["ground_truth_lines"] == 0:
        raise Invalid(
            "there are no usable ground-truth lines to train on: mark transcribed lines as ground truth "
            "(and not ignored) first",
            "no_ground_truth",
            "Transcribe some lines in the page editor and press Enter to confirm them.",
        )
    return s["warnings"]


# ---- export ----------------------------------------------------------------------------------------------------


@dataclass
class RunExport:
    run_dir: Path
    train_files: list[Path]
    val_files: list[Path]
    eval_files: list[Path]
    eval_samples: list[tuple[int, Image.Image, str]] = field(default_factory=list)  # (line id, crop, text)
    manifest: dict = field(default_factory=dict)
    skipped_unreadable: list[int] = field(default_factory=list)


def _page_xml(rows: list[sqlite3.Row], width: int, height: int, image_path: Path, path: Path) -> None:
    seg = PageSegmentation(width=width, height=height)
    for i, r in enumerate(rows):
        seg.lines.append(
            Line(
                id=f"l{r['id']}",
                baseline=[tuple(p) for p in json.loads(r["baseline"])],
                polygon=[tuple(p) for p in json.loads(r["polygon"])],
                index=i,
                text=r["text"],
                text_source="manual",
                is_ground_truth=True,
            )
        )
    # an absolute imageFilename: Kraken resolves it relative to the XML file, and an absolute path wins
    pagexml.write_pagexml(seg, str(image_path), path)


def export_run(
    conn: sqlite3.Connection,
    settings: Settings,
    run_id: str,
    splits: dict[int, str],
    val_ids: set[int],
    validation_is_training: bool,
    extra_manifest: dict | None = None,
) -> RunExport:
    """Write data/training/<run_id>/: PAGE-XML for train/ val/ eval/ (one file per page, only the lines of that
    part), the masked line crops with `.gt.txt` transcriptions under lines/<part>/, and manifest.json.

    `splits` maps every usable line id to 'train' or 'eval'; `val_ids` is the validation subset of the train lines
    (or all of them if `validation_is_training`)."""
    run_dir = settings.data_dir / "training" / run_id
    for part in ("train", "val", "eval"):
        (run_dir / part).mkdir(parents=True, exist_ok=True)
        (run_dir / "lines" / part).mkdir(parents=True, exist_ok=True)
    rows = {r["id"]: r for r in usable_lines(conn)}
    by_page: dict[int, list[sqlite3.Row]] = {}
    for r in rows.values():
        if r["id"] in splits:
            by_page.setdefault(r["page_id"], []).append(r)
    export = RunExport(run_dir, [], [], [])
    entries = []
    for page_id, prow in sorted(by_page.items()):
        page = conn.execute("SELECT width, height, working_path FROM pages WHERE id = ?", (page_id,)).fetchone()
        image_path = (settings.data_dir / page["working_path"]).resolve()
        parts: dict[str, list[sqlite3.Row]] = {"train": [], "val": [], "eval": []}
        for r in prow:
            if splits[r["id"]] == "eval":
                parts["eval"].append(r)
            else:
                parts["train"].append(r)
                if r["id"] in val_ids:
                    parts["val"].append(r)
        with Image.open(image_path) as im:
            im.load()
            for part, prs in parts.items():
                if not prs:
                    continue
                xml = run_dir / part / f"page_{page_id}.xml"
                _page_xml(prs, page["width"], page["height"], image_path, xml)
                {"train": export.train_files, "val": export.val_files, "eval": export.eval_files}[part].append(xml)
                for r in prs:
                    crop = extract_line_image(im, json.loads(r["baseline"]), json.loads(r["polygon"]))
                    if crop is None:
                        if part == "train":
                            export.skipped_unreadable.append(r["id"])
                        continue
                    stem = run_dir / "lines" / part / f"line_{r['id']}"
                    crop.save(stem.with_suffix(".png"))
                    stem.with_suffix(".gt.txt").write_text(r["text"], encoding="utf-8")
                    if part == "eval":
                        export.eval_samples.append((r["id"], crop, r["text"]))
        for r in prow:
            entries.append(
                {
                    "line_id": r["id"],
                    "page_id": page_id,
                    "split": splits[r["id"]],
                    "validation": r["id"] in val_ids,
                    "chars": len(r["text"]),
                }
            )
    n_eval = sum(1 for s in splits.values() if s == "eval")
    export.manifest = {
        "run_id": run_id,
        "created_at": now_iso(),
        "split_rule": {
            "version": SPLIT_RULE_VERSION,
            "held_out_percent": EVAL_PERMILLE_X10 / 100,
            "min_lines_for_held_out": MIN_LINES_FOR_HELD_OUT,
            "persisted_in": "line_splits",
        },
        "validation_rule": {
            "version": VAL_RULE_VERSION,
            "percent": VAL_THRESHOLD / 100,
            "validation_is_training_set": validation_is_training,
        },
        "counts": {
            "usable": len(splits),
            "train": len(splits) - n_eval,
            "validation": len(val_ids),
            "held_out": n_eval,
            "unreadable_train_lines": len(export.skipped_unreadable),
        },
        "lines": entries,
        **(extra_manifest or {}),
    }
    (run_dir / "manifest.json").write_text(json.dumps(export.manifest, indent=1), encoding="utf-8")
    return export
