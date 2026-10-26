#!/usr/bin/env python3
"""Recognizer experiments on an imported word-image project (dev tool; see docs/experiments/rxhand.md).

Uses the app's own `KrakenTrainer` and `KrakenRecognizer` (pipeline/train.py, pipeline/recognize.py) on the PAGE-XML
and working images of a project made by `scripts/rxhand_import.py`. It does not go through the job queue because the
learning curve needs (a) a validation set that is NOT part of the training set (the app's train job validates on a
subset of its training lines) and (b) the two held-out sets scored separately; everything else (export format,
augmentation, optimizer, early stopping, model conversion, prediction, confidence) is the app's code.

Subcommands (all write JSON under <project>/experiment/results/ and print one summary line):

    baseline  --project DIR [--model NAME]        pretrained model, no fine-tuning, on validation / A / B
    train     --project DIR --n N --name RUN ...  fine-tune on the first N training samples, then score validation/A/B
    ctc       --project DIR                       time steps per label vs the CTC minimum (no GPU)

Held-out sets are scored only after a run has finished; nothing here reads them to make a decision.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from PIL import Image  # noqa: E402

from app.pipeline import pagexml  # noqa: E402,F401
from app.pipeline.evaluate import bootstrap_cer_ci, evaluate, exact_match  # noqa: E402
from app.pipeline.recognize import (  # noqa: E402
    KrakenRecognizer,
    TrainObserver,
    TrainRequest,
    extract_line_image,
    free_cuda,
    make_trainer,
)
from app.services import wordsplits as WS  # noqa: E402

DEFAULT_BASE = REPO / "data" / "models" / "base" / "revcity-english-handwriting"


def base_model_file(name_or_path: str | None) -> Path:
    if name_or_path and Path(name_or_path).is_file():
        return Path(name_or_path)
    d = DEFAULT_BASE
    if name_or_path:  # a directory name under data/models/base (the app's base models) or data/models/alt
        d = next((p for p in (REPO / "data" / "models" / s / name_or_path for s in ("base", "alt")) if p.is_dir()), d)
    files = sorted(list(d.rglob("*.mlmodel")) + list(d.rglob("*.safetensors")))
    if not files:
        raise SystemExit(f"no model under {d}")
    return files[0]


class Project:
    """An imported experiment project: fixed splits, word images, line crops, PAGE-XML export."""

    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir)
        self.meta = json.loads((self.dir / "experiment" / "meta.json").read_text())
        self.entries = {e["line_id"]: e for e in self.meta["entries"]}
        self.results = self.dir / "experiment" / "results"
        self.results.mkdir(parents=True, exist_ok=True)
        import sqlite3

        conn = sqlite3.connect(self.dir / "htr.db")
        conn.row_factory = sqlite3.Row
        self.rows = {
            r["id"]: r
            for r in conn.execute(
                "SELECT lines.id, lines.text, lines.baseline, lines.polygon, pages.width, pages.height, "
                "pages.working_path FROM lines JOIN pages ON pages.id = lines.page_id"
            )
        }
        conn.close()
        self.splits = WS.WordSplits(
            train=[
                i
                for i, _ in sorted(
                    ((e["line_id"], e["train_rank"]) for e in self.entries.values() if e["train_rank"] is not None),
                    key=lambda t: t[1],
                )
            ],  # fmt: skip
            val=[i for i, e in self.entries.items() if e["part"] == "val"],
            seen=[i for i, e in self.entries.items() if e["part"] == "seen"],
            unseen=[i for i, e in self.entries.items() if e["part"] == "unseen"],
        )
        self.labels = {i: e["label"] for i, e in self.entries.items()}

    def ids(self, part: str) -> list[int]:
        return list(getattr(self.splits, part))

    def xml(self, line_id: int) -> Path:
        from app.services.training import _page_xml

        p = self.dir / "experiment" / "xml" / f"{line_id}.xml"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            r = self.rows[line_id]
            _page_xml([r], r["width"], r["height"], (self.dir / r["working_path"]).resolve(), p)
        return p

    def crop(self, line_id: int) -> Image.Image:
        r = self.rows[line_id]
        with Image.open(self.dir / r["working_path"]) as im:
            im.load()
            c = extract_line_image(im, json.loads(r["baseline"]), json.loads(r["polygon"]))
        assert c is not None
        return c


def score(project: Project, model: Path, parts=("val", "seen", "unseen"), batch: int = 16) -> dict:
    """Predict the given parts with the app's recognizer and compute the metrics; predictions are kept."""
    rec = KrakenRecognizer(model, "auto", batch)
    out = {}
    try:
        for part in parts:
            ids = project.ids(part)
            t0 = time.time()
            preds = rec.predict([project.crop(i) for i in ids])
            pairs = [(project.labels[i], p.text) for i, p in zip(ids, preds, strict=True)]
            m = evaluate(pairs)
            ci = bootstrap_cer_ci(pairs)
            out[part] = {
                "n": len(ids),
                "cer": m.cer,
                "cer_ci95": ci,
                "wer": m.wer,
                "exact": exact_match(pairs),
                "predict_seconds": round(time.time() - t0, 1),
                "predictions": [
                    {"id": i, "truth": project.labels[i], "pred": p.text, "conf": p.confidence}
                    for i, p in zip(ids, preds, strict=True)
                ],
            }
    finally:
        rec.close()
        free_cuda()
    return out


class Log(TrainObserver):
    def __init__(self, tag: str):
        self.tag = tag
        self.history = []

    def on_epoch(self, info):
        self.history.append(info.as_dict())
        print(f"[{self.tag}] epoch {info.epoch}/{info.epochs} loss {info.train_loss} val_acc {info.val_accuracy} "
              f"{info.seconds:.0f}s", flush=True)  # fmt: skip


def cmd_baseline(a) -> int:
    p = Project(a.project)
    model = base_model_file(a.model)
    t0 = time.time()
    res = score(p, model)
    res_out = {"kind": "baseline", "model": str(model.name), "project": p.meta["preprocess_key"], "sets": res}
    name = a.name or f"baseline_{p.meta['preprocess_key'].split(':')[1]}_{model.parent.name}"
    (p.results / f"{name}.json").write_text(json.dumps(res_out))
    print(summary(name, res_out), f"({time.time() - t0:.0f}s)")
    return 0


def summary(name: str, r: dict) -> str:
    s = r["sets"]
    f = lambda part: f"{part} CER {s[part]['cer']:.4f} exact {s[part]['exact']:.3f}" if part in s else ""  # noqa: E731
    return f"{name}: {f('val')} | {f('seen')} | {f('unseen')}"


def cmd_train(a) -> int:
    p = Project(a.project)
    ids = p.splits.train[: a.n] if a.n else p.splits.train
    problems = WS.check_no_leak(p.splits, p.labels, train_n=len(ids))
    if problems:
        raise SystemExit("leak: " + "; ".join(problems))
    run = p.dir / "experiment" / "runs" / a.name
    shutil.rmtree(run, ignore_errors=True)
    run.mkdir(parents=True)
    start = base_model_file(a.model) if not a.scratch else None
    req = TrainRequest(
        start_model=start,  # type: ignore[arg-type]
        out_path=run / "model.mlmodel",
        train_files=[p.xml(i) for i in ids],
        val_files=[p.xml(i) for i in p.splits.val],
        work_dir=run / "work",
        epochs=a.epochs,
        batch_size=a.batch,
        learning_rate=a.lr,
        patience=a.patience,
        device="auto",
        augment=not a.no_augment,
        seed=a.seed,
        num_workers=a.workers,
    )
    if a.scratch:
        req.from_scratch = True
        req.spec = scratch_spec(base_model_file(a.model))
    obs = Log(a.name)
    t0 = time.time()
    out = make_trainer("kraken").train(req, obs)
    res = score(p, out.model_path)
    result = {
        "kind": "train",
        "name": a.name,
        "n_train": len(ids),
        "preprocess_key": p.meta["preprocess_key"],
        "start": str(start.name) if start else "scratch",
        "recipe": {
            "lr": a.lr,
            "batch": a.batch,
            "epochs_max": a.epochs,
            "patience": a.patience,
            "augment": not a.no_augment,
            "seed": a.seed,
        },  # fmt: skip
        "epochs_run": out.epochs_run,
        "best_epoch": out.best_epoch,
        "best_val_accuracy": out.best_val_accuracy,
        "stopped_early": out.stopped_early,
        "train_seconds": round(out.seconds, 1),
        "wall_seconds": round(time.time() - t0, 1),
        "peak_vram_mb": out.peak_vram_mb,
        "new_characters": out.alphabet_added,
        "seen_coverage_in_subset": WS.seen_coverage(p.splits, p.labels, len(ids)),
        "history": obs.history,
        "sets": res,
    }
    (p.results / f"{a.name}.json").write_text(json.dumps(result))
    if not a.keep_model:
        out.model_path.unlink(missing_ok=True)
    shutil.rmtree(run / "work", ignore_errors=True)
    print(summary(a.name, result), f"epochs {out.epochs_run} best {out.best_epoch} {out.seconds:.0f}s "
          f"vram {out.peak_vram_mb:.0f}MB" if out.peak_vram_mb else "")  # fmt: skip
    return 0


def scratch_spec(model_file: Path) -> str:
    """The base model's network architecture without its output layer (Kraken appends `O1c<alphabet>` itself), so a
    from-scratch run trains exactly the same network shape as the fine-tuned runs."""
    import re

    from kraken.models import load_models

    net = [m for m in load_models(str(model_file)) if "recognition" in (m.model_type or [])][0]
    spec = re.sub(r"\s+O\S+\]$", "]", net.spec.strip())
    assert spec.endswith("]") and " O" not in spec, spec
    return spec


def ctc_min_steps(text: str) -> int:
    """Time steps CTC needs for `text`: one per character plus a blank between equal neighbours."""
    return len(text) + sum(1 for x, y in zip(text, text[1:], strict=False) if x == y)


def cmd_ctc(a) -> int:
    """Network time steps for the project's images versus what the labels need (measured with the real network)."""
    import torch
    from kraken.lib.dataset import ImageInputTransforms
    from kraken.models import load_models

    p = Project(a.project)
    net = [m for m in load_models(str(base_model_file(a.model))) if "recognition" in (m.model_type or [])][0]
    _, ch, h, w = net.input
    tf = ImageInputTransforms(1, h, w, ch, (16, 0), valid_norm=False)
    ids = list(p.entries)[:: a.every]
    steps = {}
    with torch.inference_mode():
        for i in ids:
            t = tf(p.crop(i))
            _, ol = net.nn(t.unsqueeze(0), torch.LongTensor([t.shape[2]]))
            steps[i] = int(ol[0])
    short = [i for i in ids if steps[i] < ctc_min_steps(p.labels[i])]
    tight = [i for i in ids if steps[i] < 2 * ctc_min_steps(p.labels[i])]
    n = len(ids)
    print(f"{p.meta['preprocess_key']}: model input height {h}; time steps per image min {min(steps.values())} "
          f"median {sorted(steps.values())[n // 2]} max {max(steps.values())}")  # fmt: skip
    print(f"  labels needing more CTC steps than available (impossible): {len(short)}/{n} = {len(short) / n:.2%}")
    print(f"  labels with fewer than 2 steps per required step (tight): {len(tight)}/{n} = {len(tight) / n:.2%}")
    avg = sum(steps[i] / max(1, len(p.labels[i])) for i in ids) / n
    print(f"  mean time steps per character: {avg:.2f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--project", type=Path, required=True)
    b.add_argument("--model", default=None, help="base model directory name under data/models/base, or a file")
    b.add_argument("--name", default=None)
    b.set_defaults(fn=cmd_baseline)
    t = sub.add_parser("train")
    t.add_argument("--project", type=Path, required=True)
    t.add_argument("--name", required=True)
    t.add_argument("--n", type=int, default=0, help="first N training samples (0 = all)")
    t.add_argument("--model", default=None)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--patience", type=int, default=10)
    t.add_argument("--seed", type=int, default=1234)
    t.add_argument("--no-augment", action="store_true")
    t.add_argument("--workers", type=int, default=0, help="dataloader worker processes")
    t.add_argument("--scratch", action="store_true")
    t.add_argument("--spec", default=None)
    t.add_argument("--keep-model", action="store_true")
    t.set_defaults(fn=cmd_train)
    c = sub.add_parser("ctc")
    c.add_argument("--project", type=Path, required=True)
    c.add_argument("--model", default=None)
    c.add_argument("--every", type=int, default=5)
    c.set_defaults(fn=cmd_ctc)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
