#!/usr/bin/env python3
"""Build a synthetic labeled project through the running API, for the M4 train/predict proof (dev tool only).

NOT real handwriting: pages are rendered with script-style fonts (`make fonts`) from a small random vocabulary, so
any accuracy measured on them says nothing about real handwriting. Its purpose is to exercise the loop end to end
(import -> real Kraken segmentation -> ground truth -> fine-tune -> predict -> CER) with true text known.

Everything is written under one directory (default data/synthetic_project, gitignored). Use a throw-away server:

    mkdir -p data/synthetic_project/server/models
    ln -s "$PWD/data/models/base" data/synthetic_project/server/models/base   # the base model
    HTR_DATA_DIR="$PWD/data/synthetic_project/server" make serve              # own terminal; absolute path
    make synthetic-project        # = scripts/make_synthetic_project.py build --pages 30 --label 170
    # then train and predict from the Training page, or POST /jobs/train and /jobs/predict (docs/api.md)
    make synthetic-report         # = scripts/make_synthetic_project.py report

`build`  renders pages (known text per line), imports and segments them with the real segmenter, matches every
         segmented line to the rendered line it covers (by position; unmatched or merged/split lines are left
         alone), and marks a random `--label` of the matched lines as ground truth with their true text.
`report` after a predict job: CER of the predictions on the matched lines that were NOT labeled, plus samples.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from make_synthetic_pages import available_fonts, render_page

REPO = Path(__file__).resolve().parents[1]
WORDS = (
    "the of and to in is was he for it with as his on be at by had not are but from or have an they which one you "
    "were her all she there would their we him been has when who will more no if out so said what up its about into "
    "than them can only other new some could time these two may then do first any my now such like our over man me "
    "even most made after also did many before must through back years where much your way well down should because "
    "each just those people how too little state good very make world still own see men work long get here between "
    "both life being under never day same another know while last might us great old year off come since against go "
    "came right used take three morning letter garden river harvest weather market winter evening dinner kitchen "
    "window table chair bread water coffee tuesday friday sunday monday april october tomorrow yesterday walked "
    "talked wrote called visited bought finished started waiting remember bring send return collect open close "
    "pt c/o SOB BP rx f/u Dr. Mrs. Mr. mg ml daily twice weekly 120/80 5mg 2x 3.5 12 40 1998 & + ok yes no"
).split()
STYLE_SIZES = [36, 44, 52, 60]


def build_vocabulary(rng: np.random.Generator) -> list[str]:
    """The word list with some capitalized and some punctuated variants, so lines contain mixed case and symbols."""
    out = list(WORDS)
    out += [w.capitalize() for w in WORDS if w.isalpha()][:60]
    out += [w + "," for w in WORDS[:40]] + [w + "." for w in WORDS[40:80]] + ["(" + w + ")" for w in WORDS[80:100]]
    rng.shuffle(out)
    return out


def render_project(out_dir: Path, n_pages: int, seed: int) -> list[dict]:
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    vocab = build_vocabulary(rng)
    fonts = available_fonts()
    rendered = []
    for i in range(n_pages):
        font = fonts[i % len(fonts)]
        size = STYLE_SIZES[int(rng.integers(0, len(STYLE_SIZES)))]
        truth: list[dict] = []
        page = render_page(
            rng,
            font,
            font_size=size,
            ruled=bool(i % 3 == 1),
            margin_note=bool(i % 4),
            n_lines=int(rng.integers(8, 14)),
            vocabulary=vocab,
            truth=truth,
        )
        name = f"synth_{i + 1:03d}.png"
        page.image.save(pages_dir / name, dpi=(300, 300))
        rendered.append({"file": name, "font": Path(font).name, "font_size": size, "truth": truth})
    (out_dir / "rendered.json").write_text(json.dumps(rendered, indent=1))
    return rendered


def match_lines(page: dict, truth: list[dict]) -> dict[int, dict]:
    """Assign rendered lines to segmented lines. A segmented line matches a rendered line when it covers >= 70% of
    its text box, is not much larger than it (no merged lines), and the rendered line is covered by only one
    segmented line. Returns {line id: truth entry}."""
    scale = page["scale"]
    cands = []
    for ln in page["lines"]:
        xs = [p[0] for p in ln["polygon"]]
        ys = [p[1] for p in ln["polygon"]]
        cands.append((ln["id"], min(xs), min(ys), max(xs), max(ys)))
    pairs = []
    for ti, t in enumerate(truth):
        x0, y0, x1, y1 = (v * scale for v in t["bbox"])
        area = max(1.0, (x1 - x0) * (y1 - y0))
        for lid, a0, b0, a1, b1 in cands:
            iw = min(x1, a1) - max(x0, a0)
            ih = min(y1, b1) - max(y0, b0)
            if iw <= 0 or ih <= 0:
                continue
            cover = iw * ih / area
            width_ratio = (a1 - a0) / max(1.0, x1 - x0)
            height_ratio = (b1 - b0) / max(1.0, y1 - y0)
            if cover >= 0.7 and 0.7 <= width_ratio <= 1.5 and height_ratio <= 3.0:
                pairs.append((cover, ti, lid))
    pairs.sort(reverse=True)
    used_t: set[int] = set()
    used_l: set[int] = set()
    out: dict[int, dict] = {}
    for _cover, ti, lid in pairs:
        if ti in used_t or lid in used_l:
            continue
        used_t.add(ti)
        used_l.add(lid)
        out[lid] = truth[ti]
    return out


def wait_job(client: httpx.Client, job_id: int, timeout: float = 600) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.3)
    raise TimeoutError(f"job {job_id} still running")


def cmd_build(args: argparse.Namespace) -> int:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rendered = render_project(out, args.pages, args.seed)
    client = httpx.Client(base_url=args.api, timeout=120)
    client.get("/health").raise_for_status()
    doc = client.post("/documents", json={"name": "synthetic project"}).json()
    line_truth: dict[int, dict] = {}
    n_rendered = n_segmented = 0
    for item in rendered:
        with open(out / "pages" / item["file"], "rb") as f:
            up = client.post("/pages", data={"document_id": str(doc["id"])}, files={"file": (item["file"], f.read())})
        up.raise_for_status()
        page_id = up.json()["id"]
        job = wait_job(client, client.post(f"/pages/{page_id}/segment", json={"force": True}).json()["id"])
        if job["status"] != "succeeded":
            print(f"{item['file']}: segmentation {job['status']}: {job['error']}", file=sys.stderr)
            continue
        page = client.get(f"/pages/{page_id}").json()
        matched = match_lines(page, item["truth"])
        n_rendered += len(item["truth"])
        n_segmented += len(page["lines"])
        for lid, t in matched.items():
            line_truth[lid] = {"page_id": page_id, "text": t["text"], "kind": t["kind"], "font": item["font"]}
        print(
            f"{item['file']} ({item['font']}, {item['font_size']}px): rendered {len(item['truth'])}, "
            f"segmented {len(page['lines'])}, matched {len(matched)}"
        )
    ids = sorted(line_truth)
    rng = np.random.default_rng(args.seed + 1)
    labeled = set(int(i) for i in rng.choice(ids, size=min(args.label, len(ids)), replace=False))
    for lid in sorted(labeled):
        client.patch(
            f"/lines/{lid}", json={"text": line_truth[lid]["text"], "is_ground_truth": True}
        ).raise_for_status()
    for lid in ids:
        line_truth[lid]["labeled"] = lid in labeled
    (out / "line_truth.json").write_text(json.dumps({str(k): v for k, v in line_truth.items()}, indent=1))
    print(
        f"\nrendered {n_rendered} lines, segmented {n_segmented}, matched {len(ids)} "
        f"({100 * len(ids) / max(1, n_rendered):.0f}% of rendered); labeled {len(labeled)} as ground truth, "
        f"{len(ids) - len(labeled)} matched lines left unlabeled for prediction"
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from app.pipeline.evaluate import evaluate  # backend on sys.path via `uv run --project backend` + PYTHONPATH

    truth = {int(k): v for k, v in json.loads((args.out / "line_truth.json").read_text()).items()}
    client = httpx.Client(base_url=args.api, timeout=60)
    pairs, rows = [], []
    for page_id in sorted({t["page_id"] for t in truth.values()}):
        for ln in client.get(f"/pages/{page_id}").json()["lines"]:
            t = truth.get(ln["id"])
            if t is None or t["labeled"] or ln["text_source"] != "predicted":
                continue
            pairs.append((t["text"], ln["text"]))
            rows.append((ln["confidence"], t["text"], ln["text"]))
    if not pairs:
        print("no predicted, unlabeled, matched lines found (run a predict job first)")
        return 1
    m = evaluate(pairs)
    print(f"{len(pairs)} predicted lines that were never labeled: CER {m.cer:.4f}  WER {m.wer:.4f}")
    confs = np.array([r[0] for r in rows])
    errs = np.array([evaluate([(r[1], r[2])]).cer for r in rows])
    print(f"confidence: mean {confs.mean():.3f}, min {confs.min():.3f}, max {confs.max():.3f}")
    if len(rows) > 3 and confs.std() > 0 and errs.std() > 0:
        print(f"correlation(confidence, line CER) = {np.corrcoef(confs, errs)[0, 1]:.2f} (negative is good)")
    rows.sort()
    print("\nlowest confidence:")
    for c, truth_text, pred in rows[:5]:
        print(f"  {c:.3f}  true: {truth_text!r}\n         pred: {pred!r}")
    print("\nhighest confidence:")
    for c, truth_text, pred in rows[-3:]:
        print(f"  {c:.3f}  true: {truth_text!r}\n         pred: {pred!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "synthetic_project")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="render, import, segment, label")
    b.add_argument("--pages", type=int, default=30)
    b.add_argument("--label", type=int, default=170, help="how many matched lines to mark as ground truth")
    b.add_argument("--seed", type=int, default=7)
    b.set_defaults(fn=cmd_build)
    r = sub.add_parser("report", help="CER of predictions on never-labeled lines")
    r.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
