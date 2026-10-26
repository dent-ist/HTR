#!/usr/bin/env python3
"""Simulate the M5 active-learning loop on synthetic pages through the running API (development only).

NOT real handwriting and NOT a human: pages are script-font renderings with known text, and the "reviewer" is a
script that knows the true text. What this exercises and measures is the *machinery* (rounds, review queue order,
review events, effort statistics), not how well the loop works for a real writer.

    mkdir -p data/synthetic_rounds/server/models
    ln -s "$PWD/data/models/base" data/synthetic_rounds/server/models/base
    HTR_DATA_DIR="$PWD/data/synthetic_rounds/server" make serve          # own terminal; absolute path
    make synthetic-rounds                                                # this script

Steps: render and import pages, segment them with the real segmenter, match segmented lines to the rendered text,
label `--initial` lines as ground truth, hold back `--holdback` matched lines that are never labeled or reviewed (a
fixed evaluation pool for the predictions), then repeat `--rounds` times:

  1. POST /jobs/round (train, then predict every non-ground-truth line); record the held-out CER/WER of the new model;
  2. measure, on the whole review queue, whether low-confidence-first finds errors faster than random order;
  3. the simulated reviewer takes `--review` lines from GET /review/next in the order the app serves them, accepts a
     line whose prediction equals the true text and corrects the others (PATCH text + is_ground_truth);
  4. record GET /review/stats (effort per round) and the CER of the held-back pool under the new model.

Seeds are fixed; nothing is tuned between runs. Output: a table on stdout and `<out>/rounds.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from make_synthetic_project import match_lines, render_project, wait_job

from app.pipeline.evaluate import edit_distance, evaluate

REPO = Path(__file__).resolve().parents[1]


def build(client: httpx.Client, out: Path, pages: int, seed: int) -> dict[int, dict]:
    rendered = render_project(out, pages, seed)
    doc = client.post("/documents", json={"name": "synthetic rounds"}).json()
    truth: dict[int, dict] = {}
    for item in rendered:
        with open(out / "pages" / item["file"], "rb") as f:
            up = client.post("/pages", data={"document_id": str(doc["id"])}, files={"file": (item["file"], f.read())})
        up.raise_for_status()
        pid = up.json()["id"]
        job = wait_job(client, client.post(f"/pages/{pid}/segment", json={"force": True}).json()["id"])
        if job["status"] != "succeeded":
            print(f"{item['file']}: segmentation {job['status']}: {job['error']}", file=sys.stderr)
            continue
        page = client.get(f"/pages/{pid}").json()
        for lid, t in match_lines(page, item["truth"]).items():
            truth[lid] = {"page_id": pid, "text": t["text"]}
    return truth


def page_lines(client: httpx.Client, page_ids: set[int]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for pid in sorted(page_ids):
        for ln in client.get(f"/pages/{pid}").json()["lines"]:
            out[ln["id"]] = ln
    return out


def queue_items(client: httpx.Client, exclude: set[int]) -> list[dict]:
    items: list[dict] = []
    seen = set(exclude)
    while True:
        res = client.get("/review/next", params={"limit": 200, "exclude": ",".join(map(str, sorted(seen)))}).json()
        if not res["items"]:
            return items
        items += res["items"]
        seen |= {i["line"]["id"] for i in res["items"]}


def ordering_experiment(items: list[dict], truth: dict[int, dict], k: int, rng: np.random.Generator) -> dict:
    """Does lowest-confidence-first meet errors sooner than random order? (on the lines the reviewer can reach)"""
    wrong = np.array([edit_distance(i["predicted_text"], truth[i["line"]["id"]]["text"]) > 0 for i in items])
    k = min(k, len(items))
    shuffles = [rng.permutation(len(items))[:k] for _ in range(2000)]
    return {
        "queue_lines": len(items),
        "queue_error_rate": float(wrong.mean()),
        "first_k": k,
        "errors_in_first_k_by_confidence": int(wrong[:k].sum()),
        "errors_in_first_k_random_mean": float(np.mean([wrong[s].sum() for s in shuffles])),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "synthetic_rounds")
    ap.add_argument("--pages", type=int, default=36)
    ap.add_argument("--initial", type=int, default=40, help="lines labeled before round 1")
    ap.add_argument("--holdback", type=int, default=60, help="matched lines never labeled or reviewed (fixed pool)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--review", type=int, default=50, help="lines the simulated reviewer handles per round")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)

    client = httpx.Client(base_url=args.api, timeout=120)
    client.get("/health").raise_for_status()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    truth = build(client, args.out, args.pages, args.seed)
    ids = sorted(truth)
    order = [int(i) for i in rng.permutation(ids)]
    holdback = set(order[: args.holdback])
    initial = order[args.holdback : args.holdback + args.initial]
    print(f"matched {len(ids)} lines; holding back {len(holdback)}, labeling {len(initial)} to start")
    for lid in initial:
        client.patch(f"/lines/{lid}", json={"text": truth[lid]["text"], "is_ground_truth": True}).raise_for_status()
    pages = {t["page_id"] for t in truth.values()}
    labeled = set(initial)
    unmatched = {i for i in page_lines(client, pages) if i not in truth}  # segmented lines with no rendered text
    skip = holdback | unmatched  # never offered to the reviewer

    results = []
    for r in range(1, args.rounds + 1):
        t0 = time.time()
        job = client.post("/jobs/round", json={"epochs": args.epochs})
        job.raise_for_status()
        train = wait_job(client, job.json()["id"], timeout=1800)
        if train["status"] != "succeeded":
            print(f"round {r}: training {train['status']}: {train['error']}", file=sys.stderr)
            return 1
        res = train["result"]
        predict = wait_job(client, res["next_job_id"], timeout=600)
        if predict["status"] != "succeeded":
            print(f"round {r}: prediction {predict['status']}: {predict['error']}", file=sys.stderr)
            return 1
        # fixed pool: CER of this round's model on lines that were never labeled or reviewed
        lines = page_lines(client, pages)
        hold = [(truth[i]["text"], lines[i]["text"]) for i in sorted(holdback)]
        hm = evaluate(hold)
        # ordering experiment on what the reviewer can reach (everything not held back, not labeled)
        items = queue_items(client, skip)
        order_stats = ordering_experiment(items, truth, 20, rng)
        # the simulated reviewer
        accepted = corrected = 0
        for _ in range(args.review):
            nxt = client.get("/review/next", params={"limit": 1, "exclude": ",".join(map(str, sorted(skip)))}).json()
            if not nxt["items"]:
                break
            it = nxt["items"][0]
            lid = it["line"]["id"]
            true_text = truth[lid]["text"]
            client.patch(f"/lines/{lid}", json={"text": true_text, "is_ground_truth": True}).raise_for_status()
            labeled.add(lid)
            if it["predicted_text"] == true_text:
                accepted += 1
            else:
                corrected += 1
        stats = client.get("/review/stats").json()
        mine = next((s for s in stats["rounds"] if s["model_id"] == res["model_id"]), None)
        row = {
            "round": r,
            "ground_truth_lines_at_training": res["train_lines"] + res["held_out_lines"],
            "held_out_lines": res["held_out_lines"],
            "heldout_cer": res["cer"],
            "heldout_wer": res["wer"],
            "start_model_heldout_cer": res["base_cer"],
            "fixed_pool_cer": hm.cer,
            "fixed_pool_wer": hm.wer,
            "fixed_pool_lines": hm.lines,
            "reviewed": accepted + corrected,
            "accepted_unchanged": accepted,
            "accepted_rate": accepted / max(1, accepted + corrected),
            "mean_edits_per_reviewed_line": mine["mean_edits_per_line"] if mine else None,
            "mean_edits_per_char": mine["mean_edits_per_char"] if mine else None,
            "ordering": order_stats,
            "seconds": round(time.time() - t0),
            "training_seconds": res["training_seconds"],
        }
        results.append(row)
        print(f"round {r} done in {row['seconds']} s: {json.dumps({k: v for k, v in row.items() if k != 'ordering'})}")
        print(f"   ordering: {json.dumps(order_stats)}")
    (args.out / "rounds.json").write_text(json.dumps(results, indent=1))

    print("\nround | GT lines | held-out CER (n) | held-out WER | fixed-pool CER / WER")
    for x in results:
        print(
            f"{x['round']:>5} | {x['ground_truth_lines_at_training']:>8} | {x['heldout_cer']:.3f} ({x['held_out_lines']}) "
            f"| {x['heldout_wer']:.3f} | {x['fixed_pool_cer']:.3f} / {x['fixed_pool_wer']:.3f}"
        )
    print("\nround | accepted unchanged | edits/line | edits/char | errors in first 20: confidence vs random")
    for x in results:
        o = x["ordering"]
        rand = o["errors_in_first_k_random_mean"]
        print(
            f"{x['round']:>5} | {x['accepted_rate']:.0%} ({x['accepted_unchanged']}/{x['reviewed']}) "
            f"| {x['mean_edits_per_reviewed_line']:.2f} | {x['mean_edits_per_char']:.3f} "
            f"| {o['errors_in_first_k_by_confidence']}/{o['first_k']} vs {rand:.1f}/{o['first_k']} "
            f"(queue error rate {o['queue_error_rate']:.0%})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
