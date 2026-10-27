#!/usr/bin/env python3
"""Tables and error analysis from the result files written by `scripts/rxhand_experiment.py` (dev tool).

    curve    --project DIR [--prefix curve_]   learning-curve table (A and B, with bootstrap CIs), baselines
    analyze  --project DIR --run NAME          confidence vs error, review order, error analysis, contact sheet

Everything is computed from the saved predictions, so it needs no GPU. Output images go to
<project>/experiment/out/ (inside the gitignored data directory); they contain dataset images, never commit them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

import numpy as np  # noqa: E402

from app.pipeline.evaluate import edit_distance, evaluate, exact_match  # noqa: E402


def load(project: Path, name: str) -> dict:
    return json.loads((project / "experiment" / "results" / f"{name}.json").read_text())


def pairs_of(part: dict) -> list[tuple[str, str]]:
    return [(p["truth"], p["pred"]) for p in part["predictions"]]


def fmt_ci(ci) -> str:
    return f"{ci[0]:.3f}-{ci[1]:.3f}" if ci else "-"


def row_for(part: dict) -> str:
    return f"{part['cer']:.3f} [{fmt_ci(part['cer_ci95'])}] | {part['wer']:.3f} | {part['exact']:.3f}"


def cmd_curve(a) -> int:
    proj = a.project
    res_dir = proj / "experiment" / "results"
    names = sorted(p.stem for p in res_dir.glob(f"{a.prefix}*.json"))
    all_runs = [load(proj, n) for n in names]
    all_runs.sort(key=lambda r: (r["n_train"], r["name"]))
    # several seeds per N: the reported run is the one with the lowest VALIDATION CER (never a held-out set)
    best: dict[int, dict] = {}
    for r in all_runs:
        if r["n_train"] not in best or r["sets"]["val"]["cer"] < best[r["n_train"]]["sets"]["val"]["cer"]:
            best[r["n_train"]] = r
    runs = [best[n] for n in sorted(best)]
    print("seeds (validation CER, 250 samples): " + "; ".join(
        f"N={n}: " + ", ".join(f"{r['name'].rsplit('_', 1)[-1]} {r['sets']['val']['cer']:.3f} ({r['epochs_run']} ep)"
                               for r in all_runs if r["n_train"] == n) for n in sorted(best)) + "\n")  # fmt: skip
    nA = runs[0]["sets"]["seen"]["n"] if runs else 0
    nB = runs[0]["sets"]["unseen"]["n"] if runs else 0
    print(f"| N train | epochs (best) | train s | peak VRAM MB | A seen words (n={nA}): CER [95% CI] | WER | exact "
          f"| B unseen words (n={nB}): CER [95% CI] | WER | exact | A labels in subset |")  # fmt: skip
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        s = r["sets"]
        vram = f"{r['peak_vram_mb']:.0f}" if r.get("peak_vram_mb") else "-"
        print(f"| {r['n_train']} | {r['epochs_run']} ({r['best_epoch']}) | {r['train_seconds']:.0f} | {vram} | "
              f"{row_for(s['seen'])} | {row_for(s['unseen'])} | {r['seen_coverage_in_subset']:.0%} |")  # fmt: skip
    return 0


# ---- analysis ----------------------------------------------------------------------------------------------------


def align_ops(ref: str, hyp: str) -> list[tuple[str, str, str]]:
    """Levenshtein alignment as (op, ref char, hyp char); op in 'match', 'sub', 'ins', 'del'."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("match" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("del", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", hyp[j - 1]))
            j -= 1
    return ops[::-1]


def spearman(x, y) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)


def auroc_wrong(conf, wrong) -> float:
    """Probability that a randomly chosen wrong prediction has lower confidence than a randomly chosen right one."""
    from scipy.stats import rankdata

    conf, wrong = np.asarray(conf), np.asarray(wrong, dtype=bool)
    n1, n0 = wrong.sum(), (~wrong).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(-conf)  # high rank = low confidence
    return float((ranks[wrong].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def review_stats(rows, frac: float = 0.2, seed: int = 0) -> dict:
    """rows: dicts with conf, truth, pred. Reviewing = looking at the lowest-confidence `frac` of samples first."""
    n = len(rows)
    k = max(1, round(frac * n))
    order = sorted(range(n), key=lambda i: rows[i]["conf"])
    wrong = np.array([r["truth"] != r["pred"] for r in rows])
    ed = np.array([edit_distance(r["truth"], r["pred"]) for r in rows], dtype=float)
    chars = np.array([len(r["truth"]) for r in rows], dtype=float)
    low = order[:k]
    rng = np.random.default_rng(seed)
    rand_wrong, rand_resid = [], []
    for _ in range(500):
        pick = rng.choice(n, size=k, replace=False)
        rand_wrong.append(wrong[pick].sum() / max(1, wrong.sum()))
        keep = np.ones(n, bool)
        keep[pick] = False
        rand_resid.append(ed[keep].sum() / chars.sum())
    keep = np.ones(n, bool)
    keep[low] = False
    return {
        "n": n,
        "wrong_total": int(wrong.sum()),
        "share_wrong_in_lowest": float(wrong[low].sum() / max(1, wrong.sum())),
        "share_wrong_in_random": float(np.mean(rand_wrong)),
        "share_edits_in_lowest": float(ed[low].sum() / max(1.0, ed.sum())),
        "cer_before": float(ed.sum() / chars.sum()),
        "cer_after_fixing_lowest": float(ed[keep].sum() / chars.sum()),
        "cer_after_fixing_random": float(np.mean(rand_resid)),
        "frac": frac,
    }


def cmd_analyze(a) -> int:
    proj = a.project
    meta = json.loads((proj / "experiment" / "meta.json").read_text())
    ent = {e["line_id"]: e for e in meta["entries"]}
    counts = Counter(e["key"] for e in ent.values())
    run = load(proj, a.run)
    out = proj / "experiment" / "out"
    out.mkdir(exist_ok=True)
    report = {}
    for name, key in (("A seen words", "seen"), ("B unseen words", "unseen")):
        part = run["sets"][key]
        rows = part["predictions"]
        conf = np.array([r["conf"] for r in rows])
        cer_i = np.array([edit_distance(r["truth"], r["pred"]) / max(1, len(r["truth"])) for r in rows])
        wrong = np.array([r["truth"] != r["pred"] for r in rows])
        print(f"\n## {name} (n={len(rows)})")
        print(f"confidence vs per-sample CER: Spearman {spearman(conf, cer_i):.3f}; "
              f"AUROC (wrong ranked below right) {auroc_wrong(conf, wrong):.3f}; "
              f"mean conf right {conf[~wrong].mean():.3f} wrong {conf[wrong].mean():.3f}")  # fmt: skip
        for frac in (0.1, 0.2, 0.3):
            rs = review_stats(rows, frac)
            print(f"  lowest {frac:.0%} by confidence: holds {rs['share_wrong_in_lowest']:.1%} of wrong predictions "
                  f"(random order: {rs['share_wrong_in_random']:.1%}), {rs['share_edits_in_lowest']:.1%} of all edits; "
                  f"CER {rs['cer_before']:.3f} -> {rs['cer_after_fixing_lowest']:.3f} after correcting them "
                  f"(random: {rs['cer_after_fixing_random']:.3f})")  # fmt: skip
        # error analysis
        sub, ins, dele = Counter(), Counter(), Counter()
        n_ops = Counter()
        for r in rows:
            for op, x, y in align_ops(r["truth"], r["pred"]):
                n_ops[op] += 1
                if op == "sub":
                    sub[(x, y)] += 1
                elif op == "ins":
                    ins[y] += 1
                elif op == "del":
                    dele[x] += 1
        print(f"  edit operations: {dict(n_ops)}")
        print("  top substitutions (truth->pred): " + ", ".join(f"{x}->{y} {c}" for (x, y), c in sub.most_common(15)))
        print("  top deletions (missed): " + ", ".join(f"{x} {c}" for x, c in dele.most_common(8))
              + " | top insertions: " + ", ".join(f"{y} {c}" for y, c in ins.most_common(8)))  # fmt: skip
        ci_pairs = [(r["truth"].lower(), r["pred"].lower()) for r in rows]
        print(f"  case-insensitive CER {evaluate(ci_pairs).cer:.3f} (case-sensitive {part['cer']:.3f})")
        per_label: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for r in rows:
            per_label[ent[r["id"]]["key"]].append((r["truth"], r["pred"]))
        contrib = sorted(per_label.items(), key=lambda kv: -sum(edit_distance(t, h) for t, h in kv[1]))
        total_edits = sum(edit_distance(t, h) for _, v in contrib for t, h in v)
        print("  labels contributing most edits (share of all edits, samples, own CER): " + "; ".join(
            f"'{k}' {sum(edit_distance(t, h) for t, h in v) / total_edits:.1%} n={len(v)} CER {evaluate(v).cer:.2f}"
            for k, v in contrib[:5]))  # fmt: skip
        top_key = max(per_label, key=lambda k: len(per_label[k]))
        rest = [(r["truth"], r["pred"]) for r in rows if ent[r["id"]]["key"] != top_key]
        n_top = len(per_label[top_key])
        print(f"  most frequent label '{top_key}': {n_top} samples ({n_top / len(rows):.0%}); "
              f"CER without it {evaluate(rest).cer:.3f}")  # fmt: skip
        groups = {
            "length 1-4": lambda r, e: len(r["truth"]) <= 4,
            "length 5-7": lambda r, e: 5 <= len(r["truth"]) <= 7,
            "length 8-10": lambda r, e: 8 <= len(r["truth"]) <= 10,
            "length 11+": lambda r, e: len(r["truth"]) >= 11,
            "dpi 240": lambda r, e: e["dpi"] == 240.0,
            "dpi 610": lambda r, e: e["dpi"] and e["dpi"] > 600,
            "label occurs once in dataset": lambda r, e: counts[e["key"]] == 1,
            "label occurs 2-9 times": lambda r, e: 2 <= counts[e["key"]] <= 9,
            "label occurs 10+ times": lambda r, e: counts[e["key"]] >= 10,
            "multi-word label": lambda r, e: " " in r["truth"],
            "label has a capital": lambda r, e: any(c.isupper() for c in r["truth"]),
            "all lower case": lambda r, e: r["truth"].islower(),
        }
        by = {}
        for g, f in groups.items():
            sel = [(r["truth"], r["pred"]) for r in rows if f(r, ent[r["id"]])]
            if len(sel) >= 5:
                m = evaluate(sel)
                by[g] = {"n": len(sel), "cer": m.cer, "exact": exact_match(sel)}
                print(f"  CER {m.cer:.3f} exact {exact_match(sel):.3f}  n={len(sel):4d}  {g}")
        report[key] = {"by_group": by}
    sheet(proj, run, ent, out / f"worst40_{a.run}.png")
    print(f"\ncontact sheet: {out / f'worst40_{a.run}.png'}")
    (out / f"analysis_{a.run}.json").write_text(json.dumps(report, indent=1))
    return 0


def sheet(proj: Path, run: dict, ent: dict, path: Path, k: int = 40) -> None:
    from PIL import Image, ImageDraw, ImageFont

    rows = []
    for key, tag in (("seen", "A"), ("unseen", "B")):
        for r in run["sets"][key]["predictions"]:
            ed = edit_distance(r["truth"], r["pred"])
            rows.append((ed / max(1, len(r["truth"])), ed, tag, r))
    rows.sort(key=lambda t: (-t[1], -t[0], t[3]["id"]))
    worst = rows[:k]
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    cols, cw, ch = 5, 260, 175
    rws = (len(worst) + cols - 1) // cols
    img = Image.new("RGB", (cols * cw, rws * ch), "white")
    d = ImageDraw.Draw(img)
    for n, (_c, ed, tag, r) in enumerate(worst):
        x, y = (n % cols) * cw, (n // cols) * ch
        e = ent[r["id"]]
        im = Image.open(proj / "originals" / sorted((proj / "originals").glob(Path(e["file"]).stem + "_*"))[0].name)
        img.paste(im.convert("RGB").resize((128, 128)), (x + 4, y + 4))
        d.rectangle([x + 2, y + 2, x + cw - 3, y + ch - 3], outline=(180, 180, 180))
        d.text((x + 138, y + 6), f"set {tag}  dpi {e['dpi']:.0f}", fill=(90, 90, 90), font=font)
        d.text((x + 138, y + 28), "truth:", fill=(0, 100, 0), font=font)
        d.text((x + 138, y + 44), r["truth"][:20], fill=(0, 100, 0), font=font)
        d.text((x + 138, y + 68), "pred:", fill=(170, 0, 0), font=font)
        d.text((x + 138, y + 84), r["pred"][:20], fill=(170, 0, 0), font=font)
        d.text((x + 138, y + 108), f"edits {ed}  conf {r['conf']:.2f}", fill=(60, 60, 60), font=font)
        d.text((x + 4, y + 136), e["file"], fill=(120, 120, 120), font=font)
    img.save(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curve")
    c.add_argument("--project", type=Path, required=True)
    c.add_argument("--prefix", default="curve_")
    c.set_defaults(fn=cmd_curve)
    z = sub.add_parser("analyze")
    z.add_argument("--project", type=Path, required=True)
    z.add_argument("--run", required=True)
    z.set_defaults(fn=cmd_analyze)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
