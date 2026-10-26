"""M1 review sheet: sample pages for manual overlay review, then score the filled sheet (plan section 5.2)."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

COLUMNS = ["page", "detected_lines", "correct_lines", "notes"]
PROCEED, REPAIR, TUNE = 90.0, 70.0, 70.0


def sample_pages(manifest: dict[str, dict], n: int = 20, seed: int = 0, worst_fraction: float = 0.25) -> list[str]:
    """Pick up to n segmented pages: the ones with the most warnings first (ceil(n*worst_fraction) of them),
    the rest spread by seeded random choice. Result is sorted by page id."""
    segmented = {pid: r for pid, r in manifest.items() if r.get("segmentation", {}).get("status") == "ok"}
    if len(segmented) <= n:
        return sorted(segmented)
    by_warn = sorted(segmented, key=lambda p: (-len(segmented[p]["warnings"]), p))
    k = min(n, math.ceil(n * worst_fraction))
    worst = [p for p in by_warn[:k] if segmented[p]["warnings"]]  # only pages that actually have warnings
    rest = [p for p in sorted(segmented) if p not in worst]
    picked = worst + random.Random(seed).sample(rest, n - len(worst))
    return sorted(picked)


def write_sheet(manifest: dict[str, dict], pages: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for pid in pages:
            w.writerow([pid, manifest[pid]["segmentation"]["lines"], "", ""])


@dataclass
class Score:
    pages_scored: int
    pages_blank: int
    detected: int
    correct: int
    share: float  # percent
    decision: str
    missed: int = 0


def decide(share_pct: float) -> str:
    if share_pct > PROCEED:
        return "PROCEED as planned (> 90% correct)"
    if share_pct >= REPAIR:
        return "PRIORITIZE REPAIR TOOLS: build the line repair UI early (70-90% correct)"
    return "TUNE THE SEGMENTER: try other settings, then fine-tune on corrected pages (< 70% correct)"


def score_sheet(path: Path) -> Score:
    """Share = correct / detected over rows with correct_lines filled in. An optional `missed_lines` column
    (lines the segmenter did not find) is added to the denominator when present."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or not set(COLUMNS[:3]) <= set(rows[0]):
        raise ValueError(f"{path}: expected columns {COLUMNS}")
    detected = correct = missed = scored = blank = 0
    for i, row in enumerate(rows, start=2):
        raw = (row["correct_lines"] or "").strip()
        if not raw:
            blank += 1
            continue
        try:
            det, cor = int(row["detected_lines"]), int(raw)
            mis = int((row.get("missed_lines") or "0").strip() or 0)
        except ValueError as e:
            raise ValueError(f"{path} line {i}: non-integer value ({e})") from e
        if not 0 <= cor <= det or mis < 0:
            raise ValueError(f"{path} line {i} ({row['page']}): correct_lines must be between 0 and detected_lines")
        detected, correct, missed, scored = detected + det, correct + cor, missed + mis, scored + 1
    if scored == 0:
        raise ValueError(f"{path}: no rows have correct_lines filled in")
    denom = detected + missed
    share = 100.0 * correct / denom if denom else 0.0
    return Score(scored, blank, detected, correct, share, decide(share), missed)
