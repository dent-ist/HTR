"""M1 review sheet: sample pages for manual overlay review, then score the filled sheet (plan section 5.2)."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

COLUMNS = ["page", "detected_lines", "correct_lines", "missed_lines", "notes"]
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
            w.writerow([pid, manifest[pid]["segmentation"]["lines"], "", "", ""])


@dataclass
class Score:
    pages: int
    detected: int
    correct: int
    missed: int
    share: float  # percent: correct / (detected + missed)
    decision: str


def decide(share_pct: float) -> str:
    if share_pct > PROCEED:
        return "PROCEED as planned (> 90% correct)"
    if share_pct >= REPAIR:
        return "PRIORITIZE REPAIR TOOLS: build the line repair UI early (70-90% correct)"
    return "TUNE THE SEGMENTER: try other settings, then fine-tune on corrected pages (< 70% correct)"


def _int(path: Path, line: int, page: str, column: str, raw: str | None) -> int:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError(f"{path} line {line} ({page}): {column} is blank (enter 0 if there are none)")
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{path} line {line} ({page}): {column} must be an integer, got {raw!r}") from None
    if value < 0:
        raise ValueError(f"{path} line {line} ({page}): {column} must not be negative")
    return value


def score_sheet(path: Path) -> Score:
    """Share = correct / (detected + missed) over ALL rows; every row needs correct_lines and missed_lines filled in
    (0 is a valid answer; blank is an error so that no page is silently skipped)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or not set(COLUMNS) <= set(rows[0]):
        raise ValueError(f"{path}: expected columns {COLUMNS} and at least one page")
    detected = correct = missed = 0
    for i, row in enumerate(rows, start=2):
        page = row["page"]
        det = _int(path, i, page, "detected_lines", row["detected_lines"])
        cor = _int(path, i, page, "correct_lines", row["correct_lines"])
        mis = _int(path, i, page, "missed_lines", row["missed_lines"])
        if cor > det:
            raise ValueError(f"{path} line {i} ({page}): correct_lines ({cor}) exceeds detected_lines ({det})")
        detected, correct, missed = detected + det, correct + cor, missed + mis
    denom = detected + missed
    if denom == 0:
        raise ValueError(f"{path}: no lines to score (detected and missed are all 0)")
    share = 100.0 * correct / denom
    return Score(len(rows), detected, correct, missed, share, decide(share))
