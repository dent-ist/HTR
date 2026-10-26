"""Honest train / held-out splits for datasets with many repeats of the same word (word-level experiments).

A random split of such data mostly measures memorization, so two fixed held-out sets are built and reported
separately, the same for every training-set size:

* **B, "unseen words"**: about 12% of the *distinct labels* (hash of the case-folded, whitespace-collapsed label,
  `htr-unseen-v1`) are held out with **all** their samples, so the model never sees those words. Case variants of a
  word ("Algin"/"algin") share one key and go out together.
* **A, "seen words"**: of the remaining samples, those selected by the app's existing line hash rule
  (`services.training.natural_split`, about 12%), except that a sample stays in A only if its label still occurs in
  the training pool afterwards (a sample that was the last one of its label goes back to training; the held-out set
  must be "seen words" by construction).
* **validation**: a fixed number of samples from the rest, lowest `htr-val-v1` hash (`training.val_hash`), for early
  stopping. Never A or B.
* **train**: everything else, in a fixed pseudo-random order (`htr-sub-v1`), so the first N samples form nested
  training sets for N = 100, 250, ...

`check_no_leak` verifies the properties above and is run by the import tool and by the tests. For the app's own
`line_splits` table (which has only 'train' / 'eval') use `app_split`: A and B are 'eval', everything else 'train'.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from . import training as T

UNSEEN_RULE = "htr-unseen-v1"
SUBSET_RULE = "htr-sub-v1"
UNSEEN_PERMILLE_X10 = 1200  # of 10000 distinct label keys -> about 12%
DEFAULT_VAL = 250


def clean_label(text: str) -> str:
    """Outer whitespace removed (invisible in the image, so it cannot be learned); nothing else is changed."""
    return text.strip()


def label_key(text: str) -> str:
    """Identity of a word for the unseen-words split: case-folded, whitespace collapsed."""
    return re.sub(r"\s+", " ", text.strip()).casefold()


def _h(prefix: str, s: str) -> int:
    return int(hashlib.sha256(f"{prefix}:{s}".encode()).hexdigest()[:8], 16) % 10000


@dataclass
class WordSplits:
    train: list[int] = field(default_factory=list)  # in the fixed nested-subset order
    val: list[int] = field(default_factory=list)
    seen: list[int] = field(default_factory=list)  # set A
    unseen: list[int] = field(default_factory=list)  # set B
    demoted: int = 0  # A candidates sent back to training because their label would not otherwise be seen

    def part_of(self) -> dict[int, str]:
        out = {}
        for name in ("train", "val", "seen", "unseen"):
            for i in getattr(self, name):
                out[i] = name
        return out

    def app_split(self) -> dict[int, str]:
        """The app's two-valued split: held-out (both sets) = 'eval', all the rest = 'train'."""
        return {i: ("eval" if p in ("seen", "unseen") else "train") for i, p in self.part_of().items()}


def build_word_splits(items: Sequence[tuple[int, str]], val_size: int = DEFAULT_VAL) -> WordSplits:
    """`items` = (line id, label). Deterministic: depends only on the ids and labels."""
    label = dict(items)
    key = {i: label_key(text) for i, text in items}
    unseen = sorted(i for i in label if _h(UNSEEN_RULE, key[i]) < UNSEEN_PERMILLE_X10)
    unseen_set = set(unseen)
    rest = [i for i in sorted(label) if i not in unseen_set]
    cand = {i for i in rest if T.natural_split(i) == "eval"}
    left_in_pool: dict[str, int] = {}
    for i in rest:
        if i not in cand:
            left_in_pool[key[i]] = left_in_pool.get(key[i], 0) + 1
    seen = [i for i in rest if i in cand and left_in_pool.get(key[i], 0) > 0]
    pool = [i for i in rest if i not in set(seen)]
    val = sorted(pool, key=lambda i: (T.val_hash(i), i))[: min(val_size, max(0, len(pool) // 5))]
    val_set = set(val)
    train = sorted((i for i in pool if i not in val_set), key=lambda i: (_h(SUBSET_RULE, str(i)), i))
    # a seen-words sample must have its label in the training set (not only in validation)
    train_keys = {key[i] for i in train}
    kept = []
    demoted = 0
    for i in seen:
        if key[i] in train_keys:
            kept.append(i)
        else:
            train.append(i)
            demoted += 1
    return WordSplits(train=train, val=sorted(val), seen=kept, unseen=unseen, demoted=demoted)


def check_no_leak(splits: WordSplits, labels: dict[int, str], train_n: int | None = None) -> list[str]:
    """Problems found (empty list = clean). With `train_n`, only the first `train_n` training samples count as
    trained on (a learning-curve point)."""
    problems = []
    parts = {n: set(getattr(splits, n)) for n in ("train", "val", "seen", "unseen")}
    names = list(parts)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            if parts[names[a]] & parts[names[b]]:
                problems.append(f"{names[a]} and {names[b]} share samples")
    if len(splits.train) != len(parts["train"]):
        problems.append("duplicate ids in train")
    key = {i: label_key(text) for i, text in labels.items()}
    trained = splits.train[:train_n] if train_n else splits.train
    trained_keys = {key[i] for i in trained}
    trained_ids = set(trained)
    for part in ("val", "seen", "unseen"):
        for i in getattr(splits, part):
            if i in trained_ids:
                problems.append(f"{part} sample {i} is trained on")
    unseen_keys = {key[i] for i in splits.unseen}
    if unseen_keys & trained_keys:
        problems.append("an unseen-words label (case-folded) also occurs in the training set")
    if unseen_keys & {key[i] for i in splits.val}:
        problems.append("an unseen-words label occurs in the validation set")
    if unseen_keys & {key[i] for i in splits.seen}:
        problems.append("an unseen-words label occurs in the seen-words set")
    if train_n is None:
        missing = {key[i] for i in splits.seen} - {key[i] for i in splits.train}
        if missing:
            problems.append(f"{len(missing)} seen-words labels do not occur in the training set")
    return problems


def seen_coverage(splits: WordSplits, labels: dict[int, str], train_n: int) -> float:
    """Share of the seen-words samples whose label occurs in the first `train_n` training samples."""
    if not splits.seen:
        return 0.0
    tk = {label_key(labels[i]) for i in splits.train[:train_n]}
    return sum(1 for i in splits.seen if label_key(labels[i]) in tk) / len(splits.seen)
