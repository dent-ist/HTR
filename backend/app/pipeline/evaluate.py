"""Recognition metrics: character error rate (CER) and word error rate (WER).

Definitions (the same for every model, so rounds are comparable):

* Texts are compared exactly as stored: sequences of Unicode code points. No Unicode normalization, **no case
  folding**, no punctuation stripping. A wrong capital letter is an error.
* CER uses the text as it is, whitespace included (a missing or extra space is one edit). WER first tokenizes both
  strings with `str.split()`, i.e. **whitespace is collapsed** (runs of whitespace are one separator, leading and
  trailing whitespace is ignored); words are then compared whole and case-sensitively, punctuation staying attached.
* Edit distance is the Levenshtein distance (insertion, deletion, substitution, each cost 1), computed on code
  points for CER and on tokens for WER.
* Scores over several lines are **micro averaged**: total edits divided by the total number of reference
  characters (words). Long lines weigh more than short ones, and an empty reference line contributes nothing.
  CER/WER can exceed 1.0 when the hypothesis has many insertions.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass


def edit_distance(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
    """Levenshtein distance between two sequences (strings or token lists)."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def words(text: str) -> list[str]:
    """WER tokens: whitespace-separated, whitespace collapsed."""
    return text.split()


@dataclass(frozen=True)
class Metrics:
    lines: int
    chars: int  # reference characters
    char_edits: int
    words: int  # reference words
    word_edits: int

    @property
    def cer(self) -> float | None:
        return self.char_edits / self.chars if self.chars else None

    @property
    def wer(self) -> float | None:
        return self.word_edits / self.words if self.words else None

    def as_dict(self) -> dict:
        return {
            "lines": self.lines,
            "chars": self.chars,
            "char_edits": self.char_edits,
            "words": self.words,
            "word_edits": self.word_edits,
            "cer": self.cer,
            "wer": self.wer,
        }


def line_cer(reference: str, hypothesis: str) -> float | None:
    """CER of one line (None for an empty reference)."""
    return edit_distance(reference, hypothesis) / len(reference) if reference else None


def evaluate(pairs: Sequence[tuple[str, str]]) -> Metrics:
    """Micro-averaged metrics over (reference, hypothesis) pairs."""
    chars = char_edits = n_words = word_edits = 0
    for ref, hyp in pairs:
        chars += len(ref)
        char_edits += edit_distance(ref, hyp)
        ref_w, hyp_w = words(ref), words(hyp)
        n_words += len(ref_w)
        word_edits += edit_distance(ref_w, hyp_w)
    return Metrics(len(pairs), chars, char_edits, n_words, word_edits)


def cer(pairs: Sequence[tuple[str, str]]) -> float | None:
    return evaluate(pairs).cer


def wer(pairs: Sequence[tuple[str, str]]) -> float | None:
    return evaluate(pairs).wer
