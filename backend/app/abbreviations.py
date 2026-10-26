"""Rule-based abbreviation expansion. Pure functions and small value types: no database, no I/O.

Recognition output stays diplomatic (`pt c/o SOB`); this module maps the written forms in a personal dictionary to
their expansions. The rules, which are also documented in the README, are deliberately simple and predictable.

Tokenization
    A text is split on whitespace; every maximal run of non-whitespace is a *token* (with its character offsets).
    A token is matched whole: an abbreviation is never expanded inside a longer word, and `pt-centered`, `c/o's` or
    `SOBx` do not match the entries `pt`, `c/o` or `SOB`.

Punctuation
    A token may be wrapped in punctuation that is not part of the abbreviation: leading `( [ { " ' ` and the curly
    quotes, trailing `, ; : ! ? ) ] } " '` and a sentence period. For one token the matcher tries candidate strings in
    this order and uses the first one that is in the dictionary:
      1. the whole token (this is how entries that contain punctuation themselves, `c/o`, `w/`, `s.o.b.`, `b.i.d`
         match: the longest candidate always comes first),
      2. then shorter candidates: the fewest stripped characters first, and for the same number trailing
         punctuation before leading. For `(s.o.b.),` the order is `(s.o.b.),`, `(s.o.b.)`, `s.o.b.),`, `(s.o.b.`,
         `s.o.b.)`, `(s.o.b`, `s.o.b.`, `s.o.b`: only punctuation characters are ever stripped.
    The stripped punctuation is kept in the output: `SOB.` -> `shortness of breath.`. Only the matched core is
    replaced (and reported as the span).

Case
    Entries match case-insensitively (`SOB`, `sob` and `Sob` find the entry `sob`), unless the entry has
    `case_sensitive` set, in which case the token must equal the written form exactly. The casing of the output:
      * the token is exactly the written form        -> the expansion as stored;
      * the token is ALL CAPS (two or more letters, none lowercase) and has at most 4 letters -> the expansion as
        stored (`SOB` -> `shortness of breath`; short capitals are acronym-like, not a sentence start or a shout);
      * otherwise, the token starts with an uppercase letter -> the expansion with its first character uppercased,
        the rest untouched (`Pt` -> `Patient`, and the long `APPROX` -> `Approximately`);
      * otherwise                                    -> the expansion as stored.

Ambiguity
    Several entries may share a written form (`SOB` = shortness of breath / son of a bitch / ...). When more than one
    distinct expansion matches a token, the token is *ambiguous*: it is never expanded automatically. It is returned
    as a span with `status = "ambiguous"` and the list of candidate expansions; in the output string it is left as
    written (`ambiguous="raw"`) or marked `SOB[?]` (`ambiguous="mark"`, the marker goes right after the matched core).

Single pass, no re-scanning
    The text is scanned once from left to right; expansions are written to the output but never scanned again, so
    an expansion that contains a word which is itself an entry is not expanded a second time. The functions always
    work on the stored (diplomatic) text; expanded text is never stored.

Spans
    `Expansion.spans` has one `Span` per token that matched or could not be matched: `start`/`end` are character
    offsets of the matched core in the input text (for unmatched tokens, of the whole token), `written` is the text
    at those offsets, `expansion` is the output for expanded tokens (None otherwise), `status` is `expanded`,
    `ambiguous` or `unmatched`, and `candidates` the distinct expansions that matched an ambiguous token.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

# Punctuation that wraps a token without belonging to the abbreviation.
LEADING_PUNCT = frozenset("([{\"'“‘")
TRAILING_PUNCT = frozenset(",;:!?)]}\"'.”’")
MARK = "[?]"
SHORT_CAPS_MAX = 4  # an ALL-CAPS token with at most this many letters keeps the stored casing of the expansion
MAX_WRITTEN = 64
MAX_EXPANSION = 500

_TOKEN = re.compile(r"\S+")

Ambiguous = Literal["raw", "mark"]
Status = Literal["expanded", "ambiguous", "unmatched"]


@dataclass(frozen=True)
class Entry:
    id: int
    written: str
    expansion: str
    case_sensitive: bool = False

    def matches(self, core: str) -> bool:
        """Does this entry apply to the (already punctuation-stripped) candidate `core`?"""
        if self.case_sensitive:
            return core == self.written
        return core.casefold() == self.written.casefold()


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    written: str
    expansion: str | None
    status: Status
    candidates: tuple[str, ...] = ()
    entry_ids: tuple[int, ...] = ()

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "written": self.written,
            "expansion": self.expansion,
            "status": self.status,
            "candidates": list(self.candidates),
            "entry_ids": list(self.entry_ids),
        }


@dataclass
class Expansion:
    text: str
    spans: list[Span] = field(default_factory=list)


@dataclass(frozen=True)
class TokenMatch:
    """One whitespace-delimited token and the entries that applied to it (empty when none did)."""

    start: int  # offsets of the matched core (or of the whole token when nothing matched)
    end: int
    entries: tuple[Entry, ...]


def conflicts(a: Entry, b: Entry) -> bool:
    """Can the two entries match the same token? True for a pair that shares a written form (ignoring case) unless
    both are case sensitive with different exact spellings."""
    if a.written.casefold() != b.written.casefold():
        return False
    return not (a.case_sensitive and b.case_sensitive and a.written != b.written)


def candidate_cores(token: str) -> Iterable[tuple[int, int]]:
    """(start, end) slices of `token` to try, longest first: the whole token, then fewer trailing punctuation
    characters stripped before leading ones (see the module docstring). Slices are never empty."""
    n_lead = 0
    while n_lead < len(token) and token[n_lead] in LEADING_PUNCT:
        n_lead += 1
    n_trail = 0
    while n_trail < len(token) - n_lead and token[len(token) - 1 - n_trail] in TRAILING_PUNCT:
        n_trail += 1
    # A character may be both leading and trailing punctuation (quotes); the loops above never overlap.
    for stripped in range(n_lead + n_trail + 1):
        for lead in range(min(n_lead, stripped) + 1):
            trail = stripped - lead
            if trail > n_trail or lead >= len(token) - trail:
                continue
            yield lead, len(token) - trail


class Dictionary:
    """Entries indexed by their case-folded written form."""

    def __init__(self, entries: Iterable[Entry]):
        self.entries = list(entries)
        self._by_form: dict[str, list[Entry]] = {}
        for e in self.entries:
            self._by_form.setdefault(e.written.casefold(), []).append(e)

    def lookup(self, core: str) -> tuple[Entry, ...]:
        return tuple(e for e in self._by_form.get(core.casefold(), ()) if e.matches(core))

    def match_token(self, token: str) -> tuple[int, int, tuple[Entry, ...]] | None:
        """(start, end, entries) of the first candidate core with a dictionary hit, in `token`'s own offsets."""
        for start, end in candidate_cores(token):
            hits = self.lookup(token[start:end])
            if hits:
                return start, end, hits
        return None


def apply_case(token_core: str, entry: Entry) -> str:
    """The expansion with the casing rules of the module docstring."""
    exp = entry.expansion
    if token_core == entry.written:
        return exp
    letters = [c for c in token_core if c.isalpha()]
    if len(letters) >= 2 and not any(c.islower() for c in letters) and len(letters) <= SHORT_CAPS_MAX:
        return exp
    if token_core[:1].isupper() and exp:
        return exp[0].upper() + exp[1:]
    return exp


def scan(text: str, dictionary: Dictionary) -> list[TokenMatch]:
    """Tokenize `text` and return, for every token that matched at least one entry, the matched core's offsets and
    the entries (more than one when the token is ambiguous). Tokens without a match are omitted."""
    out = []
    for m in _TOKEN.finditer(text):
        hit = dictionary.match_token(m.group())
        if hit:
            start, end, entries = hit
            out.append(TokenMatch(m.start() + start, m.start() + end, entries))
    return out


def _distinct(entries: Sequence[Entry]) -> list[Entry]:
    seen: dict[str, Entry] = {}
    for e in entries:
        seen.setdefault(e.expansion, e)
    return list(seen.values())


def expand(text: str, entries: Iterable[Entry] | Dictionary, ambiguous: Ambiguous = "raw") -> Expansion:
    """Expand the abbreviations in `text` (see the module docstring). `ambiguous` is `raw` (leave ambiguous tokens as
    written) or `mark` (append `[?]` to them)."""
    d = entries if isinstance(entries, Dictionary) else Dictionary(entries)
    out: list[str] = []
    spans: list[Span] = []
    pos = 0
    for m in _TOKEN.finditer(text):
        tok = m.group()
        hit = d.match_token(tok)
        if hit is None:
            spans.append(Span(m.start(), m.end(), tok, None, "unmatched"))
            continue
        start, end, hits = hit
        s, e = m.start() + start, m.start() + end
        core = tok[start:end]
        distinct = _distinct(hits)
        out.append(text[pos:s])
        if len(distinct) == 1:
            expansion = apply_case(core, distinct[0])
            out.append(expansion)
            spans.append(Span(s, e, core, expansion, "expanded", (), tuple(x.id for x in hits)))
        else:
            out.append(core + (MARK if ambiguous == "mark" else ""))
            spans.append(
                Span(s, e, core, None, "ambiguous", tuple(x.expansion for x in distinct), tuple(x.id for x in hits))
            )
        pos = e
    out.append(text[pos:])
    return Expansion("".join(out), spans)


def validate_written(written: str) -> str:
    """Normalize and check the written form of an entry; raises ValueError with a user-facing message."""
    w = written.strip()
    if not w:
        raise ValueError("the written form must not be empty")
    if len(w) > MAX_WRITTEN:
        raise ValueError(f"the written form is longer than {MAX_WRITTEN} characters")
    if re.search(r"\s", w):
        raise ValueError("the written form must be one token without spaces (abbreviations are matched per word)")
    if all(c in LEADING_PUNCT or c in TRAILING_PUNCT for c in w):
        raise ValueError("the written form must contain at least one letter, digit or symbol that is not punctuation")
    return w


def validate_expansion(expansion: str) -> str:
    x = " ".join(expansion.split())
    if not x:
        raise ValueError("the expansion must not be empty")
    if len(x) > MAX_EXPANSION:
        raise ValueError(f"the expansion is longer than {MAX_EXPANSION} characters")
    if any(ord(c) < 32 for c in x):
        raise ValueError("the expansion must not contain control characters")
    return x
