"""The pure expansion engine (app/abbreviations.py): tokenization, punctuation, case, ambiguity, idempotence."""

from __future__ import annotations

import pytest

from app.abbreviations import (
    Dictionary,
    Entry,
    apply_case,
    candidate_cores,
    conflicts,
    expand,
    scan,
    validate_expansion,
    validate_written,
)


def D(*pairs, cs=()):
    """Dictionary from (written, expansion) pairs; `cs` lists written forms that are case sensitive."""
    return Dictionary([Entry(i + 1, w, x, w in cs) for i, (w, x) in enumerate(pairs)])


BASE = D(
    ("pt", "patient"),
    ("c/o", "complains of"),
    ("w/", "with"),
    ("s.o.b.", "shortness of breath"),
    ("b.i.d", "twice daily"),
    ("SOB", "shortness of breath"),
    ("c", "with"),
    ("approx", "approximately"),
)


def ex(text, d=BASE, **kw):
    return expand(text, d, **kw).text


# ---- tokenization and whole-token matching -----------------------------------------------------------------------


def test_the_basic_sentence_from_the_plan():
    assert ex("pt c/o SOB") == "patient complains of shortness of breath"


def test_never_expands_inside_a_longer_word():
    for word in ("ptosis", "spt", "pt-centered", "SOBx", "xc/o", "c/ot", "withc", "cat", "w/o"):
        assert ex(word) == word, word


def test_whitespace_is_preserved_exactly():
    assert ex("  pt \t c/o\n\nSOB  ") == "  patient \t complains of\n\nshortness of breath  "


def test_empty_and_whitespace_only():
    assert ex("") == ""
    assert ex("   ") == "   "


def test_unmatched_tokens_are_untouched_and_reported():
    res = expand("the pt is fine", BASE)
    assert res.text == "the patient is fine"
    assert [s.status for s in res.spans] == ["unmatched", "expanded", "unmatched", "unmatched"]
    assert (res.spans[1].start, res.spans[1].end, res.spans[1].written) == (4, 6, "pt")


# ---- punctuation -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("pt,", "patient,"),
        ("(pt)", "(patient)"),
        ('"pt"', '"patient"'),
        ("'pt'", "'patient'"),
        ("pt;", "patient;"),
        ("pt:", "patient:"),
        ("pt!", "patient!"),
        ("pt?", "patient?"),
        ("pt.", "patient."),
        ("(pt),", "(patient),"),
        ("[pt]", "[patient]"),
        ("pt...", "patient..."),
        ("“pt”", "“patient”"),
    ],
)
def test_surrounding_punctuation_is_kept(text, want):
    assert ex(text) == want


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("c/o", "complains of"),
        ("c/o,", "complains of,"),
        ("(c/o)", "(complains of)"),
        ("w/", "with"),
        ("w/,", "with,"),
        ("s.o.b.", "shortness of breath"),
        ("S.O.B.", "shortness of breath"),
        ("s.o.b.,", "shortness of breath,"),
        ("(s.o.b.)", "(shortness of breath)"),
        ("s.o.b", "s.o.b"),  # the entry has a trailing dot; without it the token is a different word
        ("b.i.d", "twice daily"),
        ("b.i.d.", "twice daily."),  # the longer candidate "b.i.d." is not an entry, so the period is stripped
        ("b.i.d,", "twice daily,"),
    ],
)
def test_entries_that_contain_punctuation(text, want):
    assert ex(text) == want


def test_longest_whole_token_match_wins_over_stripped():
    d = D(("sob.", "shortness of breath (dot form)"), ("sob", "short"))
    assert ex("sob.", d) == "shortness of breath (dot form)"
    assert ex("sob,", d) == "short,"
    assert ex("sob", d) == "short"


def test_candidate_cores_order_and_never_empty():
    assert list(candidate_cores("pt")) == [(0, 2)]
    got = [("(s.o.b.),")[a:b] for a, b in candidate_cores("(s.o.b.),")]
    assert got[:3] == ["(s.o.b.),", "(s.o.b.)", "s.o.b.),"]
    assert "s.o.b." in got and all(got)
    assert list(candidate_cores('"')) == [(0, 1)]
    assert all(a < b for a, b in candidate_cores("..."))


def test_a_token_that_is_only_punctuation_is_unmatched():
    assert ex("- ... ,") == "- ... ,"


# ---- case --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("pt", "patient"),  # exact
        ("Pt", "Patient"),  # starts uppercase -> first letter of the expansion capitalized
        ("PT", "patient"),  # short ALL CAPS -> expansion as stored
        ("pT", "patient"),  # not uppercase-initial, mixed -> as stored
        ("C", "With"),  # single capital letter is not "ALL CAPS" (needs two letters)
        ("C/O", "complains of"),
        ("C/o", "Complains of"),
        ("APPROX", "Approximately"),  # long ALL CAPS: starts uppercase -> capitalized
        ("Approx", "Approximately"),
        ("approx", "approximately"),
    ],
)
def test_case_rules(text, want):
    assert ex(text) == want


def test_exact_match_keeps_the_stored_casing_even_for_an_uppercase_entry():
    d = D(("SOB", "Shortness Of Breath"))
    assert ex("SOB", d) == "Shortness Of Breath"
    assert ex("sob", d) == "Shortness Of Breath"  # lowercase token, stored casing
    assert ex("Sob", d) == "Shortness Of Breath"  # capitalizing an already uppercase initial changes nothing


def test_only_the_first_character_is_capitalized():
    d = D(("dx", "diagnosis of NIDDM"))
    assert ex("Dx", d) == "Diagnosis of NIDDM"


def test_case_sensitive_entries_need_the_exact_spelling():
    d = D(("US", "ultrasound"), ("us", "ours"), cs=("US",))
    assert ex("US", d) == "US"  # both entries apply to the exact spelling "US": ambiguous, left alone
    assert ex("us", d) == "ours"  # only the case-insensitive entry applies
    d = D(("US", "ultrasound"), cs=("US",))
    assert ex("US", d) == "ultrasound"
    assert ex("us", d) == "us"
    assert ex("Us", d) == "Us"


def test_apply_case_directly():
    e = Entry(1, "pt", "patient")
    assert apply_case("pt", e) == "patient"
    assert apply_case("Pt", e) == "Patient"
    assert apply_case("PT", e) == "patient"
    assert apply_case("PTS", Entry(1, "pts", "patients")) == "patients"
    assert apply_case("PTSDX", Entry(1, "ptsdx", "post trauma")) == "Post trauma"


# ---- ambiguity ---------------------------------------------------------------------------------------------------

AMB = D(("SOB", "shortness of breath"), ("SOB", "son of a bitch"), ("pt", "patient"))


def test_ambiguous_tokens_are_not_expanded_by_default():
    res = expand("pt c/o SOB.", AMB)
    assert res.text == "patient c/o SOB."
    s = res.spans[-1]
    assert (s.status, s.expansion, s.written) == ("ambiguous", None, "SOB")
    assert s.candidates == ("shortness of breath", "son of a bitch")
    assert (s.start, s.end) == (7, 10)


def test_ambiguous_mark_option_marks_after_the_core_before_punctuation():
    assert ex("pt c/o SOB.", AMB, ambiguous="mark") == "patient c/o SOB[?]."
    assert ex("(SOB)", AMB, ambiguous="mark") == "(SOB[?])"
    assert ex("sob", AMB, ambiguous="mark") == "sob[?]"  # marked as written, casing untouched


def test_two_rows_with_the_same_expansion_are_not_ambiguous():
    d = D(("SOB", "shortness of breath"), ("sob", "shortness of breath"))
    assert ex("SOB sob", d) == "shortness of breath shortness of breath"


def test_case_sensitive_entries_can_disambiguate():
    d = D(("SOB", "shortness of breath"), ("sob", "sobbing"), cs=("SOB", "sob"))
    assert ex("SOB sob Sob", d) == "shortness of breath sobbing Sob"


def test_conflicts():
    a, b = Entry(1, "SOB", "x"), Entry(2, "sob", "y")
    assert conflicts(a, b)
    assert not conflicts(Entry(1, "SOB", "x", True), Entry(2, "sob", "y", True))
    assert conflicts(Entry(1, "SOB", "x", True), Entry(2, "sob", "y", False))
    assert not conflicts(Entry(1, "pt", "x"), Entry(2, "ptx", "y"))


# ---- idempotence, spans ------------------------------------------------------------------------------------------


def test_expansions_are_not_rescanned_within_one_pass():
    d = D(("pt", "patient pt"), ("c", "with c"))
    assert ex("pt c", d) == "patient pt with c"


def test_expanding_twice_is_stable_for_a_normal_dictionary():
    once = ex("pt c/o SOB, w/ approx 3 b.i.d")
    assert ex(once) == once


def test_spans_describe_the_input_text():
    text = "Pt c/o (SOB), ok"
    res = expand(text, BASE)
    for s in res.spans:
        assert text[s.start : s.end] == s.written
    expanded = [s for s in res.spans if s.status == "expanded"]
    assert [(s.written, s.expansion) for s in expanded] == [
        ("Pt", "Patient"),
        ("c/o", "complains of"),
        ("SOB", "shortness of breath"),
    ]
    assert all(s.entry_ids for s in expanded)


def test_scan_reports_candidates_per_token():
    ms = scan("SOB pt x", AMB)
    assert [(m.start, m.end, len(m.entries)) for m in ms] == [(0, 3, 2), (4, 6, 1)]


def test_unicode_and_non_ascii():
    d = D(("ño", "nothing"), ("ß", "sharp s"))
    assert ex("Ño, ß", d) == "Nothing, sharp s"


# ---- validation --------------------------------------------------------------------------------------------------


def test_validate_written():
    assert validate_written("  c/o ") == "c/o"
    for bad in ("", "  ", "a b", "(", ",.", "x" * 65):
        with pytest.raises(ValueError):
            validate_written(bad)


def test_validate_expansion():
    assert validate_expansion("  shortness   of breath ") == "shortness of breath"
    for bad in ("", "  ", "x\x01y", "x" * 501):
        with pytest.raises(ValueError):
            validate_expansion(bad)
