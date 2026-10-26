import pytest

from app.pipeline.evaluate import cer, edit_distance, evaluate, line_cer, wer, words


def test_edit_distance_basics():
    assert edit_distance("", "") == 0
    assert edit_distance("abc", "") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("flaw", "lawn") == 2
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("ab", "ba") == 2  # no transposition operation


def test_edit_distance_is_symmetric_and_works_on_token_lists():
    assert edit_distance("saturday", "sunday") == edit_distance("sunday", "saturday") == 3
    assert edit_distance(["a", "b", "c"], ["a", "c"]) == 1


def test_cer_is_exact_about_case_and_whitespace():
    assert cer([("Hello", "hello")]) == pytest.approx(1 / 5)  # no case folding
    assert cer([("a b", "a  b")]) == pytest.approx(1 / 3)  # an extra space is an error for CER
    assert cer([("a b", "ab")]) == pytest.approx(1 / 3)


def test_wer_collapses_whitespace_but_not_case_or_punctuation():
    assert words("  a \t b  c ") == ["a", "b", "c"]
    assert wer([("a b c", "a   b\tc ")]) == 0.0
    assert wer([("Hello world", "hello world")]) == pytest.approx(1 / 2)
    assert wer([("end.", "end")]) == 1.0
    assert wer([("one two three four", "one three four")]) == pytest.approx(1 / 4)


def test_corpus_scores_are_micro_averaged():
    pairs = [("abcd", "abcd"), ("ab", "xx")]  # 0 + 2 edits over 6 reference characters
    m = evaluate(pairs)
    assert (m.lines, m.chars, m.char_edits) == (2, 6, 2)
    assert m.cer == pytest.approx(2 / 6)
    # not the mean of per-line CERs (which would be 0.5)
    assert (line_cer("abcd", "abcd") + line_cer("ab", "xx")) / 2 == 0.5


def test_cer_can_exceed_one_and_empty_cases():
    assert cer([("a", "abcd")]) == 3.0
    assert cer([("abc", "")]) == 1.0
    assert cer([]) is None and wer([]) is None
    assert cer([("", "x")]) is None  # empty reference: nothing to normalize by
    assert line_cer("", "x") is None


def test_metrics_dict_has_everything():
    d = evaluate([("a b", "a c")]).as_dict()
    assert d == {"lines": 1, "chars": 3, "char_edits": 1, "words": 2, "word_edits": 1, "cer": 1 / 3, "wer": 0.5}
