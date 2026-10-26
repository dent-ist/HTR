import csv

import pytest

from app.pipeline import review


def manifest(n=30):
    return {
        f"p{i:02d}": {
            "warnings": ["w"] * (i % 7 if i % 7 > 4 else 0),
            "segmentation": {"status": "ok", "lines": 10 + i},
        }
        for i in range(n)
    }


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(review.COLUMNS)
        w.writerows(rows)


def test_sample_includes_most_warned_pages_and_size():
    m = manifest()
    pages = review.sample_pages(m, n=20)
    assert len(pages) == len(set(pages)) == 20
    worst = sorted(m, key=lambda p: (-len(m[p]["warnings"]), p))[:5]
    assert set(worst) <= set(pages)
    assert review.sample_pages(m, n=20) == pages  # deterministic


def test_sample_small_set_and_unsegmented_excluded():
    m = manifest(5)
    m["p00"]["segmentation"] = {"status": "failed"}
    assert review.sample_pages(m, n=20) == ["p01", "p02", "p03", "p04"]


def test_sheet_columns_and_blank_sheet_rejected(tmp_path):
    path = tmp_path / "s.csv"
    review.write_sheet(manifest(4), ["p00", "p01"], path)
    with open(path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["page", "detected_lines", "correct_lines", "missed_lines", "notes"]
    assert rows[1] == ["p00", "10", "", "", ""]
    with pytest.raises(ValueError, match="correct_lines is blank"):
        review.score_sheet(path)


@pytest.mark.parametrize(
    "correct,missed,share,keyword",
    [
        (95, 0, 95, "PROCEED"),
        (90, 0, 90, "REPAIR"),
        (70, 0, 70, "REPAIR"),
        (69, 0, 69, "TUNE"),
        (90, 10, 81.8, "REPAIR"),
    ],
)
def test_share_and_decision(tmp_path, correct, missed, share, keyword):
    path = tmp_path / "s.csv"
    write(path, [["a", 100, correct, missed, ""]])
    sc = review.score_sheet(path)
    assert sc.share == pytest.approx(share, abs=0.1) and keyword in sc.decision


def test_worked_example_from_readme(tmp_path):
    path = tmp_path / "s.csv"
    write(path, [["a", 14, 12, 0, ""], ["b", 10, 7, 2, "margin note merged"], ["c", 16, 16, 1, ""]])
    sc = review.score_sheet(path)
    assert (sc.detected, sc.correct, sc.missed) == (40, 35, 3)
    assert sc.share == pytest.approx(100 * 35 / 43)  # 81.4%
    assert "REPAIR" in sc.decision


@pytest.mark.parametrize(
    "row,match",
    [
        (["a", 10, "", 0, ""], "correct_lines is blank"),
        (["a", 10, 5, "", ""], "missed_lines is blank"),
        (["a", 10, 11, 0, ""], "exceeds detected_lines"),
        (["a", 10, "x", 0, ""], "must be an integer"),
        (["a", 10, 5, -1, ""], "must not be negative"),
    ],
)
def test_invalid_rows(tmp_path, row, match):
    path = tmp_path / "s.csv"
    write(path, [row])
    with pytest.raises(ValueError, match=match):
        review.score_sheet(path)


def test_blank_row_on_any_page_fails(tmp_path):
    path = tmp_path / "s.csv"
    write(path, [["a", 10, 10, 0, ""], ["b", 10, "", "", ""]])
    with pytest.raises(ValueError, match=r"line 3 \(b\)"):
        review.score_sheet(path)


def test_old_four_column_sheet_rejected(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("page,detected_lines,correct_lines,notes" + chr(10) + "a,10,10," + chr(10))
    with pytest.raises(ValueError, match="expected columns"):
        review.score_sheet(path)
