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


def test_sheet_columns_and_empty_sheet_rejected(tmp_path):
    path = tmp_path / "s.csv"
    review.write_sheet(manifest(4), ["p00", "p01"], path)
    with open(path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["page", "detected_lines", "correct_lines", "notes"]
    assert rows[1] == ["p00", "10", "", ""]
    with pytest.raises(ValueError, match="no rows"):
        review.score_sheet(path)


@pytest.mark.parametrize(
    "correct,keyword",
    [(95, "PROCEED"), (90, "REPAIR"), (75, "REPAIR"), (70, "REPAIR"), (69, "TUNE"), (10, "TUNE")],
)
def test_decision_thresholds(tmp_path, correct, keyword):
    path = tmp_path / "s.csv"
    write(path, [["a", 100, correct, ""], ["b", 50, "", "not reviewed"]])
    sc = review.score_sheet(path)
    assert sc.share == pytest.approx(correct) and keyword in sc.decision
    assert sc.pages_scored == 1 and sc.pages_blank == 1


def test_invalid_rows(tmp_path):
    path = tmp_path / "s.csv"
    write(path, [["a", 10, 11, ""]])
    with pytest.raises(ValueError, match="between 0 and"):
        review.score_sheet(path)
    write(path, [["a", 10, "x", ""]])
    with pytest.raises(ValueError, match="non-integer"):
        review.score_sheet(path)
