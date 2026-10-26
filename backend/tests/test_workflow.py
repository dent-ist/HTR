import json

import htr
import numpy as np
import review_sheet
from make_synthetic_pages import available_fonts, render_page

from app.pipeline.workflow import Workspace, import_folder, segment_pages


def quiet(_msg):
    pass


def make_scans(folder, sizes=((900, 1200), (1000, 1400))):
    folder.mkdir()
    for i, (w, h) in enumerate(sizes):
        page = render_page(np.random.default_rng(i), available_fonts()[0], width=w, height=h, font_size=30, n_lines=5)
        page.image.save(folder / f"scan{i}.png")  # no DPI tag -> warning
    return folder


def test_import_is_idempotent_and_keeps_originals(tmp_path):
    scans = make_scans(tmp_path / "scans")
    data = Workspace(tmp_path / "data")
    assert import_folder(scans, data, log=quiet)["new"] == 2
    stored = sorted(p.read_bytes() for p in data.originals.iterdir())
    assert stored == sorted([(scans / "scan0.png").read_bytes(), (scans / "scan1.png").read_bytes()])
    s2 = import_folder(scans, data, log=quiet)
    assert s2["new"] == 0 and s2["skipped"] == 2 and len(list(data.originals.iterdir())) == 2
    (scans / "copy.png").write_bytes((scans / "scan0.png").read_bytes())  # same content, other name
    assert import_folder(scans, data, log=quiet)["new"] == 0
    rep = json.loads((data.root / "import_report.json").read_text())
    assert len(rep["pages"]) == 2 and any("missing DPI" in w for p in rep["pages"] for w in p["warnings"])
    assert "missing DPI" in (data.root / "import_report.md").read_text()
    assert len(list(data.working.glob("*.png"))) == 2


def test_changed_target_rederives_working_copy(tmp_path):
    scans = make_scans(tmp_path / "scans", sizes=((900, 1200),))
    data = Workspace(tmp_path / "data")
    import_folder(scans, data, 40, log=quiet)
    assert import_folder(scans, data, 60, log=quiet)["rederived"] == 1


def test_bad_image_does_not_abort_import(tmp_path):
    scans = make_scans(tmp_path / "scans", sizes=((900, 1200),))
    (scans / "broken.png").write_bytes(b"not an image")
    data = Workspace(tmp_path / "data")
    stats = import_folder(scans, data, log=quiet)
    assert stats["new"] == 1 and stats["failed"] == 1
    assert len(list(data.originals.iterdir())) == 1  # broken file not kept


def test_segment_resumes_after_failed_page(tmp_path, fake_blla):
    scans = make_scans(tmp_path / "scans")
    data = Workspace(tmp_path / "data")
    import_folder(scans, data, log=quiet)
    widths = {r["preprocess"]["width"] for r in data.load().values()}
    fake_blla(n_lines=3, fail_on=min(widths))
    assert segment_pages(data, device="cpu", log=quiet) == {"done": 1, "skipped": 0, "failed": 1}
    failed = [r for r in data.load().values() if r["segmentation"]["status"] == "failed"]
    assert "simulated" in failed[0]["segmentation"]["error"]
    fake_blla(n_lines=3)  # fixed: only the failed page is redone
    assert segment_pages(data, device="cpu", log=quiet) == {"done": 1, "skipped": 1, "failed": 0}
    assert len(list(data.pagexml.glob("*.xml"))) == 2 and len(list(data.overlays.glob("*.png"))) == 2
    assert all(r["segmentation"]["lines"] == 3 for r in data.load().values())


def test_cli_end_to_end_with_review(tmp_path, fake_blla, capsys):
    scans = make_scans(tmp_path / "scans")
    d = str(tmp_path / "data")
    assert htr.main(["--data", d, "import", str(scans)]) == 0
    fake_blla(n_lines=4)
    assert htr.main(["--data", d, "segment", "--device", "cpu"]) == 0
    assert review_sheet.main(["--data", d, "sample", "-n", "5"]) == 0
    sheet = tmp_path / "data" / "review_sheet.csv"
    lines = sheet.read_text().splitlines()
    sheet.write_text("\n".join([lines[0]] + [ln.replace(",4,,,", ",4,4,0,") for ln in lines[1:]]) + "\n")
    assert review_sheet.main(["--data", d, "score"]) == 0
    out = capsys.readouterr().out
    assert "100.0%" in out and "PROCEED" in out


def test_cli_errors(tmp_path):
    assert htr.main(["--data", str(tmp_path / "d"), "segment"]) == 2
    assert htr.main(["--data", str(tmp_path / "d"), "import", str(tmp_path / "nope")]) == 2
    assert review_sheet.main(["--data", str(tmp_path / "d"), "sample"]) == 2


def test_reported_line_height_matches_working_copy_measurement(tmp_path):
    """Report and rescale decision use the same (deskewed) measurement; the working copy re-measures the same."""
    from app.pipeline import measure

    scans = tmp_path / "scans"
    scans.mkdir()
    page = render_page(np.random.default_rng(3), available_fonts()[0], font_size=44, n_lines=8, skew_deg=1.5)
    page.image.save(scans / "skewed.png")
    data = Workspace(tmp_path / "data")
    import_folder(scans, data, log=quiet)
    (rec,) = data.load().values()
    assert rec["preprocess"]["deskewed"] and not rec["preprocess"]["rescaled"]
    assert rec["line_height_px"] == rec["preprocess"]["line_height_before"]
    work = measure.inspect_image(data.root / rec["working"])
    assert work.line_height_px == rec["line_height_px"]
    assert work.line_pitch_px == rec["line_pitch_px"]
