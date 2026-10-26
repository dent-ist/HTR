import inspect_images as ii
import numpy as np
import pytest
from PIL import Image


def make_page(width=1240, height=1754, line_h=30, pitch=70, ruled=False, margin=120, seed=0):
    """Synthetic page: words as ink blocks of height line_h every `pitch` rows."""
    rng = np.random.default_rng(seed)
    page = np.full((height, width), 255, dtype=np.uint8)
    y = margin
    while y + line_h < height - margin:
        if ruled:
            rule_y = y + line_h + (pitch - line_h) // 2
            page[rule_y : rule_y + 2] = 120  # full-width paper rule
        x = margin
        while x < width - margin - 60:
            wlen = min(int(rng.integers(40, 160)), width - margin - x)
            for sx in range(x, x + wlen, 6):  # vertical strokes of varying height
                top = y + int(rng.integers(0, line_h // 4))
                page[top : y + line_h - int(rng.integers(0, line_h // 4)), sx : sx + 3] = 30
            page[y + line_h // 2 : y + line_h // 2 + 3, x : x + wlen] = 30  # joining stroke
            x += wlen + int(rng.integers(15, 35))
        y += pitch
    return page


def save(arr, path, **kw):
    Image.fromarray(arr).save(path, **kw)
    return path


@pytest.mark.parametrize("ruled", [False, True])
@pytest.mark.parametrize("line_h", [24, 40, 60])
def test_line_height_estimate(ruled, line_h):
    pitch = int(line_h * 2.2)
    gray = make_page(line_h=line_h, pitch=pitch, ruled=ruled).astype(float)
    h, p, n = ii.estimate_line_height(gray)
    assert h == pytest.approx(line_h, rel=0.25)
    assert p == pytest.approx(pitch, rel=0.1)
    assert n >= 10


def test_large_scan_scaled_back():
    gray = make_page(width=2480, height=3508, line_h=60, pitch=140).astype(float)
    h, p, _ = ii.estimate_line_height(gray)
    assert h == pytest.approx(60, rel=0.25)
    assert p == pytest.approx(140, rel=0.1)


def test_blank_page_has_no_lines():
    assert ii.estimate_line_height(np.full((1000, 800), 255.0)) == (None, None, 0)


@pytest.mark.parametrize(
    "name,kw",
    [
        ("a.png", {"dpi": (300, 300)}),
        ("b.jpg", {"dpi": (200, 200)}),
        ("c.tif", {"dpi": (300, 300)}),
        ("d.png", {}),
        ("e.jpeg", {}),
        ("f.tiff", {}),
    ],
)
def test_formats_and_dpi(tmp_path, name, kw):
    p = save(make_page(line_h=40, pitch=88), tmp_path / name, **kw)
    rep = ii.inspect_image(p)
    assert rep.error is None
    assert (rep.width, rep.height) == (1240, 1754)
    if kw:
        assert rep.dpi_x == pytest.approx(kw["dpi"][0], abs=1)
        assert not any("DPI" in w for w in rep.warnings)
    else:
        assert rep.dpi_x is None
        assert "missing DPI metadata" in rep.warnings
    assert rep.line_height_px == pytest.approx(40, rel=0.25)


def test_rgba_and_16bit(tmp_path):
    g = make_page(line_h=40, pitch=88)
    rgba = np.stack([g, g, g, np.full_like(g, 255)], axis=-1)
    Image.fromarray(rgba, "RGBA").save(tmp_path / "rgba.png")
    Image.fromarray(g.astype(np.uint16) * 257).save(tmp_path / "g16.png")
    for n in ("rgba.png", "g16.png"):
        rep = ii.inspect_image(tmp_path / n)
        assert rep.error is None, rep.error
        assert rep.line_height_px == pytest.approx(40, rel=0.25)


def test_low_resolution_and_odd_size_warnings(tmp_path):
    small = save(make_page(width=500, height=700, line_h=10, pitch=22, margin=40), tmp_path / "s.png")
    wide = save(make_page(width=2400, height=900, line_h=40, pitch=88, margin=60), tmp_path / "w.png")
    rs, rw = ii.inspect_image(small), ii.inspect_image(wide)
    assert any("low resolution" in w for w in rs.warnings)
    assert any("small image" in w for w in rs.warnings)
    assert any("aspect ratio" in w for w in rw.warnings)


def test_contradictory_dpi_warnings(tmp_path):
    p = save(make_page(line_h=40, pitch=88), tmp_path / "x.png", dpi=(300, 150))
    q = save(make_page(line_h=40, pitch=88), tmp_path / "y.png", dpi=(72, 72))
    assert any("contradictory DPI" in w for w in ii.inspect_image(p).warnings)
    assert any("placeholder" in w for w in ii.inspect_image(q).warnings)


def test_corrupt_file_reported_not_raised(tmp_path):
    (tmp_path / "bad.png").write_bytes(b"not an image")
    assert ii.inspect_image(tmp_path / "bad.png").error


def test_cli_folder(tmp_path, capsys):
    save(make_page(line_h=40, pitch=88, ruled=True), tmp_path / "p1.png", dpi=(300, 300))
    save(make_page(line_h=40, pitch=88, seed=2), tmp_path / "p2.jpg")
    (tmp_path / "notes.txt").write_text("ignore me")
    assert ii.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "p1.png" in out and "p2.jpg" in out and "notes.txt" not in out
    assert "== Summary ==" in out and "images: 2" in out


def test_cli_errors(tmp_path):
    assert ii.main([str(tmp_path / "nope")]) == 2
    assert ii.main([str(tmp_path)]) == 2  # empty folder
