import numpy as np
import pytest
from PIL import Image

from app.pipeline import preprocess as pp
from tests.test_inspect_images import make_page


def test_scale_factor_only_outside_usable_range():
    assert pp.scale_factor(40, 40) == 1.0
    assert pp.scale_factor(25, 40) == 1.0  # inside the usable range: untouched
    assert pp.scale_factor(None, 40) == 1.0
    assert pp.scale_factor(10, 40) == pytest.approx(4.0)  # far too small: upscale (clamped)
    assert pp.scale_factor(200, 40) == pytest.approx(0.25)  # far too large: downscale (clamped)
    assert pp.scale_factor(1000, 40) == 0.25  # clamped


def test_normalize_contrast_flattens_gradient():
    page = make_page(line_h=30, pitch=70).astype(float)
    grad = np.linspace(1.0, 0.6, page.shape[1])[None, :]
    out = pp.normalize_contrast(page * grad)
    assert out.min() >= 0 and out.max() <= 255
    paper_left, paper_right = np.median(out[:100, :200]), np.median(out[:100, -200:])
    assert abs(paper_left - paper_right) < 10


@pytest.mark.parametrize("angle", [-2.0, 1.0, 3.0])
def test_skew_estimated_and_corrected(angle):
    page = Image.fromarray(make_page(line_h=30, pitch=70))
    skewed = np.asarray(page.rotate(angle, resample=Image.BICUBIC, fillcolor=255), dtype=float)
    est = pp.estimate_skew(skewed)
    assert est == pytest.approx(angle, abs=0.5)
    fixed = pp.rotate_gray(skewed, -est)
    assert abs(pp.estimate_skew(fixed)) < 0.4


def test_large_skew_not_corrected():
    page = Image.fromarray(make_page(line_h=30, pitch=70))
    big = np.asarray(page.rotate(12, resample=Image.BICUBIC, fillcolor=255), dtype=float)
    _, info = pp.preprocess_array(big)
    assert not info.deskewed or abs(info.skew_deg) <= pp.MAX_SKEW_DEG


def test_rescales_only_when_far_outside_range():
    ok, info_ok = pp.preprocess_array(make_page(line_h=30, pitch=70).astype(float))
    assert not info_ok.rescaled and ok.size == (1240, 1754)
    huge = make_page(width=2480, height=3508, line_h=200, pitch=440).astype(float)
    img, info = pp.preprocess_array(huge, target_line_height=40)
    assert info.rescaled and info.scale < 0.5 and img.width < 2480
    assert info.line_height_after == pytest.approx(40, rel=0.4)
    tiny = make_page(width=900, height=1200, line_h=8, pitch=18, margin=60).astype(float)
    img, info = pp.preprocess_array(tiny, target_line_height=40)
    assert info.rescaled and info.scale > 1 and img.width > 900


def test_preprocess_file_never_touches_original(tmp_path):
    src = tmp_path / "a.png"
    Image.fromarray(make_page()).save(src)
    before = src.read_bytes()
    info = pp.preprocess_file(src, tmp_path / "out" / "a.png")
    assert src.read_bytes() == before
    assert (tmp_path / "out" / "a.png").exists() and info.width > 0
