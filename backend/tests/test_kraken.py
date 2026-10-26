"""Real Kraken integration test (no mocks). Run with `make test-kraken` / `pytest -m kraken`."""

import numpy as np
import pytest
from make_synthetic_pages import available_fonts, render_page

from app.pipeline import pagexml, segment

pytestmark = pytest.mark.kraken


def test_real_blla_on_synthetic_page(tmp_path):
    page = render_page(np.random.default_rng(1), available_fonts()[0], font_size=44, n_lines=8, margin_note=True)
    seg = segment.segment_image(page.image, device="auto")
    # lenient: a smoke test of the real pipeline, not an accuracy benchmark
    assert 0.7 * page.total_lines <= len(seg.lines) <= 1.3 * page.total_lines
    assert all(len(ln.baseline) >= 2 and len(ln.polygon) >= 3 for ln in seg.lines)
    xml = tmp_path / "p.xml"
    pagexml.write_pagexml(seg, "p.png", xml)
    back, _ = pagexml.read_pagexml(xml)
    assert len(back.lines) == len(seg.lines)
    # kraken's own PAGE reader accepts our file
    from kraken.lib.xml import XMLPage

    assert len(XMLPage(xml).to_container().lines) == len(seg.lines)
