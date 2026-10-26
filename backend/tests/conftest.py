import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def fake_blla_factory(n_lines=3, fail_on=None):
    """Replacement for segment.run_blla: n horizontal lines, optionally raising for images of a given width."""

    def fake(img, device):
        if fail_on is not None and img.width == fail_on:
            raise ValueError("simulated segmentation failure")
        h = img.height // (n_lines + 1)
        lines = []
        for i in range(n_lines):
            y = h * (i + 1)
            lines.append(
                SimpleNamespace(
                    baseline=[(50, y), (img.width - 50, y)],
                    boundary=[(50, y - 15), (img.width - 50, y - 15), (img.width - 50, y + 10), (50, y + 10)],
                )
            )
        return SimpleNamespace(lines=lines)

    return fake


@pytest.fixture
def fake_blla(monkeypatch):
    from app.pipeline import segment

    def install(**kw):
        monkeypatch.setattr(segment, "run_blla", fake_blla_factory(**kw))

    return install
