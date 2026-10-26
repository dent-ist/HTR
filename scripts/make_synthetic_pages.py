#!/usr/bin/env python3
"""Generate synthetic handwriting-like page images for development and tests.

Usage: python scripts/make_synthetic_pages.py OUT_DIR [-n 6] [--seed 0]

NOT real data: pages are rendered with script-style fonts (data/fonts/*.ttf if present, see `make fonts`;
otherwise DejaVu Sans), with varied size, slight skew, ruled/unruled paper, noise and a margin note.
Each run also writes OUT_DIR/truth.json with the number of lines rendered per page.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
FONT_DIR = REPO / "data" / "fonts"
FALLBACK_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
]
WORDS = (
    "the letter arrived on tuesday morning with news of the harvest and the weather has been fine "
    "we walked along the river before dinner and spoke about the old house garden market winter "
    "please remember to send the books back by friday evening if the road is dry mother says hello"
).split()


@dataclass
class SyntheticPage:
    image: Image.Image
    main_lines: int
    margin_lines: int

    @property
    def total_lines(self) -> int:
        return self.main_lines + self.margin_lines


def available_fonts() -> list[str]:
    fonts = sorted(str(p) for p in FONT_DIR.glob("*.ttf")) if FONT_DIR.is_dir() else []
    return fonts or [f for f in FALLBACK_FONTS if Path(f).exists()]


def _text_line(rng: np.random.Generator, font: ImageFont.FreeTypeFont, max_w: int, ink: tuple) -> Image.Image:
    """One line of words that fits in max_w, drawn on a transparent strip with a slight random slope."""
    words: list[str] = []
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    target = int(rng.integers(max(3, max_w // 400), max(5, max_w // 150)))
    while len(words) < target + 8:
        words.append(str(rng.choice(WORDS)))
        if probe.textlength(" ".join(words), font=font) > max_w * rng.uniform(0.7, 0.98):
            words.pop()
            break
    text = " ".join(words) or "x"
    w = int(probe.textlength(text, font=font)) + 20
    size = font.size
    strip = Image.new("RGBA", (w, int(size * 2.2)), (255, 255, 255, 0))
    ImageDraw.Draw(strip).text((10, size * 0.4), text, font=font, fill=ink + (255,))
    return strip.rotate(float(rng.uniform(-1.2, 1.2)), resample=Image.BICUBIC, expand=True)


def render_page(
    rng: np.random.Generator,
    font_path: str,
    width: int = 1654,
    height: int = 2339,
    font_size: int = 44,
    ruled: bool = False,
    skew_deg: float = 0.0,
    margin_note: bool = True,
    n_lines: int | None = None,
    tight: bool = False,
) -> SyntheticPage:
    """Render one page. `tight` shrinks line spacing so ascenders/descenders nearly touch."""
    paper = int(rng.integers(238, 252))
    page = Image.new("RGBA", (width, height), (paper, paper - 2, paper - 8, 255))
    font = ImageFont.truetype(font_path, font_size)
    pitch = int(font_size * (1.45 if tight else 2.0))
    left, top = int(width * 0.10), int(height * 0.08)
    note_w = int(width * 0.2) if margin_note else 0
    text_w = width - left - note_w - int(width * 0.08)
    max_lines = (height - top - int(height * 0.08)) // pitch
    count = min(n_lines or max_lines, max_lines)
    draw = ImageDraw.Draw(page)
    ink = (int(rng.integers(15, 60)), int(rng.integers(15, 60)), int(rng.integers(40, 90)))
    for i in range(count):
        y = top + i * pitch
        if ruled:
            ry = y + int(font_size * 1.35)
            draw.line([(0, ry), (width, ry)], fill=(150, 170, 200, 255), width=2)
        strip = _text_line(rng, font, text_w, ink)
        page.alpha_composite(strip, (left + int(rng.integers(0, 25)), y))
    margin_lines = 0
    if margin_note:
        nfont = ImageFont.truetype(font_path, int(font_size * 0.8))
        ny = top + pitch * int(rng.integers(2, max(3, count // 2)))
        for k in range(2):
            strip = _text_line(rng, nfont, note_w - 30, ink)
            page.alpha_composite(strip, (width - note_w - 40, ny + k * int(pitch * 0.9)))
            margin_lines += 1
    arr = np.asarray(page.convert("L"), dtype=np.float64)
    arr += rng.normal(0, 3.0, arr.shape)  # scanner noise
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L")
    if skew_deg:
        img = img.rotate(skew_deg, resample=Image.BICUBIC, fillcolor=paper)
    return SyntheticPage(img, count, margin_lines)


def make_set(out_dir: Path, n: int = 6, seed: int = 0) -> dict[str, dict]:
    """Write n varied pages (png/jpg, with and without DPI tags) and truth.json; return the truth mapping."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    fonts = available_fonts()
    if not fonts:
        raise RuntimeError("no usable font found (run `make fonts` or install DejaVu)")
    truth: dict[str, dict] = {}
    for i in range(n):
        font = fonts[i % len(fonts)]
        size = int(rng.choice([36, 44, 56]))
        ruled = bool(i % 2)
        skew = float(rng.uniform(-1.5, 1.5)) if i % 3 else 0.0
        n_lines = int(rng.integers(8, 15))
        page = render_page(
            rng, font, font_size=size, ruled=ruled, skew_deg=skew, margin_note=i % 4 != 3, n_lines=n_lines
        )
        ext = ".png" if i % 2 == 0 else ".jpg"
        name = f"synthetic_{i + 1:02d}{ext}"
        kw = {"dpi": (300, 300)} if i % 3 == 0 else {}  # leave some without DPI metadata
        page.image.save(out_dir / name, **kw)
        truth[name] = {
            "main_lines": page.main_lines,
            "margin_lines": page.margin_lines,
            "total_lines": page.total_lines,
            "font": Path(font).name,
            "font_size": size,
            "ruled": ruled,
            "skew_deg": round(skew, 2),
        }
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=2))
    return truth


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    truth = make_set(args.out_dir, args.n, args.seed)
    for name, t in truth.items():
        print(f"{name}: {t['total_lines']} lines ({t['main_lines']} main + {t['margin_lines']} margin), {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
