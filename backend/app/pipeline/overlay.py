"""Overlay PNG: polygons, baselines and reading-order indices drawn over the page for visual inspection."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .segment import PageSegmentation

PALETTE = [(220, 50, 47), (38, 139, 210), (133, 153, 0), (203, 75, 22), (108, 113, 196), (42, 161, 152)]


def render_overlay(img: Image.Image, seg: PageSegmentation) -> Image.Image:
    base = img.convert("RGBA")
    fill = Image.new("RGBA", base.size, (0, 0, 0, 0))
    fd = ImageDraw.Draw(fill)
    for ln in seg.lines:
        c = PALETTE[ln.index % len(PALETTE)]
        fd.polygon(ln.polygon, fill=c + (60,), outline=c + (255,))
    out = Image.alpha_composite(base, fill)
    d = ImageDraw.Draw(out)
    fsize = max(14, base.height // 90)
    try:
        font = ImageFont.load_default(size=fsize)
    except TypeError:  # very old Pillow
        font = ImageFont.load_default()
    width = max(2, base.height // 800)
    for ln in seg.lines:
        c = PALETTE[ln.index % len(PALETTE)]
        d.line(ln.baseline, fill=(0, 0, 0, 255), width=width + 2)
        d.line(ln.baseline, fill=c + (255,), width=width)
        x, y = ln.baseline[0]
        label = str(ln.index + 1)
        box = d.textbbox((0, 0), label, font=font)
        tw, th, pad = box[2] - box[0], box[3] - box[1], 2
        # label sits left of the baseline start, clamped inside the page
        lx = max(0, x - tw - 2 * pad - 2)
        ly = max(0, y - th // 2 - pad)
        d.rectangle([lx, ly, lx + tw + 2 * pad, ly + th + 2 * pad], fill=c + (255,))
        d.text((lx + pad - box[0], ly + pad - box[1]), label, fill=(255, 255, 255, 255), font=font)
    return out.convert("RGB")


def write_overlay(img: Image.Image, seg: PageSegmentation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    render_overlay(img, seg).save(tmp, format="PNG")
    tmp.replace(path)
