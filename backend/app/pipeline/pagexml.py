"""PAGE-XML (PAGE 2019-07-15) writer and reader for baseline segmentations.

Coordinates are in the pixels of the image named by `imageFilename` (the working copy).
Structure: one TextRegion holding all lines (TextLine = Baseline + Coords polygon), a ReadingOrder that
references the region, and a `readingOrder {index:N;}` custom attribute on each line (0-based, document order).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .segment import Line, PageSegmentation

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
SCHEMA_LOC = f"{NS} {NS}/pagecontent.xsd"
XSI = "http://www.w3.org/2001/XMLSchema-instance"
CREATOR = "htr-m1"
_ORDER_RE = re.compile(r"readingOrder\s*\{\s*index\s*:\s*(\d+)")


def _q(tag: str) -> str:
    return f"{{{NS}}}{tag}"


def _points(pts) -> str:
    return " ".join(f"{x},{y}" for x, y in pts)


def _parse_points(s: str) -> list[tuple[int, int]]:
    out = []
    for tok in s.split():
        x, y = tok.split(",")
        out.append((int(round(float(x))), int(round(float(y)))))
    return out


def build_pagexml(seg: PageSegmentation, image_filename: str) -> ET.ElementTree:
    ET.register_namespace("", NS)
    ET.register_namespace("xsi", XSI)
    root = ET.Element(_q("PcGts"), {f"{{{XSI}}}schemaLocation": SCHEMA_LOC})
    meta = ET.SubElement(root, _q("Metadata"))
    ET.SubElement(meta, _q("Creator")).text = CREATOR
    # schema requires Created/LastChange timestamps
    ET.SubElement(meta, _q("Created")).text = "1970-01-01T00:00:00"
    ET.SubElement(meta, _q("LastChange")).text = "1970-01-01T00:00:00"
    attrs = {"imageFilename": image_filename, "imageWidth": str(seg.width), "imageHeight": str(seg.height)}
    page = ET.SubElement(root, _q("Page"), attrs)
    if seg.lines:
        xs = [x for ln in seg.lines for x, _ in ln.polygon]
        ys = [y for ln in seg.lines for _, y in ln.polygon]
        region_pts = [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]
        ro = ET.SubElement(page, _q("ReadingOrder"))
        og = ET.SubElement(ro, _q("OrderedGroup"), {"id": "ro1"})
        ET.SubElement(og, _q("RegionRefIndexed"), {"index": "0", "regionRef": "r1"})
        reg = ET.SubElement(page, _q("TextRegion"), {"id": "r1", "type": "paragraph"})
        ET.SubElement(reg, _q("Coords"), {"points": _points(region_pts)})
        for ln in sorted(seg.lines, key=lambda line: line.index):
            tl = ET.SubElement(reg, _q("TextLine"), {"id": ln.id, "custom": f"readingOrder {{index:{ln.index};}}"})
            ET.SubElement(tl, _q("Coords"), {"points": _points(ln.polygon)})
            ET.SubElement(tl, _q("Baseline"), {"points": _points(ln.baseline)})
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    return tree


def write_pagexml(seg: PageSegmentation, image_filename: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    build_pagexml(seg, image_filename).write(tmp, encoding="utf-8", xml_declaration=True)
    tmp.replace(path)  # atomic: a crash never leaves a half-written file


def read_pagexml(path: Path) -> tuple[PageSegmentation, str]:
    """Return (segmentation with lines in reading order, imageFilename)."""
    root = ET.parse(path).getroot()
    page = root.find(_q("Page"))
    if page is None:
        raise ValueError(f"{path}: no Page element")
    seg = PageSegmentation(width=int(page.get("imageWidth", 0)), height=int(page.get("imageHeight", 0)))
    found: list[tuple[int, int, Line]] = []
    for doc_pos, tl in enumerate(page.iter(_q("TextLine"))):
        base = tl.find(_q("Baseline"))
        coords = tl.find(_q("Coords"))
        if base is None or coords is None:
            raise ValueError(f"{path}: TextLine {tl.get('id')} lacks Baseline or Coords")
        m = _ORDER_RE.search(tl.get("custom", ""))
        order = int(m.group(1)) if m else doc_pos
        line = Line(
            id=tl.get("id", f"l{doc_pos}"),
            baseline=_parse_points(base.get("points", "")),
            polygon=_parse_points(coords.get("points", "")),
        )
        found.append((order, doc_pos, line))
    for i, (_, _, line) in enumerate(sorted(found, key=lambda t: (t[0], t[1]))):
        line.index = i
        seg.lines.append(line)
    return seg, page.get("imageFilename", "")
