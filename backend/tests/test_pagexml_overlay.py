import xml.etree.ElementTree as ET

from PIL import Image

from app.pipeline import overlay, pagexml
from app.pipeline.segment import Line, PageSegmentation


def sample_seg():
    lines = [
        Line("l001", [(10, 50), (200, 52)], [(10, 30), (200, 30), (200, 60), (10, 60)], 0),
        Line("l002", [(10, 120), (180, 118)], [(10, 100), (180, 100), (180, 130), (10, 130)], 1),
    ]
    return PageSegmentation(300, 200, lines, "cpu")


def test_roundtrip(tmp_path):
    seg = sample_seg()
    p = tmp_path / "x" / "p.xml"
    pagexml.write_pagexml(seg, "p.png", p)
    back, img = pagexml.read_pagexml(p)
    assert img == "p.png" and (back.width, back.height) == (300, 200)
    assert [(ln.id, ln.baseline, ln.polygon, ln.index) for ln in back.lines] == [
        (ln.id, ln.baseline, ln.polygon, ln.index) for ln in seg.lines
    ]


def test_structure_is_page_2019(tmp_path):
    p = tmp_path / "p.xml"
    pagexml.write_pagexml(sample_seg(), "p.png", p)
    root = ET.parse(p).getroot()
    assert root.tag == f"{{{pagexml.NS}}}PcGts"
    ns = {"p": pagexml.NS}
    assert len(root.findall(".//p:TextLine/p:Baseline", ns)) == 2
    assert len(root.findall(".//p:TextLine/p:Coords", ns)) == 2
    assert root.find(".//p:ReadingOrder/p:OrderedGroup/p:RegionRefIndexed", ns).get("regionRef") == "r1"
    assert root.find("p:Metadata/p:Created", ns) is not None


def test_reading_order_attribute_wins_over_document_order(tmp_path):
    seg = sample_seg()
    seg.lines[0].index, seg.lines[1].index = 1, 0
    p = tmp_path / "p.xml"
    pagexml.write_pagexml(seg, "p.png", p)
    back, _ = pagexml.read_pagexml(p)
    assert [ln.id for ln in back.lines] == ["l002", "l001"]


def test_empty_page(tmp_path):
    p = tmp_path / "e.xml"
    pagexml.write_pagexml(PageSegmentation(10, 10), "e.png", p)
    back, _ = pagexml.read_pagexml(p)
    assert back.lines == []


def test_overlay_draws_over_page(tmp_path):
    page = Image.new("L", (300, 200), 255)
    out = overlay.render_overlay(page, sample_seg())
    assert out.size == (300, 200) and out.mode == "RGB"
    assert out.getpixel((100, 45)) != (255, 255, 255)  # inside polygon 1
    assert out.getpixel((250, 190)) == (255, 255, 255)  # untouched area
    overlay.write_overlay(page, sample_seg(), tmp_path / "o" / "p.png")
    assert (tmp_path / "o" / "p.png").exists()
