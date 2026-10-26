"""Pure geometry for segmentation repair (no database). Coordinates are working-copy pixels."""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import unary_union

from ..errors import Invalid

Pt = tuple[int, int]
MAX_POLYGON_POINTS = 500
DEFAULT_LINE_HEIGHT = 40.0
# Share of the measured line height above / below the baseline when nothing better is known.
ASCENT_SHARE = 0.7
DESCENT_SHARE = 0.3


def _round_pts(coords) -> list[Pt]:
    return [(round(x), round(y)) for x, y in coords]


def _dedupe(points: Sequence[Pt]) -> list[Pt]:
    out: list[Pt] = []
    for p in points:
        if not out or out[-1] != p:
            out.append(p)
    return out


def orient_left_to_right(baseline: Sequence[Pt]) -> list[Pt]:
    pts = list(baseline)
    return pts[::-1] if pts and pts[0][0] > pts[-1][0] else pts


def page_box(size: tuple[int, int]) -> Polygon:
    return box(0, 0, size[0], size[1])


def polygon_from_shape(shape, size: tuple[int, int], what: str = "polygon") -> list[Pt]:
    """Largest polygon of `shape`, clipped to the page, as <= MAX_POLYGON_POINTS integer points."""
    shape = shape.intersection(page_box(size))
    if isinstance(shape, MultiPolygon):
        shape = max(shape.geoms, key=lambda g: g.area)
    if not isinstance(shape, Polygon) or shape.is_empty or shape.area < 1:
        raise Invalid(f"{what} would be empty")
    tol = 0.0
    while len(shape.exterior.coords) - 1 > MAX_POLYGON_POINTS:
        tol += 0.5
        shape = shape.simplify(tol)
    pts = _dedupe(_round_pts(shape.exterior.coords[:-1]))
    if len(pts) < 3:
        raise Invalid(f"{what} would be degenerate")
    return pts


def to_shape(points: Sequence[Pt]) -> Polygon:
    p = Polygon(points)
    return p if p.is_valid else p.buffer(0)


def merge_polygons(polygons: Sequence[Sequence[Pt]], size: tuple[int, int]) -> list[Pt]:
    """Union of the polygons (their convex hull when they are disjoint), clipped to the page."""
    union = unary_union([to_shape(p) for p in polygons])
    if not isinstance(union, Polygon):
        union = union.convex_hull
    return polygon_from_shape(union, size, "merged polygon")


def merge_baselines(baselines: Sequence[Sequence[Pt]]) -> list[Pt]:
    """All baseline points ordered left to right (stable for equal x)."""
    pts = sorted((p for b in baselines for p in b), key=lambda p: p[0])
    out = _dedupe(pts)
    if len(out) < 2:
        raise Invalid("merged baseline would be degenerate")
    return out


def split_line(
    baseline: Sequence[Pt], polygon: Sequence[Pt], x: float, size: tuple[int, int]
) -> tuple[tuple[list[Pt], list[Pt]], tuple[list[Pt], list[Pt]]]:
    """Cut a line with the vertical at `x`: ((left baseline, left polygon), (right baseline, right polygon))."""
    base = orient_left_to_right(baseline)
    xs = [p[0] for p in base]
    if not (min(xs) + 1 <= x <= max(xs) - 1):
        raise Invalid(f"split position x={x:g} must be inside the line's baseline ({min(xs)} to {max(xs)})")
    cx = round(x)
    left: list[Pt] = []
    right: list[Pt] = []
    cut: Pt | None = None
    for a, b in zip(base, base[1:], strict=False):
        if cut is None:
            left.append(a)
            lo, hi = sorted((a[0], b[0]))
            if lo <= cx <= hi and a[0] != b[0]:
                y = a[1] + (b[1] - a[1]) * (cx - a[0]) / (b[0] - a[0])
                cut = (cx, round(y))
                left.append(cut)
                right.append(cut)
                right.append(b)
        else:
            right.append(b)
    if cut is None:
        raise Invalid("cannot split a vertical baseline")
    left, right = _dedupe(left), _dedupe(right)
    if len(left) < 2 or len(right) < 2:
        raise Invalid("split would leave a half with a degenerate baseline")
    poly = to_shape(polygon)
    big = 1e7
    lp = poly.intersection(box(-big, -big, cx, big))
    rp = poly.intersection(box(cx, -big, big, big))
    return (left, polygon_from_shape(lp, size, "left half")), (right, polygon_from_shape(rp, size, "right half"))


def _ascent_descent_at(baseline: Sequence[Pt], poly: Polygon, height: float) -> list[tuple[float, float]]:
    out = []
    for bx, by in baseline:
        hit = poly.intersection(LineString([(bx, -height), (bx, 2 * height)]))
        if hit.is_empty:
            continue
        miny, maxy = hit.bounds[1], hit.bounds[3]
        if miny <= by <= maxy:
            out.append((by - miny, maxy - by))
    return out


def typical_extent(
    lines: Sequence[tuple[Sequence[Pt], Sequence[Pt]]], size: tuple[int, int], line_height: float | None
) -> tuple[float, float]:
    """Median (ascent, descent) in px of existing (baseline, polygon) pairs; falls back to the page line height."""
    asc: list[float] = []
    desc: list[float] = []
    for baseline, polygon in lines:
        try:
            for a, d in _ascent_descent_at(baseline, to_shape(polygon), size[1]):
                asc.append(a)
                desc.append(d)
        except Exception:  # noqa: BLE001 - a broken polygon must not block drawing a new line
            continue
    if asc:
        return max(1.0, statistics.median(asc)), max(1.0, statistics.median(desc))
    lh = line_height if line_height and line_height > 0 else DEFAULT_LINE_HEIGHT
    return lh * ASCENT_SHARE, lh * DESCENT_SHARE


def polygon_from_baseline(baseline: Sequence[Pt], ascent: float, descent: float, size: tuple[int, int]) -> list[Pt]:
    """Polygon around a left-to-right baseline: the baseline offset up by `ascent` and down by `descent`."""
    w, h = size
    top = [(min(max(x, 0), w), min(max(y - ascent, 0), h)) for x, y in baseline]
    bottom = [(min(max(x, 0), w), min(max(y + descent, 0), h)) for x, y in baseline]
    ring = _round_pts(top + bottom[::-1])
    shape = to_shape(ring)
    return polygon_from_shape(shape, size, "polygon (baseline too short?)")


def region_overlap_share(polygon: Sequence[Pt], region: Polygon) -> float:
    poly = to_shape(polygon)
    if poly.is_empty or poly.area <= 0:
        return 0.0
    return poly.intersection(region).area / poly.area


def region_from_rect(x: float, y: float, w: float, h: float) -> Polygon:
    return box(x, y, x + w, y + h)


def region_from_points(points: Sequence[Pt]) -> Polygon:
    return to_shape(points)


__all__ = [
    "merge_baselines",
    "merge_polygons",
    "orient_left_to_right",
    "polygon_from_baseline",
    "region_from_points",
    "region_from_rect",
    "region_overlap_share",
    "split_line",
    "typical_extent",
]
