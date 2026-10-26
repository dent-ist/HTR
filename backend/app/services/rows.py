"""Row <-> JSON-able dict helpers shared by the services and the API."""

from __future__ import annotations

import json
import sqlite3


def json_loads(s: str):
    return json.loads(s)


def json_dumps_points(points) -> str:
    return json.dumps([[int(x), int(y)] for x, y in points], separators=(",", ":"))


def line_dict(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "page_id": r["page_id"],
        "index": r["index"],
        "baseline": json.loads(r["baseline"]),
        "polygon": json.loads(r["polygon"]),
        "text": r["text"],
        "text_source": r["text_source"],
        "confidence": r["confidence"],
        "is_ground_truth": bool(r["is_ground_truth"]),
        "ignored": bool(r["ignored"]),
        "crop_url": f"/lines/{r['id']}/crop.png",
    }


def page_dict(r: sqlite3.Row) -> dict:
    keys = r.keys()
    d = {
        "id": r["id"],
        "document_id": r["document_id"],
        "original_filename": r["original_filename"],
        "status": r["status"],
        "width": r["width"],
        "height": r["height"],
        "original_width": r["original_width"],
        "original_height": r["original_height"],
        "dpi": [r["dpi_x"], r["dpi_y"]],
        "line_height_px": r["line_height_px"],
        "scale": r["scale"],
        "skew_deg": r["skew_deg"],
        "deskewed": bool(r["deskewed"]),
        "warnings": json.loads(r["warnings"]),
        "sha256": r["sha256"],
        "imported_at": r["imported_at"],
        "segmented_at": r["segmented_at"],
        "image_url": f"/pages/{r['id']}/working",
        "original_url": f"/pages/{r['id']}/original",
    }
    if "line_count" in keys:
        d["line_count"] = r["line_count"]
        d["ground_truth_count"] = r["ground_truth_count"]
        d["ignored_count"] = r["ignored_count"]
    return d
