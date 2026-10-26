"""Batch import and segmentation over a data directory (used by scripts/htr.py).

Layout under DATA:
  originals/<id><ext>   byte-identical copies of the source files (id = <stem>_<sha256[:8]>)
  working/<id>.png      normalized/deskewed/rescaled working copies
  pagexml/<id>.xml      PAGE-XML (coordinates are in working-copy pixels)
  overlays/<id>.png     inspection overlays
  manifest.json         one record per page: hashes, measurements, warnings, segmentation status
  import_report.json/.md
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from . import measure, preprocess

MANIFEST = "manifest.json"


class Workspace:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir)
        self.originals = self.root / "originals"
        self.working = self.root / "working"
        self.pagexml = self.root / "pagexml"
        self.overlays = self.root / "overlays"
        self.manifest_path = self.root / MANIFEST

    def load(self) -> dict[str, dict]:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def save(self, manifest: dict[str, dict]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.manifest_path)  # atomic, so an interrupted run never corrupts it


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "page"


def import_folder(
    folder: Path,
    ws: Workspace,
    target_line_height: float = preprocess.DEFAULT_TARGET_LINE_HEIGHT,
    recursive: bool = False,
    deskew: bool = True,
    log: Callable[[str], None] = print,
) -> dict:
    """Copy originals (skipping content already imported), derive working copies, write the import report."""
    manifest = ws.load()
    by_hash = {rec["sha256"]: pid for pid, rec in manifest.items()}
    stats = {"new": 0, "skipped": 0, "rederived": 0, "failed": 0}
    for src in measure.find_images(folder, recursive):
        try:
            digest = sha256_file(src)
            pid = by_hash.get(digest)
            if pid is None:
                pid = f"{_safe_stem(src.name)}_{digest[:8]}"
                ws.originals.mkdir(parents=True, exist_ok=True)
                dest = ws.originals / f"{pid}{src.suffix.lower()}"
                shutil.copy2(src, dest)  # bytes untouched
                rep = measure.inspect_image(src)
                if rep.error:
                    dest.unlink(missing_ok=True)
                    raise ValueError(rep.error)
                manifest[pid] = {
                    "id": pid,
                    "source_name": src.name,
                    "original": str(dest.relative_to(ws.root)),
                    "sha256": digest,
                    "width": rep.width,
                    "height": rep.height,
                    "dpi": [rep.dpi_x, rep.dpi_y],
                    "line_height_px": rep.line_height_px,
                    "line_pitch_px": rep.line_pitch_px,
                    "warnings": rep.warnings,
                    "segmentation": {"status": "pending"},
                }
                by_hash[digest] = pid
                stats["new"] += 1
                log(f"imported {src.name} -> {pid}")
                needs_working = True
            else:
                rec = manifest[pid]
                work = ws.root / rec.get("working", "")
                prev = rec.get("preprocess", {})
                needs_working = (
                    not rec.get("working") or not work.exists() or prev.get("target_line_height") != target_line_height
                )
                if needs_working:
                    stats["rederived"] += 1
                else:
                    stats["skipped"] += 1
                    log(f"skip {src.name} (already imported as {pid})")
            if needs_working:
                rec = manifest[pid]
                orig = ws.root / rec["original"]
                work = ws.working / f"{pid}.png"
                info = preprocess.preprocess_file(orig, work, target_line_height, deskew)
                rec["working"] = str(work.relative_to(ws.root))
                rec["preprocess"] = info.to_dict()
                rec["segmentation"] = {"status": "pending"}  # coordinates depend on the working copy
                for p in (ws.pagexml / f"{pid}.xml", ws.overlays / f"{pid}.png"):
                    p.unlink(missing_ok=True)
                notes = []
                if info.rescaled:
                    notes.append(f"working copy rescaled x{info.scale:g}")
                if info.deskewed:
                    notes.append(f"deskewed {info.skew_deg:g} deg")
                rec["notes"] = notes
                # One measurement for report and rescale decision: the normalized, deskewed image. (Measuring the
                # raw original gave band heights up to 35% larger on slightly skewed pages.)
                rep = measure.ImageReport(
                    path=rec["source_name"],
                    width=rec["width"],
                    height=rec["height"],
                    dpi_x=rec["dpi"][0],
                    dpi_y=rec["dpi"][1],
                    line_height_px=info.line_height_before,
                    line_pitch_px=info.line_pitch_before,
                    n_lines=info.n_lines,
                )
                rec["line_height_px"] = rep.line_height_px
                rec["line_pitch_px"] = rep.line_pitch_px
                rec["warnings"] = measure.image_warnings(rep)
        except Exception as e:  # noqa: BLE001  one bad file must not abort the batch
            stats["failed"] += 1
            log(f"FAILED {src.name}: {type(e).__name__}: {e}")
        ws.save(manifest)
    write_import_report(ws, manifest, stats)
    return stats


def write_import_report(ws: Workspace, manifest: dict[str, dict], stats: dict) -> None:
    pages = sorted(manifest.values(), key=lambda r: r["id"])
    flagged = [p for p in pages if p["warnings"]]
    (ws.root / "import_report.json").write_text(
        json.dumps({"last_run": stats, "pages": pages}, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# Import report",
        "",
        f"Pages: {len(pages)}; with warnings: {len(flagged)}. Last run: "
        + ", ".join(f"{k} {v}" for k, v in stats.items())
        + ".",
        "",
        "| page | source | size | dpi tag | line height px | working copy | warnings |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in pages:
        dpi = "none" if p["dpi"][0] is None else f"{p['dpi'][0]:g}"
        lh = "-" if p["line_height_px"] is None else f"{p['line_height_px']:.0f}"
        pre = p.get("preprocess", {})
        work = ", ".join(p.get("notes", [])) or "normalized"
        if pre:
            work += f" ({pre['width']}x{pre['height']}, line h {pre['line_height_after']})"
        warn = "; ".join(p["warnings"]) or ""
        size = f"{p['width']}x{p['height']}"
        lines.append(f"| {p['id']} | {p['source_name']} | {size} | {dpi} | {lh} | {work} | {warn} |")
    (ws.root / "import_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def segment_pages(
    ws: Workspace,
    device: str = "auto",
    force: bool = False,
    only: list[str] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Segment every pending/failed page (all pages with force). A failing page is recorded and skipped."""
    from . import overlay, pagexml, segment

    manifest = ws.load()
    stats = {"done": 0, "skipped": 0, "failed": 0}
    for pid, rec in sorted(manifest.items()):
        if only and pid not in only:
            continue
        st = rec.get("segmentation", {})
        xml = ws.pagexml / f"{pid}.xml"
        if not force and st.get("status") == "ok" and xml.exists():
            stats["skipped"] += 1
            log(f"skip {pid} (already segmented)")
            continue
        t0 = time.time()
        try:
            work = ws.root / rec["working"]
            with Image.open(work) as im:
                im.load()
                seg = segment.segment_image(im, device)
                pagexml.write_pagexml(seg, work.name, xml)
                overlay.write_overlay(im, seg, ws.overlays / f"{pid}.png")
            rec["segmentation"] = {
                "status": "ok",
                "lines": len(seg.lines),
                "device": seg.device,
                "seconds": round(time.time() - t0, 1),
                "pagexml": str(xml.relative_to(ws.root)),
                "overlay": f"overlays/{pid}.png",
            }
            stats["done"] += 1
            log(f"segmented {pid}: {len(seg.lines)} lines ({seg.device}, {time.time() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            rec["segmentation"] = {
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=5),
            }
            stats["failed"] += 1
            log(f"FAILED {pid}: {type(e).__name__}: {e}")
        ws.save(manifest)  # after every page: resumable
    return stats


__all__ = ["Workspace", "import_folder", "segment_pages", "sha256_file"]
