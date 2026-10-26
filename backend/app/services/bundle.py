"""Project bundle (`.htrbundle`): export and import of a whole project, see the README ("Project bundle").

Layout of the zip (every path is relative; the file entries mirror the data directory so the paths stored in the
database stay valid unchanged):

    manifest.json            last entry (its checksums are computed while the files stream out)
    db/htr.sqlite            consistent snapshot made with the sqlite3 backup API
    originals/...            the untouched scans (byte-identical)
    working/...              working copies
    pagexml/...              PAGE-XML, rendered from the snapshot so it matches the database exactly
    models/<id>.mlmodel      fine-tuned models
    models/base/<name>/...   the pretrained base models that model rows point at

Export never copies the live database file and never builds the zip in memory (`iter_bundle` is a generator; the CLI
and the HTTP route both consume it). Import verifies the whole bundle in a staging directory first and only then
swaps it into the project, with an undo log so a failure leaves the project as it was (see `import_bundle`).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import stat
import time
import uuid
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from ..config import APP_VERSION, Settings
from ..db import LATEST_VERSION, connect, migrate, now_iso
from ..db.migrations import current_version
from ..errors import Conflict, Invalid, NotFound
from . import abbreviations as abbreviations_svc
from . import export as export_svc
from . import models as models_svc

log = logging.getLogger(__name__)

FORMAT = "htr-bundle"
BUNDLE_FORMAT_VERSION = 1
MANIFEST = "manifest.json"
DB_ENTRY = "db/htr.sqlite"
TOP_LEVEL = frozenset({"manifest.json", "db", "originals", "working", "pagexml", "models"})
CHUNK = 1 << 20
MAX_ENTRIES = 200_000
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
ACTIVE_JOBS_MESSAGE = (
    "a {type} job is {status} (job {id}); wait for it to finish or cancel it, then try again "
    "(a bundle is only written when no segment, train or predict job is queued or running)"
)

# Tables whose row counts are recorded in the manifest and checked again on import.
COUNTED_TABLES = ("documents", "pages", "lines", "models", "abbreviations", "review_events", "jobs", "model_lines")
# Anything in these tables means the project is not empty (base-model rows registered at startup do not count).
EMPTY_CHECKS = [
    ("document", "documents", "SELECT COUNT(*) FROM documents"),
    ("page", "pages", "SELECT COUNT(*) FROM pages"),
    ("line", "lines", "SELECT COUNT(*) FROM lines"),
    ("fine-tuned model", "fine-tuned models", "SELECT COUNT(*) FROM models WHERE kind <> 'base'"),
    ("abbreviation", "abbreviations", "SELECT COUNT(*) FROM abbreviations"),
    ("review event", "review events", "SELECT COUNT(*) FROM review_events"),
    ("job", "jobs", "SELECT COUNT(*) FROM jobs"),
]


class BundleInvalid(Invalid):
    """The bundle is damaged, tampered with, unsafe or from a newer app (422)."""


def _fault(point: str) -> None:  # noqa: B027
    """Failure-injection hook for tests (monkeypatched); does nothing."""


# ---- shared helpers --------------------------------------------------------------------------------------------


def check_relative(path: str, what: str = "path", *, prefixes: frozenset[str] | None = None) -> PurePosixPath:
    """A safe bundle / database path: relative, forward slashes, no `..`, no empty or dot parts, no drive letter,
    and (optionally) under one of the allowed top-level names. Raises BundleInvalid."""
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise BundleInvalid(f"unsafe {what}: {path!r}")
    p = PurePosixPath(path)
    if p.is_absolute() or path.startswith("/") or len(path) > 1 and path[1] == ":":
        raise BundleInvalid(f"unsafe {what} (absolute path): {path!r}")
    if any(part in ("", ".", "..") for part in path.split("/")):
        raise BundleInvalid(f"unsafe {what} (relative components): {path!r}")
    if prefixes is not None and p.parts[0] not in prefixes:
        raise BundleInvalid(f"unexpected {what}: {path!r}")
    return p


def active_job(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT id, type, status FROM jobs WHERE status IN ('queued', 'running') ORDER BY id"
    ).fetchone()


def require_no_active_jobs(conn: sqlite3.Connection) -> None:
    j = active_job(conn)
    if j:
        raise Conflict(ACTIVE_JOBS_MESSAGE.format(type=j["type"], status=j["status"], id=j["id"]))


def non_empty_parts(conn: sqlite3.Connection) -> list[str]:
    """What makes the project non-empty, e.g. ['3 documents', '10 pages']; empty list when it is empty."""
    found = []
    for singular, plural, sql in EMPTY_CHECKS:
        n = conn.execute(sql).fetchone()[0]
        if n:
            found.append(f"{n} {singular if n == 1 else plural}")
    return found


def _table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    have = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    out = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in COUNTED_TABLES if t in have}
    out["ground_truth_lines"] = conn.execute("SELECT COUNT(*) FROM lines WHERE is_ground_truth = 1").fetchone()[0]
    out["ignored_lines"] = conn.execute("SELECT COUNT(*) FROM lines WHERE ignored = 1").fetchone()[0]
    out["segmented_pages"] = conn.execute("SELECT COUNT(*) FROM pages WHERE segmented_at IS NOT NULL").fetchone()[0]
    out["finetuned_models"] = conn.execute("SELECT COUNT(*) FROM models WHERE kind = 'finetuned'").fetchone()[0]
    return out


def _canonical(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def manifest_checksum(manifest: dict) -> str:
    return hashlib.sha256(_canonical({k: v for k, v in manifest.items() if k != "manifest_sha256"})).hexdigest()


# ---- export ----------------------------------------------------------------------------------------------------


@dataclass
class _Item:
    arc: str
    src: Path | None = None  # a file on disk, or...
    page_id: int | None = None  # ...a PAGE-XML rendered from the snapshot
    compress: bool = True
    size: int = 0


@dataclass
class ExportPlan:
    settings: Settings
    snapshot: Path
    conn: sqlite3.Connection
    items: list[_Item]
    counts: dict[str, int]
    schema_version: int
    missing_files: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.conn.close()
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(str(self.snapshot) + suffix).unlink(missing_ok=True)


_STORED_SUFFIXES = {".png", ".jpg", ".jpeg", ".mlmodel", ".zip"}


def _snapshot_db(settings: Settings, dest: Path) -> None:
    """Copy the live database with the sqlite3 backup API (one consistent point in time, WAL included)."""
    src = sqlite3.connect(settings.db_path, timeout=30)
    dst = sqlite3.connect(dest)
    try:
        src.backup(dst)
        dst.execute("PRAGMA journal_mode = DELETE")
        dst.commit()
    finally:
        dst.close()
        src.close()


def prepare_export(settings: Settings) -> ExportPlan:
    """Refuse while a job is queued or running, snapshot the database and list every file that goes in. Everything
    that can fail without having started a download (a job, a missing scan) fails here, before the first byte."""
    if not settings.db_path.is_file():
        raise NotFound("there is no project database yet")
    live = connect(settings.db_path)
    try:
        require_no_active_jobs(live)
        if current_version(live) == LATEST_VERSION:  # (a database that was never migrated is exported as it is)
            abbreviations_svc.refresh_counts(live)  # the cached occurrence counts travel with the bundle
    finally:
        live.close()
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    snap = settings.tmp_dir / f"snapshot-{uuid.uuid4().hex}.sqlite"
    plan = None
    try:
        _snapshot_db(settings, snap)
        conn = sqlite3.connect(snap, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        version = current_version(conn)
        plan = ExportPlan(settings, snap, conn, [], _table_counts(conn), version)
        _list_files(plan)
        return plan
    except BaseException:
        if plan:
            plan.close()
        else:
            snap.unlink(missing_ok=True)
        raise


def _stat_item(plan: ExportPlan, arc: str, *, essential: bool, compress: bool | None = None) -> None:
    check_relative(arc, "path in the database", prefixes=TOP_LEVEL - {"manifest.json", "db"})
    src = plan.settings.data_dir / arc
    if not src.is_file():
        if essential:
            raise Conflict(f"cannot write the bundle: the file {arc} is missing from the data directory")
        plan.missing_files.append(arc)
        return
    if any(i.arc == arc for i in plan.items):
        return
    stored = PurePosixPath(arc).suffix.lower() in _STORED_SUFFIXES
    plan.items.append(
        _Item(arc, src=src, compress=not stored if compress is None else compress, size=src.stat().st_size)
    )


def _list_files(plan: ExportPlan) -> None:
    c = plan.conn
    plan.items.append(_Item(DB_ENTRY, src=plan.snapshot, size=plan.snapshot.stat().st_size))
    pages = c.execute("SELECT id, image_path, working_path, xml_path, segmented_at FROM pages ORDER BY id").fetchall()
    for p in pages:
        if p["segmented_at"]:
            check_relative(p["xml_path"], "path in the database", prefixes=frozenset({"pagexml"}))
            plan.items.append(_Item(p["xml_path"], page_id=p["id"]))
    for p in pages:
        _stat_item(plan, p["image_path"], essential=True)
        _stat_item(plan, p["working_path"], essential=True)
    base_dirs: set[str] = set()
    for m in c.execute("SELECT path, kind FROM models ORDER BY id"):
        _stat_item(plan, m["path"], essential=False)
        if m["kind"] == "base":
            parent = str(PurePosixPath(m["path"]).parent)
            if parent.startswith(models_svc.BASE_DIR + "/") and parent not in base_dirs:
                base_dirs.add(parent)
                d = plan.settings.data_dir / parent
                for f in sorted(d.iterdir()) if d.is_dir() else []:
                    if f.is_file() and not f.is_symlink():
                        _stat_item(plan, f"{parent}/{f.name}", essential=False)
    plan.missing_files.sort()


def iter_bundle(plan: ExportPlan) -> Iterator[bytes]:
    """Yield the bundle zip in chunks. Writes the manifest last (it needs the checksums of the files before it) and
    cleans the snapshot up when finished or abandoned."""
    sink = _Sink()
    entries: list[dict] = []
    try:
        with zipfile.ZipFile(sink, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            when = time.localtime()[:6]
            for item in plan.items:
                h, size = hashlib.sha256(), 0
                zi = zipfile.ZipInfo(item.arc, date_time=when)
                zi.compress_type = zipfile.ZIP_DEFLATED if item.compress else zipfile.ZIP_STORED
                zi.external_attr = (stat.S_IFREG | 0o644) << 16
                if item.page_id is not None:
                    data = export_svc.pagexml_bytes(plan.conn, item.page_id, include_predicted=True)
                    h.update(data)
                    size = len(data)
                    zf.writestr(zi, data)
                    yield from sink.drain()
                else:
                    assert item.src is not None
                    zi.file_size = item.size
                    with open(item.src, "rb") as fin, zf.open(zi, "w", force_zip64=item.size >= 1 << 31) as out:
                        while chunk := fin.read(CHUNK):
                            h.update(chunk)
                            size += len(chunk)
                            out.write(chunk)
                            yield from sink.drain()
                    if size != item.size:
                        raise Conflict(f"{item.arc} changed while the bundle was being written; try again")
                entries.append({"path": item.arc, "size": size, "sha256": h.hexdigest()})
            manifest = {
                "format": FORMAT,
                "bundle_format_version": BUNDLE_FORMAT_VERSION,
                "schema_version": plan.schema_version,
                "app_version": APP_VERSION,
                "created_at": plan.created_at,
                "counts": {**plan.counts, "files": len(entries)},
                "missing_files": plan.missing_files,
                "files": entries,
            }
            manifest["manifest_sha256"] = manifest_checksum(manifest)
            zi = zipfile.ZipInfo(MANIFEST, date_time=when)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = (stat.S_IFREG | 0o644) << 16
            zf.writestr(zi, json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"))
        yield from sink.drain()
    finally:
        plan.close()


class _Sink:
    """An unseekable write target for ZipFile; `drain()` hands out what has been written so far."""

    def __init__(self) -> None:
        self._parts: list[bytes] = []

    def write(self, data: bytes) -> int:
        self._parts.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        pass

    def drain(self) -> Iterator[bytes]:
        parts, self._parts = self._parts, []
        data = b"".join(parts)
        if data:
            yield data


def export_to_path(settings: Settings, dest: Path) -> dict:
    """Write a bundle to `dest` (CLI). Written to `<dest>.part` and renamed, so a failure leaves no half file."""
    dest = Path(dest)
    if dest.exists():
        raise Conflict(f"{dest} already exists; choose another file name")
    plan = prepare_export(settings)
    part = dest.with_name(dest.name + ".part")
    gen = iter_bundle(plan)
    try:
        with open(part, "wb") as f:
            for chunk in gen:
                f.write(chunk)
        os.replace(part, dest)
    except BaseException:
        gen.close()
        part.unlink(missing_ok=True)
        raise
    return {
        "path": str(dest),
        "bytes": dest.stat().st_size,
        "counts": {**plan.counts},
        "schema_version": plan.schema_version,
        "missing_files": plan.missing_files,
    }


# ---- import ----------------------------------------------------------------------------------------------------


@dataclass
class _Verified:
    manifest: dict
    staging: Path
    files: list[dict]


def _read_manifest(zf: zipfile.ZipFile) -> dict:
    try:
        info = zf.getinfo(MANIFEST)
    except KeyError:
        raise BundleInvalid("this is not a project bundle: manifest.json is missing") from None
    if info.file_size > MAX_MANIFEST_BYTES:
        raise BundleInvalid("manifest.json is implausibly large")
    try:
        with zf.open(info) as f:
            raw = f.read(MAX_MANIFEST_BYTES + 1)
        manifest = json.loads(raw.decode("utf-8"))
    except (zipfile.BadZipFile, OSError, ValueError, EOFError) as e:
        raise BundleInvalid(f"manifest.json cannot be read: {e}") from None
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise BundleInvalid("this is not a project bundle (manifest.json has the wrong format marker)")
    return manifest


def _check_manifest(manifest: dict) -> list[dict]:
    fmt = manifest.get("bundle_format_version")
    if not isinstance(fmt, int) or fmt < 1:
        raise BundleInvalid("the manifest has no valid bundle_format_version")
    if fmt > BUNDLE_FORMAT_VERSION:
        raise BundleInvalid(
            f"this bundle uses format version {fmt}, which is newer than this app understands "
            f"({BUNDLE_FORMAT_VERSION}); update the app to import it"
        )
    sv = manifest.get("schema_version")
    if not isinstance(sv, int) or sv < 1:
        raise BundleInvalid("the manifest has no valid schema_version")
    if sv > LATEST_VERSION:
        raise BundleInvalid(
            f"this bundle was made by a newer version of the app (database schema {sv}; this app supports up to "
            f"{LATEST_VERSION}). Update the app to import it; nothing was changed"
        )
    if manifest.get("manifest_sha256") != manifest_checksum(manifest):
        raise BundleInvalid("the manifest does not match its own checksum: the bundle was modified or damaged")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise BundleInvalid("the manifest lists no files")
    seen = set()
    for f in files:
        if not isinstance(f, dict) or not isinstance(f.get("size"), int) or f["size"] < 0:
            raise BundleInvalid("the manifest has a malformed file entry")
        sha = f.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise BundleInvalid("the manifest has a malformed checksum")
        check_relative(f.get("path"), "path in the bundle", prefixes=TOP_LEVEL - {"manifest.json"})
        if f["path"].startswith("db/") and f["path"] != DB_ENTRY:
            raise BundleInvalid(f"unexpected path in the bundle: {f['path']!r}")
        if f["path"] in seen:
            raise BundleInvalid(f"the manifest lists {f['path']} twice")
        seen.add(f["path"])
    if DB_ENTRY not in seen:
        raise BundleInvalid("the bundle has no database snapshot")
    mf = manifest.get("missing_files", [])
    if not isinstance(mf, list):
        raise BundleInvalid("the manifest's missing_files is malformed")
    for p in mf:
        check_relative(p, "path in missing_files", prefixes=frozenset({"models"}))
    return files


def _check_zip_entries(zf: zipfile.ZipFile, files: list[dict]) -> None:
    infos = zf.infolist()
    if len(infos) > MAX_ENTRIES:
        raise BundleInvalid("the bundle has too many entries")
    listed = {f["path"]: f for f in files}
    names = set()
    for zi in infos:
        name = zi.filename
        if zi.is_dir():
            raise BundleInvalid(f"unexpected directory entry in the bundle: {name!r}")
        check_relative(name, "path in the bundle", prefixes=TOP_LEVEL)
        mode = zi.external_attr >> 16
        if mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG):
            raise BundleInvalid(f"the bundle contains a link or special file: {name!r}")
        if zi.flag_bits & 0x1:
            raise BundleInvalid("the bundle is encrypted")
        if name in names:
            raise BundleInvalid(f"the bundle contains {name!r} twice")
        names.add(name)
        if name == MANIFEST:
            continue
        spec = listed.get(name)
        if spec is None:
            raise BundleInvalid(f"the bundle contains {name!r}, which the manifest does not list")
        if zi.file_size != spec["size"]:
            raise BundleInvalid(
                f"{name}: size in the bundle ({zi.file_size}) differs from the manifest ({spec['size']})"
            )
    missing = sorted(set(listed) - names)
    if missing:
        raise BundleInvalid(
            f"the bundle is missing {missing[0]!r} (and {len(missing) - 1} more) listed in its manifest"
        )


def _extract_verified(zf: zipfile.ZipFile, files: list[dict], staging: Path) -> None:
    """Read every entry exactly once into the staging directory, verifying size and SHA-256 as it goes."""
    for spec in files:
        arc = spec["path"]
        dest = staging / arc
        dest.parent.mkdir(parents=True, exist_ok=True)
        h, size = hashlib.sha256(), 0
        try:
            with zf.open(arc) as src, open(dest, "wb") as out:
                while chunk := src.read(CHUNK):
                    size += len(chunk)
                    if size > spec["size"]:
                        raise BundleInvalid(f"{arc}: more data than the manifest declares")
                    h.update(chunk)
                    out.write(chunk)
        except (zipfile.BadZipFile, OSError, EOFError, zipfile.LargeZipFile) as e:
            if isinstance(e, OSError) and e.errno == 28:
                raise Conflict("not enough disk space to import this bundle") from None
            raise BundleInvalid(f"{arc}: cannot be read ({e}); the bundle is damaged or truncated") from None
        if size != spec["size"] or h.hexdigest() != spec["sha256"]:
            raise BundleInvalid(f"{arc}: checksum mismatch; the bundle is damaged or was modified")
        _fault(f"extracted:{arc}")


def _inspect_db(manifest: dict, staging: Path, files: list[dict]) -> dict:
    """Open the staged database, migrate it to the current schema if it is older, and check it against the manifest
    and the staged files. Leaves a single-file database (no WAL). Returns {schema_from, counts}."""
    db = staging / DB_ENTRY
    conn = sqlite3.connect(db, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise BundleInvalid("the database in the bundle is corrupt")
        objects = conn.execute("SELECT type, name FROM sqlite_master").fetchall()
        bad = [o["name"] for o in objects if o["type"] not in ("table", "index")]
        if bad:
            raise BundleInvalid(f"the database in the bundle contains unexpected objects ({', '.join(bad[:3])})")
        version = current_version(conn)
        if version != manifest["schema_version"]:
            raise BundleInvalid(
                f"the manifest says schema {manifest['schema_version']} but the database is at schema {version}"
            )
        conn.execute("PRAGMA foreign_keys = ON")
        schema_from = version
        if version < LATEST_VERSION:
            migrate(conn)  # the same runner the app uses at startup; each step is transactional
        counts = _table_counts(conn)
        for key, want in manifest.get("counts", {}).items():
            if key == "files":
                if want != len(files):
                    raise BundleInvalid("the manifest's file count does not match its file list")
            elif key in counts and counts[key] != want:
                raise BundleInvalid(f"the database has {counts[key]} {key} but the manifest says {want}")
        if conn.execute("PRAGMA foreign_key_check").fetchone():
            raise BundleInvalid("the database in the bundle violates its own foreign keys")
        _check_db_paths(conn, manifest, staging)
        conn.execute("PRAGMA journal_mode = DELETE")
        return {"schema_from": schema_from, "counts": counts}
    except sqlite3.DatabaseError as e:
        raise BundleInvalid(f"the database in the bundle cannot be read: {e}") from None
    finally:
        conn.close()


def _check_db_paths(conn: sqlite3.Connection, manifest: dict, staging: Path) -> None:
    """Every file path stored in the database is relative, stays inside its folder and exists in the bundle."""
    missing_ok = set(manifest.get("missing_files", []))

    def need(path: str, folder: str, *, optional: bool = False) -> None:
        check_relative(path, "path in the database", prefixes=frozenset({folder}))
        if not (staging / path).is_file() and not (optional and path in missing_ok):
            raise BundleInvalid(f"the database refers to {path}, which is not in the bundle")

    for r in conn.execute("SELECT image_path, working_path, xml_path, segmented_at FROM pages"):
        need(r["image_path"], "originals")
        need(r["working_path"], "working")
        check_relative(r["xml_path"], "path in the database", prefixes=frozenset({"pagexml"}))
        if r["segmented_at"]:
            need(r["xml_path"], "pagexml")
    for r in conn.execute("SELECT path FROM models"):
        need(r["path"], "models", optional=True)


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _plan_placement(settings: Settings, files: list[dict]) -> list[str]:
    """Which staged files must be moved into the data directory. A file that is already there with identical content
    is left alone; one with different content is a conflict (nothing of the project's is ever overwritten)."""
    place = []
    for spec in files:
        arc = spec["path"]
        if arc == DB_ENTRY:
            continue
        dest = settings.data_dir / arc
        if dest.exists():
            if dest.is_file() and dest.stat().st_size == spec["size"] and _file_sha(dest) == spec["sha256"]:
                continue
            raise Conflict(
                f"the data directory already has a different file at {arc}; refusing to overwrite it "
                "(import into an empty data directory)"
            )
        place.append(arc)
    return place


def _disk_check(path: Path, needed: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < needed * 1.1 + (64 << 20):
        raise Conflict(
            f"not enough free disk space to import: about {needed >> 20} MB needed in the data directory, "
            f"{free >> 20} MB free"
        )


def validate_bundle(source: str | os.PathLike | BinaryIO, staging: Path, settings: Settings) -> _Verified:
    """Open, validate and verify a bundle into `staging` (a new directory). Raises BundleInvalid / Conflict; touches
    nothing outside `staging`."""
    try:
        zf = zipfile.ZipFile(source)
    except (zipfile.BadZipFile, OSError, EOFError) as e:
        raise BundleInvalid(
            f"this is not a readable project bundle (damaged, truncated or not a zip file): {e}"
        ) from None
    with zf:
        manifest = _read_manifest(zf)
        files = _check_manifest(manifest)
        _check_zip_entries(zf, files)
        _disk_check(settings.data_dir, sum(f["size"] for f in files) * 2)
        staging.mkdir(parents=True)
        _extract_verified(zf, files, staging)
    return _Verified(manifest, staging, files)


def import_bundle(settings: Settings, source: str | os.PathLike | BinaryIO) -> dict:
    """Restore a bundle into an EMPTY project (never merges). Steps: refuse unless the project is empty; verify the
    bundle completely in a staging directory inside the data directory; migrate its database if it is older; check
    everything against the manifest; then move the files in and swap the database file last, with an undo log. Any
    failure removes what was added and puts the old database back, so the project is exactly as it was. The caller is
    responsible for there being no running job and no other writer (the HTTP route and the CLI do that)."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(settings.db_path)
    try:
        migrate(conn)
        require_no_active_jobs(conn)
        used = non_empty_parts(conn)
    finally:
        conn.close()
    if used:
        raise Conflict(
            "the project is not empty (" + ", ".join(used) + "); a bundle can only be imported into an empty project "
            "(v1 never merges). Use a fresh data directory"
        )
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:12]
    staging = settings.tmp_dir / f"import-{token}"
    incoming = settings.db_path.with_name(f".incoming-{token}.sqlite")
    backup = settings.db_path.with_name(f".pre-import-{token}")
    try:
        verified = validate_bundle(source, staging, settings)
        _fault("validated")
        info = _inspect_db(verified.manifest, staging, verified.files)
        to_place = _plan_placement(settings, verified.files)
        shutil.copyfile(staging / DB_ENTRY, incoming)  # next to the live database: the final step is a rename
        _fault("prepared")
        _swap(settings, staging, to_place, incoming, backup)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        incoming.unlink(missing_ok=True)
    shutil.rmtree(backup, ignore_errors=True)
    conn = connect(settings.db_path)
    try:
        migrate(conn)
        models_svc.register_base_models(conn, settings)
        counts = _table_counts(conn)
    finally:
        conn.close()
    m = verified.manifest
    return {
        "imported": True,
        "counts": counts,
        "schema_version": LATEST_VERSION,
        "migrated_from": info["schema_from"] if info["schema_from"] < LATEST_VERSION else None,
        "bundle_created_at": m.get("created_at"),
        "bundle_app_version": m.get("app_version"),
        "files": len(to_place),
        "missing_files": m.get("missing_files", []),
    }


def _swap(settings: Settings, staging: Path, to_place: list[str], incoming: Path, backup: Path) -> None:
    """Move the staged files into the data directory, then replace the database. Undone in reverse on any error."""
    undo: list[Callable[[], None]] = []
    created_dirs: list[Path] = []
    try:
        for i, arc in enumerate(to_place):
            dest = settings.data_dir / arc
            parent = dest.parent
            missing = []
            while not parent.exists():
                missing.append(parent)
                parent = parent.parent
            dest.parent.mkdir(parents=True, exist_ok=True)
            created_dirs.extend(reversed(missing))
            os.replace(staging / arc, dest)
            undo.append(lambda d=dest: d.unlink(missing_ok=True))
            _fault(f"placed:{i}")
        _fault("before-db")
        backup.mkdir()
        db = settings.db_path
        for suffix in ("", "-wal", "-shm"):
            old = Path(str(db) + suffix)
            if old.exists():
                kept = backup / old.name
                os.replace(old, kept)
                undo.append(lambda o=old, k=kept: os.replace(k, o))
        _fault("db-moved")
        os.replace(incoming, db)
        undo.append(lambda: os.replace(db, incoming))
        _fault("db-swapped")
    except BaseException:
        for fn in reversed(undo):
            try:
                fn()
            except Exception:  # noqa: BLE001  keep undoing the rest
                log.exception("could not undo a step of a failed import")
        for d in reversed(created_dirs):
            with contextlib.suppress(OSError):
                d.rmdir()  # only if empty
        shutil.rmtree(backup, ignore_errors=True)
        raise
