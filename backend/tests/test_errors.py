"""M7: one consistent error format (code + human message + hint), friendly handling of the failures a real project
runs into (damaged images, missing files, locked or full database, no GPU), and no raw stack traces for users."""

import errno
import io
import sqlite3

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import failures
from app.config import Settings
from app.errors import Conflict, Invalid, NotFound
from app.main import create_app
from tests.conftest import wait_job

SHAPE = {"detail", "code", "message", "hint"}


def assert_standard_error(r, status, code=None):
    assert r.status_code == status, r.text
    body = r.json()
    assert SHAPE <= body.keys(), body
    assert isinstance(body["code"], str) and body["code"]
    assert isinstance(body["message"], str) and body["message"]
    assert body["detail"] == body["message"] or isinstance(body["detail"], list)  # 422 keeps the field list in detail
    assert "Traceback" not in r.text and 'File "' not in r.text
    if code:
        assert body["code"] == code, body
    return body


# ---- the format itself -------------------------------------------------------------------------------------------


def test_every_kind_of_error_has_the_same_shape(client, api):
    page = api.segmented_page()
    assert_standard_error(client.get("/pages/999"), 404, "not_found")
    assert_standard_error(client.get("/no/such/route"), 404, "not_found")
    assert_standard_error(client.delete("/health"), 405, "method_not_allowed")
    assert_standard_error(client.post("/documents", json={}), 422, "validation_error")
    assert_standard_error(client.post("/documents", json={"name": "x", "extra": 1}), 422, "validation_error")
    assert_standard_error(client.patch(f"/lines/{page['lines'][0]['id']}", json={"text": "a\nb"}), 422)
    assert_standard_error(client.post("/abbreviations", json={"written": "a b", "expansion": "x"}), 422)
    assert_standard_error(client.post("/jobs/999/cancel"), 404)
    done = client.post(f"/pages/{page['id']}/segment").json()
    wait_job(client, done["id"])
    # a transcribed page needs force: 409
    lid = page["lines"][0]["id"]
    client.patch(f"/lines/{lid}", json={"text": "typed", "is_ground_truth": True})
    body = assert_standard_error(client.post(f"/pages/{page['id']}/segment"), 409)
    assert body["message"] and body["hint"] is None or isinstance(body["hint"], str)


def test_validation_errors_are_readable_and_keep_the_field_list(client):
    r = client.post("/documents", json={"name": ""})
    body = assert_standard_error(r, 422, "validation_error")
    assert isinstance(body["detail"], list) and body["detail"][0]["loc"][-1] == "name"
    assert "name" in body["message"] and body["hint"]


def test_error_codes_and_hints_are_set_where_users_need_them(client, api, fake_ml):
    body = assert_standard_error(client.post("/jobs/train", json={}), 422, "no_ground_truth")
    assert body["hint"]
    body = assert_standard_error(client.get("/pages/1/working"), 404)
    assert body["code"] == "not_found"
    doc = client.post("/documents", json={"name": "d"}).json()
    r = client.post("/pages", data={"document_id": str(doc["id"])}, files={"file": ("x.gif", b"GIF89a")})
    body = assert_standard_error(r, 415, "unsupported_file_type")
    assert "PNG" in body["hint"]


def test_unexpected_exceptions_become_a_friendly_500_without_text_or_trace(client, monkeypatch):
    from app.services import pages as pages_svc

    def boom(conn):
        raise RuntimeError("secret transcription ZEBRA-SECRET")

    monkeypatch.setattr(pages_svc, "list_documents", boom)
    r = client.get("/documents")
    body = assert_standard_error(r, 500, "internal_error")
    assert "ZEBRA-SECRET" not in r.text and "RuntimeError" in body["message"]  # the class name helps, the text leaks
    assert "htr.log" in body["hint"]


def test_database_busy_is_a_retryable_503(client, monkeypatch):
    from app.services import pages as pages_svc

    def locked(conn):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(pages_svc, "list_documents", locked)
    r = client.get("/documents")
    body = assert_standard_error(r, 503, "db_locked")
    assert r.headers["retry-after"] and "try again" in body["hint"].lower()


@pytest.mark.parametrize(
    "exc",
    [sqlite3.OperationalError("database or disk is full"), OSError(errno.ENOSPC, "No space left on device")],
)
def test_disk_full_is_a_507_with_advice(client, monkeypatch, exc):
    from app.services import pages as pages_svc

    def full(conn, name):
        raise exc

    monkeypatch.setattr(pages_svc, "create_document", full)
    body = assert_standard_error(client.post("/documents", json={"name": "d"}), 507, "disk_full")
    assert "space" in body["hint"].lower()


def test_damaged_database_gets_restore_advice(client, monkeypatch):
    from app.services import pages as pages_svc

    def damaged(conn):
        raise sqlite3.DatabaseError("database disk image is malformed")

    monkeypatch.setattr(pages_svc, "list_documents", damaged)
    body = assert_standard_error(client.get("/documents"), 500, "db_corrupt")
    assert "bundle" in body["hint"]


def test_classify_unit():
    c = failures.classify
    assert c(NotFound("x", "nf", "h")) == failures.Failure(404, "x", "nf", "h")
    assert c(Conflict("y")).code == "conflict" and c(Invalid("z")).status == 422
    assert c(RuntimeError("CUDA out of memory. Tried to allocate 20 MiB")).code == "cuda_oom"
    assert "batch size" in c(RuntimeError("CUDA out of memory.")).hint
    assert c(FileNotFoundError("/x")).code == "file_missing" and "/x" not in c(FileNotFoundError("/x")).message
    assert c(PermissionError("p")).code == "permission_denied"
    assert c(MemoryError()).code == "out_of_memory"
    job = c(ValueError("bad value"), job=True)
    assert (job.code, job.message) == ("job_failed", "ValueError: bad value")
    api = c(ValueError("secret"), job=False)
    assert "secret" not in api.message and api.code == "internal_error"


# ---- damaged or unsupported images -------------------------------------------------------------------------------


def png_bytes(size=(120, 160), mode="L", fmt="PNG", **kw):
    buf = io.BytesIO()
    Image.new(mode, size, 255).save(buf, format=fmt, **kw)
    return buf.getvalue()


def upload(client, name, data):
    doc = client.get("/documents").json()
    doc_id = doc[0]["id"] if doc else client.post("/documents", json={"name": "d"}).json()["id"]
    return client.post("/pages", data={"document_id": str(doc_id)}, files={"file": (name, data)})


def test_corrupt_and_unsupported_uploads_are_refused_with_advice(client):
    good = png_bytes()
    cases = [
        ("garbage.png", b"this is not an image at all" * 20, 415, "image_unreadable"),
        ("cut_off.png", good[: len(good) // 2], 415, "image_unreadable"),
        ("wrong_ext.png", png_bytes(fmt="JPEG"), 415, "image_type_mismatch"),
        ("scan.gif", png_bytes(fmt="GIF"), 415, "unsupported_file_type"),
        ("scan.pdf", b"%PDF-1.4", 415, "unsupported_file_type"),
        ("noext", good, 415, "unsupported_file_type"),
        ("empty.png", b"", 422, "empty_upload"),
    ]
    for name, data, status, code in cases:
        body = assert_standard_error(upload(client, name, data), status, code)
        assert body["hint"], name
    assert client.get("/pages").json() == []  # nothing half-imported
    originals = client.app.state.settings.data_dir / "originals"
    assert not originals.exists() or not list(originals.glob("*"))  # no leftovers on disk either


def test_truncated_jpeg_that_passes_verify_is_still_refused(client):
    buf = io.BytesIO()
    Image.effect_noise((400, 500), 80).save(buf, format="JPEG", quality=90)
    data = buf.getvalue()
    body = assert_standard_error(upload(client, "cut.jpg", data[: int(len(data) * 0.6)]), 415, "image_unreadable")
    assert body["hint"]
    assert client.get("/pages").json() == []


def test_multi_page_tiff_is_refused_not_silently_truncated(client):
    """Decision: only single-page images are imported. Silently keeping page 1 of a scanned document would lose the
    rest without a word, so the user is told to split the file."""
    buf = io.BytesIO()
    pages = [Image.new("L", (200, 300), v) for v in (255, 200, 150)]
    pages[0].save(buf, format="TIFF", save_all=True, append_images=pages[1:])
    body = assert_standard_error(upload(client, "notebook.tif", buf.getvalue()), 415, "multipage_image")
    assert "3 pages" in body["message"] and "one image per page" in body["hint"]
    single = io.BytesIO()
    pages[0].save(single, format="TIFF")
    assert upload(client, "page.tif", single.getvalue()).status_code == 201  # a normal TIFF is fine


def test_image_size_limits(settings, fake_blla):
    small = Settings.for_dir(settings.data_dir, worker_poll_seconds=0.02, device="cpu", max_image_pixels=50_000)
    with TestClient(create_app(small)) as c:
        ok = upload(c, "ok.png", png_bytes((200, 200)))  # 40,000 pixels
        assert ok.status_code == 201
        body = assert_standard_error(upload(c, "big.png", png_bytes((300, 300))), 413, "image_too_many_pixels")
        assert "HTR_MAX_IMAGE_MP" in body["hint"]
    tiny_upload = Settings.for_dir(settings.data_dir, worker_poll_seconds=0.02, device="cpu", max_upload_bytes=1000)
    with TestClient(create_app(tiny_upload)) as c:
        body = assert_standard_error(upload(c, "x.png", Image.effect_noise((200, 200), 90).tobytes() * 2), 413)
        assert body["code"] == "upload_too_large" and "HTR_MAX_UPLOAD_MB" in body["hint"]


def test_default_pixel_limit_and_decompression_bombs(client):
    """A 100 MP image (over the 80 MP default) and a 196 MP one (over Pillow's own bomb limit) are both refused as
    too large, before anything is decoded into memory."""
    for side in (10_000, 14_000):
        data = png_bytes((side, side), mode="1")
        body = assert_standard_error(upload(client, f"huge{side}.png", data), 413, "image_too_many_pixels")
        assert "megapixels" in body["message"]
    assert client.get("/pages").json() == []


# ---- missing files and failing jobs ------------------------------------------------------------------------------


def test_missing_files_on_disk_give_404_with_a_restore_hint(client, api):
    page = api.segmented_page()
    data = client.app.state.settings.data_dir
    row = (
        sqlite3.connect(client.app.state.settings.db_path)
        .execute("SELECT image_path, working_path FROM pages WHERE id = ?", (page["id"],))
        .fetchone()
    )
    (data / row[1]).unlink()
    (data / row[0]).unlink()
    for url in (f"/pages/{page['id']}/working", f"/pages/{page['id']}/original", page["lines"][0]["crop_url"]):
        body = assert_standard_error(client.get(url), 404, "file_missing")
        assert "bundle" in body["hint"] and "data" not in body["message"].split("missing")[0]


def test_segmenting_a_page_whose_image_is_gone_or_damaged_fails_with_a_clear_job_error(client, api, fake_blla):
    page = api.import_page()
    data = client.app.state.settings.data_dir
    working = next((data / "working").glob("*"))
    working.write_bytes(b"not a png any more")
    job = wait_job(client, client.post(f"/pages/{page['id']}/segment").json()["id"])
    assert job["status"] == "failed" and job["error_code"] == "image_unreadable"
    assert "cannot be read" in job["error"] and job["error_hint"] and "Traceback" not in job["error"]
    working.unlink()
    job = wait_job(client, client.post(f"/pages/{page['id']}/segment").json()["id"])
    assert job["status"] == "failed" and job["error_code"] == "file_missing" and "bundle" in job["error_hint"]


def test_unexpected_job_errors_keep_their_text_and_point_to_the_log(client, api, fake_blla):
    page = api.import_page()
    fake_blla(n_lines=3, fail_on=page["width"])
    job = wait_job(client, client.post(f"/pages/{page['id']}/segment").json()["id"])
    assert job["status"] == "failed" and job["error"] == "ValueError: simulated segmentation failure"
    assert job["error_code"] == "job_failed" and "htr.log" in job["error_hint"]


def test_cuda_out_of_memory_in_a_job_suggests_a_smaller_batch(client, api, fake_blla):
    from app.pipeline import segment

    def oom(img, device):
        raise RuntimeError("CUDA error: out of memory (cublas)")

    page = api.import_page()
    segment_run = segment.run_blla
    segment.run_blla = oom
    try:
        job = wait_job(client, client.post(f"/pages/{page['id']}/segment").json()["id"])
    finally:
        segment.run_blla = segment_run
    assert job["status"] == "failed" and job["error_code"] == "cuda_oom" and "batch size" in job["error_hint"]


def test_interrupted_jobs_are_marked_with_a_code(settings):
    from app.db import connect, migrate
    from app.worker import queue

    conn = connect(settings.db_path)
    migrate(conn)
    queue.enqueue(conn, "segment", {"page_id": 1})
    queue.claim_next(conn)
    assert queue.fail_interrupted(conn) == 1
    job = queue.list_jobs(conn)[0]
    assert job["error_code"] == "interrupted" and job["error_hint"]


def test_missing_gpu_falls_back_to_the_cpu_with_a_visible_notice(client, api, monkeypatch, fake_ml):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    page = api.segmented_page()
    for ln in page["lines"][:2]:  # the third line is left for the prediction
        client.patch(f"/lines/{ln['id']}", json={"text": "hello world", "is_ground_truth": True})
    job = wait_job(client, client.post("/jobs/predict", json={"device": "cuda:0"}).json()["id"])
    assert job["status"] == "succeeded", job["error"]
    assert any("No usable GPU" in n for n in job["result"]["notices"])
    assert job["result"]["device"] == "cpu"
    job = wait_job(client, client.post("/jobs/train", json={"epochs": 1, "device": "auto"}).json()["id"])
    assert job["status"] == "succeeded", job["error"]
    assert any("CPU" in n for n in job["result"]["notices"])
