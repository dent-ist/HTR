import io

from PIL import Image

from tests.conftest import wait_job


def test_documents(client):
    assert client.get("/documents").json() == []
    r = client.post("/documents", json={"name": "  Clinic notes "})
    assert r.status_code == 201
    doc = r.json()
    assert doc["name"] == "Clinic notes" and doc["page_count"] == 0 and doc["imported_at"]
    assert client.get("/documents").json()[0]["id"] == doc["id"]


def test_document_validation(client):
    assert client.post("/documents", json={"name": "   "}).status_code == 422
    assert client.post("/documents", json={}).status_code == 422
    assert client.post("/documents", json={"name": "x", "extra": 1}).status_code == 422
    assert client.post("/documents", json={"name": "x" * 201}).status_code == 422


def test_upload_png_stores_original_untouched_and_working_copy(api, client, settings, scan_bytes):
    data = scan_bytes("PNG", seed=5)
    page = api.import_page(name="my scan.png", data=data)
    assert page["duplicate"] is False and page["status"] == "unsegmented"
    assert page["original_filename"] == "my scan.png"
    assert (page["original_width"], page["original_height"]) == (700, 900)
    assert page["line_count"] == 0 and page["ground_truth_count"] == 0
    stored = list((settings.data_dir / "originals").iterdir())
    assert len(stored) == 1 and stored[0].read_bytes() == data  # byte-identical
    working = list((settings.data_dir / "working").glob("*.png"))
    assert len(working) == 1
    with Image.open(working[0]) as im:
        assert im.size == (page["width"], page["height"]) and im.mode == "L"
    # import warnings from the M1 pipeline are recorded (no DPI tag, small image)
    assert any("missing DPI" in w for w in page["warnings"])
    assert page["image_url"] == f"/pages/{page['id']}/working"
    assert list((settings.data_dir / "tmp").iterdir()) == []


def test_upload_accepts_jpeg_and_tiff(api):
    jpg = api.import_page(name="a.jpg", fmt="JPEG")
    tif = api.import_page(name="b.TIFF", fmt="TIFF")
    assert jpg["original_filename"] == "a.jpg" and tif["original_filename"] == "b.TIFF"


def test_upload_same_content_is_deduplicated(api, client, scan_bytes, settings):
    doc = api.new_document()
    first = api.import_page(doc["id"], data=scan_bytes("PNG", 9))
    r = client.post("/pages", data={"document_id": doc["id"]}, files={"file": ("copy.png", scan_bytes("PNG", 9))})
    assert r.status_code == 200 and r.json()["duplicate"] is True and r.json()["id"] == first["id"]
    assert len(list((settings.data_dir / "originals").iterdir())) == 1


def test_upload_error_cases(client, api, settings, scan_bytes):
    doc = api.new_document()["id"]

    def post(files, document_id=doc):
        data = {} if document_id is None else {"document_id": str(document_id)}
        return client.post("/pages", data=data, files=files)

    assert post({"file": ("a.png", scan_bytes())}, document_id=9999).status_code == 404
    assert post({"file": ("a.png", scan_bytes())}, document_id=None).status_code == 422
    assert client.post("/pages", data={"document_id": str(doc)}).status_code == 422  # no file
    assert post({"file": ("notes.txt", b"hello")}).status_code == 415  # extension
    assert post({"file": ("noext", scan_bytes())}).status_code == 415
    assert post({"file": ("fake.png", b"this is not an image")}).status_code == 415  # content
    assert post({"file": ("lie.jpg", scan_bytes("PNG"))}).status_code == 415  # extension/content mismatch
    assert post({"file": ("empty.png", b"")}).status_code == 422
    assert post({"file": ("trunc.png", scan_bytes()[:200])}).status_code == 415
    # nothing left behind by any failure
    assert not (settings.data_dir / "originals").exists() or list((settings.data_dir / "originals").iterdir()) == []
    assert not (settings.data_dir / "working").exists() or list((settings.data_dir / "working").iterdir()) == []
    assert list(settings.tmp_dir.iterdir()) == []
    assert client.get("/pages").json() == []


def test_upload_size_limit(settings, scan_bytes, fake_blla):
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from app.main import create_app

    small = replace(settings, max_upload_bytes=1000)
    with TestClient(create_app(small)) as c:
        doc = c.post("/documents", json={"name": "d"}).json()
        r = c.post("/pages", data={"document_id": str(doc["id"])}, files={"file": ("a.png", scan_bytes())})
        assert r.status_code == 413
        assert list(small.tmp_dir.iterdir()) == []


def test_upload_path_in_filename_is_reduced_to_basename(api, settings):
    page = api.import_page(name="../../etc/passwd.png")
    assert page["original_filename"] == "passwd.png"
    assert all(settings.data_dir in p.parents for p in (settings.data_dir / "originals").iterdir())


def test_list_pages_status_counts_and_filters(api, client):
    doc_a, doc_b = api.new_document("a")["id"], api.new_document("b")["id"]
    raw = api.import_page(doc_a)
    seg = api.segmented_page(document_id=doc_a)
    other = api.segmented_page(document_id=doc_b)
    line = seg["lines"][0]
    client.patch(f"/lines/{line['id']}", json={"text": "hello", "is_ground_truth": True})
    client.patch(f"/lines/{seg['lines'][1]['id']}", json={"ignored": True})

    pages = {p["id"]: p for p in client.get("/pages").json()}
    assert pages[raw["id"]]["status"] == "unsegmented" and pages[raw["id"]]["line_count"] == 0
    p = pages[seg["id"]]
    assert p["status"] == "partially_transcribed"
    assert (p["line_count"], p["ground_truth_count"], p["ignored_count"]) == (3, 1, 1)
    assert pages[other["id"]]["status"] == "segmented"
    assert {x["id"] for x in client.get(f"/pages?document_id={doc_b}").json()} == {other["id"]}
    assert {x["id"] for x in client.get("/pages?status=unsegmented").json()} == {raw["id"]}
    assert client.get("/pages?status=bogus").status_code == 422


def test_get_page_detail_and_404(api, client):
    page = api.segmented_page()
    assert page["status"] == "segmented" and len(page["lines"]) == 3
    ln = page["lines"][0]
    assert ln["index"] == 0 and ln["text"] == "" and ln["text_source"] is None
    assert ln["is_ground_truth"] is False and ln["ignored"] is False
    assert len(ln["baseline"]) == 2 and len(ln["polygon"]) == 4
    assert ln["crop_url"] == f"/lines/{ln['id']}/crop.png"
    assert [x["index"] for x in page["lines"]] == [0, 1, 2]
    assert client.get("/pages/99999").status_code == 404
    assert client.get("/pages/abc").status_code == 422


def test_image_endpoints(api, client, scan_bytes):
    data = scan_bytes("PNG", seed=77)
    page = api.import_page(data=data)
    orig = client.get(page["original_url"])
    assert orig.status_code == 200 and orig.content == data and orig.headers["content-type"] == "image/png"
    work = client.get(page["image_url"])
    assert work.status_code == 200 and work.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(work.content)).size == (page["width"], page["height"])
    assert client.get("/pages/12345/original").status_code == 404
    assert client.get("/pages/12345/working").status_code == 404


def test_line_crop(api, client):
    page = api.segmented_page()
    ln = page["lines"][1]
    xs = [p[0] for p in ln["polygon"]]
    ys = [p[1] for p in ln["polygon"]]
    r = client.get(ln["crop_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    im = Image.open(io.BytesIO(r.content))
    assert im.size == (max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)
    assert client.get("/lines/99999/crop.png").status_code == 404


def test_segment_enqueues_job_and_creates_lines(api, client, settings):
    page = api.import_page()
    r = client.post(f"/pages/{page['id']}/segment")
    assert r.status_code == 202
    job = r.json()
    assert job["type"] == "segment" and job["payload"] == {"page_id": page["id"], "force": False}
    done = wait_job(client, job["id"])
    assert done["status"] == "succeeded" and done["progress"] == 1.0
    assert done["result"]["lines"] == 3 and done["started_at"] and done["finished_at"]
    got = client.get(f"/pages/{page['id']}").json()
    assert got["status"] == "segmented" and got["segmented_at"] and len(got["lines"]) == 3
    assert len(list((settings.data_dir / "pagexml").glob("*.xml"))) == 1


def test_segment_errors(api, client):
    assert client.post("/pages/999/segment").status_code == 404
    page = api.import_page()
    assert client.post(f"/pages/{page['id']}/segment", json={"bogus": 1}).status_code == 422


def test_segment_zero_lines_is_still_segmented(api, client, fake_blla):
    fake_blla(n_lines=0)
    page = api.segmented_page()
    assert page["status"] == "segmented" and page["lines"] == []


def test_segmentation_failure_marks_job_failed_and_keeps_page(api, client, monkeypatch):
    from app.pipeline import segment

    page = api.import_page()

    def boom(img, device):
        raise ValueError("simulated segmentation failure")

    monkeypatch.setattr(segment, "run_blla", boom)
    job = client.post(f"/pages/{page['id']}/segment").json()
    done = wait_job(client, job["id"])
    assert done["status"] == "failed" and "simulated segmentation failure" in done["error"]
    got = client.get(f"/pages/{page['id']}").json()
    assert got["status"] == "unsegmented" and got["lines"] == []


def test_resegment_is_refused_over_transcribed_lines_unless_forced(api, client):
    page = api.segmented_page()
    lid = page["lines"][0]["id"]
    client.patch(f"/lines/{lid}", json={"text": "keep me"})
    r = client.post(f"/pages/{page['id']}/segment")
    assert r.status_code == 409 and "transcribed" in r.json()["detail"]
    forced = client.post(f"/pages/{page['id']}/segment", json={"force": True})
    assert forced.status_code == 202
    assert wait_job(client, forced.json()["id"])["status"] == "succeeded"
    again = client.get(f"/pages/{page['id']}").json()
    assert len(again["lines"]) == 3 and all(x["text"] == "" for x in again["lines"])
    assert again["status"] == "segmented"


def test_resegment_untouched_page_needs_no_force(api, client):
    page = api.segmented_page()
    job = client.post(f"/pages/{page['id']}/segment").json()
    assert wait_job(client, job["id"])["status"] == "succeeded"


def test_segment_rejected_while_job_active(api, client):
    page = api.import_page()
    client.app.state.worker.stop()  # nothing runs, so the first job stays queued
    first = client.post(f"/pages/{page['id']}/segment")
    assert first.status_code == 202
    second = client.post(f"/pages/{page['id']}/segment")
    assert second.status_code == 409 and str(first.json()["id"]) in second.json()["detail"]
