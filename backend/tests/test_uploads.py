"""Uploads and edits from several requests at once, hostile file names and large (but allowed) images."""

import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from app.db import connect


def count_files(settings, folder):
    d = settings.data_dir / folder
    return len(list(d.glob("*"))) if d.exists() else 0


def test_concurrent_uploads_of_different_scans_all_arrive(client, scan_bytes, settings):
    doc = client.post("/documents", json={"name": "bulk"}).json()
    blobs = [scan_bytes("PNG", seed=100 + i) for i in range(8)]

    def send(i):
        return client.post("/pages", data={"document_id": str(doc["id"])}, files={"file": (f"p{i}.png", blobs[i])})

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(send, range(8)))
    assert [r.status_code for r in results] == [201] * 8, [r.text for r in results if r.status_code != 201]
    ids = {r.json()["id"] for r in results}
    assert len(ids) == 8
    pages = client.get("/pages").json()
    assert len(pages) == 8 and {p["original_filename"] for p in pages} == {f"p{i}.png" for i in range(8)}
    assert count_files(settings, "originals") == 8 and count_files(settings, "working") == 8
    assert count_files(settings, "tmp") == 0 or not list((settings.data_dir / "tmp").glob("upload_*"))  # temp removed
    for p in pages:  # every page is really served
        assert client.get(p["image_url"]).status_code == 200


def test_concurrent_uploads_of_the_same_scan_create_one_page(client, scan_bytes, settings):
    doc = client.post("/documents", json={"name": "dupes"}).json()
    blob = scan_bytes("PNG", seed=7)
    barrier = threading.Barrier(6)

    def send(_):
        barrier.wait()
        return client.post("/pages", data={"document_id": str(doc["id"])}, files={"file": ("same.png", blob)})

    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(send, range(6)))
    codes = sorted(r.status_code for r in results)
    assert set(codes) <= {200, 201} and codes.count(201) == 1, codes
    assert len({r.json()["id"] for r in results}) == 1
    assert len(client.get("/pages").json()) == 1
    assert count_files(settings, "originals") == 1 and count_files(settings, "working") == 1  # no orphans


def test_concurrent_edits_to_different_lines_all_persist_in_db_and_xml(client, api, settings):
    page = api.segmented_page()
    lines = page["lines"]
    texts = {ln["id"]: f"concurrent line {i}" for i, ln in enumerate(lines)}

    def edit(lid):
        return client.patch(f"/lines/{lid}", json={"text": texts[lid], "is_ground_truth": True})

    with ThreadPoolExecutor(len(lines)) as pool:
        results = list(pool.map(edit, texts))
    assert [r.status_code for r in results] == [200] * len(lines)
    now = {ln["id"]: ln["text"] for ln in client.get(f"/pages/{page['id']}").json()["lines"]}
    assert now == texts
    xml = (settings.data_dir / "pagexml" / next(iter((settings.data_dir / "pagexml").glob("*.xml"))).name).read_text()
    assert all(t in xml for t in texts.values())  # the PAGE-XML on disk matches the database
    conn = connect(settings.db_path)
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    conn.close()


def test_hostile_file_names_are_reduced_to_a_plain_name(client):
    doc = client.post("/documents", json={"name": "d"}).json()
    img = io.BytesIO()
    Image.new("L", (200, 300), 255).save(img, format="PNG")
    r = client.post(
        "/pages", data={"document_id": str(doc["id"])}, files={"file": ("../../../etc/pass wd.png", img.getvalue())}
    )
    assert r.status_code == 201
    assert r.json()["original_filename"] == "pass wd.png"
    row = connect(client.app.state.settings.db_path).execute("SELECT image_path, working_path FROM pages").fetchone()
    for key, folder in (("image_path", "originals/"), ("working_path", "working/")):
        assert row[key].startswith(folder) and "/" not in row[key][len(folder) :] and "etc" not in row[key]
        assert (client.app.state.settings.data_dir / row[key]).is_file()


def test_a_large_but_allowed_scan_imports(client):
    """About 20 megapixels (A4 at 600 DPI) is within the default limit and must import and be served."""
    doc = client.post("/documents", json={"name": "big"}).json()
    img = Image.new("L", (4000, 5000), 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    r = client.post("/pages", data={"document_id": str(doc["id"])}, files={"file": ("a4_600dpi.png", buf.getvalue())})
    assert r.status_code == 201, r.text
    page = r.json()
    assert page["original_width"] == 4000 and page["width"] * page["height"] <= 4000 * 5000
    assert client.get(page["image_url"]).status_code == 200
    json.dumps(page)  # serialisable
