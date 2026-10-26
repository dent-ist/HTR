import threading
import time

from fastapi.testclient import TestClient

from app.db import connect, migrate
from app.main import create_app
from app.worker import queue
from app.worker.runner import Worker
from tests.conftest import wait_job


def make_db(settings):
    conn = connect(settings.db_path)
    migrate(conn)
    return conn


# ---- queue -------------------------------------------------------------------------------------------------------


def test_queue_lifecycle(settings):
    conn = make_db(settings)
    j = queue.enqueue(conn, "segment", {"page_id": 1})
    assert j["status"] == "queued" and j["progress"] == 0 and j["payload"] == {"page_id": 1}
    assert j["created_at"] and j["started_at"] is None and j["error"] is None
    claimed = queue.claim_next(conn)
    assert claimed["id"] == j["id"] and claimed["status"] == "running" and claimed["started_at"]
    assert queue.claim_next(conn) is None  # nothing else queued
    queue.set_progress(conn, j["id"], 0.4)
    assert queue.get_job(conn, j["id"])["progress"] == 0.4
    queue.set_progress(conn, j["id"], 7)  # clamped
    assert queue.get_job(conn, j["id"])["progress"] == 1.0
    queue.finish(conn, j["id"], {"lines": 3})
    done = queue.get_job(conn, j["id"])
    assert done["status"] == "succeeded" and done["progress"] == 1 and done["result"] == {"lines": 3}
    assert done["finished_at"]


def test_queue_is_fifo_and_claims_once(settings):
    conn = make_db(settings)
    ids = [queue.enqueue(conn, "segment", {"n": i})["id"] for i in range(3)]
    assert [queue.claim_next(conn)["id"] for _ in range(3)] == ids
    assert queue.claim_next(conn) is None


def test_progress_ignored_unless_running(settings):
    conn = make_db(settings)
    j = queue.enqueue(conn, "segment")
    queue.set_progress(conn, j["id"], 0.9)
    assert queue.get_job(conn, j["id"])["progress"] == 0


def test_fail_and_cancel(settings):
    conn = make_db(settings)
    a, b = queue.enqueue(conn, "x")["id"], queue.enqueue(conn, "x")["id"]
    queue.claim_next(conn)
    queue.fail(conn, a, "boom")
    assert queue.get_job(conn, a)["error"] == "boom" and queue.get_job(conn, a)["status"] == "failed"
    assert queue.cancel(conn, b)["status"] == "cancelled"
    assert queue.claim_next(conn) is None  # cancelled jobs are not run


def test_fail_interrupted_only_touches_running(settings):
    conn = make_db(settings)
    q = queue.enqueue(conn, "x")["id"]
    r = queue.enqueue(conn, "x")["id"]
    queue.claim_next(conn)  # q is now running
    assert queue.fail_interrupted(conn) == 1
    assert queue.get_job(conn, q)["status"] == "failed" and "interrupted" in queue.get_job(conn, q)["error"]
    assert queue.get_job(conn, r)["status"] == "queued"
    assert queue.fail_interrupted(conn) == 0


# ---- worker ------------------------------------------------------------------------------------------------------


def test_stub_handlers_fail_with_clear_message(settings):
    conn = make_db(settings)
    w = Worker(settings)
    for t in ("predict", "train"):
        jid = queue.enqueue(conn, t)["id"]
        assert w.run_once(conn) is True
        job = queue.get_job(conn, jid)
        assert job["status"] == "failed"
        assert job["error"] == f"'{t}' jobs are not implemented until M4"
    assert w.run_once(conn) is False


def test_unknown_job_type_and_handler_exception(settings):
    conn = make_db(settings)

    def boom(ctx):
        raise ValueError("kaput")

    w = Worker(settings, {"boom": boom})
    a, b = queue.enqueue(conn, "nonsense")["id"], queue.enqueue(conn, "boom")["id"]
    w.run_once(conn)
    w.run_once(conn)
    assert "no handler" in queue.get_job(conn, a)["error"]
    assert queue.get_job(conn, b)["error"] == "ValueError: kaput"
    assert queue.get_job(conn, b)["status"] == "failed"


def test_handler_progress_and_result(settings):
    conn = make_db(settings)
    seen = []

    def work(ctx):
        ctx.progress(0.5)
        seen.append(queue.get_job(ctx.conn, ctx.job["id"])["progress"])
        return {"ok": True}

    w = Worker(settings, {"work": work})
    jid = queue.enqueue(conn, "work")["id"]
    w.run_once(conn)
    assert seen == [0.5] and queue.get_job(conn, jid)["result"] == {"ok": True}


def test_worker_runs_one_job_at_a_time_in_order(settings):
    conn = make_db(settings)
    active, peak, order = [0], [0], []
    lock = threading.Lock()

    def slow(ctx):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            order.append(ctx.payload["n"])
        time.sleep(0.05)
        with lock:
            active[0] -= 1

    w = Worker(settings, {"slow": slow})
    ids = [queue.enqueue(conn, "slow", {"n": i})["id"] for i in range(4)]
    w.start()
    deadline = time.time() + 10
    while time.time() < deadline and queue.get_job(conn, ids[-1])["status"] != "succeeded":
        time.sleep(0.02)
    assert w.stop()
    assert order == [0, 1, 2, 3] and peak[0] == 1
    assert all(queue.get_job(conn, i)["status"] == "succeeded" for i in ids)


def test_worker_stops_cleanly_and_is_restartable(settings):
    make_db(settings).close()
    w = Worker(settings)
    w.start()
    assert w.running
    assert w.stop() and not w.running
    w.start()
    assert w.running
    assert w.stop()


def test_stop_waits_for_job_in_progress(settings):
    conn = make_db(settings)
    started, release = threading.Event(), threading.Event()

    def hold(ctx):
        started.set()
        release.wait(5)

    w = Worker(settings, {"hold": hold})
    jid = queue.enqueue(conn, "hold")["id"]
    w.start()
    assert started.wait(5)
    threading.Timer(0.2, release.set).start()
    assert w.stop(timeout=5)
    assert queue.get_job(conn, jid)["status"] == "succeeded"


# ---- through the API and app lifespan --------------------------------------------------------------------------


def test_job_endpoints(client):
    assert client.get("/jobs").json() == []
    assert client.get("/jobs/1").status_code == 404
    client.app.state.worker.stop()  # keep jobs queued
    conn = connect(client.app.state.settings.db_path)
    a = queue.enqueue(conn, "predict")["id"]
    b = queue.enqueue(conn, "segment", {"page_id": 5})["id"]
    all_ = client.get("/jobs").json()
    assert [j["id"] for j in all_] == [b, a]  # newest first
    assert [j["id"] for j in client.get("/jobs?type=predict").json()] == [a]
    assert [j["id"] for j in client.get("/jobs?status=queued&limit=1").json()] == [b]
    assert client.get("/jobs?status=bogus").status_code == 422
    one = client.get(f"/jobs/{a}").json()
    assert set(one) >= {"id", "type", "status", "payload", "progress", "error", "created_at"}


def test_cancel_endpoint(client):
    client.app.state.worker.stop()
    conn = connect(client.app.state.settings.db_path)
    jid = queue.enqueue(conn, "predict")["id"]
    assert client.post(f"/jobs/{jid}/cancel").json()["status"] == "cancelled"
    assert client.post(f"/jobs/{jid}/cancel").status_code == 409
    assert client.post("/jobs/999/cancel").status_code == 404


def test_progress_visible_through_api_while_running(client):
    started, release = threading.Event(), threading.Event()

    def stage(ctx):
        ctx.progress(0.25)
        started.set()
        release.wait(10)
        return {}

    client.app.state.worker.handlers = {"stage": stage}
    conn = connect(client.app.state.settings.db_path)
    jid = queue.enqueue(conn, "stage")["id"]
    client.app.state.worker.wake()
    assert started.wait(10)
    job = client.get(f"/jobs/{jid}").json()
    assert job["status"] == "running" and job["progress"] == 0.25
    release.set()
    assert wait_job(client, jid)["status"] == "succeeded"


def test_crash_recovery_on_startup(settings):
    conn = make_db(settings)
    crashed = queue.enqueue(conn, "segment", {"page_id": 1})["id"]
    queue.claim_next(conn)  # left 'running' by a process that then died
    waiting = queue.enqueue(conn, "predict")["id"]
    conn.close()
    with TestClient(create_app(settings)) as c:
        job = c.get(f"/jobs/{crashed}").json()
        assert job["status"] == "failed" and "interrupted" in job["error"] and job["finished_at"]
        # the queued job survived and is picked up by the new worker (its stub handler then fails it)
        assert "M4" in wait_job(c, waiting)["error"]


def test_app_startup_migrates_a_fresh_database(settings):
    assert not settings.db_path.exists()
    with TestClient(create_app(settings)) as c:
        assert c.get("/health").status_code == 200
        assert c.app.state.worker.running
        worker = c.app.state.worker
    assert settings.db_path.exists() and not worker.running  # lifespan shutdown stopped the worker
    conn = connect(settings.db_path)
    assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 1
