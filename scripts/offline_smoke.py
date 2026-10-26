#!/usr/bin/env python3
"""Real import -> segment -> train -> predict smoke flow, meant to run inside a network namespace that has no network
other than loopback (see `make verify-offline` and scripts/verify_offline.sh). Development tool; it talks only to the
server it starts itself on 127.0.0.1.

Steps (nothing is mocked: real Kraken segmentation, real fine-tune of the base model, real prediction):

  1. prove this process really is offline: a TCP connect to a public address and a DNS lookup must fail;
  2. start the app (uvicorn on 127.0.0.1) with a throw-away data directory;
  3. import two synthetic pages, segment them, label the lines that match the rendered text, train 1 epoch, predict
     the rest, export text and a bundle, read /health/details;
  4. check that the server only listens on loopback and that the data/logs file holds none of the text used;
  5. stop the server and delete the throw-away directory.

The training is a plumbing check, not an accuracy check: a handful of synthetic lines teach nothing meaningful.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "backend"), str(REPO / "scripts")]

import httpx  # noqa: E402  (dev dependency; loopback only)
from make_synthetic_project import match_lines, render_project, wait_job  # noqa: E402

PORT = 8123


def step(msg: str) -> None:
    print(f"[offline-smoke] {msg}", flush=True)


def prove_offline() -> None:
    for host in ("1.1.1.1", "8.8.8.8"):
        s = socket.socket()
        s.settimeout(3)
        try:
            s.connect((host, 53))
        except OSError as e:
            step(f"connect to {host}:53 failed as it should: {e.strerror or e}")
        else:
            raise SystemExit(f"NOT OFFLINE: connected to {host}; run this through `make verify-offline`")
        finally:
            s.close()
    try:
        socket.getaddrinfo("example.com", 443)
    except OSError as e:
        step(f"DNS lookup failed as it should: {e}")
    else:
        raise SystemExit("NOT OFFLINE: example.com resolved")
    step("this process has no route to the outside world")


def listening_addresses(pid: int) -> set[str]:
    """Local addresses of the TCP sockets in LISTEN state owned by `pid` (from /proc, so no extra tools)."""
    inodes = set()
    for fd in Path(f"/proc/{pid}/fd").iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith("socket:["):
            inodes.add(target[8:-1])
    out = set()
    for table, v6 in (("/proc/net/tcp", False), ("/proc/net/tcp6", True)):
        try:
            rows = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            f = row.split()
            if f[3] == "0A" and f[9] in inodes:  # LISTEN
                addr, port = f[1].split(":")
                if v6:
                    out.add("::1" if addr == "00000000000000000000000001000000" else f"[{addr}]")
                else:
                    ip = ".".join(str(int(addr[i : i + 2], 16)) for i in (6, 4, 2, 0))
                    out.add(ip)
                out.add(f":{int(port, 16)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda:0 for the app (default auto)")
    ap.add_argument("--pages", type=int, default=2)
    args = ap.parse_args()

    prove_offline()
    base = REPO / "data" / "models" / "base"
    if not list(base.glob("*/BASE_MODEL.json")):
        raise SystemExit("no base model: run `make base-model` once (setup step, needs the network)")

    work = Path(tempfile.mkdtemp(prefix="htr-offline-"))
    server: subprocess.Popen | None = None
    try:
        data = work / "data"
        (data / "models").mkdir(parents=True)
        (data / "models" / "base").symlink_to(base)
        env = {**os.environ, "HTR_DATA_DIR": str(data), "HTR_DEVICE": args.device, "PYTHONPATH": str(REPO / "backend")}
        for var in ("HTR_LLM_ENABLED", "HTR_LLM_URL"):
            env.pop(var, None)
        step(f"starting the app on 127.0.0.1:{PORT} (data dir {data})")
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(PORT)],
            cwd=REPO / "backend",
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        client = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=120)
        for _ in range(240):
            try:
                if client.get("/health").status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.5)
        else:
            raise SystemExit("the app did not start")

        listening = listening_addresses(server.pid)
        step(f"server listens on: {sorted(listening)}")
        if not listening <= {"127.0.0.1", ":" + str(PORT)}:
            raise SystemExit(f"server listens on more than loopback: {listening}")

        rendered = render_project(work / "synthetic", args.pages, seed=11)
        doc = client.post("/documents", json={"name": "offline smoke"}).json()
        truth: dict[int, str] = {}
        for item in rendered:
            with open(work / "synthetic" / "pages" / item["file"], "rb") as f:
                files = {"file": (item["file"], f.read())}
                up = client.post("/pages", data={"document_id": str(doc["id"])}, files=files)
            up.raise_for_status()
            pid = up.json()["id"]
            job = wait_job(client, client.post(f"/pages/{pid}/segment").json()["id"])
            if job["status"] != "succeeded":
                raise SystemExit(f"segmentation {job['status']}: {job['error']}")
            for lid, t in match_lines(client.get(f"/pages/{pid}").json(), item["truth"]).items():
                truth[lid] = t["text"]
        step(f"imported and segmented {args.pages} pages; {len(truth)} lines match the rendered text")
        ids = sorted(truth)
        if len(ids) < 12:
            raise SystemExit(f"only {len(ids)} usable lines; need at least 12 for the smoke test")
        labeled = ids[: len(ids) // 2]
        for lid in labeled:
            client.patch(f"/lines/{lid}", json={"text": truth[lid], "is_ground_truth": True}).raise_for_status()

        job = client.post("/jobs/train", json={"epochs": 1, "batch_size": 2}).json()
        done = wait_job(client, job["id"], timeout=900)
        if done["status"] != "succeeded":
            raise SystemExit(f"training {done['status']}: {done['error']}")
        r = done["result"]
        step(f"trained: {r['train_lines']} lines on {r['device']}, CER {r['base_cer']} -> {r['cer']}")
        pdone = wait_job(client, client.post("/jobs/predict", json={}).json()["id"], timeout=900)
        if pdone["status"] != "succeeded":
            raise SystemExit(f"prediction {pdone['status']}: {pdone['error']}")
        step(f"predicted {pdone['result']['predicted']} lines (model {pdone['result']['model']})")
        if pdone["result"]["predicted"] < 1:
            raise SystemExit("prediction produced nothing")

        txt = client.get("/export", params={"include_predicted": "true"})
        txt.raise_for_status()
        bundle = client.get("/bundle/export")
        bundle.raise_for_status()
        health = client.get("/health/details").json()
        step(f"export {len(txt.text)} chars, bundle {len(bundle.content) // 1024} KiB, health: {health['status']}")

        server.send_signal(signal.SIGINT)
        server.wait(timeout=60)
        server = None
        logs = "\n".join(p.read_text(errors="replace") for p in (data / "logs").glob("htr.log*"))
        leaked = [t for t in truth.values() if len(t) > 6 and t in logs]
        if not logs:
            raise SystemExit("no log file was written")
        if leaked:
            raise SystemExit(f"transcribed text found in the log file: {leaked[:2]}")
        step(f"log file has {len(logs.splitlines())} lines and none of the {len(truth)} transcribed lines")
        step("OK: import -> segment -> train -> predict -> export -> bundle all worked with no network")
        return 0
    finally:
        if server is not None:
            server.kill()
            server.wait()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
