# HTR

Local-first handwritten text recognition for one writer's scanned notes. Nothing leaves the machine.
Full spec: [docs/implementation-plan.md](docs/implementation-plan.md).

## Layout

```
backend/     uv project (Python 3.11): FastAPI app, tests, pyproject.toml + uv.lock
frontend/    placeholder (React, from M3)
scripts/     check_env.py, inspect_images.py, htr.py (import/segment), review_sheet.py, make_synthetic_pages.py
backend/app/pipeline/   measure, preprocess, segment, pagexml, overlay, workflow, review (M1 pipeline)
backend/app/db/         SQLite connection + versioned migrations
backend/app/services/   pages (import), lines (CRUD), sync (DB <-> PAGE-XML)
backend/app/worker/     SQLite job queue, handlers, in-process worker thread
backend/app/api/        FastAPI routes and request schemas
data/        gitignored: scans, PAGE-XML, models, DB (never committed)
docs/
```

## Setup (WSL2 Ubuntu + NVIDIA GPU)

Work inside WSL2 on the Linux filesystem (e.g. `~/HTR`), not under `/mnt/c`.

1. **GPU driver**: install the NVIDIA Windows driver (it provides CUDA to WSL2). `nvidia-smi` must work inside WSL.
2. **uv**: `curl -LsSf https://astral.sh/uv/install.sh | sh`, then make sure `~/.local/bin` is on `PATH`.
3. **Python 3.11**: `uv python install 3.11`. Do not use the system Python (Ubuntu may ship 3.14, which Kraken
   does not support). The project pins `requires-python = ">=3.11,<3.12"` and uv uses 3.11 automatically.
4. **Pinned install**: `make setup` (runs `uv sync --project backend --python 3.11`). Versions are pinned in
   `backend/pyproject.toml` and locked in `backend/uv.lock`:
   - `torch==2.9.1`, `torchvision==0.24.1` from the PyTorch **CUDA 12.6** index
   - `kraken==7.1.1` (latest release supporting Python 3.11; requires torch >= 2.9)
5. **Verify**: `make check`.

GPU note: the target card is a GTX 1050 Ti (Pascal, compute capability 6.1, 4 GB VRAM). Newer PyTorch CUDA builds
(12.8+) drop Pascal kernels, so the CUDA 12.6 wheels are used. These wheels list `sm_60` but not `sm_61` in
`torch.cuda.get_arch_list()`; `sm_60` machine code runs on `sm_61` GPUs, and `make check` confirms this with a real
CUDA operation. Keep training batch sizes small enough for 4 GB.

## Commands

| Command | What it does |
|---|---|
| `make setup` | Install pinned dependencies into `backend/.venv` |
| `make check` | Environment check: WSL2, real CUDA op, GPU/compute capability, Kraken CLI, free disk. Exits non-zero on failure |
| `make test` | pytest (includes a guard that fails if anything under `data/` is git-tracked) |
| `make test-kraken` | The real (unmocked) Kraken `blla` integration test, excluded from `make test` (marker `kraken`) |
| `make fonts` | Optional one-time dev download of three OFL script fonts into `data/fonts/` for nicer synthetic test pages |
| `make lint` | ruff check + ruff format --check |
| `make format` | ruff autofix + format |
| `make serve` | The API on `http://127.0.0.1:8000` (loopback only); docs at `/docs`. See "M2: backend API" below |

## Inspecting a folder of scans

```
uv run --project backend python scripts/inspect_images.py /path/to/scans [-r] [--json]
```

For every PNG/JPG/TIFF it prints pixel size, DPI tag (or `none`), estimated text-line height and line pitch in
pixels, and warnings (low resolution, unusual size or aspect ratio, missing/contradictory DPI), followed by a
summary of the whole set. DPI tags are reported but not trusted; line height drives later normalization.
Line height is estimated from a horizontal projection profile of the binarized page, with long horizontal
rules (ruled paper) removed first. Multi-page TIFFs: only the first page is inspected.

Keep scans under `data/` (gitignored), e.g. `data/scans/`.

## M1 workflow: import, segment, review

All commands run through `uv run --project backend python ...` (shown without that prefix). Output goes under `data/`
(gitignored). Nothing here makes network calls.

1. **Inspect** the scans and pick a target line height: `scripts/inspect_images.py data/scans` (see above).
2. **Import**: `scripts/htr.py import data/scans [-r] [--target-line-height 40] [--no-deskew]`
   - copies each original byte-for-byte to `data/originals/<name>_<hash8>.<ext>` (never modified);
     files whose SHA-256 is already imported are skipped, so re-running is safe;
   - writes a working copy `data/working/<id>.png`: grayscale, background/contrast normalized, deskewed (only
     0.3 to 5 degrees), and rescaled toward the target line height **only** when the measured line height is
     outside 20 to 150 px (the default target is 40 px; changing the option re-derives working copies);
   - writes `data/manifest.json` and an import report `data/import_report.md` / `.json` that flags low-resolution,
     odd-sized, missing/contradictory-DPI and blank images (same warnings as `inspect_images.py`).
   - an unreadable file is reported and skipped; it never aborts the batch.
3. **Segment**: `scripts/htr.py segment [--device auto|cpu|cuda:0] [--force] [--page ID]`
   - runs Kraken `blla` on the working copies, writing `data/pagexml/<id>.xml` (PAGE-XML 2019-07-15: baselines,
     polygons, reading order; coordinates are in working-copy pixels) and `data/overlays/<id>.png` (polygons,
     baselines and reading-order numbers over the page);
   - resumable: pages already done are skipped, failed pages are recorded in the manifest and retried next run;
     the manifest is saved after every page; exit status 1 if any page failed. `--device auto` uses the GPU if
     available, and falls back to CPU on CUDA out-of-memory.
4. **Review sample**: `scripts/review_sheet.py sample [-n 20]` writes `data/review_sheet.csv` with columns
   `page, detected_lines, correct_lines, missed_lines, notes` for N pages: the ones with the most import warnings
   first, the rest chosen at random (seeded).
5. **Judge by eye** (human step): open `data/overlays/<page>.png` for each page and fill in
   - `correct_lines`: how many of the detected lines are segmented correctly (one clean line, nothing clipped,
     merged or split);
   - `missed_lines`: how many real text lines have no detected line at all.

   Both columns are required on every row; enter `0` when there are none. A blank value is an error, so no page is
   silently skipped.
6. **Decide**: `scripts/review_sheet.py score [data/review_sheet.csv]` computes
   `share = correct / (detected + missed)` over all pages and prints the plan's decision: more than 90%: proceed;
   70 to 90%: prioritize repair tools; below 70%: tune the segmenter.

   Worked example (invented numbers):

   ```
   page,detected_lines,correct_lines,missed_lines,notes
   a,14,12,0,
   b,10,7,2,margin note merged into line 4
   c,16,16,1,
   ```

   detected = 40, correct = 35, missed = 3, so share = 35 / 43 = 81.4%, which means "prioritize repair tools".

The line height shown in the import report is measured on the normalized, deskewed image, the same measurement that
decides whether the working copy is rescaled. (Measuring the raw original was less reliable: the estimator's band
thickness grows on slightly skewed pages, so line pitch is the steadier number.)

### Kraken segmentation model

`blla` needs its default segmentation model. In kraken 7.1.1 it is `kraken/blla.mlmodel`, **shipped inside the
kraken wheel**, so `make setup` already provides it and no `kraken get` download is needed. The model repository
(`kraken get`) is only relevant for recognition models in M4. `make test-kraken` runs a real segmentation of a
generated synthetic page to verify this on your machine.

### Synthetic pages (development only)

There are no real scans in the repo. `scripts/make_synthetic_pages.py OUT_DIR -n 8` renders handwriting-like pages
(script fonts from `data/fonts/` if you ran `make fonts`, otherwise DejaVu) with varied size, skew, ruled/unruled
paper and a margin note, plus a `truth.json` with the number of lines rendered. Results on synthetic pages say
nothing about real handwriting; the go/no-go decision requires your real scans.

## M2: backend API, database, job queue

### Running the server

```
make serve        # uvicorn app.main:app --host 127.0.0.1 --port 8000   (loopback only)
```

Open `http://127.0.0.1:8000/docs` for the interactive API. A one-page endpoint summary is in
[docs/api.md](docs/api.md). On startup the app creates `data/` if needed, migrates the database, marks jobs left
`running` by a crash as failed, and starts the worker thread; on shutdown it stops the worker (a job in progress is
allowed to finish, up to 30 s).

Environment variables (all optional): `HTR_DATA_DIR` (default `<repo>/data`), `HTR_DB_PATH` (default
`<data>/htr.db`), `HTR_MAX_UPLOAD_MB` (default 100), `HTR_DEVICE` (`auto`, `cpu` or `cuda:0`, default `auto`).
The server only accepts browser calls from `http://localhost:5173` and `http://127.0.0.1:5173` (Vite dev); there is no
wildcard and no authentication (single user, loopback).

Quick try (second terminal):

```
curl -s -X POST localhost:8000/documents -H 'content-type: application/json' -d '{"name":"notes"}'
curl -s -X POST localhost:8000/pages -F document_id=1 -F file=@data/scans/page1.png
curl -s -X POST localhost:8000/pages/1/segment        # returns a job
curl -s localhost:8000/jobs/1                         # poll until "succeeded"
curl -s localhost:8000/pages/1                        # the page with its lines
```

### Endpoints

| | |
|---|---|
| `POST /documents`, `GET /documents` | create / list documents |
| `POST /pages` | multipart upload (`file`, `document_id`); PNG/JPEG/TIFF, validated; identical content is not imported twice |
| `GET /pages`, `GET /pages/{id}` | status (`unsegmented`, `segmented`, `partially_transcribed`, `done`), line and ground-truth counts; detail adds lines |
| `GET /pages/{id}/original`, `GET /pages/{id}/working` | the untouched original / the working copy |
| `POST /pages/{id}/segment` | enqueue a segmentation job |
| `PATCH /lines/{id}`, `DELETE /lines/{id}`, `POST /pages/{id}/lines` | line CRUD (text, polygon, baseline, ignored, ground truth, reorder via `index`) |
| `GET /lines/{id}/crop.png` | line crop (polygon bounding box of the working copy) |
| `GET /jobs`, `GET /jobs/{id}`, `POST /jobs/{id}/cancel` | job status, progress, errors; cancel queued jobs |

Details and rules: [docs/api.md](docs/api.md). Not in M2: line split/merge (M3), predict/train endpoints (M4;
those job types exist as stubs that fail with "not implemented until M4"), abbreviations endpoints, export.

### Data layout

```
data/
  htr.db                 SQLite (WAL mode; htr.db-wal / -shm beside it while running)
  originals/<stem>_<hash8>.<ext>   uploaded files, byte-identical, never modified
  working/<stem>_<hash8>.png       normalized / deskewed / maybe rescaled working copy (M1 pipeline)
  pagexml/<stem>_<hash8>.xml       PAGE-XML per segmented page, coordinates in working-copy pixels
  tmp/                   uploads in flight
```

The M1 CLI (`scripts/htr.py`) uses the same folders but its own `manifest.json`; the API does not read that manifest,
so pages imported with the CLI do not appear in the API (re-upload them).

### Database

Plain stdlib `sqlite3` with a small versioned migration runner (`backend/app/db/migrations.py`): for one user and six
tables an ORM would add more than it saves. Tables follow the plan: `documents`, `pages`, `lines`, `models`, `jobs`,
`abbreviations`, plus `schema_version`. Foreign keys are on, the journal is WAL, each request / the worker has its own
connection, writes are explicit `BEGIN IMMEDIATE` transactions. Notable columns beyond the plan:

- `pages`: `width`/`height` are the **working copy** size (all coordinates live there); `image_path` is the original;
  `original_filename`, `sha256` (unique, dedupes uploads), `original_width/height`, `dpi_x/y`, `line_height_px`,
  `scale`, `skew_deg`, `deskewed`, `warnings` (json import warnings), `preprocess` (json), `imported_at`,
  `segmented_at`. `status` is stored and recomputed in every transaction that changes lines.
- `lines`: `"index"` (quoted: reserved word) is contiguous per page and maintained by the service layer;
  `text_source` is NULL while the line has no text.
- `jobs`: plus `result` (json), `started_at`, `finished_at`.

To change the schema append a migration to `MIGRATIONS`; never edit an applied one. A database from a newer app
version is refused.

### PAGE-XML sync

`data/pagexml/<page>.xml` is rewritten from the database after every line mutation and every segmentation, inside the
same operation as the DB commit: the transaction opens, rows change, the XML is written to `<page>.xml.tmp` and renamed
over the old file (atomic), then the transaction commits. If any step fails the transaction is rolled back and the
previous XML stays. (If COMMIT itself fails after the rename, the XML is regenerated from the unchanged database.)

State is stored in the file so that it is complete on its own: text in `TextEquiv/Unicode` (prediction confidence in
`TextEquiv@conf`), reading order in `custom="readingOrder {index:N;}"`, and the flags PAGE-XML has no field for in an
`htr` block of the same `custom` attribute:

```
custom="readingOrder {index:3;} htr {ground_truth:true; ignored:true; text_source:manual;}"
```

`ground_truth` / `ignored` are written only when true. `app.services.sync.rebuild_lines_from_xml(conn, settings,
page_id, xml_path=None)` is the import path: it replaces a page's DB lines from a PAGE-XML file (sizes must match the
working copy) and restores text, source, confidence, ground-truth and ignored flags; the round trip DB -> XML -> DB is
covered by tests. Training (M4) must skip `ignored` lines.

### Job queue

`jobs` rows are the queue. One worker thread polls (and is woken on enqueue), claims the oldest `queued` job, runs its
handler and records `succeeded` / `failed` (with the error message). One job at a time, because the GPU is shared.
Handlers: `segment` (Kraken `blla` via the M1 `segment.segment_image`; replaces the page's lines; refuses to discard
manually transcribed lines unless `force`), `predict` and `train` (stubs). Segmentation points are clamped to the page.

### Tests

`make test` runs the API tests with Kraken mocked (fast, no GPU). `make test-kraken` runs the real-model tests,
including `tests/test_kraken_api.py`: import a synthetic page through the API, segment it with real Kraken via the
job queue, and read the lines back (run with `-s` to see the printed lines).
