# HTR

Local-first handwritten text recognition for one writer's scanned notes. Nothing leaves the machine.
Full spec: [docs/implementation-plan.md](docs/implementation-plan.md).

## Layout

```
backend/     uv project (Python 3.11): FastAPI app, tests, pyproject.toml + uv.lock
frontend/    placeholder (React, from M3)
scripts/     check_env.py, inspect_images.py, htr.py (import/segment), review_sheet.py, make_synthetic_pages.py
backend/app/pipeline/   measure, preprocess, segment, pagexml, overlay, workflow, review
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
| `make serve` | Stub API on `http://127.0.0.1:8000` (loopback only); `GET /health` |

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
   `page, detected_lines, correct_lines, notes` for N pages: the ones with the most import warnings first, the rest
   chosen at random (seeded).
5. **Judge by eye** (human step): open `data/overlays/<page>.png` for each page and fill `correct_lines` with how
   many of the detected lines are segmented correctly (one clean line, nothing clipped, merged or split). Lines the
   segmenter missed entirely are not in `detected_lines`; to count them, add an optional integer column
   `missed_lines` (they are added to the denominator).
6. **Decide**: `scripts/review_sheet.py score [data/review_sheet.csv]` prints the share of correct lines and the plan's
   decision: more than 90%: proceed; 70 to 90%: prioritize repair tools; below 70%: tune the segmenter.

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
