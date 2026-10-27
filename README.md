# HTR

Local-first handwritten text recognition for one writer's scanned notes. You transcribe a few lines, the system learns
your hand, and it predicts the rest. **Nothing leaves your machine**: the server listens on 127.0.0.1 only, makes no
outbound connections (tested), and keeps scans, transcriptions and models in a `data/` folder that git ignores.

Full spec and decisions: [docs/implementation-plan.md](docs/implementation-plan.md). HTTP API:
[docs/api.md](docs/api.md). First measurement of the recognizer on real (word-level) handwriting:
[docs/experiments/rxhand.md](docs/experiments/rxhand.md).

## Quick start

On Windows with WSL2 (Ubuntu) and an NVIDIA driver; the complete, step-by-step version with the first-use walkthrough,
troubleshooting and privacy notes is **[docs/setup-wsl2.md](docs/setup-wsl2.md)**.

```bash
sudo apt install -y git curl make xz-utils build-essential        # once
git clone <repository URL> ~/HTR && cd ~/HTR                       # on the Linux filesystem, not /mnt/c
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
make setup-node        # Node 22 LTS, user level
make setup             # Python 3.11 + pinned PyTorch (CUDA 12.6) + Kraken
make setup-frontend    # npm packages + headless Chromium for the e2e test
make base-model        # pretrained recognition model (one-time download)
make check             # GPU / Kraken / disk self-check
make dev               # then open http://127.0.0.1:5173
```

Everything marked "setup" needs the network once. After that the app runs offline (`make verify-offline` proves it).

GPU note: the target card is a GTX 1050 Ti (Pascal, compute capability 6.1, 4 GB VRAM). Newer PyTorch CUDA builds
(12.8+) drop Pascal kernels, so the CUDA 12.6 wheels are used. These wheels list `sm_60` but not `sm_61` in
`torch.cuda.get_arch_list()`; `sm_60` machine code runs on `sm_61` GPUs, and `make check` confirms this with a real
CUDA operation. Keep training batch sizes small enough for 4 GB.

## Layout

```
backend/     uv project (Python 3.11): FastAPI app, tests, pyproject.toml + uv.lock
frontend/    React + TypeScript + Vite app: src/api (typed client), src/app (status strip, toasts, error boundary),
             src/library, src/editor, src/training, src/review, src/abbreviations, src/export; e2e/ (Playwright)
scripts/     check_env.py, inspect_images.py, htr.py (import/segment/bundle), review_sheet.py, get_base_model.py,
             install_node.sh, audit_network.py + offline_smoke.py + verify_offline.sh (no-network proof),
             make_synthetic_*.py (development only)
backend/app/pipeline/   measure, preprocess, segment, pagexml, overlay, workflow, review (M1 pipeline);
                        recognize (Recognizer/Trainer interface + Kraken), train (fine-tuning), evaluate (CER/WER)
backend/app/db/         SQLite connection + versioned migrations
backend/app/services/   pages (import), lines (CRUD), repair, sync (DB <-> PAGE-XML), models, training, review,
                        abbreviations, suggestions (optional LLM), export, bundle
backend/app/worker/     SQLite job queue, handlers, in-process worker thread
backend/app/api/        FastAPI routes and request schemas
backend/app/            web.py (Host check, error guard, request log), failures.py (error classification), health.py
                        (self-checks), logging_setup.py, abbreviation_llm.py (optional, loopback-only)
data/        gitignored: scans, PAGE-XML, models, DB, logs (never committed)
docs/
```

## Commands

| Command | What it does |
|---|---|
| `make setup` | Install pinned dependencies into `backend/.venv` |
| `make setup-node` | One-time: Node 22 LTS at user level (no sudo) into `~/.local/node` |
| `make setup-frontend` | One-time: `npm ci` plus headless Chromium for the e2e test (and unpacks missing Chromium system libraries without sudo). Dev tooling downloads only |
| `make base-model` | One-time download of the pretrained recognition model into `data/models/base/` (see "M4"); the app itself never goes online |
| `make fonts` | Optional one-time dev download of three OFL script fonts into `data/fonts/` for nicer synthetic test pages |
| `make check` | Environment check: WSL2, real CUDA op, GPU/compute capability, Kraken CLI, free disk. Exits non-zero on failure |
| `make dev` | API (8000) and frontend (5173) together; Ctrl+C stops both |
| `make serve` / `make dev-frontend` | The API alone on `http://127.0.0.1:8000` (loopback only; docs at `/docs`) / the Vite dev server alone |
| `make test` | pytest (includes the guards that fail if anything under `data/` is git-tracked or any code reaches a non-loopback address) |
| `make test-kraken` | The real (unmocked) Kraken `blla` integration test, excluded from `make test` (marker `kraken`) |
| `make test-train` | Opt-in tiny real fine-tune + predict with Kraken (marker `train`; needs `make base-model` and `make fonts`; about 15 s) |
| `make test-frontend` / `make lint-frontend` | Vitest unit and component tests / ESLint plus strict `tsc` |
| `make test-e2e` | Opt-in Playwright end-to-end test (real backend on a temp data dir, real Kraken, headless Chromium); its temp data is deleted afterwards |
| `make lint` / `make format` | ruff check + format check (also covers `scripts/`) / autofix + format |
| `make verify-offline` | Static audit (no network code or external URLs in the app, frontend source and built frontend) plus the real import, segment, train, predict flow inside an offline network namespace (`unshare -rn`, loopback only). See "Privacy and the network" |
| `make audit-network` | Only the static audit |
| `make clean-test` | Delete temp data left by test runs (pytest base directories, e2e roots) |
| `make synthetic-rounds` / `make synthetic-project` / `make synthetic-report` | Development proofs of the loop on synthetic pages (see "M4" and "M5") |
| `python scripts/htr.py bundle export\|import <file>` | Project bundle from the command line, without the server (see "M6") |

## Privacy and the network

* The server binds `127.0.0.1` only (`make serve`, `uvicorn --host 127.0.0.1`), answers only requests whose `Host`
  header is `127.0.0.1`, `localhost` or `[::1]` (DNS-rebinding protection; anything else gets 421), and allows CORS only
  from `http://localhost:5173` and `http://127.0.0.1:5173`.
* **No outbound calls.** The whole backend test suite runs with a guard that makes any connect, send or DNS lookup to a
  non-loopback address raise and fail the test (`backend/tests/netguard.py`). `make verify-offline` adds a static audit
  (`scripts/audit_network.py`: no `requests`/`httpx`/`urllib.request`/CDN URLs in the app, the frontend source or the
  built frontend; only XML namespaces and loopback) and runs the real import, segment, train, predict flow in a network
  namespace that has only a loopback interface.
* **Setup-time downloads, and only those:** apt packages; uv, Python 3.11, PyTorch and the other wheels (`make setup`);
  Node (`make setup-node`); npm packages and Chromium (`make setup-frontend`); the pretrained base model
  (`make base-model`, Zenodo via HTRMoPo); optional fonts (`make fonts`, GitHub). The running app uses none of them.
* **Logs** (`data/logs/htr.log`, rotating JSON lines) hold ids, counts, timings, status codes and error codes. Never
  transcribed text, predictions, dictionary entries, query strings or scan file names (tested end to end).
* **Bundles, `data/` and exports are personal data.** `data/` and `*.htrbundle` are git-ignored; keep bundles where you
  put them on purpose.
* The optional local-LLM abbreviation suggestions are **off by default**, talk only to a loopback server and never apply
  anything on their own (see [docs/setup-wsl2.md](docs/setup-wsl2.md), section 10, and "M7" below).

## M7: hardening

* **Errors.** One format for every API error: `{detail, code, message, hint}` (see "Errors" in docs/api.md). Damaged or
  unsupported images, missing files, a locked or full database, a missing GPU (jobs fall back to the CPU and say so)
  and job failures all produce a plain message with a hint; no stack traces reach the browser. `GET /health/details`
  runs the self-checks (database migrated, data folder writable, free disk, GPU/CUDA, base model, worker, local LLM);
  the strip at the top of the UI shows the result, an offline banner with *Retry* appears when the server cannot be
  reached, and server errors raise a toast. A crash in a screen shows a friendly page instead of a blank one.
* **Uploads.** Single-page PNG/JPEG/TIFF only: a multi-page TIFF is refused with a hint to split it (silently
  importing only page 1 would lose the rest); files over `HTR_MAX_UPLOAD_MB` (100) or `HTR_MAX_IMAGE_MP` (80
  megapixels) are refused before they are decoded; identical concurrent uploads produce one page.
* **Predictions and undo.** Lines keep `predicted_by_model_id` and `predicted_text`; undo/restore gives them back
  exactly; merge and split produce unpredicted lines (the old prediction returns with the undo); ignoring a region
  records `ignored` review events for the predicted lines it ignores.
* **Local-LLM suggestions** (optional, off by default): `backend/app/abbreviation_llm.py` talks only to a loopback
  Ollama or OpenAI-compatible server (`HTR_LLM_ENABLED`, `HTR_LLM_URL`, `HTR_LLM_API`, `HTR_LLM_MODEL`); a prompt holds
  the abbreviation and its own line (plus at most two neighbouring lines, 300 characters each). Suggestions are stored
  in the `suggestions` table and reviewed on the Abbreviations page (accept, edit then accept, reject; rejected pairs are
  not suggested again); they never change a transcription. API in docs/api.md. Tested against a fake local server only;
  no real model was run.
* **Temp hygiene.** Passing tests delete their temp data immediately (`tmp_path_retention_policy = failed`), the e2e
  run deletes its root, `make clean-test` removes leftovers, and everything honours `$TMPDIR`. A full `make test` peaks
  at a fraction of the old ~2 GB and leaves nothing behind.
* **Line endings.** `.gitattributes` sets `text=auto eol=lf`; the repository is LF everywhere.

# Reference by milestone

The sections below were written milestone by milestone and describe each part in detail. Where they disagree with
[docs/setup-wsl2.md](docs/setup-wsl2.md) or [docs/api.md](docs/api.md), those two are current.

## M3: labeling UI (frontend)

Part A of M3: library and page editor. Segmentation repair tools (merge, split, redraw baseline, region ignore) come
later; the editor state machine (`frontend/src/editor/transcription.ts`) and canvas are structured so those modes can
be added. There are no prediction/confidence displays yet.

**Node**: Node 22 LTS (pinned in `frontend/.nvmrc`; 20 or newer works, `engines` says `>=20`). It is not in Ubuntu by
default; install it at user level without sudo, e.g. the official tarball:

```
mkdir -p ~/.local/node && cd ~/.local/node
curl -fsSLO https://nodejs.org/dist/latest-v22.x/node-v22.23.4-linux-x64.tar.xz   # pick the current 22.x file name
tar xf node-v22.23.4-linux-x64.tar.xz && ln -sfn "$PWD/node-v22.23.4-linux-x64" current
for b in node npm npx; do ln -sf ~/.local/node/current/bin/$b ~/.local/bin/$b; done
```

(fnm or nvm work too.) The Makefile puts `~/.local/bin` and `~/.local/node/current/bin` first on `PATH`, which also
keeps WSL from picking up a Windows `node.exe`. Then `make setup-frontend`.

**Run**: `make dev` (both servers), or in two terminals `make serve` and `make dev-frontend`, then open
`http://127.0.0.1:5173`. The API base URL defaults to `http://127.0.0.1:8000`; override with `VITE_API_BASE_URL`
(the browser origin must stay `127.0.0.1:5173` or `localhost:5173`, which is all the API's CORS allows). The built app
bundles everything and makes no outbound requests (system fonts, no CDN).

**Library**: create a document, drop images onto it (or "Choose images"), press Segment per page; job progress and
errors are polled and shown, identical re-uploads are reported as "Already imported". Re-segmenting a page that has
transcriptions asks for confirmation (it replaces all lines).

**Page editor** (open a page with Transcribe): the page canvas supports wheel zoom at the cursor, drag to pan,
click a line to select it, and keeps the selected line in view. Lines are blue (unlabeled), green (ground truth),
grey dashed (ignored), orange (selected). Below it: the line crop and a text box. Typing is optimistic: a failed
save never discards text, it shows an error with Retry, and leaving the page with unsaved edits asks first.

| Shortcut | Action |
|---|---|
| Enter | Save text, mark as ground truth, go to the next active (non-ignored) line |
| Tab / Shift+Tab | Next / previous line (all lines, no saving; typed text is kept per line) |
| Esc | Revert unsaved edits on the current line (closes the help first if it is open) |
| Ctrl+I | Ignore / un-ignore the current line |
| Alt+M (or M outside the text box) | Merge mode: click lines to add/remove them (the current line is pre-picked), Enter merges, Esc cancels |
| Alt+J (or J) | Merge the current line with the next one immediately |
| Alt+S (or S) | Split mode: click a point on a line to preview the cut, click it again or press Enter to cut |
| Alt+B (or B) | Draw baseline for a missed line: click 2+ points, Backspace removes the last, Enter creates the line |
| Alt+Shift+B (or Shift+B) | Redraw the baseline of the selected line (the server rebuilds its polygon) |
| Alt+R (or R) | Ignore region: drag a rectangle; every line at least half inside it is ignored |
| Alt+Delete / Alt+Backspace (or Delete) | Delete the current line |
| Alt+Up / Alt+Down | Move the current line up / down in reading order |
| Ctrl+Z (Alt+Z always) | Undo the last repair, multi-level (while the text box has unsaved edits Ctrl+Z is the normal text undo) |
| Esc in a mode | Cancel the mode and return to Select |
| Alt+PageDown / Alt+PageUp | Next / previous page of the document |
| Alt+0 | Fit page to width |
| Ctrl+/ or F1 (or `?` when focus is not in the text box) | Shortcut help overlay |

In the text box Tab and Enter belong to the transcription flow; toolbar buttons stay reachable with the mouse or by
clicking out of the box (buttons keep their normal Tab/Enter behaviour once focused).

**Segmentation repair** (part B of M3). The toolbar above the canvas has a button for every tool and a banner that
names the active mode; the canvas gets a pink frame and a crosshair, and pending geometry is drawn as a ghost (picked
lines in purple, the cut with its two halves, the baseline being drawn, the region rectangle with the lines it would
ignore). While a mode is active the keyboard focus moves off the text box so Enter, Esc and Backspace belong to the
mode. Merge or split of a transcribed line asks first (merge resets ground truth because the combined text is
unverified; split clears both halves and offers the old text to paste into either half). A new line from a drawn
baseline is placed in top-to-bottom reading order. Every operation can be undone with Ctrl+Z (up to 50 levels): the
editor remembers the page's lines before each operation and restores them through `POST /pages/{id}/restore`. That
endpoint refuses, and the UI asks you to confirm, if the undo would drop text you typed after the operation, so an
undo never silently loses transcription work. Errors are shown in a banner and leave the page as it was.

**Tests**: `make test-frontend` (API client, the pure transcription reducer and key mapping, zoom/pan/hit-test
maths, component tests for the keyboard flow and library). `make test-e2e` generates a synthetic page, uploads it,
segments it with real Kraken, transcribes three lines with the keyboard, reloads and checks the saved text and
ground-truth flags through the API, with no console errors. Two more e2e tests cover repair (split, merge, delete,
draw baseline, ignore region, undo, reload) and browser behaviour (wheel zoom at the cursor, drag pan, click-to-select,
Alt+0, Ctrl+I, Esc revert, a PATCH forced to fail that must keep the typed text and succeed on Retry). Screenshots land in `frontend/test-results/screenshots`.
It uses API port 8100 and web port 5173 and a temporary data dir, never `data/`.

## M4: training, prediction, evaluation

The loop: you transcribe lines and mark them ground truth, **fine-tune** a Kraken recognition model on them, the model
**predicts** the lines you have not done, and **CER/WER** on a fixed set of held-out lines tells you how good it is.
Everything runs locally; only the one-time base-model download (a dev setup step, below) goes online.

Use it from the **Training** page (link in the library header, `#/training`) or the API
([docs/api.md](docs/api.md): `POST /jobs/train`, `POST /jobs/predict`, `GET /models`, `POST /models/{id}/activate`,
`GET /training/summary`). The page shows your ground-truth count against the recommended minimum (about 100 lines for
the first round, then about 50 new lines per retrain) as a **non-blocking** warning, a form (epochs, batch size), live
progress with loss and validation accuracy and a Cancel button, a Predict button with progress, the model table (date,
start model, train and held-out lines, CER, WER, active marker, Activate) and a CER-by-round chart (inline SVG, no
library). Training and prediction are always allowed; the only refusal is "no usable ground-truth lines" (422).

### Setup: the pretrained base model (once)

```
make base-model        # downloads into data/models/base/ (gitignored), prints the model record
```

**Chosen model: "Rev City Handwritten English Base Model"** (HTRMoPo / Zenodo
[10.5281/zenodo.19238205](https://doi.org/10.5281/zenodo.19238205), `revcity_july_2025.mlmodel`, 16 MB, **Apache-2.0**).
Why: it is the only Kraken recognition model in the repository listing (109 records, queried with `kraken list`) that is
*English* and *handwriting* and in Kraken's native format; it has the same network architecture as Kraken's default
recognition model (about 4M parameters, input height 120, 118-symbol alphabet), so it fine-tunes on a 4 GB card; its
reported CER is 6.6% on its own test data (about 95k English lines, 18th-century archival hands, trained from scratch
with augmentation).

Implications, stated plainly: it was trained on **18th-century** hands (Revolutionary City dataset), not on modern
handwriting, so before fine-tuning it reads a modern writer poorly (about 29% CER on the synthetic script-font lines
below, which are an easier and cleaner problem than real pages). It is a better starting point than training from
scratch, but expect the first rounds on your real notes to be rough until you have transcribed a few hundred lines.
Other candidates looked at: *McCATMuS* (10.5281/zenodo.13788177, CC-BY-4.0: handwritten, printed and typewritten lines
from the 16th to the 21st century, mostly French, so a possible alternative for modern hands; not tested here:
`uv run --project backend python scripts/get_base_model.py --doi 10.5281/zenodo.13788177 --name mccatmus`); the
printed-English model (10.5281/zenodo.2577813) is for print; the multilingual "Party" and PP-OCR models use other
architectures than the VGSL one this code fine-tunes. If you download another model, press Activate on it in the
Training page (it is registered automatically the next time the models are listed).

The app registers every `data/models/base/*/BASE_MODEL.json` as a `kind = "base"` row in `models` and makes the first one
active. Without a base model, train and predict return a 422 that says to run `make base-model`.

### Workflow

1. Transcribe lines in the page editor (Enter marks a line as ground truth). Crossed-out or unusable lines: ignore them.
2. Open **Training**. Read the ground-truth count and the warnings; press **Start training** (defaults: 15 epochs, batch
   size 4, learning rate 0.0001, early stopping after 5 epochs without improvement). The job exports your lines, measures
   the start model on the held-out lines (the "before"), fine-tunes with augmentation, keeps the best epoch, measures it
   on the same held-out lines (the "after"), saves `data/models/<id>.mlmodel` and registers it. It becomes the active
   model unless it is worse than its start model on the held-out lines.
3. Press **Predict remaining lines**: every line that is not ground truth, not ignored and has no text you typed gets a
   prediction (`text_source = "predicted"`, with a confidence), written through the same transactional PAGE-XML sync as
   every other change. Ground truth and manual text are never overwritten, even if you edit a line while the job runs.
   (Showing predictions in the editor and the review queue come in M5; until then they are visible through the API and
   in the PAGE-XML.)
4. Correct some lines, mark them ground truth, train again: the next round starts from the active model.

Per round the files are `data/models/<id>.mlmodel` (the model) and `data/training/run-<job id>/` (PAGE-XML per page for
`train/`, `val/` and `eval/`, the masked line crops with `.gt.txt` transcriptions under `lines/`, and `manifest.json`
listing every line, its side of the split and the hyperparameters). Kraken 7.1.1 trains with PyTorch Lightning; the
augmentation is Kraken's own `augment` option (mild warp/rotation/scale, brightness and contrast, blur, noise).

### How the held-out set works

A line is held out when `sha256("htr-split-v1:<line id>") mod 10000 < 1200` (about 12%). The first run with at least 8
usable lines stores every line's side in the `line_splits` table; **a line never changes side afterwards**, so CER is
comparable across rounds, and held-out lines are never trained on. If that leaves no held-out line, the unassigned line
with the lowest hash is held out, so there is at least one whenever there are 8 or more usable lines; with fewer,
nothing is held out and the run reports no CER. Choosing the best epoch and early stopping use a separate validation
subset (about 10% of the training lines by another hash; with under 10 training lines, the training lines themselves),
never the held-out set, so the held-out CER is not tuned on. Usable = ground truth, not ignored, non-blank text.

### Metrics

- **CER** = character edit distance / reference characters, exactly as stored: no case folding, no Unicode
  normalization, whitespace counts (a missing space is an error). **WER** = word edit distance / reference words after
  collapsing whitespace (case and punctuation still count). Both are *micro averaged* over the held-out lines (total
  edits over total reference length) and can exceed 100%. Implemented in `backend/app/pipeline/evaluate.py`
  (unit-tested).
- **Reading CER**: 5% means about one wrong character in twenty. With 12-20 held-out lines (the first rounds) the number
  is noisy: two lines can move it by several points. Compare rounds, not single numbers; the Training page shows each
  model's CER next to the CER its start model had on the *same* lines ("was ..."). CER on synthetic fonts says nothing
  about real handwriting, and CER on held-out lines of your own hand says little about pages written very differently.
- **Confidence** (per predicted line, in [0, 1]) is the **geometric mean of the per-character confidences**, where a
  character's confidence is the highest softmax probability the network gave it (Kraken's `ocr_record.confidences`). A
  geometric mean lets one unsure character pull the line down (nine certain characters plus one coin flip score 0.93,
  not 0.95) and is length-normalized. Greedy maxima are optimistic and the network is not calibrated: use confidence to
  **rank** lines for review (lowest first, M5), not as the probability of a line being right. On the synthetic lines the
  line confidence and the line CER correlate at about -0.7 (lower confidence, more errors).

### Hardware notes (GTX 1050 Ti, 4 GB)

The recognition network has about 4M parameters; memory is dominated by activations, which grow with line width and
batch size. Measured on the synthetic project below with batch size 4: PyTorch peak allocation 1.3 GB; at the driver
level the process added about 1.8 GB (allocator cache + CUDA context) on top of what the desktop already used, so keep
other GPU applications closed while training and lower the batch size if you share the card (`batch_size` in the form,
or `HTR_TRAIN_BATCH_SIZE`). A CUDA out-of-memory **fails the job with a clear message** ("the GPU ran out of memory at
batch size 128. Retry with a smaller batch size (for example 64) or train on the CPU"); the server keeps running and the
GPU memory is released (verified on the real GPU: 685 MiB in use afterwards, 597 before). Prediction halves its batch
size on out-of-memory and finally continues on the CPU. CPU fallback: `HTR_DEVICE=cpu` (server-wide) or `device` in the
request; training on the CPU works but is much slower. Other knobs: `HTR_TRAIN_WORKERS` (data-loader processes, default
0) and `HTR_PREDICT_BATCH_SIZE` (default 4).

### Proof of the loop on synthetic data (development only; NOT real handwriting)

`scripts/make_synthetic_project.py` (see its docstring; `make synthetic-project` and `make synthetic-report`) renders 30
pages with the three script fonts, sizes 36-60 px, ruled and unruled paper, margin notes and a vocabulary with capitals,
digits, punctuation and abbreviations (`c/o`, `SOB`, `5mg`, ...), imports them through the API, segments them with the
**real** Kraken segmenter, matches each segmented line to the rendered line it covers (351 rendered, 351 segmented, 350
matched) and marks 170 random lines as ground truth with their true text. Run on the GTX 1050 Ti with default settings:

| | start model (Rev City base) | after fine-tuning (15 epochs) |
|---|---|---|
| held-out lines (17 lines, 885 characters): CER | 28.9% | **4.0%** |
| held-out lines: WER | 66.7% | **17.2%** |
| the 180 never-labeled lines (true text known): CER / WER | 29.9% / 68.6% | **3.1% / 12.4%** |

Training: 153 train + 15 validation lines, 15 epochs, 12.3 s per epoch (189 s in total), validation accuracy (1 - CER)
89.8% after epoch 1 and 99.4% at the best epoch (13); peak PyTorch memory 1.3 GB. Prediction of 181 lines: 4.2 s, mean
confidence 0.963 (base model: 0.783). Example predictions (true -> predicted, confidence): `with, who.` -> `with, wto.`
(0.84); `Said are (also) Mrs.` -> `said are (also Mrs.` (0.84); `ok now. or, most. 40 In` -> `&t now. or, most. 4o M` (0.86);
`make was, And it (much) Has been most. you has an (years)` -> exact (1.00). These lines come from three fonts and a
vocabulary of about 370 word forms, which a 4M-parameter network learns quickly. **The numbers only show that the pipeline works end to
end and that fine-tuning moves the model towards its data. They say nothing about how accurate the model will be on real
handwriting**; that needs your own ground truth (the plan's M4 "done when" is about ~100 of your own labeled lines, a step
that is still ahead).

### Tests for M4

`make test` covers CER/WER, the split (determinism, no side-switching across rounds, the at-least-one-held-out rule),
the export, the job lifecycle with fake engines (progress, cancellation, out-of-memory, warnings, never overwriting
ground truth, API errors, registry) and the recognizer's batching and out-of-memory logic. `make test-train` runs the
real thing on a handful of synthetic lines (train + predict through the API, about 15 s on the GPU). `make test-frontend`
and `make test-e2e` include the Training page.

## M5: active learning loop and review queue

The loop: train, **predict every line you have not done**, **review** the predictions least-certain first, corrected
lines become ground truth, train again. Nothing here blocks you: every suggestion is advice.

**Predictions in the editor.** A line that holds an unreviewed prediction shows its confidence as a number, a word and a
mark (`98% high`, `75% medium`, `40% low`; bands at 90% and 70%) in the line list and in the dock, and its outline on the
canvas changes colour *and* pattern (solid, dashed, dotted); a legend sits under the line list. When you select such a
line the text box is prefilled with the prediction and labelled "Predicted text, not yet checked". **Enter accepts it**
(ground truth, next line); edit and press Enter to correct it; **Esc restores the prediction**. A prefilled prediction is
not an unsaved edit: the leave-page guard stays quiet until you change the text.

**Review page** (`#/review`, buttons in the library and editor headers): one line at a time by ascending confidence, with
a large crop, the editable prediction, its confidence and the model that made it, lines left in the queue, an "Open page in
editor" link and this session's counts. Keyboard only:

| Key | Action |
|---|---|
| Enter | accept the prediction as is (or save your correction) and load the next line |
| Tab | skip (the line stays out of this session's queue until "Reset skipped") |
| Ctrl+I | ignore the line (recorded as such) |
| Esc | restore the prediction |

**Correction effort.** Every prediction remembers which model made it. When a predicted line becomes ground truth the
server records, in the same transaction as the change, whether it was *accepted* unchanged, *corrected* (with the edit
distance in characters) or *ignored* (`review_events`, `GET /review/stats`). The Training page charts, per round (= per
model whose predictions were reviewed): the share accepted unchanged and the mean edits per reviewed line, next to the CER
chart, with a table of the numbers. A suggestion banner ("Retraining suggested") appears in the library, the review page
and the Training page when there are at least 50 new ground-truth lines since the last round (or at least 100 in total
before the first one). **Run round** on the Training page trains and then predicts all non-ground-truth lines with the new
model (`POST /jobs/round`; both steps show progress and can be cancelled; a failed or cancelled training step ends the
round and says so; a model that is worse than its start model is kept but not activated, so no prediction is started).
Re-predicting overwrites only older predictions, never ground truth or text you typed.

### Simulated loop on synthetic data (development only; NOT real handwriting, NOT a human)

`make synthetic-rounds` (`scripts/make_synthetic_rounds.py`; setup in its docstring) drives the real API on the real GPU:
36 synthetic pages (real segmentation; 417 lines matched to their true text), 60 matched lines held back and never
labeled or reviewed (a fixed evaluation pool), 40 lines labeled to start, then 3 rounds of {`/jobs/round` (10 epochs),
a scripted reviewer that takes the next 50 lines from `/review/next` in the order the app serves them and accepts a line
whose prediction equals the true text or corrects it}. Seeds are fixed and nothing was tuned between runs; this is the
single run reported.

| Round | GT lines at training | Held-out CER (lines) | Held-out WER | Fixed-pool CER / WER (60 lines) | Accepted unchanged | Edits / reviewed line | Edits / char | Errors in the first 20 queue lines: by confidence vs random order |
|---|---|---|---|---|---|---|---|---|
| 1 | 40 | 12.9% (3; start model 34.7%) | 37.5% | 15.9% / 48.9% | 0 of 50 | 11.74 | 0.280 | 20/20 vs 17.9/20 (90% of the queue is wrong) |
| 2 | 90 | 7.3% (6; start 20.5%) | 26.7% | 6.5% / 23.9% | 0 of 50 | 4.68 | 0.096 | 20/20 vs 14.2/20 (72%) |
| 3 | 140 | 4.8% (15; start 10.4%) | 20.0% | 2.6% / 11.9% | 9 of 50 (18%) | 1.60 | 0.037 | 18/20 vs 9.4/20 (47%) |

Training took 35, 70 and 102 s (a whole round 45, 79 and 112 s). What this shows, and what it does not:

- CER and review effort both fall from round to round in this run. The **fixed-pool CER** (the same 60 never-labeled
  lines each time) is the cleanest comparison: 15.9% to 6.5% to 2.6%. The held-out CER also falls, but it rests on 3, 6 and
  15 lines and the held-out set *grows* as ground truth grows, so those numbers are noisy and not strictly like for like.
- "Edits per reviewed line" is confounded: the reviewer always takes the *least certain* lines, which are the hardest
  ones, so it understates the improvement on the average line. The accepted-unchanged share stays at 0% for two rounds
  because the least-certain 50 lines were all wrong; it only rises once the model is good (18% in round 3).
- Low-confidence-first finds errors faster than random order, increasingly so as the model improves: all 20 of the first
  20 queue lines needed a correction in rounds 1 and 2 (random order would meet 17.9 and 14.2 on average), 18 of 20 in
  round 3 (random: 9.4). In round 1 almost everything was wrong, so ordering barely mattered.
- Tests: `make test` covers queue order, exclude, review events (accepted / corrected / ignored, one per prediction,
  rolled back with a failed save), the stats, chained rounds (success, failed step, cancel), the retrain thresholds and
  never overwriting ground truth or typed text; `make test-frontend` the editor prefill/accept/Esc semantics and guard,
  the review page keyboard flow, the confidence indicator and both charts; `make test-e2e` includes
  `zz-review.spec.ts` (needs `make base-model`): accept a prediction in the editor, then a 3-line review session
  (accept, correct, ignore), verified through the API.
- This is synthetic text from three fonts and about 370 word forms, with a reviewer who never errs. It proves the
  machinery works end to end (rounds, queue order, events, statistics, charts). It does **not** show that review effort
  will fall for a real writer; that needs real rounds with your own lines.

## M6: abbreviations, export and project bundle

Recognition and transcription stay **diplomatic**: text is stored exactly as written (`pt c/o SOB`). Abbreviations are a
separate layer: a personal dictionary that is applied on demand, for the raw / expanded text view and the expanded export.
Expanded text is never stored.

### Abbreviation dictionary (`#/abbreviations`)

Add, edit and delete entries (written form, expansion, optional notes, "case sensitive"), search them, filter to the
ambiguous ones and sort by occurrences. The occurrence count is live (tokens in the text of every non-ignored line,
predicted lines included); click it to list the lines, each linking into the editor on that line
(`#/pages/<id>?line=<line id>`). "Try the dictionary" previews an expansion. A pair (written form, expansion) is unique;
**several expansions for one written form make it ambiguous** (badge in the table, `ambiguous` + `candidates` in the API).

**Expansion rules** (implemented in `backend/app/abbreviations.py`, pure and unit-tested in `tests/test_abbreviations.py`):

| Rule | Behaviour |
|---|---|
| Tokens | The text is split on whitespace; every run of non-space characters is a token. A token is matched **whole**: an entry is never expanded inside a longer word (`ptosis`, `pt-centered`, `SOBx`, `w/o` do not match `pt`, `SOB`, `w/`). |
| Punctuation | Candidates are tried longest first: the whole token, then fewer characters with trailing punctuation (`, ; : ! ? ) ] } " ' .`) stripped before leading punctuation (`( [ { " '`). So entries that contain punctuation themselves (`c/o`, `w/`, `s.o.b.`, `b.i.d`) match as written, `SOB.` matches `SOB`, and `b.i.d.` falls back to `b.i.d`. Stripped punctuation stays in the output (`(pt),` becomes `(patient),`). |
| Case of the match | Case-insensitive, unless the entry is marked case sensitive: then the token must equal the written form exactly. |
| Case of the output | The token equals the written form: the expansion as stored. A short ALL-CAPS token (2 to 4 letters, no lowercase): the expansion as stored (`SOB` becomes `shortness of breath`). Otherwise a token that starts with an uppercase letter: the expansion with its first character uppercased, the rest untouched (`Pt` becomes `Patient`, `APPROX` becomes `Approximately`). Otherwise the expansion as stored. |
| Ambiguous | More than one distinct expansion matches a token: it is **not** expanded. It is reported with its candidates (`status: ambiguous`), left as written in text (`ambiguous=raw`, default) or marked `SOB[?]` (`ambiguous=mark`). Two rows with the same expansion are not ambiguous. Case-sensitive rows with different spellings (`US`, `us`) do not conflict. |
| Single pass | Left to right, once. Expansions are never re-scanned, so an expansion that contains another entry's written form is not expanded again; expanding already expanded text changes nothing unless an expansion is itself an entry. |
| Spans | Every result carries `start`, `end`, `written`, `expansion` and `status` (`expanded` / `ambiguous` / `unmatched`) per token, offsets into the input. |

**Text view** (`#/pages/<id>/text`, "Text view" button in the editor): the page's lines with the raw text (matches
highlighted, ambiguous ones in orange) next to the expanded text; options for marking ambiguity and for showing predictions.

### Export (`#/export`, `GET /export`)

- **Plain text**: UTF-8, one block per page (`=== <document> / page <n> (<file>) ===`, `n` = position in the document by import
  order), lines in reading order, a blank line between pages. Ignored and empty lines are never exported, pages without text
  are left out. **Predicted lines are excluded by default**; with `include_predicted` they are included and each is prefixed
  `[predicted] `. `expanded=true` applies the dictionary; `ambiguous=raw|mark` decides what happens to ambiguous tokens.
- **PAGE-XML**: a zip with one file per segmented page, rendered from the database by the same writer as the project's own
  files (so text, `ground_truth` / `ignored` / `text_source` flags and confidence are included). Without
  `include_predicted` the text and confidence of predicted lines are left out.
- File names are generic (`htr-<document>-expanded.txt`, `htr-all-pagexml.zip`); no path of your machine appears in the
  file or the header.

### Project bundle (`.htrbundle`)

One zip file with everything needed to continue the work on another machine or after a disaster. **It contains your scans and
all transcriptions, so it contains personal data**: it is written only where you choose (the browser's save location, or the
path you give the CLI) and nothing is ever uploaded. Layout (paths mirror the data directory, so the relative paths stored in
the database stay valid):

```
manifest.json            written last (it holds the checksums of everything before it)
db/htr.sqlite            consistent snapshot (sqlite3 backup API, never a copy of the live file)
originals/...            the untouched scans, byte-identical
working/...              working copies (what the line coordinates refer to)
pagexml/...              PAGE-XML, rendered from the snapshot so it matches the database exactly
models/<id>.mlmodel      fine-tuned models
models/base/<name>/...   the pretrained base models that model rows point at (model file + BASE_MODEL.json)
```

The database already stores every file path relative to the data directory (`originals/...`, `working/...`,
`pagexml/...`, `models/...`), so nothing has to be rewritten on import and a bundle restores into any data directory; the
import still refuses any path that is absolute or leaves those folders.

Not included: overlays, import reports, `training/` run folders (regenerable), temporary files. The database part carries
documents, pages, lines (text, flags, geometry, order), `line_splits` and `model_lines`, models and the active flag, jobs
(history), review events and abbreviations.

```json
{
  "format": "htr-bundle",
  "bundle_format_version": 1,
  "schema_version": 4,
  "app_version": "0.3.0",
  "created_at": "2026-10-26T12:00:00.000000Z",
  "counts": {"documents": 1, "pages": 10, "lines": 30, "ground_truth_lines": 24, "models": 2, "finetuned_models": 1,
             "abbreviations": 4, "review_events": 3, "jobs": 12, "files": 33},
  "missing_files": [],
  "files": [{"path": "db/htr.sqlite", "size": 151552, "sha256": "..."}, {"path": "originals/page_1658c3e4.png", "size": 344641, "sha256": "..."}],
  "manifest_sha256": "..."
}
```

**Export** (`GET /bundle/export`, **Export project bundle** on the Export page, `python scripts/htr.py bundle export <file>`):
refused with **409** while a segment, train or predict job is queued or running (the message names the job). Otherwise the
zip is streamed in chunks (never built in memory); the browser saves the stream straight to a file. A model or base-model
file that is missing on disk is listed under `missing_files` instead of failing; a missing scan or working copy refuses
the export. The CLI writes `<file>.part` and renames it, and never overwrites an existing file; it reads through the SQLite backup
API, so it can run while the server is up.

**Import** (`POST /bundle/import` multipart `file`, **Import project bundle** on the Export page,
`python scripts/htr.py bundle import <file>`):

- **Only into an empty project; v1 never merges.** Empty means the database has no documents, pages, lines, fine-tuned
  models, abbreviations, review events and jobs. The base-model rows created at startup do not count. Anything else is
  refused with **409** naming what is there (`the project is not empty (3 documents, 10 pages)`), and nothing is touched.
- The bundle is validated completely **before** the project is touched, in a staging directory inside the data directory:
  zip entries must be regular files with relative, forward-slash paths without `..` and only under `db/ originals/ working/
  pagexml/ models/` (zip-slip, absolute paths, symlinks, duplicates, unlisted or missing entries are refused); every file's
  size and SHA-256 must match the manifest, the manifest must match its own checksum, its counts and `schema_version` must
  match the database, the database must pass `integrity_check`, contain only tables and indexes, have only relative paths
  inside the allowed folders and every referenced file must exist. Damaged, truncated or tampered bundles give **422**.
- `schema_version` newer than this app: refused (**422**, "made by a newer version of the app"). Older: the restored
  database is migrated with the normal migration runner before it is swapped in.
- The swap: files are moved into the data directory (a file that already exists with identical content is left alone; a
  different file at the same path is a conflict and nothing is overwritten), then the database file is replaced by two
  renames. Every step is recorded in an undo log: if anything fails, the added files are removed and the old database is
  put back, so the project is exactly as it was.
- **Running work:** the import is refused with 409 while a job is queued or running. While it runs the server is in
  maintenance mode: other requests get **503** (`/health` stays up), the background worker is stopped and restarted after the
  swap, and the import waits (at most 15 s) for requests that were already running (for example a bundle download);
  if they do not finish it gives up with 409. The command-line import takes an exclusive lock on the data directory, so it
  refuses (and tells you to stop the server) while a server is running there.
- After an import the app works immediately: pages, lines, ground-truth and ignored flags, models and the active model,
  abbreviations, review events, job history and PAGE-XML are exactly the exported ones (ids included).

Large bundles: the upload is spooled to disk by the server (not held in memory) and extracted file by file; you need free disk
space for about twice the bundle size while it is imported.

### Tests for M6

`make test` covers the expansion engine (`test_abbreviations.py`), the dictionary / expansion / export API
(`test_api_abbreviations.py`, `test_api_export.py`) and the bundle (`test_bundle.py`): the round trip through the real API
into a fresh data directory with every table, every project file byte for byte, PAGE-XML in sync and the API output compared;
a bundle from a schema-3 database migrated on import; refusals for a non-empty project, running jobs, corrupted checksums,
tampered manifests, a newer schema or bundle format, zip-slip / absolute / symlink entries, hostile database paths,
truncated zips; faults injected at every step of the import (the project must be untouched afterwards); streaming in
chunks; a WAL-consistent snapshot; maintenance mode; and the CLI. `make test-train` additionally runs the real thing
(`test_bundle_real.py`: real fine-tune, bundle, import into a directory without any base model, real prediction with the
imported model giving the same texts). `make test-frontend` and `make test-e2e` cover the three new pages; the e2e test
adds abbreviations, exports expanded text, exports a bundle through the UI, sees the UI refuse an import into the
non-empty project and imports the bundle into a second, fresh backend that it compares through the API.

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
(`kraken get`) is only relevant for recognition models (M4, below). `make test-kraken` runs a real segmentation of a
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

Details and rules: [docs/api.md](docs/api.md). Training, prediction and model endpoints arrived in M4 (below).
Abbreviations, export and the project bundle arrived in M6 (below).

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
manually transcribed lines unless `force`), `train` and `predict` (M4). Segmentation points are clamped to the page.
A running `train` or `predict` job can be cancelled (cooperatively); a running `segment` job cannot.

### Tests

`make test` runs the API tests with Kraken mocked (fast, no GPU). `make test-kraken` runs the real-model tests,
including `tests/test_kraken_api.py`: import a synthetic page through the API, segment it with real Kraken via the
job queue, and read the lines back (run with `-s` to see the printed lines).
