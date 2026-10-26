# Setting up HTR on Windows (WSL2)

This page takes a clean Windows machine to a running HTR in your browser. Every step was followed from a fresh
`git clone` while writing it (see "How this page was validated" at the end). Nothing in HTR uploads anything: the
steps marked **needs the network** are one-time downloads; once they are done the app runs fully offline.

## 0. What you need

| | |
|---|---|
| Windows | 10 (21H2+) or 11, with WSL2 |
| GPU (recommended) | an NVIDIA card with 4 GB or more (tested: GTX 1050 Ti, 4 GB) and the normal Windows driver. Without a GPU everything works on the CPU, only training and prediction are much slower |
| Disk | about 12 GB free inside WSL (PyTorch with CUDA is ~5 GB unpacked, plus models and your scans) |
| RAM | 16 GB recommended (8 GB works for small batches) |

## 1. Windows side (once)

1. Install WSL2 with Ubuntu: open PowerShell as administrator and run `wsl --install -d Ubuntu`, then reboot if asked
   and create your Linux user. *(Needs the network.)*
2. Install the current **NVIDIA driver for Windows** (GeForce Experience or nvidia.com). It provides CUDA to WSL2.
   Do **not** install a Linux NVIDIA driver or the CUDA toolkit inside Ubuntu: the PyTorch wheels ship their own CUDA.
3. In Ubuntu, check that the GPU is visible: `nvidia-smi` must print your card. If it does not, see "Troubleshooting".

## 2. Ubuntu packages (once)

```bash
sudo apt update
sudo apt install -y git curl make xz-utils build-essential
```

`make` runs the project commands, `xz-utils` is needed to unpack Node, `build-essential` is a safety net for Python
packages that compile.

## 3. Get the code (once)

Work on the **Linux** filesystem (`~/HTR`), never under `/mnt/c`: file access there is slow and breaks file locking.

```bash
git clone <the repository URL> ~/HTR
cd ~/HTR
```

## 4. Tools: uv, Python 3.11, Node 22 (once; needs the network)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # uv, the Python package manager
export PATH="$HOME/.local/bin:$PATH"                    # add this line to ~/.bashrc as well
make setup-node                                         # Node 22 LTS into ~/.local/node (no sudo)
node --version                                          # v22.x
```

Python 3.11 is installed by uv itself in the next step. Do not use the system Python: Kraken needs 3.11 exactly, and
Ubuntu may ship a newer one.

## 5. Install the project (once; needs the network)

```bash
make setup            # backend: Python 3.11 + pinned PyTorch (CUDA 12.6), Kraken, FastAPI. A few GB, several minutes
make setup-frontend   # frontend: npm packages from the lockfile + a headless Chromium used by the e2e test
make base-model       # the pretrained recognition model (~16 MB) into data/models/base
make fonts            # OPTIONAL: three open-licence script fonts for the synthetic test pages
make check            # verifies WSL2, a real CUDA operation, Kraken and free disk
```

`make check` ends with a table of OK / WARN / FAIL lines. WARN items do not stop you. If a line says FAIL, the message
says what to fix; "Troubleshooting" below covers the usual ones.

Why CUDA 12.6 wheels? Newer PyTorch CUDA builds dropped the older "Pascal" GPUs (GTX 10xx, compute capability 6.1).
The 12.6 wheels still run on them; `make check` proves it with a real GPU calculation.

## 6. Run it

```bash
make dev
```

This starts the API on `http://127.0.0.1:8000` and the web app on `http://127.0.0.1:5173`. Open
**http://127.0.0.1:5173** in your Windows browser. (`localhost:5173` works too.) Press Ctrl+C in the terminal to stop
both. The strip at the top of every page shows whether the app is healthy; click it to see the self-checks (database,
disk space, GPU, model, worker).

Only those two addresses work. The server refuses requests that name any other host (this blocks web pages that try
to reach your local app through DNS tricks), and it listens on 127.0.0.1 only, so other computers cannot connect.

## 7. First use: from scans to text

1. **Import.** On the *Library* page create a document ("New document name" and *Create document*), then drop scan
   files onto its drop zone or pick them with its file chooser. PNG, JPEG
   and single-page TIFF are accepted (up to 100 MB each and 80 megapixels; a multi-page TIFF must be split into one
   image per page first). The original is stored untouched; a working copy is straightened and size-normalised. Files
   that were imported before are recognised and skipped. Any quality warnings (low resolution, odd size) are shown
   with the page.
2. **Segment.** Press *Segment* next to a page. The first run loads the model and takes a few seconds; then the
   page's lines appear. Open the page with *Transcribe*. Press `?` in the editor for all keyboard shortcuts.
3. **Repair the lines** where needed: merge two lines, split one, redraw a baseline, add a missed line, ignore
   crossed-out text or doodles (single line or a drawn region), reorder. `Ctrl+Z` undoes a repair.
4. **Transcribe.** Type what the line crop shows, exactly as written, abbreviations included. `Enter` saves the line
   as ground truth and moves to the next. About **100 lines** are recommended before the first training; fewer works,
   you are only warned.
5. **Train.** On the *Training* page press *Train* (batch size 4 is right for a 4 GB card; lower it if the GPU runs
   out of memory). The app fine-tunes the pretrained model on your ground-truth lines and reports the error rate on
   held-out lines before and after. Then *Predict* fills every other line with a prediction and a confidence.
6. **Review.** The *Review* page serves the least confident predictions first. `Enter` accepts a prediction as it is;
   edit then `Enter` to correct it, `Tab` skips a line. After about 50 new lines, run another round (*Run round (train + predict)* on the Training
   page); the effort charts show whether reviewing is getting cheaper.
7. **Abbreviations.** The *Abbreviations* page holds your personal dictionary (`pt` = patient, `c/o` = complains of).
   Transcriptions stay exactly as written; the dictionary only drives the expanded views and exports.
8. **Export.** The *Export* page writes plain text (as written, or with abbreviations expanded) and PAGE-XML.
9. **Back up.** *Export project bundle* on the Export page writes one `.htrbundle` file with everything (scans,
   transcriptions, dictionary, models, history). *Import* it into an empty project to restore. Do this regularly and
   keep the file somewhere safe: **a bundle contains all your scans and text**, so treat it like the originals.

## 8. Where your data lives

Everything is in the `data/` folder of the project (git-ignored; override with `HTR_DATA_DIR`):

| Path | Contents |
|---|---|
| `data/htr.db` | the database (documents, lines, dictionary, jobs, model registry) |
| `data/originals/`, `data/working/` | your scans, untouched, and the working copies the editor shows |
| `data/pagexml/` | one PAGE-XML file per segmented page, kept in sync with the database |
| `data/models/` | the pretrained base model and the models you trained |
| `data/training/` | per-run training exports (safe to delete when no job is running) |
| `data/logs/htr.log` | the app log (rotating, 5 x 1 MB); never contains your text |
| `data/tmp/` | uploads in progress |

You can inspect these in Windows Explorer under `\\wsl$\Ubuntu\home\<you>\HTR\data`, but do not edit them while the
app runs.

## 9. Privacy

* **No outbound connections.** The app makes none; this is tested (`make test` fails if any code tries to reach a
  non-loopback address) and checked end to end with `make verify-offline`, which runs the whole import, segment, train,
  predict flow inside a network namespace that has no network at all.
* **Setup-time downloads only** (the rows marked "needs the network" above): apt packages, uv and Python, PyTorch and
  the other Python wheels, Node, npm packages, Chromium for the e2e test, the base model (Zenodo, via `make
  base-model`) and optional fonts (`make fonts`). None of them sends your data anywhere, and none is used while you
  work.
* The server binds to 127.0.0.1, answers only a loopback `Host` header, and allows cross-origin calls only from the
  two Vite addresses.
* The log file holds ids, counts, timings and error codes, never transcribed text, predictions, dictionary entries or
  scan file names.
* The optional language-model suggestions (below) talk only to a model server on the same computer, and are off unless
  you turn them on.
* **Bundles, `data/` and exports are personal data.** The repository ignores `data/` and `*.htrbundle`; do not copy
  them into cloud-synced folders, e-mail or chat unless you mean to.

## 10. Optional: suggestions from a local language model

Off by default. When enabled, the app can ask a model **running on your own computer** what an unfamiliar abbreviation
probably stands for, using only the abbreviation and its line (plus at most two neighbouring lines, each shortened to 300
characters). The answers appear on the *Abbreviations* page as suggestions you accept, edit or reject; nothing is ever
applied on its own and no transcription is changed. A rejected suggestion is remembered and not offered again.

This repository does **not** install or download any model. If you want to try it, one possible set-up (Ollama):

```bash
curl -fsSL https://ollama.com/install.sh | sh     # needs the network and sudo; installs Ollama as a local service
ollama pull llama3.2:3b                           # one-time model download (about 2 GB, needs the network)
ollama serve                                      # if the service is not already running; listens on 127.0.0.1:11434
```

Then start HTR with the feature switched on:

```bash
HTR_LLM_ENABLED=1 HTR_LLM_MODEL=llama3.2:3b make dev
```

and press *Suggest abbreviations* on the Abbreviations page. A llama.cpp server also works:
`llama-server -m model.gguf --port 8080` and `HTR_LLM_ENABLED=1 HTR_LLM_API=openai HTR_LLM_URL=http://127.0.0.1:8080`.
`HTR_LLM_URL` must be a loopback address (`127.0.0.1`, `localhost`, `::1`); anything else is refused at startup.
A 4 GB GPU is shared with training, so run suggestions when no training job is running.

*Honest status:* the connection, prompts, parsing, review flow and the "off means no connection" guarantee are covered
by tests against a fake server that mimics Ollama and llama.cpp. The behaviour of a real model (quality of the
suggestions, speed on your hardware) was not tested here.

## 11. Settings

Environment variables, read when the server starts:

| Variable | Default | Meaning |
|---|---|---|
| `HTR_DATA_DIR` | `<project>/data` | where everything is stored |
| `HTR_DEVICE` | `auto` | `auto`, `cpu` or `cuda:0`. If the GPU is not usable, jobs fall back to the CPU and say so |
| `HTR_TRAIN_BATCH_SIZE` / `HTR_PREDICT_BATCH_SIZE` | `4` / `4` | lines per step; lower them on GPU out-of-memory |
| `HTR_MAX_UPLOAD_MB` | `100` | largest accepted scan file |
| `HTR_MAX_IMAGE_MP` | `80` | largest accepted scan in megapixels (A4 at 300 DPI is about 8) |
| `HTR_LLM_ENABLED` | off | turn on local-LLM suggestions |
| `HTR_LLM_URL` / `HTR_LLM_API` / `HTR_LLM_MODEL` | `http://127.0.0.1:11434` / `ollama` / `llama3.2:3b` | the local model server (`ollama` or `openai`) |
| `HTR_LLM_TIMEOUT` / `HTR_LLM_CONTEXT_LINES` | `120` / `1` | seconds per request; neighbouring lines sent each side (0-2) |

## 12. Troubleshooting

**`nvidia-smi` does not work inside Ubuntu / "No usable GPU" in the status strip.** Update the NVIDIA driver on
*Windows* (not in Ubuntu), then run `wsl --shutdown` in PowerShell and open Ubuntu again. Check `wsl --version` shows
WSL 2 and that `nvidia-smi` works in Windows. Without a GPU the app still works on the CPU, with a visible notice.

**`make check` says the GPU architecture is not supported / "no kernel image".** Use the pinned wheels from `make
setup`. They are built for CUDA 12.6, the last PyTorch CUDA line that still includes the older Pascal cards (GTX 1050
Ti, compute capability 6.1). If you installed PyTorch some other way, run `make setup` again.

**"database or disk is full", or tests fail strangely.** Check `df -h / /tmp`. WSL's `/tmp` can be a small RAM-backed
filesystem; test runs now clean up after themselves, but older runs may have left data: `make clean-test` removes it.
You can also point temp files elsewhere: `export TMPDIR=$HOME/tmp; mkdir -p $TMPDIR`. The data drive's free space is
also shown in the status strip.

**"Address already in use" on port 8000 or 5173.** Another copy is running. Find it with `ss -ltnp | grep -E
':(8000|5173)'` and stop it (often a forgotten `make dev` in another terminal). The web app must stay on 5173 because
the API only accepts calls from there; the API port can be changed by starting it yourself:
`cd backend && uv run uvicorn app.main:app --host 127.0.0.1 --port 8001` and, in another terminal,
`cd frontend && VITE_API_BASE_URL=http://127.0.0.1:8001 npm run dev`.

**Training stops with "the graphics card ran out of memory".** Lower the batch size on the Training page (or set
`HTR_TRAIN_BATCH_SIZE=2`), close programs that use the GPU (browsers with hardware acceleration use some), or run with
`HTR_DEVICE=cpu`. Prediction halves its batch size by itself and finally falls back to the CPU. A 4 GB card needs batch
size 4 or less.

**The page says "server offline".** The API is not running or crashed: start it with `make dev` (or `make serve`)
and press *Retry*. Everything you saved is in `data/htr.db`. The reason for a crash is in `data/logs/htr.log`.

**"This request was refused ... Host header".** You opened the app by an address other than 127.0.0.1 or localhost
(for example the WSL IP address). Use `http://127.0.0.1:5173`.

**An import is refused.** The message says why: damaged or cut-off file, wrong extension for the content, more than one
page (split multi-page TIFFs, for example with ImageMagick: `magick scan.tif page-%03d.png`), or too many megapixels.

**"another process holds the data directory lock".** A second server is using the same `data/` folder. Stop it first.

**Chromium cannot start for `make test-e2e`.** `make setup-frontend` unpacks the few missing system libraries without
sudo; run it again, or `sudo npx playwright install-deps chromium`.

## 13. Everyday commands

| Command | What it does |
|---|---|
| `make dev` | API and web app together |
| `make check` | environment check (GPU, Kraken, disk) |
| `make test`, `make test-frontend`, `make lint`, `make lint-frontend` | the automated tests and linters |
| `make test-e2e` | full browser test with the real segmenter (slow) |
| `make test-kraken`, `make test-train` | real Kraken segmentation test; tiny real fine-tune test (needs `make base-model`, `make fonts`) |
| `make verify-offline` | proves the app needs no network (static audit + offline run of the real flow) |
| `make clean-test` | deletes temp files left by test runs |

## How this page was validated

The steps above were followed literally in a new directory from a fresh `git clone` of the repository, with a new
Python virtual environment and a new `node_modules`. What that did and did not cover is recorded in the milestone
report; it was not possible to test a brand-new WSL distribution or a machine without an NVIDIA driver.
