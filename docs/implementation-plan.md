# HTR: Implementation Plan

A local-first handwritten text recognition (HTR) tool for one writer's scanned notes. The user transcribes a small number of lines, the system learns that writer's hand, and it predicts the rest. Nothing leaves the machine.

Status: **draft for review**

---

## 1. Goals and non-goals

### Goals
- Import a few hundred scanned note pages (images, 300 DPI, clean).
- Segment each page into text lines automatically.
- Let the user transcribe lines in a browser UI.
- Fine-tune a recognition model on the user's corrections and predict the remaining lines.
- Prioritize review of low-confidence predictions (active learning).
- Keep abbreviations as written, with a personal dictionary that expands them on request.
- Run entirely locally (WSL2 + NVIDIA GPU). No cloud OCR or public LLM services.

### Non-goals (for now)
- Multi-user accounts, teams, billing, sharing.
- Multiple writers or per-writer model management.
- Vertical or upside-down text, orientation detection.
- Non-English languages.
- Table, form or diagram understanding.
- Mobile or desktop-native app.

### Assumptions
- Single writer, English, horizontal text, upright pages (arbitrary small skew is possible).
- Lined or unlined paper; notes in no fixed order.
- Pages are clean 300 DPI scans.
- A few hundred pages; fine-tuning, not training from scratch.
- Sensitive content (some pages are medical notes), so all processing is local.

---

## 2. Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| Training unit | **Text line** (image + text) | Standard for HTR; cursive cannot be reliably split into letters. Word boxes are a UI aid only. |
| Transcription policy | **Diplomatic**: text exactly as written | Keeps recognition honest and debuggable. Abbreviation expansion is a separate layer. |
| Interchange format | **PAGE-XML** | Standard for layout and transcription; Kraken reads and writes it. |
| Segmentation + recognition | **Kraken** (baseline segmenter + CRNN recognizer) | Mature, fine-tunable, runs locally, supports baseline polygons. |
| Backend | Python, FastAPI | Same language as the ML stack. |
| Metadata store | SQLite | Single user, local, zero setup. |
| Storage | Plain files on disk | Images and PAGE-XML are easy to inspect and back up. |
| Frontend | React + TypeScript + Vite, canvas viewer | Needed for polygon overlays and fast line-by-line editing. |
| Jobs | In-process worker with a SQLite-backed queue | Avoids Redis for a single-user app; can swap later. |
| Abbreviation expansion | Personal dictionary first, optional local LLM later | Reviewable and offline. |

Fallback recognizer if Kraken fine-tuning underperforms: **TrOCR** (HuggingFace) on line crops. The pipeline is built behind a `Recognizer` interface so the engine can be swapped.

---

## 3. Architecture

```
Browser (React) <--HTTP/JSON--> FastAPI <--> SQLite
                                   |
                                   +--> data/ (images, PAGE-XML, models)
                                   |
                                   +--> Worker (segment / predict / train)
                                            |
                                            +--> Kraken + PyTorch (CUDA)
```

### Repo layout (proposed)
```
HTR/
  README.md
  docs/
    implementation-plan.md
  backend/
    app/
      main.py            # FastAPI entrypoint
      api/               # routers: pages, lines, jobs, models, abbreviations
      db/                # SQLite schema, migrations
      pipeline/
        preprocess.py    # normalize image, deskew
        segment.py       # Kraken segmentation
        recognize.py     # Recognizer interface + Kraken impl
        train.py         # fine-tuning
        evaluate.py      # CER/WER
        pagexml.py       # read/write PAGE-XML
      worker/            # job queue + runner
      abbreviations.py
    tests/
    pyproject.toml
  frontend/
    src/
      components/        # PageViewer, LineEditor, ReviewQueue, ...
      pages/
    package.json
  data/                  # gitignored: images, xml, models, db
  scripts/               # setup, import, CLI utilities
  Makefile
```

`data/` is gitignored. Personal scans never go into the repo.

### Data model (SQLite)
- `documents`: id, name, imported_at
- `pages`: id, document_id, image_path, xml_path, width, height, status
- `lines`: id, page_id, index, baseline (json), polygon (json), text, text_source (`manual` | `predicted`), confidence, is_ground_truth, ignored
- `models`: id, path, base_model, trained_at, train_lines, cer
- `jobs`: id, type, status, payload, progress, error, created_at
- `abbreviations`: id, written, expansion, notes, count

PAGE-XML on disk is kept in sync with the DB; PAGE-XML is the export/import format and the training input.

---

## 4. Environment

- **WSL2** (Ubuntu) with the NVIDIA CUDA driver passthrough.
- Python 3.11+, PyTorch with CUDA, Kraken.
- Node 20+ for the frontend.
- Everything runs on `localhost`; the server binds to `127.0.0.1` only.
- A `scripts/check_env.py` verifies GPU visibility, Kraken install, and disk space.

---

## 5. Pipeline details

### 5.1 Import and preprocessing
- Accept PNG/JPG/TIFF. Store originals untouched; derive a working copy.
- Convert to grayscale/normalize contrast. Optional binarization for segmentation only.
- Light deskew (small angles only). No orientation detection.

### 5.2 Segmentation
- Kraken baseline segmentation (`blla`) produces baselines and polygons per line.
- Reading order: top-to-bottom by default; user can reorder or ignore lines in the UI.
- Known weak spots to surface in the UI: margin notes, arrows, doodles, crossed-out text. Users can mark lines as ignored or edit/redraw baselines.

### 5.3 Recognition
- Start from a pretrained Kraken English/Latin handwriting model if one is usable; otherwise start from the general model and fine-tune.
- Output per-line text and a confidence score.

### 5.4 Fine-tuning
- Training set = lines flagged `is_ground_truth`.
- Hold out ~10-15% of ground-truth lines for evaluation; fixed split so CER is comparable across rounds.
- Augmentation: mild rotation, scale, brightness, blur.
- Each round records the model, training size and CER so progress is visible.

### 5.5 Active learning
- After each round, predict all non-ground-truth lines.
- Review queue ordered by ascending confidence.
- Corrected lines become ground truth; retrain when enough new lines accumulate (user-triggered, with a suggested threshold).

### 5.6 Abbreviations
- Recognition output stays diplomatic (`pt c/o SOB`).
- Dictionary maps written forms to expansions. UI shows raw and expanded text side by side.
- Expansion is rule-based (whole-token match, case-aware). Ambiguous tokens (multiple expansions) are flagged for the user.
- Later (optional): local LLM via Ollama/llama.cpp suggests expansions from context, shown as reviewable suggestions only.

---

## 6. User interface

1. **Library**: list of documents/pages with status (unsegmented, segmented, partially transcribed, done) and progress.
2. **Page editor** (main screen)
   - Page image with line polygons overlaid; selected line highlighted.
   - Text input for the current line, with the line crop shown above it.
   - Keyboard-first: Enter = save and next line, Tab/Shift-Tab to move, shortcut to ignore a line.
   - Show predictions with a confidence indicator; accepting a prediction is one keystroke.
   - Tools: edit baseline, split/merge lines, reorder, mark region ignored.
3. **Review queue**: lowest-confidence lines across all pages, one at a time.
4. **Training**: start a fine-tune job, see progress, and compare models by CER.
5. **Abbreviations**: view/edit the dictionary; see occurrences.
6. **Export**: plain text (raw or expanded), PAGE-XML, searchable full-text across all pages.

---

## 7. API sketch

- `POST /documents` / `POST /pages` (upload)
- `GET /pages`, `GET /pages/{id}` (lines, image URL)
- `POST /pages/{id}/segment`
- `PATCH /lines/{id}` (text, polygon, ignored, ground truth)
- `POST /jobs/predict`, `POST /jobs/train`, `GET /jobs/{id}`
- `GET /review/next`
- `GET/POST /abbreviations`
- `GET /export?format=txt|pagexml&expanded=bool`
- `GET /search?q=`

---

## 8. Milestones

Each milestone ends in something runnable and reviewable.

**M0: Skeleton and environment**
- Repo layout, `pyproject.toml`, `Makefile`, `.gitignore` (excluding `data/`).
- `check_env.py`; confirm CUDA works under WSL2 and Kraken runs.
- *Done when:* `make check` passes on the user's machine.

**M1: Import and segmentation (CLI)**
- Import images into `data/`, run Kraken segmentation, write PAGE-XML, render an overlay PNG for inspection.
- Run on ~20 sample pages and review segmentation quality with the user.
- *Done when:* the user agrees line detection is good enough on their handwriting, or we know which fixes are needed.

**M2: Backend API + database**
- SQLite schema, import/segment endpoints, line CRUD, PAGE-XML sync, job queue.
- *Done when:* the API can import a page, segment it, and return lines.

**M3: Line-labeling UI**
- Library, page editor, keyboard-driven transcription, line crops, ignore/split/merge.
- *Done when:* the user can transcribe a page end to end in the browser.

**M4: Training, prediction, evaluation**
- Fine-tune job from ground-truth lines, held-out CER, predict job, model registry.
- *Done when:* after ~100 labeled lines, the model produces predictions and CER is reported.

**M5: Active learning loop**
- Confidence-ordered review queue, prediction display in the editor, retrain flow.
- *Done when:* repeated rounds visibly reduce CER and correction effort.

**M6: Abbreviations and export**
- Dictionary UI, expansion view, plain-text and PAGE-XML export, full-text search.
- *Done when:* a full document exports as readable expanded text.

**M7: Hardening**
- Backups, error handling, a small test suite, docs for setup on WSL2, optional local-LLM suggestions.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Segmentation is poor on messy notes (overlaps, margins, slant) | Validate in M1 before building UI; provide manual baseline editing; fall back to a different segmenter if needed. |
| Fine-tuning on a few hundred pages is not accurate enough | Measure CER early (M4); try TrOCR via the `Recognizer` interface; add augmentation; label more lines. |
| Kraken/PyTorch/CUDA setup is fragile under WSL2 | M0 exists to prove this first; pin versions; document the exact setup. |
| Heavy abbreviation use hurts recognition | Keep diplomatic transcription so the model learns the abbreviation shapes directly; expand afterward. |
| Labeling effort is too high | Active learning ordering; prediction acceptance with one key; train after small batches. |
| Privacy leaks via repo or tooling | `data/` gitignored; server binds to localhost; no outbound calls from the app; local LLM only. |

---

## 10. Evaluation

- **CER** (character error rate) on a fixed held-out set of ground-truth lines, reported per model.
- **WER** as a secondary metric.
- **Correction effort**: average edits per line over successive rounds (does review get faster?).
- Segmentation quality is judged by eye on overlays in M1 and by the share of lines the user must fix in the editor.

---

## 11. Open questions for review

1. Is the proposed milestone order acceptable, particularly validating segmentation (M1) before building the UI?
2. Should the first training round wait for a target number of lines (e.g. 100), or let the user trigger it at any time?
3. Should "crossed-out" text be transcribed (e.g. wrapped in a marker), or ignored?
4. Is full-text search across all pages needed in v1, or can it wait?
5. Backups: is copying the `data/` folder enough, or do you want an export/import bundle?
