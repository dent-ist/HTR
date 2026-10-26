# All Python commands run inside the uv project in backend/ (Python 3.11).
UV ?= uv
RUN = $(UV) run --project backend

# Node (version pinned in frontend/.nvmrc) is installed at user level under ~/.local; put it ahead of anything
# inherited from Windows on PATH. Only the *-frontend / dev / e2e targets use it; `make check` does not.
export PATH := $(HOME)/.local/bin:$(HOME)/.local/node/current/bin:$(PATH)

.PHONY: setup check test test-kraken test-train base-model synthetic-project synthetic-report synthetic-rounds fonts lint format serve dev setup-frontend dev-frontend test-frontend lint-frontend test-e2e clean-test verify-offline audit-network setup-node

setup:
	$(UV) sync --project backend --python 3.11

check:
	$(RUN) python scripts/check_env.py

# Temp files: passing tests leave nothing behind (see tmp_path_retention_* in backend/pyproject.toml); everything goes
# to $$TMPDIR (default /tmp). `make clean-test` removes whatever a crashed or failed run left.
test:
	cd backend && $(UV) run pytest

# Remove temp data left by test runs (pytest base directories, e2e roots, offline-smoke directories) and test output.
clean-test:
	rm -rf "$${TMPDIR:-/tmp}"/pytest-of-* "$${TMPDIR:-/tmp}"/htr-e2e-* "$${TMPDIR:-/tmp}"/htr-offline-* 	  backend/.pytest_cache frontend/test-results frontend/playwright-report

# Static audit: no network client code or external URLs in the app, the frontend source or the built frontend.
audit-network:
	$(RUN) python scripts/audit_network.py

# Proof that the app needs no network: static audit, then the real import -> segment -> train -> predict flow inside
# an offline network namespace (unshare -rn, loopback only). Needs `make base-model` and `make setup-frontend` first.
verify-offline:
	bash scripts/verify_offline.sh

# Real Kraken blla run on a generated synthetic page (no mocks). The default segmentation model
# (blla.mlmodel) ships inside the kraken wheel, so no model download is needed.
test-kraken:
	cd backend && $(UV) run pytest -m kraken

# Optional dev-only download (one time) of OFL script fonts into data/fonts (gitignored) for synthetic pages.
fonts:
	mkdir -p data/fonts
	curl -fsSL -o data/fonts/Caveat.ttf "https://raw.githubusercontent.com/google/fonts/main/ofl/caveat/Caveat%5Bwght%5D.ttf"
	curl -fsSL -o data/fonts/ReenieBeanie.ttf https://raw.githubusercontent.com/google/fonts/main/ofl/reeniebeanie/ReenieBeanie.ttf
	curl -fsSL -o data/fonts/ShadowsIntoLight.ttf https://raw.githubusercontent.com/google/fonts/main/ofl/shadowsintolight/ShadowsIntoLight.ttf

# One-time dev-time download of the pretrained Kraken base recognition model into data/models/base (gitignored).
# The running app never goes online.
base-model:
	$(RUN) python scripts/get_base_model.py

# Opt-in: tiny real fine-tune + predict (needs `make base-model` and `make fonts`; well under two minutes).
test-train:
	cd backend && $(UV) run pytest -m train

# Dev proof of the train/predict loop on synthetic pages (NOT real handwriting). Needs a throw-away server whose data
# dir is separate from your real one, see the header of scripts/make_synthetic_project.py:
#   mkdir -p data/synthetic_project/server/models && ln -s $$PWD/data/models/base data/synthetic_project/server/models/base
#   HTR_DATA_DIR=$$PWD/data/synthetic_project/server make serve   (own terminal)
#   make synthetic-project                                     (renders, imports, segments, labels 170 lines)
#   ... train and predict on the Training page ...
#   make synthetic-report                                      (CER of the predictions on never-labeled lines)
synthetic-project:
	PYTHONPATH=backend:scripts $(RUN) python scripts/make_synthetic_project.py build --pages 30 --label 170

# Dev proof of the active-learning loop (rounds of train + predict + simulated review) on synthetic pages through the
# API and the real GPU; needs a throw-away server like synthetic-project (see scripts/make_synthetic_rounds.py).
synthetic-rounds:
	PYTHONPATH=backend:scripts $(RUN) python scripts/make_synthetic_rounds.py

synthetic-report:
	PYTHONPATH=backend:scripts $(RUN) python scripts/make_synthetic_project.py report

lint:
	cd backend && $(UV) run ruff check .. && $(UV) run ruff format --check ..

format:
	cd backend && $(UV) run ruff check --fix .. && $(UV) run ruff format ..

# Bound to loopback only (127.0.0.1).
serve:
	cd backend && $(UV) run uvicorn app.main:app --host 127.0.0.1 --port 8000

# ---- frontend (React + TypeScript + Vite, M3) ----

# One-time, needs the network: Node 22 LTS at user level (no sudo) into ~/.local/node, linked into ~/.local/bin.
setup-node:
	bash scripts/install_node.sh

# One-time (dev tooling downloads only): npm packages from the lockfile, plus Chromium for the e2e test.
setup-frontend:
	node --version
	cd frontend && npm ci && npx playwright install chromium
	bash scripts/setup_chromium_libs.sh

# Vite dev server on http://127.0.0.1:5173 (the only origin the API's CORS allows). API base: VITE_API_BASE_URL.
dev-frontend:
	cd frontend && npm run dev

test-frontend:
	cd frontend && npm test

lint-frontend:
	cd frontend && npm run lint

# API (8000) and frontend (5173) together; Ctrl+C stops both.
dev:
	@trap 'kill 0' INT TERM EXIT; $(MAKE) serve & $(MAKE) dev-frontend & wait

# Opt-in end-to-end test: real backend on a temp data dir (real Kraken segmentation) + Vite + headless Chromium.
# Slow; not part of `make test` or `make test-frontend`.
# Temp data (the app's data dir, generated pages) is deleted afterwards, also when the run fails.
test-e2e:
	cd frontend && LD_LIBRARY_PATH="$(HOME)/.local/chromium-libs/root/usr/lib/x86_64-linux-gnu:$$LD_LIBRARY_PATH" npm run test:e2e; 	  status=$$?; rm -rf "$${TMPDIR:-/tmp}"/htr-e2e-*; exit $$status
