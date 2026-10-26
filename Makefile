# All Python commands run inside the uv project in backend/ (Python 3.11).
UV ?= uv
RUN = $(UV) run --project backend

.PHONY: setup check test test-kraken fonts lint format serve

setup:
	$(UV) sync --project backend --python 3.11

check:
	$(RUN) python scripts/check_env.py

test:
	cd backend && $(UV) run pytest

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

lint:
	cd backend && $(UV) run ruff check .. && $(UV) run ruff format --check ..

format:
	cd backend && $(UV) run ruff check --fix .. && $(UV) run ruff format ..

# Bound to loopback only (127.0.0.1).
serve:
	cd backend && $(UV) run uvicorn app.main:app --host 127.0.0.1 --port 8000
