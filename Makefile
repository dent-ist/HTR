# All Python commands run inside the uv project in backend/ (Python 3.11).
UV ?= uv
RUN = $(UV) run --project backend

.PHONY: setup check test lint format serve

setup:
	$(UV) sync --project backend --python 3.11

check:
	$(RUN) python scripts/check_env.py

test:
	cd backend && $(UV) run pytest

lint:
	cd backend && $(UV) run ruff check .. && $(UV) run ruff format --check ..

format:
	cd backend && $(UV) run ruff check --fix .. && $(UV) run ruff format ..

# Bound to loopback only (127.0.0.1).
serve:
	cd backend && $(UV) run uvicorn app.main:app --host 127.0.0.1 --port 8000
