# HTR

Local-first handwritten text recognition for one writer's scanned notes. Nothing leaves the machine.
Full spec: [docs/implementation-plan.md](docs/implementation-plan.md).

## Layout

```
backend/     uv project (Python 3.11): FastAPI app, tests, pyproject.toml + uv.lock
frontend/    placeholder (React, from M3)
scripts/     check_env.py, inspect_images.py
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
