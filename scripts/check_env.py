#!/usr/bin/env python3
"""Environment check for the HTR tool (WSL2, CUDA, GPU, Kraken, disk).

Run via `make check`. Exits non-zero if any check FAILs; WARN does not fail.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MIN_DISK_GB_FAIL = 5.0
MIN_DISK_GB_WARN = 20.0

OK, WARN, FAIL = "OK", "WARN", "FAIL"
Result = tuple[str, str]  # (status, message)


def check_wsl2() -> Result:
    if sys.platform != "linux":
        return FAIL, f"not Linux (platform={sys.platform}); run inside WSL2 Ubuntu"
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text().lower()
        version = Path("/proc/version").read_text().lower()
    except OSError as e:
        return FAIL, f"cannot read /proc to detect WSL: {e}"
    if "microsoft" not in release and "microsoft" not in version:
        return FAIL, "Linux kernel is not a Microsoft/WSL kernel; run inside WSL2"
    if "wsl2" in release or "wsl2" in version:
        return OK, f"WSL2 detected (kernel {release.strip()}, distro={os.environ.get('WSL_DISTRO_NAME', '?')})"
    return FAIL, f"WSL detected but not clearly WSL2 (kernel {release.strip()})"


def _import_torch():
    import torch  # noqa: PLC0415

    return torch


def check_torch_cuda() -> Result:
    try:
        torch = _import_torch()
    except Exception as e:  # noqa: BLE001
        return FAIL, f"cannot import torch: {e}"
    if not torch.cuda.is_available():
        return FAIL, f"torch {torch.__version__}: torch.cuda.is_available() is False (CUDA build: {torch.version.cuda})"
    try:
        a = torch.arange(1.0, 1025.0, device="cuda").reshape(32, 32)
        got = (a @ a).sum().item()
        torch.cuda.synchronize()
        want = (a.cpu() @ a.cpu()).sum().item()
    except Exception as e:  # noqa: BLE001
        return FAIL, f"torch {torch.__version__}: CUDA tensor op failed: {e}"
    if abs(got - want) > 1e-3 * abs(want):
        return FAIL, f"CUDA matmul result {got} differs from CPU {want}"
    return OK, f"torch {torch.__version__} (CUDA {torch.version.cuda}); real CUDA matmul matches CPU"


def parse_arch(arch: str) -> tuple[str, int] | None:
    """'sm_61' -> ('sm', 61); 'compute_70' -> ('compute', 70)."""
    kind, _, num = arch.partition("_")
    if num.isdigit() and kind in {"sm", "compute"}:
        return kind, int(num)
    return None


def arch_support(capability: tuple[int, int], arch_list: list[str]) -> tuple[str, str]:
    """Judge whether the installed torch can run on a GPU of this capability.

    Returns (status, detail). SASS built for sm_6X runs on any later 6.Y GPU,
    and compute_XX PTX can be JIT-compiled, so those count as supported.
    """
    major, minor = capability
    want = major * 10 + minor
    parsed = [p for p in map(parse_arch, arch_list) if p]
    if ("sm", want) in parsed:
        return OK, f"native sm_{want} kernels present"
    same_major = [n for k, n in parsed if k == "sm" and n // 10 == major and n <= want]
    if same_major:
        return OK, f"no native sm_{want}, but binary-compatible sm_{max(same_major)} kernels present"
    ptx = [n for k, n in parsed if k == "compute" and n <= want]
    if ptx:
        return WARN, f"no sm_{want} kernels; relying on PTX JIT from compute_{max(ptx)} (slow first run)"
    return WARN, f"torch has NO kernels usable on sm_{want} (arch list: {arch_list})"


def check_gpu() -> Result:
    try:
        torch = _import_torch()
    except Exception as e:  # noqa: BLE001
        return FAIL, f"cannot import torch: {e}"
    if not torch.cuda.is_available():
        return FAIL, "no CUDA device visible"
    props = torch.cuda.get_device_properties(0)
    cap = (props.major, props.minor)
    arch_list = torch.cuda.get_arch_list()
    status, detail = arch_support(cap, arch_list)
    msg = (
        f"{props.name}, {props.total_memory / 1024**3:.1f} GiB VRAM, "
        f"compute capability {cap[0]}.{cap[1]}; torch arch list {arch_list}; {detail}"
    )
    return status, msg


def check_kraken() -> Result:
    exe = Path(sys.executable).parent / "kraken"
    cmd = [str(exe) if exe.exists() else (shutil.which("kraken") or "kraken"), "--version"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        return FAIL, "kraken CLI not found; run `make setup`"
    except subprocess.TimeoutExpired:
        return FAIL, "kraken --version timed out"
    if p.returncode != 0:
        return FAIL, f"`kraken --version` exited {p.returncode}: {(p.stderr or p.stdout).strip()[-300:]}"
    return OK, (p.stdout or p.stderr).strip().splitlines()[-1]


def check_disk() -> Result:
    free_gb = shutil.disk_usage(REPO).free / 1024**3
    msg = f"{free_gb:.1f} GiB free at {REPO}"
    if free_gb < MIN_DISK_GB_FAIL:
        return FAIL, f"{msg} (< {MIN_DISK_GB_FAIL:.0f} GiB)"
    if free_gb < MIN_DISK_GB_WARN:
        return WARN, f"{msg} (< {MIN_DISK_GB_WARN:.0f} GiB recommended)"
    return OK, msg


CHECKS: list[tuple[str, Callable[[], Result]]] = [
    ("WSL2", check_wsl2),
    ("PyTorch CUDA", check_torch_cuda),
    ("GPU", check_gpu),
    ("Kraken CLI", check_kraken),
    ("Disk space", check_disk),
]


def main() -> int:
    failed = 0
    for name, fn in CHECKS:
        try:
            status, msg = fn()
        except Exception as e:  # noqa: BLE001
            status, msg = FAIL, f"unexpected error: {e!r}"
        print(f"[{status:<4}] {name}: {msg}")
        failed += status == FAIL
    print()
    print("All checks passed." if not failed else f"{failed} check(s) FAILED.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
