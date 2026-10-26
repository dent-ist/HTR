import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout


def test_no_file_under_data_is_tracked():
    tracked = git("ls-files", "--", "data").split()
    assert tracked == [], f"personal data must never be tracked: {tracked}"


def test_data_dir_is_ignored():
    probe = "data/_probe_should_be_ignored.png"
    out = subprocess.run(["git", "-C", str(REPO), "check-ignore", probe], capture_output=True, text=True)
    assert out.returncode == 0, "data/ must be gitignored"
