#!/usr/bin/env bash
# Playwright's Chromium needs a few shared libraries (nspr, nss, alsa) that a minimal WSL Ubuntu may lack, and
# `playwright install-deps` needs sudo. This unpacks the missing packages (apt-get download works without root)
# into ~/.local/chromium-libs; `make test-e2e` adds that directory to LD_LIBRARY_PATH. Dev tooling only.
set -euo pipefail

DEST="$HOME/.local/chromium-libs"
LIBDIR="$DEST/root/usr/lib/x86_64-linux-gnu"
BROWSERS="${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}"
BIN=$(ls -d "$BROWSERS"/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell 2>/dev/null | tail -1 || true)
[ -n "$BIN" ] || { echo "Chromium not installed yet: run 'make setup-frontend' first" >&2; exit 1; }

missing() { LD_LIBRARY_PATH="$LIBDIR:${LD_LIBRARY_PATH:-}" ldd "$BIN" | grep "not found" || true; }

if [ -z "$(missing)" ]; then
  echo "Chromium shared libraries OK"
  exit 0
fi

mkdir -p "$DEST/debs" && cd "$DEST/debs"
apt-get download libnspr4 libnss3 libasound2t64 2>/dev/null || apt-get download libnspr4 libnss3 libasound2
for f in *.deb; do dpkg-deb -x "$f" "$DEST/root"; done

if [ -n "$(missing)" ]; then
  echo "Still missing libraries:" >&2
  missing >&2
  exit 1
fi
echo "Chromium shared libraries unpacked into $DEST"
