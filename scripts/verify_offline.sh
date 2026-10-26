#!/usr/bin/env bash
# `make verify-offline`: prove the app works with no network at all.
#
#  1. static audit: no network client code or external URLs in the app, the frontend source or the built frontend;
#  2. dynamic proof: run the real import -> segment -> train -> predict smoke flow inside a fresh network namespace
#     whose only interface is loopback (`unshare -rn`: an unprivileged user + network namespace, no sudo needed).
#
# If the kernel does not allow unprivileged namespaces, step 2 cannot run and this script says so and exits non-zero
# (it never pretends). Needs `make setup`, `make base-model` and `make setup-frontend` to have been done.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.local/node/current/bin:$PATH"

echo "== 1/3 build the frontend (setup tooling only; the build itself makes no network calls)"
(cd frontend && npm run build >/dev/null)

echo "== 2/3 static audit"
uv run --project backend python scripts/audit_network.py --dist

echo "== 3/3 smoke flow inside an offline network namespace"
if ! unshare -rn true 2>/dev/null; then
  echo "FAIL: cannot create a network namespace here (unshare -rn is not permitted)." >&2
  echo "      The static audit passed, but the offline run could not be performed." >&2
  exit 2
fi
PY="$(pwd)/backend/.venv/bin/python"
exec unshare -rn bash -c '
  set -e
  ip link set lo up 2>/dev/null || true
  echo "network interfaces in the namespace: $(grep -o "^ *[a-z0-9]*:" /proc/net/dev | tr -d " :" | tr "\n" " ")"
  exec "$0" scripts/offline_smoke.py "$@"
' "$PY" "$@"
