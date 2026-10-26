#!/usr/bin/env bash
# SETUP ONLY (needs the network, once): install Node.js 22 LTS at user level, no sudo.
#
# Downloads the newest Node 22 Linux x64 tarball from nodejs.org, checks it against the published SHA-256 list,
# unpacks it into $NODE_PREFIX (default ~/.local/node) and links node, npm and npx into $NODE_BIN_DIR (default
# ~/.local/bin, which must be on PATH). The running app never uses the network; Node is only needed to build and test
# the frontend and to run the Vite dev server. Needs curl and xz-utils (`sudo apt install curl xz-utils`).
set -euo pipefail

PREFIX="${NODE_PREFIX:-$HOME/.local/node}"
BIN="${NODE_BIN_DIR:-$HOME/.local/bin}"
BASE="https://nodejs.org/dist/latest-v22.x"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$work"

curl -fsSL -o SHASUMS256.txt "$BASE/SHASUMS256.txt"
file="$(grep -o 'node-v22\.[0-9.]*-linux-x64\.tar\.xz' SHASUMS256.txt | head -1)"
[ -n "$file" ] || { echo "could not find a Node 22 linux-x64 tarball in $BASE" >&2; exit 1; }
echo "downloading $file"
curl -fsSL -O "$BASE/$file"
grep " $file\$" SHASUMS256.txt | sha256sum -c -

mkdir -p "$PREFIX" "$BIN"
tar -xf "$file" -C "$PREFIX"
ln -sfn "$PREFIX/${file%.tar.xz}" "$PREFIX/current"
for b in node npm npx; do ln -sf "$PREFIX/current/bin/$b" "$BIN/$b"; done
echo "installed: $("$PREFIX/current/bin/node" --version) in $PREFIX/current (links in $BIN)"
case ":$PATH:" in *":$BIN:"*) ;; *) echo "note: add $BIN to your PATH (for example in ~/.bashrc)" ;; esac
