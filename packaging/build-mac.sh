#!/bin/bash
# Build PRISM.app on this Mac:  bash packaging/build-mac.sh [--install]
# Needs nothing installed first: it fetches uv and Python 3.12 if they are missing (no sudo).
# Output: dist-mac/PRISM.app and dist-mac/PRISM-<version>-macos-<arch>.dmg (the release file).
# --install also copies PRISM.app into /Applications.
set -euo pipefail
cd "$(dirname "$0")/.."

ARCH=$(uname -m)  # arm64 (Apple Silicon) or x86_64 (Intel)

if ! command -v uv >/dev/null 2>&1; then
  echo "== installing uv (Python manager) into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "== Python 3.12"
uv python install 3.12

echo "== building (about 10 minutes the first time)"
# --no-browser: PRISM drives your installed Chrome or Edge, or downloads Chromium into
# ~/.prism on first use. Bundling Chromium inside the app breaks its macOS framework.
uv run --no-project --python 3.12 python packaging/build.py --no-browser \
  --build-dir "${TMPDIR:-/tmp}/prism-build" --dist-dir dist-mac --arch "$ARCH"

echo "== checking that it starts"
if [ -d dist-mac/PRISM.app ]; then BIN=dist-mac/PRISM.app/Contents/MacOS/PRISM; else BIN=dist-mac/PRISM/PRISM; fi
HOME_DIR=$(mktemp -d)
PRISM_HOME="$HOME_DIR" "$BIN" --no-browser --port 18765 >"$HOME_DIR/out.log" 2>&1 &
PID=$!
STARTED=""
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:18765/api/status; then STARTED=1; break; fi
  sleep 1
done
echo
kill "$PID" 2>/dev/null || true
if [ -z "$STARTED" ]; then
  echo "!! PRISM did not start - its output:"; cat "$HOME_DIR"/out.log "$HOME_DIR"/prism.log 2>/dev/null; exit 1
fi
rm -rf "$HOME_DIR"

if [ "${1:-}" = "--install" ] && [ -d dist-mac/PRISM.app ]; then
  echo "== installing to /Applications"
  rm -rf /Applications/PRISM.app
  ditto dist-mac/PRISM.app /Applications/PRISM.app
  echo "Installed: open PRISM from Launchpad, Spotlight or Applications."
fi

echo
echo "Done:"
ls -lh dist-mac/*.dmg dist-mac/*.zip 2>/dev/null || true
