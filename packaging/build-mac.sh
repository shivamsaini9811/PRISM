#!/bin/bash
# Build PRISM on this Mac:  bash packaging/build-mac.sh
# Needs nothing installed first: it fetches uv and Python 3.12 if they are missing (no sudo).
# Output: dist-mac/PRISM/PRISM (the app) and dist-mac/PRISM-<version>-macos-<arch>.zip
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
# --no-browser: PRISM drives your installed Chrome or Edge, or downloads Chromium on first
# use. Bundling Chromium inside the app folder breaks its macOS framework.
uv run --no-project --python 3.12 python packaging/build.py --no-browser \
  --build-dir "${TMPDIR:-/tmp}/prism-build" --dist-dir dist-mac --arch "$ARCH"

echo "== checking that it starts"
HOME_DIR=$(mktemp -d)
PRISM_HOME="$HOME_DIR" ./dist-mac/PRISM/PRISM --no-browser --port 18765 >"$HOME_DIR/out.log" 2>&1 &
PID=$!
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:18765/api/status; then break; fi
  sleep 1
done
echo
kill "$PID" 2>/dev/null || true
if ! grep -q "PRISM" "$HOME_DIR/out.log"; then
  echo "!! PRISM did not start - its output:"; cat "$HOME_DIR/out.log"; exit 1
fi
rm -rf "$HOME_DIR"

echo
echo "Done:"
ls -lh dist-mac/*.zip
echo "Run it:  ./dist-mac/PRISM/PRISM"
