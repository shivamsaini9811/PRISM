## Build PRISM on a Mac

From the PRISM folder (unzipped source):

    bash packaging/build-mac.sh

It installs uv and Python 3.12 if needed (no sudo), builds, checks the app starts, and leaves
`dist-mac/PRISM/PRISM` and `dist-mac/PRISM-<version>-macos-<arch>.zip`.

## PRISM on a Mac

1. Download the zip for your Mac: **arm64** for Apple Silicon (M1 and later), **x86_64** for Intel.
2. Unzip it, then open Terminal in that folder and run once:
   `xattr -dr com.apple.quarantine PRISM`
   (PRISM isn't signed with an Apple Developer ID, so macOS blocks it until you do this.)
3. Start it: `./PRISM/PRISM` - or double-click `PRISM` inside the folder. It opens
   http://127.0.0.1:8765 in your browser.

PRISM drives your installed Google Chrome or Microsoft Edge (signed in once through Connect).
If neither is installed it downloads Chromium the first time. Data lives in `~/.prism`.
