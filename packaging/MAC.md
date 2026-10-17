## Build PRISM.app (on a Mac)

From the PRISM folder (unzipped source), in Terminal:

    bash packaging/build-mac.sh --install

It installs uv and Python 3.12 if needed (no sudo), builds `PRISM.app`, checks that it starts,
copies it into /Applications (that's `--install`), and leaves the release file
`dist-mac/PRISM-<version>-macos-<arch>.dmg`. Without `--install` it only builds.

An Apple Silicon Mac builds the `arm64` version; Intel Macs need a build made on an Intel Mac.

## Install PRISM from the .dmg

1. Open `PRISM-<version>-macos-<arch>.dmg` and drag **PRISM** onto **Applications**.
2. PRISM isn't signed with an Apple Developer ID, so the first time macOS refuses to open a
   downloaded copy. Either run once in Terminal:

       xattr -dr com.apple.quarantine /Applications/PRISM.app

   or open it, then System Settings → Privacy & Security → **Open Anyway**.
   (A copy you built yourself opens straight away.)

## Using it

- Open **PRISM** from Launchpad, Spotlight or Applications. It opens http://127.0.0.1:8765 in
  your browser. Opening it again while it runs brings the page back.
- It has no window or Dock icon: stop it with the **Quit** button in the PRISM page.
- It drives your installed Google Chrome or Microsoft Edge (sign in to each site once with
  Connect). With neither, it downloads Chromium the first time.
- Data (chats, settings, sign-ins) lives in `~/.prism`; the log is `~/.prism/prism.log`.
