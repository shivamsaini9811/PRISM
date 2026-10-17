"""PRISM launcher: `prism` (or the packaged executable) starts the app and opens the UI."""

import argparse
import json
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

DEFAULT_PORT = 8765


def _frozen_setup() -> None:
    """Inside the packaged app. Windows: use the Chromium bundled next to the Playwright driver.
    macOS: PRISM.app bundles no browser (it drives Chrome or Edge); a Chromium it has to fetch
    goes into the data folder - the signed app bundle must not change. A Mac app has no
    terminal, so its output goes to a log file."""
    if not getattr(sys, "frozen", False):
        return
    from .config import data_dir
    if sys.platform == "darwin":
        os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(data_dir() / "browsers"))
        if sys.stdout is None or not sys.stdout.isatty():
            data_dir().mkdir(parents=True, exist_ok=True)
            log = open(data_dir() / "prism.log", "a", buffering=1, encoding="utf-8")
            sys.stdout = sys.stderr = log
    else:
        os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "0")


def _running_prism(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=1) as r:
            return "initialized" in json.load(r)
    except Exception:
        return False


def _port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="prism", description="PRISM multi-model chat")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="don't open the UI automatically")
    args = parser.parse_args()

    _frozen_setup()
    port = args.port
    url = f"http://127.0.0.1:{port}"

    if _running_prism(port):  # already running: just bring the UI up again
        print(f"PRISM is already running at {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return
    if not _port_free(port):
        port = _free_port()
        url = f"http://127.0.0.1:{port}"

    import uvicorn

    from . import __version__
    from .app import app
    from .config import data_dir

    def open_when_ready() -> None:
        for _ in range(100):
            if _running_prism(port):
                webbrowser.open(url)
                return
            time.sleep(0.1)

    print(f"PRISM {__version__}\n  UI:   {url}\n  Data: {data_dir()}\n"
          "  Quit: the Quit button in PRISM, or Ctrl+C here.", flush=True)
    if not args.no_browser:
        threading.Thread(target=open_when_ready, daemon=True).start()
    # Bound to localhost only: the vault and chats are never exposed to the network.
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    app.state.server = server  # the Quit button stops it
    server.run()


if __name__ == "__main__":
    main()
