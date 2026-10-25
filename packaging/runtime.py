"""PRISM's own Python, shipped inside the app: what bots' code runs on (RUN @python).

A full CPython from python-build-standalone (pinned release, checked against its published
SHA-256), with a large set of packages installed into it at build time, so code that uses
numpy, pandas, matplotlib and the rest works on a computer with no Python at all. At run
time PRISM makes a sandbox on top of it (a venv that sees these packages); what bots pip
install goes there, never into this runtime or the system's Python.

    python packaging/runtime.py --build-dir <dir>     # prepare it for this computer, cached

build.py calls prepare() and copies the result next to PRISM (dist/PRISM/runtime, or
PRISM.app/Contents/Resources/runtime on a Mac).
"""

import argparse
import hashlib
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

TAG = "20261009"          # python-build-standalone release (pinned; published 2026-10-09)
PY = "3.12.15"
BASE = f"https://github.com/astral-sh/python-build-standalone/releases/download/{TAG}/"
# platform -> (target triple, sha256 from the release's SHA256SUMS)
BUILDS = {
    "windows-x64": ("x86_64-pc-windows-msvc", "194c8f0d9e0de6f1c4576e5dfa67819a48c764be87608b74b8ad43387a57b93b"),
    "macos-arm64": ("aarch64-apple-darwin", "83dcb04777c798bd95f3687acf9ae84b09860272ee6bdfb05be8bcf9447f75b8"),
    "macos-x64": ("x86_64-apple-darwin", "4c84ef5abe4bc145646175fce9b881936d60126b9a3f9a24c6308e2d035ca5c6"),
    "linux-x64": ("x86_64-unknown-linux-gnu", "5864001bf969db329fadc2b654c4878c6027d170c6bd77e28f88666260bcaf63"),
}
# What code can use without installing anything: data, maths, charts, files, the web.
PACKAGES = [
    "numpy", "pandas", "scipy", "matplotlib", "seaborn", "scikit-learn", "statsmodels", "sympy",
    "networkx", "duckdb", "pyarrow", "openpyxl", "xlsxwriter", "python-docx", "python-pptx",
    "pypdf", "pillow", "requests", "httpx", "beautifulsoup4", "lxml", "pyyaml", "jinja2",
    "markdown", "tabulate", "python-dateutil", "pytz", "tqdm", "rich", "plotly",
]


def platform_key() -> str:
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    return {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux") + "-" + arch


def python_exe(root: Path) -> Path:
    return root / "python.exe" if (root / "python.exe").exists() else root / "bin" / "python3"


def prepare(build_dir: Path, key: str | None = None) -> Path:
    """The runtime for this computer with the packages installed: build_dir/runtime-<key>/python."""
    key = key or platform_key()
    triple, sha = BUILDS[key]
    name = f"cpython-{PY}+{TAG}-{triple}-install_only.tar.gz"
    out = build_dir / f"runtime-{key}"
    root = out / "python"
    done = out / f"ready-{TAG}-{len(PACKAGES)}"
    if done.exists():
        return root
    build_dir.mkdir(parents=True, exist_ok=True)
    archive = build_dir / name
    if not archive.exists() or _sha256(archive) != sha:
        print(f"+ downloading {name}", flush=True)
        part = archive.with_suffix(".part")
        urllib.request.urlretrieve(BASE + name.replace("+", "%2B"), part)
        if _sha256(part) != sha:
            part.unlink()
            raise SystemExit(f"{name} doesn't match its published SHA-256 - not using it")
        part.replace(archive)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    with tarfile.open(archive) as t:
        root_dir = out.resolve()
        for m in t.getmembers():
            if not (out / m.name).resolve().is_relative_to(root_dir):
                raise SystemExit("the runtime archive has paths outside its folder")
        t.extractall(out)
    exe = python_exe(root)
    print(f"+ installing {len(PACKAGES)} packages into {exe}", flush=True)
    subprocess.run([str(exe), "-m", "pip", "install", "--no-input", "--disable-pip-version-check",
                    "--no-warn-script-location", *PACKAGES], check=True)
    freeze = subprocess.run([str(exe), "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    (root / "PRISM-PACKAGES.txt").write_text(freeze)
    _trim(root)
    done.write_text(freeze)
    return root


def _trim(root: Path) -> None:
    """Test suites nobody runs here: the standard library's own, and packages' tests folders."""
    for lib in list(root.glob("Lib")) + list(root.glob("lib/python3*")):
        shutil.rmtree(lib / "test", ignore_errors=True)
        for tests in (lib / "site-packages").glob("*/tests"):
            if tests.is_dir():
                shutil.rmtree(tests, ignore_errors=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", type=Path, default=Path(tempfile.gettempdir()) / "prism-build")
    ap.add_argument("--platform", default=None, help="windows-x64 | macos-arm64 | macos-x64 | linux-x64")
    args = ap.parse_args()
    root = prepare(args.build_dir, args.platform)
    print(f"Runtime ready: {python_exe(root)}")


if __name__ == "__main__":
    main()
