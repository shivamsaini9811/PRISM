"""A ready-to-run PRISM for macOS, made on any computer (no Mac, no build on the Mac).

    python packaging/mac_portable.py --out dist-mac-portable [--arch arm64|x86_64]

Each zip holds PRISM's Python for macOS (python-build-standalone, the pinned release in
runtime.py, checked against its SHA-256) with PRISM's own dependencies and the data packages
(numpy, pandas, ...) installed as ready-made macOS wheels by uv, PRISM's code, and
start.command. On the Mac:

    cd ~/Downloads && unzip -q PRISM-<version>-macos-arm64.zip && xattr -dr com.apple.quarantine PRISM && ./PRISM/start.command

It needs macOS 14 or later (the wheels are chosen for it). Unlike PRISM.app from
build-mac.sh it has a Terminal window while it runs; closing it stops PRISM.
"""

import argparse
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packaging"))
import llama  # noqa: E402
import runtime  # noqa: E402

ARCHES = {"arm64": ("macos-arm64", "aarch64-apple-darwin"), "x86_64": ("macos-x64", "x86_64-apple-darwin"),
          # the same package for Linux: lets the layout be tried on a machine that isn't a Mac
          "linux-test": ("linux-x64", "x86_64-unknown-linux-gnu")}
ENGINE = {"arm64": "macos-arm64", "x86_64": "macos-x86_64", "linux-test": "linux"}  # packaging/llama.py
MACOS = "14.0"

START = """#!/bin/bash
# PRISM for macOS: double-click this file, or run ./start.command in Terminal.
# PRISM opens in your browser; stop it with Quit in the page (or close this window).
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1
exec ./python/bin/python3 -m prism "$@"
"""

README = """PRISM {version} for macOS ({arch})
=====================================

Start it (in Terminal, from the folder where you unzipped it - once):

    xattr -dr com.apple.quarantine PRISM
    ./PRISM/start.command

After that, double-click PRISM/start.command to start PRISM. It opens http://127.0.0.1:8765
in your browser. Stop it with the Quit button in PRISM, or by closing its Terminal window.

- Needs macOS {macos} or later, on {kind}.
- It uses your Google Chrome or Microsoft Edge to reach Claude, ChatGPT, Gemini and Grok
  (sign in once with Connect in Settings); with neither, it downloads Chromium the first time.
- Your data (chats, settings, sign-ins) is in ~/.prism; bots' Python runs on the Python in
  this folder, in a sandbox under ~/.prism/python.
- Move or rename the PRISM folder only while PRISM is stopped.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def runtime_tarball(key: str, cache: Path) -> Path:
    triple, digest = runtime.BUILDS[key]
    name = f"cpython-{runtime.PY}+{runtime.TAG}-{triple}-install_only.tar.gz"
    path = cache / name
    if not path.exists() or sha256(path) != digest:
        print(f"+ downloading {name}", flush=True)
        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(runtime.BASE + name.replace("+", "%2B"), path)
        if sha256(path) != digest:
            path.unlink()
            raise SystemExit(f"{name} doesn't match its published SHA-256")
    return path


def build(arch: str, out: Path, cache: Path) -> Path:
    key, target = ARCHES[arch]
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    stage = cache / f"stage-{arch}"
    shutil.rmtree(stage, ignore_errors=True)
    app = stage / "PRISM"
    app.mkdir(parents=True)
    with tarfile.open(runtime_tarball(key, cache)) as t:
        t.extractall(app, filter="tar")              # -> PRISM/python
    site = app / "python" / "lib" / f"python{runtime.PY.rsplit('.', 1)[0]}" / "site-packages"
    print(f"+ installing PRISM's dependencies and {len(runtime.PACKAGES)} packages for macOS {arch}", flush=True)
    subprocess.run(["uv", "pip", "install", "--quiet", "--target", str(site),
                    "--python-platform", target, "--python-version", runtime.PY.rsplit(".", 1)[0],
                    "--only-binary", ":all:", *deps, *runtime.PACKAGES], check=True,
                   env={**os.environ, "MACOSX_DEPLOYMENT_TARGET": MACOS})
    shutil.copytree(ROOT / "src" / "prism", site / "prism",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    llama.install(ENGINE[arch], app, cache / "llama")   # -> PRISM/llama/<build>/llama-server
    (app / "start.command").write_text(START)
    (app / "start.command").chmod(0o755)
    (app / "README-MAC.txt").write_text(README.format(
        version=version, arch=arch, macos=MACOS,
        kind="Apple Silicon (M1 or later)" if arch == "arm64" else "an Intel Mac"))
    # uv puts console scripts in <target>/bin; they point at this machine's Python - not needed.
    shutil.rmtree(site / "bin", ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"PRISM-{version}-macos-{arch}.zip"
    write_zip(stage, dest)
    return dest


def write_zip(root: Path, dest: Path) -> None:
    """A zip that keeps what macOS needs: executable bits and symbolic links."""
    print(f"+ writing {dest.name}", flush=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(root.rglob("*")):
            name = path.relative_to(root).as_posix()
            if path.is_symlink():
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o755) << 16
                z.writestr(info, os.readlink(path))
            elif path.is_dir():
                info = zipfile.ZipInfo(name + "/")
                info.create_system = 3
                info.external_attr = (stat.S_IFDIR | 0o755) << 16
                z.writestr(info, "")
            else:
                info = zipfile.ZipInfo.from_file(path, name)
                info.create_system = 3
                mode = path.stat().st_mode & 0o777
                if path.parent.name == "bin" or path.name == "start.command" or os.access(path, os.X_OK):
                    mode |= 0o755
                info.external_attr = (stat.S_IFREG | mode) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                with open(path, "rb") as f:
                    z.writestr(info, f.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "dist-mac-portable")
    ap.add_argument("--cache", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "prism-mac-portable")
    ap.add_argument("--arch", choices=list(ARCHES), action="append")
    args = ap.parse_args()
    for arch in args.arch or ["arm64", "x86_64"]:
        print(f"Ready: {build(arch, args.out, args.cache)}")


if __name__ == "__main__":
    main()
