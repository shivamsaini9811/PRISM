"""Build the standalone PRISM app for the OS this script runs on.

    python packaging/build.py              # full build, Chromium bundled
    python packaging/build.py --no-browser # smaller build, relies on Edge/Chrome on the target

Output: dist/PRISM/ (PRISM.exe, or PRISM on macOS, + its files) and
dist/PRISM-<version>-<os>[-<arch>].zip.
PyInstaller does not cross-compile: run this on Windows for the Windows build, on a Mac for
the Mac build (.github/workflows/mac.yml does that on GitHub's Macs).
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES_WITH_METADATA = ["anthropic", "openai", "google-genai", "playwright", "fastapi",
                          "pydantic", "pydantic-settings", "uvicorn", "cryptography"]


def run(*cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-browser", action="store_true", help="don't bundle Chromium")
    # Build on the system drive: venv + Chromium are thousands of small files, and slow drives
    # (e.g. an external D:) make pip and Playwright's installer lock time out.
    ap.add_argument("--build-dir", type=Path,
                    default=Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()) / "prism-build")
    ap.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    ap.add_argument("--arch", default="", help="label for the zip name, e.g. arm64 or x86_64")
    args = ap.parse_args()

    # Isolated build venv so the result never depends on what this machine has installed.
    venv_dir = args.build_dir / f"venv-{sys.platform}"
    if not venv_dir.exists():
        venv.create(venv_dir, with_pip=True)
    py = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run(py, "-m", "pip", "install", "-q", "--upgrade", "pip")
    # Install from a clean copy of just the sources: the project folder also holds build/ and
    # dist/ (venvs, Chromium), which pip would otherwise crawl for minutes.
    src_copy = args.build_dir / "src-copy"
    shutil.rmtree(src_copy, ignore_errors=True)
    shutil.copytree(ROOT / "src", src_copy / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(ROOT / name, src_copy / name)
    # Exact versions from uv.lock (packaging/requirements.lock.txt): pip has nothing to resolve.
    # Regenerate after dependency changes:
    #   uv export --format requirements-txt --no-dev --no-hashes --no-emit-project \
    #     -o packaging/requirements.lock.txt
    run(py, "-m", "pip", "install", "-q", "-r", ROOT / "packaging/requirements.lock.txt")
    run(py, "-m", "pip", "install", "-q", "--no-deps", str(src_copy))
    run(py, "-m", "pip", "install", "-q", "pyinstaller>=6.10")

    env = {**os.environ}
    if not args.no_browser:
        # Browsers path 0 = install inside the playwright package, so PyInstaller ships it.
        env["PLAYWRIGHT_BROWSERS_PATH"] = "0"
        run(py, "-m", "playwright", "install", "chromium", env=env)

    sep = ";" if os.name == "nt" else ":"
    cmd = [py, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--name", "PRISM",
           "--distpath", args.dist_dir, "--workpath", args.build_dir / "pyinstaller",
           "--specpath", args.build_dir,
           "--add-data", f"{ROOT / 'src/prism/static'}{sep}prism/static",
           "--collect-data", "playwright",
           "--collect-submodules", "uvicorn"]
    for pkg in PACKAGES_WITH_METADATA:
        cmd += ["--copy-metadata", pkg]
    run(*cmd, ROOT / "packaging/prism_launcher.py", env=env)

    version = next(line.split('"')[1] for line in (ROOT / "src/prism/__init__.py").read_text().splitlines()
                   if line.startswith("__version__"))
    osname = {"Windows": "windows", "Darwin": "macos"}.get(platform.system(), "linux")
    stem = args.dist_dir / f"PRISM-{version}-{osname}{'-' + args.arch if args.arch else ''}"
    if platform.system() == "Darwin":  # ditto keeps symlinks and permissions (zipfile doesn't)
        archive = f"{stem}.zip"
        run("ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", args.dist_dir / "PRISM", archive)
    else:
        archive = shutil.make_archive(str(stem), "zip", args.dist_dir, "PRISM")
    print(f"\nBuilt {args.dist_dir / 'PRISM'}\nZip   {archive}")


if __name__ == "__main__":
    main()
