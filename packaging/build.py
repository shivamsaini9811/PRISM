"""Build the standalone PRISM app for the OS this script runs on.

    python packaging/build.py              # full build, Chromium bundled
    python packaging/build.py --no-browser # smaller build, relies on Edge/Chrome on the target

Output - Windows/Linux: dist/PRISM/ (PRISM.exe + its files) and dist/PRISM-<version>-<os>.zip.
macOS: dist/PRISM.app and dist/PRISM-<version>-macos-<arch>.dmg (drag PRISM to Applications).
PyInstaller does not cross-compile: run this on Windows for the Windows build, on a Mac for
the Mac build (.github/workflows/mac.yml does that on GitHub's Macs).
"""

import argparse
import json
import os
import plistlib
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
    ap.add_argument("--no-runtime", action="store_true",
                    help="leave out PRISM's own Python (bots can't RUN code; a much smaller build)")
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
    run(py, "-m", "pip", "install", "-q", "pyinstaller>=6.10", "pillow")  # pillow: the icon

    env = {**os.environ}
    if not args.no_browser:
        # Browsers path 0 = install inside the playwright package, so PyInstaller ships it.
        env["PLAYWRIGHT_BROWSERS_PATH"] = "0"
        run(py, "-m", "playwright", "install", "chromium", env=env)

    sep = ";" if os.name == "nt" else ":"
    cmd = [py, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--name", "PRISM",
           "--distpath", args.dist_dir, "--workpath", args.build_dir / "pyinstaller",
           "--specpath", args.build_dir,
           "--icon", ROOT / "packaging/icon.png",
           "--add-data", f"{ROOT / 'src/prism/static'}{sep}prism/static",
           "--collect-data", "playwright",
           "--collect-submodules", "uvicorn"]
    if (ROOT / "src/prism/publisher.pub").is_file():  # the catalog publisher's key, trusted out of the box
        cmd += ["--add-data", f"{ROOT / 'src/prism/publisher.pub'}{sep}prism"]
    for pkg in PACKAGES_WITH_METADATA:
        cmd += ["--copy-metadata", pkg]
    mac = platform.system() == "Darwin"
    if mac:  # PRISM.app: opened from Applications, no terminal window
        cmd += ["--windowed", "--osx-bundle-identifier", "app.prism.desktop"]
    run(*cmd, ROOT / "packaging/prism_launcher.py", env=env)

    version = next(line.split('"')[1] for line in (ROOT / "src/prism/__init__.py").read_text().splitlines()
                   if line.startswith("__version__"))
    osname = {"Windows": "windows", "Darwin": "macos"}.get(platform.system(), "linux")
    stem = args.dist_dir / f"PRISM-{version}-{osname}{'-' + args.arch if args.arch else ''}"
    if not args.no_runtime:  # PRISM's own Python with its packages (packaging/runtime.py)
        sys.path.insert(0, str(ROOT / "packaging"))
        import runtime
        src = runtime.prepare(args.build_dir)
        # One archive, unpacked by PRISM into its data folder on first use: ~30,000 small
        # files next to the app would take a terabyte on an exFAT drive (32 MB clusters).
        dest = (args.dist_dir / "PRISM.app/Contents/Resources") if mac else (args.dist_dir / "PRISM")
        archive = args.build_dir / f"runtime-{runtime.TAG}-{runtime.platform_key()}.tar.gz"
        if not archive.exists():
            print(f"+ packing PRISM's Python into {archive.name}", flush=True)
            shutil.make_archive(str(archive).removesuffix(".tar.gz"), "gztar", src)
        shutil.copy2(archive, dest / "runtime.tar.gz")
        (dest / "runtime.json").write_text(json.dumps({"python": runtime.PY, "tag": runtime.TAG,
                                                       "packages": len(runtime.PACKAGES)}))
    if mac:
        app = args.dist_dir / "PRISM.app"
        finish_mac_app(app, version)
        print(f"\nBuilt {app}\nDMG   {make_dmg(app, args.build_dir, stem.with_suffix('.dmg'))}")
        return
    archive = shutil.make_archive(str(stem), "zip", args.dist_dir, "PRISM")
    print(f"\nBuilt {args.dist_dir / 'PRISM'}\nZip   {archive}")


def finish_mac_app(app: Path, version: str) -> None:
    """No Dock icon (PRISM has no window; the Quit button in the page stops it), the real
    version in Finder, then sign again: editing Info.plist breaks PyInstaller's signature.
    Ad-hoc signed only - without an Apple Developer ID, a downloaded copy needs the quarantine
    flag cleared once (see packaging/MAC.md)."""
    plist_path = app / "Contents/Info.plist"
    info = plistlib.loads(plist_path.read_bytes())
    info.update({"LSUIElement": True, "CFBundleShortVersionString": version,
                 "CFBundleVersion": version, "CFBundleDisplayName": "PRISM",
                 "NSHumanReadableCopyright": "PRISM - one group chat for your AI models"})
    plist_path.write_bytes(plistlib.dumps(info))
    run("codesign", "--force", "--deep", "--sign", "-", app)
    run("codesign", "--verify", "--deep", "--strict", app)


def make_dmg(app: Path, build_dir: Path, dmg: Path) -> Path:
    """The usual Mac installer: a disk image with PRISM.app next to an Applications shortcut."""
    stage = build_dir / "dmg"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    run("ditto", app, stage / "PRISM.app")  # ditto keeps the bundle's symlinks and signature
    (stage / "Applications").symlink_to("/Applications")
    dmg.unlink(missing_ok=True)
    run("hdiutil", "create", "-volname", "PRISM", "-srcfolder", stage, "-ov", "-format", "UDZO", dmg)
    return dmg


if __name__ == "__main__":
    main()
