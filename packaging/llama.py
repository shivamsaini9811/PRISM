"""PRISM's model engine: llama.cpp's llama-server, pinned to one release and checked.

    python packaging/llama.py --platform windows --dest dist/PRISM

puts llama/<build>/llama-server(.exe) and its libraries under dest - the place
prism.engine.server_binary() looks. Downloads are cached and checked against the SHA-256
GitHub publishes for each file.
"""

import argparse
import hashlib
import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

TAG = "b11559"  # 2026-10-11
BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/"
# platform -> {build: (file, sha256)}
BUILDS = {
    "windows": {
        "cpu": (f"llama-{TAG}-bin-win-cpu-x64.zip", "1228f133868c02a424bcf663a688e66d8c7d5499077649f305188f756f584377"),
        "vulkan": (f"llama-{TAG}-bin-win-vulkan-x64.zip", "bf2fb2a7fc5d7669ccabbb6de3f6dd6b54058f213b32e731abc62b590f8566dc"),
    },
    "macos-arm64": {
        "metal": (f"llama-{TAG}-bin-macos-arm64.tar.gz", "c6cfeaa709b8d2e47e8a50f6833d22071f7ee7079b0b1a6d704e482afe6ee2c6"),
    },
    "macos-x86_64": {
        "cpu": (f"llama-{TAG}-bin-macos-x64.tar.gz", "630d5ab669876748b4f715def6b1cbbc2a8cf836ca8310cb9e4585d5f61fe8d4"),
    },
    "linux": {
        "cpu": (f"llama-{TAG}-bin-ubuntu-x64.tar.gz", "a454867b975dec288f731c6d3589f132b0e356f7dbad8d711f48d231ee67d7d9"),
    },
}
# Only what llama-server needs; the other tools (bench, quantize, ...) stay out.
SKIP = ("llama-bench", "llama-batched", "llama-perplexity", "llama-quantize", "llama-imatrix", "llama-tts",
        "llama-gguf", "llama-tokenize", "llama-results", "llama-fit-params", "llama-llava", "llama-minicpmv",
        "llama-qwen2vl", "llama-gemma3", "llama-mtmd", "llama-cli", "llama-completion", "llama-run",
        "llama-simple", "llama-embedding", "llama-export", "llama-cvector", "llama-lookup", "llama-parallel",
        "llama-passkey", "llama-retrieval", "llama-save", "llama-speculative", "llama-gen", "llama-eval",
        "ggml-rpc-server", "rpc-server", "test-")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, digest: str, cache: Path) -> Path:
    path = cache / name
    if not path.exists() or sha256(path) != digest:
        print(f"+ downloading {name}", flush=True)
        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BASE + name, path)
        if sha256(path) != digest:
            path.unlink()
            raise SystemExit(f"{name} doesn't match its published SHA-256")
    return path


def wanted(name: str) -> bool:
    stem = name.rsplit("/", 1)[-1]
    return not stem.startswith(SKIP) or stem.startswith(("llama-server", "llama-common"))


def install(platform: str, dest: Path, cache: Path) -> list[Path]:
    """Unpack each build of this platform into dest/llama/<build>/; returns the servers."""
    servers = []
    for build, (name, digest) in BUILDS[platform].items():
        archive = fetch(name, digest, cache)
        stage = cache / f"x-{build}-{platform}"
        shutil.rmtree(stage, ignore_errors=True)
        if name.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                z.extractall(stage)
        else:
            with tarfile.open(archive) as t:
                t.extractall(stage, filter="tar")
        exe = next(p for p in stage.rglob("llama-server*") if p.is_file() and p.name in ("llama-server", "llama-server.exe"))
        out = dest / "llama" / build
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        for p in exe.parent.iterdir():
            if p.is_dir() or not wanted(p.name):
                continue
            target = out / p.name
            if p.is_symlink():
                os.symlink(os.readlink(p), target)
            else:
                shutil.copy2(p, target)
        shutil.rmtree(stage, ignore_errors=True)
        servers.append(out / exe.name)
        print(f"+ {build}: {out / exe.name}", flush=True)
    return servers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=list(BUILDS), required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "prism-llama")
    a = ap.parse_args()
    install(a.platform, a.dest, a.cache)


if __name__ == "__main__":
    main()
