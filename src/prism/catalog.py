"""The catalog: Python packages and models the publisher (the user's own server) offers, signed.

The server holds two files next to each other: catalog.json and catalog.json.sig (the Ed25519
signature of catalog.json's exact bytes, base64). PRISM installs nothing until it has the
publisher's public key (Settings, or packaged as prism/publisher.pub), and then only from a
catalog whose signature checks out, whose version is not older than the last one accepted
and that has not expired - so a stolen or replayed server can't push anything. Each download
must match the catalog's SHA-256 before it is used, and archives can't write outside their
folder. Make keys and sign with `prism-publish` (prism/publish.py).

catalog.json:
    {"format": 1, "name": "My tools", "version": 7, "expires": "2027-01-31T00:00:00Z",
     "entries": [
       {"name": "mytools", "version": "1.2.0", "kind": "python", "description": "...",
        "files": {"any": {"url": "mytools-1.2.0-py3-none-any.whl", "sha256": "...", "size": 12345}}},
       {"name": "qwen-7b", "version": "q4", "kind": "model", "as": "qwen",
        "files": {"any": {"url": "https://.../qwen.gguf", "sha256": "...", "size": 4800000000}},
        "needs": {"ram_mb": 6000}}]}
File URLs may be relative to the catalog's address; a file per platform is allowed
(windows-x64, macos-arm64, macos-x64, linux-x64, any). Python packages (wheels or source
archives) install into PRISM's Python sandbox, where bots' code can import them; models go
to Ollama and join the room.
"""

import asyncio
import base64
import hashlib
import json
import os
import platform
import shutil
import stat
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

KEY_PREFIX = "ed25519:"
BUNDLED_KEY = Path(__file__).parent / "publisher.pub"


class CatalogError(Exception):
    pass


# --- keys and signatures ---------------------------------------------------------------------

def parse_key(text: str) -> Ed25519PublicKey:
    text = text.strip()
    if not text.startswith(KEY_PREFIX):
        raise CatalogError(f"a public key looks like {KEY_PREFIX}… (prism-publish keygen prints it)")
    try:
        raw = base64.b64decode(text[len(KEY_PREFIX):], validate=True)
        return Ed25519PublicKey.from_public_bytes(raw)
    except Exception as e:
        raise CatalogError("that public key is damaged - copy the whole line again") from e


def fingerprint(text: str) -> str:
    raw = base64.b64decode(text.strip()[len(KEY_PREFIX):])
    h = hashlib.sha256(raw).hexdigest()[:16]
    return " ".join(h[i:i + 4] for i in range(0, 16, 4))


def bundled_keys() -> list[str]:
    if BUNDLED_KEY.is_file():
        return [line.strip() for line in BUNDLED_KEY.read_text().splitlines() if line.strip().startswith(KEY_PREFIX)]
    return []


def verify(raw: bytes, sig_b64: str, keys: list[str]) -> None:
    """The catalog's exact bytes must carry a signature from one of the keys."""
    if not keys:
        raise CatalogError("no publisher key yet - add yours in Settings → Catalog before installing anything")
    try:
        sig = base64.b64decode(sig_b64.strip(), validate=True)
    except Exception as e:
        raise CatalogError("the signature file is damaged") from e
    for k in keys:
        try:
            parse_key(k).verify(sig, raw)
            return
        except InvalidSignature:
            continue
    raise CatalogError("the catalog's signature doesn't match your publisher key - not trusted, nothing installed")


def check(raw: bytes, last_version: int | None, now: datetime | None = None) -> dict:
    """After the signature: a catalog PRISM can use (format, not older, not expired)."""
    try:
        doc = json.loads(raw)
    except ValueError as e:
        raise CatalogError("the catalog isn't valid JSON") from e
    if doc.get("format") != 1 or not isinstance(doc.get("entries"), list):
        raise CatalogError("not a PRISM catalog (format 1 with entries)")
    version = doc.get("version")
    if not isinstance(version, int):
        raise CatalogError("the catalog has no version number")
    if last_version is not None and version < last_version:
        raise CatalogError(f"this catalog (version {version}) is older than one already accepted ({last_version}) - refused")
    try:
        expires = datetime.fromisoformat(str(doc["expires"]).replace("Z", "+00:00"))
    except Exception as e:
        raise CatalogError("the catalog has no valid expiry date") from e
    if expires <= (now or datetime.now(timezone.utc)):
        raise CatalogError(f"the catalog expired on {expires:%Y-%m-%d} - sign a fresh one with prism-publish")
    return doc


async def fetch(url: str, keys: list[str], last_version: int | None) -> tuple[dict, bytes]:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as http:
        r = await http.get(url)
        s = await http.get(url + ".sig")
    if r.status_code != 200:
        raise CatalogError(f"{url} answered {r.status_code}")
    if s.status_code != 200:
        raise CatalogError(f"no signature at {url}.sig ({s.status_code}) - sign the catalog with prism-publish")
    verify(r.content, s.text, keys)
    return check(r.content, last_version), r.content


# --- this computer ---------------------------------------------------------------------------

def platform_key() -> str:
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    osname = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    return f"{osname}-{arch}"


def file_for(entry: dict) -> dict | None:
    files = entry.get("files") or {}
    return files.get(platform_key()) or files.get("any")


# --- installing ------------------------------------------------------------------------------

_installs: dict[str, dict] = {}
_tasks: set[asyncio.Task] = set()


def installs() -> dict[str, dict]:
    return {k: dict(v) for k, v in _installs.items()}


def _safe_extract(archive: Path, target: Path) -> None:
    root = target.resolve()

    def inside(name: str) -> bool:
        dest = (target / name).resolve()
        return dest == root or root in dest.parents

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            if not all(inside(m.filename) for m in z.infolist()):
                raise CatalogError("the archive has files outside its folder - refused")
            z.extractall(target)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            members = t.getmembers()
            if not all(inside(m.name) and not m.issym() and not m.islnk() and not m.isdev() for m in members):
                raise CatalogError("the archive has links or files outside its folder - refused")
            t.extractall(target)
    else:
        raise CatalogError("not a zip or tar archive")


def start_install(entry: dict, catalog_url: str, home: Path, on_done, install_python=None) -> None:
    """Download, check the SHA-256, then install a Python package into the sandbox
    (install_python(path)) or hand a model to Ollama; on_done(name, record) records it.
    Progress in installs()."""
    name = entry["name"]
    if name in _installs and not _installs[name]["done"]:
        return
    f = file_for(entry)
    if not f:
        raise CatalogError(f"{name} has no download for this computer ({platform_key()})")
    p = _installs[name] = {"status": "downloading", "completed": 0, "total": int(f.get("size") or 0),
                           "error": None, "done": False}

    async def run() -> None:
        url = urljoin(catalog_url, f["url"])
        downloads = home / "downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        part = downloads / f"{name}-{entry['version']}.part"
        try:
            h = hashlib.sha256()
            async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=300), follow_redirects=True) as http:
                async with http.stream("GET", url) as r:
                    if r.status_code != 200:
                        raise CatalogError(f"the download answered {r.status_code}")
                    with open(part, "wb") as out:
                        async for block in r.aiter_bytes(1 << 16):
                            out.write(block)
                            h.update(block)
                            p["completed"] += len(block)
            if h.hexdigest() != str(f["sha256"]).lower():
                raise CatalogError("the download doesn't match the catalog's checksum - deleted, nothing installed")
            p["status"] = "installing"
            if entry.get("kind") == "model":
                from . import local
                models = home / "models"
                models.mkdir(parents=True, exist_ok=True)
                gguf = models / f"{name}-{entry['version']}.gguf"
                part.replace(gguf)
                ollama_name = local.start_import(str(gguf))
                while not local.pulls().get(ollama_name, {}).get("done"):
                    await asyncio.sleep(1)
                    p["status"] = "adding to Ollama: " + local.pulls().get(ollama_name, {}).get("status", "")
                if err := local.pulls()[ollama_name].get("error"):
                    raise CatalogError(err)
                record = {"version": entry["version"], "kind": "model", "path": str(gguf),
                          "ollama": ollama_name, "as": entry.get("as") or name, "commands": {}}
            else:  # a Python package: a wheel or source archive, into the sandbox
                if entry.get("kind") != "python" or install_python is None:
                    raise CatalogError(f"{name}: unknown kind {entry.get('kind')!r} (python or model)")
                packages = home / "packages"
                packages.mkdir(parents=True, exist_ok=True)
                fname = Path(f["url"].split("?")[0]).name
                if not fname.endswith((".whl", ".tar.gz", ".zip")):
                    raise CatalogError(f"{name}: a Python package is a .whl, .tar.gz or .zip")
                dest = packages / fname
                part.replace(dest)
                await asyncio.to_thread(install_python, str(dest))
                record = {"version": entry["version"], "kind": "python", "path": str(dest),
                          "package": entry["name"]}  # the entry is named after its pip package
            on_done(name, record)
            p["status"] = "installed"
        except Exception as e:
            p["error"] = str(e) or type(e).__name__
            part.unlink(missing_ok=True)
        finally:
            p["done"] = True

    task = asyncio.create_task(run())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


def remove(record: dict, uninstall_python=None) -> None:
    if record.get("kind") == "python" and uninstall_python and record.get("package"):
        uninstall_python(record["package"])
    if record.get("path"):
        Path(record["path"]).unlink(missing_ok=True)
