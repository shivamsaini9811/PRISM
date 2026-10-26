"""PRISM's own engine for models on this computer: llama.cpp's llama-server, shipped with PRISM.

Models are downloaded on first use from Hugging Face - each pinned to a revision and checked
against its SHA-256 - into ~/.prism/models. One model runs at a time (memory): asking another
one stops the first and starts the server with the new one. The server is started when a
built-in model is asked something, on a free port, with a random key so no web page in the
browser can use it, and stopped after IDLE_S without use.

Builds: "cpu" (default; as fast as the graphics on a laptop for small models) and "vulkan"
(graphics: Intel/AMD/NVIDIA) on Windows; "metal" on a Mac.
"""

import asyncio
import hashlib
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .local import LMStudioProvider
from .providers.base import Message, Provider, ProviderError

NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
IDLE_S = 600.0          # no request this long: the server stops (memory back)
START_TIMEOUT_S = 180.0
CONTEXT = 8192
SPARE_DISK = 1 << 30    # a download must leave this much free


@dataclass(frozen=True)
class Model:
    id: str
    label: str
    repo: str
    revision: str
    file: str
    sha256: str
    size: int
    ram_gb: float
    license: str
    good_for: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.file}"


# Checked 2026-10-26: not gated, sizes and SHA-256 from Hugging Face, pinned to a revision.
MODELS = [
    Model("qwen2.5-1.5b", "Qwen 2.5 1.5B", "Qwen/Qwen2.5-1.5B-Instruct-GGUF", "91cad51170dc346986eccefdc2dd33a9da36ead9",
          "qwen2.5-1.5b-instruct-q4_k_m.gguf", "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
          1117320736, 2.5, "Apache 2.0", "fastest; quick, simple tasks"),
    Model("llama3.2-3b", "Llama 3.2 3B", "bartowski/Llama-3.2-3B-Instruct-GGUF", "5ab33fa94d1d04e903623ae72c95d1696f09f9e8",
          "Llama-3.2-3B-Instruct-Q4_K_M.gguf", "6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff",
          2019377696, 4.0, "Llama 3.2 Community License", "general chat"),
    Model("qwen2.5-3b", "Qwen 2.5 3B", "Qwen/Qwen2.5-3B-Instruct-GGUF", "7dabda4d13d513e3e842b20f0d435c732f172cbe",
          "qwen2.5-3b-instruct-q4_k_m.gguf", "626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d",
          2104932768, 4.0, "Qwen Research License", "general, code"),
    Model("gemma3-4b", "Gemma 3 4B", "bartowski/google_gemma-3-4b-it-GGUF", "71506238f970075ca85125cd749c28b1b0eee84e",
          "google_gemma-3-4b-it-Q4_K_M.gguf", "4996030242583a40aa151ff93f49ed787ac8c25e4120c3ae4588b2e2a7d1ae94",
          2489757856, 5.0, "Gemma Terms of Use", "general, writing"),
    Model("qwen3-4b", "Qwen 3 4B", "Qwen/Qwen3-4B-GGUF", "bc640142c66e1fdd12af0bd68f40445458f3869b",
          "Qwen3-4B-Q4_K_M.gguf", "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
          2497281120, 5.0, "Apache 2.0", "reasoning (thinks before answering)"),
    Model("qwen2.5-7b", "Qwen 2.5 7B", "bartowski/Qwen2.5-7B-Instruct-GGUF", "8911e8a47f92bac19d6f5c64a2e2095bd2f7d031",
          "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423",
          4683073952, 7.5, "Apache 2.0", "better quality, slower"),
    Model("qwen3-8b", "Qwen 3 8B", "Qwen/Qwen3-8B-GGUF", "7c41481f57cb95916b40956ab2f0b139b296d974",
          "Qwen3-8B-Q4_K_M.gguf", "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
          5027783488, 8.0, "Apache 2.0", "best of these, slowest"),
]
BY_ID = {m.id: m for m in MODELS}


def models_dir(home: Path) -> Path:
    return home / "models"


def installed(home: Path) -> set[str]:
    """Models whose file is there and was checked (a .ok marker is written after the check)."""
    d = models_dir(home)
    return {m.id for m in BY_ID.values() if (d / m.file).is_file() and (d / (m.file + ".ok")).is_file()}


# --- downloads ----------------------------------------------------------------------------------

_downloads: dict[str, dict] = {}
_tasks: set[asyncio.Task] = set()


def downloads() -> dict[str, dict]:
    return {k: dict(v) for k, v in _downloads.items()}


def start_download(model_id: str, home: Path, url: str | None = None) -> None:
    m = BY_ID[model_id]
    if model_id in _downloads and not _downloads[model_id]["done"]:
        return
    d = models_dir(home)
    d.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(d).free
    if free < m.size + SPARE_DISK:
        raise ProviderError(f"{m.label} needs {m.size / 1e9:.1f} GB and the disk has {free / 1e9:.1f} GB free "
                            "- free some space first")
    p = _downloads[model_id] = {"completed": 0, "total": m.size, "error": None, "done": False}

    async def run() -> None:
        part = d / (m.file + ".part")
        try:
            h = hashlib.sha256()
            async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=120), follow_redirects=True) as http:
                async with http.stream("GET", url or m.url) as r:
                    if r.status_code != 200:
                        raise ProviderError(f"the download answered {r.status_code}")
                    with open(part, "wb") as f:
                        async for block in r.aiter_bytes(1 << 20):
                            f.write(block)
                            h.update(block)
                            p["completed"] += len(block)
            if h.hexdigest() != m.sha256:
                raise ProviderError("the download doesn't match its published checksum - deleted, try again")
            part.replace(d / m.file)
            (d / (m.file + ".ok")).write_text(m.sha256)
        except Exception as e:
            p["error"] = str(e) or type(e).__name__
            part.unlink(missing_ok=True)
        finally:
            p["done"] = True

    task = asyncio.create_task(run())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


def remove(model_id: str, home: Path) -> None:
    m = BY_ID[model_id]
    for f in (m.file, m.file + ".ok", m.file + ".part"):
        (models_dir(home) / f).unlink(missing_ok=True)


# --- the server ---------------------------------------------------------------------------------

def server_binary(build: str) -> Path | None:
    """llama-server: PRISM_LLAMA_SERVER, else shipped next to the app (llama/<build>/)."""
    if env := os.environ.get("PRISM_LLAMA_SERVER"):
        return Path(env)
    exe = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    bases = []
    if getattr(sys, "frozen", False):
        app = Path(sys.executable).parent
        bases = [app / "llama", app.parent / "Resources" / "llama"]
    else:
        bases = [Path(sys.prefix).parent / "llama", Path(__file__).resolve().parents[2] / "llama"]
    for base in bases:
        for b in [build, "metal", "cpu", "vulkan"]:
            if (base / b / exe).is_file():
                return base / b / exe
    return None


def builds_available() -> list[str]:
    exe = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    found = server_binary("cpu")
    if not found:
        return []
    base = found.parent.parent
    return [b for b in ("cpu", "vulkan", "metal") if (base / b / exe).is_file()] or ["cpu"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Engine:
    """One llama-server at a time, started when needed, stopped when idle or asked."""

    def __init__(self, home: Path, build: str = "cpu"):
        self.home, self.build = home, build
        self.proc: subprocess.Popen | None = None
        self.model: str | None = None
        self.port = 0
        self.key = ""
        self.last_used = 0.0
        self.lock = asyncio.Lock()
        self.busy = 0
        self._idle_task: asyncio.Task | None = None

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def status(self) -> dict:
        return {"running": self.running(), "model": self.model if self.running() else None, "build": self.build,
                "binary": str(server_binary(self.build) or ""), "builds": builds_available()}

    async def ensure(self, model_id: str) -> tuple[str, str]:
        """The server with this model, started if need be: (base url, key). Counts the caller
        as busy until it calls done() - another model waits for it rather than cutting it off."""
        async with self.lock:
            self.last_used = time.monotonic()
            while self.running() and self.model != model_id and self.busy:
                await asyncio.sleep(0.5)  # another model is answering: switch when it is done
            if self.running() and self.model == model_id:
                self.busy += 1
                return f"http://127.0.0.1:{self.port}/v1", self.key
            await self._stop()
            m = BY_ID.get(model_id)
            if not m or model_id not in installed(self.home):
                raise ProviderError(f"{m.label if m else model_id} isn't downloaded - Settings → Local models")
            exe = server_binary(self.build)
            if not exe:
                raise ProviderError("PRISM's model engine is missing from this install - reinstall PRISM")
            self.port, self.key = free_port(), secrets.token_urlsafe(24)
            log = self.home / "models" / "engine.log"
            args = [str(exe), "-m", str(models_dir(self.home) / m.file), "--host", "127.0.0.1",
                    "--port", str(self.port), "-c", str(CONTEXT), "--api-key", self.key,
                    "-ngl", "0" if exe.parent.name == "cpu" else "99", "--reasoning-format", "deepseek"]
            with open(log, "w") as out:
                self.proc = subprocess.Popen(args, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                             creationflags=NO_WINDOW)
            self.model = model_id
            try:
                await self._wait_ready(log)
            except Exception:
                await self._stop()
                raise
            if not self._idle_task or self._idle_task.done():
                self._idle_task = asyncio.create_task(self._idle_stop())
            self.busy += 1
            return f"http://127.0.0.1:{self.port}/v1", self.key

    def done(self) -> None:
        self.busy = max(0, self.busy - 1)
        self.last_used = time.monotonic()

    async def _wait_ready(self, log: Path) -> None:
        """Ready when /v1/models answers with our key - so it is our server on that port,
        not another program that happens to listen there."""
        deadline = time.monotonic() + START_TIMEOUT_S
        async with httpx.AsyncClient(timeout=3) as http:
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    tail = log.read_text(errors="replace").strip().splitlines()[-4:] if log.exists() else []
                    raise ProviderError("the model engine stopped while starting: " + " | ".join(tail)[-400:]
                                        + (" - try the CPU build (Settings → Local models)" if self.build != "cpu" else ""))
                try:
                    r = await http.get(f"http://127.0.0.1:{self.port}/v1/models",
                                       headers={"Authorization": f"Bearer {self.key}"})
                    if r.status_code == 200 and "data" in r.json():
                        return
                except (httpx.HTTPError, ValueError):
                    pass
                await asyncio.sleep(0.5)
        raise ProviderError(f"the model engine didn't start within {round(START_TIMEOUT_S)} s")

    async def _idle_stop(self) -> None:
        while self.running():
            await asyncio.sleep(30)
            if not self.busy and time.monotonic() - self.last_used > IDLE_S:
                async with self.lock:
                    if not self.busy and time.monotonic() - self.last_used > IDLE_S:
                        await self._stop()

    async def _stop(self) -> None:
        proc, self.proc, self.model = self.proc, None, None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                await asyncio.to_thread(proc.wait, 10)
            except subprocess.TimeoutExpired:
                proc.kill()

    async def stop(self) -> None:
        async with self.lock:
            await self._stop()

    def kill(self) -> None:
        """PRISM is closing: no model server is left behind."""
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                pass


class BuiltinProvider(Provider):
    """A model run by PRISM's own engine; talks to it as an OpenAI-compatible server."""

    def __init__(self, name: str, model_id: str, engine: Engine):
        self.name, self.model, self.engine = name, model_id, engine
        self._client: LMStudioProvider | None = None

    @property
    def last_activity(self) -> float | None:
        return self._client.last_activity if self._client else None

    async def complete(self, messages: list[Message], system: str | None = None) -> str:
        base, key = await self.engine.ensure(self.model)
        try:
            self._client = LMStudioProvider(self.name, self.model, base, key)
            self._client.label = f"PRISM's engine ({BY_ID[self.model].label})"
            return await self._client.complete(messages, system)
        finally:
            self.engine.done()
