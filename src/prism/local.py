"""Local models: Ollama and LM Studio on this computer, and models that run inside the PRISM
page itself (WebGPU). They join the room like any other bot, under a name the user picks.

- Ollama / LM Studio: PRISM finds them on their usual ports, lists the models they have, and
  (Ollama) downloads new ones with progress. They answer over HTTP on this machine.
- In the browser: the model runs in the PRISM tab (WebLLM). The meeting engine runs in the
  PRISM server, so a turn becomes a job; the page picks up jobs for the models it has loaded,
  runs them and posts the answer back.

Local models remember nothing between calls, so each turn sends the room history; small
models have small context windows, so the history is trimmed to fit (newest kept).
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field

import httpx

from .providers.base import Message, Provider, ProviderError

log = logging.getLogger(__name__)

RUNTIMES = {
    "ollama": {"label": "Ollama", "url": "http://localhost:11434",
               "install": "https://ollama.com/download"},
    "lmstudio": {"label": "LM Studio", "url": "http://localhost:1234",
                 "install": "https://lmstudio.ai"},
}
# Suggestions for a first download: small enough for a laptop, good at following instructions.
SUGGESTED = ["llama3.2:3b", "qwen2.5:7b", "gemma3:4b", "mistral:7b", "phi4-mini", "deepseek-r1:7b"]
OLLAMA_CTX = 8192          # tokens Ollama keeps per request (its default is smaller)
LOCAL_MAX_CHARS = 24_000   # about 6k tokens of history, so the system prompt is never cut off
BROWSER_MAX_CHARS = 12_000  # in-browser models have the smallest windows


def fit(messages: list[Message], system: str | None, max_chars: int) -> list[Message]:
    """System prompt first, then as many of the newest messages as fit; the newest message
    is kept even when it alone is too long (its start is cut)."""
    sys_msg = [{"role": "system", "content": system[:max_chars // 3]}] if system else []
    budget = max_chars - sum(len(m["content"]) for m in sys_msg)
    kept: list[Message] = []
    for m in reversed(messages):
        if kept and len(m["content"]) > budget:
            break
        content = m["content"] if len(m["content"]) <= budget else "…" + m["content"][-max(budget, 1000):]
        kept.append({"role": m["role"], "content": content})
        budget -= len(content)
    return sys_msg + list(reversed(kept))


def free_ram_mb() -> int | None:
    if sys.platform == "win32":
        from .browser.manager import free_memory_mb
        return free_memory_mb()
    try:  # Linux: what can be had without swapping (free alone leaves out the page cache)
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None  # macOS: no simple counter


# --- finding what is installed ------------------------------------------------------------

async def detect() -> list[dict]:
    """Each runtime: running or not, and the models it has."""
    async with httpx.AsyncClient(timeout=3) as http:
        async def ollama() -> dict:
            base = RUNTIMES["ollama"]["url"]
            try:
                r = await http.get(f"{base}/api/tags")
                r.raise_for_status()
                models = [{"name": m["name"], "size_mb": round(m.get("size", 0) / 1024 / 1024)}
                          for m in r.json().get("models", [])]
                return {"id": "ollama", **RUNTIMES["ollama"], "running": True, "models": models}
            except Exception:
                return {"id": "ollama", **RUNTIMES["ollama"], "running": False, "models": []}

        async def lmstudio() -> dict:
            base = RUNTIMES["lmstudio"]["url"]
            try:
                r = await http.get(f"{base}/v1/models")
                r.raise_for_status()
                models = [{"name": m["id"], "size_mb": None} for m in r.json().get("data", [])
                          if "embed" not in m["id"]]
                return {"id": "lmstudio", **RUNTIMES["lmstudio"], "running": True, "models": models}
            except Exception:
                return {"id": "lmstudio", **RUNTIMES["lmstudio"], "running": False, "models": []}

        return list(await asyncio.gather(ollama(), lmstudio()))


# --- downloading a model with Ollama ---------------------------------------------------------

_pulls: dict[str, dict] = {}   # model -> {status, completed, total, error, done}
_pull_tasks: set[asyncio.Task] = set()


def pulls() -> dict[str, dict]:
    return {k: dict(v) for k, v in _pulls.items()}


def start_pull(model: str) -> None:
    """Ollama downloads it; progress is kept in pulls() for the Settings page to show."""
    if model in _pulls and not _pulls[model].get("done"):
        return
    _pulls[model] = {"status": "starting", "completed": 0, "total": 0, "error": None, "done": False}

    async def run() -> None:
        p = _pulls[model]
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=600)) as http:
                async with http.stream("POST", f"{RUNTIMES['ollama']['url']}/api/pull",
                                       json={"model": model, "stream": True}) as r:
                    if r.status_code != 200:
                        raise ProviderError(f"Ollama answered {r.status_code}: {(await r.aread())[:200]!r}")
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        ev = json.loads(line)
                        if ev.get("error"):
                            raise ProviderError(ev["error"])
                        p["status"] = ev.get("status", p["status"])
                        if ev.get("total"):
                            p["total"], p["completed"] = ev["total"], ev.get("completed", 0)
        except httpx.ConnectError:
            p["error"] = "Ollama isn't running - start it, then try again"
        except Exception as e:
            p["error"] = str(e) or type(e).__name__
        finally:
            p["done"] = True

    task = asyncio.create_task(run())
    _pull_tasks.add(task)
    task.add_done_callback(_pull_tasks.discard)


# --- talking to a local model ----------------------------------------------------------------

class OllamaProvider(Provider):
    """Ollama's own chat API: unlike its OpenAI-style one, it lets PRISM set the context
    window, so a long room isn't silently cut from the front (where the instructions are)."""

    def __init__(self, name: str, model: str, base_url: str | None = None):
        self.name, self.model = name, model
        self.base_url = (base_url or RUNTIMES["ollama"]["url"]).rstrip("/").removesuffix("/v1")

    async def complete(self, messages: list[Message], system: str | None = None) -> str:
        body = {"model": self.model, "messages": fit(messages, system, LOCAL_MAX_CHARS),
                "stream": False, "options": {"num_ctx": OLLAMA_CTX}}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=900)) as http:
                r = await http.post(f"{self.base_url}/api/chat", json=body)
        except httpx.ConnectError:
            raise ProviderError("Ollama isn't running - start it (or open Settings → Local models)")
        if r.status_code == 404:
            raise ProviderError(f"Ollama doesn't have {self.model} - download it in Settings → Local models")
        if r.status_code != 200:
            raise ProviderError(f"Ollama answered {r.status_code}: {r.text[:200]}")
        return r.json().get("message", {}).get("content", "")


class LMStudioProvider(Provider):
    def __init__(self, name: str, model: str, base_url: str | None = None):
        self.name, self.model = name, model
        base = (base_url or RUNTIMES["lmstudio"]["url"]).rstrip("/")
        self.base_url = base if base.endswith("/v1") else base + "/v1"

    async def complete(self, messages: list[Message], system: str | None = None) -> str:
        body = {"model": self.model, "messages": fit(messages, system, LOCAL_MAX_CHARS)}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=900)) as http:
                r = await http.post(f"{self.base_url}/chat/completions", json=body)
        except httpx.ConnectError:
            raise ProviderError("LM Studio's server isn't running - start it in LM Studio (Developer → Start server)")
        if r.status_code != 200:
            raise ProviderError(f"LM Studio answered {r.status_code}: {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"] or ""


# --- models running in the PRISM page --------------------------------------------------------

CLAIM_TIMEOUT_S = 90  # no open PRISM page took the job by then


@dataclass
class Job:
    id: str
    model: str
    messages: list[Message]
    created: float = field(default_factory=time.monotonic)
    claimed: float | None = None
    result: asyncio.Future | None = None


class ClientJobs:
    """Turns for in-browser models, waiting for a PRISM page that has the model loaded."""

    def __init__(self):
        self.jobs: dict[str, Job] = {}

    def take(self, models: list[str]) -> list[dict]:
        """A page with these models loaded takes the waiting jobs for them (each job once)."""
        out = []
        for job in self.jobs.values():
            if job.claimed is None and job.model in models:
                job.claimed = time.monotonic()
                out.append({"id": job.id, "model": job.model, "messages": job.messages})
        return out

    def waiting_models(self) -> list[str]:
        return sorted({j.model for j in self.jobs.values() if j.claimed is None})

    def finish(self, job_id: str, text: str | None, error: str | None) -> bool:
        """The first answer wins (two tabs may both have run it)."""
        job = self.jobs.get(job_id)
        if not job or job.result.done():
            return False
        if error:
            job.result.set_exception(ProviderError(f"in-browser model: {error}"))
        else:
            job.result.set_result(text or "")
        return True

    async def run(self, model: str, messages: list[Message], timeout: float) -> str:
        job = Job(uuid.uuid4().hex, model, messages, result=asyncio.get_running_loop().create_future())
        self.jobs[job.id] = job
        try:
            deadline = time.monotonic() + timeout
            while not job.result.done():
                if job.claimed is None and time.monotonic() - job.created > CLAIM_TIMEOUT_S:
                    raise ProviderError(
                        f"no PRISM page is running {model} - open PRISM in Edge or Chrome on this "
                        "computer and keep the tab open (the model runs there)")
                if time.monotonic() > deadline:
                    raise ProviderError("the in-browser model didn't answer in time (was the PRISM tab closed?)")
                try:
                    return await asyncio.wait_for(asyncio.shield(job.result), 1.0)
                except asyncio.TimeoutError:
                    continue
            return job.result.result()
        finally:
            self.jobs.pop(job.id, None)


client_jobs = ClientJobs()


class BrowserModelProvider(Provider):
    def __init__(self, name: str, model: str, timeout: float = 900):
        self.name, self.model, self.timeout = name, model, timeout

    async def complete(self, messages: list[Message], system: str | None = None) -> str:
        return await client_jobs.run(self.model, fit(messages, system, BROWSER_MAX_CHARS), self.timeout)


def local_provider(cred: dict, timeout: float = 900) -> Provider:
    runtime = (cred.get("base_url") or "").strip()
    if cred["transport"] == "webgpu":
        return BrowserModelProvider(cred["platform"], cred["model"], timeout)
    if runtime.startswith("lmstudio") or ":1234" in runtime:
        return LMStudioProvider(cred["platform"], cred["model"],
                                None if runtime.startswith("lmstudio") else runtime)
    return OllamaProvider(cred["platform"], cred["model"],
                          None if runtime in ("", "ollama") else runtime)
