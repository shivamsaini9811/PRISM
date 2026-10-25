"""Readers for what isn't text: a vision model reads images and scanned pages (the words in
them, then what they show), a speech model transcribes audio and video, and an embedding
model adds meaning search. Each one is found on this computer first (Ollama, LM Studio,
whisper.cpp) and otherwise taken from an API key the user saved; none is required - without
one, those files are found by name only and the Context panel says so.

Reader ids: "ollama:<model>", "lmstudio:<model>", "api:<platform>", "whisper.cpp",
"faster-whisper". The user's choice is kept in meta `readers`; "auto" takes the first found.
"""

import asyncio
import base64
import io
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from array import array
from pathlib import Path

import httpx

from .local import RUNTIMES

log = logging.getLogger(__name__)

VISION_PROMPT = ("This image is being indexed so it can be found by searching. First copy out "
                 "every piece of text in it exactly, line by line (write 'No text.' if there is "
                 "none). Then, under 'Shows:', describe in a few sentences what it shows: the "
                 "kind of image, people, objects, charts and what their data says. Plain text only.")
TRANSCRIBE_PROMPT = ("Transcribe the speech in this recording word for word, as plain text with "
                     "a new paragraph when the speaker or topic changes. For a video, add short "
                     "[on screen: ...] notes for text or slides shown.")
VISION_NAMES = ("llava", "vision", "gemma3", "qwen2.5vl", "qwen2.5-vl", "qwen3-vl", "minicpm-v",
                "moondream", "llama4", "mistral-small3", "granite3.2-vision", "bakllava")
EMBED_NAMES = ("embed", "bge-", "minilm", "e5-", "gte-", "arctic-embed")
API_VISION = ("claude", "gpt", "gemini", "grok")
OPENAI_URL = "https://api.openai.com/v1"
ANTHROPIC_URL = "https://api.anthropic.com/v1"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
INLINE_MAX = 18 * 1024 * 1024    # Gemini: request size limit for inline files
OPENAI_AUDIO_MAX = 25 * 1024 * 1024
DEFAULT_CONFIG = {"vision": "auto", "transcribe": "auto", "embed": ""}
SUGGESTED_EMBED = "nomic-embed-text"
SUGGESTED_VISION = "gemma3:4b"
NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0  # no console flashing up on Windows


class ReaderError(Exception):
    pass


# --- what is on this computer ------------------------------------------------------------------

def tools_dir(home: Path) -> Path:
    return home / "tools"


def find_tool(home: Path, *names: str) -> str | None:
    """On PATH, or downloaded into PRISM's tools folder."""
    for name in names:
        if found := shutil.which(name):
            return found
    exe = ".exe" if sys.platform == "win32" else ""
    if tools_dir(home).exists():
        for name in names:
            for p in tools_dir(home).rglob(name + exe):
                if p.is_file():
                    return str(p)
    return None


def whisper_bin(home: Path) -> str | None:
    return find_tool(home, "whisper-cli", "whisper-cpp")


def whisper_model(home: Path) -> str | None:
    for folder in (tools_dir(home) / "whisper", tools_dir(home)):
        models = sorted(folder.glob("ggml-*.bin")) if folder.exists() else []
        if models:
            return str(models[0])
    return None


def ffmpeg_bin(home: Path) -> str | None:
    return find_tool(home, "ffmpeg")


def has_faster_whisper() -> bool:
    try:
        import faster_whisper  # noqa: F401 - only in installs from source that added it
        return True
    except Exception:
        return False


async def detect(home: Path, creds: dict[str, dict]) -> dict:
    """Every reader PRISM could use now, by job."""
    vision, embed, transcribe = [], [], []
    async with httpx.AsyncClient(timeout=6) as http:
        try:
            base = RUNTIMES["ollama"]["url"]
            tags = (await http.get(f"{base}/api/tags")).json().get("models", [])

            async def caps(name: str) -> list[str]:
                try:
                    r = await http.post(f"{base}/api/show", json={"model": name})
                    return r.json().get("capabilities") or []
                except Exception:
                    return []
            names = [m["name"] for m in tags][:40]
            for name, c in zip(names, await asyncio.gather(*(caps(n) for n in names))):
                low = name.lower()
                if "vision" in c or (not c and any(v in low for v in VISION_NAMES)):
                    vision.append({"id": f"ollama:{name}", "label": f"{name} (Ollama, on this computer)"})
                if "embedding" in c or any(v in low for v in EMBED_NAMES):
                    embed.append({"id": f"ollama:{name}", "label": f"{name} (Ollama)"})
        except Exception:
            pass
        try:
            base = RUNTIMES["lmstudio"]["url"]
            r = await http.get(f"{base}/api/v0/models")
            for m in r.json().get("data", []):
                if m.get("type") == "vlm":
                    vision.append({"id": f"lmstudio:{m['id']}", "label": f"{m['id']} (LM Studio)"})
                elif m.get("type") == "embeddings":
                    embed.append({"id": f"lmstudio:{m['id']}", "label": f"{m['id']} (LM Studio)"})
        except Exception:
            pass
    if whisper_bin(home) and whisper_model(home):
        transcribe.append({"id": "whisper.cpp", "label": f"whisper.cpp ({Path(whisper_model(home)).name}, on this computer)"})
    if has_faster_whisper():
        transcribe.append({"id": "faster-whisper", "label": "faster-whisper (on this computer)"})
    for platform, c in creds.items():
        if c.get("transport") != "api" or not c.get("secret"):
            continue
        if platform in API_VISION:
            vision.append({"id": f"api:{platform}", "label": f"{platform} API ({c['model']}) - uses your API credit"})
        if platform == "gpt" and not c.get("base_url"):
            transcribe.append({"id": "api:gpt", "label": "OpenAI API (whisper-1) - uses your API credit"})
        if platform == "gemini":
            transcribe.append({"id": "api:gemini", "label": f"Gemini API ({c['model']}) - uses your API credit"})
    return {"vision": vision, "transcribe": transcribe, "embed": embed,
            "tools": {"whisper": whisper_bin(home), "whisper_model": whisper_model(home),
                      "ffmpeg": ffmpeg_bin(home)}}


def pick(choice: str, found: list[dict]) -> str | None:
    """The chosen reader if it is there; with "auto", the first (local ones come first)."""
    ids = [f["id"] for f in found]
    if choice in ("", "off", None):
        return None
    if choice == "auto":
        return ids[0] if ids else None
    return choice if choice in ids else None


# --- images ------------------------------------------------------------------------------------

def to_jpeg(data: bytes | None = None, image=None, max_side: int = 1600) -> bytes:
    from PIL import Image
    img = image if image is not None else Image.open(io.BytesIO(data))
    img.thumbnail((max_side, max_side))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    out = io.BytesIO()
    img.save(out, "JPEG", quality=85)
    return out.getvalue()


def pdf_page_images(path: str, limit: int = 60) -> list[tuple[int, bytes]]:
    """Scanned PDFs: each page is a picture; take the biggest one on each page."""
    import pypdf
    out = []
    for no, page in enumerate(pypdf.PdfReader(path).pages, start=1):
        if len(out) >= limit:
            break
        try:
            images = [im.image for im in page.images]
        except Exception:
            continue
        images = [im for im in images if im is not None]
        if images:
            out.append((no, to_jpeg(image=max(images, key=lambda im: im.width * im.height))))
    return out


async def describe_image(reader: str, jpeg: bytes, creds: dict[str, dict]) -> str:
    b64 = base64.b64encode(jpeg).decode()
    async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=600)) as http:
        if reader.startswith("ollama:"):
            r = await http.post(f"{RUNTIMES['ollama']['url']}/api/chat", json={
                "model": reader.split(":", 1)[1], "stream": False,
                "messages": [{"role": "user", "content": VISION_PROMPT, "images": [b64]}]})
            _check(r, "Ollama")
            return r.json().get("message", {}).get("content", "")
        if reader.startswith("lmstudio:"):
            return await _openai_vision(http, f"{RUNTIMES['lmstudio']['url']}/v1", "lm-studio",
                                        reader.split(":", 1)[1], b64)
        platform = reader.split(":", 1)[1]
        c = creds.get(platform)
        if not c:
            raise ReaderError(f"{platform} has no API key saved (or PRISM is locked)")
        if platform == "claude":
            r = await http.post(f"{c.get('base_url') or ANTHROPIC_URL}/messages", headers={
                "x-api-key": c["secret"], "anthropic-version": "2023-06-01"}, json={
                "model": c["model"], "max_tokens": 1500, "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": VISION_PROMPT}]}]})
            _check(r, "Anthropic")
            return "".join(b.get("text", "") for b in r.json().get("content", []))
        if platform == "gemini":
            return await _gemini(http, c, "image/jpeg", b64, VISION_PROMPT)
        base = c.get("base_url") or ("https://api.x.ai/v1" if platform == "grok" else OPENAI_URL)
        return await _openai_vision(http, base, c["secret"], c["model"], b64)


async def _openai_vision(http: httpx.AsyncClient, base: str, key: str, model: str, b64: str) -> str:
    r = await http.post(f"{base.rstrip('/')}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                        json={"model": model, "messages": [{"role": "user", "content": [
                            {"type": "text", "text": VISION_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]})
    _check(r, "the model")
    return r.json()["choices"][0]["message"]["content"] or ""


async def _gemini(http: httpx.AsyncClient, c: dict, mime: str, b64: str, prompt: str) -> str:
    r = await http.post(f"{c.get('base_url') or GEMINI_URL}/models/{c['model']}:generateContent",
                        headers={"x-goog-api-key": c["secret"]},
                        json={"contents": [{"parts": [{"inline_data": {"mime_type": mime, "data": b64}},
                                                      {"text": prompt}]}]})
    _check(r, "Gemini")
    parts = (r.json().get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def _check(r: httpx.Response, who: str) -> None:
    if r.status_code != 200:
        raise ReaderError(f"{who} answered {r.status_code}: {r.text[:200]}")


# --- audio and video ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: float) -> None:
    p = subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=NO_WINDOW)
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or b"").decode(errors="replace").strip().splitlines()[-3:]
        raise ReaderError(f"{Path(cmd[0]).name} failed: {' '.join(tail)[:300]}")


def _ffmpeg(home: Path, src: str, dst: str, *args: str, timeout: float = 3600) -> None:
    ff = ffmpeg_bin(home)
    if not ff:
        raise ReaderError("needs ffmpeg to read this file - download it in Settings → Readers")
    _run([ff, "-nostdin", "-y", "-loglevel", "error", "-i", src, "-vn", *args, dst], timeout)


def _transcribe_local(reader: str, home: Path, path: str) -> str:
    with tempfile.TemporaryDirectory() as d:
        if reader == "faster-whisper":
            from faster_whisper import WhisperModel
            segments, _ = WhisperModel("base", compute_type="int8").transcribe(path)
            return "\n".join(s.text.strip() for s in segments)
        src = path
        if ffmpeg_bin(home):
            src = str(Path(d) / "audio.wav")
            _ffmpeg(home, path, src, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le")
        elif Path(path).suffix.lower() not in (".wav", ".mp3", ".flac", ".ogg"):
            raise ReaderError("needs ffmpeg to read this file - download it in Settings → Readers")
        _run([whisper_bin(home), "-m", whisper_model(home), "-f", src, "-l", "auto", "-np",
              "-otxt", "-of", str(Path(d) / "out")], timeout=4 * 3600)
        return (Path(d) / "out.txt").read_text(encoding="utf-8", errors="replace")


def _small_audio(home: Path, path: str, limit: int, d: str) -> tuple[str, str]:
    """The file itself if small enough, else its sound as a small mono mp3 (an hour ≈ 14 MB)."""
    if os.path.getsize(path) <= limit:
        return path, MIME.get(Path(path).suffix.lower().lstrip("."), "application/octet-stream")
    out = str(Path(d) / "audio.mp3")
    _ffmpeg(home, path, out, "-ac", "1", "-ar", "16000", "-b:a", "32k")
    if os.path.getsize(out) > limit:
        raise ReaderError("too long to send in one piece (over about an hour of sound)")
    return out, "audio/mpeg"


MIME = {"mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/mp4", "flac": "audio/flac",
        "ogg": "audio/ogg", "opus": "audio/ogg", "aac": "audio/aac", "mp4": "video/mp4",
        "mov": "video/quicktime", "webm": "video/webm", "mkv": "video/x-matroska", "avi": "video/x-msvideo",
        "m4v": "video/mp4", "wmv": "video/x-ms-wmv", "wma": "audio/x-ms-wma"}


async def transcribe(reader: str, home: Path, path: str, creds: dict[str, dict]) -> str:
    if reader in ("whisper.cpp", "faster-whisper"):
        return await asyncio.to_thread(_transcribe_local, reader, home, path)
    platform = reader.split(":", 1)[1]
    c = creds.get(platform)
    if not c:
        raise ReaderError(f"{platform} has no API key saved (or PRISM is locked)")
    with tempfile.TemporaryDirectory() as d:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=1800)) as http:
            if platform == "gemini":
                src, mime = await asyncio.to_thread(_small_audio, home, path, INLINE_MAX, d)
                b64 = base64.b64encode(Path(src).read_bytes()).decode()
                return await _gemini(http, c, mime, b64, TRANSCRIBE_PROMPT)
            src, mime = await asyncio.to_thread(_small_audio, home, path, OPENAI_AUDIO_MAX, d)
            with open(src, "rb") as f:
                r = await http.post(f"{c.get('base_url') or OPENAI_URL}/audio/transcriptions",
                                    headers={"Authorization": f"Bearer {c['secret']}"},
                                    data={"model": "whisper-1", "response_format": "text"},
                                    files={"file": (Path(src).name, f, mime)})
            _check(r, "OpenAI")
            return r.text


# --- meaning search ----------------------------------------------------------------------------

async def embed(spec: str, texts: list[str]) -> list[bytes]:
    """Vectors for the texts, normalised (so cosine = dot product), as float32 bytes."""
    runtime, model = spec.split(":", 1)
    async with httpx.AsyncClient(timeout=httpx.Timeout(10, read=300)) as http:
        if runtime == "ollama":
            r = await http.post(f"{RUNTIMES['ollama']['url']}/api/embed", json={"model": model, "input": texts})
            _check(r, "Ollama")
            vecs = r.json()["embeddings"]
        else:
            r = await http.post(f"{RUNTIMES['lmstudio']['url']}/v1/embeddings", json={"model": model, "input": texts})
            _check(r, "LM Studio")
            vecs = [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d.get("index", 0))]
    out = []
    for v in vecs:
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append(array("f", (x / n for x in v)).tobytes())
    return out


def unpack(vec: bytes) -> array:
    a = array("f")
    a.frombytes(vec)
    return a


# --- downloads (whisper.cpp, its model, ffmpeg) ------------------------------------------------

DOWNLOADS = {
    "whisper": {"label": "whisper.cpp", "windows_only": True,
                "release": "https://api.github.com/repos/ggml-org/whisper.cpp/releases/latest",
                "asset": "whisper-bin-x64.zip", "into": "whisper"},
    "whisper-model": {"label": "speech model (ggml-base, 148 MB)",
                      "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
                      "into": "whisper", "file": "ggml-base.bin"},
    "ffmpeg": {"label": "ffmpeg", "windows_only": True,
               "url": "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip", "into": "ffmpeg"},
}
MAC_HINTS = {"whisper": "brew install whisper-cpp", "ffmpeg": "brew install ffmpeg"}
_downloads: dict[str, dict] = {}
_download_tasks: set[asyncio.Task] = set()


def downloads() -> dict[str, dict]:
    return {k: dict(v) for k, v in _downloads.items()}


def start_download(what: str, home: Path) -> None:
    spec = DOWNLOADS[what]
    if what in _downloads and not _downloads[what]["done"]:
        return
    _downloads[what] = p = {"completed": 0, "total": 0, "error": None, "done": False}

    async def run() -> None:
        target = tools_dir(home) / spec["into"]
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=300), follow_redirects=True) as http:
                url = spec.get("url")
                if "release" in spec:
                    assets = (await http.get(spec["release"])).json().get("assets", [])
                    url = next((a["browser_download_url"] for a in assets if a["name"] == spec["asset"]), None)
                    if not url:
                        raise ReaderError(f"{spec['asset']} is not in the latest release")
                target.mkdir(parents=True, exist_ok=True)
                part = target / ((spec.get("file") or "download.zip") + ".part")
                async with http.stream("GET", url) as r:
                    _check(r, "the download site") if r.status_code != 200 else None
                    p["total"] = int(r.headers.get("content-length") or 0)
                    with open(part, "wb") as f:
                        async for block in r.aiter_bytes(1 << 16):
                            f.write(block)
                            p["completed"] += len(block)
            if spec.get("file"):
                part.replace(target / spec["file"])
            else:
                await asyncio.to_thread(_unzip, part, target)
                part.unlink()
        except Exception as e:
            p["error"] = str(e) or type(e).__name__
        finally:
            p["done"] = True

    task = asyncio.create_task(run())
    _download_tasks.add(task)
    task.add_done_callback(_download_tasks.discard)


def _unzip(path: Path, target: Path) -> None:
    with zipfile.ZipFile(path) as z:
        for m in z.infolist():  # nothing may land outside the target folder
            dest = (target / m.filename).resolve()
            if not str(dest).startswith(str(target.resolve())):
                raise ReaderError("the download has unsafe paths")
        z.extractall(target)


def load_config(raw) -> dict:
    try:
        return {**DEFAULT_CONFIG, **json.loads(raw or "{}")}
    except ValueError:
        return dict(DEFAULT_CONFIG)
