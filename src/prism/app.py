"""FastAPI web app: vault unlock, credentials, mother/daughter chats, context store."""

import asyncio
import json
import logging
import re
import sys
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from cryptography.fernet import Fernet
import httpx
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import __version__
from .browser import SITES, BrowserManager, BrowserProvider, SiteAdapter, SiteConfig
from .config import Settings, data_dir
from .crypto import WrongPassword
from .db import Store, context_name
from .meeting import (BUDGETS, FORMATS, MAX_ROUNDS, MODERATOR, Entry, Member, Session,
                      context_block, make_script)
from . import enrich, flows, local, readers
from .library import Library, attachable, passages_text
from .providers import PLATFORMS, Provider, build_provider, wait_reply

STATIC = Path(__file__).parent / "static"
log = logging.getLogger(__name__)



@lru_cache
def settings() -> Settings:
    return Settings()


@lru_cache
def store() -> Store:
    return Store(settings().prism_db_path)


@lru_cache
def browsers() -> BrowserManager:
    channel = settings().prism_browser_channel  # "bundled" forces the Chromium shipped with PRISM
    channels = [None if channel == "bundled" else channel] if channel else None
    return BrowserManager(settings().prism_profiles_dir, headless=settings().prism_browser_headless,
                          channels=channels)


@lru_cache
def library() -> Library:
    def creds() -> dict:
        return {c["platform"]: c for c in store().enabled_credentials(vault())}
    return Library(store(), data_dir(), creds=creds, readers_config=lambda: store().get_meta("readers"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    recover_runs()
    yield
    for task in list(_session_tasks.values()) + list(_run_tasks.values()):
        task.cancel()
    await asyncio.gather(*_session_tasks.values(), *_run_tasks.values(), return_exceptions=True)
    await browsers().close()


app = FastAPI(title="PRISM", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def same_origin_only(request: Request, call_next):
    """Changes come only from PRISM's own page. Another website open in the browser, or a
    preview the bots wrote (its sandbox has the origin "null"), can still send a request to
    127.0.0.1 - refuse it."""
    origin = request.headers.get("origin")
    if request.method not in ("GET", "HEAD", "OPTIONS") and origin:
        host = request.headers.get("host", "")
        if origin == "null" or origin.split("://", 1)[-1] != host:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Not from PRISM's own page"}, status_code=403)
    return await call_next(request)


# The vault key lives only in this process's memory; restarting the server locks it again.
_vault: dict[str, Fernet | None] = {"key": None}


def vault() -> Fernet:
    if _vault["key"] is None:
        raise HTTPException(423, "Vault is locked - unlock with your master password")
    return _vault["key"]


def chat_or_404(chat_id: int) -> dict:
    chat = store().get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    return chat


def _meta_json(key: str) -> dict:
    raw = store().get_meta(key)
    return json.loads(raw) if raw else {}


def site_config(platform: str) -> SiteConfig:
    """Built-in selectors, then what PRISM learned when the site changed its layout, then
    the user's own overrides."""
    return SITES[platform].with_overrides(_meta_json(f"learned:{platform}")) \
        .with_overrides(_meta_json(f"site:{platform}"))


def remember_layout(platform: str, learned: dict) -> None:
    log.warning("%s changed its layout - learned %s", platform, learned)
    store().set_meta(f"learned:{platform}", json.dumps({**_meta_json(f"learned:{platform}"), **learned}))


LAYOUT_ADVICE_TIMEOUT_S = 120


def layout_advisor(platform: str):
    """When PRISM can't find a site's message box itself, it asks another assistant: an API
    one if there is one (fast), else another signed-in site that isn't busy (waiting on a
    busy one could wait on this very site's turn)."""
    async def advise(question: str) -> str:
        try:
            creds = [c for c in store().enabled_credentials(vault()) if c["platform"] != platform]
        except HTTPException:
            creds = []
        creds.sort(key=lambda c: c["transport"] != "api")
        for cred in creds:
            if cred["transport"] == "webgpu":
                continue  # needs an open PRISM page; this may run with none
            if cred["transport"] == "browser" and (cred["platform"] not in SITES
                                                   or browsers().lock(cred["platform"]).locked()):
                continue
            try:
                helper = make_provider(cred, fresh=True, advisor=False)
                return await asyncio.wait_for(
                    helper.complete([{"role": "user", "content": question}]), LAYOUT_ADVICE_TIMEOUT_S)
            except Exception as e:
                log.warning("layout advice from %s failed: %s", cred["platform"], e)
        return ""
    return advise


def media_dir() -> Path:
    """Files the user attaches, and images saved from the bots' replies."""
    return data_dir() / "media"


def site_adapter(platform: str, advisor: bool = True) -> SiteAdapter:
    # Failed page loads leave a screenshot here, named in the error message.
    return SiteAdapter(site_config(platform), screenshot_dir=data_dir() / "screenshots",
                       media_dir=media_dir(),
                       on_learn=lambda learned: remember_layout(platform, learned),
                       advisor=layout_advisor(platform) if advisor else None)


def make_provider(cred: dict, thread_url: str | None = None, fresh: bool = False,
                  resend_system: bool = False, advisor: bool = True) -> Provider:
    """API providers are stateless; browser providers bind to a remote conversation."""
    if cred["transport"] in LOCAL_TRANSPORTS:
        return local.local_provider(cred, settings().reply_max_s)
    if cred["transport"] == "browser":
        if cred["platform"] not in SITES:
            raise ValueError(f"{cred['platform']} has no web app support")
        return BrowserProvider(cred["platform"], browsers(), site_adapter(cred["platform"], advisor),
                               thread_url=thread_url, fresh=fresh, resend_system=resend_system)
    return build_provider(cred["platform"], cred["model"], cred["secret"], cred["base_url"])


# --- UI ----------------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


# --- media: files the user attaches, and pictures and files saved from replies ------------

MEDIA_NAME = re.compile(r"^[\w][\w.\- ]{0,120}$")
MAX_UPLOAD = 100 * 1024 * 1024
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}


def media_path(name: str) -> Path:
    if not MEDIA_NAME.match(name) or ".." in name:
        raise HTTPException(400, f"Bad file name {name!r}")
    path = media_dir() / name
    if not path.is_file():
        raise HTTPException(404, f"No file {name!r} - attach it again")
    return path


@app.post("/api/media")
async def upload(request: Request, name: str) -> dict:
    """The file is the request body (no multipart): saved under a unique name."""
    vault()
    data = await request.body()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Files up to 100 MB")
    shown = re.sub(r"[^\w.\- ]+", "_", Path(name).name).strip(" .")[:80] or "file"
    media_dir().mkdir(parents=True, exist_ok=True)
    stored = f"{uuid.uuid4().hex[:12]}-{shown}"
    (media_dir() / stored).write_bytes(data)
    return {"id": stored, "name": shown, "size": len(data)}


@app.get("/media/{name}")
def get_media(name: str) -> FileResponse:
    path = media_path(name)
    shown = name.split("-", 1)[-1]
    if path.suffix.lower() in IMAGE_EXT:
        return FileResponse(path)
    return FileResponse(path, filename=shown)  # a download, under its own name


class QuitIn(BaseModel):
    confirm: bool


@app.post("/api/quit")
def quit_prism(body: QuitIn) -> dict:
    """The Quit button: the Mac app has no window or Dock icon to quit from. Shutting down
    closes the browser too (lifespan). A JSON body, so another website open in the browser
    can't send it (a cross-site JSON POST needs a CORS preflight PRISM never grants)."""
    if not body.confirm:
        raise HTTPException(400, "Not confirmed")
    server = getattr(app.state, "server", None)
    if server is None:
        raise HTTPException(409, "PRISM was not started by its launcher - stop it where it runs")
    server.should_exit = True
    return {"ok": True}


# --- vault -------------------------------------------------------------------------------

class UnlockRequest(BaseModel):
    password: str


@app.get("/api/status")
def status() -> dict:
    return {"version": __version__, "initialized": store().is_initialized(),
            "unlocked": _vault["key"] is not None}


@app.post("/api/unlock")
def unlock(req: UnlockRequest) -> dict:
    if len(req.password) < 8:
        raise HTTPException(400, "Master password must be at least 8 characters")
    try:
        _vault["key"] = store().unlock(req.password)
    except WrongPassword:
        raise HTTPException(401, "Wrong master password")
    return status()


@app.post("/api/lock")
def lock() -> dict:
    _vault["key"] = None
    return status()


# --- credentials & primary (mother) LLM -------------------------------------------------

class CredentialIn(BaseModel):
    transport: str = "api"  # "api" | "browser"
    model: str | None = None
    secret: str | None = None  # None = keep stored secret, "" = clear it
    base_url: str | None = None
    enabled: bool = True


# Local models: "local" = Ollama / LM Studio on this computer (base_url names which), "webgpu" =
# running inside the PRISM page. Each is a room member under a name the user picks.
LOCAL_TRANSPORTS = ("local", "webgpu")
LOCAL_NAME = re.compile(r"^[a-z][a-z0-9-]{1,23}$")
RESERVED = {"all", "any", "everyone", "moderator", "mod", "mother", "prism", "docs", "context",
            "user", "you"}


def known_platform(platform: str) -> bool:
    return platform in PLATFORMS or any(c["platform"] == platform for c in store().list_credentials())


@app.get("/api/platforms")
def platforms() -> dict:
    catalog = {p: {**spec, "web": p in SITES} for p, spec in PLATFORMS.items()}
    for c in store().list_credentials():
        if c["transport"] in LOCAL_TRANSPORTS:
            where = "in this browser" if c["transport"] == "webgpu" else \
                local.RUNTIMES.get(c["base_url"] or "ollama", {}).get("label", c["base_url"])
            catalog[c["platform"]] = {"label": f"{c['platform']} (local · {c['model']} · {where})",
                                      "kind": "local", "default_model": c["model"], "needs_secret": False,
                                      "web": False, "local": c["transport"]}
    return {"catalog": catalog, "credentials": store().list_credentials(),
            "primary": store().get_meta("primary_platform")}


@app.put("/api/credentials/{platform}")
def put_credential(platform: str, c: CredentialIn) -> dict:
    if c.transport in LOCAL_TRANSPORTS:
        if not LOCAL_NAME.match(platform) or platform in RESERVED or platform in PLATFORMS:
            raise HTTPException(400, f"Pick another name than {platform!r}: lowercase letters, digits "
                                     "and -, starting with a letter, and not a platform's name")
        if not c.model:
            raise HTTPException(400, "Which model? Pick one of the installed models")
        store().upsert_credential(vault(), platform, c.model, None, c.base_url or None, c.enabled, c.transport)
        return platforms()
    if platform not in PLATFORMS:
        raise HTTPException(404, f"Unknown platform {platform!r}")
    if c.transport not in ("api", "browser"):
        raise HTTPException(400, "transport must be 'api', 'browser', 'local' or 'webgpu'")
    if c.transport == "browser" and platform not in SITES:
        raise HTTPException(400, f"{platform} has no web app support")
    store().upsert_credential(vault(), platform, c.model or PLATFORMS[platform]["default_model"],
                              c.secret, c.base_url or None, c.enabled, c.transport)
    return platforms()


# --- browser platforms: log in once, check selectors, override them ---------------------

def _web_platform(platform: str) -> None:
    if platform not in SITES:
        raise HTTPException(404, f"{platform} has no web app support")


@app.post("/api/platforms/{platform}/connect")
async def connect(platform: str, stop: bool = False) -> dict:
    """Open the site in a visible window on PRISM's shared browser profile; the user signs in there.
    Check (or the next message) switches the browser back to headless; the login is kept.
    The window means relaunching the shared browser, which cuts off every bot mid-reply: with a
    meeting running it asks first, and stop=true ends those meetings now."""
    _web_platform(platform)
    running = [cid for cid, m in _sessions.items() if m.phase != "done"]
    if running and not stop:
        titles = ", ".join(repr((store().get_chat(c) or {}).get("title", "?")[:40]) for c in running)
        raise HTTPException(409, f"Signing in restarts PRISM's browser, which cuts off the meeting "
                                 f"running in {titles}. Stop it and open the sign-in window?")
    for cid in running:
        task = _session_tasks.get(cid)
        if task:
            task.cancel()
    await asyncio.gather(*(t for c in running if (t := _session_tasks.get(c))), return_exceptions=True)
    page = await browsers().new_page(visible=True)
    await page.goto(site_config(platform).new_chat_url, wait_until="domcontentloaded")
    return {"opened": page.url}


@app.get("/api/platforms/{platform}/doctor")
async def doctor(platform: str) -> dict:
    _web_platform(platform)
    async with browsers().lock(platform):
        page = await browsers().new_page()
        try:
            return await site_adapter(platform).doctor(page)
        finally:
            await page.close()


@app.get("/api/platforms/{platform}/selectors")
def get_selectors(platform: str) -> dict:
    _web_platform(platform)
    return {"defaults": SITES[platform].to_dict(), "learned": _meta_json(f"learned:{platform}"),
            "overrides": _meta_json(f"site:{platform}"), "effective": site_config(platform).to_dict()}


@app.put("/api/platforms/{platform}/selectors")
def put_selectors(platform: str, overrides: dict) -> dict:
    """Saves only what differs from built-in + learned. {} = reset: forgets what was learned too."""
    _web_platform(platform)
    if not overrides:
        store().set_meta(f"learned:{platform}", json.dumps({}))
    base = SITES[platform].with_overrides(_meta_json(f"learned:{platform}")).to_dict()
    store().set_meta(f"site:{platform}", json.dumps(
        {k: v for k, v in overrides.items() if v and base.get(k) != v}))
    return get_selectors(platform)


@app.delete("/api/credentials/{platform}")
def delete_credential(platform: str) -> dict:
    store().delete_credential(platform)
    return platforms()


class PrimaryIn(BaseModel):
    platform: str


@app.put("/api/primary")
def set_primary(p: PrimaryIn) -> dict:
    if not known_platform(p.platform):
        raise HTTPException(404, f"Unknown platform {p.platform!r}")
    store().set_meta("primary_platform", p.platform)
    return platforms()


# --- local models: Ollama / LM Studio on this computer, and models in the PRISM page --------

@app.get("/api/local")
async def local_models() -> dict:
    return {"runtimes": await local.detect(), "pulls": local.pulls(), "suggested": local.SUGGESTED,
            "free_ram_mb": local.free_ram_mb()}


class PullIn(BaseModel):
    model: str = Field(min_length=1, max_length=120, pattern=r"^[\w.:/-]+$")


@app.post("/api/local/pull")
async def pull_model(body: PullIn) -> dict:
    """Ollama downloads the model; GET /api/local shows the progress."""
    local.start_pull(body.model)
    return {"pulls": local.pulls()}


class ImportIn(BaseModel):
    path: str = Field(min_length=1, max_length=1000)


@app.post("/api/local/import")
async def import_model(body: ImportIn) -> dict:
    """A .gguf already on this computer becomes an Ollama model (progress as for downloads)."""
    try:
        model = local.start_import(body.path)
    except local.ProviderError as e:
        raise HTTPException(400, str(e)) from e
    return {"model": model, "pulls": local.pulls()}


@app.get("/api/client-jobs")
def take_client_jobs(models: str = "") -> dict:
    """A PRISM page with in-browser models loaded takes the turns waiting for them."""
    loaded = [m for m in models.split(",") if m]
    return {"jobs": local.client_jobs.take(loaded), "waiting": local.client_jobs.waiting_models()}


class JobResultIn(BaseModel):
    text: str | None = None
    error: str | None = None


@app.post("/api/client-jobs/{job_id}")
def finish_client_job(job_id: str, body: JobResultIn) -> dict:
    return {"accepted": local.client_jobs.finish(job_id, body.text, body.error)}


# --- chats -------------------------------------------------------------------------------

@app.get("/api/chats")
def list_chats() -> list[dict]:
    return store().list_chats()


@app.post("/api/chats")
def create_chat() -> dict:
    return store().get_chat(store().create_chat())


def chat_view(chat_id: int) -> dict:
    s = store()
    m = _sessions.get(chat_id)
    # Status before messages: a session ending in between must not hide its last message.
    meeting = m.status() if m and m.phase != "done" else None
    return {
        "chat": chat_or_404(chat_id),
        "messages": s.thread(chat_id),
        "daughters": [{**d, "messages": s.thread(chat_id, d["id"])} for d in s.daughters(chat_id)],
        "context": [{**c, "status": library().status(c["library"])} if c.get("library") else c
                    for c in s.context_for(chat_id)],
        "meeting": meeting,
        "queued": [q.content for q in _queued.get(chat_id, [])],
        "meeting_defaults": meeting_defaults(),
        "profile": profile_state(),
        "run": run_view(chat_id),
    }


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: int) -> dict:
    return chat_view(chat_id)


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: int) -> dict:
    m = _sessions.get(chat_id)
    if m and m.phase != "done":
        raise HTTPException(409, "Stop the meeting before deleting the chat")
    if (r := store().run_for_chat(chat_id)) and r["id"] in _run_tasks:
        raise HTTPException(409, "Stop the run before deleting its chat")
    store().delete_chat(chat_id)
    return {"ok": True}


# --- the room: the user, the bots and the moderator in one group chat --------------------

_sessions: dict[int, Session] = {}
_session_tasks: dict[int, asyncio.Task] = {}
_queued: dict[int, list["SendIn"]] = {}  # @mentioned messages waiting for the running session

# speakers: platform -> budget | "off"; format: how @all (or no mention) is run.
DEFAULTS = {"rounds": 3, "speakers": {}, "deep": False, "format": "roundtable", "asks": 3,
            "research": False}
MENTION = re.compile(r"(?<![\w.])@([A-Za-z][\w-]*)")
CONTEXT_REF = re.compile(r"(?<![\w&#])#([A-Za-z][\w-]*)")   # "C#" and "#1" are not references
COMMAND = re.compile(r"^/(save|save-global|forget)\s+#([A-Za-z][\w-]*)\s*(.*)$", re.S)
RECALL = re.compile(r"^/recall\s+@([A-Za-z][\w-]*)\s+#([A-Za-z][\w-]*)\s*(.*)$", re.S)
PROFILE_CMD = re.compile(r"^/profile(?:\s+(on|off|update))?\s*$", re.I)
# A task plan: lines that start with @mentions ("1. @gpt write it", "then @claude review").
PLAN_LINE = re.compile(r"^\s*(?:\d+[.)]\s*|[-*\u2022]\s*|then\s+|->\s*|\u2192\s*)?"
                       r"((?:@[A-Za-z][\w-]*[\s,]*(?:and\s+|&\s*)?)+)(.*)$", re.I)
PLAN_SPLIT = re.compile(r"\s*(?:,?\s*\bthen\b|->|\u2192|;)\s*(?=@)", re.I)
ALIASES = {"everyone": "all", "mother": MODERATOR, "mod": MODERATOR}


class SendIn(BaseModel):
    """A message to the room. Outside a session it starts one: @mentions pick who answers
    (@all / no mention = every bot switched on), format/rounds/speakers/deep set how - and are
    remembered. During a session it joins the room, and every bot sees it on its next turn."""
    content: str
    format: str | None = None
    rounds: int | None = Field(None, ge=1, le=MAX_ROUNDS)
    asks: int | None = Field(None, ge=0, le=5)  # advice budget: ASKs per bot per turn
    speakers: dict[str, str] | None = None
    deep: bool | None = None
    research: bool | None = None  # web bots switch on their site's deep research mode
    files: list[str] = []  # ids from POST /api/media, attached to this message
    artifact: "ArtifactIn | None" = None  # "revise this": the version on screen goes with it


class ArtifactIn(BaseModel):
    title: str = ""
    kind: str = "html"     # html | svg | react
    code: str = Field(max_length=2_000_000)
    who: str = ""
    version: int = 0


SendIn.model_rebuild()


def meeting_defaults() -> dict:
    raw = store().get_meta("meeting_defaults")
    return {**DEFAULTS, **(json.loads(raw) if raw else {})}


def parse_mentions(text: str, members: list[str]) -> list[str]:
    """@names in the order written (deduplicated); unknown names are an error."""
    found: list[str] = []
    for raw in MENTION.findall(text):
        name = ALIASES.get(raw.lower(), raw.lower())
        if name not in members + ["all", MODERATOR]:
            raise HTTPException(400, f"Unknown @{raw} - in this room: "
                                     + ", ".join(f"@{n}" for n in members + ["all", MODERATOR]))
        if name not in found:
            found.append(name)
    return found


def parse_plan(text: str, members: list[str]) -> list[list[tuple[str, str]]] | None:
    """Two or more lines starting with @mentions make a plan: each line is a step, run in
    order; the bots named on one line work on it in parallel. "@gpt write it, then @claude
    review it" counts as two lines. Lines without a mention are the brief everyone reads."""
    steps = []
    for line in PLAN_SPLIT.sub("\n", text).splitlines():
        m = PLAN_LINE.match(line)
        if not m:
            continue
        task = m.group(2).strip(" :,-") or "your part of the plan above"
        steps.append([(n, task) for n in parse_mentions(m.group(1), members)])
    if len(steps) < 2:
        return None
    if any(n in ("all", MODERATOR) for step in steps for n, _ in step):
        raise HTTPException(400, "A plan step names bots: @all and @moderator can't take a task")
    return steps


def _roles(thread: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in thread]


def load_room(thread: list[dict]) -> list[Entry]:
    """The room as bots see it: everything said, minus failed turns."""
    room = []
    for m in thread:
        meta = m["meta"] or {}
        if not m["content"]:
            continue
        who = (meta.get("speaker") or "prism") if m["role"] != "user" or meta.get("kind") == "context" \
            else "user"
        room.append(Entry(who, meta.get("kind") or ("user" if who == "user" else "final"),
                          m["content"], m["id"],
                          files=[str(media_dir() / f) for f in meta.get("files") or []]))
    return room


def seen_index(room: list[Entry], seen_id: int | None) -> int:
    return 0 if seen_id is None else sum(1 for e in room if e.id is not None and e.id <= seen_id)


@app.post("/api/chats/{chat_id}/messages")
async def send(chat_id: int, body: SendIn) -> dict:
    chat = chat_or_404(chat_id)
    text = body.content.strip()
    for f in body.files:
        media_path(f)  # attached files must exist before anything starts
    if not text and body.files:
        text = "(see the attached files)"
    if not text:
        raise HTTPException(400, "Empty message")
    if cmd := PROFILE_CMD.match(text):
        notice = profile_command((cmd.group(1) or "update").lower())
        return {**chat_view(chat_id), "notice": notice}
    if cmd := RECALL.match(text):
        notice = start_recall(chat_id, *cmd.groups())  # first: the view shows its pending card
        return {**chat_view(chat_id), "notice": notice}
    if cmd := COMMAND.match(text):
        notice = context_command(chat_id, *cmd.groups())
        return {**chat_view(chat_id), "notice": notice}
    r = store().run_for_chat(chat_id)
    if r and r["status"] in ("running", "waiting", "paused"):
        # A run's room: what the user says goes to the steps still to come.
        store().add_message(chat_id, None, "user", text, {"kind": "user", "run": r["id"]})
        if run := _live_runs.get(r["id"]):
            run.add_note(text)
            store().save_run(r["id"], run.state.to_dict())
        else:
            store().save_run(r["id"], {**r["state"], "notes": r["state"].get("notes", []) + [text]})
        return chat_view(chat_id)
    running = _sessions.get(chat_id)
    if running and running.phase != "done":
        enabled = [c["platform"] for c in store().list_credentials() if c["enabled"]]
        if parse_mentions(text, enabled + [a["name"] for a in store().list_agents()]):
            # Addressed to someone: its own turn, once the running one has finished.
            _queued.setdefault(chat_id, []).append(body.model_copy(update={"content": text}))
            return chat_view(chat_id)
        if not running.accepts_messages:
            raise HTTPException(409, "The final answer is being written - send again in a moment")
        opens_files = any(isinstance(m.provider, BrowserProvider) for m in running.members.values())
        running.add_user_message(text, await attach_context(chat_id, text, opens_files=opens_files)
                                 + ([attach_artifact(chat_id, body.artifact)] if body.artifact else []),
                                 [str(media_path(f)) for f in body.files])
        return chat_view(chat_id)
    await start_session(chat, text, body)
    return chat_view(chat_id)


# --- named context: #name uses it, /save #name writes it -----------------------------------

def context_command(chat_id: int, verb: str, name: str, text: str) -> str:
    s, name = store(), context_name(name)
    if verb == "forget":
        item = s.find_context(chat_id, name)
        if not item:
            raise HTTPException(404, f"No #{name} here")
        s.delete_context(item["id"])
        return f"Forgot #{name}."
    content = text.strip() or latest_answer(chat_id)
    scope = None if verb == "save-global" else chat_id
    if scope is None and name == PROFILE:  # the user editing their profile by hand
        s.add_context(None, PROFILE_TITLE, content, name=PROFILE, always=True)
        return "Saved your profile (#profile) - every chat gets it."
    s.add_context(scope, name, content, name=name)
    return f"Saved #{name}{' (global)' if scope is None else ''} - use it with #{name}."


RECALL_PROMPT = (
    "Using your memory of me - saved memories, past conversations and projects - recall "
    "everything relevant to: {direction}\n\n"
    "Write it as complete, specific notes I can hand to another assistant: facts, decisions, "
    "preferences, numbers, names, dates, and open questions. Keep my own wording where it "
    "matters. Don't invent anything: if you remember nothing relevant, say so plainly. Plain "
    "text in this chat; don't create documents.")
ASK_PROMPT = (
    "Another AI assistant working for me asks you this: {question}\n\nIf it is about me, answer "
    "from your memory of me - saved memories and our past conversations - with the specific "
    "facts you know (dates, numbers, decisions, my own words where they matter). If it asks for "
    "advice, give your best advice. If you don't know or have nothing about it, say so plainly "
    "in one line; don't guess. Plain text, no documents. If it asks you to make a file (a page, "
    "a component, an SVG), put the complete file in one fenced code block in this chat (```html, "
    "```svg, ```jsx) - not in an artifact, canvas or side panel, which can't be passed on.")
_recalls: set[asyncio.Task] = set()


def start_recall(chat_id: int, platform: str, name: str, direction: str) -> str:
    """/recall @gpt #name <what to recall>: a new chat on that platform (so its own memory of
    the user applies) writes notes, saved as a global #name (it is about the user, so every
    chat can use it). Runs in the background."""
    s, name, platform = store(), context_name(name), ALIASES.get(platform.lower(), platform.lower())
    creds = {c["platform"]: c for c in s.enabled_credentials(vault())}
    if platform not in creds:
        raise HTTPException(400, f"@{platform} isn't set up - in this room: "
                                 + ", ".join(f"@{p}" for p in creds))
    direction = direction.strip() or name.replace("-", " ")
    provider = make_provider(creds[platform], fresh=True)
    note = s.add_message(chat_id, None, "assistant", "", {
        "kind": "recall", "speaker": platform, "model": provider.model, "name": name,
        "pending": True, "direction": direction})

    async def run() -> None:
        start = asyncio.get_running_loop().time()
        meta = {"kind": "recall", "speaker": platform, "model": provider.model, "name": name,
                "direction": direction}
        try:
            text = await wait_reply(provider.complete(
                [{"role": "user", "content": RECALL_PROMPT.format(direction=direction)}]),
                provider, settings().request_timeout_s, settings().reply_max_s)
            s.add_context(None, f"{name} (recalled from {platform})", text, name=name)
            meta["chars"] = len(text)
        except Exception as e:
            meta["error"] = f"{type(e).__name__}: {e}"
        meta["latency_s"] = round(asyncio.get_running_loop().time() - start, 1)
        s.update_message(note, meta)

    task = asyncio.create_task(run())
    _recalls.add(task)
    task.add_done_callback(_recalls.discard)
    return f"Recalling #{name} from {platform}: it appears under Context (global) when done."


# --- the user's profile: what PRISM's own chats say about the user -------------------------

PROFILE, PROFILE_TITLE = "profile", "About you (kept by PRISM from your chats)"
PROFILE_EVERY = 3        # new things the user said before an automatic update
PROFILE_MAX_CHARS = 2500
PROFILE_FAILURES = 3     # then automatic updates pause until /profile (e.g. signed out)
PROFILE_PROMPT = (
    "PRISM profile update. You keep a short profile of me for the AI assistants I use, so they know who I am. Below "
    "is my current profile, then things I wrote recently in my chats.\n\n"
    "CURRENT PROFILE:\n{profile}\n\nWHAT I WROTE RECENTLY (chat title: message):\n{said}\n\n"
    "Write my full updated profile: only lasting facts about me that I stated or clearly "
    "implied - who I am, my work and projects, goals, preferences (including how I like answers), "
    "constraints, and decisions I made. Leave out one-off requests and anything about the AI "
    "assistants themselves. Keep everything from the current profile that is still true. Short "
    "bullet points under a few headings, at most 300 words. Plain text in this chat, no "
    "documents. If nothing lasting is new, reply with exactly: UNCHANGED")
_profile_task: dict[str, asyncio.Task | None] = {"task": None}


def profile_mode() -> str:
    return store().get_meta("profile_mode") or "auto"  # auto | off


def profile_state() -> dict:
    task = _profile_task["task"]
    return {"mode": profile_mode(), "updating": bool(task and not task.done()),
            "last": store().get_meta("profile_last") or ""}


def profile_command(arg: str) -> str:
    if arg in ("on", "off"):
        store().set_meta("profile_mode", "auto" if arg == "on" else "off")
        return ("Profile on: PRISM keeps #profile (about you) up to date from your chats, and "
                "every chat gets it." if arg == "on" else
                "Profile off: #profile is no longer updated (it stays until you /forget #profile).")
    if not maybe_update_profile(force=True):
        return "Nothing new to learn about you yet, or no primary LLM is set up."
    return "Updating your profile from your chats - it appears as #profile under Context."


def maybe_update_profile(force: bool = False) -> bool:
    """After a meeting (or on /profile): if the user said enough new things since the last
    update, the primary LLM rewrites #profile from them, in the background. The profile is
    a global, always-sent context item; it is only rewritten when it actually changed,
    because a change makes every bot re-read the context on its next turn."""
    s = store()
    if (profile_mode() == "off" and not force) or _vault["key"] is None:
        return False
    failures = int(s.get_meta("profile_failures") or 0)
    if failures >= PROFILE_FAILURES and not force:
        return False
    task = _profile_task["task"]
    if task and not task.done():
        return False
    seen = int(s.get_meta("profile_seen_id") or 0)
    said = s.user_messages_after(seen)
    if not said or (len(said) < PROFILE_EVERY and not force):
        return False
    primary = s.get_meta("primary_platform")
    creds = {c["platform"]: c for c in s.enabled_credentials(vault())}
    if primary not in creds:
        return False
    current = next((c for c in s.context_for(0) if c["name"] == PROFILE and c["chat_id"] is None), None)
    lines, used = [], 0
    for m in reversed(said):  # the newest first, up to about 12k characters
        line = f"{m['title'][:50]}: {' '.join(m['content'].split())[:1500]}"
        if used + len(line) > 12_000:
            break
        lines.append(line)
        used += len(line)
    prompt = PROFILE_PROMPT.format(profile=current["content"] if current else "(none yet)",
                                   said="\n".join(reversed(lines)))
    provider = make_provider(creds[primary], fresh=True)

    async def run() -> None:
        try:
            text = (await wait_reply(provider.complete([{"role": "user", "content": prompt}]), provider,
                                     settings().request_timeout_s, settings().reply_max_s)).strip()
        except Exception as e:
            log.warning("profile update failed: %s", e)
            n = 1 if force else failures + 1
            s.set_meta("profile_failures", str(n))
            s.set_meta("profile_last", f"failed: {type(e).__name__}" + (
                " - paused until you press Update now" if n >= PROFILE_FAILURES else ""))
            return  # not marked as read: the next update tries these messages again
        s.set_meta("profile_failures", "0")
        new = text if len(text) >= 40 and "UNCHANGED" not in text[:40] else None
        if new and (not current or new != current["content"]):
            s.add_context(None, PROFILE_TITLE, new[:PROFILE_MAX_CHARS], name=PROFILE, always=True)
            s.set_meta("profile_last", f"updated from {len(said)} messages")
        else:
            s.set_meta("profile_last", "checked - nothing new")
        s.set_meta("profile_seen_id", str(said[-1]["id"]))

    _profile_task["task"] = asyncio.create_task(run())
    return True


class ProfileIn(BaseModel):
    mode: str


@app.put("/api/profile")
def put_profile(body: ProfileIn) -> dict:
    if body.mode not in ("auto", "off"):
        raise HTTPException(400, "mode is auto or off")
    store().set_meta("profile_mode", body.mode)
    return profile_state()


@app.post("/api/profile/update")
async def update_profile() -> dict:
    vault()
    return {**profile_state(), "started": maybe_update_profile(force=True)}


# --- ASK @prism: the user's past chats in PRISM itself, searched locally --------------------

PRISM_BOT = "prism"
STOPWORDS = set("""a an and are as at be but by can did do does for from has have how i in is it
its me my of on or so that the their them they this to user user's was what when where which who
why will with you your about any all know tell past chat chats""".split())


def search_history(question: str, chat_id: int | None) -> str:
    words = [w for w in dict.fromkeys(re.findall(r"[\w'-]{3,}", question.lower())) if w not in STOPWORDS]
    hits = store().search_messages(words[:8], exclude_chat=chat_id)
    if not hits:
        return "Nothing in the user's past PRISM chats matches that."
    out = []
    for h in hits:
        who = "the user" if h["role"] == "user" else h["meta"].get("speaker") or "a bot"
        text, at = " ".join(h["content"].split()), 0
        for w in words:
            if (i := text.lower().find(w)) >= 0:
                at = max(0, i - 150)
                break
        snippet = ("…" if at else "") + text[at:at + 700] + ("…" if len(text) > at + 700 else "")
        day = time.strftime("%Y-%m-%d", time.localtime(h["created_at"]))
        out.append(f"- {day}, chat \"{h['title'][:50]}\", {who}: {snippet}")
    return "From the user's past PRISM chats:\n" + "\n".join(out)


def latest_answer(chat_id: int) -> str:
    """What /save #name keeps when given no text: the final answer, else the last thing a bot
    or the moderator said."""
    said = [m for m in store().thread(chat_id) if m["role"] == "assistant" and m["content"]]
    finals = [m for m in said if (m["meta"] or {}).get("kind") in ("final", "summary")]
    if not (finals or said):
        raise HTTPException(400, "Nothing to save yet - add the text: /save #name <text>")
    return (finals or said)[-1]["content"]


FENCE = {"html": "html", "svg": "svg", "react": "jsx", "doc": "markdown"}


def attach_artifact(chat_id: int, art: "ArtifactIn") -> Entry:
    """The version the user is looking at, for the bot asked to change it: the room may hold
    it far back (or not at all in that bot's own chat)."""
    lang = FENCE.get(art.kind, "html")
    text = (f"The user is looking at v{art.version} of \"{art.title}\"" + (f" (by {art.who})" if art.who else "")
            + f" and asks for a change. Here it is:\n```{lang}\n{art.code}\n```\n"
            f"Reply with the complete revised file in one ```{lang} block (not a diff), then a "
            "line or two on what you changed.")
    entry = Entry("context", "context", text)
    entry.id = store().add_message(chat_id, None, "user", text,
                                   {"kind": "context", "name": "artifact", "speaker": "context",
                                    "title": art.title, "version": art.version})
    return entry


async def attach_context(chat_id: int, text: str, always: list[dict] = (),
                         opens_files: bool = False) -> list[Entry]:
    """#names in a message: each item joins the room (and is stored) just before it. A folder
    or web pages (#name, or marked always) gives the passages that match the message."""
    s, entries = store(), []
    names = list(dict.fromkeys(context_name(n) for n in CONTEXT_REF.findall(text)))
    items = {n: s.find_context(chat_id, n) for n in names}
    for name in names:  # check all before storing any
        if not items[name]:
            known = ", ".join(f"#{c['name']}" for c in s.context_for(chat_id)) or "none yet"
            raise HTTPException(400, f"No #{name} - known: {known}. Write it with /save #{name} <text>")
    for item in always:
        items.setdefault(item["name"], item)
    for name, item in items.items():
        if item.get("library"):
            entry, meta = await library_entry(chat_id, item, text, opens_files)
        else:
            entry = Entry("context", "context", f"#{name}: {item['title']}\n{item['content']}")
            meta = {"kind": "context", "name": name, "speaker": "context"}
        entry.id = s.add_message(chat_id, None, "user", entry.text, meta)
        entries.append(entry)
    return entries


DOCS_BOT = "docs"


def library_query(chat_id: int, text: str) -> str:
    """What to search for: the message without its #names; a short one ("go on", "more")
    borrows the words of the message before it."""
    query = CONTEXT_REF.sub(" ", text).strip()
    if len(re.findall(r"\w{3,}", query)) < 3:
        topics = [m["content"] for m in store().thread(chat_id)
                  if (m["meta"] or {}).get("kind") in ("topic", "user") and m["content"] != text]
        query = f"{topics[-1] if topics else ''} {query}".strip()
    return query


async def library_entry(chat_id: int, item: dict, text: str, opens_files: bool) -> tuple[Entry, dict]:
    lib, sid, name = library(), item["library"], item["name"]
    lib.scan_if_stale(sid)
    hits = await lib.retrieve([sid], library_query(chat_id, text))
    status = lib.status(sid)
    files: list[str] = []
    if opens_files and status["options"].get("attach", True):
        already = {p for m in store().thread(chat_id) for p in (m["meta"] or {}).get("attached_paths", [])}
        files = attachable(hits, already)
    where = (status["location"] if status["kind"] == "folder" else "the saved web pages")
    head = f"#{name}: passages from {where} that match the message (ask for more with ASK @docs: <keywords>)"
    if hits:
        body = passages_text(hits)
    elif status["busy"] or not status["scanned_at"]:
        body = "(PRISM is still reading these files - nothing matching yet)"
    else:
        body = "(nothing in it matches this message)"
    if files:
        body += "\n\nAttached for you to open: " + ", ".join(Path(f).name for f in files)
    entry = Entry("context", "context", f"{head}\n{body}", files=files)
    meta = {"kind": "context", "name": name, "speaker": "context", "library": sid,
            "sources": [{"path": h["path"], "page": h["page"]} for h in hits],
            **({"attached": [Path(f).name for f in files], "attached_paths": files} if files else {})}
    return entry, meta


async def search_docs(question: str, chat_id: int) -> str:
    """ASK @docs: every folder and web-page source this chat can use."""
    items = [c for c in store().context_for(chat_id) if c.get("library")]
    if not items:
        return "No folders or web pages are in this chat's context."
    hits = await library().retrieve([c["library"] for c in items], question, k=6)
    names = ", ".join(f"#{c['name']}" for c in items)
    if not hits:
        return f"Nothing in {names} matches that."
    return f"From the user's files ({names}):\n" + passages_text(hits, 6000)


@app.post("/api/chats/{chat_id}/meeting/stop")
async def stop_meeting(chat_id: int) -> dict:  # on the event loop: stop() is not thread-safe
    chat_or_404(chat_id)
    m = _sessions.get(chat_id)
    _queued.pop(chat_id, None)  # Stop means stop: queued messages are dropped too
    if m and m.phase != "done":
        m.stop()
    return chat_view(chat_id)


async def start_session(chat: dict, text: str, body: SendIn) -> None:
    s, f = store(), vault()
    chat_id = chat["id"]
    opts = meeting_defaults()
    for key in ("format", "rounds", "speakers", "deep", "asks", "research"):
        if getattr(body, key) is not None:
            opts[key] = getattr(body, key)
    if opts["format"] not in FORMATS:
        raise HTTPException(400, f"Unknown format {opts['format']!r} - one of {', '.join(FORMATS)}")
    s.set_meta("meeting_defaults", json.dumps(opts))

    creds = {c["platform"]: c for c in s.enabled_credentials(f)}
    agents = room_agents(creds)  # saved agents answer when @named, on their model
    primary = s.get_meta("primary_platform")
    mentions = parse_mentions(text, list(creds) + list(agents))
    named = [m for m in mentions if m not in ("all", MODERATOR)]
    plan = parse_plan(text, list(creds) + list(agents))
    if plan:
        fmt = "plan"
        speakers = list(dict.fromkeys(name for step in plan for name, _ in step))
    elif mentions == [MODERATOR]:
        fmt, speakers = "summary", []
    else:
        # Named bots answer, in the order named; @all or no mention adds every bot switched
        # on. Naming a bot overrides its "Off".
        on = [p for p in creds if opts["speakers"].get(p, "full") != "off"]
        speakers = named if named and "all" not in mentions else named + [p for p in on if p not in named]
        fmt = opts["format"]
        if not speakers:
            raise HTTPException(400, "No bot to answer - @mention one, or switch one on")
    script = make_script(fmt, speakers, opts["rounds"], plan)
    if script.moderated:
        if not primary:
            raise HTTPException(400, "Pick a primary (moderator) LLM in Settings first")
        if primary not in creds:
            raise HTTPException(400, f"Primary LLM {primary!r} has no enabled credentials")

    bound = {d["platform"]: d for d in s.daughters(chat_id)}
    always = [c for c in s.context_for(chat_id) if c["always"]]  # the rest: by #name
    context_items = [c for c in always if not c.get("library")]   # folders: searched per message

    def context_changed_since_last_turn(name: str) -> bool:
        """Remote chats only see the context when it is sent; resend after additions."""
        d = bound.get(name)
        if not d:
            return False
        last = max((m["created_at"] for m in s.thread(chat_id, d["id"])), default=0)
        return any(ci["created_at"] > last for ci in context_items)

    def tune(p: Provider) -> Provider:
        if opts["deep"] and hasattr(p, "effort"):
            p.effort = "high"  # API models with a thinking-effort setting think longer
        if opts.get("research") and isinstance(p, BrowserProvider):
            p.research = True  # the site's own deep research (can take 5-30 min)
        return p

    room = load_room(s.thread(chat_id))
    seats = {p: creds[p] if p in creds else creds[agent_platform(agents[p])] for p in speakers}
    if script.moderated:
        seats[MODERATOR] = creds[primary]  # the primary, in its own moderator conversation
    members: dict[str, Member] = {}
    daughter_ids: dict[str, int] = {}
    try:
        for name, cred in seats.items():
            d = bound.get(name, {})
            provider = tune(make_provider(cred, thread_url=d.get("remote_url"),
                                          resend_system=context_changed_since_last_turn(name)))
            daughter_ids[name] = s.bind_daughter(chat_id, name, provider.model)
            budget = opts["speakers"].get(name, "full")
            members[name] = Member(name, provider, BUDGETS.get(budget, BUDGETS["full"]),
                                   _roles(s.thread(chat_id, daughter_ids[name])),
                                   seen_index(room, d.get("seen_message_id")),
                                   role=agents[name].get("instructions", "") if name in agents else "")
    except Exception as e:
        raise HTTPException(400, f"Could not set up the meeting: {e}") from e

    def emit(kind: str, **data) -> int | None:
        if kind == "seen":
            if data["entry"].id is not None:
                s.set_seen(daughter_ids[data["member"]], data["entry"].id)
            return None
        if kind != "entry":
            return None
        entry, meta = data["entry"], data["meta"]
        st, prompt = data.get("statement"), data.get("prompt")
        name = entry.who
        if entry.text and name in members and meta["kind"] != "lookup":  # its own thread
            s.add_message(chat_id, daughter_ids[name], "user", st.prompt if st else prompt)
            s.add_message(chat_id, daughter_ids[name], "assistant", entry.text,
                          {k: meta[k] for k in ("html", "documents") if meta.get(k)} or None)
            url = getattr(members[name].provider, "thread_url", None)
            if url:
                s.set_remote_url(daughter_ids[name], url)
        return s.add_message(chat_id, None, "user" if name == "user" else "assistant",
                             entry.text, meta)

    opens_files = any(cred["transport"] == "browser" for cred in seats.values())
    room += await attach_context(chat_id, text, [c for c in always if c.get("library")], opens_files)
    if body.artifact:
        room.append(attach_artifact(chat_id, body.artifact))
    topic = Entry("user", "topic", text, files=[str(media_path(f)) for f in body.files])
    topic.id = s.add_message(chat_id, None, "user", text, {
        "kind": "topic", "format": script.format, "rounds": script.rounds,
        "speakers": speakers, "mentions": mentions, **({"files": body.files} if body.files else {})})
    room.append(topic)
    async def ask(platform: str, question: str) -> str:
        """A bot's ASK: a new chat on that platform, so its memory of the user applies;
        @prism searches the user's other PRISM chats."""
        if platform == PRISM_BOT:
            return search_history(question, chat_id)
        if platform == DOCS_BOT:
            return await search_docs(question, chat_id)
        return await make_provider(creds[platform], fresh=True).complete(
            [{"role": "user", "content": ASK_PROMPT.format(question=question)}])

    # Memory lives in the signed-in web apps; API models remember nothing between calls.
    askable = ([PRISM_BOT] + ([DOCS_BOT] if any(c.get("library") for c in s.context_for(chat_id)) else [])
               + [p for p, c in creds.items() if c["transport"] == "browser"])
    session = Session(room, members, script, emit, context=context_block(context_items),
                      timeout=settings().request_timeout_s, max_s=settings().reply_max_s, deep=opts["deep"],
                      ask=ask, askable=askable, max_asks=opts["asks"])
    _sessions[chat_id] = session
    if chat["title"] == "New chat":
        s.update_chat(chat_id, title=text[:60], primary_platform=primary)
    else:
        s.update_chat(chat_id, primary_platform=primary)

    async def run() -> None:
        cancelled = False
        try:
            await session.run()
        except asyncio.CancelledError:
            cancelled = True  # shutting down: don't start what is queued
            raise
        except Exception:
            log.exception("session in chat %s failed", chat_id)
            s.add_message(chat_id, None, "assistant", "", {
                "kind": "final", "speaker": MODERATOR, "round": session.round,
                "error": "the session stopped unexpectedly - see the PRISM window"})
        finally:
            _session_tasks.pop(chat_id, None)
            if not cancelled:
                try:
                    maybe_update_profile()
                except Exception:
                    log.exception("could not start the profile update")
                await start_next_queued(chat_id)

    _session_tasks[chat_id] = asyncio.create_task(run())


async def start_next_queued(chat_id: int) -> None:
    """Run the next @mentioned message that arrived while the room was busy."""
    while _queued.get(chat_id):
        nxt = _queued[chat_id].pop(0)
        chat = store().get_chat(chat_id)
        if chat is None:
            break
        try:
            await start_session(chat, nxt.content, nxt)
            return
        except HTTPException as e:  # e.g. vault locked meanwhile: say so in the room
            store().add_message(chat_id, None, "user", nxt.content, {"kind": "topic", "mentions": []})
            store().add_message(chat_id, None, "assistant", "", {
                "kind": "final", "speaker": MODERATOR, "error": f"Not sent: {e.detail}"})
    _queued.pop(chat_id, None)


# --- context store -----------------------------------------------------------------------

class ContextIn(BaseModel):
    title: str = ""
    content: str
    name: str | None = None     # #name; defaults to one made from the title
    always: bool = False        # True = sent with every message, not only when #named
    global_: bool = Field(False, alias="global")  # True = shared with every chat


@app.post("/api/chats/{chat_id}/context")
def add_context(chat_id: int, c: ContextIn) -> dict:
    chat_or_404(chat_id)
    title = c.title.strip() or (c.name or "").strip() or "Untitled"
    store().add_context(None if c.global_ else chat_id, title, c.content, name=c.name or title,
                        always=c.always)
    return chat_view(chat_id)


@app.delete("/api/context/{item_id}")
def delete_context(item_id: int) -> dict:
    store().delete_context(item_id)
    return {"ok": True}


# --- library: folders and web pages as searchable context -------------------------------

class LibraryIn(BaseModel):
    name: str
    location: str               # a folder, or URLs one per line
    always: bool = False
    global_: bool = Field(False, alias="global")
    vision: bool = False        # read images and scanned pages with a vision model
    transcribe: bool = False    # transcribe audio and video
    attach: bool = True         # upload the best PDFs/images to bots that open files


class LibraryOptionsIn(BaseModel):
    vision: bool | None = None
    transcribe: bool | None = None
    attach: bool | None = None


@app.post("/api/chats/{chat_id}/library")
async def add_library(chat_id: int, body: LibraryIn) -> dict:
    chat_or_404(chat_id)
    name = context_name(body.name.lstrip("#"))
    if not name or not body.location.strip():
        raise HTTPException(400, "Give it a #name and a folder (or web addresses)")
    try:
        sid = library().add(None if body.global_ else chat_id, name, body.location, body.always,
                            {"vision": body.vision, "transcribe": body.transcribe, "attach": body.attach})
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    library().scan(sid)
    return chat_view(chat_id)


@app.get("/api/library/{source_id}")
def library_status(source_id: int) -> dict:
    if not store().get_source(source_id):
        raise HTTPException(404, "No such folder")
    return library().status(source_id)


@app.put("/api/library/{source_id}")
async def library_options(source_id: int, body: LibraryOptionsIn) -> dict:
    src = store().get_source(source_id)
    if not src:
        raise HTTPException(404, "No such folder")
    options = {**src["options"], **body.model_dump(exclude_none=True)}
    store().update_source(source_id, options=options)
    library().scan(source_id)  # a reader switched on reads what it can now
    return library().status(source_id)


@app.post("/api/library/{source_id}/scan")
async def library_scan(source_id: int, full: bool = False) -> dict:
    if not store().get_source(source_id):
        raise HTTPException(404, "No such folder")
    return {"started": library().scan(source_id, full=full), **library().status(source_id)}


class ReadersIn(BaseModel):
    vision: str | None = None       # "auto" | "off" | a reader id
    transcribe: str | None = None
    embed: str | None = None        # "" = words only | "ollama:<model>" | "lmstudio:<model>"


def _all_sources() -> list[int]:
    return [r["id"] for r in store()._q("SELECT id FROM library_sources")]


@app.get("/api/readers")
async def readers_status() -> dict:
    try:
        creds = {c["platform"]: c for c in store().enabled_credentials(vault())}
    except HTTPException:
        creds = {}
    found = await enrich.detect(data_dir(), creds)
    return {**found, "config": enrich.load_config(store().get_meta("readers")),
            "downloads": enrich.downloads(),
            "can_download": {k: sys.platform == "win32" or not v.get("windows_only")
                             for k, v in enrich.DOWNLOADS.items()},
            "hints": {} if sys.platform == "win32" else enrich.MAC_HINTS,
            "suggested": {"embed": enrich.SUGGESTED_EMBED, "vision": enrich.SUGGESTED_VISION}}


@app.put("/api/readers")
async def put_readers(body: ReadersIn) -> dict:
    cfg = enrich.load_config(store().get_meta("readers"))
    new = {**cfg, **body.model_dump(exclude_none=True)}
    store().set_meta("readers", json.dumps(new))
    if new["embed"] != cfg["embed"]:  # vectors from another model don't compare
        store().clear_vecs()
        for sid in _all_sources():
            store().bump_source(sid)
            if new["embed"]:
                library().scan(sid)
    return new


class DownloadIn(BaseModel):
    what: str


@app.post("/api/readers/download")
async def download_tool(body: DownloadIn) -> dict:
    if body.what not in enrich.DOWNLOADS:
        raise HTTPException(400, f"Unknown download {body.what!r}")
    enrich.start_download(body.what, data_dir())
    return {"downloads": enrich.downloads()}


# --- agents and flows ------------------------------------------------------------------------

AGENT_NAME = re.compile(r"^[a-z][a-z0-9-]{1,23}$")


def agent_platform(agent: dict) -> str:
    return str(agent.get("model") or "").strip().lstrip("@").lower()


def room_agents(creds: dict) -> dict[str, dict]:
    """Saved agents whose model is set up and switched on."""
    return {a["name"]: a for a in store().list_agents() if agent_platform(a) in creds}


class AgentIn(BaseModel):
    model: str = Field(min_length=1, max_length=40)
    instructions: str = Field("", max_length=20_000)
    context: list[str] = []      # #names it always gets
    tools: list[str] = []        # ask | docs | prism
    output: dict[str, str] = {}  # field: A|B, or text
    research: bool = False       # web bots: the site's deep research mode for each step


@app.get("/api/agents")
def list_agents() -> dict:
    return {"agents": store().list_agents()}


@app.put("/api/agents/{name}")
def put_agent(name: str, body: AgentIn) -> dict:
    name = name.lower()
    platforms = {c["platform"] for c in store().list_credentials()}
    if not AGENT_NAME.match(name) or name in RESERVED or known_platform(name) or name in platforms:
        raise HTTPException(400, f"@{name} can't be an agent's name: use 2-24 lowercase letters, digits or -, "
                                 "not a platform's or a room member's name")
    if agent_platform({"model": body.model}) not in platforms:
        raise HTTPException(400, f"No room member {body.model} - set it up in Settings first")
    bad = [t for t in body.tools if t not in ("ask", "docs", "prism")]
    if bad:
        raise HTTPException(400, f"Unknown tools {bad} - use ask, docs, prism")
    spec = body.model_dump()
    spec["model"] = "@" + agent_platform(spec)
    spec["context"] = [context_name(c.lstrip("#")) for c in body.context if c.strip("# ")]
    store().put_agent(name, spec)
    return list_agents()


@app.delete("/api/agents/{name}")
def delete_agent(name: str) -> dict:
    store().delete_agent(name)
    return list_agents()


def parse_flow(source: str) -> flows.Flow:
    return flows.parse(source, agents=store().get_agent, flows=store().get_flow)


class FlowIn(BaseModel):
    source: str = Field(max_length=200_000)


@app.post("/api/flows/validate")
def validate_flow(body: FlowIn) -> dict:
    try:
        return {"ok": True, "errors": [], "graph": parse_flow(body.source).to_json()}
    except flows.FlowError as e:
        return {"ok": False, "errors": e.errors, "graph": None}


@app.get("/api/flows")
def list_flows() -> dict:
    out = []
    for f in store().list_flows():
        v = validate_flow(FlowIn(source=f["source"]))
        out.append({**f, "ok": v["ok"], "errors": v["errors"]})
    return {"flows": out, "agents": store().list_agents()}


@app.post("/api/flows")
def save_flow(body: FlowIn) -> dict:
    try:
        flow = parse_flow(body.source)
    except flows.FlowError as e:
        raise HTTPException(400, "Not saved: " + "; ".join(e.errors)) from e
    store().put_flow(flow.name, body.source)
    return {**list_flows(), "saved": flow.name}


@app.delete("/api/flows/{name}")
def delete_flow(name: str) -> dict:
    store().delete_flow(name)
    return list_flows()


DRAFT_PROMPT = (
    "Write a workflow for PRISM, an app that runs AI agents step by step. The user wants:\n\n"
    "{description}\n\nThe format, with an example:\n```yaml\n{example}\n```\n"
    "Rules: node types are agent (a model answers its prompt), tool (docs = search the user's "
    "files, prism = search past chats, fetch = read a web page; give query: or url:), ask_you "
    "(pause for the user's approval), wait (e.g. 10m), flow (another saved flow). `join: all` on "
    "a node waits for every node leading into it. Prompts read earlier results as "
    "{{{{node.text}}}} or {{{{node.field}}}} and inputs as {{{{input.name}}}}. Every loop needs "
    "max_rounds on one of its edges. Agents you may use as they are: {agents}. New agents go "
    "under agents: with model one of: {models}. Reply with only the YAML in one ```yaml block.")
DRAFT_EXAMPLE = flows.__doc__.split("    name: market-brief", 1)[1].split("\n\nNode types", 1)[0]


class DraftIn(BaseModel):
    description: str = Field(min_length=3, max_length=5000)


@app.post("/api/flows/draft")
async def draft_flow(body: DraftIn) -> dict:
    """The primary model writes the YAML; PRISM checks it and asks once more to fix errors."""
    creds = {c["platform"]: c for c in store().enabled_credentials(vault())}
    primary = store().get_meta("primary_platform")
    if primary not in creds:
        raise HTTPException(400, "Pick a primary LLM in Settings first - it drafts the flow")
    provider = make_provider(creds[primary], fresh=True)
    example = "name: market-brief" + "\n".join(line[4:] for line in DRAFT_EXAMPLE.splitlines())
    agents = ", ".join(f"{a['name']} ({a['model']})" for a in store().list_agents()) or "none yet"
    msgs = [{"role": "user", "content": DRAFT_PROMPT.format(
        description=body.description, example=example, agents=agents,
        models=", ".join("@" + p for p in creds))}]
    source, errors = "", []
    for _ in range(2):
        try:
            reply = await wait_reply(provider.complete(list(msgs)), provider,
                                     settings().request_timeout_s, settings().reply_max_s)
        except Exception as e:
            raise HTTPException(502, f"{primary} could not draft it: {e}") from e
        m = re.search(r"```(?:ya?ml)?\s*\n(.*?)```", reply, re.S)
        source = (m.group(1) if m else reply).strip() + "\n"
        errors = validate_flow(FlowIn(source=source))["errors"]
        if not errors:
            break
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": "PRISM found these problems - reply with the whole corrected "
                                             "YAML:\n- " + "\n- ".join(errors)}]
    return {"source": source, "errors": errors}


# --- runs: a flow running in its own chat room -----------------------------------------------

_live_runs: dict[int, flows.Run] = {}
_run_tasks: dict[int, asyncio.Task] = {}
FETCH_MAX = 20_000


class AppRunEnv(flows.RunEnv):
    def __init__(self, run_id: int, chat_id: int):
        self.run_id, self.chat_id = run_id, chat_id
        self.timeout, self.max_s = settings().request_timeout_s, settings().reply_max_s

    def _creds(self) -> dict:
        return {c["platform"]: c for c in store().enabled_credentials(vault())}

    def agent(self, name: str) -> dict | None:
        return store().get_agent(name)

    def provider(self, agent: dict, memory: dict) -> Provider:
        platform = agent_platform(agent)
        cred = self._creds().get(platform)
        if not cred:
            raise flows.StepError(f"@{platform} (for {agent['name']}) is not set up or is switched off")
        url = memory.get("thread_url")
        p = make_provider(cred, thread_url=url, fresh=not url)
        if agent.get("research") and isinstance(p, BrowserProvider):
            p.research = True
        return p

    async def context(self, agent: dict, prompt: str) -> str:
        s, parts = store(), []
        items = [c for c in s.context_for(self.chat_id) if c["always"] and c["chat_id"] is None and not c.get("library")]
        for name in agent.get("context") or []:
            item = s.find_context(self.chat_id, name)
            if not item:
                continue
            if item.get("library"):
                hits = await library().retrieve([item["library"]], prompt)
                if hits:
                    parts.append(f"#{name} - passages that match this step:\n" + passages_text(hits))
            elif item not in items:
                items.append(item)
        if items:
            parts.insert(0, context_block(items))
        return "\n\n".join(parts)

    def askable(self, agent: dict) -> list[str]:
        tools = agent.get("tools") or []
        out = [t for t in ("docs", "prism") if t in tools or "ask" in tools]
        if "ask" in tools:
            out += [p for p, c in self._creds().items() if c["transport"] == "browser"]
        return out

    async def ask(self, target: str, question: str) -> str:
        if target == PRISM_BOT:
            return search_history(question, self.chat_id)
        if target == DOCS_BOT:
            return await search_docs(question, self.chat_id)
        return await make_provider(self._creds()[target], fresh=True).complete(
            [{"role": "user", "content": ASK_PROMPT.format(question=question)}])

    async def tool(self, name: str, query: str) -> str:
        if name == "docs":
            return await search_docs(query, self.chat_id)
        if name == "prism":
            return search_history(query, self.chat_id)
        url = query.strip()
        if not re.match(r"https?://", url):
            raise flows.StepError(f"fetch needs a web address, got {url[:80]!r}")
        async with httpx.AsyncClient(timeout=30, follow_redirects=True,
                                     headers={"User-Agent": "Mozilla/5.0 PRISM"}) as http:
            r = await http.get(url)
        if r.status_code != 200:
            raise flows.StepError(f"{url} answered {r.status_code}")
        text = readers.html_text(r.text) if "html" in r.headers.get("content-type", "html") else r.text
        return text[:FETCH_MAX]

    def emit(self, kind: str, **data) -> int | None:
        s, base = store(), {"run": self.run_id}
        if kind == "run":
            if data["status"] in ("running", "waiting"):
                return None  # the room shows the run's state (a question shows as its own card)
            return s.add_message(self.chat_id, None, "assistant", data.get("error") or "", {
                **base, "kind": "run", "speaker": "prism", "status": data["status"], "error": data.get("error")})
        if kind == "ask_you":
            return s.add_message(self.chat_id, None, "assistant", data["show"], {
                **base, "kind": "ask_you", "speaker": "you", "node": data["node"], "question": data["question"]})
        if kind == "lookup":
            return s.add_message(self.chat_id, None, "assistant",
                                 f"{data['node']} asked: {data['question']}\n\n{data['text']}", {
                **base, "kind": "lookup", "speaker": data["target"], "asker": data["node"], "question": data["question"]})
        provider = data.get("provider")
        meta = {**base, "kind": "step", "node": data["node"], "type": data["type"], "speaker": data["label"],
                "round": data["round"], "latency_s": data.get("latency_s"), "error": data.get("error"),
                "fields": data.get("fields") or {}}
        if provider is not None:
            meta["model"] = provider.model
            if notes := getattr(provider, "notes", None):
                meta["notes"] = list(notes)
            if html := getattr(provider, "last_html", None):
                meta["html"] = html
            if docs := getattr(provider, "last_documents", None):
                meta["documents"] = docs
            # Each agent has its own tab, like a bot in a meeting.
            d = s.bind_daughter(self.chat_id, data["label"], provider.model)
            s.add_message(self.chat_id, d, "user", data["prompt"])
            s.add_message(self.chat_id, d, "assistant", data["text"],
                          {k: meta[k] for k in ("html", "documents") if meta.get(k)} or None)
            if url := getattr(provider, "thread_url", None):
                s.set_remote_url(d, url)
        return s.add_message(self.chat_id, None, "assistant", data["text"], meta)

    def save(self, state: flows.RunState) -> None:
        store().save_run(self.run_id, state.to_dict())


def recover_runs() -> None:
    """Runs that were going when PRISM closed wait to be continued."""
    for r in store().runs_with_status("running"):
        if r["id"] not in _run_tasks:
            store().save_run(r["id"], {**r["state"], "status": "paused",
                                       "error": "interrupted when PRISM closed - continue to pick up where it stopped"})


def load_run(run_id: int) -> flows.Run:
    if run := _live_runs.get(run_id):
        return run
    r = store().get_run(run_id)
    if not r:
        raise HTTPException(404, "No such run")
    try:
        flow = parse_flow(r["source"])
    except flows.FlowError as e:
        raise HTTPException(400, "The flow can't be read any more: " + "; ".join(e.errors)) from e
    run = flows.Run(flow, flows.RunState.from_dict(r["state"]), AppRunEnv(run_id, r["chat_id"]))
    _live_runs[run_id] = run
    return run


def launch(run_id: int, run: flows.Run) -> None:
    if (t := _run_tasks.get(run_id)) and not t.done():
        raise HTTPException(409, "It is already running")

    async def go() -> None:
        try:
            await run.run()
        except asyncio.CancelledError:
            pass  # stopped, or PRISM is closing: the state is saved
        except Exception as e:
            log.exception("run %s failed", run_id)
            run.state.status, run.state.error = "paused", f"PRISM error: {e}"
            run.env.save(run.state)
        finally:
            _run_tasks.pop(run_id, None)

    _run_tasks[run_id] = asyncio.create_task(go())


def run_view(chat_id: int) -> dict | None:
    r = store().run_for_chat(chat_id)
    if not r:
        return None
    st = r["state"]
    try:
        graph = parse_flow(r["source"]).to_json()
    except flows.FlowError:
        graph = None
    return {"id": r["id"], "flow": r["flow"], "status": r["status"], "error": st.get("error"),
            "waiting": st.get("waiting"), "steps": st.get("steps", 0), "budget": st.get("budget"),
            "queue": st.get("queue", []), "done": list(st.get("outputs", {})), "graph": graph,
            "live": r["id"] in _run_tasks}


class RunIn(BaseModel):
    inputs: dict[str, str] = {}


@app.post("/api/flows/{name}/runs")
async def start_run(name: str, body: RunIn) -> dict:
    vault()
    source = store().get_flow(name)
    if source is None:
        raise HTTPException(404, f"No flow named {name}")
    try:
        flow = parse_flow(source)
    except flows.FlowError as e:
        raise HTTPException(400, "Fix the flow first: " + "; ".join(e.errors)) from e
    missing = [k for k in flow.inputs if not body.inputs.get(k, "").strip()]
    if missing:
        raise HTTPException(400, "Fill in: " + ", ".join(missing))
    s = store()
    chat_id = s.create_chat(f"Run: {name}")
    run_id = s.add_run(name, source, chat_id, flows.RunState({}).to_dict())
    run = flows.Run.start(flow, body.inputs, AppRunEnv(run_id, chat_id))
    run.env.save(run.state)
    shown = "\n".join(f"{k}: {v}" for k, v in run.state.inputs.items())
    s.add_message(chat_id, None, "user", f"Run {name}" + (f"\n{shown}" if shown else ""),
                  {"kind": "topic", "run": run_id, "format": "flow", "mentions": [], "speakers": []})
    _live_runs[run_id] = run
    launch(run_id, run)
    return chat_view(chat_id)


@app.post("/api/runs/{run_id}/stop")
async def stop_run(run_id: int) -> dict:
    r = store().get_run(run_id)
    if not r:
        raise HTTPException(404, "No such run")
    if run := _live_runs.get(run_id):
        run.stop()
    if task := _run_tasks.get(run_id):
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    r = store().get_run(run_id)
    if r["status"] not in ("done", "stopped"):
        store().save_run(run_id, {**r["state"], "status": "stopped"})
        store().add_message(r["chat_id"], None, "assistant", "", {"kind": "run", "run": run_id,
                                                                   "speaker": "prism", "status": "stopped"})
    _live_runs.pop(run_id, None)
    return chat_view(r["chat_id"])


class ResumeIn(BaseModel):
    more_budget: bool = False


@app.post("/api/runs/{run_id}/resume")
async def resume_run(run_id: int, body: ResumeIn) -> dict:
    vault()
    run = load_run(run_id)
    if run.state.status in ("done", "stopped"):
        raise HTTPException(400, f"The run is {run.state.status}")
    if run.state.waiting:
        raise HTTPException(400, "It is waiting for your answer above")
    run.resume(more_budget=body.more_budget or (run.state.error or "").startswith("budget"))
    launch(run_id, run)
    return chat_view(run.env.chat_id)


class AnswerIn(BaseModel):
    node: str
    approved: bool
    text: str = ""


@app.post("/api/runs/{run_id}/answer")
async def answer_run(run_id: int, body: AnswerIn) -> dict:
    vault()
    run = load_run(run_id)
    if not run.answer(body.node, body.approved, body.text.strip()):
        raise HTTPException(409, "That question was already answered")
    s, chat_id = store(), run.env.chat_id
    for m in reversed(s.thread(chat_id)):
        meta = m["meta"] or {}
        if meta.get("kind") == "ask_you" and meta.get("node") == body.node and "answer" not in meta:
            s.update_message(m["id"], {**meta, "answer": "yes" if body.approved else "no", "answer_text": body.text.strip()})
            break
    launch(run_id, run)
    return chat_view(chat_id)
