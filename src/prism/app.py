"""FastAPI web app: vault unlock, credentials, mother/daughter chats, context store."""

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from cryptography.fernet import Fernet
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
from .providers import PLATFORMS, Provider, build_provider

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


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    for task in _session_tasks.values():
        task.cancel()
    await asyncio.gather(*_session_tasks.values(), return_exceptions=True)
    await browsers().close()


app = FastAPI(title="PRISM", version=__version__, lifespan=lifespan)


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


@app.get("/api/platforms")
def platforms() -> dict:
    catalog = {p: {**spec, "web": p in SITES} for p, spec in PLATFORMS.items()}
    return {"catalog": catalog, "credentials": store().list_credentials(),
            "primary": store().get_meta("primary_platform")}


@app.put("/api/credentials/{platform}")
def put_credential(platform: str, c: CredentialIn) -> dict:
    if platform not in PLATFORMS:
        raise HTTPException(404, f"Unknown platform {platform!r}")
    if c.transport not in ("api", "browser"):
        raise HTTPException(400, "transport must be 'api' or 'browser'")
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
    if p.platform not in PLATFORMS:
        raise HTTPException(404, f"Unknown platform {p.platform!r}")
    store().set_meta("primary_platform", p.platform)
    return platforms()


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
        "context": s.context_for(chat_id),
        "meeting": meeting,
        "queued": [q.content for q in _queued.get(chat_id, [])],
        "meeting_defaults": meeting_defaults(),
        "profile": profile_state(),
    }


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: int) -> dict:
    return chat_view(chat_id)


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: int) -> dict:
    m = _sessions.get(chat_id)
    if m and m.phase != "done":
        raise HTTPException(409, "Stop the meeting before deleting the chat")
    store().delete_chat(chat_id)
    return {"ok": True}


# --- the room: the user, the bots and the moderator in one group chat --------------------

_sessions: dict[int, Session] = {}
_session_tasks: dict[int, asyncio.Task] = {}
_queued: dict[int, list["SendIn"]] = {}  # @mentioned messages waiting for the running session

# speakers: platform -> budget | "off"; format: how @all (or no mention) is run.
DEFAULTS = {"rounds": 3, "speakers": {}, "deep": False, "format": "roundtable", "asks": 3}
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
    files: list[str] = []  # ids from POST /api/media, attached to this message


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
    running = _sessions.get(chat_id)
    if running and running.phase != "done":
        if parse_mentions(text, [c["platform"] for c in store().list_credentials() if c["enabled"]]):
            # Addressed to someone: its own turn, once the running one has finished.
            _queued.setdefault(chat_id, []).append(body.model_copy(update={"content": text}))
            return chat_view(chat_id)
        if not running.accepts_messages:
            raise HTTPException(409, "The final answer is being written - send again in a moment")
        running.add_user_message(text, attach_context(chat_id, text),
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
    "in one line; don't guess. Plain text, no documents.")
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
            text = await asyncio.wait_for(provider.complete(
                [{"role": "user", "content": RECALL_PROMPT.format(direction=direction)}]),
                settings().request_timeout_s)
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
            text = (await asyncio.wait_for(provider.complete([{"role": "user", "content": prompt}]),
                                           settings().request_timeout_s)).strip()
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


def attach_context(chat_id: int, text: str) -> list[Entry]:
    """#names in a message: each item joins the room (and is stored) just before it."""
    s, entries = store(), []
    names = list(dict.fromkeys(context_name(n) for n in CONTEXT_REF.findall(text)))
    items = {n: s.find_context(chat_id, n) for n in names}
    for name in names:  # check all before storing any
        if not items[name]:
            known = ", ".join(f"#{c['name']}" for c in s.context_for(chat_id)) or "none yet"
            raise HTTPException(400, f"No #{name} - known: {known}. Write it with /save #{name} <text>")
    for name, item in items.items():
        entry = Entry("context", "context", f"#{name}: {item['title']}\n{item['content']}")
        entry.id = s.add_message(chat_id, None, "user", entry.text,
                                 {"kind": "context", "name": name, "speaker": "context"})
        entries.append(entry)
    return entries


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
    for key in ("format", "rounds", "speakers", "deep", "asks"):
        if getattr(body, key) is not None:
            opts[key] = getattr(body, key)
    if opts["format"] not in FORMATS:
        raise HTTPException(400, f"Unknown format {opts['format']!r} - one of {', '.join(FORMATS)}")
    s.set_meta("meeting_defaults", json.dumps(opts))

    creds = {c["platform"]: c for c in s.enabled_credentials(f)}
    primary = s.get_meta("primary_platform")
    mentions = parse_mentions(text, list(creds))
    named = [m for m in mentions if m not in ("all", MODERATOR)]
    plan = parse_plan(text, list(creds))
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
    context_items = [c for c in s.context_for(chat_id) if c["always"]]  # the rest: by #name

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
        return p

    room = load_room(s.thread(chat_id))
    seats = {p: creds[p] for p in speakers}
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
                                   seen_index(room, d.get("seen_message_id")))
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
            s.add_message(chat_id, daughter_ids[name], "assistant", entry.text)
            url = getattr(members[name].provider, "thread_url", None)
            if url:
                s.set_remote_url(daughter_ids[name], url)
        return s.add_message(chat_id, None, "user" if name == "user" else "assistant",
                             entry.text, meta)

    room += attach_context(chat_id, text)
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
        return await make_provider(creds[platform], fresh=True).complete(
            [{"role": "user", "content": ASK_PROMPT.format(question=question)}])

    # Memory lives in the signed-in web apps; API models remember nothing between calls.
    askable = [PRISM_BOT] + [p for p, c in creds.items() if c["transport"] == "browser"]
    session = Session(room, members, script, emit, context=context_block(context_items),
                      timeout=settings().request_timeout_s, deep=opts["deep"],
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
