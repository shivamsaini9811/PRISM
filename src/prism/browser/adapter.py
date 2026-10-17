"""Generic chat-site driver: type into the composer, send, wait for the reply to finish, read it."""

import asyncio
import base64
import re
import time
import uuid
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from playwright.async_api import Page

from ..providers.base import ProviderError


class NotLoggedIn(ProviderError):
    pass


class BotCheck(ProviderError):
    pass


class PageNotReady(ProviderError):
    pass


_LOGIN_URL = re.compile(r"login|signin|sign-in|sign_in|/auth|accounts\.google\.com|accounts\.x\.ai",
                        re.IGNORECASE)
_BOT_TITLE = re.compile(r"just a moment|verify you are human|attention required|are you a robot",
                        re.IGNORECASE)


@dataclass(frozen=True)
class SiteConfig:
    new_chat_url: str
    composer: str               # where the prompt is typed (textarea or contenteditable)
    assistant_message: str      # every assistant reply block; the last one is the answer
    send_button: str = ""       # empty -> press Enter in the composer
    streaming: str = ""         # visible while the reply is still being generated
    signed_out: str = ""        # visible only when signed out (sites that still let you chat)
    file_input: str = 'input[type="file"]'  # where attachments go (hidden inputs are fine)
    attach_button: str = ""     # opens the file picker when the site has no file input yet
    verified: bool = False

    def with_overrides(self, overrides: dict | None) -> "SiteConfig":
        allowed = {f.name for f in fields(self)} - {"verified"}
        return replace(self, **{k: v for k, v in (overrides or {}).items() if k in allowed and v})

    def to_dict(self) -> dict:
        return asdict(self)


# Icon-font glyphs (Unicode private use area) and the page chrome that follows a reply
# ("Copy", "just now", "2 minutes ago", ...) get captured along with the reply text.
_ICON_CHARS = re.compile("[\ue000-\uf8ff\U000f0000-\U000fffff\u25a1\ufffd]")
_CHROME_LINE = re.compile(
    r"^(just now|\d+\s*(s|sec|second|m|min|minute|h|hr|hour|d|day)s?\s*ago|copy|retry|edit|share|"
    r"good response|bad response|read aloud|regenerate|create a doc)$", re.IGNORECASE)
# claude.ai: a screen-reader label "Claude responded: <first line>", then the collapsed
# thinking summary, then the answer itself starting with that first line again.
_SR_LABEL = re.compile(r"^\w[\w .-]{0,20} responded:\s*", re.IGNORECASE)
# Status lines above a reply: grok.com's "Grok" heading and "Worked for 4s".
_STATUS_LINE = re.compile(r"^(?:Grok|ChatGPT|Claude|Gemini|"
                          r"(?:Worked|Thought|Thinking|Reasoned) for \d+\s*(?:s|sec|seconds?|m|min|minutes?)?)$",
                          re.IGNORECASE)


def _is_echo(text: str, sent: str) -> bool:
    """The page is showing what we just typed (a too-broad selector caught the user's bubble)."""
    t, s = " ".join(text.split()), " ".join(sent.split())
    return bool(t) and (t == s or (len(t) >= 20 and s.startswith(t))
                        or (len(s) >= 20 and t.startswith(s)))


def resp_ok(data: bytes) -> bool:
    return len(data) > 200 and not data.lstrip()[:15].lower().startswith((b"<!doctype", b"<html"))


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.\- ]+", "_", name or "").strip(" .")[:80]


def clean_reply(text: str) -> str:
    lines = text.splitlines()
    while lines and (not lines[0].strip() or _STATUS_LINE.match(lines[0].strip())):
        lines.pop(0)
    text = "\n".join(lines)
    label = _SR_LABEL.match(text)
    if label:
        text = text[label.end():]
    # claude.ai: a preview of the first line, then thinking or tool notes ("Used Search Past
    # Conversations, ..."), then the answer itself starting with that line again.
    first = text.split("\n", 1)[0].strip()
    again = text.find("\n" + first, len(first)) if first else -1
    if again > 0 and (label or (again < 800 and all(len(x) < 160 for x in text[:again].splitlines()))):
        text = text[again + 1:]
    lines = [_ICON_CHARS.sub("", line).rstrip() for line in text.splitlines()]
    while lines and (not lines[-1].strip() or _CHROME_LINE.match(lines[-1].strip())):
        lines.pop()
    return "\n".join(lines).strip()


# Cookie notices that cover the message box (grok.com's sits on top of it): refuse, else close.
_COOKIE_BUTTONS = ", ".join([
    "#onetrust-reject-all-handler",
    *(f'{box} button:has-text("{label}")'
      for box in ('[data-cookie-banner]', '[aria-label*="cookie" i]', '[id*="cookie" i]', '[class*="cookie" i]')
      for label in ("Reject all", "Reject", "Decline", "Only necessary", "Necessary only")),
    'button[aria-label*="cookie" i][aria-label*="dismiss" i]',
    '[data-cookie-banner] button[aria-label="Close"]',
])

# When a site renames its message box: every chat page has exactly one visible text box.
_ANY_COMPOSER = 'textarea, [contenteditable="true"], [role="textbox"]'

# What Check lists when a site changes its layout: the visible boxes and buttons to pick from.
_INPUTS_JS = """() => [...document.querySelectorAll(
  'textarea, [contenteditable="true"], input[type="file"], button[aria-label], button[data-testid]')]
  .filter(el => el.type === 'file' || el.offsetWidth || el.offsetHeight)
  .sort((a, b) => (a.tagName === 'BUTTON') - (b.tagName === 'BUTTON')).slice(0, 40)
  .map(el => Object.fromEntries(Object.entries({tag: el.tagName.toLowerCase(), id: el.id,
    name: el.getAttribute('name'), testid: el.getAttribute('data-testid'),
    aria: el.getAttribute('aria-label'), placeholder: el.getAttribute('placeholder')})
    .filter(([, v]) => v)))"""


class SiteAdapter:
    def __init__(self, config: SiteConfig, load_timeout_s: float = 60.0,
                 screenshot_dir: Path | None = None, media_dir: Path | None = None):
        self.cfg = config
        self.load_timeout_ms = load_timeout_s * 1000
        self.screenshot_dir = screenshot_dir
        self.media_dir = media_dir  # images in replies are saved here

    async def open(self, page: Page, thread_url: str | None = None) -> None:
        """Open a new chat, or the bound remote conversation, and wait for the composer.
        A login or bot-check page fails at once instead of after the whole load timeout."""
        await page.goto(thread_url or self.cfg.new_chat_url, wait_until="domcontentloaded",
                        timeout=self.load_timeout_ms)
        composer = self._composer(page)
        deadline = time.monotonic() + self.load_timeout_ms / 1000
        while True:
            try:
                await composer.wait_for(state="visible", timeout=1000)
                if await self._signed_out(page):
                    raise await self.diagnose(page)
                await self._dismiss_cookie_notice(page)
                return
            except ProviderError:
                raise
            except Exception:
                pass
            try:
                blocked = bool(_LOGIN_URL.search(page.url) or _BOT_TITLE.search(await page.title()))
            except Exception:  # mid-navigation
                blocked = False
            if blocked or time.monotonic() > deadline:
                raise await self.diagnose(page)

    def _composer(self, page: Page):
        """The configured message box, else any visible text box (the site changed its layout)."""
        return page.locator(self.cfg.composer).or_(page.locator(_ANY_COMPOSER)) \
            .filter(visible=True).first

    async def _signed_out(self, page: Page) -> bool:
        return bool(self.cfg.signed_out) and \
            await page.locator(self.cfg.signed_out).filter(visible=True).count() > 0

    @staticmethod
    def _visible(page: Page, selector: str):
        """Sites keep hidden copies of their inputs (mobile layout, fallbacks): use the shown one."""
        return page.locator(selector).filter(visible=True).first

    async def diagnose(self, page: Page) -> ProviderError:
        """Work out why the composer is missing, instead of assuming the user is logged out."""
        title = await page.title()
        shot = await self._screenshot(page)
        where = f"{page.url}" + (f" (screenshot: {shot})" if shot else "")
        if _BOT_TITLE.search(title):
            return BotCheck(f"bot check at {where} - complete it with Connect, then retry")
        if _LOGIN_URL.search(page.url) or await self._signed_out(page):
            return NotLoggedIn(f"signed out at {where} - sign in again with Connect")
        return PageNotReady(
            f"message box not found within {self.load_timeout_ms / 1000:.0f}s at {where} - "
            "slow load, or the site changed its layout (press Check)")

    async def _screenshot(self, page: Page) -> str | None:
        if not self.screenshot_dir:
            return None
        try:
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
            path = self.screenshot_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{id(page)}.png"
            await page.screenshot(path=str(path))
            return str(path)
        except Exception:
            return None

    async def send(self, page: Page, text: str, timeout_s: float = 840.0,
                   files: list[str] = ()) -> str:
        await self._mark_old_replies(page)
        # The new reply is the one not marked: counting fails on sites that drop old messages
        # from the page as new ones arrive (long chats on claude.ai).
        replies = page.locator(f":is({self.cfg.assistant_message}):not([data-prism-seen])")

        composer = self._composer(page)
        try:
            await composer.click(timeout=3000)
        except Exception:  # something still covers it: focus and type without clicking
            await composer.focus()
        if files:
            await self._attach(page, files)
        await composer.fill(text)
        button = self._visible(page, self.cfg.send_button) if self.cfg.send_button else None
        try:
            if files:  # the send button stays disabled while the files upload
                await self._wait_enabled(button, 180)
            await button.click(timeout=2000)
        except Exception:  # no send button configured, or it was renamed: Enter sends
            await composer.press("Enter")

        deadline = time.monotonic() + timeout_s
        while not await replies.count():
            if time.monotonic() > deadline:
                raise ProviderError("no reply appeared")
            await asyncio.sleep(0.25)
        reply = replies.last
        text_out = await self._wait_until_stable(page, reply, deadline, text)
        return "\n\n".join([text_out] + await self._save_media(page, reply))

    async def _dismiss_cookie_notice(self, page: Page) -> None:
        try:
            button = page.locator(_COOKIE_BUTTONS).filter(visible=True).first
            if await button.count():
                await button.click(timeout=2000)
                await asyncio.sleep(0.5)
        except Exception:
            pass  # best effort: typing works without clicking anyway

    async def _mark_old_replies(self, page: Page) -> None:
        """Once the conversation has finished loading, tag every reply already on the page."""
        mark = """s => { const all = document.querySelectorAll(s);
                         all.forEach(e => e.setAttribute('data-prism-seen', '')); return all.length; }"""
        last, settle = -1, time.monotonic() + 10
        while time.monotonic() < settle:
            n = await page.evaluate(mark, self.cfg.assistant_message)
            if n == last:
                return
            last = n
            await asyncio.sleep(0.7)

    async def _attach(self, page: Page, files: list[str]) -> None:
        """Hand the files to the site's own upload: its file input, else its attach button."""
        inputs = page.locator(self.cfg.file_input)
        if await inputs.count():
            await inputs.last.set_input_files(list(files))
        elif self.cfg.attach_button:
            async with page.expect_file_chooser(timeout=10_000) as chooser:
                await self._visible(page, self.cfg.attach_button).click()
            await (await chooser.value).set_files(list(files))
        else:
            raise ProviderError("can't attach files here: no upload control found - set "
                                "file_input or attach_button under Selectors")
        await asyncio.sleep(1.5)  # let the site start the upload and show the preview

    @staticmethod
    async def _wait_enabled(button, timeout_s: float) -> None:
        if button is None:
            await asyncio.sleep(5)
            return
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                if await button.is_enabled(timeout=1000):
                    return
            except Exception:
                pass
            await asyncio.sleep(0.5)

    async def _save_media(self, page: Page, reply) -> list[str]:
        """Pictures and files in the reply, saved so PRISM can show and download them. Returned
        as markdown lines: ![image](media/x.png) and [file name](media/x.pdf)."""
        if not self.media_dir:
            return []
        self.media_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for name in await self._save_images(page, reply):
            lines.append(f"![image](media/{name})")
        for name, shown in await self._save_downloads(page, reply):
            lines.append(f"[\U0001F4C4 {shown}](media/{name})")
        return lines

    async def _save_images(self, page: Page, reply) -> list[str]:
        """Pictures (generated or quoted); small ones are icons and avatars. Only real images:
        a URL that answers with a web page (a login wall, an expired link) is skipped."""
        try:
            srcs = await reply.evaluate("""el => [...el.querySelectorAll('img')]
                .filter(i => (i.naturalWidth || i.width) >= 96 && (i.naturalHeight || i.height) >= 96)
                .map(i => i.currentSrc || i.src)""")
        except Exception:
            return []
        saved = []
        for src in dict.fromkeys(srcs):
            try:
                if src.startswith(("blob:", "data:")):  # only the page itself can read these
                    kind, data64 = await page.evaluate(
                        """async u => { const b = await (await fetch(u)).blob();
                           return [b.type, await new Promise(r => { const f = new FileReader();
                             f.onload = () => r(f.result.split(',')[1]); f.readAsDataURL(b); })]; }""", src)
                    data = base64.b64decode(data64)
                else:  # with the site's cookies
                    resp = await page.request.get(src)
                    data, kind = await resp.body(), resp.headers.get("content-type", "")
                kind = kind.split(";")[0].strip().lower()
                if not kind.startswith("image/") or not resp_ok(data):
                    continue
                ext = {"image/jpeg": "jpg", "image/svg+xml": "svg"}.get(kind, kind.split("/")[1])
                name = f"{uuid.uuid4().hex[:12]}-image.{ext}"
                (self.media_dir / name).write_bytes(data)
                saved.append(name)
            except Exception:
                continue
        return saved

    async def _save_downloads(self, page: Page, reply) -> list[tuple[str, str]]:
        """Files the bot made (claude's "Download" card on a PDF, links marked download)."""
        buttons = reply.locator('a[download], button:has-text("Download"), a:has-text("Download")')
        saved = []
        for i in range(min(await buttons.count(), 5)):
            try:
                async with page.expect_download(timeout=20_000) as info:
                    await buttons.nth(i).click(timeout=5000)
                download = await info.value
                shown = _safe_name(download.suggested_filename) or "file"
                name = f"{uuid.uuid4().hex[:12]}-{shown}"
                await download.save_as(str(self.media_dir / name))
                saved.append((name, shown))
            except Exception:
                continue
        return saved

    async def _wait_until_stable(self, page: Page, reply, deadline: float, sent: str = "",
                                 settle_s: float = 1.5, stuck_s: float = 60.0) -> str:
        """The reply is done when nothing is streaming and its text stopped changing.
        `reply` is re-resolved each time, so an echo of `sent` is skipped for the real reply."""
        last, stable_since = None, time.monotonic()
        while time.monotonic() < deadline:
            text = (await reply.inner_text()).strip()
            streaming = bool(self.cfg.streaming) and await page.locator(self.cfg.streaming).count() > 0
            quiet = time.monotonic() - stable_since
            if text != last or not text or _is_echo(text, sent):
                last, stable_since = text, time.monotonic()
            elif (quiet >= settle_s and not streaming) or quiet >= stuck_s:
                return clean_reply(text)  # done, or the site's "writing" marker got stuck
            await asyncio.sleep(0.3)
        raise ProviderError("reply did not finish before the timeout")

    async def doctor(self, page: Page) -> dict:
        """Report how many elements each configured selector matches on the new-chat page."""
        await page.goto(self.cfg.new_chat_url, wait_until="domcontentloaded")
        found = None
        try:
            box = self._composer(page)
            await box.wait_for(state="visible", timeout=20_000)
            found = await box.evaluate(
                "el => el.tagName.toLowerCase() + (el.id ? '#' + el.id : '')"
                " + (el.getAttribute('name') ? `[name=${el.getAttribute('name')}]` : '')")
        except Exception:
            pass
        signed_out = await self._signed_out(page)
        checks = {}
        for name in ("composer", "send_button", "assistant_message", "streaming", "signed_out"):
            selector = getattr(self.cfg, name)
            checks[name] = {"selector": selector,
                            "matches": await page.locator(selector).count() if selector else None}
        return {"url": page.url, "title": await page.title(),
                "logged_in": bool(found) and not signed_out and not _LOGIN_URL.search(page.url),
                "message_box": found, "checks": checks,
                "inputs_on_page": await page.evaluate(_INPUTS_JS)}
