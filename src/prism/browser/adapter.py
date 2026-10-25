"""Generic chat-site driver: type into the composer, send, wait for the reply to finish, read it."""

import asyncio
import base64
import re
import time
import uuid
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Awaitable, Callable

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
    # Deep research: what to click, in order, to switch it on for the next message (steps
    # separated by " ;; "; a last step that is a checkbox already on is not clicked again).
    research_mode: str = ""
    # A button the site shows mid-reply and waits on (Gemini's "Start research"): clicked.
    continue_button: str = ""
    # Text on the page that means research is still going, even with no Stop button
    # (a regular expression, e.g. ChatGPT's "Deep Research has started investigating").
    research_pending: str = ""
    # A document the reply made, shown as a card that opens in a side panel (claude.ai's
    # reports and artifacts): PRISM opens each card and adds the document to the reply.
    document_card: str = ""
    document_panel: str = ""
    # Text that means the research finished (looked for on the page and in its frames), and
    # an embedded frame holding the report (ChatGPT draws it in an app frame of its own).
    research_done: str = ""
    document_frame: str = ""
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


# The document in a side panel: its own content area when there is one (claude.ai's
# #markdown-artifact), else the panel without its toolbar buttons.
_DOC_TEXT_JS = """p => { const body = p.querySelector('#markdown-artifact, [class*="markdown"], pre, code') || p;
  const c = body.cloneNode(true); c.querySelectorAll('button, [role=toolbar], nav').forEach(x => x.remove());
  return c.innerText; }"""

RELOAD_STALLED_S = 120.0

_DOC_HTML_JS = """p => { const body = p.querySelector('#markdown-artifact, [class*="markdown"]') || p;
  return (REPLY_HTML)(body); }"""


def _doc_as_text(doc: dict) -> str:
    """A document as the bots read it: its title, then the text (code fenced)."""
    body = doc["text"] if doc["kind"] == "doc" else f"```{doc['kind']}\n{doc['text']}\n```"
    head = doc.get("title") or ""
    return body if not head or body.lstrip().startswith(head) else f"## {head}\n\n{body}"


PROCEED = "No preference - use your best judgement and go ahead with the research."


def _asks_first(reply: str) -> bool:
    """A short reply ending in questions: deep research asking before it starts."""
    tail = reply.strip()[-400:]
    return len(reply) < 1500 and tail.count("?") >= 1 and tail.rstrip().endswith(("?", ":"))


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
    cut = first.rstrip("\u2026.").rstrip() if first.endswith(("\u2026", "...")) else ""  # a shortened preview
    key = cut or first
    again = text.find("\n" + key, len(first)) if len(key) >= 8 or (key and not cut) else -1
    if again > 0 and (label or (cut and again < 1500) or
                      (again < 800 and all(len(x) < 160 for x in text[:again].splitlines()))):
        text = text[again + 1:]
    lines = []
    for line in text.splitlines():
        line = _ICON_CHARS.sub("", line).rstrip()
        prev = next((x for x in reversed(lines) if x.strip()), None)
        if line.strip() and line == prev and len(line) < 100:
            continue  # claude.ai shows each step ("Searched past chats") twice
        lines.append(line)
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
# Every chat site shows a Stop button while it writes; used when the reply had to be found
# by watching the page (its own "writing" marker may be gone too).
_ANY_STOP = 'button[aria-label*="Stop" i], button[data-testid*="stop" i]'
# How much the conversation shows: changes while a model thinks, searches or writes (progress
# lines, sources, the reply), even when the reply's own text does not.
_ACTIVITY_JS = """() => { const m = document.querySelector('main') || document.body, t = m.innerText;
  let h = 0; for (let i = 0; i < t.length; i++) h = (h * 31 + t.charCodeAt(i)) | 0;
  return t.length + ':' + h + ':' + m.getElementsByTagName('*').length; }"""

# --- finding things when the site changed its layout -----------------------------------
# The message box: the biggest visible text box, the lower the better (the composer sits at
# the bottom; a search box in the sidebar is small and at the top). Tagged for later use.
_PICK_COMPOSER_JS = """sel => {
  const vis = e => { const r = e.getBoundingClientRect(); return r.width > 20 && r.height > 10
    && getComputedStyle(e).visibility !== 'hidden'; };
  const all = [...document.querySelectorAll(sel)].filter(e => vis(e) && !e.closest('nav, aside'));
  if (!all.length) return null;
  const score = e => { const r = e.getBoundingClientRect(); return r.width * Math.max(r.height, 20) + r.top * 50; };
  const best = all.sort((a, b) => score(b) - score(a))[0];
  document.querySelectorAll('[data-prism-composer]').forEach(e => e.removeAttribute('data-prism-composer'));
  best.setAttribute('data-prism-composer', '');
  return best.tagName.toLowerCase();
}"""

# Before sending: record every piece of text the page adds from now on.
_WATCH_JS = """() => {
  if (window.__prismWatch) window.__prismWatch.obs.disconnect();
  const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
  const w = window.__prismWatch = {nodes: new Set(), before: norm(document.body.innerText)};
  w.obs = new MutationObserver(muts => { for (const m of muts) {
    if (m.type === 'characterData') w.nodes.add(m.target);
    else m.addedNodes.forEach(n => w.nodes.add(n));
  }});
  w.obs.observe(document.body, {childList: true, subtree: true, characterData: true});
}"""

# After sending: the reply is the deepest element holding most of the text added since -
# leaving out the message box, the sidebar, and the echo of what we sent. Tagged
# data-prism-reply; returns its length (0 = nothing yet).
_FIND_REPLY_JS = """prompt => {
  const w = window.__prismWatch; if (!w) return 0;
  const norm = s => (s || '').replace(/\\s+/g, ' ').trim(), p = norm(prompt);
  const items = []; let total = 0;
  for (const n of w.nodes) {
    if (!n.isConnected) continue;
    const el = n.nodeType === 1 ? n : n.parentElement;
    if (!el || el.closest('textarea, [contenteditable="true"], nav, aside, header, [data-prism-seen-user]')) continue;
    const t = norm(n.textContent);
    if (!t || (p && p.includes(t))) continue;   // our own message, shown back
    if (t.length >= 30 && w.before.includes(t)) continue;  // old messages the page re-drew
    items.push([el, t.length]); total += t.length;
  }
  if (!total) return 0;
  const held = new Map();
  for (const [el, n] of items)
    for (let a = el; a && a !== document.body; a = a.parentElement) held.set(a, (held.get(a) || 0) + n);
  let best = null;
  for (const [a, n] of held) if (n >= 0.7 * total && (!best || best.contains(a))) best = a;
  if (!best) return 0;
  document.querySelectorAll('[data-prism-reply]').forEach(e => e.removeAttribute('data-prism-reply'));
  best.setAttribute('data-prism-reply', '');
  return norm(best.innerText).length;
}"""

# A selector for what was found, to remember for next time: built from the element's (or,
# for a reply, its containers') stable attributes. A reply selector must match every reply
# and nothing else: not the message box, not the user's own message.
_NAME_IT_JS = """([kind, prompt]) => {
  const el = document.querySelector(kind === 'composer' ? '[data-prism-composer]' : '[data-prism-reply]');
  if (!el) return null;
  const norm = s => (s || '').replace(/\\s+/g, ' ').trim(), p = norm(prompt);
  const stable = v => v && v.length <= 60 && !/\\d{3,}|[:][a-z0-9]+[:]|^radix-|^headlessui-/i.test(v);
  const q = v => '"' + v.replace(/["\\\\]/g, '\\\\$&') + '"';
  const chain = [el]; if (kind !== 'composer') for (let a = el.parentElement, i = 0; a && a !== document.body && i < 4; a = a.parentElement, i++) chain.push(a);
  const cands = [];
  for (const e of chain) {
    const tag = e.tagName.toLowerCase();
    for (const at of ['data-testid', 'data-message-author-role', 'data-role', 'role', 'aria-label', 'placeholder', 'name'])
      if (stable(e.getAttribute(at))) cands.push(`${tag}[${at}=${q(e.getAttribute(at))}]`);
    if (stable(e.id)) cands.push(`${tag}#${CSS.escape(e.id)}`);
    for (const c of e.classList) if (/^[a-z][a-z-]{2,30}$/i.test(c)) cands.push(`${tag}.${c}`);
  }
  const vis = e => e.offsetWidth || e.offsetHeight;
  for (const c of cands) {
    let all; try { all = [...document.querySelectorAll(c)]; } catch (e) { continue; }
    if (!all.some(m => m === el || m.contains(el))) continue;
    if (kind === 'composer') { if (all.filter(vis).length === 1) return c; continue; }
    const last = all[all.length - 1];
    if (!(last === el || last.contains(el))) continue;
    if (all.some(m => m.matches('textarea, [contenteditable="true"]') || m.querySelector('textarea, [contenteditable="true"]'))) continue;
    if (p && all.some(m => { const t = norm(m.innerText); return t && (t === p || (p.length >= 20 && t.includes(p))); })) continue;
    return c;
  }
  return null;
}"""

# Why the message box is blocked: dialogs, alerts and notices the page shows (limit reached,
# conversation too long, upgrade offers) - their text, briefly.
_NOTICES_JS = """() => [...document.querySelectorAll('[role="dialog"], [role="alertdialog"], [role="alert"], [role="status"], [aria-live]')]
  .filter(e => e.offsetWidth || e.offsetHeight).map(e => (e.innerText || '').replace(/\\s+/g, ' ').trim())
  .filter(t => t.length > 3).join(' | ').slice(0, 300)"""

# The finished reply as text, with code blocks fenced again: innerText of a rendered code
# block is the language label and the code with no fences ("html\n<!doctype html>...").
_REPLY_TEXT_JS = """el => {
  const copy = el.cloneNode(true);
  copy.style.cssText = 'position:fixed;left:-99999px;top:0;width:900px';
  document.body.appendChild(copy);
  try {
    for (const pre of copy.querySelectorAll('pre')) {
      const code = pre.querySelector('code') || pre;
      const cls = [...code.classList, ...pre.classList].find(c => /^(language|lang)-/.test(c));
      let lang = cls ? cls.replace(/^(language|lang)-/, '') : '';
      if (!lang) {  // claude.ai and ChatGPT name the language in a header above the code
        const head = pre.parentElement && pre.parentElement.querySelector(':scope > div:first-child');
        const t = head && head !== pre ? (head.innerText || '').trim().split(/\\s/)[0].toLowerCase() : '';
        if (/^[a-z+#]{1,12}$/.test(t)) lang = t;
      }
      const text = (code.innerText || code.textContent || '').replace(/\\n$/, '');
      // The code block's own wrapper (a header naming the language, a Copy button) goes with
      // it - but only while everything beside it is such chrome: short, no paragraphs. A short
      // reply ("RUN @python" + one line of code) must keep its text.
      const chrome = (parent, keep) => [...parent.children].every(ch => ch === keep || (
        !/^(P|LI|UL|OL|H[1-6]|TABLE|BLOCKQUOTE)$/.test(ch.tagName)
        && !ch.querySelector('p, li, h1, h2, h3, h4, h5, h6, table, blockquote')
        && (ch.innerText || '').trim().length <= 40));
      let wrap = pre;
      while (wrap.parentElement && wrap.parentElement !== copy && wrap.parentElement.querySelectorAll('pre').length === 1
             && chrome(wrap.parentElement, wrap)) {
        if (!lang) {  // grok.com: the language is a label beside the code's own box
          for (const ch of wrap.parentElement.children) {
            const t = ch === wrap ? '' : (ch.innerText || '').trim().split(/\\s/)[0].toLowerCase();
            if (/^[a-z+#]{1,12}$/.test(t) && t !== 'copy') { lang = t; break; }
          }
        }
        wrap = wrap.parentElement;
      }
      const box = document.createElement('div');
      box.textContent = '\\n```' + lang + '\\n' + text + '\\n```\\n';
      box.style.whiteSpace = 'pre';
      wrap.replaceWith(box);
    }
    return copy.innerText;
  } finally { copy.remove(); }
}"""

# The finished reply as the site showed it (headings, lists, tables, code), for PRISM to show
# the same way. Only structure is kept: the page sanitizes it again before showing it.
_REPLY_HTML_JS = """el => {
  const copy = el.cloneNode(true);
  copy.querySelectorAll('script, style, link, button, svg, input, textarea, select, form, nav, iframe, object, '
    + 'canvas, video, audio, [aria-hidden="true"], .sr-only, .visually-hidden, [role="button"], [role="tooltip"]')
    .forEach(e => e.remove());
  for (const e of copy.querySelectorAll('*')) {
    for (const a of [...e.attributes]) {
      const keep = (a.name === 'href' && /^https?:/i.test(a.value)) || a.name === 'colspan' || a.name === 'rowspan'
        || (a.name === 'class' && /language-/.test(a.value));
      if (!keep) e.removeAttribute(a.name);
      else if (a.name === 'class') e.setAttribute('class', (a.value.match(/language-[\\w+#-]+/) || [''])[0]);
    }
  }
  copy.querySelectorAll('img').forEach(e => e.remove());  // saved separately, with the site's cookies
  return copy.innerHTML.replace(/<!--[\\s\\S]*?-->/g, '').replace(/\\s{3,}/g, '  ').slice(0, 400000);
}"""

# For asking another assistant: the page's controls, numbered - structure only, no chat text.
_OUTLINE_JS = """() => {
  const vis = e => { const r = e.getBoundingClientRect(); return r.width > 4 && r.height > 4; };
  const els = [...document.querySelectorAll('textarea, input:not([type=hidden]):not([type=file]):not([type=checkbox]):not([type=radio]),'
    + ' [contenteditable], [role="textbox"], button, [role="button"]')].filter(vis).slice(0, 80);
  return els.map((e, i) => { e.setAttribute('data-prism-pick', i); const r = e.getBoundingClientRect();
    const o = {n: i, tag: e.tagName.toLowerCase(), type: e.getAttribute('type'), role: e.getAttribute('role'),
      editable: e.isContentEditable || undefined, label: e.getAttribute('aria-label'),
      placeholder: e.getAttribute('placeholder') || e.getAttribute('data-placeholder'),
      testid: e.getAttribute('data-testid'),
      button_text: e.tagName === 'BUTTON' ? (e.innerText || '').trim().slice(0, 24) : undefined,
      at: `${Math.round(r.left)},${Math.round(r.top)} ${Math.round(r.width)}x${Math.round(r.height)}`};
    return Object.fromEntries(Object.entries(o).filter(([, v]) => v !== null && v !== undefined && v !== ''));
  });
}"""
_LAYOUT_QUESTION = (
    "I'm automating a chat website and can't find its message box (where a user types a new "
    "chat message). Here are the visible controls on the page, numbered, with their position "
    "and size (x,y widthxheight; the page is about 1280x720):\n\n{outline}\n\nWhich number is "
    "the message box? Reply with the number only, or 'none'.")

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
                 screenshot_dir: Path | None = None, media_dir: Path | None = None,
                 on_learn: Callable[[dict], None] | None = None,
                 advisor: Callable[[str], Awaitable[str]] | None = None,
                 fallback_after_s: float = 30.0, idle_s: float = 1200.0, stuck_s: float = 300.0):
        self.cfg = config
        # No fixed time for a reply: PRISM waits while the site shows it is working, and gives
        # up after idle_s with nothing changing on the page (3x that while a Stop button still
        # shows). A Stop button that stays up with the reply unchanged for stuck_s is stale.
        self.idle_s, self.stuck_s = idle_s, stuck_s
        self.research_settle_s = 15.0  # research replies arrive in parts: wait this long for quiet
        self.last_activity: float | None = None  # heartbeat while PRISM is still judging the reply
        self._page_mark: str | None = None
        self.load_timeout_ms = load_timeout_s * 1000
        self.screenshot_dir = screenshot_dir
        self.media_dir = media_dir  # images in replies are saved here
        # When the site changed its layout: what worked is passed to on_learn (saved, used
        # first next time); advisor(question) asks another assistant when PRISM can't tell.
        self.on_learn = on_learn
        self.advisor = advisor
        self.fallback_after_s = fallback_after_s  # no reply on the known selector by then: watch the page
        self.notes: list[str] = []  # what changed in the last send, for the room to show
        self.last_html: str | None = None  # the last reply as the site formatted it
        self.last_documents: list[dict] = []  # documents the reply made: {title, kind, text, html}
        self._composer_guessed = False

    async def open(self, page: Page, thread_url: str | None = None) -> None:
        """Open a new chat, or the bound remote conversation, and wait for the composer.
        A login or bot-check page fails at once instead of after the whole load timeout."""
        await page.goto(thread_url or self.cfg.new_chat_url, wait_until="domcontentloaded",
                        timeout=self.load_timeout_ms)
        start = time.monotonic()
        deadline = start + self.load_timeout_ms / 1000
        self.notes, self._composer_guessed = [], False
        while True:
            try:
                # The known message box; only well after it should have shown up, the likeliest
                # text box (sites load slowly; a guess is learned once a reply comes back).
                guess = time.monotonic() - start > min(15.0, self.load_timeout_ms / 2000)
                if await self._find_composer(page, guess=guess):
                    if await self._signed_out(page):
                        raise await self.diagnose(page)
                    await self._dismiss_cookie_notice(page)
                    return
            except ProviderError:
                raise
            except Exception:
                pass  # mid-navigation
            try:
                blocked = bool(_LOGIN_URL.search(page.url) or _BOT_TITLE.search(await page.title()))
            except Exception:  # mid-navigation
                blocked = False
            if blocked or time.monotonic() > deadline:
                error = await self.diagnose(page)
                if isinstance(error, PageNotReady) and await self._ask_for_composer(page):
                    return
                raise error
            await asyncio.sleep(1)

    async def _find_composer(self, page: Page, guess: bool = True):
        """The configured message box, else (guess) the likeliest visible text box on the page."""
        box = page.locator(self.cfg.composer).filter(visible=True).first
        if await box.count():
            if self._composer_guessed:  # it was only slow to load: nothing to learn
                self._composer_guessed = False
                await page.evaluate("() => document.querySelectorAll('[data-prism-composer]')"
                                    ".forEach(e => e.removeAttribute('data-prism-composer'))")
            return box
        tagged = page.locator("[data-prism-composer]").filter(visible=True).first
        if await tagged.count():
            return tagged
        if guess and await page.evaluate(_PICK_COMPOSER_JS, _ANY_COMPOSER):
            self._composer_guessed = True
            return tagged
        return None

    async def _ask_for_composer(self, page: Page) -> bool:
        """Last resort: show another assistant the page's controls (no chat text) and try the
        one it names - kept only if it really is a box we can type into."""
        if not self.advisor:
            return False
        try:
            outline = await page.evaluate(_OUTLINE_JS)
            if not outline:
                return False
            lines = "\n".join(" ".join(f"{k}={v}" for k, v in o.items()) for o in outline)
            answer = await self.advisor(_LAYOUT_QUESTION.format(outline=lines))
            n = re.search(r"\d+", answer or "")
            if not n:
                return False
            box = page.locator(f'[data-prism-pick="{n.group()}"]')
            if not await box.count() or not await box.evaluate(
                    "e => e.isContentEditable || ['TEXTAREA', 'INPUT'].includes(e.tagName)"):
                return False
            await page.evaluate("() => document.querySelectorAll('[data-prism-composer]')"
                                ".forEach(e => e.removeAttribute('data-prism-composer'))")
            await box.evaluate("e => e.setAttribute('data-prism-composer', '')")
            self._composer_guessed = True
            self.notes.append("another assistant found the new message box")
            return True
        except Exception:
            return False

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

    async def send(self, page: Page, text: str, timeout_s: float | None = None,
                   files: list[str] = (), research: bool = False) -> str:
        """timeout_s: give up after this long with no activity on the page (default idle_s).
        research: switch the site's deep research mode on for this message first."""
        self._research, self._continued = research, False
        self.last_documents = []
        reply = await self._send_once(page, text, timeout_s, files, research)
        if research and _asks_first(reply):
            # Deep research on some sites asks clarifying questions before it starts.
            self.notes.append("research asked questions first - told it to go ahead with its best judgement")
            reply = await self._send_once(page, PROCEED, timeout_s)
        return reply

    async def _send_once(self, page: Page, text: str, timeout_s: float | None = None,
                         files: list[str] = (), research: bool = False) -> str:
        idle = timeout_s or self.idle_s
        self.last_html = None
        self._beat()
        await self._mark_old_replies(page)
        # The new reply is the one not marked: counting fails on sites that drop old messages
        # from the page as new ones arrive (long chats on claude.ai).
        replies = page.locator(f":is({self.cfg.assistant_message}):not([data-prism-seen])")

        composer = await self._find_composer(page)
        if composer is None:
            raise await self.diagnose(page)
        await page.evaluate(_WATCH_JS)  # in case the reply selector no longer matches
        try:
            await composer.click(timeout=3000)
        except Exception:  # something still covers it: focus and type without clicking
            await composer.focus()
        if research:
            await self._research_on(page)
            try:
                await composer.click(timeout=3000)
            except Exception:
                await composer.focus()
        if files:
            await self._attach(page, files)
        await self._type(page, composer, text)
        button = self._visible(page, self.cfg.send_button) if self.cfg.send_button else None
        try:
            if files:  # the send button stays disabled while the files upload
                await self._wait_enabled(button, 180)
            await button.click(timeout=2000)
        except Exception:  # no send button configured, or it was renamed: Enter sends
            await composer.press("Enter")

        watch_from = time.monotonic() + self.fallback_after_s
        watching = False
        marker = ", ".join(x for x in (self.cfg.streaming, _ANY_STOP) if x)
        changed_at = time.monotonic()
        while not await replies.count():
            self._beat()
            await self._press_continue(page)
            if await self._page_changed(page) or await self._showing(page, marker):
                changed_at = time.monotonic()
            elif time.monotonic() - changed_at > idle:
                raise ProviderError(f"no reply appeared, and the page showed no activity for {round(idle / 60)} min")
            # The site may have renamed its replies: read whatever the page added instead.
            if time.monotonic() > watch_from and await page.evaluate(_FIND_REPLY_JS, text):
                watching = True
                break
            await asyncio.sleep(0.25)
        if watching:
            async def current():  # the known selector may still turn up; else re-find it
                if await replies.count():
                    return replies.last
                await page.evaluate(_FIND_REPLY_JS, text)
                return page.locator("[data-prism-reply]")
            text_out = await self._wait_until_stable(page, current, idle, text, settle_s=5.0,
                                                     streaming=_ANY_STOP)
            reply = await current()
        else:
            reply = replies.last
            text_out = await self._wait_until_stable(page, reply, idle, text)
        await self._learn(page, text, watching and not await replies.count())
        return "\n\n".join([text_out] + await self._save_media(page, reply))

    async def _learn(self, page: Page, sent: str, reply_found_by_watching: bool) -> None:
        """The turn worked with what PRISM found itself: name it, and remember it."""
        learned = {}
        try:
            if self._composer_guessed:
                if sel := await page.evaluate(_NAME_IT_JS, ["composer", sent]):
                    learned["composer"] = sel
                self.notes.append("the message box moved - found it" + (" and saved it" if sel else ""))
            if reply_found_by_watching:
                if sel := await page.evaluate(_NAME_IT_JS, ["reply", sent]):
                    learned["assistant_message"] = sel
                self.notes.append("the reply layout changed - read it by watching the page"
                                  + (" and saved the new layout" if sel else ""))
        except Exception:
            pass
        if learned:
            self.cfg = self.cfg.with_overrides(learned)
            if self.on_learn:
                try:
                    self.on_learn(learned)
                except Exception:
                    pass

    async def _type(self, page: Page, composer, text: str) -> None:
        """Put the message in the box. When the site has marked the box read-only or covered
        it (a dialog, a usage-limit notice), close what can be closed and type with the
        keyboard; if nothing lands in the box, say what the page shows instead."""
        try:
            if await composer.is_editable(timeout=3000):
                await composer.fill(text, timeout=30_000)
                return
        except Exception:
            pass
        try:
            await page.keyboard.press("Escape")
            try:
                await composer.click(timeout=3000)
            except Exception:
                await composer.focus()
            await page.keyboard.press("ControlOrMeta+A")
            await page.keyboard.insert_text(text)
            typed = await composer.evaluate("e => ((e.value !== undefined ? e.value : e.innerText) || '').trim().length")
        except Exception:
            typed = 0
        if not typed:
            raise await self._blocked(page)

    async def _blocked(self, page: Page) -> ProviderError:
        shot = await self._screenshot(page)
        try:
            shown = await page.evaluate(_NOTICES_JS)
        except Exception:
            shown = ""
        return PageNotReady(
            "the message box can't be typed into" + (f" - the page says: {shown}" if shown else "")
            + (f" (screenshot: {shot})" if shot else "")
            + " - a usage limit, a dialog, or a conversation that is too long; open the site "
              "with Connect to see it")

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

    def _beat(self) -> None:
        self.last_activity = time.monotonic()

    async def _research_on(self, page: Page) -> None:
        steps = [x.strip() for x in self.cfg.research_mode.split(";;") if x.strip()]
        if not steps:
            self.notes.append("this site has no research mode set up - sent as a normal message")
            return
        try:
            for i, sel in enumerate(steps):
                el = page.locator(sel).filter(visible=True).first
                await el.wait_for(timeout=5000)
                last = i == len(steps) - 1
                on = await el.evaluate("e => e.getAttribute('aria-checked') || e.getAttribute('aria-pressed') || ''")
                if last and on == "true":
                    break  # already on
                await el.click(timeout=3000)
                await page.wait_for_timeout(600)
            if await page.locator('[role="menu"]').filter(visible=True).count():
                await page.keyboard.press("Escape")
            self.notes.append("research mode on")
        except Exception as e:
            await page.keyboard.press("Escape")
            self.notes.append(f"couldn't switch research mode on ({str(e).splitlines()[0][:80]}) - sent as a normal message")

    async def _page_text(self, page: Page) -> str:
        """The end of the conversation's text, and of each embedded frame's."""
        parts = []
        for f in page.frames:
            try:
                parts.append(await f.evaluate(
                    "() => (document.querySelector('main') || document.body)?.innerText.slice(-4000) || ''"))
            except Exception:
                continue
        return "\n".join(parts)

    async def _frame_documents(self, page: Page) -> list[dict]:
        """Reports drawn inside an embedded frame (and the frames within it)."""
        out = []
        frames = page.locator(self.cfg.document_frame)
        for i in range(await frames.count())[-1:]:  # the newest: older ones belong to earlier turns
            try:
                top = await (await frames.nth(i).element_handle(timeout=3000)).content_frame()
                if top is None:
                    continue
                best, text = None, ""
                for f in [top] + top.child_frames + [g for c in top.child_frames for g in c.child_frames]:
                    try:
                        t = await f.evaluate("() => document.body ? document.body.innerText : ''")
                    except Exception:
                        continue
                    if len(t) > len(text):
                        best, text = f, t
                text = re.sub(r"(?:\n\d(?=\n)){5,}\n?", " ", text)  # animated counters: runs of digits
                text = re.sub(r"[ \t]+\n", "\n", text).strip()
                if len(text) > 40:
                    try:
                        html = await best.evaluate(f"() => ({_REPLY_HTML_JS})(document.body)")
                    except Exception:
                        html = None
                    title = (await frames.nth(i).get_attribute("title")) or "Report"
                    first = next((x for x in text.splitlines() if len(x.strip()) > 8 and not x.startswith("Research completed")), title)
                    out.append({"title": first.strip()[:120], "kind": "doc", "text": text, "html": html})
                    self.notes.append("read the research report from its embedded frame")
            except Exception as e:
                self.notes.append(f"couldn't read the embedded report ({str(e).splitlines()[0][:80]})")
        return out

    async def _read_documents(self, page: Page, reply) -> list[dict]:
        """Open each document card in the reply and read the panel it opens; read reports
        drawn in embedded frames."""
        out = await self._frame_documents(page) if self.cfg.document_frame else []
        if not (self.cfg.document_card and self.cfg.document_panel):
            return out
        try:
            cards = reply.locator(self.cfg.document_card)
            for i in range(min(await cards.count(), 5)):
                card = cards.nth(i)
                kind = (await card.get_attribute("data-sheet-kind")) or ""
                await card.click(timeout=5000)
                panel = page.locator(self.cfg.document_panel).filter(visible=True).last
                await panel.wait_for(timeout=10_000)
                await page.wait_for_timeout(1500)  # the document renders after the panel
                title = ((await panel.get_attribute("aria-label")) or "").split(":", 1)[-1].strip()
                body = (await panel.evaluate(_DOC_TEXT_JS)).strip()
                if not body:
                    continue
                code = bool(kind) and kind not in ("markdown", "text", "document")
                try:
                    html = None if code else await panel.evaluate(_DOC_HTML_JS.replace("REPLY_HTML", _REPLY_HTML_JS))
                except Exception:
                    html = None
                lang = {"react": "jsx", "application/vnd.ant.react": "jsx", "text/html": "html"}.get(kind, kind)
                out.append({"title": title or "Untitled", "kind": lang if code else "doc", "text": body, "html": html})
                self.notes.append(f"read the document \"{title or 'untitled'}\" from its panel")
        except Exception as e:
            self.notes.append(f"couldn't read a document the reply made ({str(e).splitlines()[0][:80]})")
        return out

    async def _press_continue(self, page: Page) -> None:
        """The site paused for the user (Gemini shows its research plan and waits): go on."""
        if not self.cfg.continue_button or getattr(self, "_continued", False):
            return  # once per message: the button can stay on screen after it was used
        try:
            button = page.locator(self.cfg.continue_button).filter(visible=True).first
            if await button.count():
                label = (await button.inner_text(timeout=1000)).strip() or "continue"
                await button.click(timeout=3000)
                self._continued = True
                self.notes.append(f"clicked \"{label}\" for you")
        except Exception:
            pass

    async def _showing(self, page: Page, selector: str) -> bool:
        try:
            return bool(selector) and await page.locator(selector).filter(visible=True).count() > 0
        except Exception:
            return False

    async def _page_changed(self, page: Page) -> bool:
        try:
            mark = await page.evaluate(_ACTIVITY_JS)
        except Exception:
            return False
        changed, self._page_mark = mark != self._page_mark, mark
        return changed

    async def _wait_until_stable(self, page: Page, reply, idle_s: float, sent: str = "",
                                 settle_s: float = 1.5, streaming: str = "") -> str:
        """The reply is done when the site no longer shows it is working (its Stop button or
        streaming marker is gone) and the text stopped changing. There is no time limit while
        the page keeps changing - deep research can take an hour. It ends early only when:
        the Stop button stays up but nothing at all changed for stuck_s (a stale marker), or
        nothing changed for idle_s (3x that while a Stop button shows).
        `reply` is re-resolved each time, so an echo of `sent` is skipped for the real reply;
        it may also be an async function returning the locator to read."""
        research = getattr(self, "_research", False)
        # Research mode: any Stop button counts (claude.ai marks its message done while the
        # research card keeps working), and the reply must stay put a while before it counts.
        marker = ", ".join(s for s in (self.cfg.streaming, streaming, _ANY_STOP if research else "") if s)
        settle_s = max(settle_s, self.research_settle_s) if research else settle_s
        pending = re.compile(self.cfg.research_pending, re.I) if self.cfg.research_pending else None
        finished = re.compile(self.cfg.research_done, re.I) if self.cfg.research_done else None
        last, stable_since = None, time.monotonic()
        still_since = time.monotonic()  # the last time anything on the page changed
        reloaded_at = time.monotonic()
        while True:
            self._beat()
            await self._press_continue(page)
            loc = await reply() if callable(reply) else reply
            try:
                text = (await loc.inner_text(timeout=5000)).strip()
            except Exception:  # replaced while we read it
                text = ""
            streaming_now = await self._showing(page, marker)
            researching = False
            if pending and research:  # only in research mode: old research text stays on the page
                seen = await self._page_text(page)
                researching = bool(pending.search(seen)) and not (finished and finished.search(seen))
            now = time.monotonic()
            if text != last or not text or _is_echo(text, sent):
                last, stable_since = text, now
                still_since = now
            elif await self._page_changed(page):
                still_since = now  # thinking, searching, sources arriving: still working
            quiet, still = now - stable_since, now - still_since
            usable = bool(text) and not _is_echo(text, sent)
            done = usable and not researching and (
                (quiet >= settle_s and not streaming_now) or (streaming_now and still >= self.stuck_s))
            # Research can sit for half an hour with nothing new on the page (ChatGPT works
            # server-side and shows the report when it is done).
            limit = max(idle_s * 3, 5400.0) if researching else idle_s * (3 if streaming_now else 1)
            if researching and still >= RELOAD_STALLED_S and now - reloaded_at >= RELOAD_STALLED_S:
                # Some sites stop updating the open page while research runs (gemini.google.com
                # makes the plan and the report but the page keeps saying "Generating research
                # plan"): open the chat again to see where it is.
                reloaded_at = now
                try:
                    await page.goto(page.url, wait_until="domcontentloaded")
                    await page.wait_for_timeout(6000)
                    if not any("reloaded" in n for n in self.notes):
                        self.notes.append("the page stopped updating during research - reloaded it to follow along")
                except Exception:
                    pass
            if not done and still >= limit:
                if not usable:
                    raise ProviderError(f"no reply: nothing changed on the page for {round(still / 60)} min")
                self.notes.append(f"the reply stopped changing for {round(still / 60)} min - took it as finished")
                done = True
            if done:
                try:  # read it once more, with code blocks fenced
                    if "<pre" in await loc.inner_html(timeout=5000):
                        text = (await loc.evaluate(_REPLY_TEXT_JS)).strip() or text
                except Exception:
                    pass
                try:
                    self.last_html = await loc.evaluate(_REPLY_HTML_JS)
                except Exception:
                    self.last_html = None
                docs = await self._read_documents(page, loc)
                self.last_documents = docs
                # The bots get the documents as text; the page shows them in its document view.
                return clean_reply(text) + "".join(f"\n\n{_doc_as_text(d)}" for d in docs)
            await asyncio.sleep(0.3 if quiet < 30 else 1.0)

    async def doctor(self, page: Page) -> dict:
        """Report how many elements each configured selector matches on the new-chat page."""
        await page.goto(self.cfg.new_chat_url, wait_until="domcontentloaded")
        found = None
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not found:
            try:
                box = await self._find_composer(page, guess=time.monotonic() > deadline - 15)
                if box:
                    found = await box.evaluate(
                        "el => el.tagName.toLowerCase() + (el.id ? '#' + el.id : '')"
                        " + (el.getAttribute('name') ? `[name=${el.getAttribute('name')}]` : '')")
            except Exception:
                pass
            if not found:
                await asyncio.sleep(1)
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
