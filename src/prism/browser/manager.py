"""One shared browser for every platform: a single persistent profile (logins for all sites
live side by side in it, and survive restarts) and one tab per request.

Work runs headless. Signing in needs a person at a real window, so Connect relaunches the same
profile with a visible window; the next request switches back to headless. One browser process
instead of one per platform keeps memory flat however many platforms run in parallel."""

import asyncio
import ctypes
import logging
import os
import sys
from pathlib import Path

from playwright.async_api import BrowserContext, Page, Playwright, async_playwright

log = logging.getLogger(__name__)

# Try the browsers the recipient already has before falling back to Playwright's Chromium.
# Edge ships with every Windows 10/11 install.
CHANNELS = ["msedge", "chrome", None]

# Platforms run in parallel in tabs of one browser; keep background tabs at full speed.
COMMON_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-renderer-backgrounding",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
]
# Only used when headless is turned off: a real window parked off-screen.
HIDDEN_ARGS = ["--window-position=-32000,-32000", "--window-size=1280,900"]

PROFILE = "shared"
MIN_FREE_MB = 1024  # refuse to start a browser below this, instead of starving the machine

# Errors that mean the Playwright driver or the browser is gone, not that a page misbehaved.
_DEAD = ("connection closed", "target page, context or browser has been closed",
         "browser has been closed", "browser closed", "has been disconnected")


def is_dead_browser_error(e: BaseException) -> bool:
    return any(s in str(e).lower() for s in _DEAD)


def free_memory_mb() -> int | None:
    """Physical memory available to new processes (Windows only; None elsewhere)."""
    if sys.platform != "win32":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = MemoryStatus(dwLength=ctypes.sizeof(MemoryStatus))
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullAvailPhys // (1024 * 1024)


class BrowserManager:
    def __init__(self, profiles_dir: Path, headless: bool = True,
                 channels: list[str | None] | None = None):
        """headless: how requests run. Connect always shows a window."""
        self.profiles_dir = Path(profiles_dir)
        self.headless = headless
        self.channels = channels or CHANNELS
        self._pw: Playwright | None = None
        self._ctx: BrowserContext | None = None
        self._visible = False
        self._user_agent: str | None = None
        self._locks: dict[str, asyncio.Lock] = {}
        self._start_lock = asyncio.Lock()

    def lock(self, platform: str) -> asyncio.Lock:
        """One request at a time per platform (its tabs share the site's session)."""
        return self._locks.setdefault(platform, asyncio.Lock())

    async def context(self, *, visible: bool = False) -> BrowserContext:
        """The shared browser. visible=True (sign-in) relaunches it with a window on screen;
        the next visible=False call relaunches it headless. The profile, and so every login,
        is kept across relaunches - but tabs open at that moment are closed."""
        async with self._start_lock:
            if self._ctx is not None and self._visible == visible:
                return self._ctx
            await self._shutdown()
            mb = free_memory_mb()
            if mb is not None and mb < MIN_FREE_MB:
                raise RuntimeError(f"only {mb} MB of memory free - close some programs, then retry "
                                   f"(PRISM needs about {MIN_FREE_MB} MB to start its browser)")
            self._pw = await async_playwright().start()
            ctx = await self._launch(self.profiles_dir / PROFILE, visible)
            ctx.on("close", lambda c: self._forget(c))
            self._ctx, self._visible = ctx, visible
            self._user_agent = None
            return ctx

    def _forget(self, ctx: BrowserContext) -> None:
        if self._ctx is ctx:
            self._ctx = None

    async def _launch(self, user_data_dir: Path, visible: bool) -> BrowserContext:
        user_data_dir.mkdir(parents=True, exist_ok=True)
        headless = self.headless and not visible
        args = COMMON_ARGS + ([] if visible or headless else HIDDEN_ARGS)
        opts = dict(headless=headless, no_viewport=not headless, args=args)
        errors = []
        for channel in self.channels:
            try:
                return await self._pw.chromium.launch_persistent_context(
                    str(user_data_dir), channel=channel, **opts)
            except Exception as e:
                errors.append(f"{channel or 'chromium'}: {str(e).splitlines()[0]}")
        # Nothing usable on this machine: fetch Playwright's Chromium once, then retry.
        await install_chromium()
        try:
            return await self._pw.chromium.launch_persistent_context(str(user_data_dir), **opts)
        except Exception as e:
            raise RuntimeError("No usable browser: " + "; ".join(errors + [str(e)])) from e

    async def new_page(self, *, visible: bool = False) -> Page:
        """A new tab. If the browser or its driver died (crash, out of memory), start a
        fresh one and retry once instead of failing every request until PRISM restarts."""
        for attempt in (1, 2):
            try:
                page = await (await self.context(visible=visible)).new_page()
                await self._hide_headless(page)
                return page
            except Exception as e:
                if attempt == 2 or not is_dead_browser_error(e):
                    raise
                log.warning("browser is gone (%s) - restarting it", str(e).splitlines()[0])
                async with self._start_lock:
                    await self._shutdown()

    async def _hide_headless(self, page: Page) -> None:
        """Headless Chromium says "HeadlessChrome" in its user agent, which some sites refuse."""
        if self._visible or not self.headless:
            return
        try:
            if self._user_agent is None:
                self._user_agent = (await page.evaluate("navigator.userAgent")).replace(
                    "HeadlessChrome", "Chrome")
            cdp = await page.context.new_cdp_session(page)
            await cdp.send("Network.setUserAgentOverride", {"userAgent": self._user_agent})
        except Exception as e:
            log.debug("could not override user agent: %s", e)

    async def _shutdown(self) -> None:
        """Close the browser and the driver, tolerating either being dead already."""
        ctx, pw = self._ctx, self._pw
        self._ctx, self._pw = None, None
        for closer in ((ctx.close if ctx else None), (pw.stop if pw else None)):
            if closer is None:
                continue
            try:
                await asyncio.wait_for(closer(), 15)
            except Exception as e:
                log.debug("ignoring error while closing the browser: %s", e)

    async def close(self) -> None:
        async with self._start_lock:
            await self._shutdown()


async def install_chromium() -> None:
    """Run Playwright's bundled installer (works inside a frozen executable too)."""
    from playwright._impl._driver import compute_driver_executable, get_driver_env

    driver, cli = compute_driver_executable()
    log.warning("No Edge/Chrome found - downloading Chromium (one time)…")
    proc = await asyncio.create_subprocess_exec(
        driver, cli, "install", "chromium", env={**os.environ, **get_driver_env()})
    await proc.wait()
