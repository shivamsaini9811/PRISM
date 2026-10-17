"""Browser transport: drive a platform's own web app (claude.ai, chatgpt.com, ...) via Playwright."""

from .adapter import BotCheck, NotLoggedIn, PageNotReady, SiteAdapter, SiteConfig
from .manager import BrowserManager
from .provider import BrowserProvider
from .sites import SITES

__all__ = ["BotCheck", "BrowserManager", "BrowserProvider", "NotLoggedIn", "PageNotReady", "SITES", "SiteAdapter", "SiteConfig"]
