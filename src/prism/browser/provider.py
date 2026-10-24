"""Provider that talks to a platform through its web app instead of its API."""

from ..providers.base import Message, Provider
from .adapter import SiteAdapter
from .manager import BrowserManager


class BrowserProvider(Provider):
    """A daughter bound to one remote conversation (thread_url), or - with fresh=True -
    a mother that starts a new remote chat for every call.

    The system prompt (which carries the shared context) is sent when a remote chat starts,
    and again when resend_system is set - e.g. context was added since the last turn."""

    def __init__(self, platform: str, manager: BrowserManager, adapter: SiteAdapter,
                 thread_url: str | None = None, fresh: bool = False, resend_system: bool = False):
        self.name = platform
        self.model = f"{platform} (web)"
        self.manager = manager
        self.adapter = adapter
        self.thread_url = thread_url
        self.fresh = fresh
        self.resend_system = resend_system
        self.notes: list[str] = []
        self.last_html: str | None = None

    async def complete(self, messages: list[Message], system: str | None = None) -> str:
        return await self.complete_with_files(messages, system)

    async def complete_with_files(self, messages: list[Message], system: str | None = None,
                                  files: list[str] = ()) -> str:
        # The remote chat already holds the history - send only the new message.
        text = messages[-1]["content"]
        new_chat = self.fresh or not self.thread_url
        if system and (new_chat or self.resend_system):
            text = f"{system}\n\n---\n\n{text}"

        async with self.manager.lock(self.name):
            page = await self.manager.new_page()  # a tab in the shared headless browser
            try:
                await self.adapter.open(page, None if self.fresh else self.thread_url)
                reply = await self.adapter.send(page, text, files=files)
                self.notes = list(self.adapter.notes)  # the site's layout changed: what PRISM did
                self.last_html = self.adapter.last_html  # the reply as the site showed it
                if not self.fresh:
                    self.thread_url = page.url  # binds the daughter to the remote conversation
                self.resend_system = False  # the remote chat has it now
                return reply
            finally:
                await page.close()
