"""Common interface every model provider implements."""

import asyncio
import time
from abc import ABC, abstractmethod
import re
from pathlib import Path
from typing import TypedDict


class Message(TypedDict):
    role: str  # "user" | "assistant"
    content: str


class ProviderError(Exception):
    pass


class Provider(ABC):
    name: str
    model: str
    # time.monotonic() of the last sign the model is still working on its reply (set by
    # providers that can tell: web apps, streaming local models). None = no signal.
    last_activity: float | None = None

    @abstractmethod
    async def complete(self, messages: list[Message], system: str | None = None) -> str:
        """Return the model's text reply to the conversation."""

    async def complete_with_files(self, messages: list[Message], system: str | None = None,
                                  files: list[str] = ()) -> str:
        """The same, with files the user attached to the latest message. This default tells the
        model their names only; providers that can upload files override it."""
        names = ", ".join(re.sub(r"^[0-9a-f]{12}-", "", Path(f).name) for f in files)
        note = f"\n\n[The user attached: {names} - you can't see the files through this connection.]"
        return await self.complete(messages[:-1] + [{**messages[-1],
                                                     "content": messages[-1]["content"] + note}], system)


class NoReply(asyncio.TimeoutError):
    pass


async def wait_reply(call, provider: "Provider | None" = None, idle_s: float = 1200.0,
                     max_s: float = 4 * 3600.0):
    """Await a model's reply for as long as it is working on it: give up only after idle_s
    with no sign of activity from the provider (or since the start, for providers that give
    none), or after max_s in all. Long research runs as long as it needs."""
    task = asyncio.ensure_future(call)
    start = time.monotonic()
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=2.0)
            if done:
                return task.result()
            now = time.monotonic()
            last = max(start, getattr(provider, "last_activity", None) or start)
            if now - last > idle_s:
                raise NoReply(f"no sign of activity for {round(idle_s / 60)} min - the reply seems stuck")
            if now - start > max_s:
                raise NoReply(f"still not finished after {round(max_s / 3600, 1)} h")
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
