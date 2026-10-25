"""Common interface every model provider implements."""

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
