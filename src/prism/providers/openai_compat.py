"""OpenAI Chat Completions API - also used for xAI Grok and Ollama (OpenAI-compatible endpoints)."""

from openai import AsyncOpenAI

from .base import Message, Provider


class OpenAICompatProvider(Provider):
    def __init__(self, name: str, model: str, api_key: str, base_url: str | None = None):
        self.name = name
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(self, messages: list[Message], system: str | None = None) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        resp = await self._client.chat.completions.create(model=self.model, messages=msgs)
        return resp.choices[0].message.content or ""
