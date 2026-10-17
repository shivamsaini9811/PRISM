"""Claude via the official Anthropic SDK."""

import anthropic

from .base import Message, Provider, ProviderError

# Models that accept server-side refusal fallbacks (fallbacks="default").
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


class AnthropicProvider(Provider):
    def __init__(self, api_key: str, model: str):
        self.name = "claude"
        self.model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self.effort = "medium"  # "high" for a deep meeting

    async def complete(self, messages: list[Message], system: str | None = None) -> str:
        kwargs: dict = {}
        if system:
            kwargs["system"] = system
        if self.model in _FALLBACK_MODELS:
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        resp = await self._client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            output_config={"effort": self.effort},
            messages=messages,
            **kwargs,
        )
        if resp.stop_reason == "refusal":
            raise ProviderError("Claude declined to answer")
        return "".join(b.text for b in resp.content if b.type == "text")
