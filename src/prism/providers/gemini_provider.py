"""Google Gemini via the google-genai SDK."""

from google import genai
from google.genai import types

from .base import Message, Provider


class GeminiProvider(Provider):
    def __init__(self, api_key: str, model: str):
        self.name = "gemini"
        self.model = model
        self._client = genai.Client(api_key=api_key)

    async def complete(self, messages: list[Message], system: str | None = None) -> str:
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
        ]
        config = types.GenerateContentConfig(system_instruction=system) if system else None
        resp = await self._client.aio.models.generate_content(
            model=self.model, contents=contents, config=config
        )
        return resp.text or ""
