"""Platform catalog and provider construction from stored credentials."""

from .base import Message, NoReply, Provider, ProviderError, wait_reply

# kind -> which adapter; needs_secret=False for local runtimes.
PLATFORMS: dict[str, dict] = {
    "claude": {"label": "Claude (Anthropic)", "kind": "anthropic", "default_model": "claude-opus-5-5"},
    "gpt":    {"label": "GPT (OpenAI)", "kind": "openai", "default_model": "gpt-5"},
    "gemini": {"label": "Gemini (Google)", "kind": "gemini", "default_model": "gemini-2.5-pro"},
    "grok":   {"label": "Grok (xAI)", "kind": "openai", "default_model": "grok-4",
               "base_url": "https://api.x.ai/v1"},
    "ollama": {"label": "Ollama (local)", "kind": "openai", "default_model": "llama3.1",
               "base_url": "http://localhost:11434/v1", "needs_secret": False},
}


def build_provider(platform: str, model: str, secret: str | None,
                   base_url: str | None = None) -> Provider:
    spec = PLATFORMS[platform]
    if spec.get("needs_secret", True) and not secret:
        raise ProviderError(f"{platform}: no API key stored")
    if spec["kind"] == "anthropic":
        from .anthropic_provider import AnthropicProvider
        p: Provider = AnthropicProvider(secret, model)
    elif spec["kind"] == "gemini":
        from .gemini_provider import GeminiProvider
        p = GeminiProvider(secret, model)
    else:
        from .openai_compat import OpenAICompatProvider
        p = OpenAICompatProvider(platform, model, secret or "none", base_url or spec.get("base_url"))
    p.name = platform
    return p


__all__ = ["Message", "NoReply", "PLATFORMS", "Provider", "ProviderError", "build_provider", "wait_reply"]
