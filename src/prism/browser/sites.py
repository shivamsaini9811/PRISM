"""Selector configs for each platform's web app.

UNVERIFIED: these selectors are best guesses and break whenever a site changes its UI.
Use Settings -> Check (the /doctor endpoint) to see which ones match, and override them
from the UI without touching code.
"""

from .adapter import SiteConfig

SITES: dict[str, SiteConfig] = {
    "claude": SiteConfig(
        new_chat_url="https://claude.ai/new",
        composer='[data-testid="chat-input"], div[contenteditable="true"]',
        send_button='button[data-testid="chat-input-send"], button[aria-label="Send message"]',
        assistant_message='div[data-testid="assistant-message"]',
        # Seen on claude.ai 2026-10-16: no data-is-streaming while writing, "false" when done.
        streaming='div[data-testid="assistant-message"]:not([data-is-streaming="false"])',
    ),
    "gpt": SiteConfig(
        new_chat_url="https://chatgpt.com/",
        composer='div[contenteditable="true"][aria-label="Ask ChatGPT"], #prompt-textarea, textarea[name="prompt"]',
        send_button='button[data-testid="send-button"], button[aria-label="Send message"]',
        # Seen on chatgpt.com 2026-10-16 (signed in): the older layout used the first ones.
        assistant_message='div[data-message-author-role="assistant"], [data-dil-source-message-id]',
        streaming='button[data-testid="stop-button"], button[aria-label*="Stop"]',
        # Signed out, ChatGPT still chats - anonymously, without your memory.
        signed_out='button:has-text("Log in"), a:has-text("Log in")',
    ),
    "gemini": SiteConfig(
        new_chat_url="https://gemini.google.com/app",
        composer='rich-textarea div[contenteditable="true"]',
        send_button='button[aria-label="Send message"]',
        assistant_message="message-content",
        streaming='button[aria-label="Stop response"]',
    ),
    "grok": SiteConfig(
        new_chat_url="https://grok.com/",
        composer='textarea, div[contenteditable="true"]',
        send_button='button[data-testid="chat-submit"], button[type="submit"]',
        # The user's own messages are message-bubbles too.
        assistant_message='div.message-bubble:not([data-testid="user-message"])',
        # Signed out, grok.com takes your message and never answers ("Sign up to continue").
        signed_out='a:has-text("Sign in"), button:has-text("Sign in")',
        streaming='button[aria-label="Stop model response"]',
    ),
}
