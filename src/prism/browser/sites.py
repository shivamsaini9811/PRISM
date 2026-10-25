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
        # Seen 2026-10-25: + menu (chat-input-attach) -> Research switch (menuitemcheckbox).
        research_mode='[data-testid="chat-input-attach"] ;; [data-testid="add-menu-research"]',
        research_pending=r"sources and counting",  # the card says "Research complete" when done
        research_done=r"Research complete",
        # Reports and artifacts: a card in the reply opens an "Artifact panel" region.
        document_card="[data-sheet-kind]",
        document_panel='[role="region"][aria-label^="Artifact panel"]',
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
        # Seen 2026-10-25: + menu -> Deep research; it shows a plan with Start (and starts on
        # its own after a countdown), then works server-side with nothing on the page.
        research_mode='button[aria-label="Add files and more"] ;; button:has-text("Deep research")',
        continue_button='button:text-matches("^\\s*Start\\b")',
        research_pending=r"Deep Research has started|findings will appear in the Deep Research report",
        # The report is an app frame titled "Deep research" (a frame inside it holds the text).
        research_done=r"Research completed in",
        document_frame='iframe[title="Deep research"]',
    ),
    "gemini": SiteConfig(
        new_chat_url="https://gemini.google.com/app",
        composer='rich-textarea div[contenteditable="true"]',
        send_button='button[aria-label="Send message"]',
        assistant_message="message-content",
        streaming='button[aria-label="Stop response"]',
        # Seen 2026-10-25: Upload & tools -> More tools -> Deep research (menuitemcheckbox);
        # it writes a plan and waits for "Start research"; no Stop button meanwhile.
        research_mode=('button[aria-label="Upload & tools"] ;; button[aria-label="More tools"]'
                       ' ;; [role="menuitemcheckbox"]:has-text("Deep research")'),
        continue_button='button:has-text("Start research")',
        research_pending=r"Generating research plan|Researching [0-9]+ websites",
        # Then "I've completed your research" and the report opens in a side panel (its
        # text is the newest message-content, so it is read as the reply).
        research_done=r"I've completed your research",
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
        # Seen 2026-10-25: no DeepSearch any more; the model menu's Expert searches deepest.
        research_mode='button[aria-label^="Model:"] ;; [role="menuitemradio"]:has-text("Expert")',
    ),
}
