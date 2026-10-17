# PRISM

**P**arallel **R**outing, **I**nter-model **S**coring & **M**erging

A standalone desktop app that wraps several AI platforms (Claude, ChatGPT, Gemini, Grok,
local models) behind one chat. Run the executable, it opens its UI in your browser; your
chats, context and sign-ins are kept between runs.

## How it works - a group chat with your models

Every PRISM chat is a **room**: you, your AI models (the daughters) and a **moderator** (the
primary LLM you pick). Each model keeps its own thread; web-connected ones are bound to a
*real* conversation on that site (e.g. a claude.ai chat URL). Every member has a bookmark, and
when it is asked to speak it is sent everything said in the room since it last spoke, so
models read and answer each other.

**Who answers - @mentions**

| You write | What happens |
|---|---|
| `@claude explain X` | only Claude answers |
| `@gpt is that right?` | GPT answers, having seen Claude's answer |
| `@claude @gpt ...` | both answer, run in the chosen format |
| `@all ...` or no mention | every model switched on answers, in the chosen format |
| `@moderator where are we?` | the moderator sums up the room |

**How they talk - the format** (for several models; one model is always a plain answer)

- **Round table** - everyone answers at once; after each round the moderator posts the
  analysis (agreement, disagreement, **questions for you**, best answer so far); next round
  everyone responds to the others. Final answer by the moderator.
- **Round robin** - one after another, each seeing the ones before; analysis each round.
- **Ask & review** - the first @named model answers, the others review it, it revises; the
  revision is the final answer.
- **Chat** - each answers once, side by side.

Also set before sending: **rounds** (1-8), how long each model may talk (Off / Short ~80 words
/ Medium ~200 / Long ~400; an instruction the web apps can't enforce; naming a model
overrides Off) and **Deep** (API Claude thinks at high effort; everyone is told to reason step
by step). While models talk, a message from you joins the discussion (in the last round it
adds one more); a message with an @mention is queued and runs next. **Stop & conclude** cuts
the current step short and still writes the final answer.

**Named context - #names**

| You write | What happens |
|---|---|
| `/save #spec Budget 10k, ship by May` | saves that text as `#spec` (same name again replaces it) |
| `/save #plan` | saves the latest answer as `#plan` |
| `/save-global #style Be terse` | a `#name` every chat can use |
| `@gpt plan it within #spec` | `#spec` is attached to the room for this message |
| `/forget #spec` | deletes it |

Answers also have a **Save as #...** button, and the Context panel adds items by name. Items
marked *always* are sent with every message instead of only when named.

A round takes as long as its slowest model plus the moderator: seconds over APIs, 1-2 minutes
with web platforms.

## Connecting platforms

Each platform connects one of two ways (Settings):

| | API key | Browser |
|---|---|---|
| How | Paste the provider's API key | **Connect** opens the site in PRISM's own browser; sign in once |
| Platforms | Claude, GPT, Gemini, Grok, Ollama | claude.ai, chatgpt.com, gemini.google.com, grok.com |
| Speed | Seconds | Tens of seconds to minutes per answer |
| Reliability | Stable | Breaks when a site changes its layout - see below |

Browser platforms:
- PRISM never sees or stores your web passwords. You type them into the real site (2FA and
  CAPTCHAs work). Every site's session lives in one shared browser profile in the data folder.
- All platforms run as tabs of **one headless browser**, which keeps memory use flat. **Connect**
  reopens that browser with a visible window so you can sign in; **Check** (or the next message)
  switches it back to headless. If the browser dies (for example, the PC ran low on memory),
  PRISM restarts it on the next request. It won't start one with less than 1 GB of RAM free.
- Some sites challenge headless browsers ("Just a moment..."). PRISM reports that as a bot
  check with a screenshot; complete it via **Connect**, or set `PRISM_BROWSER_HEADLESS=false`.
- The page selectors in `src/prism/browser/sites.py` are **unverified best guesses**. After
  signing in, press **Check** to see which selectors match. If one doesn't, fix it in
  **Selectors** without rebuilding.
- These sites' consumer terms generally prohibit automated use. Using API keys is the
  supported path.
- Use an API connection for the primary (mother) if you can. A browser mother opens one extra
  web chat per meeting (its moderator conversation) and is slower.

## Security model

- Provider API keys are encrypted at rest (scrypt-derived key + Fernet). The master password is
  never stored, and you enter it on each launch.
- Chats and context are stored unencrypted in SQLite.
- The server listens on 127.0.0.1 only.
- Data folder: `~/.prism` (`C:\Users\<you>\.prism` on Windows); override with `PRISM_HOME`.
  It holds `prism.db` and `profiles/` (browser sign-ins).

## Running the packaged app

Unzip `PRISM-<version>-windows.zip` and run `PRISM\PRISM.exe`. It prints the URL and opens it.
Nothing else needs to be installed: Python, all libraries and Chromium are inside the folder.
PRISM prefers Edge/Chrome on the machine if present (better for Google sign-in) and falls back
to the bundled Chromium.

## Development

```bash
# On WSL keep the venv on the Linux side (NTFS can't hold venv symlinks).
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/prism
uv sync
uv run prism            # http://127.0.0.1:8765
uv run pytest           # fakes only, no keys or network needed
```

## Building the executable

```bash
python packaging/build.py               # on Windows -> dist/PRISM/PRISM.exe + zip
python packaging/build.py --no-browser  # smaller; relies on Edge/Chrome on the target
```

PyInstaller does not cross-compile, so build on the OS you are targeting.

## Layout

```
src/prism/
  __main__.py      launcher: starts the server, opens the UI
  app.py           FastAPI: vault, credentials, chats, context, browser connect/check
  meeting.py       the room: bookmarks, formats as scripts, sessions, moderator
  db.py            SQLite store + migrations
  crypto.py        master-password vault
  providers/       API adapters (Anthropic, OpenAI-compatible, Gemini)
  browser/         Playwright transport: profiles, generic site driver, site selectors
  static/index.html
packaging/         PyInstaller build
tests/
```

## Status

v0.5.0 - group chat: @mentions, formats (round table, round robin, ask & review, chat),
moderator analysis/summary, named #context with /save and /forget, live updates, Stop &
conclude. Replaces the one-shot route/judge/merge turn.
v0.3.1 - one shared headless browser for all web platforms, recovery when it dies.
v0.3.0 - mother/daughter pipeline, encrypted vault, context store, API + browser transports,
standalone packaging. Not yet tested against live platforms.
