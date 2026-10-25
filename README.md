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

**Agents and flows - work that runs on its own**

An **agent** (Flows → Agents) is a role on one of your models: instructions, the #context it
always gets, what it may look up (`ASK @docs`, `@prism`, your other assistants) and fields
PRISM reads from its reply (`verdict: PASS|FIX`). `@critic` in any chat makes it answer.

A **flow** is a graph of steps, written in YAML (or drafted from a description by the
primary model) and drawn as you type:

```yaml
name: market-brief
inputs: {topic: text}
nodes:
  draft:  {agent: writer, prompt: "Write a brief on {{input.topic}}. Fix: {{review.notes}}"}
  review: {agent: critic, prompt: "Review {{draft.text}}"}
  ok:     {ask_you: "Publish it?", show: "{{draft.text}}"}
edges:
  - draft -> review
  - review -> draft: {when: "verdict == FIX", max_rounds: 2}
  - review -> ok:    {when: "verdict == PASS or rounds_done"}
```

Steps: `agent`, `tool` (docs, prism, fetch), `ask_you` (pauses with Approve / Reject), `wait`
(e.g. `10m`), `flow` (another flow inside this one); `join: all` waits for every step that
leads in. Every loop needs `max_rounds`; a run also has a step and time budget (`budget:`),
and pauses rather than fails when one runs out. Each run is its own chat: a card per step, a
tab per agent, the graph lit up as it goes; notes you type reach the steps still to come.
State is saved after every step, so a run continues after PRISM restarts.

**Research** (the checkbox under the message box, or "deep research" on an agent): web bots
switch on their site's own deep research - Claude Research, ChatGPT deep research, Gemini Deep
Research, Grok Expert - for the message. PRISM clicks the plan's Start for you, answers
"go ahead" when it asks clarifying questions first, waits through placeholders like "Deep
Research has started…", reloads a page that stops updating (Gemini), and brings back the
full report: from Claude's document panel, ChatGPT's report frame, Gemini's report panel.
Checked against all four sites on 2026-10-25: 1 min (Grok) to 37 min (Gemini).

**Knowledge graph** (Knowledge, in the sidebar): the things in your work - people,
organisations, projects, decisions, documents, places, dates - and how they connect, read out
of your chats ("Read this chat in now", or automatically after each chat), folders ("Add to
knowledge graph" on a folder in Context) and flows (`tool: kg` with `text:` adds, with
`query:` reads). A model lists the things and links as JSON - a local model when you have
one, else the primary (or the one you pick) - and names that mean the same thing are merged.
Every link keeps where it came from (the chat, with a quote; the file; the flow), shown on
the Knowledge screen, where you can search, rename, merge, change, and delete. `#kg` in a
message gives the room what is known about what the message names; bots look things up with
`ASK @kg`.

**Running Python**: a bot (or a flow agent with the *run Python* tool) can end its reply with
`RUN @python` and a code block. The chat shows a card with the exact code and folder: **Run
once**, **Always allow Python in this chat**, **Always allow Python**, or **Don't run**. The
code runs on PRISM's own Python, shipped with the app (CPython 3.12 with numpy, pandas,
scipy, matplotlib, seaborn, scikit-learn, statsmodels, sympy, networkx, duckdb, pyarrow,
openpyxl, xlsxwriter, python-docx, python-pptx, pypdf, pillow, requests, httpx,
beautifulsoup4, lxml, plotly and more - `runtime/PRISM-PACKAGES.txt` lists them), inside a
sandbox (`~/.prism/python/env`, a virtual environment that sees those packages). A first line
`# pip: name1 name2` installs packages into the sandbox - always asked, even where Python is
allowed - so neither PRISM's Python nor the computer's changes; Settings → Running Python
lists them, installs and removes, and **Reset sandbox** empties it. A run uses the chat's
work folder (`~/.prism/work/`), stops after 5 minutes (`RUN_TIMEOUT_S`), gets none of PRISM's
saved keys, and what it prints - plus files it made, downloadable from the card - goes back
to the bot. It is not an operating-system sandbox: approved code has your account's rights.
Flows also have a `tool: run` step with `code:`. Builds include the runtime (~1 GB unpacked;
`build.py --no-runtime` leaves it out).

**Catalog - Python packages and models from your own server, signed by you**

```bash
prism-publish keygen --out ~/prism-keys          # once; keep the .key file off the server
prism-publish new catalog.json --name "My tools"
prism-publish add catalog.json --name mytools --version 1.2.0 --file any=mytools-1.2.0-py3-none-any.whl
prism-publish add catalog.json --name qwen-7b --version q4 --kind model --file any=qwen.gguf --as qwen
prism-publish sign catalog.json --key ~/prism-keys/prism-publisher.key   # bumps version, 90-day expiry
```

Upload `catalog.json`, `catalog.json.sig` and the files next to each other (https). In PRISM,
Settings → Catalog: the address and your public key (`ed25519:…`, printed by keygen; or ship
it in a build as `src/prism/publisher.pub`). PRISM installs nothing without a key, and only
from a catalog whose signature matches, whose version isn't older than the last one it took,
and that hasn't expired; every download must match its SHA-256 and archives can't write
outside their folder. Python packages (`--kind python`, the default: a wheel or source
archive named after its pip package) install into the Python sandbox, where bots' code can
import them; models are handed to Ollama and join the room under their `as` name.

**Local models by address**: Settings → Local models → *A server by address* takes any
OpenAI-compatible server (llama.cpp's `llama-server`, vLLM, LocalAI, Jan, a GPU box) with an
optional key, lists its models and adds them to the room.

**No fixed time limit on a reply**: PRISM waits while the model shows it is working - the
site's Stop button, progress or text still changing, a local model still streaming - so long
research is not cut off. It gives up after 20 minutes with nothing changing on the page
(`REQUEST_TIMEOUT_S`), or 4 hours in all (`REPLY_MAX_S`).

**Folders and web pages - your documents as context**

In the Context panel, **Add a folder or web pages** gives a folder (or a list of URLs) a
`#name`. PRISM reads it in the background - PDF, Word, PowerPoint, Excel, OpenDocument, EPUB,
text, Markdown, code, CSV/JSON, HTML, notebooks, email (`.eml`, `.mbox`) and zip exports
(Notion, Takeout) - and checks for changes every few minutes while it is used. Each message
that names it (or every message, if marked *always*) gets the passages that best match, with
file and page; the best-matching PDFs and images are also uploaded to web-app bots (once per
chat). Bots search further with `ASK @docs: <keywords>`.

Images, scanned PDFs, audio and video are found by name; with a **reader** (Settings →
Readers) PRISM also reads them: a vision model copies out the text in images and scans and
says what they show (Ollama, LM Studio, or an API key), and recordings are transcribed
(whisper.cpp + ffmpeg - one-click download on Windows, `brew install whisper-cpp ffmpeg` on a
Mac - or the OpenAI / Gemini API). **Meaning search** with an embedding model from Ollama or
LM Studio (e.g. `nomic-embed-text`) finds passages worded differently from the question;
without one, search is by words (BM25), which needs nothing installed.

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
  flows.py         agents in a graph: parsing, checks, conditions, runs with saved state
  library.py       folders / web pages: scanning, passages, search (#name, ASK @docs)
  readers.py       files of every kind to text
  enrich.py        reader models: images and scans, transcription, embeddings
  local.py         Ollama / LM Studio / OpenAI-compatible servers / in-browser models, .gguf import
  runner.py        RUN @python: approval, the Python sandbox and pip, work folders
  kg.py            the knowledge graph: extraction, merging names, evidence, #kg / ASK @kg
  catalog.py       the signed catalog: verify, install Python packages and models
  publish.py       prism-publish: keys, checksums, signing
  db.py            SQLite store + migrations
  crypto.py        master-password vault
  providers/       API adapters (Anthropic, OpenAI-compatible, Gemini)
  browser/         Playwright transport: profiles, generic site driver, site selectors
  static/index.html
packaging/         PyInstaller build; runtime.py = PRISM's own Python and its packages
tests/
```

## Status

v0.5.0 - group chat: @mentions, formats (round table, round robin, ask & review, chat),
moderator analysis/summary, named #context with /save and /forget, live updates, Stop &
conclude. Replaces the one-shot route/judge/merge turn.
v0.3.1 - one shared headless browser for all web platforms, recovery when it dies.
v0.3.0 - mother/daughter pipeline, encrypted vault, context store, API + browser transports,
standalone packaging. Not yet tested against live platforms.
