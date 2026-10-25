"""PRISM's group chat: the user, the bots and a moderator in one room.

The room is one shared transcript. Every member (each bot, and the moderator) keeps a
bookmark: when it is asked to speak it is sent everything new since its bookmark - what the
user, the other bots and the moderator said - plus its instruction. So "@claude answer" and
then "@gpt check that" just works: gpt's turn carries claude's answer.

A session runs a script: rounds of steps, each step a set of turns that run in parallel.

  chat         the addressed bots answer once (@claude, @claude @gpt, @all)
  roundtable   every bot speaks each round, in parallel; the moderator analyses each round
  round_robin  bots speak one after another, each seeing the previous; analysis each round
  review       the lead answers, the others review it, the lead revises; repeat
  summary      @moderator: the moderator sums up the room as it stands
  plan         the user assigns tasks, one step per line ("@gpt write the script" then
               "@claude review it"); steps run in order, bots named on one line in parallel

A bot can ask another platform for what it remembers of the user (its saved memory and
past chats) by writing "ASK @gpt: <question>" on a line: the question goes to a new chat
there, the answer comes back to the bot, which then finishes its turn. The exchange also
joins the room, so everyone sees what was looked up.

Messages the user sends while a session runs join the room at once, so every bot sees them
on its next turn. Everything is reported through `emit` as it happens.
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from .providers import Message, Provider, wait_reply
from .runner import RUN_HELP, SHOWN_PACKAGES, Request, parse_request, result_for_bot

MAX_RUNS = 5  # programs a bot may run per turn (each one approved by the user)

# How long each bot may talk per turn (words; 0 = no limit). A prompt instruction - web apps
# can't enforce it.
BUDGETS = {"short": 80, "medium": 200, "long": 400, "full": 0}
MAX_ROUNDS = 8
FORMATS = ("chat", "roundtable", "round_robin", "review")
MODERATOR = "moderator"
DELTA_CHARS = 60_000  # a member away for long gets the newest part of the room in full

MEMBER_SYSTEM = (
    "You are {name}, a member of a group chat with a user, other AI models ({others}) and a "
    "moderator. Messages from the room reach you labelled by who said them. Build on good "
    "points and challenge weak or wrong ones by name. Be concrete. Answer in plain text in this "
    "chat: don't create documents or use connected apps (email, Drive, Docs) unless the user "
    "asks for that. When you make a file to look at or use (a web page, a component, an SVG, "
    "a script), put the complete file in one fenced code block in your reply (```html, ```svg, "
    "```jsx ...) - not in an artifact, canvas or side panel, which the room can't see; PRISM "
    "shows it as a live preview (for React: one component with a default export, using only "
    "React itself and Tailwind classes - no other packages load there).{deep}"
)
MODERATOR_SYSTEM = (
    "You moderate a group chat between a user and AI models. You see the room labelled by who "
    "said what. When asked for an analysis, use exactly this structure:\n"
    "## Agreement\n## Disagreement\n## Questions for you\n(what the user could answer to move "
    "things forward; write 'None' if nothing)\n## Best answer so far\n"
    "and finish with one line: STATUS: converged | open. Be specific about which model said what. "
    "Answer in plain text in this chat: don't create documents or use connected apps."
)
ASK_HELP = (" You can ask the other assistants the user uses ({askable}): each knows the user "
            "from its own saved memory and past chats, and can also give advice. Write one line "
            "`ASK @name: <question>` - or `ASK @any: <question>` to ask all of them at once when "
            "you don't know who has it - and nothing after it; the answers come back to you and "
            "you then finish your turn. The one you ask gets a new chat with only your question - "
            "nothing from this room - so put everything it needs in the question (to have it "
            "build or write something, paste the full brief). When the user mentions their own "
            "things you don't have (a routine, a plan, past decisions, personal history), ASK "
            "before guessing. At most {max_asks} asks per turn.{prism}")
ASK_TOOLS_ONLY = (" You can search the user's own material: write one line `ASK @name: <keywords>` "
                  "and nothing after it; the results come back to you and you then finish your "
                  "turn. At most {max_asks} asks per turn.{tools}")
ASK_PRISM = (" `prism` is not an assistant: `ASK @prism: <keywords>` searches the user's past "
             "chats in this app and returns the matching messages.")
ASK_DOCS = (" `docs` is not an assistant: `ASK @docs: <keywords>` searches the user's files and "
            "web pages given to this chat (#folders) and returns the matching passages with file "
            "and page - use it to check facts and find more than the passages you were given.")
LOCAL_TOOLS = {"prism": ASK_PRISM, "docs": ASK_DOCS}  # searched by PRISM itself, not assistants
ASK_LINE = re.compile(r"^[\s*_`>-]*ASK\s+@([A-Za-z][\w-]*)[*_`]*\s*:[\s*_`]*(.+?)[\s*_`]*$", re.I)
MAX_ASKS = 3  # default advice budget per turn; the user sets 0-5
DEEP = (" Take your time: reason step by step, check your own claims, and prefer depth over "
        "speed.")

ROLES = {
    "answer": "Answer the user's latest message.",
    "discuss": "Respond to the others: where do you agree, what is wrong and why, and what is "
               "your refined answer now?",
    "review": "Review {lead}'s latest answer: what is right, what is wrong or missing, and how "
              "to fix it. Be specific.",
    "revise": "Revise your answer using the reviews. Give the complete improved answer.",
    "task": "Your task: {task}\nDo your part fully, building on what the others produced; "
            "the rest of the plan is theirs.",
}
ANALYSIS = "Write the current analysis of the discussion."
FINAL = ("The discussion is over. Write the final answer for the user: the single best answer, "
         "using what the room established and fixing what it found wrong. Reply with the answer "
         "only - no analysis headings, no mention of rounds or STATUS.")
SUMMARY = ("Sum up the room for the user as it stands, using the analysis structure: what the "
           "models agree and disagree on, questions for the user, and the best answer so far.")


# --- the room ------------------------------------------------------------------------------

@dataclass
class Entry:
    who: str            # "user", a platform, "moderator", or "context"
    kind: str           # topic | user | context | speaker | analysis | final | summary
    text: str
    id: int | None = None  # the stored message's id, once persisted
    files: list[str] = field(default_factory=list)  # paths of files attached to it


@dataclass
class Member:
    name: str
    provider: Provider
    words: int = BUDGETS["full"]
    history: list[Message] = field(default_factory=list)  # its own thread so far
    seen: int = 0  # room entries [:seen] have been sent to it
    role: str = ""  # a saved agent's instructions: what this member is for


def render(entries: list[Entry], limit: int = DELTA_CHARS) -> str:
    """The room as a member reads it; when too long, the newest part is kept in full."""
    def label(e: Entry) -> str:
        if e.who == "user":
            return "the user"
        if e.who == "context":
            return "shared context the user attached"
        return f"moderator - {e.kind}" if e.who == MODERATOR else e.who
    parts, used = [], 0
    for i in range(len(entries) - 1, -1, -1):
        part = f"[{label(entries[i])}]\n{entries[i].text}"
        if parts and used + len(part) > limit:
            parts.append(f"(... {i + 1} earlier messages omitted)")
            break
        parts.append(part)
        used += len(part)
    return "\n\n".join(reversed(parts))


# --- scripts: what a format does each round -----------------------------------------------

@dataclass
class Turn:
    speaker: str
    role: str  # answer | discuss | review | revise | task
    task: str = ""  # role "task": what the user assigned


@dataclass
class Script:
    format: str
    rounds: int
    steps: Callable[[int], list[list[Turn]]]  # round number -> steps (each runs in parallel)
    moderated: bool = False                   # moderator analyses each round, writes the final
    closing: list[Turn] = field(default_factory=list)  # last step instead of a moderator final
    extendable: bool = True                   # a user message in the last round adds a round
    lead: str = ""


def make_script(fmt: str, speakers: list[str], rounds: int,
                plan: list[list[tuple[str, str]]] | None = None) -> Script:
    """speakers in order: mention order, else the configured order. review: the first leads.
    plan: the steps the user assigned, each a list of (bot, task) run in parallel."""
    rounds = max(1, min(MAX_ROUNDS, rounds))
    if fmt == "plan" and plan:
        return Script("plan", 1, lambda r: [[Turn(s, "task", task) for s, task in step]
                                            for step in plan], extendable=False)
    if fmt == "summary":
        return Script("summary", 0, lambda r: [], moderated=True, extendable=False)
    if fmt == "review" and len(speakers) >= 2:
        lead, reviewers = speakers[0], speakers[1:]
        return Script("review", rounds,
                      lambda r: [[Turn(lead, "answer" if r == 1 else "revise")],
                                 [Turn(x, "review") for x in reviewers]],
                      closing=[Turn(lead, "revise")], lead=lead)
    many = len(speakers) > 1
    if fmt == "roundtable" and many:
        return Script(fmt, rounds, lambda r: [[Turn(s, "answer" if r == 1 else "discuss")
                                               for s in speakers]], moderated=True)
    if fmt == "round_robin" and many:
        return Script(fmt, rounds,
                      lambda r: [[Turn(s, "answer" if r == 1 and i == 0 else "discuss")]
                                 for i, s in enumerate(speakers)], moderated=True)
    return Script("chat", 1, lambda r: [[Turn(s, "answer") for s in speakers]])


# --- a session -----------------------------------------------------------------------------

@dataclass
class Statement:
    speaker: str
    role: str
    round: int
    step: int
    prompt: str
    text: str | None = None
    error: str | None = None
    latency_s: float = 0.0


# emit(kind, **data) -> the stored message id for "entry", else None. Kinds:
#   entry(entry, meta[, statement])  persist a room message (failed turns too, with meta.error)
#   seen(member, entry)              member has now been sent everything up to entry
#   status(status)
Emit = Callable[..., int | None]
# ask(platform, question) -> what that platform's own memory of the user answers
Ask = Callable[[str, str], Awaitable[str]]
# run(member, request) -> the request, run (or refused) after the user's approval
RunFn = Callable[[str, "Request"], Awaitable["Request"]]


class Session:
    def __init__(self, room: list[Entry], members: dict[str, Member], script: Script, emit: Emit,
                 context: str = "", timeout: float = 300.0, deep: bool = False,
                 ask: Ask | None = None, askable: list[str] = (), max_asks: int = MAX_ASKS,
                 max_s: float = 4 * 3600.0, run: "RunFn | None" = None, runnable: list[str] = ()):
        if script.moderated and MODERATOR not in members:
            raise ValueError("a moderated format needs the moderator")
        self.room, self.members, self.script, self.emit = room, members, script, emit
        # timeout: give up on a turn after this long with no sign the model is working;
        # max_s: in all. A model that keeps working (long research) is not cut off.
        self.context, self.timeout, self.deep, self.max_s = context, timeout, deep, max_s
        # run(member, request) -> the finished request: the app asks the user, then runs it
        self.run_fn, self.runnable, self.approving = run, set(runnable), set()
        self.python = ""  # the version RUN @python runs on here ("" = can't run; set by the app)
        self.ask, self.askable, self.max_asks = ask, list(askable), max_asks
        self.rounds = script.rounds
        self.round = 0
        self.phase = "starting"  # starting | speaking | analysing | final | done
        self.waiting_on: set[str] = set()
        self._stopped = asyncio.Event()
        self._step_task: asyncio.Future | None = None
        self._step_no = 0

    # --- control from outside (the UI) ---------------------------------------------------

    def add_user_message(self, text: str, attached: list[Entry] = (),
                         files: list[str] = ()) -> int | None:
        """The user speaks while the session runs: into the room at once, after any #context
        it attached. In the last round it adds one more, so the message is always answered.
        Returns the stored id."""
        for e in attached:
            self.room.append(e)
        entry = Entry("user", "user", text, files=list(files))
        meta = {"kind": "user", "round": self.round, "during": True}
        if files:
            meta["files"] = [Path(f).name for f in files]
        entry.id = self.emit("entry", entry=entry, meta=meta)
        self.room.append(entry)
        if (self.script.extendable and self.phase in ("speaking", "analysing")
                and self.round >= self.rounds):
            self.rounds = min(MAX_ROUNDS, self.rounds + 1)
        return entry.id

    def stop(self) -> None:
        """End after the current step: bots still talking are cut off, then the last step
        (the moderator's final answer, or the lead's revision) still runs."""
        self._stopped.set()
        if self._step_task and not self._step_task.done():
            self._step_task.cancel()

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    @property
    def accepts_messages(self) -> bool:
        return self.phase not in ("final", "done")

    def status(self) -> dict:
        return {"running": self.phase != "done", "format": self.script.format,
                "phase": self.phase, "round": self.round, "rounds": self.rounds,
                "waiting_on": sorted(self.waiting_on), "stopping": self.stopped,
                "approving": sorted(self.approving)}

    # --- running -------------------------------------------------------------------------

    async def run(self) -> None:
        try:
            while self.round < self.rounds and not self.stopped:
                self.round += 1
                answered = False
                for i, step in enumerate(self.script.steps(self.round), start=1):
                    if self.stopped:
                        break
                    said = any(st.text for st in await self._step(step, i))
                    answered |= said
                    if not said and self.script.format == "plan":
                        break  # the next step builds on this one: nothing to build on
                if self.stopped or not answered:
                    break
                if self.script.moderated:
                    await self._moderate("analysis", ANALYSIS)
            if self.script.closing:
                await self._step(self.script.closing, 0, final=True)
            elif self.script.moderated:
                await self._moderate(*(("summary", SUMMARY) if self.script.format == "summary"
                                       else ("final", FINAL)))
        finally:
            self.phase, self.waiting_on = "done", set()
            self._status()

    async def _step(self, turns: list[Turn], step: int, final: bool = False) -> list[Statement]:
        """Turns in one step run in parallel and see the room as it was when the step began."""
        self.phase = "final" if final else "speaking"
        self._step_no = step
        snapshot = len(self.room)
        prompts = {t.speaker: self._prompt(t, snapshot, final) for t in turns}
        # Files attached to what each bot has not seen yet go with its turn.
        files = {t.speaker: [f for e in self.room[self.members[t.speaker].seen:snapshot]
                             if e.who != t.speaker for f in e.files] for t in turns}
        self.waiting_on = {t.speaker for t in turns}
        self._status()
        done: list[Statement] = []

        async def one(t: Turn) -> None:
            m = self.members[t.speaker]
            st = Statement(t.speaker, t.role, self.round, step, prompts[t.speaker])
            start = time.perf_counter()
            try:
                msgs = m.history + [{"role": "user", "content": st.prompt}]
                first = (m.provider.complete_with_files(list(msgs), self._system(m), files[t.speaker])
                         if files[t.speaker] else m.provider.complete(list(msgs), self._system(m)))
                st.text = await wait_reply(first, m.provider, self.timeout, self.max_s)
                msgs.append({"role": "assistant", "content": st.text})
                asks = runs = 0
                while True:  # ASKs and RUNs at the end of the reply: answer them, let it go on
                    if (q := _ask_request(st.text)) and (targets := self._targets(m, q.group(1))):
                        if asks >= self.max_asks:  # budget spent: it answers with what it has
                            answer = ("[No asks left this turn.] Finish your answer now with what "
                                      "you have, and say what is still unknown.")
                        else:
                            asks += 1
                            answer = await self._lookup(m.name, targets, q.group(2).strip())
                    elif self.run_fn and m.name in self.runnable and (req := parse_request(st.text, m.name)):
                        if runs >= MAX_RUNS:
                            answer = "[No runs left this turn.] Finish your answer with what you have."
                        else:
                            runs += 1
                            answer = await self._run(m.name, req)
                    else:
                        break
                    msgs.append({"role": "user", "content": answer})
                    st.text = await wait_reply(m.provider.complete(list(msgs), self._system(m)),
                                               m.provider, self.timeout, self.max_s)
                    msgs.append({"role": "assistant", "content": st.text})
                    if answer.startswith(("[No asks left", "[No runs left")):
                        break
                m.history = msgs
                # Only now: a bot whose turn failed gets the same messages again next time.
                self._mark_seen(m, snapshot)
            except asyncio.CancelledError:
                st.error = "stopped"
                raise
            except Exception as e:  # one bot failing must not end the session
                st.error = _err(e)
            finally:
                st.latency_s = time.perf_counter() - start
                self.waiting_on.discard(t.speaker)
                done.append(st)
                self._record(st, final)
                self._status()

        task = asyncio.ensure_future(asyncio.gather(*(one(t) for t in turns)))
        if not final:  # Stop cuts a step short, but never the final one
            self._step_task = task
        try:
            await task
        except asyncio.CancelledError:
            if not self.stopped:  # cancelled from outside (shutdown), not by Stop
                raise
        return done

    def _record(self, st: Statement, final: bool) -> None:
        """A finished turn becomes a room entry (when it said something) and is persisted."""
        kind = "final" if final else "speaker"
        meta = {"kind": kind, "speaker": st.speaker,
                "model": self.members[st.speaker].provider.model, "role": st.role,
                "round": st.round, "step": st.step, "latency_s": round(st.latency_s, 1),
                "error": st.error, "format": self.script.format}
        provider = self.members[st.speaker].provider
        if notes := getattr(provider, "notes", None):
            meta["notes"] = list(notes)
        if st.text and (html := getattr(provider, "last_html", None)):
            meta["html"] = html  # shown the way the site formatted it
        if st.text and (docs := getattr(provider, "last_documents", None)):
            meta["documents"] = docs  # opened in the page's document view
        entry = Entry(st.speaker, kind, st.text or "")
        entry.id = self.emit("entry", entry=entry, meta=meta, statement=st)
        if st.text:
            # Not marked as seen: peers may have spoken since this member's turn began, and
            # its own words are left out of what it is sent anyway.
            self.room.append(entry)

    def _prompt(self, t: Turn, snapshot: int, final: bool) -> str:
        m = self.members[t.speaker]
        new = [e for e in self.room[m.seen:snapshot] if e.who != m.name]
        where = ("final turn" if final else
                 f"round {self.round} of {self.rounds}" if self.rounds > 1 else "your turn")
        if self.script.format == "plan":
            where = f"step {self._step_no} of {len(self.script.steps(1))}"
        limit = f" (Keep it under about {m.words} words.)" if m.words else ""
        return (f"{render(new) if new else '(nothing new in the room since your last turn)'}\n\n"
                f"--- YOUR TURN ({where}) ---\n"
                f"{ROLES[t.role].format(lead=self.script.lead, task=t.task)}{limit}")

    def _askable(self, m: Member) -> list[str]:
        return [p for p in self.askable if p != m.name] if self.ask else []

    def _targets(self, m: Member, name: str) -> list[str]:
        """Who an ASK goes to: one platform, or with @any / @all every other one."""
        askable = self._askable(m)
        name = name.lower()
        return askable if name in ("any", "all") else [name] if name in askable else []

    async def _run(self, who: str, req: Request) -> str:
        """A RUN: waits for the user's approval (the card is in the room), then runs it."""
        self.approving.add(who)
        self._status()
        try:
            done = await self.run_fn(who, req)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return f"[the run could not start: {_err(e)}] Finish your reply without it."
        finally:
            self.approving.discard(who)
            self._status()
        return result_for_bot(done)

    async def _lookup(self, asker: str, targets: list[str], question: str) -> str:
        """One ASK, to one platform or several at once; each exchange joins the room."""
        async def one(target: str) -> str:
            try:
                answer = await wait_reply(self.ask(target, question), None, self.timeout, self.max_s)
                error = None
            except Exception as e:
                answer, error = "", _err(e)
            entry = Entry(target, "lookup", f"{asker} asked: {question}\n\n{answer}" if answer else "")
            entry.id = self.emit("entry", entry=entry, meta={
                "kind": "lookup", "speaker": target, "asker": asker, "question": question,
                "round": self.round, "error": error, "format": self.script.format})
            if answer:
                self.room.append(entry)
            return (f"[{target} answered]\n{answer}" if answer else
                    f"[{target} could not be reached: {error}]")
        added = set(targets) - self.waiting_on  # some may be speaking in this step too
        self.waiting_on |= added
        self._status()
        try:
            answers = await asyncio.gather(*(one(t) for t in targets))
        finally:
            self.waiting_on -= added
        return "\n\n".join(answers) + "\n\nNow finish your turn (you may ASK again if you must)."

    def _system(self, m: Member) -> str:
        others = ", ".join(n for n in self.members if n not in (m.name, MODERATOR)) or "none"
        askable = self._askable(m)
        assistants = [a for a in askable if a not in LOCAL_TOOLS]
        tools = "".join(LOCAL_TOOLS[a] for a in askable if a in LOCAL_TOOLS)
        help_ = ("" if not self.max_asks else
                 ASK_HELP.format(askable=", ".join(assistants), max_asks=self.max_asks,
                                 prism=tools) if assistants else
                 ASK_TOOLS_ONLY.format(max_asks=self.max_asks, tools=tools) if askable else "")
        role = f" Your role here, set by the user: {m.role}" if m.role else ""
        if self.run_fn and m.name in self.runnable and self.python:
            help_ += RUN_HELP.format(version=self.python, packages=SHOWN_PACKAGES, max_runs=MAX_RUNS)
        return _with_context(MEMBER_SYSTEM.format(name=m.name, others=others,
                                                  deep=role + (DEEP if self.deep else "") + help_),
                             self.context)

    async def _moderate(self, kind: str, instruction: str) -> None:
        self.phase = "analysing" if kind == "analysis" else "final"
        self.waiting_on = {MODERATOR}
        self._status()
        m = self.members[MODERATOR]
        snapshot = len(self.room)
        new = [e for e in self.room[m.seen:snapshot] if e.who != MODERATOR]
        prompt = f"{render(new) if new else '(nothing new in the room)'}\n\n--- {instruction}"
        meta = {"kind": kind, "speaker": MODERATOR, "model": m.provider.model,
                "round": self.round, "format": self.script.format}
        try:
            msgs = m.history + [{"role": "user", "content": prompt}]
            system = _with_context(MODERATOR_SYSTEM, self.context)
            text = await wait_reply(m.provider.complete(msgs, system), m.provider, self.timeout, self.max_s)
            m.history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": text}]
            self._mark_seen(m, snapshot)
            if html := getattr(m.provider, "last_html", None):
                meta["html"] = html
            if docs := getattr(m.provider, "last_documents", None):
                meta["documents"] = docs
            entry = Entry(MODERATOR, kind, text)
            entry.id = self.emit("entry", entry=entry, meta=meta, prompt=prompt)
            self.room.append(entry)
        except Exception as e:  # the session goes on without it
            self.emit("entry", entry=Entry(MODERATOR, kind, ""), meta={**meta, "error": _err(e)})
        finally:
            self.waiting_on = set()

    def _mark_seen(self, m: Member, upto: int) -> None:
        if upto > m.seen:
            m.seen = upto
            self.emit("seen", member=m.name, entry=self.room[upto - 1])

    def _status(self) -> None:
        self.emit("status", status=self.status())


# --- helpers -------------------------------------------------------------------------------

def context_block(items: list[dict]) -> str:
    if not items:
        return ""
    body = "\n\n".join(f"### {i['title']}\n{i['content']}" for i in items)
    return f"SHARED CONTEXT (from the PRISM workspace - use it where relevant):\n{body}"


def _with_context(system: str, context: str) -> str:
    return f"{system}\n\n{context}" if context else system


_SPLIT_BIT = re.compile(r"^\s*#[0-9A-Fa-f]{3,8}\b")


def _ask_request(text: str) -> re.Match | None:
    """An ASK counts only at the end of the reply - not when a bot quotes or narrates one.
    The question may run over a few short lines (web apps break lines inside it, e.g.
    claude.ai shows colour codes as chips on their own line)."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    for i in range(len(lines) - 1, max(-1, len(lines) - 6), -1):
        if ASK_LINE.match(lines[i]):
            tail = lines[i + 1:]
            # Lines the app broke out of the question, not narration after it: they start
            # with a colour code, or the question runs on and ends with its question mark.
            broken = all(_SPLIT_BIT.match(x) for x in tail) or (
                len(tail) <= 2 and not re.search(r"[.?!:]\W*$", lines[i])
                and tail[-1].rstrip(" *_`").endswith("?"))
            if tail and (not broken or sum(len(x) for x in tail) > 400):
                return None
            return ASK_LINE.match(" ".join([lines[i]] + [x.strip() for x in tail]))
    return None


def _err(e: Exception) -> str:
    """Short enough to show: browser errors carry a call log that repeats the whole message
    that was being typed."""
    text = re.sub(r'(fill|type|insert_text)\((["\']).*?\2\)', r"\1(…)", str(e), flags=re.S)
    text = text.split("\nCall log:")[0].strip() or text
    return f"{type(e).__name__}: {text[:500]}"
