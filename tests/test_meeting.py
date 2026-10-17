"""The group-chat engine with fake providers (no network)."""

import asyncio

import pytest

from prism.meeting import MODERATOR, Entry, Member, Session, make_script, render
from prism.providers import Provider


class Fake(Provider):
    """As a bot it answers "<name> r<n>"; as the moderator it writes analyses, finals and
    summaries. gate: an asyncio.Event the bot waits on before answering (holds a step open)."""

    def __init__(self, name, fail=False, gate=None):
        self.name, self.model = name, f"{name}-fake"
        self.fail, self.gate = fail, gate
        self.calls: list[tuple[list, str | None]] = []
        self.turns = 0

    async def complete(self, messages, system=None):
        self.calls.append((messages, system))
        if system and "You moderate a group chat" in system:
            last = messages[-1]["content"]
            if "discussion is over" in last:
                return f"FINAL by {self.name}"
            if "Sum up the room" in last:
                return f"SUMMARY by {self.name}"
            return f"ANALYSIS {sum(1 for m in messages if m['role'] == 'user')}"
        if self.gate:
            await self.gate.wait()
        if self.fail:
            raise RuntimeError("boom")
        self.turns += 1
        return f"{self.name} r{self.turns}"

    def prompt(self, i=-1) -> str:
        return self.calls[i][0][-1]["content"]


class Room:
    """A room with bots, a moderator and a persisted-message log, like the app keeps."""

    def __init__(self, *names):
        self.entries: list[Entry] = []
        self.members = {n: Member(n, Fake(n)) for n in names}
        self.members[MODERATOR] = Member(MODERATOR, Fake("mother"))
        self.log: list[tuple[str, dict]] = []
        self.seen: dict[str, int] = {}
        self.ids = 0

    def bot(self, name) -> Fake:
        return self.members[MODERATOR if name == "mother" else name].provider

    def emit(self, kind, **d):
        if kind == "entry":
            self.ids += 1
            self.log.append((d["entry"].who, d["meta"]))
            return self.ids
        if kind == "seen":
            self.seen[d["member"]] = d["entry"].id
        return None

    def session(self, text, fmt, speakers, rounds=1, **kw) -> Session:
        topic = Entry("user", "topic", text, id=self.emit("entry", entry=Entry("user", "topic", text),
                                                           meta={"kind": "topic"}))
        self.entries.append(topic)
        return Session(self.entries, self.members, make_script(fmt, speakers, rounds), self.emit, **kw)

    async def say(self, text, fmt, speakers, rounds=1, **kw) -> Session:
        s = self.session(text, fmt, speakers, rounds, **kw)
        await s.run()
        return s

    def kinds(self):
        return [(m["kind"], m.get("speaker")) for _, m in self.log if m["kind"] != "topic"]


# --- chat with @mentions ------------------------------------------------------------------

async def test_ask_one_then_ask_another_to_validate():
    r = Room("claude", "gpt")
    await r.say("@claude what is 2+2?", "chat", ["claude"])
    await r.say("@gpt is that right?", "chat", ["gpt"])
    p = r.bot("gpt").prompt()
    assert "[claude]\nclaude r1" in p and "what is 2+2?" in p and "is that right?" in p
    # claude again: sees gpt's check, not its own earlier answer (its thread has that)
    await r.say("@claude reply to gpt", "chat", ["claude"])
    p = r.bot("claude").prompt()
    assert "[gpt]\ngpt r1" in p and "claude r1" not in p and "what is 2+2?" not in p
    assert r.bot("claude").calls[-1][0][1]["content"] == "claude r1"  # passed as its history
    assert r.kinds() == [("speaker", "claude"), ("speaker", "gpt"), ("speaker", "claude")]


async def test_at_all_in_chat_answers_in_parallel_without_a_moderator():
    r = Room("claude", "gpt", "gemini")
    await r.say("@all hi", "chat", ["claude", "gpt", "gemini"])
    assert sorted(k for k in r.kinds()) == [("speaker", "claude"), ("speaker", "gemini"), ("speaker", "gpt")]
    assert "[gpt]" not in r.bot("claude").prompt()  # same step: nobody sees the others yet
    assert r.bot("mother").calls == []


async def test_moderator_summary():
    r = Room("claude", "gpt")
    await r.say("hi", "chat", ["claude", "gpt"])
    await r.say("@moderator where are we?", "summary", [])
    assert r.kinds()[-1] == ("summary", MODERATOR)
    p = r.bot("mother").prompt()
    assert "claude r1" in p and "gpt r1" in p and "Sum up the room" in p


# --- formats ------------------------------------------------------------------------------

async def test_roundtable_rounds_analysis_and_final():
    r = Room("claude", "gpt", "gemini")
    s = await r.say("TOPIC", "roundtable", ["claude", "gpt", "gemini"], rounds=2)
    kinds = [k for k, _ in r.kinds()]
    assert kinds == ["speaker"] * 3 + ["analysis"] + ["speaker"] * 3 + ["analysis", "final"]
    assert r.log[-1][1]["speaker"] == MODERATOR and s.round == 2 and not s.status()["running"]
    p = r.bot("gpt").prompt()  # round 2
    assert "claude r1" in p and "gemini r1" in p and "ANALYSIS 1" in p and "gpt r1" not in p
    assert "round 2 of 2" in p and "Keep it under" not in p  # no length limit by default


async def test_round_robin_each_sees_the_one_before_in_the_same_round():
    r = Room("claude", "gpt", "gemini")
    await r.say("TOPIC", "round_robin", ["claude", "gpt", "gemini"], rounds=1)
    assert [k for k in r.kinds()] == [("speaker", "claude"), ("speaker", "gpt"),
                                      ("speaker", "gemini"), ("analysis", MODERATOR),
                                      ("final", MODERATOR)]
    assert "claude r1" in r.bot("gpt").prompt()
    assert "claude r1" in r.bot("gemini").prompt() and "gpt r1" in r.bot("gemini").prompt()


async def test_review_lead_answers_others_review_lead_revises():
    r = Room("claude", "gpt", "gemini")
    await r.say("@claude @gpt @gemini design it", "review", ["claude", "gpt", "gemini"], rounds=1)
    assert [(k, sp) for k, sp in r.kinds()] == [("speaker", "claude"), ("speaker", "gpt"),
                                                 ("speaker", "gemini"), ("final", "claude")]
    assert "Review claude's latest answer" in r.bot("gpt").prompt()
    revise = r.bot("claude").prompt()
    assert "gpt r1" in revise and "gemini r1" in revise and "Revise your answer" in revise
    assert r.bot("mother").calls == []  # review needs no moderator


def test_formats_fall_back_to_chat_with_one_bot():
    for fmt in ("roundtable", "round_robin", "review"):
        assert make_script(fmt, ["claude"], 3).format == "chat"


# --- the user in the meeting, Stop, failures -----------------------------------------------

async def test_user_message_reaches_every_bot_on_its_next_turn():
    gate = asyncio.Event()
    r = Room("claude", "gpt")
    r.bot("gpt").gate = gate
    s = r.session("TOPIC", "roundtable", ["claude", "gpt"], rounds=2)
    task = asyncio.create_task(s.run())
    await asyncio.sleep(0.01)  # round 1 open, gpt still talking
    s.add_user_message("focus on cost")
    gate.set()
    await task
    assert "focus on cost" not in r.bot("claude").prompt(0)
    assert "[the user]\nfocus on cost" in r.bot("claude").prompt(1)
    assert "focus on cost" in r.bot("mother").prompt(0)  # the analysis sees it too


async def test_user_message_in_the_last_round_adds_a_round():
    gate = asyncio.Event()
    r = Room("claude", "gpt")
    r.bot("claude").gate = gate
    s = r.session("TOPIC", "roundtable", ["claude", "gpt"], rounds=1)
    task = asyncio.create_task(s.run())
    await asyncio.sleep(0.01)
    s.add_user_message("one more thing")
    gate.set()
    await task
    assert s.round == 2 and "one more thing" in r.bot("gpt").prompt(1)


async def test_stop_cuts_the_step_short_and_still_writes_the_final():
    r = Room("claude", "gpt")
    r.bot("gpt").gate = asyncio.Event()  # never set: gpt would talk forever
    s = r.session("TOPIC", "roundtable", ["claude", "gpt"], rounds=3)
    task = asyncio.create_task(s.run())
    await asyncio.sleep(0.01)
    s.stop()
    await task
    by = {m["speaker"]: m for _, m in r.log if m["kind"] == "speaker"}
    assert by["gpt"]["error"] == "stopped" and by["claude"]["error"] is None
    assert r.kinds()[-1] == ("final", MODERATOR)
    assert "claude r1" in r.bot("mother").prompt() and s.round == 1


async def test_a_failing_bot_does_not_end_the_session():
    r = Room("claude", "gpt")
    r.bot("gpt").fail = True
    s = await r.say("TOPIC", "roundtable", ["claude", "gpt"], rounds=2)
    assert all("boom" in m["error"] for _, m in r.log if m.get("speaker") == "gpt")
    assert s.round == 2 and r.kinds()[-1] == ("final", MODERATOR)


async def test_bookmarks_are_reported_for_persistence():
    r = Room("claude", "gpt")
    await r.say("@claude hi", "chat", ["claude"])
    assert r.seen["claude"] == 1  # the topic; its own answer never needs sending to it
    assert "gpt" not in r.seen


async def test_deep_asks_for_careful_reasoning():
    r = Room("claude")
    await r.say("hi", "chat", ["claude"], deep=True)
    assert "step by step" in r.bot("claude").calls[0][1]


def test_long_rooms_keep_the_newest_messages():
    entries = [Entry("gpt", "speaker", f"msg{i} " + "x" * 100) for i in range(50)]
    text = render(entries, limit=1000)
    assert "msg49" in text and "msg0 " not in text and "earlier messages omitted" in text


async def test_plan_runs_assigned_tasks_in_order_and_stops_when_a_step_fails():
    room = Room("gpt", "claude", "grok")
    plan = [[("gpt", "write the script")], [("claude", "review it"), ("grok", "draft posts")]]
    s = Session(room.entries, room.members, make_script("plan", ["gpt", "claude", "grok"], 3, plan),
                room.emit)
    room.entries.append(Entry("user", "topic", "launch video"))
    await s.run()
    assert room.kinds() == [("speaker", "gpt"), ("speaker", "claude"), ("speaker", "grok")]
    sent = room.bot("claude").prompt()
    assert "gpt r1" in sent and "Your task: review it" in sent and "step 2 of 2" in sent
    assert "Keep it under" not in sent  # no length limit by default

    room = Room("gpt", "claude")
    room.members["gpt"].provider.fail = True
    s = Session(room.entries, room.members,
                make_script("plan", ["gpt", "claude"], 1, [[("gpt", "a")], [("claude", "b")]]), room.emit)
    await s.run()
    assert room.kinds() == [("speaker", "gpt")] and not room.bot("claude").calls


async def test_word_budget_is_only_asked_for_when_set():
    room = Room("gpt")
    room.members["gpt"].words = 200
    await room.say("q", "chat", ["gpt"])
    assert "Keep it under about 200 words" in room.bot("gpt").prompt()


class Asker(Fake):
    """Asks gpt's memory once, then answers with what came back."""

    async def complete(self, messages, system=None):
        self.calls.append((messages, system))
        if len(self.calls) == 1:
            return "Let me check.\nASK @gpt: what did I decide about my knee?"
        return "grok used: " + messages[-1]["content"].splitlines()[1]


async def test_a_bot_can_ask_another_platforms_memory():
    room = Room("grok")
    room.members["grok"] = Member("grok", Asker("grok"))
    asked = []

    async def ask(platform, question):
        asked.append((platform, question))
        return "You chose physio, no surgery (March)."

    s = Session(room.entries, room.members, make_script("chat", ["grok"], 1), room.emit,
                ask=ask, askable=["gpt", "grok"])
    room.entries.append(Entry("user", "topic", "plan my runs"))
    await s.run()
    assert asked == [("gpt", "what did I decide about my knee?")]
    assert "ASK @name" in room.bot("grok").calls[0][1] and "(gpt)" in room.bot("grok").calls[0][1]
    assert room.kinds() == [("lookup", "gpt"), ("speaker", "grok")]
    assert room.entries[-1].text == "grok used: You chose physio, no surgery (March)."
    assert "physio" in room.entries[-2].text  # the lookup is in the room for everyone


async def test_ask_is_ignored_without_a_lookup_or_for_unknown_platforms():
    room = Room("grok")
    room.members["grok"] = Member("grok", Asker("grok"))
    await room.say("q", "chat", ["grok"])  # no ask function: the ASK line is just its answer
    assert room.entries[-1].text.endswith("ASK @gpt: what did I decide about my knee?")


async def test_a_bot_whose_turn_failed_is_sent_the_missed_messages_again():
    room = Room("claude", "gpt")
    room.bot("gpt").fail = True
    await room.say("first question", "chat", ["claude", "gpt"])
    assert "gpt" not in room.seen  # its bookmark did not move
    room.bot("gpt").fail = False
    await room.say("second question", "chat", ["gpt"])
    sent = room.bot("gpt").prompt()
    assert "first question" in sent and "claude r1" in sent and "second question" in sent


def test_ask_counts_only_as_the_last_line():
    from prism.meeting import _ask_request
    assert _ask_request("Checking.\n**ASK @GPT:** my knee?").groups() == ("GPT", "my knee?")
    assert _ask_request("I used ASK @gpt: my knee? earlier.\nHere is the plan.") is None
    assert _ask_request("ASK @gpt: x\nso the plan is...") is None


class AlwaysAsks(Fake):
    """Asks @any every time it is allowed to; reports what it got."""

    async def complete(self, messages, system=None):
        self.calls.append((messages, system))
        last = messages[-1]["content"]
        if last.startswith("[No asks left"):
            return "done without more help"
        return f"saw {last.count('answered]')} answers\nASK @any: where is my skin routine?"


async def test_ask_any_asks_every_other_platform_within_the_budget():
    room = Room("claude")
    room.members["claude"] = Member("claude", AlwaysAsks("claude"))
    asked = []

    async def ask(platform, question):
        asked.append(platform)
        return "Cleanser, tretinoin 0.05%, SPF." if platform == "gpt" else "I don't have that."

    s = Session(room.entries, room.members, make_script("chat", ["claude"], 1), room.emit,
                ask=ask, askable=["gpt", "grok", "claude"], max_asks=1)
    room.entries.append(Entry("user", "topic", "print my skin care routine"))
    await s.run()
    assert sorted(asked) == ["gpt", "grok"]  # everyone else, once: the budget was 1
    assert "ASK @any" in room.bot("claude").calls[0][1] and "1 asks per turn" in room.bot("claude").calls[0][1]
    second = room.bot("claude").calls[1][0][-1]["content"]
    assert "[gpt answered]\nCleanser" in second and "[grok answered]\nI don't have that." in second
    assert room.entries[-1].text == "done without more help"
    assert [k for k, _ in room.kinds()].count("lookup") == 2


async def test_no_ask_budget_means_no_ask_instructions():
    room = Room("claude")
    s = Session(room.entries, room.members, make_script("chat", ["claude"], 1), room.emit,
                ask=lambda p, q: None, askable=["gpt"], max_asks=0)
    room.entries.append(Entry("user", "topic", "hi"))
    await s.run()
    assert "ASK" not in room.bot("claude").calls[0][1]
