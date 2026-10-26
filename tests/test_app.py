"""End-to-end through the HTTP API with fake providers (no network)."""

import time

import pytest
from fastapi.testclient import TestClient

from prism import app as app_mod
from test_meeting import Fake


@pytest.fixture
def client(tmp_path, monkeypatch, shared_sandbox):
    monkeypatch.setenv("PRISM_HOME", str(tmp_path))
    monkeypatch.setenv("PRISM_DB_PATH", str(tmp_path / "t.db"))
    for fn in (app_mod.settings, app_mod.store, app_mod.browsers, app_mod.library, app_mod.runner,
               app_mod.python_version, app_mod.kg, app_mod.engine):
        fn.cache_clear()
    monkeypatch.setattr(app_mod, "sandbox", lambda: shared_sandbox)  # made once per test run
    app_mod._vault["key"] = None

    calls = []

    def fake_provider(cred, **kw):
        p = cred["platform"]
        calls.append((p, kw.get("resend_system", False)))
        f = Fake(p)
        f.thread_url = (kw.get("thread_url") or f"https://{p}.example/c/{len(calls)}") \
            if cred["transport"] == "browser" else None
        return f

    monkeypatch.setattr(app_mod, "make_provider", fake_provider)
    with TestClient(app_mod.app) as c:
        c.provider_calls = calls
        yield c


def wait_meeting(client, chat_id: int) -> dict:
    """Poll like the UI does until the meeting has finished."""
    for _ in range(200):
        v = client.get(f"/api/chats/{chat_id}").json()
        if not v["meeting"]:
            return v
        time.sleep(0.02)
    raise AssertionError("meeting did not finish")


def test_locked_until_unlocked_and_wrong_password(client):
    assert client.get("/api/status").json() == {"version": app_mod.__version__,
                                                 "initialized": False, "unlocked": False}
    assert client.put("/api/credentials/gpt", json={"secret": "k"}).status_code == 423
    assert client.post("/api/unlock", json={"password": "longenough"}).json()["unlocked"]
    client.post("/api/lock")
    assert client.post("/api/unlock", json={"password": "wrongwrong"}).status_code == 401


def test_full_chat_turn_binds_daughters_and_persists(client):
    client.post("/api/unlock", json={"password": "longenough"})
    client.put("/api/credentials/claude", json={"secret": "sk-x"})
    r = client.put("/api/credentials/gpt", json={"transport": "browser"})
    assert {c["platform"]: c["transport"] for c in r.json()["credentials"]} == {"claude": "api", "gpt": "browser"}
    assert client.post("/api/chats/1/messages", json={"content": "hi"}).status_code == 404

    chat = client.post("/api/chats").json()
    assert client.post(f"/api/chats/{chat['id']}/messages", json={"content": "Q?"}).status_code == 400  # no primary
    client.put("/api/primary", json={"platform": "claude"})
    client.post(f"/api/chats/{chat['id']}/context", json={"title": "Rule", "content": "be brief", "global": True})

    v = client.post(f"/api/chats/{chat['id']}/messages",
                    json={"content": "Q?", "rounds": 2, "speakers": {"claude": "short"}}).json()
    assert v["meeting"]["running"] and v["meeting"]["rounds"] == 2
    v = wait_meeting(client, chat["id"])
    assert v["chat"]["title"] == "Q?" and v["chat"]["primary_platform"] == "claude"
    kinds = [(m["meta"]["kind"], m["meta"].get("round")) for m in v["messages"]]
    assert kinds[0] == ("topic", None) and kinds[-1] == ("final", 2)
    assert v["messages"][0]["meta"]["format"] == "roundtable"
    assert kinds.count(("analysis", 1)) == 1 and kinds.count(("speaker", 2)) == 2
    assert v["messages"][-1]["content"] == "FINAL by claude"
    daughters = {d["platform"]: d for d in v["daughters"]}
    assert set(daughters) == {"claude", "gpt", "moderator"}
    assert daughters["gpt"]["remote_url"].startswith("https://gpt.example/c/")  # browser bot bound
    assert daughters["claude"]["remote_url"] is None
    assert [m["content"] for m in daughters["gpt"]["messages"]][1::2] == ["gpt r1", "gpt r2"]
    assert daughters["moderator"]["messages"][-1]["content"] == "FINAL by claude"
    assert daughters["gpt"]["seen_message_id"] is not None
    assert v["context"][0]["chat_id"] is None
    # Settings are remembered for the next meeting.
    assert v["meeting_defaults"]["rounds"] == 2 and v["meeting_defaults"]["speakers"] == {"claude": "short"}

    # Data survives a "restart": fresh caches, same DB file, re-unlock.
    for fn in (app_mod.settings, app_mod.store):
        fn.cache_clear()
    app_mod._vault["key"] = None
    assert client.get(f"/api/chats/{chat['id']}").json()["messages"][-1]["content"] == "FINAL by claude"


def test_user_joins_the_meeting_and_can_stop_it(client, monkeypatch):
    client.post("/api/unlock", json={"password": "longenough"})
    client.put("/api/credentials/claude", json={"secret": "sk-x"})
    client.put("/api/credentials/gpt", json={"secret": "sk-y"})
    client.put("/api/primary", json={"platform": "claude"})
    chat = client.post("/api/chats").json()["id"]

    import asyncio
    gates = []
    real = app_mod.make_provider
    def slow(cred, **kw):  # speakers wait until released, so the meeting stays open
        f = real(cred, **kw)
        if kw:
            f.gate = asyncio.Event()
            gates.append(f.gate)
        return f
    monkeypatch.setattr(app_mod, "make_provider", slow)

    v = client.post(f"/api/chats/{chat}/messages",
                    json={"content": "Topic", "rounds": 5, "format": "roundtable"}).json()
    assert v["meeting"]["waiting_on"] == ["claude", "gpt"] or v["meeting"]["phase"] == "starting"
    v = client.post(f"/api/chats/{chat}/messages", json={"content": "my input"}).json()
    assert v["messages"][-1]["meta"] == {"kind": "user", "round": 1, "during": True}
    assert client.post(f"/api/chats/{chat}/meeting/stop").status_code == 200
    v = wait_meeting(client, chat)
    assert v["messages"][-1]["meta"]["kind"] == "final"
    assert {m["meta"].get("error") for m in v["messages"] if m["meta"]["kind"] == "speaker"} == {"stopped"}
    # A message after the meeting ended starts a new one in the same chat.
    monkeypatch.setattr(app_mod, "make_provider", real)
    client.post(f"/api/chats/{chat}/messages", json={"content": "Next topic", "rounds": 1})
    v = wait_meeting(client, chat)
    assert [m["content"] for m in v["messages"] if m["meta"]["kind"] == "topic"] == ["Topic", "Next topic"]


def test_browser_transport_rejected_for_platform_without_web_app(client):
    client.post("/api/unlock", json={"password": "longenough"})
    assert client.put("/api/credentials/ollama", json={"transport": "browser"}).status_code == 400
    assert client.get("/api/platforms").json()["catalog"]["claude"]["web"] is True


def test_context_added_mid_chat_is_resent_to_bound_daughters(client):
    client.post("/api/unlock", json={"password": "longenough"})
    client.put("/api/credentials/claude", json={"secret": "sk-x"})
    client.put("/api/credentials/gpt", json={"transport": "browser"})
    client.put("/api/primary", json={"platform": "claude"})
    client.put("/api/profile", json={"mode": "off"})  # its updates are calls too
    chat = client.post("/api/chats").json()["id"]
    def send():
        client.post(f"/api/chats/{chat}/messages", json={"content": "Q?", "format": "chat"})
        wait_meeting(client, chat)
    daughters = lambda: {(p, r) for p, r in client.provider_calls}

    send()
    assert daughters() == {("claude", False), ("gpt", False)}   # first turn: new chats anyway
    client.provider_calls.clear(); send()
    assert daughters() == {("claude", False), ("gpt", False)}   # nothing new
    client.post(f"/api/chats/{chat}/context", json={"title": "New", "content": "fact", "always": True})
    client.provider_calls.clear(); send()
    assert daughters() == {("claude", True), ("gpt", True)}     # context added -> resend
    client.provider_calls.clear(); send()
    assert daughters() == {("claude", False), ("gpt", False)}


def test_mentions_pick_who_answers(client):
    client.post("/api/unlock", json={"password": "longenough"})
    for p in ("claude", "gpt", "gemini"):
        client.put(f"/api/credentials/{p}", json={"secret": "k"})
    client.put("/api/primary", json={"platform": "claude"})
    chat = client.post("/api/chats").json()["id"]
    say = lambda text, **kw: (client.post(f"/api/chats/{chat}/messages", json={"content": text, **kw}),
                              wait_meeting(client, chat))[1]

    v = say("@gpt what is 2+2?", format="roundtable", speakers={"gpt": "off"})
    assert [m["meta"]["speaker"] for m in v["messages"] if m["meta"]["kind"] == "speaker"] == ["gpt"]
    assert v["messages"][0]["meta"]["format"] == "chat"  # one bot: just a chat
    v = say("@gemini @claude check gpt", format="chat")
    last = [m["meta"]["speaker"] for m in v["messages"] if m["meta"]["kind"] == "speaker"][1:]
    assert sorted(last) == ["claude", "gemini"]
    gem = next(d for d in v["daughters"] if d["platform"] == "gemini")
    assert "gpt r1" in gem["messages"][0]["content"]  # it saw gpt's answer
    v = say("@moderator status?")
    assert v["messages"][-1]["meta"]["kind"] == "summary"
    r = client.post(f"/api/chats/{chat}/messages", json={"content": "@bard hi"})
    assert r.status_code == 400 and "@claude" in r.json()["detail"]
    assert client.post(f"/api/chats/{chat}/messages", json={"content": "mail me at a@b.com"}).status_code == 200


def test_named_context_write_use_forget(client):
    client.post("/api/unlock", json={"password": "longenough"})
    for p in ("claude", "gpt"):
        client.put(f"/api/credentials/{p}", json={"secret": "k"})
    client.put("/api/primary", json={"platform": "claude"})
    chat = client.post("/api/chats").json()["id"]
    post = lambda text: client.post(f"/api/chats/{chat}/messages", json={"content": text, "format": "chat"})

    v = post("/save #spec Budget is 10k. Ship by May.").json()
    assert v["notice"].startswith("Saved #spec") and v["messages"] == []  # not a room message
    item = next(c for c in v["context"] if c["name"] == "spec")
    assert item["always"] is False and item["chat_id"] == chat

    post("@gpt plan it within #spec"); v = wait_meeting(client, chat)
    assert [m["meta"]["kind"] for m in v["messages"]][:2] == ["context", "topic"]
    gpt = next(d for d in v["daughters"] if d["platform"] == "gpt")
    sent = gpt["messages"][0]["content"]
    assert "Budget is 10k" in sent and "plan it within #spec" in sent

    # /save with no text keeps the latest answer; same name again replaces it
    v = post("/save #plan").json()
    assert next(c for c in v["context"] if c["name"] == "plan")["content"] == "gpt r1"
    post("/save #plan revised"); v = client.get(f"/api/chats/{chat}").json()
    assert [c["content"] for c in v["context"] if c["name"] == "plan"] == ["revised"]

    assert post("@gpt use #nope").status_code == 400
    assert post("@gpt is C# better? see item #1").status_code == 200
    wait_meeting(client, chat)
    assert post("/forget #plan").json()["notice"] == "Forgot #plan."
    assert post("/forget #plan").status_code == 404
    v = post("/save-global #style Be terse.").json()
    assert next(c for c in v["context"] if c["name"] == "style")["chat_id"] is None


def test_mentions_sent_while_busy_wait_their_turn(client, monkeypatch):
    import asyncio
    client.post("/api/unlock", json={"password": "longenough"})
    for p in ("claude", "gpt"):
        client.put(f"/api/credentials/{p}", json={"secret": "k"})
    client.put("/api/primary", json={"platform": "claude"})
    chat = client.post("/api/chats").json()["id"]
    real, gates = app_mod.make_provider, []

    def slow(cred, **kw):
        f = real(cred, **kw)
        if cred["platform"] == "claude" and not gates:
            f.gate = asyncio.Event()
            gates.append(f.gate)
        return f
    monkeypatch.setattr(app_mod, "make_provider", slow)

    client.post(f"/api/chats/{chat}/messages", json={"content": "@claude think", "format": "chat"})
    v = client.post(f"/api/chats/{chat}/messages", json={"content": "@gpt check it"}).json()
    assert v["queued"] == ["@gpt check it"]
    v = client.post(f"/api/chats/{chat}/messages", json={"content": "also consider cost"}).json()
    assert v["messages"][-1]["meta"]["during"] is True  # no mention: joins the running turn
    portal_loop = app_mod._session_tasks[chat].get_loop()
    portal_loop.call_soon_threadsafe(gates[0].set)
    v = wait_meeting(client, chat)
    for _ in range(100):  # the queued turn starts right after
        v = client.get(f"/api/chats/{chat}").json()
        if not v["meeting"] and not v["queued"] and v["messages"][-1]["meta"]["kind"] == "speaker" \
                and v["messages"][-1]["meta"]["speaker"] == "gpt":
            break
        __import__("time").sleep(0.02)
    speakers = [m["meta"]["speaker"] for m in v["messages"] if m["meta"]["kind"] == "speaker"]
    assert speakers[0] == "claude" and speakers[-1] == "gpt"
    gpt = next(d for d in v["daughters"] if d["platform"] == "gpt")
    assert "claude r1" in gpt["messages"][0]["content"]


def test_plan_lines_assign_tasks(client):
    client.post("/api/unlock", json={"password": "longenough"})
    for p in ("claude", "gpt", "grok"):
        client.put(f"/api/credentials/{p}", json={"secret": "k"})
    chat = client.post("/api/chats").json()["id"]
    text = "Launch video for our protein bar.\n1. @gpt write the script\n2. @claude review it, then @grok draft the posts"
    assert client.post(f"/api/chats/{chat}/messages", json={"content": text}).status_code == 200
    v = wait_meeting(client, chat)
    assert v["messages"][0]["meta"]["format"] == "plan"
    assert [m["meta"]["speaker"] for m in v["messages"][1:]] == ["gpt", "claude", "grok"], [m["meta"] for m in v["messages"]]
    grok = next(d for d in v["daughters"] if d["platform"] == "grok")["messages"][0]["content"]
    assert "Your task: draft the posts" in grok and "claude r1" in grok and "gpt r1" in grok
    # one line with two names is not a plan: an ordinary chat
    client.post(f"/api/chats/{chat}/messages", json={"content": "@gpt @claude compare", "format": "chat"})
    assert wait_meeting(client, chat)["messages"][-3]["meta"]["format"] == "chat"
    r = client.post(f"/api/chats/{chat}/messages", json={"content": "@gpt a\n@all b"})
    assert r.status_code == 400


def test_recall_saves_platform_memory_as_global_context(client):
    client.post("/api/unlock", json={"password": "longenough"})
    client.put("/api/credentials/gpt", json={"transport": "browser"})
    chat = client.post("/api/chats").json()["id"]
    r = client.post(f"/api/chats/{chat}/messages",
                    json={"content": "/recall @gpt #workout my training plan and injuries"}).json()
    assert r["notice"].startswith("Recalling #workout from gpt")
    assert r["messages"][-1]["meta"]["kind"] == "recall"  # the UI keeps refreshing until done
    for _ in range(200):
        v = client.get(f"/api/chats/{chat}").json()
        if not v["messages"][-1]["meta"].get("pending"):
            break
        time.sleep(0.02)
    item = next(c for c in v["context"] if c["name"] == "workout")
    assert item["content"] == "gpt r1" and item["chat_id"] is None
    assert v["messages"][-1]["meta"]["chars"] == len("gpt r1")
    assert client.post(f"/api/chats/{chat}/messages",
                       json={"content": "/recall @grok #x y"}).status_code == 400


def test_ask_lookup_is_stored_in_the_room(client, monkeypatch):
    from test_meeting import Asker
    def provider(cred, **kw):
        p = cred["platform"]
        f = Asker(p) if p == "grok" and not kw.get("fresh") else Fake(p)
        f.thread_url = None
        return f
    monkeypatch.setattr(app_mod, "make_provider", provider)
    client.post("/api/unlock", json={"password": "longenough"})
    for p in ("gpt", "grok"):
        client.put(f"/api/credentials/{p}", json={"transport": "browser"})
    chat = client.post("/api/chats").json()["id"]
    client.post(f"/api/chats/{chat}/messages", json={"content": "@grok plan my runs", "format": "chat"})
    v = wait_meeting(client, chat)
    assert [m["meta"]["kind"] for m in v["messages"]] == ["topic", "lookup", "speaker"]
    assert v["messages"][2]["content"] == "grok used: gpt r1" and not v["messages"][2]["meta"]["error"]


def test_connect_during_a_meeting_asks_then_stops_it(client, monkeypatch):
    client.post("/api/unlock", json={"password": "longenough"})
    client.put("/api/credentials/gpt", json={"transport": "browser"})
    chat = client.post("/api/chats").json()["id"]
    import asyncio
    real = app_mod.make_provider
    def held(cred, **kw):  # the bot never answers on its own: the meeting stays open
        f = real(cred, **kw)
        f.gate = asyncio.Event()
        return f
    monkeypatch.setattr(app_mod, "make_provider", held)
    opened = []

    class Page:
        url = "https://chatgpt.com/"
        async def goto(self, url, **kw):
            opened.append(url)

    class Browsers:
        async def new_page(self, visible=False):
            return Page()
        async def close(self):
            pass
    monkeypatch.setattr(app_mod, "browsers", lambda: Browsers())

    client.post(f"/api/chats/{chat}/messages", json={"content": "@gpt hi", "format": "chat"})
    r = client.post("/api/platforms/gpt/connect")
    assert r.status_code == 409 and r.json()["detail"].startswith("Signing in restarts")
    assert not opened
    r = client.post("/api/platforms/gpt/connect?stop=true")
    assert r.status_code == 200 and opened
    assert client.get(f"/api/chats/{chat}").json()["meeting"] is None


def test_attached_files_reach_the_bots_and_media_is_served(client, monkeypatch):
    client.post("/api/unlock", json={"password": "longenough"})
    client.put("/api/credentials/gpt", json={"transport": "browser"})
    chat = client.post("/api/chats").json()["id"]
    up = client.post("/api/media?name=my plan.pdf", content=b"%PDF-1.4 fake").json()
    assert up["name"] == "my plan.pdf" and up["id"].endswith("-my plan.pdf")
    got = []
    real = app_mod.make_provider
    def capture(cred, **kw):
        f = real(cred, **kw)
        async def with_files(messages, system=None, files=()):
            got.append(list(files))
            return "read it"
        f.complete_with_files = with_files
        return f
    monkeypatch.setattr(app_mod, "make_provider", capture)
    client.post(f"/api/chats/{chat}/messages", json={"content": "", "files": [up["id"]]})
    v = wait_meeting(client, chat)
    assert v["messages"][0]["meta"]["files"] == [up["id"]] and got[0][0].endswith(up["id"])
    r = client.get(f"/media/{up['id']}")
    assert r.content == b"%PDF-1.4 fake" and "my%20plan.pdf" in r.headers["content-disposition"]
    # the next turn does not resend it
    client.post(f"/api/chats/{chat}/messages", json={"content": "@gpt and now?"})
    wait_meeting(client, chat)
    assert len(got) == 1
    assert client.get("/media/..%2Fprism.db").status_code in (400, 404)
    assert client.post(f"/api/chats/{chat}/messages", json={"content": "x", "files": ["nope"]}).status_code == 404


def test_ask_budget_is_a_remembered_setting(client):
    client.post("/api/unlock", json={"password": "longenough"})
    client.put("/api/credentials/gpt", json={"transport": "browser"})
    chat = client.post("/api/chats").json()["id"]
    assert client.get(f"/api/chats/{chat}").json()["meeting_defaults"]["asks"] == 3
    client.post(f"/api/chats/{chat}/messages", json={"content": "@gpt hi", "asks": 1})
    assert wait_meeting(client, chat)["meeting_defaults"]["asks"] == 1
    assert client.post(f"/api/chats/{chat}/messages", json={"content": "@gpt hi", "asks": 9}).status_code == 422


def test_quit_stops_the_server_only_with_a_json_confirm(client):
    class Server:
        should_exit = False
    assert client.post("/api/quit", json={"confirm": True}).status_code == 409  # no launcher
    app_mod.app.state.server = server = Server()
    try:
        # a plain cross-site form post can't carry JSON
        assert client.post("/api/quit", content=b"confirm=true",
                           headers={"content-type": "application/x-www-form-urlencoded"}).status_code == 422
        assert not server.should_exit
        assert client.post("/api/quit", json={"confirm": True}).json() == {"ok": True}
        assert server.should_exit
    finally:
        del app_mod.app.state.server


def test_learned_layout_is_used_shown_and_reset(client):
    client.post("/api/unlock", json={"password": "longenough"})
    app_mod.remember_layout("gpt", {"assistant_message": "div.answer"})
    r = client.get("/api/platforms/gpt/selectors").json()
    assert r["learned"] == {"assistant_message": "div.answer"}
    assert r["effective"]["assistant_message"] == "div.answer"
    # Saving the whole effective set (what the UI does) keeps only the user's own change.
    edited = {**r["effective"], "composer": "#mine"}
    r = client.put("/api/platforms/gpt/selectors", json=edited).json()
    assert r["overrides"] == {"composer": "#mine"} and r["effective"]["assistant_message"] == "div.answer"
    r = client.put("/api/platforms/gpt/selectors", json={}).json()
    assert r["learned"] == {} and r["overrides"] == {}
    assert r["effective"] == r["defaults"]


async def test_layout_advice_comes_from_another_assistant(client, monkeypatch):
    client.post("/api/unlock", json={"password": "longenough"})
    client.put("/api/credentials/gpt", json={"transport": "browser"})
    client.put("/api/credentials/claude", json={"transport": "browser"})
    client.put("/api/credentials/gemini", json={"secret": "k"})
    asked = []

    def provider(cred, **kw):
        asked.append((cred["platform"], kw.get("advisor")))
        return Fake(cred["platform"])
    monkeypatch.setattr(app_mod, "make_provider", provider)
    assert await app_mod.layout_advisor("gpt")("which box?") == "gemini r1"
    assert asked == [("gemini", False)]  # an API one first; never the broken site itself
    async with app_mod.browsers().lock("claude"):  # busy: asking it could wait on us
        client.delete("/api/credentials/gemini")
        assert await app_mod.layout_advisor("gpt")("which box?") == ""


def wait_profile(client, chat, last):
    """The update starts just after the meeting ends: wait for its outcome, not for 'idle'."""
    for _ in range(300):
        v = client.get(f"/api/chats/{chat}").json()
        if v["profile"]["last"] == last and not v["profile"]["updating"]:
            return v
        time.sleep(0.02)
    raise AssertionError(f"profile update did not finish: {v['profile']}")


def test_profile_is_kept_from_what_the_user_says(client, monkeypatch):
    prompts = []

    class ProfileWriter(Fake):
        async def complete(self, messages, system=None):
            prompts.append(messages[-1]["content"])
            return "UNCHANGED" if len(prompts) > 1 else "- Runs marathons; knee injury in 2025\n- Prefers short answers"

    def provider(cred, **kw):
        f = ProfileWriter(cred["platform"]) if kw.get("fresh") else Fake(cred["platform"])
        f.thread_url = None
        return f
    monkeypatch.setattr(app_mod, "make_provider", provider)
    client.post("/api/unlock", json={"password": "longenough"})
    client.put("/api/credentials/gpt", json={"transport": "browser"})
    client.put("/api/primary", json={"platform": "gpt"})
    chat = client.post("/api/chats").json()["id"]
    for text in ("@gpt I run marathons", "@gpt my knee hurts since 2025"):
        client.post(f"/api/chats/{chat}/messages", json={"content": text, "format": "chat"})
        wait_meeting(client, chat)
    assert not prompts  # two new messages: not enough for an update yet
    client.post(f"/api/chats/{chat}/messages", json={"content": "@gpt keep answers short", "format": "chat"})
    wait_meeting(client, chat)
    v = wait_profile(client, chat, "updated from 3 messages")
    assert len(prompts) == 1 and "I run marathons" in prompts[0] and "keep answers short" in prompts[0]
    item = next(c for c in v["context"] if c["name"] == "profile")
    assert item["chat_id"] is None and item["always"] and "marathons" in item["content"]
    assert v["profile"]["last"] == "updated from 3 messages"

    # Every chat gets it.
    other = client.post("/api/chats").json()["id"]
    client.post(f"/api/chats/{other}/messages", json={"content": "@gpt hi", "format": "chat"})
    wait_meeting(client, other)
    # /profile forces an update; UNCHANGED keeps the profile as it was.
    r = client.post(f"/api/chats/{chat}/messages", json={"content": "/profile"}).json()
    assert r["notice"].startswith("Updating your profile")
    v = wait_profile(client, chat, "checked - nothing new")
    assert "@gpt hi" in prompts[1] and "I run marathons" not in prompts[1].split("RECENTLY")[1]
    assert next(c for c in v["context"] if c["name"] == "profile")["content"] == item["content"]
    assert v["profile"]["last"] == "checked - nothing new"

    # Off: no more updates. Editing it by hand keeps it going to every chat.
    client.post(f"/api/chats/{chat}/messages", json={"content": "/profile off"})
    assert client.get(f"/api/chats/{chat}").json()["profile"]["mode"] == "off"
    client.post(f"/api/chats/{chat}/messages", json={"content": "/save-global #profile I am a runner."})
    item = next(c for c in client.get(f"/api/chats/{chat}").json()["context"] if c["name"] == "profile")
    assert item["always"] and item["content"] == "I am a runner."


def test_ask_prism_searches_the_users_other_chats(client, monkeypatch):
    from test_meeting import Asker
    questions = []

    class PrismAsker(Asker):
        async def complete(self, messages, system=None):
            self.calls.append((messages, system))
            if len(self.calls) == 1:
                questions.append(system)
                return "ASK @prism: knee injury decision"
            return "grok used: " + messages[-1]["content"]

    def provider(cred, **kw):
        f = PrismAsker(cred["platform"]) if cred["platform"] == "grok" else Fake(cred["platform"])
        f.thread_url = None
        return f
    client.post("/api/unlock", json={"password": "longenough"})
    client.put("/api/credentials/gpt", json={"secret": "k"})
    old = client.post("/api/chats").json()["id"]
    client.post(f"/api/chats/{old}/messages", json={"content": "@gpt I decided to rest my knee injury for 6 weeks", "format": "chat"})
    wait_meeting(client, old)
    client.put("/api/credentials/grok", json={"secret": "k"})
    monkeypatch.setattr(app_mod, "make_provider", provider)
    chat = client.post("/api/chats").json()["id"]
    client.post(f"/api/chats/{chat}/messages", json={"content": "@grok what about my knee?", "format": "chat"})
    v = wait_meeting(client, chat)
    assert "ASK @prism" in questions[0]  # offered even with no web platforms set up
    lookup = next(m for m in v["messages"] if m["meta"]["kind"] == "lookup")
    assert lookup["meta"]["speaker"] == "prism" and "rest my knee injury for 6 weeks" in lookup["content"]
    assert "rest my knee" in v["messages"][-1]["content"]


def test_revise_sends_the_version_on_screen_to_the_named_bot(client, monkeypatch):
    seen = []

    class Reviser(Fake):
        async def complete(self, messages, system=None):
            seen.append(messages[-1]["content"])
            return "```html\n<h1>Darker</h1>\n```\nMade it darker."

    def provider(cred, **kw):
        f = Reviser(cred["platform"]); f.thread_url = None
        return f
    monkeypatch.setattr(app_mod, "make_provider", provider)
    client.post("/api/unlock", json={"password": "longenough"})
    client.put("/api/credentials/gpt", json={"secret": "k"})
    chat = client.post("/api/chats").json()["id"]
    r = client.post(f"/api/chats/{chat}/messages", json={
        "content": "@gpt make the hero darker", "format": "chat",
        "artifact": {"title": "Dilek", "kind": "html", "code": "<h1>Light</h1>", "who": "claude", "version": 2}})
    assert r.status_code == 200, r.text
    v = wait_meeting(client, chat)
    assert '```html\n<h1>Light</h1>\n```' in seen[0] and "complete revised file" in seen[0]
    ctx = [m for m in v["messages"] if m["meta"]["kind"] == "context"]
    assert ctx and ctx[0]["meta"]["name"] == "artifact" and ctx[0]["meta"]["version"] == 2
    assert v["messages"][-1]["content"].startswith("```html\n<h1>Darker</h1>")


def test_changes_only_from_prisms_own_page(client):
    # A preview the bots wrote runs in a sandbox (origin "null"); other sites have their own origin.
    assert client.post("/api/lock", headers={"origin": "null"}).status_code == 403
    assert client.post("/api/lock", headers={"origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/lock", headers={"origin": "http://testserver"}).status_code == 200
    assert client.post("/api/lock").status_code == 200  # no Origin: not from a browser page
    assert client.get("/api/status", headers={"origin": "null"}).status_code == 200


def wait_library(client, chat_id: int) -> dict:
    for _ in range(300):
        v = client.get(f"/api/chats/{chat_id}").json()
        if all(not c.get("status", {}).get("busy") for c in v["context"]):
            return v
        time.sleep(0.02)
    raise AssertionError("the folder was not read")


def test_folder_passages_files_and_ask_docs(client, monkeypatch, tmp_path):
    from test_library import pdf
    from test_meeting import Asker
    folder = tmp_path / "Trip"
    folder.mkdir()
    (folder / "hotels.md").write_text("We booked the Alfama hotel in Lisbon for three nights.")
    (folder / "visa.pdf").write_bytes(pdf(["Visa: none needed for stays under ninety days"]))
    (folder / "budget.txt").write_text("Food budget is forty euros per day.")

    client.post("/api/unlock", json={"password": "longenough"})
    client.put("/api/credentials/gpt", json={"transport": "browser"})
    client.put("/api/credentials/grok", json={"secret": "k"})
    chat = client.post("/api/chats").json()["id"]
    assert client.post(f"/api/chats/{chat}/library", json={"name": "trip", "location": str(tmp_path / "nope")}
                       ).status_code == 400
    r = client.post(f"/api/chats/{chat}/library", json={"name": "#trip", "location": str(folder)})
    assert r.status_code == 200
    v = wait_library(client, chat)
    item = v["context"][0]
    assert item["name"] == "trip" and item["status"]["files"] == 3 and "3 files" in item["content"]

    got, systems = [], []
    real = app_mod.make_provider

    class DocsAsker(Asker):
        async def complete(self, messages, system=None):
            self.calls.append((messages, system))
            systems.append(system)
            if len(self.calls) == 1:
                return "ASK @docs: food budget per day"
            return "grok used: " + messages[-1]["content"]

    def capture(cred, **kw):
        if cred["platform"] == "grok":
            f = DocsAsker("grok")
            f.thread_url = None
            return f
        f = real(cred, **kw)
        async def with_files(messages, system=None, files=()):
            got.append((messages[-1]["content"], list(files)))
            return "gpt read it"
        f.complete_with_files = with_files
        return f
    monkeypatch.setattr(app_mod, "make_provider", capture)

    client.post(f"/api/chats/{chat}/messages", json={"content": "@gpt do I need a visa for Lisbon? #trip", "format": "chat"})
    v = wait_meeting(client, chat)
    ctx = next(m for m in v["messages"] if m["meta"]["kind"] == "context")
    assert ctx["meta"]["library"] and ctx["meta"]["sources"][0]["path"] == "visa.pdf"
    assert "[1] visa.pdf, p.1\nVisa: none needed" in ctx["content"]
    assert ctx["meta"]["attached"] == ["visa.pdf"]
    prompt, files = got[0]
    assert "Visa: none needed" in prompt and files == [str(folder / "visa.pdf")]

    # The same file is not uploaded again in this chat.
    client.post(f"/api/chats/{chat}/messages", json={"content": "@gpt and the visa rules again? #trip", "format": "chat"})
    wait_meeting(client, chat)
    assert len(got) == 1  # no files this time, so a plain turn
    again = [m for m in client.get(f"/api/chats/{chat}").json()["messages"] if m["meta"]["kind"] == "context"][-1]
    assert "Visa: none needed" in again["content"] and "attached" not in again["meta"]

    client.post(f"/api/chats/{chat}/messages", json={"content": "@grok how much can I spend on food?", "format": "chat"})
    v = wait_meeting(client, chat)
    assert "ASK @docs" in systems[0]
    lookup = next(m for m in v["messages"] if m["meta"]["kind"] == "lookup")
    assert lookup["meta"]["speaker"] == "docs" and "forty euros per day" in lookup["content"]

    # Always: searched for every message without #trip.
    client.delete(f"/api/context/{item['id']}")
    assert client.get(f"/api/chats/{chat}").json()["context"] == []
    client.post(f"/api/chats/{chat}/library", json={"name": "trip", "location": str(folder), "always": True})
    wait_library(client, chat)
    client.post(f"/api/chats/{chat}/messages", json={"content": "@grok which hotel did we book in Lisbon?", "format": "chat"})
    v = wait_meeting(client, chat)
    ctx = [m for m in v["messages"] if m["meta"]["kind"] == "context"][-1]
    assert "Alfama hotel" in ctx["content"] and "attached" not in ctx["meta"]


def test_readers_settings(client, monkeypatch):
    async def detect(home, creds):
        return {"vision": [], "transcribe": [], "embed": [{"id": "ollama:nomic-embed-text", "label": "x"}],
                "tools": {"whisper": None, "whisper_model": None, "ffmpeg": None}}
    monkeypatch.setattr(app_mod.enrich, "detect", detect)
    client.post("/api/unlock", json={"password": "longenough"})
    r = client.get("/api/readers").json()
    assert r["config"] == {"vision": "auto", "transcribe": "auto", "embed": ""}
    assert r["suggested"]["embed"] == "nomic-embed-text" and "whisper-model" in r["can_download"]
    assert client.put("/api/readers", json={"embed": "ollama:nomic-embed-text"}).json()["embed"] == "ollama:nomic-embed-text"
    assert client.post("/api/readers/download", json={"what": "evil"}).status_code == 400
    assert client.post("/api/local/import", json={"path": "/no/such/model.gguf"}).status_code == 400
    assert client.post("/api/local/import", json={"path": "/x/model.bin"}).json()["detail"].startswith("Point at a .gguf")


def wait_run(client, chat_id: int, until=("done", "waiting", "paused", "stopped")) -> dict:
    for _ in range(300):
        v = client.get(f"/api/chats/{chat_id}").json()
        if v["run"] and v["run"]["status"] in until and not v["run"]["live"]:
            return v
        time.sleep(0.02)
    raise AssertionError(f"run did not reach {until}: {v['run']}")


FLOW = """
name: brief
inputs: {topic: text}
agents:
  checker: {model: "@gpt", output: {verdict: PASS|FIX}}
nodes:
  write:  {agent: critic, prompt: "Write about {{input.topic}}."}
  check:  {agent: checker, prompt: "Check {{write.text}}"}
  ok:     {ask_you: "Publish it?", show: "{{write.text}}"}
edges:
  - write -> check
  - check -> write: {when: "verdict == FIX", max_rounds: 1}
  - check -> ok:    {when: "verdict == PASS or rounds_done"}
"""


def test_agents_answer_by_name_in_a_chat(client, monkeypatch):
    client.post("/api/unlock", json={"password": "longenough"})
    client.put("/api/credentials/gpt", json={"secret": "k"})
    assert client.put("/api/agents/gpt", json={"model": "@gpt"}).status_code == 400       # a platform's name
    assert client.put("/api/agents/critic", json={"model": "@nobody"}).status_code == 400
    assert client.put("/api/agents/critic", json={"model": "@gpt", "tools": ["shell"]}).status_code == 400
    r = client.put("/api/agents/critic", json={"model": "gpt", "instructions": "Find what is wrong.",
                                               "context": ["#spec"], "output": {}})
    assert r.json()["agents"] == [{"name": "critic", "model": "@gpt", "instructions": "Find what is wrong.",
                                   "context": ["spec"], "tools": [], "output": {}, "research": False}]
    systems = []
    real = app_mod.make_provider

    def capture(cred, **kw):
        f = real(cred, **kw)
        orig = f.complete
        async def complete(messages, system=None):
            systems.append(system)
            return await orig(messages, system)
        f.complete = complete
        return f
    monkeypatch.setattr(app_mod, "make_provider", capture)
    chat = client.post("/api/chats").json()["id"]
    client.post(f"/api/chats/{chat}/messages", json={"content": "@critic is this plan sound?", "format": "chat"})
    v = wait_meeting(client, chat)
    said = [m for m in v["messages"] if m["meta"]["kind"] == "speaker"]
    assert said[0]["meta"]["speaker"] == "critic" and "Your role here, set by the user: Find what is wrong." in systems[0]
    assert [d["platform"] for d in v["daughters"]] == ["critic"]


def test_flow_runs_in_its_own_room_with_notes_and_approval(client, monkeypatch):
    client.post("/api/unlock", json={"password": "longenough"})
    client.put("/api/credentials/gpt", json={"secret": "k"})
    client.put("/api/agents/critic", json={"model": "@gpt", "instructions": "Write well."})
    bad = client.post("/api/flows/validate", json={"source": FLOW.replace("max_rounds: 1", "")}).json()
    assert not bad["ok"] and "has no limit" in bad["errors"][0]
    good = client.post("/api/flows/validate", json={"source": FLOW}).json()
    assert good["ok"] and [n["id"] for n in good["graph"]["nodes"]] == ["write", "check", "ok"]
    assert client.post("/api/flows", json={"source": "name: x\nnodes: {}"}).status_code == 400
    assert client.post("/api/flows", json={"source": FLOW}).json()["saved"] == "brief"
    assert client.post("/api/flows/brief/runs", json={"inputs": {}}).status_code == 400  # topic missing

    sent = []

    class Writer:
        def __init__(self):
            self.name, self.model, self.thread_url, self.n = "gpt", "gpt-fake", None, 0

        async def complete(self, messages, system=None):
            sent.append((system, messages[-1]["content"]))
            self.n += 1
            if "Check" in messages[-1]["content"]:
                return "Fine.\nVERDICT: " + ("FIX" if self.n < 3 else "PASS")
            return f"draft {self.n}"
    writer = Writer()
    monkeypatch.setattr(app_mod, "make_provider", lambda cred, **kw: writer)

    v = client.post("/api/flows/brief/runs", json={"inputs": {"topic": "tides"}}).json()
    chat = v["chat"]["id"]
    assert v["chat"]["title"] == "Run: brief"
    v = wait_run(client, chat)
    run = v["run"]
    assert run["status"] == "waiting" and run["waiting"] == "ok" and run["done"] == ["write", "check"]
    kinds = [(m["meta"]["kind"], m["meta"].get("node")) for m in v["messages"]]
    assert kinds == [("topic", None), ("step", "write"), ("step", "check"), ("step", "write"),
                     ("step", "check"), ("ask_you", "ok")]
    assert v["messages"][2]["meta"]["fields"] == {"verdict": "FIX"}
    assert "Write well." in sent[0][0] and sent[0][1].startswith("Write about tides.")
    assert {d["platform"] for d in v["daughters"]} == {"critic", "checker"}  # a tab per agent

    # Said while it waits: kept for the next steps.
    client.post(f"/api/chats/{chat}/messages", json={"content": "make it shorter"})
    r = client.post(f"/api/runs/{run['id']}/answer", json={"node": "ok", "approved": True, "text": "go"}).json()
    v = wait_run(client, chat)
    assert v["run"]["status"] == "done"
    ask = next(m for m in v["messages"] if m["meta"]["kind"] == "ask_you")
    assert ask["meta"]["answer"] == "yes"
    assert v["messages"][-1]["meta"] == {**v["messages"][-1]["meta"], "kind": "run", "status": "done"}
    assert client.post(f"/api/runs/{run['id']}/answer", json={"node": "ok", "approved": True}).status_code == 409


def test_run_stops_and_resumes_after_a_restart(client, monkeypatch):
    import asyncio
    client.post("/api/unlock", json={"password": "longenough"})
    client.put("/api/credentials/gpt", json={"secret": "k"})
    client.post("/api/flows", json={"source": "name: slow\nagents: {w: {model: '@gpt'}}\nnodes: {a: {agent: w}, b: {agent: w}}\nedges: [a -> b]"})

    class Hang:
        name, model, thread_url = "gpt", "gpt-fake", None

        async def complete(self, messages, system=None):
            await asyncio.sleep(30)
    monkeypatch.setattr(app_mod, "make_provider", lambda cred, **kw: Hang())
    chat = client.post("/api/flows/slow/runs", json={}).json()["chat"]["id"]
    run_id = client.get(f"/api/chats/{chat}").json()["run"]["id"]
    assert client.delete(f"/api/chats/{chat}").status_code == 409
    v = client.post(f"/api/runs/{run_id}/stop").json()
    assert v["run"]["status"] == "stopped" and v["messages"][-1]["meta"]["status"] == "stopped"

    # A run that was going when PRISM closed comes back paused, and continues from its step.
    s = app_mod.store()
    state = s.get_run(run_id)["state"]
    s.save_run(run_id, {**state, "status": "running", "queue": ["a"]})   # as if PRISM died mid-step
    app_mod._live_runs.pop(run_id, None)
    app_mod.recover_runs()
    assert s.get_run(run_id)["status"] == "paused"
    monkeypatch.setattr(app_mod, "make_provider", lambda cred, **kw: __import__("test_meeting").Fake("gpt"))
    assert client.post(f"/api/runs/{run_id}/resume", json={}).status_code == 200
    v = wait_run(client, chat)
    assert v["run"]["status"] == "done" and v["run"]["done"] == ["a", "b"]


def test_draft_flow_is_checked_and_fixed_once(client, monkeypatch):
    client.post("/api/unlock", json={"password": "longenough"})
    client.put("/api/credentials/claude", json={"secret": "k"})
    client.put("/api/primary", json={"platform": "claude"})
    prompts = []

    class Drafter:
        name, model, thread_url = "claude", "c", None

        async def complete(self, messages, system=None):
            prompts.append(messages[-1]["content"])
            if len(prompts) == 1:  # forgets the loop limit
                return "```yaml\nname: d\nagents: {w: {model: '@claude'}}\nnodes: {a: {agent: w}, b: {agent: w}}\nedges: [a -> b, b -> a]\n```"
            return "```yaml\nname: d\nagents: {w: {model: '@claude'}}\nnodes: {a: {agent: w}, b: {agent: w}}\nstart: a\nedges: [a -> b, {b -> a: {max_rounds: 2}}]\n```"
    monkeypatch.setattr(app_mod, "make_provider", lambda cred, **kw: Drafter())
    r = client.post("/api/flows/draft", json={"description": "write then check, twice"}).json()
    assert "write then check, twice" in prompts[0] and "{{input.name}}" in prompts[0] and "@claude" in prompts[0]
    assert "has no limit" in prompts[1]
    assert r["errors"] == [] and r["source"].startswith("name: d")


def test_bot_runs_a_program_after_approval(client, monkeypatch):
    import sys
    client.post("/api/unlock", json={"password": "longenough"})
    client.put("/api/credentials/gpt", json={"secret": "k"})
    seen = []

    class Coder:
        name, model, thread_url = "gpt", "gpt-fake", None

        async def complete(self, messages, system=None):
            seen.append((system, messages[-1]["content"]))
            if "[ran python" in messages[-1]["content"]:
                return "The answer is " + messages[-1]["content"].split("\n")[1]
            return "Let me compute it.\nRUN @python\n```python\nprint(6 * 7)\n```"
    monkeypatch.setattr(app_mod, "make_provider", lambda cred, **kw: Coder())
    chat = client.post("/api/chats").json()["id"]
    client.post(f"/api/chats/{chat}/messages", json={"content": "@gpt what is 6 x 7?", "format": "chat"})
    for _ in range(200):
        v = client.get(f"/api/chats/{chat}").json()
        card = next((m for m in v["messages"] if m["meta"]["kind"] == "run_request"), None)
        if card:
            break
        time.sleep(0.02)
    assert "RUN @python" in seen[0][0] and "pandas" in seen[0][0]
    assert card["meta"]["status"] == "pending" and card["meta"]["code"] == "print(6 * 7)"
    assert v["meeting"]["approving"] == ["gpt"]
    r = client.post(f"/api/commands/{card['meta']['id']}", json={"allow": "chat", "chat_id": chat})
    assert r.status_code == 200
    v = wait_meeting(client, chat)
    card = next(m for m in v["messages"] if m["meta"]["kind"] == "run_request")
    assert card["meta"]["status"] == "done" and card["meta"]["output"].strip() == "42"
    assert v["messages"][-1]["content"] == "The answer is 42"
    assert client.get("/api/runner").json()["rules"] == {"chats": {str(chat): ["python"]}}
    assert client.post(f"/api/commands/{card['meta']['id']}", json={"allow": "once"}).status_code == 409

    # Off: bots are not offered RUN at all.
    client.put("/api/runner", json={"mode": "off"})
    seen.clear()
    client.post(f"/api/chats/{chat}/messages", json={"content": "@gpt again", "format": "chat"})
    wait_meeting(client, chat)
    assert "RUN @python" not in seen[0][0]
    # A card left waiting when PRISM closed is marked expired.
    s = app_mod.store()
    mid = s.add_message(chat, None, "assistant", "", {"kind": "run_request", "status": "pending", "id": "x"})
    app_mod.recover_runs()
    assert next(m for m in s.thread(chat) if m["id"] == mid)["meta"]["status"] == "expired"


def test_the_page_is_never_shown_from_an_old_cache(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    assert 'id="openKg"' in r.text and 'id="openFlows"' in r.text


def test_engine_models_listed_added_switched_and_removed(client, tmp_path):
    from prism import engine as eng
    client.post("/api/unlock", json={"password": "longenough"})
    v = client.get("/api/engine").json()
    assert [m["id"] for m in v["models"]] == [m.id for m in eng.MODELS]
    assert not any(m["installed"] for m in v["models"]) and v["engine"]["build"] == "cpu" and v["disk_free"] > 0
    m = eng.MODELS[0]
    (tmp_path / "models" / m.file).write_text("gguf")       # as if downloaded and checked
    (tmp_path / "models" / (m.file + ".ok")).write_text(m.sha256)
    client.put("/api/credentials/tiny", json={"transport": "local", "model": m.id, "base_url": "builtin"})
    row = client.get("/api/engine").json()["models"][0]
    assert row["installed"] and row["in_room"] == "tiny"
    assert "PRISM's engine" in client.get("/api/platforms").json()["catalog"]["tiny"]["label"]
    assert client.put("/api/engine", json={"build": "vulkan"}).json()["engine"]["build"] == "vulkan"
    assert client.put("/api/engine", json={"build": "cuda"}).status_code == 400
    assert client.post("/api/engine/models/nope/download").status_code == 404
    assert not client.delete(f"/api/engine/models/{m.id}").json()["models"][0]["installed"]
    assert not (tmp_path / "models" / m.file).exists()
