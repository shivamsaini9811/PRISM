"""End-to-end through the HTTP API with fake providers (no network)."""

import time

import pytest
from fastapi.testclient import TestClient

from prism import app as app_mod
from test_meeting import Fake


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISM_HOME", str(tmp_path))
    monkeypatch.setenv("PRISM_DB_PATH", str(tmp_path / "t.db"))
    for fn in (app_mod.settings, app_mod.store, app_mod.browsers):
        fn.cache_clear()
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
