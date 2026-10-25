"""The knowledge graph: extraction parsing, merging names, evidence, and using it."""

import json
import time

import pytest

from prism import app as app_mod
from prism.db import Store
from prism.kg import ExtractError, Graph, norm, parse_extraction
from test_app import client, wait_meeting  # noqa: F401 - the app fixture

EXTRACTED = {
    "entities": [{"name": "Priya Shah", "type": "person", "aliases": ["Priya"], "summary": "leads design"},
                 {"name": "Launch website", "type": "project", "aliases": [], "summary": ""},
                 {"name": "Omar", "type": "person"}],
    "links": [{"src": "Priya", "rel": "Leads", "dst": "Launch website", "quote": "Priya leads the site"},
              {"src": "Omar", "rel": "runs", "dst": "Infrastructure", "quote": ""}],
}


def test_parse_extraction():
    out = parse_extraction("Here you go:\n```json\n" + json.dumps(EXTRACTED) + "\n```")
    assert [e["name"] for e in out["entities"]] == ["Priya Shah", "Launch website", "Omar"]
    assert out["links"][0] == {"src": "Priya", "rel": "leads", "dst": "Launch website", "quote": "Priya leads the site"}
    assert parse_extraction('{"entities": [{"name": "X", "type": "weird"}], "links": [{"src": "X"}]}') == {
        "entities": [{"name": "X", "type": "topic", "aliases": [], "summary": ""}], "links": []}
    for bad in ["no json here", "```json\n[1, 2]\n```", '{"entities": "x"}']:
        with pytest.raises(ExtractError):
            parse_extraction(bad)
    assert norm("The Launch-Website!") == "launch-website"


def test_graph_merges_names_and_keeps_evidence(tmp_path):
    g = Graph(Store(tmp_path / "t.db"))
    got = g.add(parse_extraction(json.dumps(EXTRACTED)), {"chat": 1, "message": 5, "where": 'chat "Plan"'})
    assert got == {"entities": 3, "links": 2} and g.counts() == {"entities": 4, "links": 2}
    priya = g.find("priya")  # by alias
    assert priya["name"] == "Priya Shah" and priya["summary"] == "leads design"
    # Said again elsewhere: the same link, with one more piece of evidence.
    g.link("Priya Shah", "leads", "launch website", {"chat": 2, "message": 9, "where": 'chat "Standup"'})
    links = g.links_of(priya["id"])
    assert len(links) == 1 and len(links[0]["evidence"]) == 2 and links[0]["confidence"] > 0.8
    text = g.describe(g.mentioned("What is Priya doing on the launch website?"))
    assert "Priya Shah (person) - leads design" in text and '-leads-> Launch website "Priya leads the site"' in text
    assert g.mentioned("nothing relevant") == []
    assert "Priya Shah" in g.answer("who leads the website?") or "Launch website" in g.answer("who leads the website?")

    # Two entries for one thing: renaming one to the other merges them.
    o = g.upsert("O. Haddad", "person")
    g.link("O. Haddad", "on call", "Infrastructure", {"where": "x"})
    g.edit(o, name="Omar")
    omar = g.find("Omar")
    assert "O. Haddad" in omar["aliases"] and {l["rel"] for l in g.links_of(omar["id"])} == {"runs", "on call"}
    g.delete_link(g.links_of(omar["id"])[0]["id"])
    g.delete_entity(priya["id"])
    assert g.find("Priya") is None and g.counts()["links"] == 1


def test_chat_extraction_kg_context_and_ask(client, monkeypatch):
    client.post("/api/unlock", json={"password": "longenough"})
    client.put("/api/credentials/gpt", json={"secret": "k"})
    client.put("/api/primary", json={"platform": "gpt"})
    prompts = []

    bad_once = [True]

    class Reader:
        name, model, thread_url = "gpt", "gpt-fake", None

        async def complete(self, messages, system=None):
            prompts.append(messages[-1]["content"])
            extracting = "--- MATERIAL" in messages[-1]["content"]
            if extracting and bad_once and bad_once.pop():
                return "Sure! Priya leads it."           # not JSON: asked again
            if extracting or "wasn't usable" in messages[-1]["content"]:
                return "```json\n" + json.dumps(EXTRACTED) + "\n```"
            if "ASK @kg" in (system or "") and "[kg answered]" not in messages[-1]["content"] and "kg answered" not in str(messages):
                return "ASK @kg: Priya"
            return "noted: " + messages[-1]["content"][-200:]
    monkeypatch.setattr(app_mod, "make_provider", lambda cred, **kw: Reader())
    chat = client.post("/api/chats").json()["id"]
    client.post(f"/api/chats/{chat}/messages", json={"content": "@gpt Priya leads the launch website", "format": "chat"})
    wait_meeting(client, chat)
    assert client.post("/api/kg/extract", json={}).status_code == 400
    prompts.clear()
    client.post("/api/kg/extract", json={"chat_id": chat})
    for _ in range(200):
        v = client.get("/api/kg").json()
        if not v["job"].get("running") and "result" in v["job"]:
            break
        time.sleep(0.02)
    assert v["job"]["result"]["failed"] == 0 and v["entities"] and v["links"] == 2
    assert "[the user] @gpt Priya leads the launch website" in prompts[0] and "wasn't usable" in prompts[1]
    priya = next(e for e in v["entities"] if e["name"] == "Priya Shah")
    detail = client.get(f"/api/kg/{priya['id']}").json()
    assert detail["links"][0]["evidence"][0]["chat"] == chat and detail["links"][0]["evidence"][0]["chat_title"]

    # #kg gives the room what is known; bots can ASK @kg.
    prompts.clear()
    client.post(f"/api/chats/{chat}/messages", json={"content": "@gpt what does Priya do? #kg", "format": "chat"})
    v = wait_meeting(client, chat)
    ctx = [m for m in v["messages"] if m["meta"]["kind"] == "context"][-1]
    assert ctx["meta"]["name"] == "kg" and "Priya Shah -leads-> Launch website" in ctx["content"]
    lookup = [m for m in v["messages"] if m["meta"]["kind"] == "lookup"]
    assert lookup and lookup[-1]["meta"]["speaker"] == "kg" and "Priya Shah" in lookup[-1]["content"]

    # Correct it by hand: rename merges into the existing entry, delete removes.
    r = client.put(f"/api/kg/{priya['id']}", json={"summary": "design lead"}).json()
    assert r["summary"] == "design lead"
    assert client.put("/api/kg-settings", json={"mode": "auto"}).json()["mode"] == "auto"
    client.delete(f"/api/kg/{priya['id']}")
    assert client.get(f"/api/kg/{priya['id']}").status_code == 404
