"""Building with words: agents, flows, runs and knowledge from a chat message."""

import json
import time

from prism import app as app_mod, builder
from test_app import client, wait_run  # noqa: F401 - the app fixture

FLOW_YAML = """name: brief
inputs: {topic: text}
nodes:
  write: {agent: reviewer, prompt: "Write about {{input.topic}}."}
"""


def test_which_messages_are_builder_requests():
    for yes in ["create an agent that reviews code", "Please make a workflow for my weekly report",
                "can you build a flow that drafts and reviews", "run the flow brief on solar",
                "remember that Priya leads design", "add to my knowledge graph: launch is 12 May"]:
        assert builder.is_request(yes), yes
    for no in ["what is an agent?", "make a plan for my week", "write a poem about flows of rivers",
               "I remember the old version"]:
        assert not builder.is_request(no), no
    assert builder.parse_actions('```json\n{"actions": [{"action": "explain", "answer": "hi"}]}\n```') == [
        {"action": "explain", "answer": "hi"}]


def wait_card(client, chat):
    for _ in range(300):
        v = client.get(f"/api/chats/{chat}").json()
        card = [m for m in v["messages"] if m["meta"]["kind"] == "build"][-1]
        if not card["meta"].get("pending"):
            return card["meta"]
        time.sleep(0.02)
    raise AssertionError("the request was not handled")


def test_build_agents_flows_runs_and_knowledge_from_chat(client, monkeypatch):
    client.post("/api/unlock", json={"password": "longenough"})
    client.put("/api/credentials/claude", json={"secret": "k"})
    client.put("/api/primary", json={"platform": "claude"})
    asked = []

    class Model:
        name, model, thread_url = "claude", "claude-fake", None

        async def complete(self, messages, system=None):
            last = messages[-1]["content"]
            asked.append(last)
            if "into actions for PRISM" in last:
                req = last.split("The request:\n", 1)[1].split("\n\nReply with", 1)[0]
                if "reviewer" in req:
                    return json.dumps({"actions": [
                        {"action": "create_agent", "name": "Code Reviewer", "model": "claude",
                         "instructions": "Review code harshly.", "tools": ["run", "shell"],
                         "output": {"verdict": "PASS|FIX"}},
                        {"action": "create_flow", "description": "write a brief on a topic with the reviewer"}]})
                if "run" in req:
                    return '```json\n{"actions": [{"action": "run_flow", "flow": "brief", "inputs": {"topic": "solar"}}]}\n```'
                if "remember" in req:
                    return '{"actions": [{"action": "remember", "text": "Priya leads design."}]}'
                return '{"actions": [{"action": "recall", "question": "Priya"}]}'
            if "Write a workflow for PRISM" in last:
                return "```yaml\n" + FLOW_YAML.replace("reviewer", "code-reviewer") + "```"
            if "--- MATERIAL" in last:
                return ('```json\n{"entities": [{"name": "Priya", "type": "person"}, {"name": "Design", "type": "topic"}],'
                        ' "links": [{"src": "Priya", "rel": "leads", "dst": "Design", "quote": "Priya leads design"}]}\n```')
            return "done: " + last[-40:]
    monkeypatch.setattr(app_mod, "make_provider", lambda cred, **kw: Model())
    chat = client.post("/api/chats").json()["id"]

    client.post(f"/api/chats/{chat}/messages", json={"content": "@prism make a code reviewer agent and a flow that uses it"})
    card = wait_card(client, chat)
    agent, flow = card["results"]
    assert agent["ok"] and agent["agent"] == "code-reviewer" and "@code-reviewer" in agent["summary"]
    assert client.get("/api/agents").json()["agents"][0] == {
        "name": "code-reviewer", "model": "@claude", "instructions": "Review code harshly.", "context": [],
        "tools": ["run"], "output": {"verdict": "PASS|FIX"}, "research": False}
    assert flow["ok"] and flow["flow"] == "brief" and flow["inputs"] == ["topic"]
    assert "code-reviewer (@claude)" in asked[1]           # the flow was drafted knowing the new agent
    v = client.get(f"/api/chats/{chat}").json()
    assert v["messages"][0]["meta"]["format"] == "build" and v["chat"]["title"].startswith("@prism make")

    # No @prism needed for an unmistakable request; it starts the run in its own chat.
    client.post(f"/api/chats/{chat}/messages", json={"content": "run the flow brief on solar"})
    run = wait_card(client, chat)["results"][0]
    assert run["ok"] and run["chat"] != chat
    assert wait_run(client, run["chat"])["run"]["status"] == "done"

    client.post(f"/api/chats/{chat}/messages", json={"content": "remember that Priya leads design"})
    assert wait_card(client, chat)["results"][0]["summary"].startswith("Remembered it in the knowledge graph: 2 things, 1 links")
    client.post(f"/api/chats/{chat}/messages", json={"content": "@prism what do we know about Priya?"})
    assert "Priya -leads-> Design" in wait_card(client, chat)["results"][0]["summary"]

    # A message to a bot is not a builder request.
    asked.clear()
    client.post(f"/api/chats/{chat}/messages", json={"content": "@claude make a plan for my week", "format": "chat"})
    time.sleep(0.3)
    assert not any("into actions for PRISM" in a for a in asked)
