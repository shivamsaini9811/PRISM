"""Local models against a fake Ollama, and the job relay for models running in the page."""

import asyncio
import json
import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from prism import app as app_mod, local
from test_app import client, wait_meeting  # noqa: F401  (fixture)

REAL_MAKE_PROVIDER = app_mod.make_provider  # before the client fixture swaps in fakes
fake = FastAPI()
seen: list[dict] = []


@fake.get("/api/tags")
def tags():
    return {"models": [{"name": "qwen2.5:7b", "size": 4_700_000_000}]}


@fake.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    seen.append(body)
    words = f"local answer to {body['messages'][-1]['content'][-20:]}".split(" ")

    async def chunks():  # Ollama streams one JSON object per line
        for i, w in enumerate(words):
            yield json.dumps({"message": {"role": "assistant", "content": (" " if i else "") + w}, "done": False}) + "\n"
        yield json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}) + "\n"
    return StreamingResponse(chunks(), media_type="application/x-ndjson")


@fake.post("/api/pull")
async def pull(request: Request):
    name = (await request.json())["model"]

    async def events():
        yield json.dumps({"status": "pulling manifest"}) + "\n"
        for done in (50, 100):
            yield json.dumps({"status": f"downloading {name}", "total": 100, "completed": done}) + "\n"
        yield json.dumps({"status": "success"}) + "\n"
    return StreamingResponse(events(), media_type="application/x-ndjson")


@pytest.fixture(scope="module")
def ollama():
    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
    server = uvicorn.Server(uvicorn.Config(fake, port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    old = dict(local.RUNTIMES["ollama"])
    local.RUNTIMES["ollama"]["url"] = f"http://127.0.0.1:{port}"
    local.RUNTIMES["lmstudio"]["url"] = "http://127.0.0.1:9"  # nothing there
    yield f"http://127.0.0.1:{port}"
    local.RUNTIMES["ollama"].update(old)
    server.should_exit = True


def test_fit_keeps_instructions_and_the_newest_messages():
    msgs = [{"role": "user", "content": "old " * 3000}, {"role": "assistant", "content": "mid"},
            {"role": "user", "content": "newest"}]
    out = local.fit(msgs, "SYSTEM", 2000)
    assert out[0] == {"role": "system", "content": "SYSTEM"}
    assert [m["content"] for m in out[1:]] == ["mid", "newest"]
    huge = local.fit([{"role": "user", "content": "x" * 50_000 + "END"}], None, 2000)
    assert huge[0]["content"].endswith("END") and len(huge[0]["content"]) < 3000


def test_ollama_is_found_downloads_and_answers_in_the_room(client, ollama, monkeypatch):
    client.post("/api/unlock", json={"password": "longenough"})
    r = client.get("/api/local").json()
    rt = {x["id"]: x for x in r["runtimes"]}
    assert rt["ollama"]["running"] and rt["ollama"]["models"] == [{"name": "qwen2.5:7b", "size_mb": 4482}]
    assert not rt["lmstudio"]["running"] and rt["lmstudio"]["install"].startswith("https://")

    client.post("/api/local/pull", json={"model": "gemma3:4b"})
    for _ in range(100):
        p = client.get("/api/local").json()["pulls"]["gemma3:4b"]
        if p["done"]:
            break
        time.sleep(0.05)
    assert p["done"] and not p["error"] and p["status"] == "success"

    # A local member, under its own name, next to the real fakes of the other tests.
    assert client.put("/api/credentials/claude", json={"transport": "local", "model": "x"}).status_code == 400
    assert client.put("/api/credentials/all", json={"transport": "local", "model": "x"}).status_code == 400
    r = client.put("/api/credentials/qwen", json={"transport": "local", "model": "qwen2.5:7b", "base_url": "ollama"})
    assert r.status_code == 200 and r.json()["catalog"]["qwen"]["local"] == "local"
    monkeypatch.setattr(app_mod, "make_provider", REAL_MAKE_PROVIDER)
    chat = client.post("/api/chats").json()["id"]
    client.post(f"/api/chats/{chat}/messages", json={"content": "@qwen say hello", "format": "chat"})
    v = wait_meeting(client, chat)
    said = v["messages"][-1]
    assert said["meta"]["speaker"] == "qwen" and not said["meta"]["error"], said
    assert said["content"].startswith("local answer to") and said["meta"]["model"] == "qwen2.5:7b"


async def test_ollama_provider_sets_the_window_and_trims(ollama):
    p = local.OllamaProvider("qwen", "qwen2.5:7b")
    seen.clear()
    reply = await p.complete([{"role": "user", "content": "a" * 60_000}, {"role": "user", "content": "hello"}], "SYS")
    assert reply == "local answer to hello"
    body = seen[0]
    assert body["options"]["num_ctx"] == local.OLLAMA_CTX and body["stream"] is True
    assert p.last_activity is not None  # it counted as working while the reply streamed
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert [m["content"] for m in body["messages"][1:]] == ["hello"]  # the huge one didn't fit


async def test_missing_runtime_says_how_to_fix_it():
    p = local.OllamaProvider("qwen", "qwen2.5:7b", "http://127.0.0.1:9")
    with pytest.raises(local.ProviderError, match="Ollama isn't running"):
        await p.complete([{"role": "user", "content": "hi"}])


async def test_in_browser_model_turns_go_to_the_page_that_has_it():
    jobs = local.ClientJobs()
    turn = asyncio.create_task(jobs.run("Llama-3.2-3B", [{"role": "user", "content": "hi"}], timeout=10))
    await asyncio.sleep(0.05)
    assert jobs.take(["other-model"]) == []
    taken = jobs.take(["Llama-3.2-3B"])
    assert [j["messages"] for j in taken] == [[{"role": "user", "content": "hi"}]]
    assert jobs.take(["Llama-3.2-3B"]) == []  # once only
    assert jobs.finish(taken[0]["id"], "hello from the tab", None)
    assert not jobs.finish(taken[0]["id"], "second tab", None)  # first answer wins
    assert await turn == "hello from the tab"


async def test_in_browser_model_with_no_page_fails_clearly(monkeypatch):
    monkeypatch.setattr(local, "CLAIM_TIMEOUT_S", 0.2)
    with pytest.raises(local.ProviderError, match="no PRISM page is running"):
        await local.ClientJobs().run("m", [{"role": "user", "content": "hi"}], timeout=5)
