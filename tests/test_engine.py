"""PRISM's own model engine: starting llama-server, switching models, downloads."""

import asyncio
import hashlib
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response

from prism import engine as eng
from prism.providers.base import ProviderError

FAKE = Path(__file__).parent / "fakes" / "fake_llama_server.py"


@pytest.fixture
def fake_server(tmp_path, monkeypatch):
    launcher = tmp_path / ("llama-server.bat" if sys.platform == "win32" else "llama-server")
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
    launcher.chmod(0o755)
    monkeypatch.setenv("PRISM_LLAMA_SERVER", str(launcher))
    home = tmp_path / "home"
    d = eng.models_dir(home)
    d.mkdir(parents=True)
    for m in eng.MODELS[:2]:  # as if downloaded and checked
        (d / m.file).write_text("gguf")
        (d / (m.file + ".ok")).write_text(m.sha256)
    return home


async def test_starts_answers_switches_and_stops(fake_server):
    e = eng.Engine(fake_server)
    a, b = eng.MODELS[0], eng.MODELS[1]
    p = eng.BuiltinProvider("small", a.id, e)
    reply = await p.complete([{"role": "user", "content": "hello there"}], "SYS")
    assert reply.strip() == f"model {a.file} says hello there"  # the thinking is not in the reply
    first = e.proc.pid
    assert e.running() and e.model == a.id and e.busy == 0 and p.last_activity
    assert (await eng.BuiltinProvider("small", a.id, e).complete([{"role": "user", "content": "again"}])).startswith("model")
    assert e.proc.pid == first                         # same model: same server
    await eng.BuiltinProvider("other", b.id, e).complete([{"role": "user", "content": "hi"}])
    assert e.proc.pid != first and e.model == b.id     # another model: restarted with it
    with pytest.raises(ProviderError, match="isn't downloaded"):
        await e.ensure(eng.MODELS[3].id)
    await e.stop()
    assert not e.running()


async def test_a_switch_waits_for_an_answer_in_progress(fake_server):
    e = eng.Engine(fake_server)
    a, b = eng.MODELS[0], eng.MODELS[1]
    await e.ensure(a.id)                 # an answer in progress (busy)
    switching = asyncio.create_task(e.ensure(b.id))
    await asyncio.sleep(0.8)
    assert not switching.done() and e.model == a.id
    e.done()
    await switching
    assert e.model == b.id
    e.done()
    await e.stop()


async def test_crash_while_starting_is_reported(fake_server, tmp_path):
    m = eng.MODELS[2]
    d = eng.models_dir(fake_server)
    bad = eng.Model(**{**m.__dict__, "file": "crash.gguf"})
    (d / "crash.gguf").write_text("x")
    (d / "crash.gguf.ok").write_text("x")
    eng.BY_ID[bad.id] = bad
    try:
        e = eng.Engine(fake_server, build="vulkan")
        with pytest.raises(ProviderError, match="stopped while starting.*failed to load model.*CPU build"):
            await e.ensure(bad.id)
        assert not e.running()
    finally:
        eng.BY_ID[m.id] = m


# --- downloads -----------------------------------------------------------------------------------

fake_hub = FastAPI()
CONTENT = b"GGUF" + b"x" * 5000


@fake_hub.get("/model.gguf")
def model_file():
    return Response(CONTENT)


@pytest.fixture(scope="module")
def hub():
    srv = uvicorn.Server(uvicorn.Config(fake_hub, host="127.0.0.1", port=11496, log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield "http://127.0.0.1:11496/model.gguf"
    srv.should_exit = True


async def wait(model_id):
    for _ in range(200):
        if eng.downloads()[model_id]["done"]:
            return eng.downloads()[model_id]
        await asyncio.sleep(0.02)


async def test_download_checked_refused_when_wrong_and_when_disk_is_full(hub, tmp_path, monkeypatch):
    real = eng.MODELS[0]
    good = eng.Model(**{**real.__dict__, "id": "t-good", "file": "good.gguf", "sha256": hashlib.sha256(CONTENT).hexdigest(), "size": len(CONTENT)})
    bad = eng.Model(**{**real.__dict__, "id": "t-bad", "file": "bad.gguf", "size": len(CONTENT)})
    monkeypatch.setitem(eng.BY_ID, good.id, good)
    monkeypatch.setitem(eng.BY_ID, bad.id, bad)
    monkeypatch.setattr(eng, "MODELS", [good, bad])
    eng.start_download(good.id, tmp_path, url=hub)
    assert (await wait(good.id))["error"] is None and eng.installed(tmp_path) == {good.id}
    eng.start_download(bad.id, tmp_path, url=hub)
    assert "checksum" in (await wait(bad.id))["error"]
    assert not (eng.models_dir(tmp_path) / "bad.gguf").exists() and not (eng.models_dir(tmp_path) / "bad.gguf.part").exists()
    huge = eng.Model(**{**real.__dict__, "id": "t-huge", "size": 1 << 60})
    monkeypatch.setitem(eng.BY_ID, huge.id, huge)
    with pytest.raises(ProviderError, match="free some space"):
        eng.start_download(huge.id, tmp_path, url=hub)
    eng.remove(good.id, tmp_path)
    assert eng.installed(tmp_path) == set()
