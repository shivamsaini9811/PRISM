"""Running Python for bots: the RUN syntax, the sandbox, pip, and approval."""

import asyncio
import os
import subprocess
import sys

import pytest

from conftest import make_wheel
from prism.runner import Request, Runner, execute, packages_in, parse_request, result_for_bot


def test_parse_run_requests():
    r = parse_request("Let me check.\nRUN @python\n```python\n# pip: rapidfuzz pandas>=2\nprint(6 * 7)\n```", "claude")
    assert (r.who, r.code.splitlines()[-1], r.packages) == ("claude", "print(6 * 7)", ["rapidfuzz", "pandas>=2"])
    assert parse_request("**RUN @python**\n```\nprint(1)\n```", "a").code == "print(1)"
    assert parse_request("RUN @python\n\nPython\nCopy\n```\nprint(2)\n```", "grok").code == "print(2)"
    assert parse_request("RUN: git log --oneline -3", "gpt") is None          # Python only now
    assert parse_request("RUN @node\n```js\nconsole.log(1)\n```", "a") is None
    assert parse_request("RUN @python\n```python\nx\n```\nand then I explain more", "a") is None  # not at the end
    assert parse_request("```python\nprint(1)\n```", "a") is None            # code alone is not a request
    assert packages_in("# pip: ok==1.0 -e . https://evil/x.whl --index-url=x ok2[extra]") == (
        ["ok==1.0", "ok2[extra]"], ["-e", ".", "https://evil/x.whl", "--index-url=x"])


def test_runs_in_the_sandbox_without_secrets(shared_sandbox, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    code = ("import os, sys, pathlib\nprint(os.environ.get('ANTHROPIC_API_KEY'))\nprint(pathlib.Path.cwd().name)\n"
            "print(sys.prefix)\npathlib.Path('out.csv').write_text('a,b')\n")
    req = execute(Request("bot", code), tmp_path / "work", shared_sandbox, 30)
    lines = req.output.splitlines()
    assert req.status == "done" and req.exit_code == 0 and lines[:2] == ["None", "work"]
    assert lines[2] == str(shared_sandbox.path) and req.files == ["out.csv"]  # inside the sandbox's Python
    assert "[ran python" in result_for_bot(req) and "out.csv" in result_for_bot(req)

    slow = execute(Request("bot", "import time\ntime.sleep(10)"), tmp_path / "w2", shared_sandbox, 1)
    assert slow.status == "failed" and "time limit" in slow.output
    big = execute(Request("bot", "print('x' * 50000)"), tmp_path / "w3", shared_sandbox, 30)
    assert "characters left out" in big.output and len(big.output) < 21_000
    err = execute(Request("bot", "raise SystemExit('boom')"), tmp_path / "w4", shared_sandbox, 30)
    assert err.exit_code == 1 and "boom" in err.output


def test_pip_installs_into_the_sandbox_only(shared_sandbox, tmp_path):
    wheels = tmp_path / "wheels"
    make_wheel(wheels, "prismtestpkg")
    assert shared_sandbox.missing(["prismtestpkg", "pip"]) == ["prismtestpkg"]
    shared_sandbox.install(["prismtestpkg"], ["--no-index", "--find-links", str(wheels)])
    assert shared_sandbox.missing(["prismtestpkg"]) == []
    assert "prismtestpkg" in [p["name"] for p in shared_sandbox.installed()]
    req = execute(Request("bot", "import prismtestpkg\nprint(prismtestpkg.VALUE)"), tmp_path / "w", shared_sandbox, 30)
    assert req.output.strip() == "42"
    # The Python the sandbox is built on did not get it.
    base = subprocess.run([sys.executable, "-c", "import prismtestpkg"], capture_output=True)
    assert base.returncode != 0
    shared_sandbox.uninstall("prismtestpkg")
    assert shared_sandbox.missing(["prismtestpkg"]) == ["prismtestpkg"]


async def test_approval_rules_new_packages_still_ask_and_cancel(shared_sandbox, tmp_path):
    rules = {}
    cards = []
    r = Runner(tmp_path, rules=lambda: rules, save_rules=lambda x: rules.update(x), sandbox=shared_sandbox, timeout=30)
    emit = lambda req: cards.append(dict(req.card())) or len(cards)
    update = lambda req, mid: cards.__setitem__(mid - 1, dict(req.card()))

    task = asyncio.create_task(r.request(Request("claude", "print('hi')"), 7, emit, update))
    await asyncio.sleep(0.3)
    assert cards[0]["status"] == "pending"
    assert r.decide(cards[0]["id"], "chat", chat_id=7)
    done = await task
    assert done.status == "done" and done.output.strip() == "hi" and done.approved_by == "chat"
    assert cards[0]["status"] == "done" and rules == {"chats": {"7": ["python"]}}

    done = await r.request(Request("claude", "print(1)"), 7, emit, update)  # allowed here: no wait
    assert done.status == "done" and cards[1]["status"] == "done"

    # A package that isn't installed yet asks even here (installing runs its setup code).
    task = asyncio.create_task(r.request(Request("claude", "# pip: notinstalledpkg\nprint(2)",
                                                 packages=["notinstalledpkg"]), 7, emit, update))
    await asyncio.sleep(0.5)
    assert cards[2]["status"] == "pending" and cards[2]["install"] == ["notinstalledpkg"]
    r.decide(cards[2]["id"], None)
    assert (await task).status == "denied" and "did not allow" in result_for_bot(task.result())

    task = asyncio.create_task(r.request(Request("claude", "print(3)"), 9, emit, update))
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cards[3]["status"] == "cancelled" and not r.decide(cards[3]["id"], "once")


def test_packaged_runtime_is_unpacked_into_the_data_folder_once(tmp_path, monkeypatch):
    import io
    import json
    import tarfile
    from prism import runner
    app = tmp_path / "PRISM"
    app.mkdir()
    exe_name = "python.exe" if sys.platform == "win32" else "bin/python3"
    with tarfile.open(app / "runtime.tar.gz", "w:gz") as t:
        data = b"#!/bin/sh\necho fake\n"
        info = tarfile.TarInfo(exe_name)
        info.size, info.mode = len(data), 0o755
        t.addfile(info, io.BytesIO(data))
    (app / "runtime.json").write_text(json.dumps({"python": "3.12.15", "tag": "20261009"}))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(app / "PRISM.exe"))
    monkeypatch.setenv("PRISM_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("PRISM_PYTHON", raising=False)
    assert runner.runtime_version() == "3.12.15"
    assert not (tmp_path / "home" / "python").exists()          # the version needs no unpacking
    exe = runner.runtime_python()
    assert exe == str(tmp_path / "home" / "python" / "runtime-20261009" / exe_name)
    assert (tmp_path / "home" / "python" / "runtime-20261009" / exe_name).is_file()
    mtime = os.path.getmtime(exe)
    assert runner.runtime_python() == exe and os.path.getmtime(exe) == mtime  # once


async def test_stop_while_running_marks_the_card(shared_sandbox, tmp_path):
    cards = []
    rules = {"always": ["python"]}
    r = Runner(tmp_path, rules=lambda: rules, save_rules=lambda x: None, sandbox=shared_sandbox, timeout=30)
    emit = lambda req: cards.append(dict(req.card())) or len(cards)
    update = lambda req, mid: cards.__setitem__(mid - 1, dict(req.card()))
    task = asyncio.create_task(r.request(Request("claude", "import time\ntime.sleep(3)"), 1, emit, update))
    await asyncio.sleep(1)
    assert cards[0]["status"] == "running"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cards[0]["status"] == "cancelled"


def test_forgiving_reading_for_small_local_models():
    from prism.runner import parse_request
    after = 'Here:\n```python\nprint("Hello, world!")\n```\n\nRUN @python'      # RUN after the block
    inside = 'Now run it:\n```python\n# RUN @python\nprint(1)\n```'             # RUN inside the block
    bare = '```python\nprint("Hello, world!")\n```'                               # no RUN at all
    shown = 'Like this:\n```python\nprint(1)\n```\nThen call it from your app.'  # code shown, not run
    for text in (after, inside, bare, shown):
        assert parse_request(text, "q") is None                # the big models' strict format only
    assert parse_request(after, "q", forgiving=True).code == 'print("Hello, world!")'
    assert parse_request(inside, "q", forgiving=True).code == "print(1)"
    assert parse_request(bare, "q", forgiving=True) is None    # the user didn't ask to run
    assert parse_request(bare, "q", forgiving=True, asked=True).code == 'print("Hello, world!")'
    assert parse_request(shown, "q", forgiving=True, asked=True) is None  # text after it: an example
    assert parse_request("RUN @python\n```python\nprint(2)\n```", "q", forgiving=True).code == "print(2)"
