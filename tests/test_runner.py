"""Running programs for bots: the RUN syntax, running safely, and approval."""

import asyncio
import os
import sys

import pytest

from prism.runner import Request, Runner, execute, parse_request, result_for_bot

PY = {"python": sys.executable}


def test_parse_run_requests():
    r = parse_request("Let me check.\nRUN @python\n```python\nprint(6 * 7)\n```", "claude")
    assert (r.tool, r.lang, r.code, r.who) == ("python", "python", "print(6 * 7)", "claude")
    r = parse_request("I'll look at the log.\n**RUN:** `git log --oneline -3`", "gpt")
    assert (r.tool, r.command) == ("git", "git log --oneline -3")
    assert parse_request("RUN @python\n```python\nx\n```\nand then I explain more", "a") is None  # not at the end
    assert parse_request("To run it, use RUN @python like this.", "a") is None
    assert parse_request("```python\nprint(1)\n```", "a") is None  # code alone is not a request


def test_execute_runs_in_the_work_folder_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    code = ("import os, pathlib\nprint(os.environ.get('ANTHROPIC_API_KEY'))\nprint(pathlib.Path.cwd().name)\n"
            "pathlib.Path('out.csv').write_text('a,b')\n")
    req = execute(Request("bot", "python", code=code, lang="python"), tmp_path / "work", PY, 30)
    assert req.status == "done" and req.exit_code == 0
    assert req.output.splitlines()[:2] == ["None", "work"] and req.files == ["out.csv"]
    assert "[ran python" in result_for_bot(req) and "out.csv" in result_for_bot(req)

    slow = execute(Request("bot", "python", code="import time\ntime.sleep(10)", lang="python"), tmp_path / "w2", PY, 1)
    assert slow.status == "failed" and "time limit" in slow.output
    big = execute(Request("bot", "python", code="print('x' * 50000)", lang="python"), tmp_path / "w3", PY, 30)
    assert "characters left out" in big.output and len(big.output) < 21_000
    missing = execute(Request("bot", "cobol", code="x", lang="cobol"), tmp_path / "w4", PY, 5)
    assert missing.status == "failed" and "no cobol" in missing.output
    err = execute(Request("bot", "python", code="raise SystemExit('boom')", lang="python"), tmp_path / "w5", PY, 30)
    assert err.exit_code == 1 and "boom" in err.output


async def test_approval_rules_and_cancel(tmp_path):
    rules = {}
    cards = []
    r = Runner(tmp_path, rules=lambda: rules, save_rules=lambda x: rules.update(x), toolchains=lambda: PY, timeout=30)
    emit = lambda req: cards.append(dict(req.card())) or len(cards)
    update = lambda req, mid: cards.__setitem__(mid - 1, dict(req.card()))

    def code():
        return Request("claude", "python", code="print('hi')", lang="python")

    task = asyncio.create_task(r.request(code(), 7, emit, update))
    await asyncio.sleep(0.05)
    assert cards[0]["status"] == "pending"
    assert r.decide(cards[0]["id"], "chat", chat_id=7)
    done = await task
    assert done.status == "done" and done.output.strip() == "hi" and done.approved_by == "chat"
    assert cards[0]["status"] == "done" and rules == {"chats": {"7": ["python"]}}

    done = await r.request(code(), 7, emit, update)  # allowed in this chat now: no card wait
    assert done.status == "done" and cards[1]["status"] == "done" and done.approved_by == "chat"

    task = asyncio.create_task(r.request(code(), 8, emit, update))  # another chat asks again
    await asyncio.sleep(0.05)
    r.decide(cards[2]["id"], None)
    assert (await task).status == "denied" and "did not allow" in result_for_bot(task.result())

    task = asyncio.create_task(r.request(code(), 9, emit, update))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cards[3]["status"] == "cancelled" and not r.decide(cards[3]["id"], "once")
