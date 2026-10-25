"""Running programs on the user's computer, for bots and flows - with the user's approval.

A bot ends its reply with one of:

    RUN @python                 RUN: git log --oneline -5
    ```python
    print(2 + 2)
    ```

PRISM shows an approval card with the exact code or command, the folder it runs in and the
tool. The user runs it once, allows that tool for this chat (or always), or refuses. It runs
in the chat's own work folder under PRISM's data folder, with a time limit, a cap on output
and none of PRISM's saved keys in its environment; what it printed (and the files it made)
goes back to the bot. Every run is logged with the chat's messages.

This is not a sandbox: an approved program can do anything the user's account can. The card
says so. Tools come from the catalog (installed toolchains, callable by name) or this
computer's PATH.
"""

import asyncio
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
OUTPUT_MAX = 20_000
FILES_MAX = 5                       # files a run made, handed back as attachments
FILE_BYTES_MAX = 10 * 1024 * 1024
DEFAULT_TIMEOUT_S = 300.0

RUN_LINE = re.compile(r"^[\s*_`>-]*RUN\s*(?:@([\w.+-]+))?[*_`]*\s*(?::[\s*_`]*(.*?))?[\s*_`]*$", re.I)
FENCE = re.compile(r"```([\w.+-]*)[^\n]*\n(.*?)\n?```\s*$", re.S)
# Code blocks: how to run each language (the tool is looked up by its first name found).
LANGS = {
    "python": (["python", "python3", "py"], ".py", []),
    "py": (["python", "python3", "py"], ".py", []),
    "node": (["node"], ".js", []), "javascript": (["node"], ".js", []), "js": (["node"], ".js", []),
    "powershell": (["pwsh", "powershell"], ".ps1", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]),
    "ps1": (["pwsh", "powershell"], ".ps1", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]),
    "bash": (["bash"], ".sh", []), "sh": (["sh", "bash"], ".sh", []),
    "cmd": (["cmd"], ".bat", ["/c"]), "bat": (["cmd"], ".bat", ["/c"]),
}
# Environment passed on: what programs need to start, nothing PRISM keeps (keys, tokens).
KEEP_ENV = {"PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP",
            "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA",
            "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA", "LANG", "LC_ALL", "TERM",
            "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS", "USER", "USERNAME", "SHELL", "TMPDIR"}

RUN_HELP = (" You can run programs on the user's computer - each run needs the user's approval: "
            "end your reply with a line `RUN @<language>` followed by one fenced code block "
            "(languages here: {langs}), or a line `RUN: <command>` for a command line; nothing "
            "after it. It runs in a work folder of its own; what it prints comes back to you, "
            "then finish your reply. Use it to compute, test code or process files rather than "
            "guessing. At most {max_runs} runs per turn.")


@dataclass
class Request:
    who: str                         # the bot or agent asking
    tool: str                        # python, node, git, ...
    code: str | None = None          # a code block, or
    command: str | None = None       # a command line
    lang: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "pending"          # pending | running | done | denied | failed | cancelled | expired
    output: str = ""
    exit_code: int | None = None
    seconds: float = 0.0
    files: list[str] = field(default_factory=list)
    approved_by: str = ""            # once | chat | flow | always | rule

    def card(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def parse_request(text: str, who: str) -> Request | None:
    """A RUN at the end of a reply: `RUN @lang` + one fenced block, or `RUN: command`."""
    body = text.rstrip()
    fence = FENCE.search(body)
    lines = body.splitlines()
    if fence:  # the block must be the last thing, right after a RUN line
        before = body[:fence.start()].rstrip().splitlines()
        if before and (m := RUN_LINE.match(before[-1])) and not m.group(2):
            lang = (m.group(1) or fence.group(1) or "").lower()
            code = fence.group(2)
            if lang and code.strip():
                return Request(who, lang, code=code, lang=lang)
    if lines and (m := RUN_LINE.match(lines[-1])) and m.group(2):
        command = m.group(2).strip().strip("`")
        if command:
            return Request(who, command.split()[0].strip('"').lower(), command=command)
    return None


def find_tool(name: str, toolchains: dict[str, str]) -> str | None:
    """A catalog toolchain's command, else a program on PATH."""
    return toolchains.get(name) or shutil.which(name)


def argv_for(req: Request, work: Path, toolchains: dict[str, str]) -> list[str]:
    if req.code is not None:
        names, ext, pre = LANGS.get(req.lang, ([req.lang], "", []))
        exe = next((p for n in names if (p := find_tool(n, toolchains))), None)
        if not exe:
            raise RunError(f"no {req.lang} on this computer - install it, or get it from the catalog (Settings)")
        script = work / f"prism_run_{req.id}{ext}"
        script.write_text(req.code, encoding="utf-8")
        return [exe, *pre, str(script)]
    first, *rest = _split(req.command)
    exe = find_tool(first, toolchains) or (first if Path(first).is_absolute() and Path(first).exists() else None)
    if not exe:
        raise RunError(f"{first} isn't installed here (not in the catalog's tools, not on PATH)")
    if sys.platform == "win32" and exe.lower().endswith((".bat", ".cmd")):
        return ["cmd", "/c", exe, *rest]
    return [exe, *rest]


def _split(command: str) -> list[str]:
    if sys.platform == "win32":
        return [p.strip('"') for p in re.findall(r'"[^"]*"|\S+', command)]
    import shlex
    return shlex.split(command)


class RunError(Exception):
    pass


def environment(toolchains: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k.upper() in KEEP_ENV}
    dirs = list(dict.fromkeys(str(Path(p).parent) for p in toolchains.values()))
    env["PATH"] = os.pathsep.join(dirs + [env.get("PATH", "")])
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def execute(req: Request, work: Path, toolchains: dict[str, str], timeout: float) -> Request:
    """Run it (blocking - call in a thread): output capped, files it made noted."""
    work.mkdir(parents=True, exist_ok=True)
    before = {p: p.stat().st_mtime for p in work.rglob("*") if p.is_file()}
    start = time.monotonic()
    try:
        argv = argv_for(req, work, toolchains)
        p = subprocess.run(argv, cwd=work, env=environment(toolchains), capture_output=True,
                           timeout=timeout, creationflags=NO_WINDOW, stdin=subprocess.DEVNULL)
        out = p.stdout.decode("utf-8", errors="replace")
        err = p.stderr.decode("utf-8", errors="replace")
        req.exit_code = p.returncode
        req.output = out + (("\n[stderr]\n" + err) if err.strip() else "")
        req.status = "done"
    except subprocess.TimeoutExpired as e:
        req.output = ((e.stdout or b"").decode("utf-8", "replace") + f"\n[stopped after {round(timeout)} s - the time limit]")
        req.status, req.exit_code = "failed", None
    except (RunError, OSError) as e:
        req.output, req.status = str(e), "failed"
    req.seconds = round(time.monotonic() - start, 1)
    if len(req.output) > OUTPUT_MAX:
        req.output = req.output[:OUTPUT_MAX // 2] + f"\n[… {len(req.output) - OUTPUT_MAX} characters left out …]\n" + req.output[-OUTPUT_MAX // 2:]
    made = [p for p in work.rglob("*") if p.is_file() and before.get(p) != p.stat().st_mtime
            and not p.name.startswith("prism_run_")]
    req.files = [str(p.relative_to(work)).replace("\\", "/") for p in made][:20]
    return req


def result_for_bot(req: Request) -> str:
    if req.status == "denied":
        return "[The user did not allow this run. Don't try it again; finish your reply without it.]"
    if req.status in ("cancelled", "expired"):
        return f"[The run was {req.status}. Finish your reply without it.]"
    head = (f"[ran {req.tool} in {req.seconds}s, exit code {req.exit_code}]" if req.status == "done"
            else f"[{req.tool} failed]")
    files = f"\n[files it made: {', '.join(req.files)}]" if req.files else ""
    return f"{head}\n{req.output.strip() or '(no output)'}{files}\n\nNow finish your reply (you may RUN again if you must)."


class Runner:
    """Pending approvals, allow rules, and the runs themselves.

    rules: {"always": [tool], "chats": {chat_id: [tool]}, "flows": {flow: [tool]}} kept by the
    app (meta `run_rules`); emit(req, chat_id) persists the card and returns its message id;
    update(req, message_id) changes it."""

    def __init__(self, work_root: Path, rules: Callable[[], dict], save_rules: Callable[[dict], None],
                 toolchains: Callable[[], dict[str, str]] = dict, timeout: float = DEFAULT_TIMEOUT_S):
        self.work_root, self.rules, self.save_rules = work_root, rules, save_rules
        self.toolchains, self.timeout = toolchains, timeout
        self.pending: dict[str, tuple[Request, asyncio.Future, int | None]] = {}

    def work_dir(self, scope: str) -> Path:
        return self.work_root / re.sub(r"[^\w-]", "_", scope)

    def allowed(self, tool: str, chat_id: int | None, flow: str | None) -> str | None:
        r = self.rules() or {}
        if tool in r.get("always", []):
            return "always"
        if chat_id is not None and tool in r.get("chats", {}).get(str(chat_id), []):
            return "chat"
        if flow and tool in r.get("flows", {}).get(flow, []):
            return "flow"
        return None

    async def request(self, req: Request, chat_id: int, emit: Callable, update: Callable,
                      flow: str | None = None) -> Request:
        """Ask the user (unless a rule already allows this tool), then run it."""
        rule = self.allowed(req.tool, chat_id, flow)
        req.status = "running" if rule else "pending"
        message_id = emit(req)
        if not rule:
            fut = asyncio.get_running_loop().create_future()
            self.pending[req.id] = (req, fut, message_id)
            try:
                decision = await fut
            except asyncio.CancelledError:
                req.status = "cancelled"
                update(req, message_id)
                raise
            finally:
                self.pending.pop(req.id, None)
            if decision is None:
                req.status = "denied"
                update(req, message_id)
                return req
            rule = decision
            req.status = "running"
            update(req, message_id)
        req.approved_by = rule
        await asyncio.to_thread(execute, req, self.work_dir(f"chat-{chat_id}"), self.toolchains(), self.timeout)
        update(req, message_id)
        return req

    def decide(self, request_id: str, allow: str | None, chat_id: int | None = None, flow: str | None = None) -> bool:
        """allow: once | chat | flow | always, or None to refuse. Chat/flow/always add a rule."""
        item = self.pending.get(request_id)
        if not item or item[1].done():
            return False
        req, fut, _ = item
        if allow in ("chat", "flow", "always"):
            r = dict(self.rules() or {})
            if allow == "always":
                r["always"] = sorted(set(r.get("always", [])) | {req.tool})
            elif allow == "chat" and chat_id is not None:
                chats = dict(r.get("chats", {}))
                chats[str(chat_id)] = sorted(set(chats.get(str(chat_id), [])) | {req.tool})
                r["chats"] = chats
            elif allow == "flow" and flow:
                fl = dict(r.get("flows", {}))
                fl[flow] = sorted(set(fl.get(flow, [])) | {req.tool})
                r["flows"] = fl
            self.save_rules(r)
        fut.set_result(allow)
        return True

    def languages(self) -> list[str]:
        tc = self.toolchains()
        seen = []
        for lang, (names, _, _) in LANGS.items():
            if lang in ("py", "js", "javascript", "ps1", "bat", "sh") or lang in seen:
                continue
            if any(find_tool(n, tc) for n in names):
                seen.append(lang)
        return seen
