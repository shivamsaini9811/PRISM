"""Running Python for bots and flows, on the user's computer, with the user's approval.

A bot ends its reply with:

    RUN @python
    ```python
    # pip: rapidfuzz            (optional: packages it needs that aren't there yet)
    import pandas as pd
    print(pd.read_csv("data.csv").describe())
    ```

PRISM shows an approval card with the exact code, the folder it runs in and any packages to
install. The user runs it once, allows Python for this chat (or always), or refuses. New
packages always ask - installing runs the package's own setup code - even where Python
itself is allowed.

The code runs on PRISM's own Python (shipped with the app, with numpy, pandas, matplotlib
and many more installed) inside a sandbox: a virtual environment under PRISM's data folder
that sees those packages and takes what bots install, so neither PRISM's Python nor the
computer's is changed (Settings → Running Python → Reset empties it). It runs in the chat's
own work folder, with a time limit, capped output and none of PRISM's saved keys in its
environment. What it printed and the files it made go back to the bot.

This is not an operating-system sandbox: approved code can do what the user's account can.
The card says so.
"""

import asyncio
import json
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
PIP_TIMEOUT_S = 900.0

RUN_LINE = re.compile(r"^[\s*_`>-]*RUN\s*(?:@([\w.+-]+))?[*_`]*\s*:?[\s*_`]*$", re.I)
FENCE = re.compile(r"```([\w.+-]*)[^\n]*\n(.*?)\n?```\s*$", re.S)
LABEL = re.compile(r"^\s*(python3?|py|code|copy|copy code)?\s*$", re.I)
PIP_LINE = re.compile(r"^\s*#\s*pip\s*:\s*(.+)$", re.I | re.M)
# A requirement as pip takes it - a name, extras and a version - and nothing else: no
# options, paths or URLs (they could point pip anywhere).
PACKAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?"
                     r"((==|>=|<=|~=|!=|>|<)[A-Za-z0-9.*+!_-]+(,(==|>=|<=|~=|!=|>|<)[A-Za-z0-9.*+!_-]+)*)?$")
# Environment passed on: what Python needs to start, nothing PRISM keeps (keys, tokens).
KEEP_ENV = {"PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP",
            "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA",
            "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA", "LANG", "LC_ALL", "TERM",
            "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS", "USER", "USERNAME", "TMPDIR",
            "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"}

RUN_HELP = (" You can run Python on the user's computer - each run needs the user's approval: end "
            "your reply with a line `RUN @python` followed by one ```python block, and nothing "
            "after it. Python {version} with {packages} and more is there; if you need another "
            "package, make the block's first line `# pip: name1 name2`. The code runs in a work "
            "folder of its own (save files there); what it prints comes back to you, then finish "
            "your reply. Use it to compute, test code or process data rather than guessing. At "
            "most {max_runs} runs per turn.")
SHOWN_PACKAGES = "numpy, pandas, scipy, matplotlib, scikit-learn, requests, openpyxl, python-docx, pypdf, pillow"


class RunError(Exception):
    pass


@dataclass
class Request:
    who: str                         # the bot or agent asking
    code: str
    tool: str = "python"
    packages: list[str] = field(default_factory=list)   # asked for with # pip:
    install: list[str] = field(default_factory=list)    # of those, what isn't there yet
    rejected: list[str] = field(default_factory=list)   # not package names: refused
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "pending"          # pending | installing | running | done | denied | failed | cancelled | expired
    output: str = ""
    exit_code: int | None = None
    seconds: float = 0.0
    files: list[str] = field(default_factory=list)
    approved_by: str = ""            # once | chat | flow | always

    def card(self) -> dict:
        return dict(self.__dict__)


def packages_in(code: str) -> tuple[list[str], list[str]]:
    """`# pip: a b==1.2` lines: (requirements, things that aren't requirements)."""
    good, bad = [], []
    for line in PIP_LINE.findall(code):
        for token in line.replace(",", " ").split():
            (good if PACKAGE.match(token) and not token.startswith("-") else bad).append(token)
    return list(dict.fromkeys(good)), bad


def parse_request(text: str, who: str) -> Request | None:
    """`RUN @python` and one python block, at the very end of a reply."""
    body = text.rstrip()
    fence = FENCE.search(body)
    if not fence:
        return None
    before = body[:fence.start()].rstrip().splitlines()
    while before and LABEL.match(before[-1]):  # a site's code header left as text ("Python", "Copy")
        before.pop()
    m = RUN_LINE.match(before[-1]) if before else None
    lang = ((m.group(1) if m else "") or fence.group(1) or "").lower()
    if not m or lang not in ("python", "py", "python3", ""):
        return None
    code = fence.group(2)
    if not code.strip():
        return None
    good, bad = packages_in(code)
    return Request(who, code, packages=good, rejected=bad)


# --- the sandbox ------------------------------------------------------------------------------

def _shipped(name: str) -> Path | None:
    """A file shipped next to the app (Windows folder) or in PRISM.app's Resources."""
    if not getattr(sys, "frozen", False):
        return None
    app = Path(sys.executable).parent
    return next((p for p in (app / name, app.parent / "Resources" / name) if p.is_file()), None)


def runtime_version() -> str:
    """The version of PRISM's own Python, without unpacking it ("" if it isn't there)."""
    if os.environ.get("PRISM_PYTHON"):
        return "custom"
    if not getattr(sys, "frozen", False):
        return "%d.%d.%d" % sys.version_info[:3]
    info = _shipped("runtime.json")
    try:
        return json.loads(info.read_text())["python"] if info else ""
    except (OSError, ValueError, KeyError):
        return ""


def runtime_python() -> str | None:
    """PRISM's own Python: PRISM_PYTHON, else the one shipped with the app - one archive,
    unpacked into PRISM's data folder the first time (a few minutes). It isn't kept unpacked
    next to the app: on drives with large clusters (exFAT) its ~30,000 small files would take
    a terabyte. When running from source: the Python running PRISM. Blocking: call in a thread."""
    if env := os.environ.get("PRISM_PYTHON"):
        return env
    if not getattr(sys, "frozen", False):
        return sys.executable
    archive, info = _shipped("runtime.tar.gz"), _shipped("runtime.json")
    if not archive or not info:
        return None
    from .config import data_dir
    tag = json.loads(info.read_text()).get("tag", "")
    target = data_dir() / "python" / f"runtime-{tag}"
    exe = target / ("python.exe" if sys.platform == "win32" else "bin/python3")
    if not exe.is_file():
        import tarfile
        part = target.with_name(target.name + ".part")
        shutil.rmtree(part, ignore_errors=True)
        part.mkdir(parents=True)
        with tarfile.open(archive) as t:
            root = part.resolve()
            for m in t.getmembers():
                if not (part / m.name).resolve().is_relative_to(root):
                    raise RunError("PRISM's Python archive is damaged - reinstall PRISM")
            t.extractall(part)
        shutil.rmtree(target, ignore_errors=True)
        part.rename(target)
        for old in target.parent.glob("runtime-*"):  # an earlier PRISM's Python
            if old != target and not old.name.endswith(".part"):
                shutil.rmtree(old, ignore_errors=True)
    return str(exe)


class Sandbox:
    """A virtual environment on PRISM's Python that sees its packages and takes new ones."""

    def __init__(self, path: Path, runtime: Callable[[], str | None] = runtime_python):
        self.path, self.runtime = path, runtime

    @property
    def python(self) -> Path:
        return self.path / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    def ready(self) -> bool:
        cfg = self.path / "pyvenv.cfg"
        if not (cfg.is_file() and self.python.is_file()):
            return False
        home = next((line.split("=", 1)[1].strip() for line in cfg.read_text().splitlines()
                     if line.startswith("home")), "")
        return Path(home).is_dir()  # PRISM moved or updated: its Python is elsewhere now

    def ensure(self) -> Path:
        """Create it (or again, if PRISM's Python moved). Blocking: call in a thread."""
        if self.ready():
            return self.python
        base = self.runtime()
        if not base:
            raise RunError("PRISM's Python is missing from this install - reinstall PRISM")
        shutil.rmtree(self.path, ignore_errors=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        p = subprocess.run([base, "-m", "venv", "--system-site-packages", str(self.path)],
                           capture_output=True, timeout=300, creationflags=NO_WINDOW)
        if p.returncode != 0:
            raise RunError("couldn't make the Python sandbox: " + p.stderr.decode(errors="replace")[-300:])
        return self.python

    def reset(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)

    def env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k.upper() in KEEP_ENV}
        bindir = str(self.python.parent)
        env.update({"PATH": os.pathsep.join([bindir, env.get("PATH", "")]), "VIRTUAL_ENV": str(self.path),
                    "PYTHONNOUSERSITE": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                    "MPLBACKEND": "Agg", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1"})
        return env

    def missing(self, packages: list[str]) -> list[str]:
        """Which of these aren't installed (here or in PRISM's Python); a pinned version
        always counts as missing, so pip checks it."""
        if not packages:
            return []
        names = [re.split(r"[\[=<>!~]", p, maxsplit=1)[0] for p in packages]
        code = ("import json, sys, importlib.metadata as m\nout = []\nfor n in json.loads(sys.argv[1]):\n"
                "    try: m.distribution(n)\n    except m.PackageNotFoundError: out.append(n)\nprint(json.dumps(out))")
        p = subprocess.run([str(self.ensure()), "-c", code, json.dumps(names)], capture_output=True,
                           timeout=60, env=self.env(), creationflags=NO_WINDOW)
        absent = set(json.loads(p.stdout or "[]")) if p.returncode == 0 else set(names)
        return [r for r, n in zip(packages, names) if n in absent or any(op in r for op in "=<>!~")]

    def install(self, requirements: list[str], extra: list[str] = (), timeout: float = PIP_TIMEOUT_S) -> str:
        p = subprocess.run([str(self.ensure()), "-m", "pip", "install", "--no-input", *extra, *requirements],
                           capture_output=True, timeout=timeout, env=self.env(), creationflags=NO_WINDOW)
        out = (p.stdout + p.stderr).decode("utf-8", errors="replace")
        if p.returncode != 0:
            raise RunError("pip couldn't install " + " ".join(requirements) + ":\n" + out[-1500:])
        return out

    def uninstall(self, name: str) -> None:
        subprocess.run([str(self.ensure()), "-m", "pip", "uninstall", "-y", name], capture_output=True,
                       timeout=300, env=self.env(), creationflags=NO_WINDOW)

    def installed(self) -> list[dict]:
        """What bots (or the catalog) installed into the sandbox - not PRISM's own packages."""
        if not self.ready():
            return []
        p = subprocess.run([str(self.python), "-m", "pip", "list", "--format", "json",
                            "--path", str(self._site())], capture_output=True, timeout=60, env=self.env(),
                           creationflags=NO_WINDOW)
        try:
            return [x for x in json.loads(p.stdout or "[]") if x["name"].lower() not in ("pip", "setuptools")]
        except ValueError:
            return []

    def _site(self) -> Path:
        lib = self.path / "Lib" / "site-packages"
        return lib if lib.is_dir() else next((self.path / "lib").glob("python3*/site-packages"), lib)

    def version(self) -> str:
        return runtime_version()


def execute(req: Request, work: Path, sandbox: Sandbox, timeout: float) -> Request:
    """Install what it asked for (it was approved), then run it - blocking: call in a thread."""
    work.mkdir(parents=True, exist_ok=True)
    before = {p: p.stat().st_mtime for p in work.rglob("*") if p.is_file()}
    start = time.monotonic()
    notes = ""
    try:
        python = sandbox.ensure()
        if req.install:
            req.status = "installing"
            sandbox.install(req.install)
            notes = f"[installed into the sandbox: {', '.join(req.install)}]\n"
        script = work / f"prism_run_{req.id}.py"
        script.write_text(req.code, encoding="utf-8")
        req.status = "running"
        p = subprocess.run([str(python), str(script)], cwd=work, env=sandbox.env(), capture_output=True,
                           timeout=timeout, creationflags=NO_WINDOW, stdin=subprocess.DEVNULL)
        out = p.stdout.decode("utf-8", errors="replace")
        err = p.stderr.decode("utf-8", errors="replace")
        req.exit_code = p.returncode
        req.output = notes + out + (("\n[stderr]\n" + err) if err.strip() else "")
        req.status = "done"
    except subprocess.TimeoutExpired as e:
        req.output = notes + (e.stdout or b"").decode("utf-8", "replace") + f"\n[stopped after {round(timeout)} s - the time limit]"
        req.status, req.exit_code = "failed", None
    except (RunError, OSError) as e:
        req.output, req.status = notes + str(e), "failed"
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
    head = (f"[ran python in {req.seconds}s, exit code {req.exit_code}]" if req.status == "done"
            else "[python failed]")
    refused = (f"\n[not installed - not package names: {', '.join(req.rejected)}]" if req.rejected else "")
    files = f"\n[files it made: {', '.join(req.files)}]" if req.files else ""
    return (f"{head}{refused}\n{req.output.strip() or '(no output)'}{files}\n\n"
            "Now finish your reply (you may RUN again if you must).")


class Runner:
    """Pending approvals, allow rules, and the runs.

    rules: {"always": ["python"], "chats": {chat_id: ["python"]}, "flows": {flow: ["python"]}},
    kept by the app (meta `run_rules`). A rule lets code run without asking; packages it
    wants that aren't installed yet still ask."""

    def __init__(self, work_root: Path, rules: Callable[[], dict], save_rules: Callable[[dict], None],
                 sandbox: Sandbox, timeout: float = DEFAULT_TIMEOUT_S):
        self.work_root, self.rules, self.save_rules = work_root, rules, save_rules
        self.sandbox, self.timeout = sandbox, timeout
        self.pending: dict[str, tuple[Request, asyncio.Future, int | None]] = {}

    def work_dir(self, scope: str) -> Path:
        return self.work_root / re.sub(r"[^\w-]", "_", scope)

    def allowed(self, chat_id: int | None, flow: str | None) -> str | None:
        r = self.rules() or {}
        if "python" in r.get("always", []):
            return "always"
        if chat_id is not None and "python" in r.get("chats", {}).get(str(chat_id), []):
            return "chat"
        if flow and "python" in r.get("flows", {}).get(flow, []):
            return "flow"
        return None

    async def request(self, req: Request, chat_id: int, emit: Callable, update: Callable,
                      flow: str | None = None) -> Request:
        """Ask the user (unless a rule allows it and nothing new is installed), then run it."""
        try:
            req.install = await asyncio.to_thread(self.sandbox.missing, req.packages)
        except Exception:
            req.install = list(req.packages)
        rule = None if req.install else self.allowed(chat_id, flow)
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
            req.status = "installing" if req.install else "running"
            update(req, message_id)
        req.approved_by = rule
        await asyncio.to_thread(execute, req, self.work_dir(f"chat-{chat_id}"), self.sandbox, self.timeout)
        update(req, message_id)
        return req

    def decide(self, request_id: str, allow: str | None, chat_id: int | None = None, flow: str | None = None) -> bool:
        """allow: once | chat | flow | always, or None to refuse. Chat/flow/always add a rule
        (for running Python; new packages keep asking)."""
        item = self.pending.get(request_id)
        if not item or item[1].done():
            return False
        _req, fut, _ = item
        if allow in ("chat", "flow", "always"):
            r = dict(self.rules() or {})
            if allow == "always":
                r["always"] = ["python"]
            elif allow == "chat" and chat_id is not None:
                r["chats"] = {**r.get("chats", {}), str(chat_id): ["python"]}
            elif allow == "flow" and flow:
                r["flows"] = {**r.get("flows", {}), flow: ["python"]}
            self.save_rules(r)
        fut.set_result(allow)
        return True
