"""Flows: agents connected in a graph, run step by step with their state saved after each step.

A flow is written in YAML:

    name: market-brief
    inputs: {topic: text}
    agents:                      # optional: agents defined just for this flow
      critic: {model: "@claude", instructions: "Find what is wrong.", output: {verdict: PASS|FIX, notes: text}}
    nodes:
      gather:  {agent: researcher, prompt: "Research {{input.topic}}."}
      draft:   {agent: writer, prompt: "Write it from {{gather.text}}. Fix: {{review.notes}}"}
      review:  {agent: critic, prompt: "Review {{draft.text}}."}
      publish: {ask_you: "Publish this?", show: "{{draft.text}}"}
    edges:
      - gather -> draft
      - draft -> review
      - review -> draft:   {when: "verdict == FIX", max_rounds: 3}
      - review -> publish: {when: "verdict == PASS or rounds_done"}

Node types (by the key they carry): `agent` (a model answers), `tool` (docs | prism | fetch:
no model), `ask_you` (pauses for the user), `wait` (a delay, e.g. 10m), `flow` (another flow,
copied in as a subflow). `join: all` makes a node wait for every node that leads into it.

Running: a node that finishes activates the targets of its outgoing edges whose `when`
holds. An edge with `max_rounds` is taken at most that many times; once it is used up, the
node's other edges see `rounds_done` as true. Every cycle must contain such an edge, so a flow
always ends; a run also has a step and time budget, and pauses (not fails) when it is spent.
The run's state - outputs, counters, the queue of activated nodes, each agent's conversation -
is saved after every step, so a run continues after PRISM restarts.
"""

import ast
import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import yaml

from .meeting import _ask_request, _err
from .providers import Provider, wait_reply

NODE_TYPES = ("agent", "tool", "ask_you", "wait", "flow")
TOOLS = ("docs", "prism", "fetch", "run", "kg")
NAME = re.compile(r"^[A-Za-z][\w-]{0,40}$")
TEMPLATE = re.compile(r"\{\{\s*([\w/-]+)(?:\.([\w-]+))?\s*\}\}")
DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(s|sec|m|min|h|hr|d)?\s*$", re.I)
DEFAULT_BUDGET = {"steps": 40, "minutes": 120}
MAX_NODE_RUNS = 25           # one node, in one run (a backstop under max_rounds)
MAX_SUBFLOW_DEPTH = 3
HISTORY_KEEP = 24            # messages of an API/local agent's conversation kept in the state
MAX_ASKS = 3
MAX_RUNS = 5


class FlowError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


# --- the flow as written ---------------------------------------------------------------------

@dataclass
class Edge:
    src: str
    dst: str
    when: str | None = None
    max_rounds: int | None = None
    back: bool = False       # closes a loop (for drawing)

    @property
    def key(self) -> str:
        return f"{self.src}->{self.dst}"


@dataclass
class Node:
    id: str
    type: str
    spec: dict
    join_all: bool = False
    label: str = ""          # for drawing: the agent, tool or question


@dataclass
class Flow:
    name: str
    inputs: dict[str, str]
    nodes: dict[str, Node]
    edges: list[Edge]
    agents: dict[str, dict]  # defined inline in this flow
    budget: dict
    aliases: dict[str, list[str]] = field(default_factory=dict)  # subflow node -> its last nodes
    start: list[str] = field(default_factory=list)                # named first steps (start:)

    def outgoing(self, node: str) -> list[Edge]:
        return [e for e in self.edges if e.src == node]

    def incoming(self, node: str) -> list[Edge]:
        return [e for e in self.edges if e.dst == node]

    @property
    def starts(self) -> list[str]:
        """The first steps: as named, else those nothing leads into, else those only a loop
        leads back into (a draft that a review sends back)."""
        if self.start:
            return list(self.start)
        plain = [n for n in self.nodes if not self.incoming(n)]
        return plain or [n for n in self.nodes
                         if all(e.max_rounds is not None for e in self.incoming(n))]

    def to_json(self) -> dict:
        """What the page draws: nodes in layers (left to right), loops marked."""
        levels = layers(self)
        return {"name": self.name, "inputs": self.inputs, "starts": self.starts,
                "nodes": [{"id": n.id, "type": n.type, "label": n.label, "level": levels.get(n.id, 0),
                           "join": n.join_all} for n in self.nodes.values()],
                "edges": [{"src": e.src, "dst": e.dst, "when": e.when, "max_rounds": e.max_rounds,
                           "back": e.back} for e in self.edges]}


def parse(source: str, agents: Callable[[str], dict | None] = lambda n: None,
          flows: Callable[[str], str | None] = lambda n: None, _depth: int = 0) -> Flow:
    """YAML -> Flow, checked: every error found is reported at once (FlowError.errors)."""
    try:
        doc = yaml.safe_load(source) or {}
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f" (line {mark.line + 1})" if mark else ""
        raise FlowError([f"not valid YAML{where}: {getattr(e, 'problem', None) or e}"]) from e
    if not isinstance(doc, dict):
        raise FlowError(["a flow is a set of keys: name, inputs, nodes, edges"])
    errors: list[str] = []
    name = str(doc.get("name") or "").strip()
    if not NAME.match(name):
        errors.append("name: letters, digits, - or _, starting with a letter")
    inputs = doc.get("inputs") or {}
    if isinstance(inputs, list):
        inputs = {str(i): "text" for i in inputs}
    if not isinstance(inputs, dict):
        errors.append("inputs: a set of name: text")
        inputs = {}
    inline = doc.get("agents") or {}
    if not isinstance(inline, dict):
        errors.append("agents: a set of name: {model, instructions, ...}")
        inline = {}
    for aname, a in inline.items():
        if not isinstance(a, dict) or not a.get("model"):
            errors.append(f"agents.{aname}: needs a model, e.g. model: \"@claude\"")
    budget = {**DEFAULT_BUDGET, **(doc.get("budget") or {})}

    raw_nodes = doc.get("nodes") or {}
    if not isinstance(raw_nodes, dict) or not raw_nodes:
        raise FlowError(errors + ["nodes: at least one step"])
    nodes: dict[str, Node] = {}
    subflows: dict[str, tuple[Flow, dict]] = {}
    for nid, spec in raw_nodes.items():
        nid = str(nid)
        if not NAME.match(nid) or nid == "input":
            errors.append(f"node {nid!r}: a name of letters, digits, - or _ (not 'input')")
            continue
        if not isinstance(spec, dict):
            errors.append(f"nodes.{nid}: a set of keys, e.g. {{agent: critic, prompt: ...}}")
            continue
        types = [t for t in NODE_TYPES if t in spec]
        if len(types) != 1:
            errors.append(f"nodes.{nid}: say what it is with exactly one of {', '.join(NODE_TYPES)}")
            continue
        t = types[0]
        node = Node(nid, t, spec, join_all=str(spec.get("join", "")).lower() == "all")
        if t == "agent":
            aname = str(spec["agent"])
            node.label = aname
            if aname not in inline and not agents(aname):
                errors.append(f"nodes.{nid}: no agent named {aname!r} - define it under agents: or in Flows → Agents")
        elif t == "tool":
            node.label = str(spec["tool"])
            if node.label not in TOOLS:
                errors.append(f"nodes.{nid}: tool is one of {', '.join(TOOLS)}")
            elif node.label == "run" and (not spec.get("code") or str(spec.get("lang", "python")).lower() != "python"):
                errors.append(f"nodes.{nid}: tool run takes code: (Python; a first line # pip: a b installs packages)")
        elif t == "ask_you":
            node.label = str(spec["ask_you"])[:60]
        elif t == "wait":
            node.label = str(spec["wait"])
            if duration_s(spec["wait"]) is None:
                errors.append(f"nodes.{nid}: wait takes a time like 30s, 10m, 2h")
        elif t == "flow":
            child = str(spec["flow"])
            node.label = child
            src = flows(child)
            if src is None:
                errors.append(f"nodes.{nid}: no flow named {child!r}")
            elif _depth >= MAX_SUBFLOW_DEPTH:
                errors.append(f"nodes.{nid}: subflows nest at most {MAX_SUBFLOW_DEPTH} deep (or they include each other)")
            else:
                try:
                    subflows[nid] = (parse(src, agents, flows, _depth + 1), spec.get("inputs") or {})
                except FlowError as e:
                    errors += [f"nodes.{nid} (flow {child}): {x}" for x in e.errors]
        nodes[nid] = node

    edges: list[Edge] = []
    for i, raw in enumerate(doc.get("edges") or []):
        opts: dict = {}
        if isinstance(raw, dict) and len(raw) == 1:
            raw, opts = next(iter(raw.items()))
            opts = opts or {}
        parts = [p.strip() for p in str(raw).split("->")]
        if len(parts) < 2 or not all(parts) or not isinstance(opts, dict):
            errors.append(f"edges[{i}]: write it as `a -> b` (or `a -> b: {{when: ..., max_rounds: 3}}`)")
            continue
        for a, b in zip(parts, parts[1:]):  # a -> b -> c is two edges
            for end in (a, b):
                if end not in nodes and end not in raw_nodes:
                    errors.append(f"edges[{i}]: no node named {end!r}")
            mr = opts.get("max_rounds")
            if mr is not None and (not isinstance(mr, int) or mr < 1):
                errors.append(f"edges[{i}]: max_rounds is a whole number from 1")
                mr = None
            when = opts.get("when")
            if when is not None:
                try:
                    check_condition(str(when))
                except FlowError as e:
                    errors.append(f"edges[{i}] when: {e}")
            edges.append(Edge(a, b, str(when) if when is not None else None, mr))
    for k in set(opts_keys(doc.get("edges") or [])) - {"when", "max_rounds"}:
        errors.append(f"edges: unknown option {k!r} (use when and max_rounds)")

    start = doc.get("start") or []
    start = [str(x) for x in (start if isinstance(start, list) else [start])]
    for x in start:
        if x not in raw_nodes:
            errors.append(f"start: no node named {x!r}")
    flow = Flow(name, {str(k): str(v) for k, v in inputs.items()}, nodes, edges, inline, budget, start=start)
    if not errors:
        for nid, (child, mapping) in subflows.items():
            expand(flow, nid, child, mapping)
        errors += check_templates(flow)
        errors += check_cycles(flow)
        if not flow.starts:
            errors.append("no first step: every node has an edge into it")
    if errors:
        raise FlowError(errors)
    mark_back_edges(flow)
    return flow


def opts_keys(raw_edges) -> list[str]:
    out = []
    for raw in raw_edges:
        if isinstance(raw, dict) and len(raw) == 1:
            opts = next(iter(raw.values()))
            if isinstance(opts, dict):
                out += list(opts)
    return out


def duration_s(value) -> float | None:
    m = DURATION.match(str(value))
    if not m:
        return None
    unit = (m.group(2) or "s").lower()[0]
    return float(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def expand(flow: Flow, nid: str, child: Flow, mapping: dict) -> None:
    """A subflow node is replaced by a copy of the child's nodes, named `nid/<child node>`;
    edges into it go to the child's first steps and edges out of it leave from its last ones.
    `{{nid.x}}` then reads the child's last step to finish."""
    prefix = f"{nid}/"

    def rename(text: str) -> str:
        def sub(m: re.Match) -> str:
            ref, fld = m.group(1), m.group(2)
            if ref == "input":
                return str(mapping.get(fld, ""))  # the parent's template, read in the parent
            return "{{" + prefix + ref + (f".{fld}" if fld else "") + "}}"
        return TEMPLATE.sub(sub, text)

    starts = child.starts
    ends = [n for n in child.nodes if not child.outgoing(n)] or list(child.nodes)[-1:]
    del flow.nodes[nid]
    for n in child.nodes.values():
        spec = {k: rename(v) if isinstance(v, str) else v for k, v in n.spec.items()}
        flow.nodes[prefix + n.id] = Node(prefix + n.id, n.type, spec, n.join_all, n.label)
    for name, a in child.agents.items():
        flow.agents.setdefault(name, a)
    for alias, targets in child.aliases.items():
        flow.aliases[prefix + alias] = [prefix + t for t in targets]
    outer = [e for e in flow.edges if nid in (e.src, e.dst)]
    flow.edges = [e for e in flow.edges if nid not in (e.src, e.dst)]
    flow.edges += [Edge(prefix + e.src, prefix + e.dst, e.when, e.max_rounds) for e in child.edges]
    for e in outer:
        srcs = [prefix + x for x in ends] if e.src == nid else [e.src]
        dsts = [prefix + x for x in starts] if e.dst == nid else [e.dst]
        flow.edges += [Edge(s, d, e.when, e.max_rounds) for s in srcs for d in dsts]
    flow.aliases[nid] = [prefix + x for x in ends]


def check_templates(flow: Flow) -> list[str]:
    errors = []
    for n in flow.nodes.values():
        for v in n.spec.values():
            for ref, fld in TEMPLATE.findall(v) if isinstance(v, str) else []:
                if ref == "input":
                    if fld not in flow.inputs:
                        errors.append(f"nodes.{n.id}: {{{{input.{fld}}}}} is not one of the inputs ({', '.join(flow.inputs) or 'none'})")
                elif ref not in flow.nodes and ref not in flow.aliases:
                    errors.append(f"nodes.{n.id}: {{{{{ref}}}}} names no node")
    return errors


def check_cycles(flow: Flow) -> list[str]:
    """Every loop needs an edge with max_rounds: look for a cycle among the other edges."""
    graph: dict[str, list[str]] = {n: [] for n in flow.nodes}
    for e in flow.edges:
        if e.max_rounds is None:
            graph[e.src].append(e.dst)
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(n: str) -> list[str] | None:
        state[n] = 1
        stack.append(n)
        for m in graph[n]:
            if state.get(m) == 1:
                return stack[stack.index(m):] + [m]
            if m not in state and (cyc := visit(m)):
                return cyc
        stack.pop()
        state[n] = 2
        return None

    for n in graph:
        if n not in state and (cyc := visit(n)):
            return [f"the loop {' -> '.join(cyc)} has no limit: give one of its edges max_rounds"]
    return []


def mark_back_edges(flow: Flow) -> None:
    seen, on_stack = set(), set()

    def visit(n: str) -> None:
        seen.add(n)
        on_stack.add(n)
        for e in flow.outgoing(n):
            if e.dst in on_stack:
                e.back = True
            elif e.dst not in seen:
                visit(e.dst)
        on_stack.discard(n)

    for s in flow.starts + list(flow.nodes):
        if s not in seen:
            visit(s)


def layers(flow: Flow) -> dict[str, int]:
    """Longest path from a first step, loops left out: the column each node is drawn in."""
    level = {n: 0 for n in flow.nodes}
    for _ in range(len(flow.nodes)):
        changed = False
        for e in flow.edges:
            if not e.back and level[e.dst] < level[e.src] + 1:
                level[e.dst] = level[e.src] + 1
                changed = True
        if not changed:
            break
    return level


# --- conditions ----------------------------------------------------------------------------

_ALLOWED = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.Compare,
            ast.Eq, ast.NotEq, ast.In, ast.NotIn, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.Name, ast.Load, ast.Constant)


def check_condition(text: str) -> ast.Expression:
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError:
        raise FlowError([f"can't read {text!r} - e.g. verdict == PASS, or score >= 7 and not done"])
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise FlowError([f"{text!r}: only comparisons (== != < > in), and, or, not"])
    return tree


def evaluate(text: str, fields: dict[str, Any]) -> bool:
    """A field name reads the step's output; any other bare word is itself (PASS = "PASS").
    Text compares without case or surrounding punctuation."""
    env = {k.lower(): v for k, v in fields.items()}

    def norm(v):
        if isinstance(v, str):
            s = v.strip().strip("*_`.!\"' ").lower()
            try:
                return float(s)
            except ValueError:
                return s
        return v

    def val(node):
        if isinstance(node, ast.Expression):
            return val(node.body)
        if isinstance(node, ast.BoolOp):
            vals = [bool(val(v)) for v in node.values]
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if isinstance(node, ast.UnaryOp):
            return not val(node.operand)
        if isinstance(node, ast.Name):
            return env[node.id.lower()] if node.id.lower() in env else node.id
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Compare):
            left = norm(val(node.left))
            for op, right_node in zip(node.ops, node.comparators):
                right = norm(val(right_node))
                try:
                    ok = {ast.Eq: lambda: left == right, ast.NotEq: lambda: left != right,
                          ast.In: lambda: str(left) in str(right), ast.NotIn: lambda: str(left) not in str(right),
                          ast.Lt: lambda: left < right, ast.LtE: lambda: left <= right,
                          ast.Gt: lambda: left > right, ast.GtE: lambda: left >= right}[type(op)]()
                except TypeError:
                    ok = False
                if not ok:
                    return False
                left = right
            return True
        raise FlowError([f"{text!r}: not allowed"])

    result = val(check_condition(text))
    return bool(norm(result)) if not isinstance(result, bool) else result


# --- templates and output fields -------------------------------------------------------------

def fill(text: str, outputs: dict[str, dict], inputs: dict[str, str], aliases: dict[str, list[str]]) -> str:
    def sub(m: re.Match) -> str:
        ref, fld = m.group(1), m.group(2)
        if ref == "input":
            return str(inputs.get(fld, ""))
        out = outputs.get(ref)
        if out is None and ref in aliases:  # a subflow: its last step that has run
            out = next((outputs[a] for a in reversed(aliases[ref]) if a in outputs), None)
        if out is None:
            return ""
        return str(out.get(fld or "text", ""))
    return TEMPLATE.sub(sub, text)


def output_spec(spec) -> dict[str, list[str] | None]:
    """{field: [allowed values] or None for free text}"""
    out = {}
    for k, v in (spec or {}).items():
        v = str(v).strip()
        out[str(k).lower()] = None if v.lower() in ("text", "number", "") else [x.strip() for x in v.split("|") if x.strip()]
    return out


def output_request(spec: dict[str, list[str] | None]) -> str:
    lines = [f"{k.upper()}: " + (" or ".join(v) if v else "<text>") for k, v in spec.items()]
    return ("\n\nEnd your reply with these lines, exactly in this form (PRISM reads them):\n"
            + "\n".join(lines))


FIELD_LINE = re.compile(r"^[\s*_`>#-]*([A-Za-z][\w ]{0,30}?)[\s*_`]*:[\s*_`]*(.*?)[\s*_`]*$")


def read_fields(text: str, spec: dict[str, list[str] | None]) -> tuple[dict[str, str], list[str]]:
    """The fields from the end of a reply; returns (fields, missing). A text field runs to the
    next field line."""
    lines = text.strip().splitlines()
    found: dict[str, str] = {}
    current = None
    for line in lines:
        m = FIELD_LINE.match(line)
        key = m.group(1).strip().lower().replace(" ", "_") if m else None
        if key in spec:
            current = key
            found[key] = m.group(2).strip()
        elif current and spec[current] is None and line.strip():
            found[current] += "\n" + line.strip()
        elif m and key not in spec:
            current = None
    out, missing = {}, []
    for k, allowed in spec.items():
        v = found.get(k, "").strip()
        if allowed:
            word = re.split(r"[\s,.;:!]+", v.strip("*_` "))[0] if v else ""
            pick = next((a for a in allowed if a.lower() == word.lower()), None)
            if pick is None:
                missing.append(k)
            else:
                out[k] = pick
        elif v:
            out[k] = v
        else:
            missing.append(k)
    return out, missing


# --- a run ---------------------------------------------------------------------------------

@dataclass
class RunState:
    inputs: dict[str, str]
    status: str = "running"           # running | waiting | paused | done | stopped | failed
    error: str | None = None
    queue: list[str] = field(default_factory=list)       # activated, not finished yet
    outputs: dict[str, dict] = field(default_factory=dict)
    node_runs: dict[str, int] = field(default_factory=dict)
    edge_runs: dict[str, int] = field(default_factory=dict)
    arrived: dict[str, list[str]] = field(default_factory=dict)  # join: all - who has reported in
    answers: dict[str, dict] = field(default_factory=dict)       # ask_you node -> the user's answer
    waiting: str | None = None
    wake: dict[str, float] = field(default_factory=dict)          # wait node -> wall-clock time
    agents: dict[str, dict] = field(default_factory=dict)         # agent -> {history, thread_url, notes_seen}
    notes: list[str] = field(default_factory=list)                # what the user said during the run
    steps: int = 0
    seconds: float = 0.0
    budget: dict = field(default_factory=lambda: dict(DEFAULT_BUDGET))
    last: list[str] = field(default_factory=list)                 # the nodes that finished last

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "RunState":
        s = cls(d.get("inputs", {}))
        for k, v in d.items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s


class StepError(Exception):
    pass


class RunEnv:
    """What a run needs from PRISM: overridden by the app (and by tests)."""
    timeout: float = 1200.0       # no sign of activity this long: the step failed
    max_s: float = 4 * 3600.0     # one call in all

    def agent(self, name: str) -> dict | None:
        return None

    def provider(self, agent: dict, memory: dict) -> Provider:
        raise NotImplementedError

    async def context(self, agent: dict, prompt: str) -> str:
        return ""

    def askable(self, agent: dict) -> list[str]:
        return []

    async def ask(self, target: str, question: str) -> str:
        return ""

    async def tool(self, name: str, query: str) -> str:
        return ""

    async def run(self, who: str, req) -> "object":
        """A program to run, after the user approves it: returns the finished request."""
        raise StepError("running programs is off (Settings → Running programs)")

    def can_run(self, agent: dict) -> str:
        """The Python version an agent's RUN gets ("": it may not run code)."""
        return ""

    def emit(self, kind: str, **data) -> int | None:
        return None

    def save(self, state: RunState) -> None:
        pass


AGENT_SYSTEM = ("You are {name}, one step in an automated workflow the user set up in PRISM. "
                "Your role: {instructions}\nDo your part fully and concretely; the other steps "
                "are done by others. Answer in plain text in this chat: don't create documents "
                "or use connected apps. When you make a file, put the complete file in one "
                "fenced code block.")
ASK_TOOL = (" You can look things up: write one line `ASK @name: <question>` and nothing after "
            "it, where name is one of: {names}. ({hints}) The answer comes back to you, then "
            "finish your reply. At most {max_asks} asks.")
ASK_HINTS = {"docs": "docs searches the user's files", "prism": "prism searches the user's past chats"}


class Run:
    def __init__(self, flow: Flow, state: RunState, env: RunEnv):
        self.flow, self.state, self.env = flow, state, env
        self._stop = asyncio.Event()
        self._running: set[str] = set()

    @classmethod
    def start(cls, flow: Flow, inputs: dict, env: RunEnv) -> "Run":
        state = RunState({k: str(inputs.get(k, "")) for k in flow.inputs},
                         budget=dict(flow.budget), queue=list(flow.starts))
        return cls(flow, state, env)

    # --- control from outside ---

    def stop(self) -> None:
        self._stop.set()

    def add_note(self, text: str) -> None:
        self.state.notes.append(text)

    def answer(self, node: str, approved: bool, text: str = "") -> bool:
        if self.state.waiting != node:
            return False
        self.state.answers[node] = {"approved": "yes" if approved else "no", "text": text}
        self.state.waiting = None
        self.state.status = "running"
        self.env.save(self.state)
        return True

    def resume(self, more_budget: bool = False) -> None:
        if more_budget:
            self.state.budget["steps"] = self.state.steps + DEFAULT_BUDGET["steps"] // 2
            self.state.budget["minutes"] = self.state.seconds / 60 + DEFAULT_BUDGET["minutes"] / 2
        self.state.status, self.state.error = "running", None
        self._stop.clear()

    # --- the loop ---

    async def run(self) -> RunState:
        s = self.state
        s.status, s.error = "running", None
        self.env.emit("run", status="running")
        try:
            while s.queue and not self._stop.is_set():
                if s.waiting:
                    s.status = "waiting"
                    break
                if s.steps >= s.budget["steps"] or s.seconds / 60 >= s.budget["minutes"]:
                    s.status = "paused"
                    s.error = (f"budget reached: {s.steps} steps, {round(s.seconds / 60)} min - "
                               "continue to allow more")
                    break
                wave = [n for n in dict.fromkeys(s.queue) if self._ready(n)]
                if not wave:  # only joins still waiting for their other inputs, and nothing else can come
                    s.queue = []
                    break
                started = time.monotonic()
                results = await asyncio.gather(*(self._step(n) for n in wave), return_exceptions=True)
                s.seconds += time.monotonic() - started
                failed = [(n, r) for n, r in zip(wave, results) if isinstance(r, BaseException)]
                for n, r in failed:
                    if isinstance(r, asyncio.CancelledError):
                        raise r
                if failed:
                    n, r = failed[0]
                    s.status, s.error = "paused", f"{n}: {r if isinstance(r, StepError) else _err(r)}"
                    break
            else:
                if not self._stop.is_set():
                    s.status = "done"
            if self._stop.is_set() and s.status == "running":
                s.status = "stopped"
        except asyncio.CancelledError:
            if self._stop.is_set():
                s.status = "stopped"
            else:
                s.status, s.error = "paused", "interrupted - continue to pick up where it stopped"
            self.env.save(s)
            self.env.emit("run", status=s.status, error=s.error)
            raise
        if s.status == "running":
            s.status = "done"
        self.env.save(s)
        self.env.emit("run", status=s.status, error=s.error, last=s.last)
        return s

    def _ready(self, node: str) -> bool:
        n = self.flow.nodes[node]
        if not n.join_all:
            return True
        need = {e.src for e in self.flow.incoming(node)}
        return need <= set(self.state.arrived.get(node, []))

    async def _step(self, node: str) -> None:
        s, n = self.state, self.flow.nodes[node]
        if s.node_runs.get(node, 0) >= MAX_NODE_RUNS:
            raise StepError(f"ran {MAX_NODE_RUNS} times in this run - check its loop")
        if n.type == "ask_you":
            if node not in s.answers:
                s.waiting = node
                s.status = "waiting"
                self.env.emit("ask_you", node=node, question=fill(str(n.spec["ask_you"]), s.outputs, s.inputs, self.flow.aliases),
                              show=fill(str(n.spec.get("show", "")), s.outputs, s.inputs, self.flow.aliases))
                self.env.save(s)
                return  # stays in the queue until answered
            out = dict(s.answers.pop(node))
        elif n.type == "wait":
            wake = s.wake.setdefault(node, time.time() + duration_s(n.spec["wait"]))
            self.env.save(s)
            while (left := wake - time.time()) > 0:
                try:
                    await asyncio.wait_for(self._stop.wait(), min(left, 5))
                    return  # stopped while waiting: stays queued
                except asyncio.TimeoutError:
                    pass
            s.wake.pop(node, None)
            out = {"text": f"waited {n.spec['wait']}"}
        elif n.type == "tool" and n.spec["tool"] == "run":
            from .runner import Request, packages_in
            code = fill(str(n.spec["code"]), s.outputs, s.inputs, self.flow.aliases)
            good, bad = packages_in(code)
            req = Request(node, code, packages=good, rejected=bad)
            started = time.perf_counter()
            done = await self.env.run(node, req)
            if done.status != "done":
                raise StepError(f"the program didn't run ({done.status}): {done.output[:200]}")
            out = {"text": done.output, "exit_code": str(done.exit_code), "files": ", ".join(done.files)}
            self.env.emit("step", node=node, type="tool", label="run python", prompt=req.code,
                          text=done.output, round=s.node_runs.get(node, 0) + 1,
                          latency_s=round(time.perf_counter() - started, 1))
        elif n.type == "tool":
            query = fill(str(n.spec.get("query") or n.spec.get("url") or n.spec.get("prompt") or n.spec.get("text") or ""),
                         s.outputs, s.inputs, self.flow.aliases)
            name = "kg-write" if n.spec["tool"] == "kg" and n.spec.get("text") else n.spec["tool"]
            started = time.perf_counter()
            text = await wait_reply(self.env.tool(name, query), None, self.env.timeout, self.env.max_s)
            out = {"text": text}
            self.env.emit("step", node=node, type="tool", label=n.label, prompt=query, text=text,
                          round=s.node_runs.get(node, 0) + 1, latency_s=round(time.perf_counter() - started, 1))
        else:
            out = await self._agent_step(node, n)
        s.steps += 1
        s.node_runs[node] = s.node_runs.get(node, 0) + 1
        s.outputs[node] = out
        s.queue.remove(node)
        s.arrived.pop(node, None)
        nxt = self._next(node, out)
        s.queue += nxt
        for e in self.flow.outgoing(node):
            if self.flow.nodes[e.dst].join_all:
                arrived = s.arrived.setdefault(e.dst, [])
                if node not in arrived:
                    arrived.append(node)
        s.last = [node] if not nxt else s.last
        self.env.save(s)

    def _next(self, node: str, out: dict) -> list[str]:
        """Targets of the edges this result takes. A used-up loop edge is skipped and tells the
        node's other edges `rounds_done`."""
        s = self.state
        edges = self.flow.outgoing(node)
        fields = {**out, "round": s.node_runs.get(node, 0), "rounds_done": False}
        bounded = [e for e in edges if e.max_rounds is not None]
        for e in bounded:
            if s.edge_runs.get(e.key, 0) >= e.max_rounds and (e.when is None or evaluate(e.when, fields)):
                fields["rounds_done"] = True
        take = []
        for e in edges:
            if e.max_rounds is not None and s.edge_runs.get(e.key, 0) >= e.max_rounds:
                continue
            if e.when is None or evaluate(e.when, fields):
                take.append(e)
        for e in take:
            s.edge_runs[e.key] = s.edge_runs.get(e.key, 0) + 1
        targets = []
        for e in take:
            dst = self.flow.nodes[e.dst]
            if dst.join_all and e.dst in s.queue:
                continue
            targets.append(e.dst)
        return list(dict.fromkeys(targets))

    async def _agent_step(self, node: str, n: Node) -> dict:
        s = self.state
        name = str(n.spec["agent"])
        agent = {**(self.env.agent(name) or {}), **self.flow.agents.get(name, {}), "name": name}
        if "output" in n.spec:
            agent["output"] = n.spec["output"]
        spec = output_spec(agent.get("output"))
        memory = s.agents.setdefault(name, {"history": [], "thread_url": None, "notes_seen": 0})
        prompt = fill(str(n.spec.get("prompt") or self._default_prompt(node)), s.outputs, s.inputs, self.flow.aliases)
        parts = []
        if s.notes[memory["notes_seen"]:]:
            parts.append("The user added, while this workflow runs:\n" + "\n".join(f"- {x}" for x in s.notes[memory["notes_seen"]:]))
        if ctx := await self.env.context(agent, prompt):
            parts.append(ctx)
        parts.append(prompt)
        full = "\n\n".join(parts) + (output_request(spec) if spec else "")
        askable = self.env.askable(agent)
        system = AGENT_SYSTEM.format(name=name, instructions=agent.get("instructions") or "as the prompt says")
        if askable:
            system += ASK_TOOL.format(names=", ".join(askable), max_asks=MAX_ASKS,
                                      hints="; ".join(ASK_HINTS.get(a, f"{a} is another assistant the user uses") for a in askable))
        from .runner import RUN_HELP, SHOWN_PACKAGES, parse_request, result_for_bot
        langs = self.env.can_run(agent)
        if langs:
            system += RUN_HELP.format(version=langs, packages=SHOWN_PACKAGES, max_runs=MAX_RUNS)
        provider = self.env.provider(agent, memory)
        started = time.perf_counter()
        msgs = list(memory["history"]) + [{"role": "user", "content": full}]
        text = await wait_reply(provider.complete(list(msgs), system), provider, self.env.timeout, self.env.max_s)
        msgs.append({"role": "assistant", "content": text})
        runs = 0
        while langs and runs < MAX_RUNS and (req := parse_request(text, name)):
            runs += 1
            done = await self.env.run(name, req)
            msgs.append({"role": "user", "content": result_for_bot(done)})
            text = await wait_reply(provider.complete(list(msgs), system), provider, self.env.timeout, self.env.max_s)
            msgs.append({"role": "assistant", "content": text})
        asks = 0
        while askable and (q := _ask_request(text)) and q.group(1).lower() in askable and asks < MAX_ASKS:
            asks += 1
            target, question = q.group(1).lower(), q.group(2).strip()
            try:
                answer = await wait_reply(self.env.ask(target, question), None, self.env.timeout, self.env.max_s)
            except Exception as e:
                answer = f"[{target} could not be reached: {_err(e)}]"
            self.env.emit("lookup", node=node, target=target, question=question, text=answer)
            msgs.append({"role": "user", "content": f"[{target} answered]\n{answer}\n\nNow finish your reply."})
            text = await wait_reply(provider.complete(list(msgs), system), provider, self.env.timeout, self.env.max_s)
            msgs.append({"role": "assistant", "content": text})
        fields, missing = read_fields(text, spec) if spec else ({}, [])
        if missing:  # web bots don't always follow the form: ask once more
            again = ("PRISM couldn't read " + ", ".join(m.upper() for m in missing)
                     + " from your reply. Reply with only these lines:" + output_request({k: spec[k] for k in missing}).split(":\n", 1)[1])
            msgs.append({"role": "user", "content": again})
            extra = await wait_reply(provider.complete(list(msgs), system), provider, self.env.timeout, self.env.max_s)
            msgs.append({"role": "assistant", "content": extra})
            more, missing = read_fields(extra, {k: spec[k] for k in missing})
            fields.update(more)
        memory["history"] = msgs[-HISTORY_KEEP:]
        memory["thread_url"] = getattr(provider, "thread_url", None) or memory.get("thread_url")
        memory["notes_seen"] = len(s.notes)
        latency = round(time.perf_counter() - started, 1)
        error = f"no {', '.join(m.upper() for m in missing)} in the reply" if missing else None
        self.env.emit("step", node=node, type="agent", label=name, agent=agent, prompt=full, text=text,
                      fields=fields, error=error, round=s.node_runs.get(node, 0) + 1, latency_s=latency,
                      provider=provider)
        if missing:
            raise StepError(error + " - continue to try the step again")
        return {"text": _strip_fields(text, spec) if spec else text, **fields}

    def _default_prompt(self, node: str) -> str:
        """No prompt given: the work of the steps that lead here."""
        srcs = [e.src for e in self.flow.incoming(node) if e.src in self.state.outputs]
        if not srcs:
            return "\n".join(f"{k}: {{{{input.{k}}}}}" for k in self.flow.inputs) or "Begin."
        return "\n\n".join(f"From {x}:\n{{{{{x}.text}}}}" for x in srcs)


def _strip_fields(text: str, spec: dict) -> str:
    """The reply without the field lines at its end (they are kept as fields)."""
    lines = text.rstrip().splitlines()
    tail = max(0, len(lines) - 30)
    for i in range(tail, len(lines)):
        m = FIELD_LINE.match(lines[i])
        if m and m.group(1).strip().lower().replace(" ", "_") in spec:
            return "\n".join(lines[:i]).rstrip() or text
    return text

