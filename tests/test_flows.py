"""Flows: parsing and checks, conditions, and runs with scripted agents."""

import asyncio

import pytest

from prism import flows
from prism.flows import FlowError, Run, RunEnv, RunState, evaluate, parse, read_fields

BRIEF = """
name: market-brief
inputs: {topic: text}
agents:
  writer: {model: "@gpt", instructions: "Write clearly."}
  critic: {model: "@claude", instructions: "Find what is wrong.", output: {verdict: PASS|FIX, notes: text}}
nodes:
  gather:  {agent: writer, prompt: "Research {{input.topic}}."}
  draft:   {agent: writer, prompt: "Write it from {{gather.text}}. Fix: {{review.notes}}"}
  review:  {agent: critic, prompt: "Review {{draft.text}}."}
  publish: {ask_you: "Publish this?", show: "{{draft.text}}"}
edges:
  - gather -> draft
  - draft -> review
  - review -> draft:   {when: "verdict == FIX", max_rounds: 2}
  - review -> publish: {when: "verdict == PASS or rounds_done"}
"""


def test_parse_both_edge_spellings_layers_and_loops():
    f = parse(BRIEF)
    assert f.starts == ["gather"]
    assert [(e.src, e.dst, e.max_rounds, e.back) for e in f.edges] == [
        ("gather", "draft", None, False), ("draft", "review", None, False),
        ("review", "draft", 2, True), ("review", "publish", None, False)]
    drawn = f.to_json()
    assert {n["id"]: n["level"] for n in drawn["nodes"]} == {"gather": 0, "draft": 1, "review": 2, "publish": 3}
    chain = parse("name: c\nnodes: {a: {agent: x}, b: {agent: x}, c: {agent: x}}\nagents: {x: {model: '@gpt'}}\nedges: [a -> b -> c]")
    assert [(e.src, e.dst) for e in chain.edges] == [("a", "b"), ("b", "c")]


def test_parse_reports_every_problem():
    with pytest.raises(FlowError) as e:
        parse("""
name: bad
nodes:
  a: {agent: nobody, prompt: "{{input.x}} {{ghost.text}}"}
  b: {tool: hammer}
  c: {agent: nobody, tool: docs}
  d: {wait: soon}
edges:
  - a -> zz
  - a -> b: {when: "__import__('os')", max_rounds: 0}
""")
    text = "\n".join(e.value.errors)
    for bit in ["no agent named 'nobody'", "tool is one of", "exactly one of", "wait takes a time",
                "no node named 'zz'", "only comparisons", "max_rounds is a whole number"]:
        assert bit in text, bit
    with pytest.raises(FlowError, match="not valid YAML"):
        parse("name: x\nnodes: [unclosed")
    with pytest.raises(FlowError, match=r"loop a -> b -> a has no limit"):
        parse("name: x\nagents: {w: {model: '@gpt'}}\nnodes: {s: {agent: w}, a: {agent: w}, b: {agent: w}}\nedges: [s -> a, a -> b, b -> a]")
    with pytest.raises(FlowError, match=r"input.x.*not one of the inputs"):
        parse("name: x\nagents: {w: {model: '@gpt'}}\nnodes: {a: {agent: w, prompt: '{{input.x}}'}}")


def test_conditions():
    assert evaluate("verdict == PASS", {"verdict": "pass."})
    assert evaluate("verdict != PASS and not rounds_done", {"verdict": "FIX", "rounds_done": False})
    assert evaluate("score >= 7", {"score": "8"}) and not evaluate("score >= 7", {"score": "six"})
    assert evaluate("'urgent' in notes", {"notes": "This is URGENT"})
    for bad in ["__import__('os').system('x')", "a.b", "f(x)", "[1][0]", "x + 1"]:
        with pytest.raises(FlowError):
            evaluate(bad, {})


def test_read_fields():
    spec = flows.output_spec({"verdict": "PASS|FIX", "notes": "text"})
    reply = "The draft is close.\n\n**VERDICT:** fix\nNOTES: tighten the intro\nand cite sources"
    assert read_fields(reply, spec) == ({"verdict": "FIX", "notes": "tighten the intro\nand cite sources"}, [])
    assert read_fields("Looks fine to me.", spec) == ({}, ["verdict", "notes"])
    assert flows._strip_fields(reply, spec) == "The draft is close."


# --- runs with scripted agents ---------------------------------------------------------------

class Scripted:
    """Answers from a list per agent; remembers what it was sent."""

    def __init__(self, name, replies):
        self.name, self.model, self.replies = name, f"{name}-fake", replies
        self.calls = []
        self.thread_url = None

    async def complete(self, messages, system=None):
        self.calls.append((list(messages), system))
        self.thread_url = f"https://x.example/{self.name}"
        r = self.replies.pop(0) if self.replies else "ok"
        if isinstance(r, Exception):
            raise r
        if callable(r):
            return await r(messages)
        return r


class Env(RunEnv):
    timeout = 5

    def __init__(self, replies: dict[str, list]):
        self.providers = {name: Scripted(name, list(r)) for name, r in replies.items()}
        self.events, self.saved = [], []

    def provider(self, agent, memory):
        return self.providers[agent["name"]]

    def askable(self, agent):
        return ["docs"] if "docs" in (agent.get("tools") or []) else []

    async def ask(self, target, question):
        return f"passage about {question}"

    async def tool(self, name, query):
        return f"{name} found: {query}"

    def emit(self, kind, **data):
        self.events.append((kind, {k: v for k, v in data.items() if k not in ("agent", "provider")}))

    def save(self, state):
        self.saved.append(state.to_dict())


def steps(env):
    return [(d["node"], d.get("fields")) for k, d in env.events if k == "step"]


async def test_loop_until_pass_then_ask_you():
    env = Env({"writer": ["notes on EVs", "draft 1", "draft 2"],
               "critic": ["Weak.\nVERDICT: FIX\nNOTES: add numbers", "Good.\nVERDICT: PASS\nNOTES: none"]})
    run = Run.start(parse(BRIEF), {"topic": "EV market"}, env)
    s = await run.run()
    assert [n for n, _ in steps(env)] == ["gather", "draft", "review", "draft", "review"]
    assert s.status == "waiting" and s.waiting == "publish"
    ask = next(d for k, d in env.events if k == "ask_you")
    assert ask["show"] == "draft 2" and ask["question"] == "Publish this?"
    writer = env.providers["writer"]
    assert "Research EV market." in writer.calls[0][0][-1]["content"]
    assert "Fix: add numbers" in writer.calls[2][0][-1]["content"]          # the review's notes reach the redraft
    assert len(writer.calls[2][0]) == 5                                     # its own conversation continues
    critic = env.providers["critic"]
    assert "VERDICT: PASS or FIX" in critic.calls[0][0][-1]["content"] and "Find what is wrong" in critic.calls[0][1]

    assert run.answer("publish", approved=True, text="ship it")
    s = await run.run()
    assert s.status == "done" and s.outputs["publish"] == {"approved": "yes", "text": "ship it"}


async def test_loop_stops_at_max_rounds():
    env = Env({"writer": ["n", "d1", "d2", "d3"],
               "critic": ["VERDICT: FIX\nNOTES: a", "VERDICT: FIX\nNOTES: b", "VERDICT: FIX\nNOTES: c"]})
    run = Run.start(parse(BRIEF), {"topic": "t"}, env)
    s = await run.run()
    assert [n for n, _ in steps(env)].count("review") == 3  # 1 + max_rounds 2
    assert s.waiting == "publish"  # rounds_done lets it out


async def test_parallel_join_all_and_tools():
    f = parse("""
name: p
inputs: [q]
agents: {a: {model: "@gpt"}, b: {model: "@claude"}, sum: {model: "@gpt"}}
nodes:
  start: {tool: docs, query: "{{input.q}}"}
  left:  {agent: a, prompt: "left {{start.text}}"}
  right: {agent: b, prompt: "right"}
  merge: {agent: sum, join: all, prompt: "{{left.text}} + {{right.text}}"}
edges: [start -> left, start -> right, left -> merge, right -> merge]
""")
    order = []

    def slow(tag, delay):
        async def go(messages):
            await asyncio.sleep(delay)
            order.append(tag)
            return tag
        return go
    env = Env({"a": [slow("L", 0.05)], "b": [slow("R", 0.0)], "sum": ["merged"]})
    s = await Run.start(f, {"q": "pricing"}, env).run()
    assert s.status == "done" and order == ["R", "L"]   # ran side by side
    assert env.providers["sum"].calls[0][0][-1]["content"] == "L + R"
    assert "left docs found: pricing" in env.providers["a"].calls[0][0][-1]["content"]
    assert [n for n, _ in steps(env)].count("merge") == 1


async def test_missing_field_retried_then_paused_and_resumed_from_saved_state():
    f = parse("name: r\nagents: {c: {model: '@claude', output: {verdict: PASS|FIX}}}\nnodes: {check: {agent: c}}")
    env = Env({"c": ["I think it's fine.", "Still no form."]})
    run = Run.start(f, {}, env)
    s = await run.run()
    assert s.status == "paused" and "no VERDICT" in s.error and s.queue == ["check"]
    assert "Reply with only these lines" in env.providers["c"].calls[1][0][-1]["content"]

    # PRISM restarts: a new Run from the saved state reruns the step that failed.
    env2 = Env({"c": ["VERDICT: pass"]})
    again = Run(f, RunState.from_dict(env.saved[-1]), env2)
    s = await again.run()
    assert s.status == "done" and s.outputs["check"]["verdict"] == "PASS"


async def test_budget_pauses_and_more_budget_continues():
    src = "name: b\nagents: {w: {model: '@gpt'}}\nbudget: {steps: 2}\nnodes: {a: {agent: w}, b: {agent: w}, c: {agent: w}}\nedges: [a -> b -> c]"
    env = Env({"w": ["1", "2", "3"]})
    run = Run.start(parse(src), {}, env)
    s = await run.run()
    assert s.status == "paused" and s.error.startswith("budget reached: 2 steps") and s.queue == ["c"]
    run.resume(more_budget=True)
    assert (await run.run()).status == "done"


async def test_notes_asks_and_stop():
    f = parse("name: n\nagents: {w: {model: '@gpt', tools: [docs]}}\nnodes: {a: {agent: w, prompt: go}, b: {agent: w, prompt: again}}\nedges: [a -> b]")
    env = Env({"w": ["ASK @docs: refund policy", "done a", "done b"]})
    run = Run.start(f, {}, env)
    run.add_note("keep it short")
    s = await run.run()
    calls = env.providers["w"].calls
    assert "The user added" in calls[0][0][-1]["content"] and "keep it short" in calls[0][0][-1]["content"]
    assert "passage about refund policy" in calls[1][0][-1]["content"]
    assert "The user added" not in calls[2][0][-1]["content"]  # each note once
    assert s.outputs["a"]["text"] == "done a" and ("lookup", {"node": "a", "target": "docs", "question": "refund policy",
                                                    "text": "passage about refund policy"}) in env.events

    gate = asyncio.Event()

    async def hang(messages):
        await gate.wait()
        return "never"
    env = Env({"w": [hang]})
    run = Run.start(f, {}, env)
    task = asyncio.create_task(run.run())
    await asyncio.sleep(0.05)
    run.stop()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert run.state.status == "stopped" and run.state.queue == ["a"]


async def test_wait_node_and_subflow():
    child = "name: child\ninputs: [topic]\nagents: {w: {model: '@gpt'}}\nnodes: {one: {agent: w, prompt: 'child on {{input.topic}}'}, two: {agent: w}}\nedges: [one -> two]"
    parent = """
name: parent
inputs: [subject]
agents: {w: {model: "@gpt"}}
nodes:
  pause: {wait: 0.05s}
  sub:   {flow: child, inputs: {topic: "{{input.subject}}"}}
  last:  {agent: w, prompt: "final from {{sub.text}}"}
edges: [pause -> sub -> last]
"""
    f = parse(parent, flows=lambda n: child if n == "child" else None)
    assert set(f.nodes) == {"pause", "sub/one", "sub/two", "last"}
    env = Env({"w": ["c1", "c2", "end"]})
    s = await Run.start(f, {"subject": "solar"}, env).run()
    calls = env.providers["w"].calls
    assert calls[0][0][-1]["content"] == "child on solar" and calls[2][0][-1]["content"] == "final from c2"
    assert s.status == "done" and s.outputs["pause"]["text"] == "waited 0.05s"
    with pytest.raises(FlowError, match="nest at most"):
        parse("name: me\nnodes: {x: {flow: me}}", flows=lambda n: "name: me\nnodes: {x: {flow: me}}")


async def test_run_step_and_agent_runs(tmp_path):
    import sys
    from prism.runner import execute
    with pytest.raises(FlowError, match="tool run needs"):
        parse("name: r\nnodes: {x: {tool: run}}")
    f = parse("""
name: r
agents: {w: {model: "@gpt", tools: [run]}}
nodes:
  calc: {tool: run, lang: python, code: "print(len('{{input.word}}'))"}
  say:  {agent: w, prompt: "The count was {{calc.text}}"}
edges: [calc -> say]
inputs: [word]
""")

    class RunEnv_(Env):
        def can_run(self, agent):
            return ["python"]

        async def run(self, who, req):
            return execute(req, tmp_path, {"python": sys.executable}, 30)
    env = RunEnv_({"w": ["RUN @python\n```python\nprint(2 + 2)\n```", "It is 4."]})
    s = await Run.start(f, {"word": "prism"}, env).run()
    assert s.status == "done" and s.outputs["calc"]["text"].strip() == "5"
    w = env.providers["w"]
    assert w.calls[0][0][-1]["content"].startswith("The count was 5") and "RUN @<language>" in w.calls[0][1]
    assert "[ran python" in w.calls[1][0][-1]["content"] and s.outputs["say"]["text"] == "It is 4."
