"""Building with words: "@prism make an agent that…", "create a flow that…", "remember…".

The primary model reads the request and answers with actions as JSON - create an agent,
create a flow (described in plain words; the flow itself is then drafted, checked and saved
as YAML), run a flow, remember facts in the knowledge graph, look something up there, or
just explain how to do it. PRISM carries them out and shows a card with what it did.

Messages that start like a builder request ("create an agent…", "make a workflow…", "run the
flow…", "remember that…") are treated as one even without @prism.
"""

import json
import re

ACTIONS = ("create_agent", "create_flow", "run_flow", "remember", "recall", "explain")

# Without @prism: only requests that can't be meant for the bots.
INTENT = re.compile(
    r"^\s*(?:please\s+|can you\s+|could you\s+|i want (?:you )?to\s+|i need\s+)?"
    r"(?:(?:create|make|build|set ?up|add|define|design|draft|new)\b.{0,80}?\b(?:agents?|flows?|workflows?|pipelines?|automations?)\b"
    r"|run (?:the |my )?(?:flow|workflow)\b"
    r"|(?:remember|note down|save to (?:the )?knowledge|add to (?:the |my )?knowledge(?: graph)?)\b)",
    re.I | re.S)

PROMPT = (
    "You turn a user's request into actions for PRISM, an app where the user chats with AI "
    "models. PRISM has: agents (a saved role on one of the user's models, @named in chats), "
    "flows (workflows of steps that run on their own: agents answer, results pass along, a "
    "review can send work back, it can stop to ask the user, run Python, search the user's "
    "files), and a knowledge graph (facts about the user's people, projects, decisions).\n\n"
    "The user's models: {members}\nAgents that exist: {agents}\nFlows that exist (name: inputs): {flows}\n\n"
    "The request:\n{text}\n\n"
    "Reply with only one ```json block: {{\"actions\": [ ... ]}} using these actions:\n"
    '{{"action": "create_agent", "name": "critic", "model": "@claude", "instructions": "what it does, '
    'in the second person, specific", "tools": [], "output": {{}}, "research": false}}\n'
    '{{"action": "create_flow", "description": "the whole workflow in plain words: its steps in order, '
    'who does each, any loop and its limit, where it asks the user, its inputs"}}\n'
    '{{"action": "run_flow", "flow": "an existing flow", "inputs": {{"input name": "value"}}}}\n'
    '{{"action": "remember", "text": "the facts to keep, as plain sentences"}}\n'
    '{{"action": "recall", "question": "what to look up in the knowledge graph"}}\n'
    '{{"action": "explain", "answer": "a short answer"}} - for questions about using PRISM, or when '
    "nothing else fits\n"
    "Rules: agent names are 2-24 lowercase letters, digits or -, and not a model's name; pick the "
    "model the user names, else the best fit from the user's models. tools may include: ask (ask "
    "the user's other assistants), docs (search their folders), prism (search past chats), kg "
    "(the knowledge graph), run (run Python on their computer). output is only for agents that must "
    "give a verdict or score, e.g. {{\"verdict\": \"PASS|FIX\", \"notes\": \"text\"}}. research true "
    "only if the user asks for deep research. Several actions are fine (an agent, then a flow that "
    "uses it). Don't invent requirements the user didn't give.")


class BuildError(ValueError):
    pass


def parse_actions(reply: str) -> list[dict]:
    m = re.search(r"```(?:json)?\s*\n(.*?)```", reply, re.S)
    raw = (m.group(1) if m else reply[reply.find("{"):reply.rfind("}") + 1]).strip()
    try:
        doc = json.loads(raw)
    except ValueError as e:
        raise BuildError(f"not valid JSON ({e})") from e
    actions = doc.get("actions") if isinstance(doc, dict) else doc
    if isinstance(doc, dict) and "action" in doc:
        actions = [doc]
    if not isinstance(actions, list) or not actions:
        raise BuildError('expected {"actions": [...]}')
    out = []
    for a in actions[:6]:
        if not isinstance(a, dict) or a.get("action") not in ACTIONS:
            raise BuildError(f"unknown action {a.get('action') if isinstance(a, dict) else a!r}")
        out.append(a)
    return out


def is_request(text: str) -> bool:
    return bool(INTENT.match(text))
