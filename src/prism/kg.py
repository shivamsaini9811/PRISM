"""The knowledge graph: the things in the user's work and how they connect.

Things (people, organisations, projects, products, places, decisions, documents, events) and
links between them ("Priya -leads-> Design", "Launch -planned for-> 12 May") are read out of
chats, folders and flow results by a model, and kept in PRISM's database. Every link keeps
where it came from (a chat message, a file, a flow run), so a bot can cite it and the user
can check, correct or remove it. Names that mean the same thing are merged ("Priya",
"Priya S." as an alias).

Used through `#kg` in a message (what is known about the things the message mentions),
`ASK @kg: <name or question>` for bots, a `tool: kg` flow step, and the Knowledge screen.
"""

import json
import re
import time
from dataclasses import dataclass

from .db import Store

TYPES = ("person", "organisation", "project", "product", "place", "decision", "document",
         "event", "topic", "fact")
EXTRACT_PROMPT = (
    "Read the material below and list the things in it and how they are linked, for the "
    "user's knowledge graph. Things: people, organisations, projects, products, places, "
    "decisions, documents, events and key topics or facts the user cares about. Links: short "
    "verb phrases (\"leads\", \"works at\", \"decided\", \"due on\", \"depends on\", \"prefers\").\n"
    "Only what the material states - no guesses, no general knowledge. Use the fullest name "
    "given; put other names for the same thing in aliases. The user is \"the user\".\n"
    "Reply with only one ```json block in this form:\n"
    '{{"entities": [{{"name": "Priya Shah", "type": "person", "aliases": ["Priya"], '
    '"summary": "leads design for the launch"}}], '
    '"links": [{{"src": "Priya Shah", "rel": "leads", "dst": "Launch website", '
    '"quote": "Priya leads the website work"}}]}}\n'
    "Types: {types}. Up to 40 entities and 60 links; nothing to list: empty lists.\n\n"
    "--- MATERIAL ({where}) ---\n{text}")
MAX_MATERIAL = 24_000


def norm(name: str) -> str:
    """How names are compared: lowercase, no punctuation, no leading 'the'."""
    n = re.sub(r"[^\w\s&+-]", " ", name.lower())
    n = re.sub(r"\s+", " ", n).strip()
    return re.sub(r"^the ", "", n)


class ExtractError(ValueError):
    pass


def parse_extraction(text: str) -> dict:
    """The model's JSON block, checked; raises ExtractError if it isn't usable."""
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    raw = (m.group(1) if m else text[text.find("{"):text.rfind("}") + 1]).strip()
    try:
        doc = json.loads(raw)
    except ValueError as e:
        raise ExtractError(f"not valid JSON ({e})") from e
    if not isinstance(doc, dict) or not isinstance(doc.get("entities", []), list) or not isinstance(doc.get("links", []), list):
        raise ExtractError("expected {\"entities\": [...], \"links\": [...]}")
    entities = []
    for e in doc.get("entities", [])[:80]:
        if isinstance(e, dict) and str(e.get("name", "")).strip():
            t = str(e.get("type", "topic")).lower()
            entities.append({"name": str(e["name"]).strip()[:120], "type": t if t in TYPES else "topic",
                             "aliases": [str(a).strip()[:120] for a in (e.get("aliases") or []) if str(a).strip()][:8],
                             "summary": str(e.get("summary") or "").strip()[:400]})
    links = []
    for l in doc.get("links", [])[:120]:
        if isinstance(l, dict) and all(str(l.get(k, "")).strip() for k in ("src", "rel", "dst")):
            links.append({"src": str(l["src"]).strip()[:120], "rel": str(l["rel"]).strip().lower()[:60],
                          "dst": str(l["dst"]).strip()[:120], "quote": str(l.get("quote") or "").strip()[:300]})
    return {"entities": entities, "links": links}


@dataclass
class Graph:
    store: Store

    # --- writing ---

    def find(self, name: str) -> dict | None:
        n = norm(name)
        if not n:
            return None
        rows = self.store._q("SELECT * FROM kg_entities WHERE norm = ?", (n,))
        if rows:
            return self._entity(rows[0])
        for r in self.store._q("SELECT * FROM kg_entities WHERE aliases LIKE ?", (f"%{json.dumps(n)[1:-1]}%",)):
            if n in [norm(a) for a in json.loads(r["aliases"] or "[]")]:
                return self._entity(r)
        return None

    def upsert(self, name: str, type_: str = "topic", aliases: list[str] = (), summary: str = "") -> int:
        found = self.find(name) or next((self.find(a) for a in aliases if self.find(a)), None)
        now = time.time()
        if found:
            al = list(dict.fromkeys(found["aliases"] + [a for a in [name, *aliases]
                                                        if norm(a) != found["norm"]]))[:20]
            # The fuller name wins as the shown name; a better summary replaces a missing one.
            shown = name if len(name) > len(found["name"]) and norm(found["name"]) in norm(name) else found["name"]
            if shown != found["name"]:
                al = [a for a in al if norm(a) != norm(shown)] + [found["name"]]
            self.store._q("UPDATE kg_entities SET name = ?, norm = ?, aliases = ?, summary = ?, "
                          "type = CASE WHEN type = 'topic' THEN ? ELSE type END, updated_at = ? WHERE id = ?",
                          (shown, norm(shown), json.dumps(al), summary or found["summary"], type_, now, found["id"]))
            return found["id"]
        return self.store._insert(
            "INSERT INTO kg_entities (type, name, norm, aliases, summary, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (type_, name, norm(name), json.dumps([a for a in aliases if norm(a) != norm(name)]), summary, now, now))

    def link(self, src: str, rel: str, dst: str, evidence: dict, confidence: float = 0.8) -> int | None:
        a, b = self.upsert(src), self.upsert(dst)
        if a == b:
            return None
        rel = re.sub(r"\s+", " ", rel.strip().lower())
        rows = self.store._q("SELECT * FROM kg_links WHERE src = ? AND rel = ? AND dst = ?", (a, rel, b))
        if rows:  # the same link again: one more piece of evidence
            ev = json.loads(rows[0]["evidence"] or "[]")
            if evidence not in ev:
                ev = (ev + [evidence])[-10:]
                self.store._q("UPDATE kg_links SET evidence = ?, confidence = MIN(1.0, confidence + 0.1) WHERE id = ?",
                              (json.dumps(ev), rows[0]["id"]))
            return rows[0]["id"]
        return self.store._insert(
            "INSERT INTO kg_links (src, rel, dst, evidence, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (a, rel, b, json.dumps([evidence]), confidence, time.time()))

    def add(self, extraction: dict, evidence: dict) -> dict:
        """Write what a model extracted; evidence says where it came from."""
        for e in extraction["entities"]:
            self.upsert(e["name"], e["type"], e["aliases"], e["summary"])
        n = 0
        for l in extraction["links"]:
            if self.link(l["src"], l["rel"], l["dst"], {**evidence, "quote": l["quote"]} if l["quote"] else evidence):
                n += 1
        return {"entities": len(extraction["entities"]), "links": n}

    def delete_entity(self, entity_id: int) -> None:
        self.store._q("DELETE FROM kg_links WHERE src = ? OR dst = ?", (entity_id, entity_id))
        self.store._q("DELETE FROM kg_entities WHERE id = ?", (entity_id,))

    def delete_link(self, link_id: int) -> None:
        self.store._q("DELETE FROM kg_links WHERE id = ?", (link_id,))

    def edit(self, entity_id: int, name: str | None = None, type_: str | None = None, summary: str | None = None) -> None:
        e = self.get(entity_id)
        if not e:
            return
        if name and name.strip() and name.strip() != e["name"]:
            other = self.find(name)
            if other and other["id"] != entity_id:
                return self.merge(entity_id, other["id"])
            self.store._q("UPDATE kg_entities SET name = ?, norm = ?, aliases = ? WHERE id = ?",
                          (name.strip(), norm(name), json.dumps(e["aliases"] + [e["name"]]), entity_id))
        if type_ in TYPES:
            self.store._q("UPDATE kg_entities SET type = ? WHERE id = ?", (type_, entity_id))
        if summary is not None:
            self.store._q("UPDATE kg_entities SET summary = ? WHERE id = ?", (summary.strip(), entity_id))

    def merge(self, from_id: int, into_id: int) -> None:
        """Two entries for the same thing: links and names move to `into`."""
        a, b = self.get(from_id), self.get(into_id)
        if not a or not b or from_id == into_id:
            return
        for l in self.store._q("SELECT * FROM kg_links WHERE src = ? OR dst = ?", (from_id, from_id)):
            src = into_id if l["src"] == from_id else l["src"]
            dst = into_id if l["dst"] == from_id else l["dst"]
            self.store._q("DELETE FROM kg_links WHERE id = ?", (l["id"],))
            if src != dst:
                for ev in json.loads(l["evidence"] or "[]") or [{}]:
                    self._link_ids(src, l["rel"], dst, ev, l["confidence"])
        aliases = list(dict.fromkeys(b["aliases"] + [a["name"]] + a["aliases"]))
        self.store._q("UPDATE kg_entities SET aliases = ?, summary = ? WHERE id = ?",
                      (json.dumps([x for x in aliases if norm(x) != b["norm"]]), b["summary"] or a["summary"], into_id))
        self.store._q("DELETE FROM kg_entities WHERE id = ?", (from_id,))

    def _link_ids(self, a: int, rel: str, b: int, evidence: dict, confidence: float) -> None:
        rows = self.store._q("SELECT * FROM kg_links WHERE src = ? AND rel = ? AND dst = ?", (a, rel, b))
        if rows:
            ev = json.loads(rows[0]["evidence"] or "[]")
            if evidence and evidence not in ev:
                self.store._q("UPDATE kg_links SET evidence = ? WHERE id = ?", (json.dumps(ev + [evidence]), rows[0]["id"]))
            return
        self.store._insert("INSERT INTO kg_links (src, rel, dst, evidence, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                           (a, rel, b, json.dumps([evidence] if evidence else []), confidence, time.time()))

    # --- reading ---

    def _entity(self, r) -> dict:
        return {"id": r["id"], "type": r["type"], "name": r["name"], "norm": r["norm"],
                "aliases": json.loads(r["aliases"] or "[]"), "summary": r["summary"] or "",
                "updated_at": r["updated_at"]}

    def get(self, entity_id: int) -> dict | None:
        rows = self.store._q("SELECT * FROM kg_entities WHERE id = ?", (entity_id,))
        return self._entity(rows[0]) if rows else None

    def links_of(self, entity_id: int) -> list[dict]:
        rows = self.store._q(
            "SELECT l.*, s.name AS src_name, d.name AS dst_name, s.type AS src_type, d.type AS dst_type"
            " FROM kg_links l JOIN kg_entities s ON s.id = l.src JOIN kg_entities d ON d.id = l.dst"
            " WHERE l.src = ? OR l.dst = ? ORDER BY l.confidence DESC, l.id DESC", (entity_id, entity_id))
        return [{"id": r["id"], "src": r["src"], "dst": r["dst"], "rel": r["rel"], "src_name": r["src_name"],
                 "dst_name": r["dst_name"], "src_type": r["src_type"], "dst_type": r["dst_type"],
                 "confidence": r["confidence"], "evidence": json.loads(r["evidence"] or "[]")} for r in rows]

    def search(self, q: str = "", limit: int = 200) -> list[dict]:
        if q.strip():
            like = f"%{norm(q)}%"
            rows = self.store._q("SELECT e.*, (SELECT COUNT(*) FROM kg_links l WHERE l.src = e.id OR l.dst = e.id) AS n"
                                 " FROM kg_entities e WHERE e.norm LIKE ? OR e.aliases LIKE ? OR e.summary LIKE ?"
                                 " ORDER BY n DESC, e.name LIMIT ?", (like, f"%{q.strip()}%", f"%{q.strip()}%", limit))
        else:
            rows = self.store._q("SELECT e.*, (SELECT COUNT(*) FROM kg_links l WHERE l.src = e.id OR l.dst = e.id) AS n"
                                 " FROM kg_entities e ORDER BY e.updated_at DESC LIMIT ?", (limit,))
        return [{**self._entity(r), "links": r["n"]} for r in rows]

    def counts(self) -> dict:
        return {"entities": self.store._q("SELECT COUNT(*) AS n FROM kg_entities")[0]["n"],
                "links": self.store._q("SELECT COUNT(*) AS n FROM kg_links")[0]["n"]}

    def mentioned(self, text: str, limit: int = 8) -> list[dict]:
        """Entities whose name or an alias appears in the text, as whole words."""
        t = " " + norm(text) + " "
        found = []
        for r in self.store._q("SELECT * FROM kg_entities"):
            names = [r["norm"]] + [norm(a) for a in json.loads(r["aliases"] or "[]")]
            if any(len(n) >= 3 and f" {n} " in t for n in names if n):
                found.append(self._entity(r))
        found.sort(key=lambda e: -len(e["norm"]))
        return found[:limit]

    def describe(self, entities: list[dict], max_links: int = 12) -> str:
        """What is known about these things, as the bots read it (with where it came from)."""
        out = []
        for e in entities:
            head = f"{e['name']} ({e['type']})" + (f" - {e['summary']}" if e["summary"] else "")
            if e["aliases"]:
                head += f" [also: {', '.join(e['aliases'][:4])}]"
            lines = [head]
            for l in self.links_of(e["id"])[:max_links]:
                src = next((ev for ev in l["evidence"] if ev.get("quote")), (l["evidence"] or [{}])[0])
                where = src.get("where") or ""
                quote = f' "{src["quote"]}"' if src.get("quote") else ""
                lines.append(f"  - {l['src_name']} -{l['rel']}-> {l['dst_name']}{quote}{f' ({where})' if where else ''}")
            out.append("\n".join(lines))
        return "\n".join(out)

    def answer(self, question: str) -> str:
        """ASK @kg: what is known about the things the question names (or matches)."""
        found = self.mentioned(question) or [e for e in self.search(question, 5)]
        if not found:
            words = [w for w in re.findall(r"[\w-]{4,}", question.lower())][:6]
            seen: dict[int, dict] = {}
            for w in words:
                for e in self.search(w, 3):
                    seen.setdefault(e["id"], e)
            found = list(seen.values())[:5]
        if not found:
            return "The knowledge graph has nothing about that yet."
        return "From the user's knowledge graph:\n" + self.describe(found)
