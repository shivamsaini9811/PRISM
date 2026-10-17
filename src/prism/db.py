"""SQLite store: vault meta, encrypted credentials, mother/daughter chats, messages, context."""

import json
import re
import sqlite3
import threading
import time
from pathlib import Path

from cryptography.fernet import Fernet

from . import crypto

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BLOB);
CREATE TABLE IF NOT EXISTS credentials (
    platform   TEXT PRIMARY KEY,
    model      TEXT NOT NULL,
    secret_enc BLOB,
    base_url   TEXT,
    enabled    INTEGER NOT NULL DEFAULT 1,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chats (
    id         INTEGER PRIMARY KEY,
    title      TEXT NOT NULL,
    primary_platform TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
-- One daughter thread per member (platform, or "moderator"), bound to its mother (room) chat.
CREATE TABLE IF NOT EXISTS daughter_chats (
    id         INTEGER PRIMARY KEY,
    chat_id    INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    platform   TEXT NOT NULL,
    model      TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE (chat_id, platform)
);
-- daughter_id NULL = message in the mother thread.
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY,
    chat_id     INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    daughter_id INTEGER REFERENCES daughter_chats(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    meta_json   TEXT,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_thread ON messages (chat_id, daughter_id, id);
-- chat_id NULL = global context shared with every chat.
CREATE TABLE IF NOT EXISTS context_items (
    id         INTEGER PRIMARY KEY,
    chat_id    INTEGER REFERENCES chats(id) ON DELETE CASCADE,
    title      TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


# Versioned migrations applied on top of SCHEMA; index i upgrades user_version i -> i+1.
MIGRATIONS = [
    # v1: browser transport + binding of a daughter thread to a real remote conversation.
    """
    ALTER TABLE credentials ADD COLUMN transport TEXT NOT NULL DEFAULT 'api';
    ALTER TABLE daughter_chats ADD COLUMN remote_url TEXT;
    """,
    # v2: group chat - each member's bookmark: the last room message it was sent.
    """
    ALTER TABLE daughter_chats ADD COLUMN seen_message_id INTEGER;
    """,
    # v3: named context (#name). Items from before were sent with every message: keep that.
    """
    ALTER TABLE context_items ADD COLUMN name TEXT;
    ALTER TABLE context_items ADD COLUMN always INTEGER NOT NULL DEFAULT 0;
    UPDATE context_items SET always = 1;
    """,
]


def context_name(text: str) -> str:
    """A #name from a title: lowercase letters, digits, - and _."""
    slug = re.sub(r"[^a-z0-9_-]+", "-", text.lower()).strip("-")
    return slug if slug and slug[0].isalpha() else f"c-{slug}".rstrip("-")


class Store:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._lock = threading.Lock()

    def _migrate(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        for i, script in enumerate(MIGRATIONS[version:], start=version):
            self._conn.executescript(script)
            self._conn.execute(f"PRAGMA user_version = {i + 1}")
            self._conn.commit()

    def _q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
            self._conn.commit()
            return rows

    def _insert(self, sql: str, args: tuple) -> int:
        with self._lock:
            cur = self._conn.execute(sql, args)
            self._conn.commit()
            return cur.lastrowid

    # --- meta / vault ---------------------------------------------------------------

    def get_meta(self, key: str):
        rows = self._q("SELECT value FROM meta WHERE key = ?", (key,))
        return rows[0]["value"] if rows else None

    def set_meta(self, key: str, value) -> None:
        self._q("INSERT INTO meta VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value))

    def is_initialized(self) -> bool:
        return self.get_meta("salt") is not None

    def unlock(self, password: str) -> Fernet:
        """Create the vault on first use, otherwise verify the master password."""
        if not self.is_initialized():
            salt = crypto.new_salt()
            f = crypto.derive_fernet(password, salt)
            self.set_meta("salt", salt)
            self.set_meta("verifier", crypto.make_verifier(f))
            return f
        f = crypto.derive_fernet(password, self.get_meta("salt"))
        crypto.check_verifier(f, self.get_meta("verifier"))
        return f

    # --- credentials ----------------------------------------------------------------

    def upsert_credential(self, f: Fernet, platform: str, model: str, secret: str | None,
                          base_url: str | None, enabled: bool, transport: str = "api") -> None:
        """secret=None keeps the stored secret; an empty string clears it."""
        existing = self._q("SELECT secret_enc FROM credentials WHERE platform = ?", (platform,))
        if secret is None:
            secret_enc = existing[0]["secret_enc"] if existing else None
        else:
            secret_enc = f.encrypt(secret.encode()) if secret else None
        self._q(
            """INSERT INTO credentials
                 (platform, model, secret_enc, base_url, enabled, updated_at, transport)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(platform) DO UPDATE SET model = excluded.model,
                 secret_enc = excluded.secret_enc, base_url = excluded.base_url,
                 enabled = excluded.enabled, updated_at = excluded.updated_at,
                 transport = excluded.transport""",
            (platform, model, secret_enc, base_url, int(enabled), time.time(), transport),
        )

    def list_credentials(self) -> list[dict]:
        """Public view - never includes the secret."""
        return [
            {"platform": r["platform"], "model": r["model"], "base_url": r["base_url"],
             "enabled": bool(r["enabled"]), "has_secret": r["secret_enc"] is not None,
             "transport": r["transport"]}
            for r in self._q("SELECT * FROM credentials ORDER BY platform")
        ]

    def enabled_credentials(self, f: Fernet) -> list[dict]:
        out = []
        for r in self._q("SELECT * FROM credentials WHERE enabled = 1 ORDER BY platform"):
            secret = f.decrypt(r["secret_enc"]).decode() if r["secret_enc"] else None
            out.append({"platform": r["platform"], "model": r["model"],
                        "base_url": r["base_url"], "secret": secret, "transport": r["transport"]})
        return out

    def delete_credential(self, platform: str) -> None:
        self._q("DELETE FROM credentials WHERE platform = ?", (platform,))

    # --- chats (mother) and daughter bindings -----------------------------------------

    def create_chat(self, title: str = "New chat") -> int:
        now = time.time()
        return self._insert("INSERT INTO chats (title, created_at, updated_at) VALUES (?, ?, ?)",
                            (title, now, now))

    def list_chats(self) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM chats ORDER BY updated_at DESC")]

    def get_chat(self, chat_id: int) -> dict | None:
        rows = self._q("SELECT * FROM chats WHERE id = ?", (chat_id,))
        return dict(rows[0]) if rows else None

    def update_chat(self, chat_id: int, **fields) -> None:
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._q(f"UPDATE chats SET {cols} WHERE id = ?", (*fields.values(), chat_id))

    def delete_chat(self, chat_id: int) -> None:
        self._q("DELETE FROM chats WHERE id = ?", (chat_id,))

    def bind_daughter(self, chat_id: int, platform: str, model: str) -> int:
        """Return the daughter thread for this platform, creating the binding if needed."""
        rows = self._q("SELECT id FROM daughter_chats WHERE chat_id = ? AND platform = ?",
                       (chat_id, platform))
        if rows:
            self._q("UPDATE daughter_chats SET model = ? WHERE id = ?", (model, rows[0]["id"]))
            return rows[0]["id"]
        return self._insert(
            "INSERT INTO daughter_chats (chat_id, platform, model, created_at) VALUES (?, ?, ?, ?)",
            (chat_id, platform, model, time.time()))

    def set_remote_url(self, daughter_id: int, url: str | None) -> None:
        self._q("UPDATE daughter_chats SET remote_url = ? WHERE id = ?", (url, daughter_id))

    def set_seen(self, daughter_id: int, message_id: int) -> None:
        self._q("UPDATE daughter_chats SET seen_message_id = ? WHERE id = ?", (message_id, daughter_id))

    def daughters(self, chat_id: int) -> list[dict]:
        return [dict(r) for r in self._q(
            "SELECT * FROM daughter_chats WHERE chat_id = ? ORDER BY platform", (chat_id,))]

    # --- messages -------------------------------------------------------------------

    def add_message(self, chat_id: int, daughter_id: int | None, role: str, content: str,
                    meta: dict | None = None) -> int:
        return self._insert(
            "INSERT INTO messages (chat_id, daughter_id, role, content, meta_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, daughter_id, role, content, json.dumps(meta) if meta else None, time.time()))

    def update_message(self, message_id: int, meta: dict) -> None:
        self._q("UPDATE messages SET meta_json = ? WHERE id = ?", (json.dumps(meta), message_id))

    def thread(self, chat_id: int, daughter_id: int | None = None) -> list[dict]:
        rows = self._q(
            "SELECT * FROM messages WHERE chat_id = ? AND daughter_id IS ? ORDER BY id",
            (chat_id, daughter_id))
        return [{"id": r["id"], "role": r["role"], "content": r["content"],
                 "meta": json.loads(r["meta_json"]) if r["meta_json"] else None,
                 "created_at": r["created_at"]} for r in rows]

    def user_messages_after(self, after_id: int, limit: int = 60) -> list[dict]:
        """What the user wrote in any chat (room messages), newest `limit`, oldest first."""
        rows = self._q(
            "SELECT m.id, m.content, m.meta_json, m.created_at, c.title FROM messages m"
            " JOIN chats c ON c.id = m.chat_id WHERE m.id > ? AND m.daughter_id IS NULL"
            " AND m.role = 'user' AND m.content != '' ORDER BY m.id DESC LIMIT ?", (after_id, limit * 3))
        said = [dict(r) for r in rows  # not the #context items attached to a message
                if (json.loads(r["meta_json"] or "{}")).get("kind") in ("topic", "user")]
        return list(reversed(said[:limit]))

    def search_messages(self, words: list[str], exclude_chat: int | None = None,
                        limit: int = 6, scan: int = 400) -> list[dict]:
        """Room messages from any chat containing the words, best match first (most words,
        then newest). Plain LIKE: no full-text index needed."""
        if not words:
            return []
        where = " OR ".join("m.content LIKE ?" for _ in words)
        rows = self._q(
            "SELECT m.id, m.chat_id, m.role, m.content, m.meta_json, m.created_at, c.title"
            " FROM messages m JOIN chats c ON c.id = m.chat_id"
            f" WHERE m.daughter_id IS NULL AND m.content != '' AND m.chat_id IS NOT ? AND ({where})"
            " ORDER BY m.id DESC LIMIT ?",
            (exclude_chat, *(f"%{w}%" for w in words), scan))
        def score(r):
            text = r["content"].lower()
            return sum(w.lower() in text for w in words)
        best = sorted(rows, key=lambda r: (-score(r), -r["id"]))[:limit]
        return [{**dict(r), "meta": json.loads(r["meta_json"]) if r["meta_json"] else {}}
                for r in best]

    # --- context store --------------------------------------------------------------

    def add_context(self, chat_id: int | None, title: str, content: str,
                    name: str | None = None, always: bool = False) -> int:
        """Write named context: a #name already in that scope (this chat, or global) is
        replaced, so writing the same name again updates it."""
        name = context_name(name or title)
        existing = self._q("SELECT id FROM context_items WHERE name = ? AND chat_id IS ?",
                           (name, chat_id))
        if existing:
            self._q("UPDATE context_items SET title = ?, content = ?, always = ?, created_at = ?"
                    " WHERE id = ?", (title, content, int(always), time.time(), existing[0]["id"]))
            return existing[0]["id"]
        return self._insert(
            "INSERT INTO context_items (chat_id, title, content, name, always, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)", (chat_id, title, content, name, int(always), time.time()))

    def context_for(self, chat_id: int) -> list[dict]:
        """Global items first, then this chat's items."""
        rows = self._q("SELECT * FROM context_items WHERE chat_id IS NULL OR chat_id = ?"
                       " ORDER BY chat_id IS NOT NULL, id", (chat_id,))
        return [{**dict(r), "name": r["name"] or context_name(r["title"]),
                 "always": bool(r["always"])} for r in rows]

    def find_context(self, chat_id: int, name: str) -> dict | None:
        """#name as this chat sees it: its own item wins over a global one."""
        items = [c for c in self.context_for(chat_id) if c["name"] == name]
        return items[-1] if items else None

    def delete_context(self, item_id: int) -> None:
        self._q("DELETE FROM context_items WHERE id = ?", (item_id,))
