import pytest

from prism.crypto import WrongPassword
from prism.db import Store


@pytest.fixture
def store():
    return Store(":memory:")


def test_vault_create_unlock_and_wrong_password(store):
    assert not store.is_initialized()
    f1 = store.unlock("correct horse")
    assert store.is_initialized()
    f2 = store.unlock("correct horse")
    assert f2.decrypt(f1.encrypt(b"x")) == b"x"
    with pytest.raises(WrongPassword):
        store.unlock("wrong password")


def test_secret_encrypted_kept_and_cleared(store):
    f = store.unlock("correct horse")
    store.upsert_credential(f, "claude", "m1", "sk-secret", None, True)
    raw = store._q("SELECT secret_enc FROM credentials")[0]["secret_enc"]
    assert b"sk-secret" not in raw
    store.upsert_credential(f, "claude", "m2", None, None, True)  # keep
    assert store.enabled_credentials(f)[0] == {"platform": "claude", "model": "m2",
                                               "base_url": None, "secret": "sk-secret",
                                               "transport": "api"}
    assert store.list_credentials()[0]["has_secret"] and "secret" not in store.list_credentials()[0]
    store.upsert_credential(f, "claude", "m2", "", None, True)  # clear
    assert store.enabled_credentials(f)[0]["secret"] is None


def test_mother_and_daughter_threads_are_separate(store):
    chat = store.create_chat()
    d = store.bind_daughter(chat, "gpt", "gpt-5")
    assert store.bind_daughter(chat, "gpt", "gpt-6") == d  # same binding, model updated
    store.add_message(chat, None, "user", "mother q")
    store.add_message(chat, d, "user", "daughter q")
    assert [m["content"] for m in store.thread(chat)] == ["mother q"]
    assert [m["content"] for m in store.thread(chat, d)] == ["daughter q"]
    assert store.daughters(chat)[0]["model"] == "gpt-6"


def test_context_globals_first_and_cascade_delete(store):
    a, b = store.create_chat(), store.create_chat()
    store.add_context(a, "chat note", "x")
    store.add_context(None, "global note", "y")
    store.add_context(b, "other chat", "z")
    assert [c["title"] for c in store.context_for(a)] == ["global note", "chat note"]
    d = store.bind_daughter(a, "gpt", "m")
    store.add_message(a, d, "user", "hi")
    store.delete_chat(a)
    assert store.daughters(a) == [] and store.thread(a, d) == []
    assert [c["title"] for c in store.context_for(b)] == ["global note", "other chat"]


def test_migrates_v0_database(tmp_path):
    import sqlite3
    from prism.db import SCHEMA
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)  # the v0.2.0 schema, user_version 0
    conn.execute("INSERT INTO credentials VALUES ('gpt', 'gpt-5', NULL, NULL, 1, 0)")
    conn.execute("INSERT INTO chats (title, created_at, updated_at) VALUES ('old', 0, 0)")
    conn.execute("INSERT INTO daughter_chats (chat_id, platform, model, created_at) VALUES (1, 'gpt', 'm', 0)")
    conn.commit(); conn.close()

    store = Store(path)
    assert store.list_credentials()[0]["transport"] == "api"
    d = store.daughters(1)[0]
    assert d["remote_url"] is None
    store.set_remote_url(d["id"], "https://example.test/c/1")
    assert store.daughters(1)[0]["remote_url"] == "https://example.test/c/1"
    assert Store(path).daughters(1)[0]["remote_url"] == "https://example.test/c/1"  # re-open is a no-op
