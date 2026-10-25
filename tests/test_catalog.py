"""The signed catalog: signatures, versions, expiry, checksums, safe unpacking, installing."""

import base64
import io
import json
import sys
import threading
import time
import zipfile
from datetime import datetime, timedelta, timezone

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from prism import app as app_mod, catalog, publish
from prism.catalog import CatalogError, check, verify
from test_app import client  # noqa: F401 - the app fixture


@pytest.fixture(scope="module")
def keys(tmp_path_factory):
    out = tmp_path_factory.mktemp("keys")
    line = publish.keygen(out, None)
    return line, publish.load_private(out / "prism-publisher.key", None)


def test_keys_signatures_versions_and_expiry(keys, tmp_path):
    line, private = keys
    cat = tmp_path / "catalog.json"
    publish.new(cat, "Test")
    doc = publish.sign(cat, private)
    raw = cat.read_bytes()
    sig = (tmp_path / "catalog.json.sig").read_text()
    verify(raw, sig, [line])
    assert doc["version"] == 1 and check(raw, None)["name"] == "Test"
    assert len(catalog.fingerprint(line).split()) == 4

    with pytest.raises(CatalogError, match="no publisher key"):
        verify(raw, sig, [])
    with pytest.raises(CatalogError, match="doesn't match"):
        verify(raw.replace(b'"Test"', b'"Evil"'), sig, [line])           # one changed byte
    other = publish.keygen(tmp_path / "other", None)
    with pytest.raises(CatalogError, match="doesn't match"):
        verify(raw, sig, [other])                                        # someone else's key
    with pytest.raises(CatalogError, match="older than"):
        check(raw, last_version=2)                                       # an old catalog replayed
    with pytest.raises(CatalogError, match="expired"):
        check(raw, None, now=datetime.now(timezone.utc) + timedelta(days=91))
    with pytest.raises(CatalogError, match="looks like"):
        catalog.parse_key("ssh-ed25519 AAAA")


def test_archives_cannot_escape(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../../evil.txt", "x")
    with pytest.raises(CatalogError, match="outside its folder"):
        catalog._safe_extract(bad, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


# --- a server with a signed catalog, through the app ---------------------------------------------

@pytest.fixture(scope="module")
def server(tmp_path_factory, keys):
    line, private = keys
    root = tmp_path_factory.mktemp("srv")
    tool = io.BytesIO()
    with zipfile.ZipFile(tool, "w") as z:
        info = zipfile.ZipInfo("bin/hello.sh")
        info.external_attr = 0o755 << 16
        z.writestr(info, "#!/bin/sh\necho hello-from-catalog\n")
        z.writestr("bin/hello.bat", "@echo hello-from-catalog\n")
    (root / "hello.zip").write_bytes(tool.getvalue())
    (root / "broken.zip").write_bytes(tool.getvalue())
    cat = root / "catalog.json"
    publish.new(cat, "Test tools")
    cmd = "bin/hello.bat" if sys.platform == "win32" else "bin/hello.sh"
    publish.add(cat, "hello", "1.0", "toolchain", [f"any={root / 'hello.zip'}"], [f"hello={cmd}"], "Says hello")
    publish.add(cat, "broken", "1.0", "toolchain", [f"any={root / 'broken.zip'}"], [f"hello={cmd}"])
    doc = json.loads(cat.read_text())
    doc["entries"][1]["files"]["any"]["sha256"] = "0" * 64  # its download won't match
    cat.write_text(json.dumps(doc, indent=2))
    publish.sign(cat, private)

    fake = FastAPI()
    fake.mount("/", StaticFiles(directory=root))
    srv = uvicorn.Server(uvicorn.Config(fake, host="127.0.0.1", port=11497, log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield "http://127.0.0.1:11497/catalog.json", root, line, private
    srv.should_exit = True
    t.join(5)


def wait_install(client, name):
    for _ in range(200):
        v = client.get("/api/catalog").json()
        if v["installs"].get(name, {}).get("done"):
            return v
        time.sleep(0.05)
    raise AssertionError("install did not finish")


def test_catalog_install_through_the_app(client, server):
    url, root, line, private = server
    client.post("/api/unlock", json={"password": "longenough"})
    assert client.put("/api/catalog", json={"url": "http://example.com/c.json"}).status_code == 400  # not https
    client.put("/api/catalog", json={"url": url})
    r = client.post("/api/catalog/refresh")
    assert r.status_code == 400 and "no publisher key" in r.json()["detail"]
    assert client.post("/api/catalog/install", json={"name": "hello"}).status_code == 400  # nothing unsigned
    assert client.put("/api/catalog", json={"add_key": "nonsense"}).status_code == 400
    v = client.put("/api/catalog", json={"add_key": line}).json()
    assert v["keys"][0]["fingerprint"] == catalog.fingerprint(line)
    v = client.post("/api/catalog/refresh").json()
    assert v["name"] == "Test tools" and [e["name"] for e in v["entries"]] == ["hello", "broken"]

    client.post("/api/catalog/install", json={"name": "hello"})
    v = wait_install(client, "hello")
    assert v["installs"]["hello"]["error"] is None, v["installs"]["hello"]
    assert v["installed"]["hello"]["version"] == "1.0"
    tools = app_mod.installed_toolchains()
    assert set(tools) == {"hello"}
    from prism.runner import Request, execute
    done = execute(Request("bot", "hello", command="hello"), root / "work", tools, 30)
    assert done.output.strip() == "hello-from-catalog"
    assert client.get("/api/runner").json()["toolchains"] == tools

    client.post("/api/catalog/install", json={"name": "broken"})
    v = wait_install(client, "broken")
    assert "checksum" in v["installs"]["broken"]["error"] and "broken" not in v["installed"]

    # The same server later serves an older catalog (say, restored from a backup): refused.
    cat = root / "catalog.json"
    doc = json.loads(cat.read_text())
    doc["version"] = 0
    raw = (json.dumps(doc, indent=2) + "\n").encode()
    cat.write_bytes(raw)
    (root / "catalog.json.sig").write_text(base64.b64encode(private.sign(raw)).decode())
    r = client.post("/api/catalog/refresh")
    assert r.status_code == 400 and "older than" in r.json()["detail"]

    v = client.delete("/api/catalog/hello").json()
    assert "hello" not in v["installed"] and app_mod.installed_toolchains() == {}


def test_prism_publish_command(tmp_path, capsys):
    tool = tmp_path / "tool.zip"
    tool.write_bytes(b"PK\x05\x06" + b"\0" * 18)  # an empty zip
    publish.main(["keygen", "--out", str(tmp_path / "k")])
    out = capsys.readouterr().out
    line = next(x for x in out.splitlines() if x.startswith("ed25519:"))
    with pytest.raises(SystemExit, match="not overwriting"):
        publish.main(["keygen", "--out", str(tmp_path / "k")])
    cat = tmp_path / "catalog.json"
    publish.main(["new", str(cat), "--name", "Mine"])
    publish.main(["add", str(cat), "--name", "tool", "--version", "2", "--file", f"windows-x64={tool}",
                  "--command", "tool=bin/tool.exe", "--url-base", "https://lib.example.com/files"])
    publish.main(["sign", str(cat), "--key", str(tmp_path / "k" / "prism-publisher.key")])
    publish.main(["sign", str(cat), "--key", str(tmp_path / "k" / "prism-publisher.key")])
    publish.main(["verify", str(cat), "--pub", line])
    out = capsys.readouterr().out
    assert "Signed version 2" in out and "Valid: version 2, 1 entries" in out
    entry = json.loads(cat.read_text())["entries"][0]
    assert entry["files"]["windows-x64"]["url"] == "https://lib.example.com/files/tool.zip"
    assert len(entry["files"]["windows-x64"]["sha256"]) == 64 and entry["commands"] == {"tool": "bin/tool.exe"}
