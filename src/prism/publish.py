"""prism-publish: make your publisher key, add files to your catalog, sign it.

    prism-publish keygen --out ~/prism-keys          # once: private key + public key line
    prism-publish new catalog.json --name "My tools"
    prism-publish add catalog.json --name mytools --version 1.2.0 --kind python \\
        --file any=dist/mytools-1.2.0-py3-none-any.whl
    prism-publish sign catalog.json --key ~/prism-keys/prism-publisher.key
    prism-publish verify catalog.json --pub "ed25519:..."

Upload catalog.json, catalog.json.sig and the files next to each other. Paste the public key
into PRISM (Settings → Catalog) - or ship it in the build as src/prism/publisher.pub. Keep the
private key off the server: whoever has it can publish to every PRISM that trusts you.
"""

import argparse
import base64
import getpass
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .catalog import KEY_PREFIX, CatalogError, check, fingerprint, verify


def public_line(private: Ed25519PrivateKey) -> str:
    raw = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return KEY_PREFIX + base64.b64encode(raw).decode()


def keygen(out: Path, passphrase: str | None) -> str:
    out.mkdir(parents=True, exist_ok=True)
    key_path = out / "prism-publisher.key"
    if key_path.exists():
        raise SystemExit(f"{key_path} exists - not overwriting your key")
    private = Ed25519PrivateKey.generate()
    enc = (serialization.BestAvailableEncryption(passphrase.encode()) if passphrase
           else serialization.NoEncryption())
    key_path.write_bytes(private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, enc))
    try:
        os.chmod(key_path, 0o600)
    except OSError:
        pass
    line = public_line(private)
    (out / "publisher.pub").write_text(line + "\n")
    return line


def load_private(path: Path, passphrase: str | None) -> Ed25519PrivateKey:
    data = path.read_bytes()
    try:
        return serialization.load_pem_private_key(data, password=passphrase.encode() if passphrase else None)
    except TypeError:  # it is encrypted and no passphrase was given
        return serialization.load_pem_private_key(data, password=getpass.getpass("Key passphrase: ").encode())


def new(path: Path, name: str) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists")
    path.write_text(json.dumps({"format": 1, "name": name, "version": 0, "expires": "", "entries": []}, indent=2) + "\n")


def file_info(path: Path) -> dict:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return {"url": path.name, "sha256": h.hexdigest(), "size": path.stat().st_size}


def add(path: Path, name: str, version: str, kind: str, files: list[str], commands: list[str],
        description: str = "", as_name: str | None = None, url_base: str = "") -> dict:
    doc = json.loads(path.read_text())
    entry = {"name": name, "version": version, "kind": kind, "files": {}}
    if description:
        entry["description"] = description
    if as_name:
        entry["as"] = as_name
    for spec in files:
        plat, _, file = spec.partition("=")
        if not file:
            plat, file = "any", plat
        info = file_info(Path(file))
        if url_base:
            info["url"] = url_base.rstrip("/") + "/" + info["url"]
        entry["files"][plat] = info
    if commands:
        entry["commands"] = dict(c.split("=", 1) for c in commands)
    doc["entries"] = [e for e in doc["entries"] if e["name"] != name] + [entry]
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return entry


def sign(path: Path, private: Ed25519PrivateKey, days: int = 90) -> dict:
    """Bump the version, set the expiry, then sign the file's exact bytes."""
    doc = json.loads(path.read_text())
    doc["version"] = int(doc.get("version") or 0) + 1
    doc["expires"] = (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw = (json.dumps(doc, indent=2) + "\n").encode()
    path.write_bytes(raw)
    sig = base64.b64encode(private.sign(raw)).decode()
    Path(str(path) + ".sig").write_text(sig + "\n")
    verify(raw, sig, [public_line(private)])  # it checks out the way PRISM will check it
    check(raw, None)
    return doc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="prism-publish", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keygen", help="make your publisher key (once)")
    k.add_argument("--out", type=Path, default=Path.home() / "prism-keys")
    k.add_argument("--passphrase", action="store_true", help="protect the private key with a passphrase")
    n = sub.add_parser("new", help="start a catalog")
    n.add_argument("catalog", type=Path)
    n.add_argument("--name", default="My PRISM catalog")
    a = sub.add_parser("add", help="add or replace an entry, with its files' checksums")
    a.add_argument("catalog", type=Path)
    a.add_argument("--name", required=True)
    a.add_argument("--version", required=True)
    a.add_argument("--kind", choices=["python", "model"], default="python")
    a.add_argument("--file", action="append", default=[], help="platform=path (windows-x64, macos-arm64, macos-x64, linux-x64, any)")
    a.add_argument("--description", default="")
    a.add_argument("--as", dest="as_name", help="models: the @name it joins the room as")
    a.add_argument("--url-base", default="", help="where the files will be, if not next to catalog.json")
    s = sub.add_parser("sign", help="bump the version, set the expiry and sign")
    s.add_argument("catalog", type=Path)
    s.add_argument("--key", type=Path, required=True)
    s.add_argument("--days", type=int, default=90)
    v = sub.add_parser("verify", help="check a signed catalog the way PRISM does")
    v.add_argument("catalog", type=Path)
    v.add_argument("--pub", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "keygen":
        pw = getpass.getpass("Passphrase for the private key: ") if args.passphrase else None
        line = keygen(args.out, pw)
        print(f"Private key: {args.out / 'prism-publisher.key'}  (keep it off the server, back it up)")
        print(f"Public key (paste into PRISM → Settings → Catalog):\n{line}\nFingerprint: {fingerprint(line)}")
    elif args.cmd == "new":
        new(args.catalog, args.name)
        print(f"Started {args.catalog}")
    elif args.cmd == "add":
        e = add(args.catalog, args.name, args.version, args.kind, args.file, [],
                args.description, args.as_name, args.url_base)
        print(f"Added {e['name']} {e['version']} for {', '.join(e['files'])}. Sign it next.")
    elif args.cmd == "sign":
        doc = sign(args.catalog, load_private(args.key, None))
        print(f"Signed version {doc['version']}, valid until {doc['expires']}. Upload {args.catalog}, "
              f"{args.catalog}.sig and the files.")
    elif args.cmd == "verify":
        raw = args.catalog.read_bytes()
        try:
            verify(raw, Path(str(args.catalog) + ".sig").read_text(), [args.pub])
            doc = check(raw, None)
        except CatalogError as e:
            sys.exit(f"Not valid: {e}")
        print(f"Valid: version {doc['version']}, {len(doc['entries'])} entries, until {doc['expires']}")


if __name__ == "__main__":
    main()
