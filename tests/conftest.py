"""Shared test helpers: one Python sandbox for the whole run, and tiny wheels for pip."""

import base64
import hashlib
import zipfile
from pathlib import Path

import pytest

from prism.runner import Sandbox


@pytest.fixture(scope="session")
def shared_sandbox(tmp_path_factory) -> Sandbox:
    """A sandbox on the Python running the tests (making one takes seconds: once per run)."""
    sb = Sandbox(tmp_path_factory.mktemp("python") / "env")
    sb.ensure()
    return sb


def make_wheel(folder: Path, name: str, version: str = "1.0", body: str = "VALUE = 42\n") -> Path:
    """A minimal pure-Python wheel pip installs offline (--no-index --find-links)."""
    folder.mkdir(parents=True, exist_ok=True)
    dist = f"{name}-{version}.dist-info"
    files = {
        f"{name}/__init__.py": body,
        f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        f"{dist}/WHEEL": "Wheel-Version: 1.0\nGenerator: prism-tests\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = []
    for path, text in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b"=").decode()
        record.append(f"{path},sha256={digest},{len(text.encode())}")
    record.append(f"{dist}/RECORD,,")
    files[f"{dist}/RECORD"] = "\n".join(record) + "\n"
    wheel = folder / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        for path, text in files.items():
            z.writestr(path, text)
    return wheel
