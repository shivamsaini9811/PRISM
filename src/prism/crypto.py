"""Master-password vault: scrypt-derived key + Fernet field encryption for stored secrets."""

import base64
import os

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

VERIFIER = b"prism-vault-v1"


class WrongPassword(Exception):
    pass


def new_salt() -> bytes:
    return os.urandom(16)


def derive_fernet(password: str, salt: bytes) -> Fernet:
    key = Scrypt(salt=salt, length=32, n=2**15, r=8, p=1).derive(password.encode())
    return Fernet(base64.urlsafe_b64encode(key))


def make_verifier(f: Fernet) -> bytes:
    return f.encrypt(VERIFIER)


def check_verifier(f: Fernet, token: bytes) -> None:
    try:
        ok = f.decrypt(token) == VERIFIER
    except InvalidToken:
        ok = False
    if not ok:
        raise WrongPassword("wrong master password")
