"""Settings from environment variables (no .env file - the app ships as a standalone executable)."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings


def data_dir() -> Path:
    """Everything PRISM keeps between runs lives here (override with PRISM_HOME)."""
    return Path(os.environ.get("PRISM_HOME") or Path.home() / ".prism")


class Settings(BaseSettings):
    prism_db_path: Path = data_dir() / "prism.db"
    prism_profiles_dir: Path = data_dir() / "profiles"   # one shared browser profile, all platforms
    prism_browser_headless: bool = True   # False: a real window parked off-screen
    prism_browser_channel: str | None = None  # msedge | chrome | bundled; default: first that works
    request_timeout_s: float = 900.0  # one bot turn; deep answers on the web can take minutes
