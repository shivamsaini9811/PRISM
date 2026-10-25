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
    # A turn has no fixed time limit: it runs while the model shows it is working (the site's
    # Stop button, text or progress still changing, a local model still streaming). It gives up
    # after this long with no sign of activity, or after reply_max_s in all.
    request_timeout_s: float = 1200.0
    reply_max_s: float = 4 * 3600.0
    run_timeout_s: float = 300.0  # a program a bot runs (RUN) is stopped after this long
