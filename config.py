"""Central configuration for ContextBridge.

Every Azure credential is read from environment variables (or a local .env file)
so nothing secret ever lands in the repository.  Import this module once, early,
and the rest of the app can just read the constants below.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    """Load .env into os.environ.

    Uses python-dotenv when available and falls back to a tiny parser so the app
    still boots on a machine where the package was not installed.
    """
    env_file = ROOT / ".env"
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_file, override=False)
        return
    except Exception:
        pass

    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def get(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


# --- Azure AI Foundry (text model used by every feature) ---------------------
FOUNDRY_ENDPOINT = get("AZURE_FOUNDRY_ENDPOINT") or get("AZURE_OPENAI_ENDPOINT")
FOUNDRY_API_KEY = get("AZURE_FOUNDRY_API_KEY") or get("AZURE_OPENAI_API_KEY")
FOUNDRY_DEPLOYMENT = (
    get("AZURE_FOUNDRY_DEPLOYMENT") or get("AZURE_OPENAI_DEPLOYMENT") or "gpt-4o-mini"
)
FOUNDRY_API_VERSION = get("AZURE_OPENAI_API_VERSION", "2024-10-21")

# --- Azure AI Speech (text-to-speech and speech-to-text) ---------------------
SPEECH_KEY = get("AZURE_SPEECH_KEY")
SPEECH_REGION = get("AZURE_SPEECH_REGION", "westeurope")

APP_NAME = "ContextBridge"
APP_TAGLINE = "One bridge between how you communicate and how the world listens."
