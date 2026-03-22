"""Credential management for the autoresearch agent on Linux/IBM Cloud.

Resolves API keys from:
1. ANTHROPIC_API_KEY environment variable (explicit, always wins)
2. File-based credential store (~/.config/autoresearch/api_key)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


CONFIG_DIR = Path.home() / ".config" / "autoresearch"
API_KEY_FILE = CONFIG_DIR / "api_key"


@dataclass
class CredentialSource:
    """Describes where a credential came from."""
    api_key: str
    source: str  # "env", "file"


def resolve_api_key() -> CredentialSource:
    """Resolve an Anthropic API key from available sources.

    Priority:
    1. ANTHROPIC_API_KEY environment variable
    2. File-based credential store

    Raises RuntimeError if no credentials found.
    """
    # 1. Environment variable (highest priority)
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return CredentialSource(api_key=env_key, source="env")

    # 2. File-based credential store
    if API_KEY_FILE.exists():
        key = API_KEY_FILE.read_text().strip()
        if key:
            return CredentialSource(api_key=key, source="file")

    raise RuntimeError(
        "No Anthropic API key found. Set up credentials using one of:\n"
        "\n"
        "  Option 1 — Environment variable:\n"
        "    export ANTHROPIC_API_KEY=sk-ant-...\n"
        "\n"
        "  Option 2 — Docker compose env:\n"
        "    ANTHROPIC_API_KEY=sk-ant-... docker compose run agent\n"
        "\n"
        "  Option 3 — File-based store:\n"
        "    mkdir -p ~/.config/autoresearch\n"
        "    echo 'sk-ant-...' > ~/.config/autoresearch/api_key\n"
        "    chmod 600 ~/.config/autoresearch/api_key\n"
    )


def setup_api_key() -> None:
    """Interactive setup: prompt for API key and store in file."""
    print("Autoresearch Agent -- API Key Setup")
    print("=" * 40)
    print()
    print("This will store your Anthropic API key in ~/.config/autoresearch/api_key")
    print("Get a key at: https://console.anthropic.com/settings/keys")
    print()

    # Check for existing key
    if API_KEY_FILE.exists():
        existing = API_KEY_FILE.read_text().strip()
        if existing:
            masked = existing[:12] + "..." + existing[-4:]
            print(f"Existing key found: {masked}")
            response = input("Replace it? [y/N]: ").strip().lower()
            if response not in ("y", "yes"):
                print("Keeping existing key.")
                return

    api_key = input("Paste your API key (sk-ant-...): ").strip()

    if not api_key:
        print("No key entered. Aborting.")
        return

    if not api_key.startswith("sk-ant-"):
        print("Warning: key doesn't start with 'sk-ant-' -- are you sure this is correct?")
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Aborting.")
            return

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    API_KEY_FILE.write_text(api_key + "\n")
    os.chmod(API_KEY_FILE, 0o600)

    print()
    print("API key stored.")
    print(f"  File: {API_KEY_FILE}")
    print()
    print("You can now run the agent without ANTHROPIC_API_KEY env var.")


def clear_api_key() -> None:
    """Remove the stored API key file."""
    if API_KEY_FILE.exists():
        API_KEY_FILE.unlink()
        print("API key removed.")
    else:
        print("No stored API key found (or already removed).")
