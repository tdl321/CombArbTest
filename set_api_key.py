#!/usr/bin/env python3
"""Securely set API keys in .env file.

Usage: python set_api_key.py
"""

import getpass
import os
from pathlib import Path


def set_key(key_name: str, prompt: str) -> None:
    """Securely prompt for and save an API key."""
    env_path = Path(__file__).parent / ".env"
    
    # Read existing content
    existing = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    existing[k] = v
    
    # Prompt for new key (hidden input)
    print(f"\n{prompt}")
    value = getpass.getpass("Paste key (hidden): ")
    
    if not value.strip():
        print("No key provided, skipping.")
        return
    
    existing[key_name] = value.strip()
    
    # Write back with secure permissions
    with open(env_path, "w") as f:
        for k, v in existing.items():
            f.write(f"{k}={v}\n")
    
    os.chmod(env_path, 0o600)
    print(f"✓ {key_name} saved to .env (permissions: 600)")


if __name__ == "__main__":
    print("=== API Key Setup ===")
    print("Keys are stored in .env with restricted permissions (owner-only read/write)")
    print("The .env file is in .gitignore and will never be committed.\n")
    
    set_key("OPENROUTER_API_KEY", "Enter your OpenRouter API key:")
    
    print("\n✓ Setup complete. Keys are securely stored.")
