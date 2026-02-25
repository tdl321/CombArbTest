"""Secure configuration loader.

Loads secrets from .env file with protections against accidental exposure.
"""

import logging
import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class SecretString:
    """A string that never exposes its value in repr/str.

    Prevents accidental logging of secrets.
    """
    __slots__ = ("_value",)

    def __init__(self, value: str):
        self._value = value

    def __repr__(self) -> str:
        return "SecretString(***)"

    def __str__(self) -> str:
        return "***"

    def get_secret_value(self) -> str:
        """Explicitly retrieve the secret value."""
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)


class Config:
    """Application configuration with secure secret handling."""

    def __init__(self):
        logger.debug("[CONFIG] Initializing configuration")

        project_root = Path(__file__).parent.parent
        env_path = project_root / ".env"

        if env_path.exists():
            mode = env_path.stat().st_mode & 0o777
            if mode & 0o077:
                logger.error("[CONFIG] .env file has insecure permissions: %s", oct(mode))
                raise PermissionError(
                    ".env file has insecure permissions (%s). "
                    "Run: chmod 600 %s" % (oct(mode), env_path)
                )
            load_dotenv(env_path)
            logger.debug("[CONFIG] Loaded .env from %s", env_path)
        else:
            logger.warning("[CONFIG] No .env file found at %s", env_path)

        self._openrouter_api_key = self._load_secret("OPENROUTER_API_KEY")

        if self._openrouter_api_key:
            logger.info("[CONFIG] OpenRouter API key loaded")
        else:
            logger.warning("[CONFIG] OpenRouter API key not configured")

    def _load_secret(self, key: str) -> SecretString | None:
        """Load a secret from environment, wrapped in SecretString."""
        value = os.getenv(key)
        return SecretString(value) if value else None

    @property
    def openrouter_api_key(self) -> SecretString | None:
        """OpenRouter API key (wrapped in SecretString)."""
        return self._openrouter_api_key

    def has_openrouter(self) -> bool:
        """Check if OpenRouter is configured."""
        return bool(self._openrouter_api_key)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Get singleton config instance."""
    logger.debug("[CONFIG] Getting config singleton")
    return Config()
