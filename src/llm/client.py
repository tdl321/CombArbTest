"""LLM client for OpenRouter API (Kimi 2.5).

Implements LLM-01: Connect to Kimi 2.5 via OpenRouter.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import httpx

from ..config import get_config, SecretString

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "moonshotai/kimi-k2"


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    model: str
    usage: dict[str, int]
    raw: dict[str, Any]


def extract_json_from_response(raw_response: str) -> dict[str, Any]:
    """Extract JSON from potentially malformed or markdown-wrapped response."""
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        pass

    json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(json_pattern, raw_response)
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass

    json_obj_pattern = r"\{[\s\S]*\}"
    matches = re.findall(json_obj_pattern, raw_response)
    if matches:
        for match in reversed(matches):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    raise json.JSONDecodeError("Could not extract JSON from response", raw_response, 0)


class LLMClient:
    """Client for OpenRouter API."""

    def __init__(
        self,
        api_key: SecretString | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
    ):
        logger.info("[LLM] Initializing client with model=%s, timeout=%.1fs", model, timeout)
        config = get_config()
        self._api_key = api_key or config.openrouter_api_key

        if not self._api_key:
            logger.error("[LLM] OpenRouter API key not found")
            raise ValueError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY in .env file."
            )

        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=OPENROUTER_BASE_URL,
            timeout=timeout,
            headers={
                "Authorization": "Bearer %s" % self._api_key.get_secret_value(),
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/combarbbot",
                "X-Title": "CombarBot",
            },
        )
        logger.debug("[LLM] Client initialized successfully")

    def chat(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send a chat completion request."""
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        logger.info("[LLM] Chat request: temp=%.1f, max_tokens=%d, json_mode=%s",
                    temperature, max_tokens, json_mode)
        logger.debug("[LLM] Prompt preview: %s", prompt_preview)
        
        start_time = time.time()
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error("[LLM] HTTP error: %s", e)
            raise
        except httpx.RequestError as e:
            logger.error("[LLM] Request error: %s", e)
            raise

        data = response.json()
        elapsed = time.time() - start_time
        
        usage = data.get("usage", {})
        logger.info("[LLM] Response received in %.3fs: prompt_tokens=%d, completion_tokens=%d",
                    elapsed, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", self.model),
            usage=usage,
            raw=data,
        )

    def chat_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Send a chat request and parse JSON response."""
        logger.debug("[LLM] JSON chat request")
        response = self.chat(
            prompt=prompt,
            system=system,
            temperature=temperature,
            json_mode=True,
        )
        
        try:
            result = extract_json_from_response(response.content)
            logger.debug("[LLM] JSON parsed successfully with %d keys", len(result))
            return result
        except json.JSONDecodeError as e:
            logger.error("[LLM] Failed to parse JSON response: %s", e)
            raise

    def close(self) -> None:
        """Close the HTTP client."""
        logger.debug("[LLM] Closing client")
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()


@lru_cache(maxsize=1)
def get_client() -> LLMClient:
    """Get singleton LLM client."""
    logger.debug("[LLM] Getting singleton client")
    return LLMClient()
