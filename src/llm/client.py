"""LLM client for OpenRouter API (Kimi 2.5).

Implements LLM-01: Connect to Kimi 2.5 via OpenRouter.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import httpx

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from config import get_config, SecretString


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "moonshotai/kimi-k2"  # Kimi 2.5


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    model: str
    usage: dict[str, int]
    raw: dict[str, Any]


class LLMClient:
    """Client for OpenRouter API.
    
    Usage:
        client = LLMClient()
        response = client.chat("What is 2+2?")
        print(response.content)
    """
    
    def __init__(
        self,
        api_key: SecretString | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
    ):
        config = get_config()
        self._api_key = api_key or config.openrouter_api_key
        
        if not self._api_key:
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
                "Authorization": f"Bearer {self._api_key.get_secret_value()}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/combarbbot",  # Required by OpenRouter
                "X-Title": "CombarBot",
            },
        )
    
    def chat(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send a chat completion request.
        
        Args:
            prompt: User message
            system: Optional system prompt
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum response tokens
            json_mode: If True, request JSON output
        
        Returns:
            LLMResponse with content and metadata
        """
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
        
        response = self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", self.model),
            usage=data.get("usage", {}),
            raw=data,
        )
    
    def chat_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,  # Lower for structured output
    ) -> dict[str, Any]:
        """Send a chat request and parse JSON response.
        
        Args:
            prompt: User message (should request JSON output)
            system: Optional system prompt
            temperature: Sampling temperature
        
        Returns:
            Parsed JSON dict
        """
        response = self.chat(
            prompt=prompt,
            system=system,
            temperature=temperature,
            json_mode=True,
        )
        return json.loads(response.content)
    
    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
    
    def __enter__(self) -> "LLMClient":
        return self
    
    def __exit__(self, *args) -> None:
        self.close()


@lru_cache(maxsize=1)
def get_client() -> LLMClient:
    """Get singleton LLM client."""
    return LLMClient()
