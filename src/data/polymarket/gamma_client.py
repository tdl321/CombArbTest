"""HTTP client for Polymarket Gamma API (public, no auth).

Gamma API provides event and market metadata — discovery, search, and
structural information like negRisk.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from .config import PolymarketConfig
from .types import PolymarketEvent, PolymarketMarket

logger = logging.getLogger(__name__)


class GammaClient:
    """Client for the Polymarket Gamma API.

    Public API — no authentication required.
    Provides market/event metadata, search, and discovery.
    """

    def __init__(self, config: PolymarketConfig | None = None):
        self._config = config or PolymarketConfig()
        self._client: httpx.Client | None = None
        self._last_request_time: float = 0.0

    @property
    def client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=self._config.gamma_base_url,
                timeout=self._config.request_timeout,
                headers={"Accept": "application/json"},
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Rate Limiting ──────────────────────────────────────

    def _wait_for_rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        if self._config.rate_limit_rps <= 0:
            return
        min_interval = 1.0 / self._config.rate_limit_rps
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.monotonic()

    # ── HTTP with Retry ────────────────────────────────────

    def _request(self, method: str, path: str, params: dict | None = None) -> Any:
        """Make an HTTP request with rate limiting and retry on 429."""
        for attempt in range(self._config.max_retries + 1):
            self._wait_for_rate_limit()
            try:
                response = self.client.request(method, path, params=params)

                if response.status_code == 429:
                    backoff = self._config.retry_backoff_factor * (2 ** attempt)
                    logger.warning(
                        "Rate limited (429), retrying in %.1fs (attempt %d/%d)",
                        backoff, attempt + 1, self._config.max_retries,
                    )
                    time.sleep(backoff)
                    continue

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                if attempt < self._config.max_retries and e.response.status_code >= 500:
                    backoff = self._config.retry_backoff_factor * (2 ** attempt)
                    logger.warning(
                        "Server error %d, retrying in %.1fs",
                        e.response.status_code, backoff,
                    )
                    time.sleep(backoff)
                    continue
                raise
            except httpx.RequestError as e:
                if attempt < self._config.max_retries:
                    backoff = self._config.retry_backoff_factor * (2 ** attempt)
                    logger.warning("Request error: %s, retrying in %.1fs", e, backoff)
                    time.sleep(backoff)
                    continue
                raise

        return None

    # ── Event Methods ──────────────────────────────────────

    def get_event_by_slug(self, slug: str) -> PolymarketEvent | None:
        """Get an event by its URL slug.

        This is the primary discovery method — slugs are human-readable
        and stable (e.g., "f1-drivers-championship-2026").
        """
        data = self._request("GET", "/events", params={"slug": slug})
        if not data:
            return None
        # Gamma returns a list for slug queries
        events = data if isinstance(data, list) else [data]
        if not events:
            return None
        return PolymarketEvent.from_api_response(events[0])

    def get_event_by_id(self, event_id: str) -> PolymarketEvent | None:
        """Get an event by its ID."""
        data = self._request("GET", f"/events/{event_id}")
        if not data:
            return None
        return PolymarketEvent.from_api_response(data)

    def search_events(
        self,
        query: str | None = None,
        tag: str | None = None,
        active: bool | None = None,
        closed: bool | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[PolymarketEvent]:
        """Search for events with filters."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if query:
            params["title"] = query
        if tag:
            params["tag"] = tag
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()

        data = self._request("GET", "/events", params=params)
        if not data:
            return []
        events = data if isinstance(data, list) else [data]
        return [PolymarketEvent.from_api_response(e) for e in events]

    # ── Market Methods ─────────────────────────────────────

    def get_market_by_id(self, condition_id: str) -> PolymarketMarket | None:
        """Get a single market by its condition ID."""
        data = self._request("GET", f"/markets/{condition_id}")
        if not data:
            return None
        return PolymarketMarket.from_api_response(data)

    def get_markets(
        self,
        event_slug: str | None = None,
        active: bool | None = None,
        closed: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PolymarketMarket]:
        """Query markets with filters."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if event_slug:
            params["event_slug"] = event_slug
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()

        data = self._request("GET", "/markets", params=params)
        if not data:
            return []
        markets = data if isinstance(data, list) else [data]
        return [PolymarketMarket.from_api_response(m) for m in markets]
