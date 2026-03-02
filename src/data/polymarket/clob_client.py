"""HTTP client for Polymarket CLOB API (public read, no auth).

CLOB API provides live prices, orderbooks, and price history.
Best source for current midpoint prices.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

import httpx

from .config import PolymarketConfig
from .types import PolymarketOrderBook, PolymarketPriceHistory

logger = logging.getLogger(__name__)


class ClobClient:
    """Client for the Polymarket CLOB API.

    Public read-only — no authentication required for price data.
    """

    def __init__(self, config: PolymarketConfig | None = None):
        self._config = config or PolymarketConfig()
        self._client: httpx.Client | None = None
        self._last_request_time: float = 0.0

    @property
    def client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=self._config.clob_base_url,
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
        if self._config.rate_limit_rps <= 0:
            return
        min_interval = 1.0 / self._config.rate_limit_rps
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.monotonic()

    # ── HTTP with Retry ────────────────────────────────────

    def _request(self, method: str, path: str, params: dict | None = None) -> Any:
        """Make an HTTP request with rate limiting and retry."""
        for attempt in range(self._config.max_retries + 1):
            self._wait_for_rate_limit()
            try:
                response = self.client.request(method, path, params=params)

                if response.status_code == 429:
                    backoff = self._config.retry_backoff_factor * (2 ** attempt)
                    logger.warning(
                        "CLOB rate limited (429), retrying in %.1fs (attempt %d/%d)",
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
                        "CLOB server error %d, retrying in %.1fs",
                        e.response.status_code, backoff,
                    )
                    time.sleep(backoff)
                    continue
                raise
            except httpx.RequestError as e:
                if attempt < self._config.max_retries:
                    backoff = self._config.retry_backoff_factor * (2 ** attempt)
                    logger.warning("CLOB request error: %s, retrying in %.1fs", e, backoff)
                    time.sleep(backoff)
                    continue
                raise

        return None

    # ── Price Methods ──────────────────────────────────────

    def get_midpoint(self, token_id: str) -> float | None:
        """Get the midpoint price for a token.

        This is the best single-price source — more accurate than
        Gamma snapshot prices.
        """
        data = self._request("GET", "/midpoint", params={"token_id": token_id})
        if data and "mid" in data:
            try:
                return float(data["mid"])
            except (ValueError, TypeError):
                return None
        return None

    def get_price(
        self,
        token_id: str,
        side: Literal["buy", "sell"] = "buy",
    ) -> float | None:
        """Get the best price on a given side."""
        data = self._request("GET", "/price", params={
            "token_id": token_id,
            "side": side.upper(),
        })
        if data and "price" in data:
            try:
                return float(data["price"])
            except (ValueError, TypeError):
                return None
        return None

    def get_orderbook(self, token_id: str) -> PolymarketOrderBook | None:
        """Get the full order book for a token."""
        data = self._request("GET", "/book", params={"token_id": token_id})
        if not data:
            return None
        return PolymarketOrderBook.from_api_response(token_id, data)

    def get_price_history(
        self,
        token_id: str,
        interval: Literal["1m", "5m", "1h", "1d", "1w", "max"] = "1d",
        fidelity: int = 60,
    ) -> PolymarketPriceHistory | None:
        """Get price history for a token.

        Args:
            token_id: CLOB token ID (YES or NO side)
            interval: Time range — "1m", "5m", "1h", "1d", "1w", "max"
            fidelity: Resolution in minutes (e.g., 60 = hourly points)
        """
        data = self._request("GET", "/prices-history", params={
            "tokenID": token_id,
            "interval": interval,
            "fidelity": fidelity,
        })
        if not data:
            return None
        return PolymarketPriceHistory.from_api_response(token_id, data)

    def get_midpoints_batch(self, token_ids: list[str]) -> dict[str, float]:
        """Get midpoint prices for multiple tokens.

        Falls back to individual requests since CLOB batch endpoint
        may not be available.
        """
        results: dict[str, float] = {}
        for token_id in token_ids:
            mid = self.get_midpoint(token_id)
            if mid is not None:
                results[token_id] = mid
        return results
