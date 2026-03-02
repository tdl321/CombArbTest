"""Polymarket API configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolymarketConfig:
    """Configuration for Polymarket API clients."""

    gamma_base_url: str = "https://gamma-api.polymarket.com"
    clob_base_url: str = "https://clob.polymarket.com"
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    request_timeout: float = 30.0
    max_retries: int = 3
    retry_backoff_factor: float = 0.5
    rate_limit_rps: float = 10.0
