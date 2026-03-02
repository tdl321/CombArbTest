"""Raw Polymarket API response types.

Mirror JSON structure exactly, handle JSON-encoded string fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PolymarketMarket:
    """A single Polymarket market (condition).

    Handles parsing of JSON-encoded strings from the API response
    for outcomes, outcomePrices, and clobTokenIds.
    """

    condition_id: str
    question: str
    slug: str
    outcomes: list[str]
    outcome_prices: list[float]
    clob_token_ids: list[str]
    volume: float = 0.0
    liquidity: float = 0.0
    active: bool = True
    closed: bool = False
    accepting_orders: bool = True
    description: str = ""
    end_date: str | None = None
    created_at: str | None = None
    category: str | None = None
    group_item_title: str | None = None
    # Event-level fields that may be on market objects
    event_slug: str | None = None
    neg_risk: bool = False
    neg_risk_market_id: str | None = None
    neg_risk_request_id: str | None = None

    @classmethod
    def from_api_response(cls, data: dict) -> PolymarketMarket:
        """Parse a market from Gamma API JSON response.

        Handles JSON-encoded strings for outcomes, outcomePrices, clobTokenIds.
        """
        outcomes = _parse_json_or_list(data.get("outcomes", "[]"))
        outcome_prices = [
            float(p) for p in _parse_json_or_list(data.get("outcomePrices", "[]"))
        ]
        clob_token_ids = _parse_json_or_list(data.get("clobTokenIds", "[]"))

        return cls(
            condition_id=data.get("conditionId") or data.get("condition_id", ""),
            question=data.get("question", ""),
            slug=data.get("slug", ""),
            outcomes=outcomes,
            outcome_prices=outcome_prices,
            clob_token_ids=clob_token_ids,
            volume=_safe_float(data.get("volume", 0)),
            liquidity=_safe_float(data.get("liquidity", 0)),
            active=bool(data.get("active", True)),
            closed=bool(data.get("closed", False)),
            accepting_orders=bool(data.get("acceptingOrders", True)),
            description=data.get("description", ""),
            end_date=data.get("endDate"),
            created_at=data.get("createdAt"),
            category=data.get("category"),
            group_item_title=data.get("groupItemTitle"),
            event_slug=data.get("eventSlug"),
            neg_risk=bool(data.get("negRisk", False)),
            neg_risk_market_id=data.get("negRiskMarketID"),
            neg_risk_request_id=data.get("negRiskRequestID"),
        )

    @property
    def yes_token_id(self) -> str | None:
        """CLOB token ID for the YES outcome."""
        return self.clob_token_ids[0] if self.clob_token_ids else None

    @property
    def no_token_id(self) -> str | None:
        """CLOB token ID for the NO outcome."""
        return self.clob_token_ids[1] if len(self.clob_token_ids) > 1 else None

    @property
    def yes_price(self) -> float:
        """Current YES price."""
        return self.outcome_prices[0] if self.outcome_prices else 0.5

    @property
    def no_price(self) -> float:
        """Current NO price."""
        return self.outcome_prices[1] if len(self.outcome_prices) > 1 else 1.0 - self.yes_price

    @property
    def is_active(self) -> bool:
        """Whether the market is active and accepting orders."""
        return self.active and not self.closed and self.accepting_orders


@dataclass
class PolymarketEvent:
    """A Polymarket event containing one or more markets.

    Events with neg_risk=True are mutually exclusive partitions.
    """

    event_id: str
    title: str
    slug: str
    description: str = ""
    markets: list[PolymarketMarket] = field(default_factory=list)
    neg_risk: bool = False
    category: str | None = None
    end_date: str | None = None
    created_at: str | None = None
    volume: float = 0.0
    liquidity: float = 0.0
    closed: bool = False
    active: bool = True

    @classmethod
    def from_api_response(cls, data: dict) -> PolymarketEvent:
        """Parse an event from Gamma API JSON response."""
        raw_markets = data.get("markets", [])
        markets = [PolymarketMarket.from_api_response(m) for m in raw_markets]

        return cls(
            event_id=data.get("id", ""),
            title=data.get("title", ""),
            slug=data.get("slug", ""),
            description=data.get("description", ""),
            markets=markets,
            neg_risk=bool(data.get("negRisk", False)),
            category=data.get("category"),
            end_date=data.get("endDate"),
            created_at=data.get("createdAt"),
            volume=_safe_float(data.get("volume", 0)),
            liquidity=_safe_float(data.get("liquidity", 0)),
            closed=bool(data.get("closed", False)),
            active=bool(data.get("active", True)),
        )

    @property
    def active_markets(self) -> list[PolymarketMarket]:
        """Markets that are currently active and accepting orders."""
        return [m for m in self.markets if m.is_active]


@dataclass
class OrderBookEntry:
    """A single entry in an order book."""

    price: float
    size: float


@dataclass
class PolymarketOrderBook:
    """Order book for a single token."""

    token_id: str
    bids: list[OrderBookEntry] = field(default_factory=list)
    asks: list[OrderBookEntry] = field(default_factory=list)
    timestamp: datetime | None = None

    @classmethod
    def from_api_response(cls, token_id: str, data: dict) -> PolymarketOrderBook:
        """Parse order book from CLOB API response."""
        bids = [
            OrderBookEntry(price=float(b.get("price", 0)), size=float(b.get("size", 0)))
            for b in data.get("bids", [])
        ]
        asks = [
            OrderBookEntry(price=float(a.get("price", 0)), size=float(a.get("size", 0)))
            for a in data.get("asks", [])
        ]
        return cls(token_id=token_id, bids=bids, asks=asks)

    @property
    def best_bid(self) -> float | None:
        """Best (highest) bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        """Best (lowest) ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def midpoint(self) -> float | None:
        """Midpoint between best bid and best ask."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> float | None:
        """Bid-ask spread."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


@dataclass
class PriceHistoryPoint:
    """A single point in a price history."""

    timestamp: int  # Unix timestamp
    price: float

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)


@dataclass
class PolymarketPriceHistory:
    """Price history for a single token."""

    token_id: str
    points: list[PriceHistoryPoint] = field(default_factory=list)

    @classmethod
    def from_api_response(cls, token_id: str, data: dict) -> PolymarketPriceHistory:
        """Parse price history from CLOB API response."""
        history = data.get("history", [])
        points = [
            PriceHistoryPoint(
                timestamp=int(p.get("t", 0)),
                price=float(p.get("p", 0)),
            )
            for p in history
        ]
        return cls(token_id=token_id, points=points)


def _parse_json_or_list(value) -> list:
    """Parse a value that may be a JSON string or already a list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [parsed]
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _safe_float(value) -> float:
    """Safely convert to float, defaulting to 0.0."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0
