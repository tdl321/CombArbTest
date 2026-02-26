"""Core types shared across all layers.

These are the canonical data types. All modules reference these,
not their own copies.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────
# Market Data Types
# ─────────────────────────────────────────────────────────

class MarketStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    RESOLVED = "resolved"


class MarketMeta(BaseModel):
    """Immutable market metadata. Shared everywhere."""
    id: str
    question: str
    slug: str
    outcomes: list[str]
    clob_token_ids: list[str] | None = None
    volume: float = 0.0
    liquidity: float = 0.0
    status: MarketStatus = MarketStatus.ACTIVE
    category: str | None = None
    subcategory: str | None = None
    end_date: datetime | None = None
    created_at: datetime | None = None


class PricePoint(BaseModel):
    """A single price observation."""
    timestamp: datetime
    prices: dict[str, float]  # outcome_name -> price
    volume: float = 0.0
    source: str = "trade"  # "trade", "orderbook_mid", "api"

    @property
    def total(self) -> float:
        return sum(self.prices.values())


class MarketSnapshot(BaseModel):
    """Point-in-time state of one market. The universal currency of the pipeline."""
    market: MarketMeta
    price_point: PricePoint
    block_number: int | None = None

    @property
    def yes_price(self) -> float:
        """Convenience: first outcome price (typically YES)."""
        if self.market.outcomes:
            return self.price_point.prices.get(self.market.outcomes[0], 0.5)
        return 0.5

    @property
    def timestamp(self) -> datetime:
        return self.price_point.timestamp


class MarketTimeSeries(BaseModel):
    """Historical prices for a single market. For strategies needing history."""
    market: MarketMeta
    points: list[PricePoint]

    @property
    def timestamps(self) -> list[datetime]:
        return [p.timestamp for p in self.points]

    @property
    def yes_prices(self) -> list[float]:
        outcome = self.market.outcomes[0] if self.market.outcomes else "Yes"
        return [p.prices.get(outcome, 0.5) for p in self.points]

    def window(self, start: datetime, end: datetime) -> MarketTimeSeries:
        """Extract a time window."""
        filtered = [p for p in self.points if start <= p.timestamp <= end]
        return MarketTimeSeries(market=self.market, points=filtered)

    def latest(self, n: int = 1) -> list[PricePoint]:
        """Get the N most recent price points."""
        return self.points[-n:]


class GroupSnapshot(BaseModel):
    """Point-in-time state of a group of markets."""
    group_id: str
    timestamp: datetime
    snapshots: dict[str, MarketSnapshot]  # market_id -> snapshot
    block_number: int | None = None

    @property
    def market_ids(self) -> list[str]:
        return list(self.snapshots.keys())

    @property
    def yes_prices(self) -> dict[str, float]:
        return {mid: snap.yes_price for mid, snap in self.snapshots.items()}

    def has_all_prices(self) -> bool:
        return all(snap.price_point.prices for snap in self.snapshots.values())


# ─────────────────────────────────────────────────────────
# Grouping Types
# ─────────────────────────────────────────────────────────

class GroupingType(str, Enum):
    """How markets should be grouped for a strategy."""
    PARTITION = "partition"           # Mutually exclusive + exhaustive
    SEMANTIC = "semantic"             # LLM-identified relationships
    CORRELATION = "correlation"       # Price correlation clustering
    CATEGORY = "category"             # Rule-based category
    CROSS_PLATFORM = "cross_platform" # Same event on different platforms
    MANUAL = "manual"                 # Hardcoded groupings


class ConstraintType(str, Enum):
    """Types of logical constraints between markets."""
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"
    EXHAUSTIVE = "exhaustive"
    IMPLIES = "implies"
    PREREQUISITE = "prerequisite"
    INCOMPATIBLE = "incompatible"
    EQUIVALENT = "equivalent"
    OPPOSITE = "opposite"


class Constraint(BaseModel):
    """A single logical constraint between markets."""
    type: ConstraintType
    from_market: str
    to_market: str | None = None
    confidence: float = 1.0
    reasoning: str | None = None


class MarketGroup(BaseModel):
    """A group of related markets with their constraints."""
    group_id: str
    name: str
    market_ids: list[str]
    group_type: GroupingType
    constraints: list[Constraint] = Field(default_factory=list)
    is_partition: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.market_ids)


# ─────────────────────────────────────────────────────────
# Trade / Opportunity Types
# ─────────────────────────────────────────────────────────

class TradeDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradeLeg(BaseModel):
    """One leg of a multi-leg trade."""
    market_id: str
    direction: TradeDirection
    target_price: float
    size: float = 1.0  # In units of the base currency


class TradeType(str, Enum):
    INSTANT_ARB = "instant_arb"     # Enter and exit immediately
    HELD_POSITION = "held_position" # Enter now, exit later
    REBALANCE = "rebalance"         # Adjust existing position


class Opportunity(BaseModel):
    """A detected trading opportunity from a strategy."""
    strategy_name: str
    group_id: str
    timestamp: datetime
    legs: list[TradeLeg]
    expected_profit: float          # Gross expected profit
    confidence: float = 1.0
    trade_type: TradeType = TradeType.INSTANT_ARB
    hold_duration: float | None = None  # Expected hold in hours (None=instant)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def num_legs(self) -> int:
        return len(self.legs)

    @property
    def market_ids(self) -> list[str]:
        return [leg.market_id for leg in self.legs]


class ExecutedTrade(BaseModel):
    """A trade that has been sized, validated, and marked for execution."""
    opportunity: Opportunity
    entry_time: datetime
    exit_time: datetime | None = None  # None if still open
    entry_prices: dict[str, float]     # Actual fill prices
    exit_prices: dict[str, float] | None = None
    size_usd: float = 1.0             # Total position size in USD
    fees_paid: float = 0.0
    realized_pnl: float | None = None  # Set when position closes
    unrealized_pnl: float = 0.0        # Mark-to-market

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def total_pnl(self) -> float:
        if self.realized_pnl is not None:
            return self.realized_pnl - self.fees_paid
        return self.unrealized_pnl - self.fees_paid


# ─────────────────────────────────────────────────────────
# Strategy Configuration
# ─────────────────────────────────────────────────────────

class DataRequirements(BaseModel):
    """What data a strategy needs."""
    needs_snapshots: bool = True       # Point-in-time prices
    needs_time_series: bool = False    # Historical price bars
    lookback_periods: int = 0          # How many historical bars needed
    needs_volume: bool = False
    needs_orderbook: bool = False
    tick_level: bool = False           # Needs raw tick data
    min_markets_per_group: int = 2


class StrategyConfig(BaseModel):
    """Base configuration for any strategy."""
    strategy_name: str
    min_profit_threshold: float = 0.001
    fee_per_leg: float = 0.01
    max_position_size: float = 1.0
    extra: dict[str, Any] = Field(default_factory=dict)
