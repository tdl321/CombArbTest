"""Convert raw Polymarket types to existing core types.

Bridges the gap between Polymarket API responses and the
canonical types in src/core/types.py.
"""

from __future__ import annotations

from datetime import datetime

from src.core.types import (
    GroupSnapshot,
    MarketMeta,
    MarketSnapshot,
    MarketStatus,
    MarketTimeSeries,
    PricePoint,
)

from .types import PolymarketEvent, PolymarketMarket, PolymarketPriceHistory


def to_market_meta(market: PolymarketMarket) -> MarketMeta:
    """Convert a PolymarketMarket to a canonical MarketMeta."""
    if market.closed:
        status = MarketStatus.CLOSED
    elif not market.active:
        status = MarketStatus.RESOLVED
    else:
        status = MarketStatus.ACTIVE

    end_date = None
    if market.end_date:
        try:
            end_date = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
        except ValueError:
            pass

    created_at = None
    if market.created_at:
        try:
            created_at = datetime.fromisoformat(market.created_at.replace("Z", "+00:00"))
        except ValueError:
            pass

    return MarketMeta(
        id=market.condition_id,
        question=market.question,
        slug=market.slug,
        outcomes=market.outcomes if market.outcomes else ["Yes", "No"],
        clob_token_ids=market.clob_token_ids if market.clob_token_ids else None,
        volume=market.volume,
        liquidity=market.liquidity,
        status=status,
        category=market.category,
        end_date=end_date,
        created_at=created_at,
    )


def to_price_point(market: PolymarketMarket, source: str = "api") -> PricePoint:
    """Convert a PolymarketMarket to a PricePoint (current prices)."""
    outcomes = market.outcomes if market.outcomes else ["Yes", "No"]
    prices = {}
    for i, outcome in enumerate(outcomes):
        if i < len(market.outcome_prices):
            prices[outcome] = market.outcome_prices[i]
        else:
            prices[outcome] = 0.5

    return PricePoint(
        timestamp=datetime.utcnow(),
        prices=prices,
        volume=market.volume,
        source=source,
    )


def to_market_snapshot(market: PolymarketMarket, source: str = "api") -> MarketSnapshot:
    """Convert a PolymarketMarket to a full MarketSnapshot."""
    return MarketSnapshot(
        market=to_market_meta(market),
        price_point=to_price_point(market, source=source),
    )


def to_market_time_series(
    market: PolymarketMarket,
    history: PolymarketPriceHistory,
) -> MarketTimeSeries:
    """Convert a market + price history to a MarketTimeSeries."""
    meta = to_market_meta(market)
    outcomes = meta.outcomes if meta.outcomes else ["Yes", "No"]
    yes_outcome = outcomes[0]

    points = []
    for hp in history.points:
        prices = {yes_outcome: hp.price}
        if len(outcomes) > 1:
            prices[outcomes[1]] = 1.0 - hp.price
        points.append(PricePoint(
            timestamp=hp.datetime,
            prices=prices,
            source="clob_history",
        ))

    return MarketTimeSeries(market=meta, points=points)


def to_group_snapshot(
    event: PolymarketEvent,
    markets: dict[str, PolymarketMarket],
    source: str = "api",
) -> GroupSnapshot:
    """Convert an event + markets dict to a GroupSnapshot."""
    snapshots = {}
    for market_id, market in markets.items():
        snapshots[market_id] = to_market_snapshot(market, source=source)

    timestamp = datetime.utcnow()
    if snapshots:
        # Use the first snapshot timestamp
        first = next(iter(snapshots.values()))
        timestamp = first.timestamp

    return GroupSnapshot(
        group_id=event.event_id,
        timestamp=timestamp,
        snapshots=snapshots,
    )
