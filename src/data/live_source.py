"""LiveMarketSource: implements MarketDataSource protocol for live Polymarket data.

Drop-in replacement for ParquetMarketSource in the Pipeline.
Wraps GammaClient + ClobClient behind the standard interface.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterator

from src.core.types import (
    GroupSnapshot,
    MarketMeta,
    MarketSnapshot,
    MarketTimeSeries,
)

from .polymarket.clob_client import ClobClient
from .polymarket.config import PolymarketConfig
from .polymarket.gamma_client import GammaClient
from .polymarket.mapping import (
    to_market_meta,
    to_market_snapshot,
    to_market_time_series,
)

logger = logging.getLogger(__name__)


class LiveMarketSource:
    """MarketDataSource implementation backed by live Polymarket APIs.

    Wraps GammaClient (metadata) and ClobClient (prices) to implement
    the standard MarketDataSource protocol from src/core/protocols.py.

    Usage:
        source = LiveMarketSource()
        market = source.get_market("0x1234...")
        snapshot = source.get_snapshot("0x1234...")
    """

    def __init__(
        self,
        config: PolymarketConfig | None = None,
        gamma: GammaClient | None = None,
        clob: ClobClient | None = None,
    ):
        self._config = config or PolymarketConfig()
        self._gamma = gamma or GammaClient(self._config)
        self._clob = clob or ClobClient(self._config)
        # Simple in-memory cache for metadata
        self._meta_cache: dict[str, MarketMeta] = {}

    def close(self) -> None:
        """Release resources."""
        self._gamma.close()
        self._clob.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── MarketDataSource Protocol ──────────────────────────

    def get_market(self, market_id: str) -> MarketMeta | None:
        """Get metadata for a single market."""
        if market_id in self._meta_cache:
            return self._meta_cache[market_id]

        pm = self._gamma.get_market_by_id(market_id)
        if pm is None:
            return None

        meta = to_market_meta(pm)
        self._meta_cache[market_id] = meta
        return meta

    def get_markets(
        self,
        market_ids: list[str] | None = None,
        category: str | None = None,
        min_volume: float | None = None,
        limit: int | None = None,
    ) -> list[MarketMeta]:
        """Query markets with filters."""
        if market_ids:
            results = []
            for mid in market_ids:
                meta = self.get_market(mid)
                if meta:
                    results.append(meta)
            return results

        # Use Gamma search for broader queries
        pm_markets = self._gamma.get_markets(
            active=True,
            limit=limit or 100,
        )

        metas = [to_market_meta(m) for m in pm_markets]

        # Apply filters
        if category:
            metas = [m for m in metas if m.category == category]
        if min_volume is not None:
            metas = [m for m in metas if m.volume >= min_volume]

        # Cache results
        for meta in metas:
            self._meta_cache[meta.id] = meta

        return metas

    def get_snapshot(
        self,
        market_id: str,
        at_time: datetime | None = None,
    ) -> MarketSnapshot | None:
        """Get point-in-time market snapshot.

        For live data, at_time is ignored (always returns current state).
        Refreshes prices from CLOB midpoint for accuracy.
        """
        pm = self._gamma.get_market_by_id(market_id)
        if pm is None:
            return None

        # Refresh from CLOB midpoint (more accurate than Gamma snapshot)
        if pm.yes_token_id:
            midpoint = self._clob.get_midpoint(pm.yes_token_id)
            if midpoint is not None:
                pm.outcome_prices = [midpoint, 1.0 - midpoint]

        return to_market_snapshot(pm, source="clob_midpoint")

    def get_snapshots(
        self,
        market_ids: list[str],
        at_time: datetime | None = None,
    ) -> dict[str, MarketSnapshot]:
        """Get snapshots for multiple markets."""
        results: dict[str, MarketSnapshot] = {}
        for mid in market_ids:
            snap = self.get_snapshot(mid, at_time)
            if snap:
                results[mid] = snap
        return results

    def get_time_series(
        self,
        market_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
        interval_minutes: int = 60,
    ) -> MarketTimeSeries:
        """Get historical price time series from CLOB prices-history."""
        pm = self._gamma.get_market_by_id(market_id)
        if pm is None or not pm.yes_token_id:
            meta = self.get_market(market_id)
            from src.core.types import MarketMeta, PricePoint
            if meta is None:
                meta = MarketMeta(id=market_id, question="Unknown", slug="", outcomes=["Yes", "No"])
            return MarketTimeSeries(market=meta, points=[])

        # Map interval_minutes to CLOB interval param
        if interval_minutes <= 60:
            interval = "1d"
            fidelity = interval_minutes
        elif interval_minutes <= 1440:
            interval = "1w"
            fidelity = interval_minutes
        else:
            interval = "max"
            fidelity = interval_minutes

        history = self._clob.get_price_history(
            pm.yes_token_id,
            interval=interval,
            fidelity=max(1, fidelity),
        )

        if history is None:
            return MarketTimeSeries(market=to_market_meta(pm), points=[])

        ts = to_market_time_series(pm, history)

        # Filter by start/end if provided
        if start or end:
            s = start or datetime.min
            e = end or datetime.max
            ts = ts.window(s, e)

        return ts

    def iter_snapshots(
        self,
        market_ids: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Iterator[GroupSnapshot]:
        """Yield a single snapshot with current state for all markets.

        For live data, we yield one snapshot (the current state).
        For streaming, use a separate WebSocket-based monitor.
        """
        snapshots = self.get_snapshots(market_ids)
        if snapshots:
            yield GroupSnapshot(
                group_id="live",
                timestamp=datetime.utcnow(),
                snapshots=snapshots,
            )
