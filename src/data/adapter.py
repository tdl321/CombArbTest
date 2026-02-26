"""Data adapter: wraps existing DuckDB/parquet loaders behind the MarketDataSource protocol.

This is a thin adapter layer — all business logic stays in the existing
loader classes. The adapter just translates between the core types and
the existing API.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Iterator

from src.core.types import (
    GroupSnapshot,
    MarketMeta,
    MarketSnapshot,
    MarketStatus,
    MarketTimeSeries,
    PricePoint,
)

from .loader import BlockLoader, MarketLoader, TradeLoader
from .tick_stream import CrossMarketIterator, CrossMarketSnapshot

logger = logging.getLogger(__name__)


class ParquetMarketSource:
    """Implements MarketDataSource protocol by wrapping existing loaders.

    Delegates all data access to MarketLoader, TradeLoader, BlockLoader,
    and CrossMarketIterator. No logic duplication.
    """

    def __init__(
        self,
        data_dir: str | Path,
        db_path: str | Path | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.db_path = db_path
        self._market_loader = MarketLoader(data_dir, db_path)
        self._trade_loader = TradeLoader(data_dir, db_path)
        self._block_loader = BlockLoader(data_dir, db_path)
        logger.info("[ADAPTER] ParquetMarketSource initialized: data_dir=%s", data_dir)

    def get_market(self, market_id: str) -> MarketMeta | None:
        """Get metadata for a single market."""
        market = self._market_loader.get_market(market_id)
        if market is None:
            return None
        return self._to_market_meta(market)

    def get_markets(
        self,
        market_ids: list[str] | None = None,
        category: str | None = None,
        min_volume: float | None = None,
        limit: int | None = None,
    ) -> list[MarketMeta]:
        """Query markets with filters."""
        df = self._market_loader.query_markets(
            market_ids=market_ids,
            min_volume=min_volume,
            limit=limit,
        )
        results = []
        for row in df.iter_rows(named=True):
            market = self._market_loader._row_to_market(row)
            meta = self._to_market_meta(market)
            if category is not None and meta.category != category:
                continue
            results.append(meta)
        return results

    def get_snapshot(
        self,
        market_id: str,
        at_time: datetime | None = None,
    ) -> MarketSnapshot | None:
        """Get point-in-time market snapshot."""
        market = self._market_loader.get_market(market_id)
        if market is None:
            return None

        meta = self._to_market_meta(market)

        # Build price point from outcome prices
        prices = {}
        for i, outcome in enumerate(market.outcomes):
            if i < len(market.outcome_prices):
                prices[outcome] = market.outcome_prices[i]

        price_point = PricePoint(
            timestamp=at_time or datetime.now(),
            prices=prices,
            source="parquet",
        )

        return MarketSnapshot(
            market=meta,
            price_point=price_point,
        )

    def get_snapshots(
        self,
        market_ids: list[str],
        at_time: datetime | None = None,
    ) -> dict[str, MarketSnapshot]:
        """Get snapshots for multiple markets."""
        result = {}
        for mid in market_ids:
            snap = self.get_snapshot(mid, at_time)
            if snap is not None:
                result[mid] = snap
        return result

    def get_time_series(
        self,
        market_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
        interval_minutes: int = 60,
    ) -> MarketTimeSeries:
        """Get historical price time series for a market.

        Uses TradeLoader to fetch trades and aggregate into price bars.
        """
        market = self._market_loader.get_market(market_id)
        if market is None or market.clob_token_ids is None:
            meta = MarketMeta(
                id=market_id, question="", slug="", outcomes=["Yes", "No"]
            )
            return MarketTimeSeries(market=meta, points=[])

        meta = self._to_market_meta(market)

        # Fetch trades for this market
        trades_df = self._trade_loader.get_trades_for_market(
            clob_token_ids=market.clob_token_ids,
            limit=50000,
        )

        if trades_df.is_empty():
            return MarketTimeSeries(market=meta, points=[])

        # Enrich with timestamps
        trades_df = self._trade_loader.enrich_with_timestamps(trades_df)

        # Filter by time range if specified
        if start is not None or end is not None:
            import polars as pl
            if start is not None:
                trades_df = trades_df.filter(pl.col("block_timestamp") >= start)
            if end is not None:
                trades_df = trades_df.filter(pl.col("block_timestamp") <= end)

        if trades_df.is_empty():
            return MarketTimeSeries(market=meta, points=[])

        # Aggregate into interval bars
        points = self._aggregate_to_bars(
            trades_df, market, interval_minutes
        )

        return MarketTimeSeries(market=meta, points=points)

    def iter_snapshots(
        self,
        market_ids: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Iterator[GroupSnapshot]:
        """Iterate over synchronized snapshots across markets.

        Wraps CrossMarketIterator and converts CrossMarketSnapshot
        to GroupSnapshot.
        """
        iterator = CrossMarketIterator(
            trade_loader=self._trade_loader,
            block_loader=self._block_loader,
            market_loader=self._market_loader,
            market_ids=market_ids,
        )

        # Load market metadata for conversion
        market_meta_cache: dict[str, MarketMeta] = {}
        for mid in market_ids:
            market = self._market_loader.get_market(mid)
            if market is not None:
                market_meta_cache[mid] = self._to_market_meta(market)

        for cross_snap in iterator.iter_snapshots():
            group_snap = self._cross_to_group_snapshot(
                cross_snap, market_meta_cache
            )
            if group_snap is not None:
                # Filter by time range if specified
                if start and group_snap.timestamp < start:
                    continue
                if end and group_snap.timestamp > end:
                    continue
                yield group_snap

    def close(self) -> None:
        """Release resources."""
        self._market_loader.close()
        self._trade_loader.close()
        self._block_loader.close()

    # ── Internal helpers ──

    @staticmethod
    def _to_market_meta(market) -> MarketMeta:
        """Convert existing Market model to MarketMeta."""
        status = MarketStatus.ACTIVE
        if hasattr(market, "closed") and market.closed:
            status = MarketStatus.CLOSED
        elif hasattr(market, "status"):
            try:
                status = MarketStatus(market.status.value)
            except (ValueError, AttributeError):
                pass

        return MarketMeta(
            id=market.id,
            question=market.question,
            slug=market.slug,
            outcomes=market.outcomes if market.outcomes else ["Yes", "No"],
            clob_token_ids=market.clob_token_ids,
            volume=getattr(market, "volume", 0.0) or 0.0,
            liquidity=getattr(market, "liquidity", 0.0) or 0.0,
            status=status,
            end_date=getattr(market, "end_date", None),
            created_at=getattr(market, "created_at", None),
        )

    def _cross_to_group_snapshot(
        self,
        cross_snap: CrossMarketSnapshot,
        meta_cache: dict[str, MarketMeta],
    ) -> GroupSnapshot | None:
        """Convert CrossMarketSnapshot to GroupSnapshot."""
        if cross_snap.timestamp is None:
            return None

        snapshots = {}
        for mid, state in cross_snap.states.items():
            meta = meta_cache.get(mid)
            if meta is None or state.last_price is None:
                continue

            price = float(state.last_price)
            outcomes = meta.outcomes if meta.outcomes else ["Yes", "No"]
            prices = {outcomes[0]: price}
            if len(outcomes) > 1:
                prices[outcomes[1]] = 1.0 - price

            price_point = PricePoint(
                timestamp=cross_snap.timestamp,
                prices=prices,
                source="trade",
            )

            snapshots[mid] = MarketSnapshot(
                market=meta,
                price_point=price_point,
                block_number=cross_snap.position.block_number,
            )

        if not snapshots:
            return None

        return GroupSnapshot(
            group_id="cross_market",
            timestamp=cross_snap.timestamp,
            snapshots=snapshots,
            block_number=cross_snap.position.block_number,
        )

    def _aggregate_to_bars(
        self,
        trades_df,
        market,
        interval_minutes: int,
    ) -> list[PricePoint]:
        """Aggregate trade data into fixed-interval price bars."""
        import polars as pl

        target_token = market.clob_token_ids[0] if market.clob_token_ids else None
        if target_token is None:
            return []

        # Compute prices from maker/taker amounts
        # Filter to trades involving the target token and compute price
        trades_with_price = trades_df.with_columns([
            (pl.col("maker_amount").cast(pl.Float64) / 1_000_000).alias("maker_usd"),
            (pl.col("taker_amount").cast(pl.Float64) / 1_000_000).alias("taker_usd"),
        ])

        # Group by time interval
        if "block_timestamp" not in trades_with_price.columns:
            return []

        trades_with_price = trades_with_price.filter(
            pl.col("block_timestamp").is_not_null()
        )

        if trades_with_price.is_empty():
            return []

        # Use truncated timestamp for grouping
        interval_str = f"{interval_minutes}m"
        grouped = trades_with_price.group_by_dynamic(
            "block_timestamp",
            every=interval_str,
        ).agg([
            pl.col("maker_usd").mean().alias("avg_maker"),
            pl.col("taker_usd").mean().alias("avg_taker"),
            pl.col("maker_usd").sum().alias("total_volume"),
        ])

        points = []
        outcomes = market.outcomes if market.outcomes else ["Yes", "No"]

        for row in grouped.iter_rows(named=True):
            ts = row["block_timestamp"]
            avg_maker = row["avg_maker"]
            avg_taker = row["avg_taker"]

            if avg_taker and avg_taker > 0:
                price = avg_maker / avg_taker
                price = max(0.01, min(0.99, price))
            else:
                continue

            prices = {outcomes[0]: price}
            if len(outcomes) > 1:
                prices[outcomes[1]] = 1.0 - price

            points.append(PricePoint(
                timestamp=ts,
                prices=prices,
                volume=row.get("total_volume", 0.0) or 0.0,
                source="trade",
            ))

        return points
