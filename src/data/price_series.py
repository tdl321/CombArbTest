"""Price series builder for Polymarket trade data."""

import logging
import time
from datetime import datetime, timedelta
from enum import Enum

import polars as pl

from .loader import BlockLoader, MarketLoader, TradeLoader

logger = logging.getLogger(__name__)


class Resolution(str, Enum):
    MINUTE = "1m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    HOUR = "1h"
    FOUR_HOUR = "4h"
    DAY = "1d"


RESOLUTION_SECONDS = {
    Resolution.MINUTE: 60,
    Resolution.FIVE_MINUTE: 300,
    Resolution.FIFTEEN_MINUTE: 900,
    Resolution.HOUR: 3600,
    Resolution.FOUR_HOUR: 14400,
    Resolution.DAY: 86400,
}


class PriceSeriesBuilder:
    """Build OHLCV price series from trade data."""

    def __init__(
        self,
        trade_loader: TradeLoader,
        market_loader: MarketLoader,
        block_loader: BlockLoader,
    ):
        self.trade_loader = trade_loader
        self.market_loader = market_loader
        self.block_loader = block_loader
        logger.debug("[DATA] PriceSeriesBuilder initialized")

    def build_price_series(
        self,
        market_id: str,
        resolution: Resolution = Resolution.HOUR,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        outcome_index: int = 0,
        limit: int = 50000,
    ) -> pl.DataFrame:
        """Build OHLCV price series for a market."""
        logger.info("[DATA] Building price series for market %s, resolution=%s, outcome=%d",
                    market_id, resolution.value, outcome_index)
        start_ts = time.time()
        
        market = self.market_loader.get_market(market_id)
        if market is None:
            logger.error("[DATA] Market not found: %s", market_id)
            raise ValueError("Market %s not found" % market_id)
        
        if market.clob_token_ids is None or len(market.clob_token_ids) <= outcome_index:
            logger.error("[DATA] Market %s has no CLOB token ID for outcome %d", market_id, outcome_index)
            raise ValueError("Market %s has no CLOB token ID for outcome %d" % (market_id, outcome_index))

        target_token_id = market.clob_token_ids[outcome_index]
        logger.debug("[DATA] Target token ID: %s", target_token_id)

        # Get trades
        trades_df = self.trade_loader.get_trades_for_market(
            market.clob_token_ids,
            limit=limit,
        )

        if trades_df.is_empty():
            logger.warning("[DATA] No trades found for market %s", market_id)
            return self._empty_series()

        # Enrich with timestamps
        trades_df = self.trade_loader.enrich_with_timestamps(trades_df)

        # Filter by time range
        if start_time is not None:
            trades_df = trades_df.filter(pl.col("block_timestamp") >= start_time)
        if end_time is not None:
            trades_df = trades_df.filter(pl.col("block_timestamp") < end_time)

        if trades_df.is_empty():
            logger.warning("[DATA] No trades in time range for market %s", market_id)
            return self._empty_series()

        # Calculate price for each trade
        trades_df = trades_df.with_columns([
            (pl.col("maker_amount") / 1_000_000).alias("maker_usd"),
            (pl.col("taker_amount") / 1_000_000).alias("taker_usd"),
        ]).with_columns([
            pl.when(pl.col("maker_asset_id") == "0")
            .then(pl.col("maker_usd") / pl.col("taker_usd"))
            .when(pl.col("taker_asset_id") == "0")
            .then(pl.col("taker_usd") / pl.col("maker_usd"))
            .otherwise(None)
            .alias("price"),
            pl.when(pl.col("maker_asset_id") == "0")
            .then(pl.col("taker_usd"))
            .when(pl.col("taker_asset_id") == "0")
            .then(pl.col("maker_usd"))
            .otherwise(pl.lit(0.0))
            .alias("volume"),
            pl.when(
                (pl.col("maker_asset_id") == target_token_id) | 
                (pl.col("taker_asset_id") == target_token_id)
            ).then(True).otherwise(False).alias("is_target"),
        ])

        trades_df = trades_df.filter(pl.col("is_target"))
        trades_df = trades_df.filter(pl.col("price").is_not_null())

        if trades_df.is_empty():
            logger.warning("[DATA] No valid price data for market %s", market_id)
            return self._empty_series()

        # Aggregate into OHLCV buckets
        bucket_seconds = RESOLUTION_SECONDS[resolution]
        
        ohlcv = (
            trades_df
            .with_columns([
                (pl.col("block_timestamp").dt.truncate("%ds" % bucket_seconds)).alias("bucket"),
            ])
            .group_by("bucket")
            .agg([
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.len().alias("trade_count"),
            ])
            .sort("bucket")
            .rename({"bucket": "timestamp"})
        )

        elapsed = time.time() - start_ts
        logger.info("[DATA] Built price series: %d candles in %.3fs", len(ohlcv), elapsed)
        return ohlcv

    def get_price_at_time(
        self,
        market_id: str,
        as_of: datetime,
        outcome_index: int = 0,
        lookback_trades: int = 1000,
    ) -> float | None:
        """Get the most recent price as of a specific time (DATA-04)."""
        logger.debug("[DATA] Getting price at %s for market %s", as_of, market_id)
        
        market = self.market_loader.get_market(market_id)
        if market is None or market.clob_token_ids is None:
            logger.warning("[DATA] Market not found or no token IDs: %s", market_id)
            return None

        target_token_id = market.clob_token_ids[outcome_index]

        trades_df = self.trade_loader.get_trades_for_market(
            market.clob_token_ids,
            limit=lookback_trades,
        )

        if trades_df.is_empty():
            logger.debug("[DATA] No trades found for market %s", market_id)
            return None

        trades_df = self.trade_loader.enrich_with_timestamps(trades_df)
        trades_df = trades_df.filter(pl.col("block_timestamp") <= as_of)

        if trades_df.is_empty():
            logger.debug("[DATA] No trades before %s for market %s", as_of, market_id)
            return None

        trades_df = trades_df.with_columns([
            (pl.col("maker_amount") / 1_000_000).alias("maker_usd"),
            (pl.col("taker_amount") / 1_000_000).alias("taker_usd"),
        ]).with_columns([
            pl.when(pl.col("maker_asset_id") == "0")
            .then(pl.col("maker_usd") / pl.col("taker_usd"))
            .when(pl.col("taker_asset_id") == "0")
            .then(pl.col("taker_usd") / pl.col("maker_usd"))
            .otherwise(None)
            .alias("price"),
            pl.when(
                (pl.col("maker_asset_id") == target_token_id) | 
                (pl.col("taker_asset_id") == target_token_id)
            ).then(True).otherwise(False).alias("is_target"),
        ])

        trades_df = trades_df.filter(
            pl.col("is_target") & pl.col("price").is_not_null()
        ).sort("block_timestamp", descending=True)

        if trades_df.is_empty():
            logger.debug("[DATA] No valid prices for market %s at %s", market_id, as_of)
            return None

        price = trades_df["price"][0]
        logger.debug("[DATA] Price for %s at %s: %.4f", market_id, as_of, price)
        return price

    def _empty_series(self) -> pl.DataFrame:
        return pl.DataFrame({
            "timestamp": pl.Series([], dtype=pl.Datetime),
            "open": pl.Series([], dtype=pl.Float64),
            "high": pl.Series([], dtype=pl.Float64),
            "low": pl.Series([], dtype=pl.Float64),
            "close": pl.Series([], dtype=pl.Float64),
            "volume": pl.Series([], dtype=pl.Float64),
            "trade_count": pl.Series([], dtype=pl.UInt32),
        })


class PointInTimeDataAccess:
    """Wrapper for point-in-time market data access (DATA-04)."""

    def __init__(
        self,
        market_loader: MarketLoader,
        trade_loader: TradeLoader,
        price_builder: PriceSeriesBuilder,
        as_of: datetime,
    ):
        self.market_loader = market_loader
        self.trade_loader = trade_loader
        self.price_builder = price_builder
        self.as_of = as_of
        logger.debug("[DATA] PointInTimeDataAccess initialized for %s", as_of)

    def get_market(self, market_id: str):
        logger.debug("[DATA] Getting market %s as of %s", market_id, self.as_of)
        market = self.market_loader.get_market(market_id)
        if market is None:
            return None
        if market.created_at is not None and market.created_at > self.as_of:
            logger.debug("[DATA] Market %s did not exist at %s", market_id, self.as_of)
            return None
        return market

    def get_price(self, market_id: str, outcome_index: int = 0) -> float | None:
        return self.price_builder.get_price_at_time(
            market_id, self.as_of, outcome_index
        )

    def get_price_series(
        self,
        market_id: str,
        resolution: Resolution = Resolution.HOUR,
        lookback: timedelta | None = None,
        outcome_index: int = 0,
    ) -> pl.DataFrame:
        start_time = None
        if lookback is not None:
            start_time = self.as_of - lookback

        return self.price_builder.build_price_series(
            market_id,
            resolution=resolution,
            start_time=start_time,
            end_time=self.as_of,
            outcome_index=outcome_index,
        )
