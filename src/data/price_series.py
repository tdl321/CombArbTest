"""Price series builder for Polymarket trade data."""

from datetime import datetime, timedelta
from enum import Enum

import polars as pl

from .loader import BlockLoader, MarketLoader, TradeLoader


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
        market = self.market_loader.get_market(market_id)
        if market is None:
            raise ValueError(f"Market {market_id} not found")
        
        if market.clob_token_ids is None or len(market.clob_token_ids) <= outcome_index:
            raise ValueError(f"Market {market_id} has no CLOB token ID for outcome {outcome_index}")

        target_token_id = market.clob_token_ids[outcome_index]

        # Get trades
        trades_df = self.trade_loader.get_trades_for_market(
            market.clob_token_ids,
            limit=limit,
        )

        if trades_df.is_empty():
            return self._empty_series()

        # Enrich with timestamps
        trades_df = self.trade_loader.enrich_with_timestamps(trades_df)

        # Filter by time range
        if start_time is not None:
            trades_df = trades_df.filter(pl.col("block_timestamp") >= start_time)
        if end_time is not None:
            trades_df = trades_df.filter(pl.col("block_timestamp") < end_time)

        if trades_df.is_empty():
            return self._empty_series()

        # Calculate price for each trade
        # Price is always USDC per outcome token
        # maker_asset_id == '0' means maker provides USDC (buying tokens)
        # taker_asset_id == '0' means taker provides USDC (selling tokens)
        trades_df = trades_df.with_columns([
            (pl.col("maker_amount") / 1_000_000).alias("maker_usd"),
            (pl.col("taker_amount") / 1_000_000).alias("taker_usd"),
        ]).with_columns([
            # Price calculation: USDC / tokens
            pl.when(pl.col("maker_asset_id") == "0")
            .then(pl.col("maker_usd") / pl.col("taker_usd"))  # BUY: USDC paid / tokens received
            .when(pl.col("taker_asset_id") == "0")
            .then(pl.col("taker_usd") / pl.col("maker_usd"))  # SELL: USDC received / tokens sold
            .otherwise(None)
            .alias("price"),
            # Volume in tokens
            pl.when(pl.col("maker_asset_id") == "0")
            .then(pl.col("taker_usd"))  # Tokens bought
            .when(pl.col("taker_asset_id") == "0")
            .then(pl.col("maker_usd"))  # Tokens sold
            .otherwise(pl.lit(0.0))
            .alias("volume"),
            # Is this trade for our target outcome?
            pl.when(
                (pl.col("maker_asset_id") == target_token_id) | 
                (pl.col("taker_asset_id") == target_token_id)
            ).then(True).otherwise(False).alias("is_target"),
        ])

        # Filter to only trades for the target outcome
        trades_df = trades_df.filter(pl.col("is_target"))

        # Filter out null prices
        trades_df = trades_df.filter(pl.col("price").is_not_null())

        if trades_df.is_empty():
            return self._empty_series()

        # Aggregate into OHLCV buckets
        bucket_seconds = RESOLUTION_SECONDS[resolution]
        
        ohlcv = (
            trades_df
            .with_columns([
                (pl.col("block_timestamp").dt.truncate(f"{bucket_seconds}s")).alias("bucket"),
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

        return ohlcv

    def get_price_at_time(
        self,
        market_id: str,
        as_of: datetime,
        outcome_index: int = 0,
        lookback_trades: int = 1000,
    ) -> float | None:
        """Get the most recent price as of a specific time (DATA-04)."""
        market = self.market_loader.get_market(market_id)
        if market is None or market.clob_token_ids is None:
            return None

        target_token_id = market.clob_token_ids[outcome_index]

        # Get recent trades
        trades_df = self.trade_loader.get_trades_for_market(
            market.clob_token_ids,
            limit=lookback_trades,
        )

        if trades_df.is_empty():
            return None

        # Enrich with timestamps
        trades_df = self.trade_loader.enrich_with_timestamps(trades_df)
        
        # Filter to trades before as_of
        trades_df = trades_df.filter(pl.col("block_timestamp") <= as_of)

        if trades_df.is_empty():
            return None

        # Calculate prices
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

        # Filter to target outcome with valid prices
        trades_df = trades_df.filter(
            pl.col("is_target") & pl.col("price").is_not_null()
        ).sort("block_timestamp", descending=True)

        if trades_df.is_empty():
            return None

        return trades_df["price"][0]

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

    def get_market(self, market_id: str):
        market = self.market_loader.get_market(market_id)
        if market is None:
            return None
        if market.created_at is not None and market.created_at > self.as_of:
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
