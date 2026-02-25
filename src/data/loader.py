"""Data loaders for Polymarket parquet files using DuckDB.

Memory-efficient implementation for 8GB servers.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl

from .models import Market

logger = logging.getLogger(__name__)


class DataLoader:
    """Base class for parquet data loading with DuckDB."""

    def __init__(self, data_dir: str | Path, db_path: str | Path | None = None):
        self.data_dir = Path(data_dir)
        self.db_path = Path(db_path) if db_path else None
        self._conn: duckdb.DuckDBPyConnection | None = None
        logger.debug("[DATA] Initializing DataLoader with data_dir=%s", data_dir)

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            logger.debug("[DATA] Creating DuckDB connection")
            if self.db_path:
                self._conn = duckdb.connect(str(self.db_path))
            else:
                self._conn = duckdb.connect()
            self._conn.execute("SET enable_progress_bar = false")
            self._conn.execute("SET memory_limit = '3GB'")
            logger.debug("[DATA] DuckDB connection established with 3GB memory limit")
        return self._conn

    def _get_parquet_glob(self, subdir: str) -> str:
        return str(self.data_dir / subdir / "[!._]*.parquet")

    def close(self) -> None:
        if self._conn is not None:
            logger.debug("[DATA] Closing DuckDB connection")
            self._conn.close()
            self._conn = None


class MarketLoader(DataLoader):
    """Load and query Polymarket market metadata."""

    def __init__(self, data_dir: str | Path, db_path: str | Path | None = None):
        super().__init__(data_dir, db_path)
        self._markets_dir = "markets"
        logger.debug("[DATA] MarketLoader initialized")

    def query_markets(
        self,
        *,
        market_ids: list[str] | None = None,
        min_volume: float | None = None,
        max_volume: float | None = None,
        active: bool | None = None,
        closed: bool | None = None,
        limit: int | None = None,
    ) -> pl.DataFrame:
        logger.debug("[DATA] Querying markets: ids=%s, min_vol=%s, max_vol=%s, active=%s, closed=%s, limit=%s",
                     len(market_ids) if market_ids else None, min_volume, max_volume, active, closed, limit)
        start_time = time.time()
        
        glob_pattern = self._get_parquet_glob(self._markets_dir)

        conditions = []
        params = []

        if market_ids is not None:
            placeholders = ", ".join(["?" for _ in market_ids])
            conditions.append("id IN (%s)" % placeholders)
            params.extend(market_ids)

        if min_volume is not None:
            conditions.append("volume >= ?")
            params.append(min_volume)

        if max_volume is not None:
            conditions.append("volume <= ?")
            params.append(max_volume)

        if active is not None:
            conditions.append("active = ?")
            params.append(active)

        if closed is not None:
            conditions.append("closed = ?")
            params.append(closed)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        limit_clause = "LIMIT %d" % limit if limit else ""

        query = """
            SELECT * FROM read_parquet('%s')
            WHERE %s
            ORDER BY volume DESC
            %s
        """ % (glob_pattern, where_clause, limit_clause)

        result = self.conn.execute(query, params).pl()
        elapsed = time.time() - start_time
        logger.info("[DATA] Query returned %d markets in %.3fs", len(result), elapsed)
        return result

    def get_market(self, market_id: str) -> Market | None:
        logger.debug("[DATA] Getting market: %s", market_id)
        df = self.query_markets(market_ids=[market_id], limit=1)
        if df.is_empty():
            logger.warning("[DATA] Market not found: %s", market_id)
            return None
        market = self._row_to_market(df.row(0, named=True))
        logger.debug("[DATA] Market found: %s (%s)", market_id, market.question[:50])
        return market

    def _row_to_market(self, row: dict) -> Market:
        outcomes = json.loads(row["outcomes"]) if row.get("outcomes") else []
        outcome_prices = json.loads(row["outcome_prices"]) if row.get("outcome_prices") else []
        clob_token_ids = json.loads(row["clob_token_ids"]) if row.get("clob_token_ids") else None

        if outcome_prices and isinstance(outcome_prices[0], str):
            outcome_prices = [float(p) for p in outcome_prices]

        return Market(
            id=row["id"],
            condition_id=row["condition_id"],
            question=row["question"],
            slug=row["slug"],
            outcomes=outcomes,
            outcome_prices=outcome_prices,
            clob_token_ids=clob_token_ids,
            volume=row.get("volume", 0.0),
            liquidity=row.get("liquidity", 0.0),
            active=row.get("active", False),
            closed=row.get("closed", False),
            end_date=row.get("end_date"),
            created_at=row.get("created_at"),
            market_maker_address=row.get("market_maker_address"),
            fetched_at=row.get("_fetched_at"),
        )


class BlockLoader(DataLoader):
    """Load block number to timestamp mappings.
    
    Uses on-demand queries instead of loading all blocks into memory.
    """

    def __init__(self, data_dir: str | Path, db_path: str | Path | None = None):
        super().__init__(data_dir, db_path)
        self._blocks_dir = "blocks"
        self._view_created = False
        logger.debug("[DATA] BlockLoader initialized")

    def _ensure_view(self) -> None:
        if self._view_created:
            return
        logger.debug("[DATA] Creating blocks view")
        glob_pattern = self._get_parquet_glob(self._blocks_dir)
        self.conn.execute("""
            CREATE OR REPLACE VIEW blocks AS
            SELECT * FROM read_parquet('%s')
        """ % glob_pattern)
        self._view_created = True
        logger.debug("[DATA] Blocks view created")

    def get_timestamp(self, block_number: int) -> datetime | None:
        """Get timestamp for a specific block number."""
        logger.debug("[DATA] Getting timestamp for block %d", block_number)
        self._ensure_view()
        result = self.conn.execute(
            "SELECT timestamp FROM blocks WHERE block_number = ?",
            [block_number]
        ).fetchone()
        if result is None:
            logger.warning("[DATA] Block not found: %d", block_number)
            return None
        ts_str = result[0]
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

    def get_timestamps_batch(self, block_numbers: list[int]) -> dict[int, datetime]:
        """Get timestamps for multiple block numbers efficiently."""
        if not block_numbers:
            return {}
        
        logger.debug("[DATA] Getting timestamps for %d blocks", len(block_numbers))
        start_time = time.time()
        
        self._ensure_view()
        placeholders = ", ".join(["?" for _ in block_numbers])
        result = self.conn.execute(
            "SELECT block_number, timestamp FROM blocks WHERE block_number IN (%s)" % placeholders,
            block_numbers
        ).fetchall()
        
        timestamps = {}
        for bn, ts_str in result:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            timestamps[bn] = ts
        
        elapsed = time.time() - start_time
        logger.debug("[DATA] Retrieved %d timestamps in %.3fs", len(timestamps), elapsed)
        return timestamps


class TradeLoader(DataLoader):
    """Load and query Polymarket trade data."""

    def __init__(self, data_dir: str | Path, db_path: str | Path | None = None, block_loader: BlockLoader | None = None):
        super().__init__(data_dir, db_path)
        self._trades_dir = "trades"
        self.block_loader = block_loader or BlockLoader(data_dir, db_path)
        self._view_created = False
        logger.debug("[DATA] TradeLoader initialized")

    def _ensure_view(self) -> None:
        if self._view_created:
            return
        logger.debug("[DATA] Creating trades view")
        glob_pattern = self._get_parquet_glob(self._trades_dir)
        self.conn.execute("""
            CREATE OR REPLACE VIEW trades AS
            SELECT * FROM read_parquet('%s')
        """ % glob_pattern)
        self._view_created = True
        logger.debug("[DATA] Trades view created")

    def query_trades(
        self,
        *,
        asset_ids: list[str] | None = None,
        min_block: int | None = None,
        max_block: int | None = None,
        limit: int | None = 10000,
    ) -> pl.DataFrame:
        """Query trades with filters."""
        logger.debug("[DATA] Querying trades: assets=%s, min_block=%s, max_block=%s, limit=%s",
                     len(asset_ids) if asset_ids else None, min_block, max_block, limit)
        start_time = time.time()
        
        self._ensure_view()

        conditions = []
        params = []

        if asset_ids is not None:
            placeholders = ", ".join(["?" for _ in asset_ids])
            conditions.append("(maker_asset_id IN (%s) OR taker_asset_id IN (%s))" % (placeholders, placeholders))
            params.extend(asset_ids)
            params.extend(asset_ids)

        if min_block is not None:
            conditions.append("block_number >= ?")
            params.append(min_block)

        if max_block is not None:
            conditions.append("block_number < ?")
            params.append(max_block)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        limit_clause = "LIMIT %d" % limit if limit else ""

        query = """
            SELECT * FROM trades
            WHERE %s
            ORDER BY block_number ASC
            %s
        """ % (where_clause, limit_clause)

        result = self.conn.execute(query, params).pl()
        elapsed = time.time() - start_time
        logger.info("[DATA] Query returned %d trades in %.3fs", len(result), elapsed)
        return result

    def get_trades_for_market(
        self,
        clob_token_ids: list[str],
        min_block: int | None = None,
        max_block: int | None = None,
        limit: int | None = 10000,
    ) -> pl.DataFrame:
        """Get trades for a market by its CLOB token IDs."""
        logger.debug("[DATA] Getting trades for market with %d token IDs", len(clob_token_ids))
        return self.query_trades(
            asset_ids=clob_token_ids,
            min_block=min_block,
            max_block=max_block,
            limit=limit,
        )

    def enrich_with_timestamps(self, trades_df: pl.DataFrame) -> pl.DataFrame:
        """Add timestamps to trades using block loader."""
        if trades_df.is_empty():
            return trades_df.with_columns(pl.lit(None).alias("block_timestamp"))

        logger.debug("[DATA] Enriching %d trades with timestamps", len(trades_df))
        block_numbers = trades_df["block_number"].unique().to_list()
        timestamps = self.block_loader.get_timestamps_batch(block_numbers)
        
        return trades_df.with_columns(
            pl.col("block_number").map_elements(
                lambda bn: timestamps.get(bn),
                return_dtype=pl.Datetime("us", "UTC")
            ).alias("block_timestamp")
        )
