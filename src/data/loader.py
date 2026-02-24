"""Data loaders for Polymarket parquet files using DuckDB.

Memory-efficient implementation for 8GB servers.
"""

import json
from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl

from .models import Market


class DataLoader:
    """Base class for parquet data loading with DuckDB."""

    def __init__(self, data_dir: str | Path, db_path: str | Path | None = None):
        self.data_dir = Path(data_dir)
        self.db_path = Path(db_path) if db_path else None
        self._conn: duckdb.DuckDBPyConnection | None = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            if self.db_path:
                self._conn = duckdb.connect(str(self.db_path))
            else:
                self._conn = duckdb.connect()
            self._conn.execute("SET enable_progress_bar = false")
            self._conn.execute("SET memory_limit = '3GB'")
        return self._conn

    def _get_parquet_glob(self, subdir: str) -> str:
        return str(self.data_dir / subdir / "[!._]*.parquet")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class MarketLoader(DataLoader):
    """Load and query Polymarket market metadata."""

    def __init__(self, data_dir: str | Path, db_path: str | Path | None = None):
        super().__init__(data_dir, db_path)
        self._markets_dir = "markets"

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
        glob_pattern = self._get_parquet_glob(self._markets_dir)

        conditions = []
        params = []

        if market_ids is not None:
            placeholders = ", ".join(["?" for _ in market_ids])
            conditions.append(f"id IN ({placeholders})")
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
        limit_clause = f"LIMIT {limit}" if limit else ""

        query = f"""
            SELECT * FROM read_parquet('{glob_pattern}')
            WHERE {where_clause}
            ORDER BY volume DESC
            {limit_clause}
        """

        result = self.conn.execute(query, params).pl()
        return result

    def get_market(self, market_id: str) -> Market | None:
        df = self.query_markets(market_ids=[market_id], limit=1)
        if df.is_empty():
            return None
        return self._row_to_market(df.row(0, named=True))

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

    def _ensure_view(self) -> None:
        if self._view_created:
            return
        glob_pattern = self._get_parquet_glob(self._blocks_dir)
        self.conn.execute(f'''
            CREATE OR REPLACE VIEW blocks AS
            SELECT * FROM read_parquet('{glob_pattern}')
        ''')
        self._view_created = True

    def get_timestamp(self, block_number: int) -> datetime | None:
        """Get timestamp for a specific block number."""
        self._ensure_view()
        result = self.conn.execute(
            "SELECT timestamp FROM blocks WHERE block_number = ?",
            [block_number]
        ).fetchone()
        if result is None:
            return None
        ts_str = result[0]
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

    def get_timestamps_batch(self, block_numbers: list[int]) -> dict[int, datetime]:
        """Get timestamps for multiple block numbers efficiently."""
        if not block_numbers:
            return {}
        
        self._ensure_view()
        placeholders = ", ".join(["?" for _ in block_numbers])
        result = self.conn.execute(
            f"SELECT block_number, timestamp FROM blocks WHERE block_number IN ({placeholders})",
            block_numbers
        ).fetchall()
        
        timestamps = {}
        for bn, ts_str in result:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            timestamps[bn] = ts
        return timestamps


class TradeLoader(DataLoader):
    """Load and query Polymarket trade data."""

    def __init__(self, data_dir: str | Path, db_path: str | Path | None = None, block_loader: BlockLoader | None = None):
        super().__init__(data_dir, db_path)
        self._trades_dir = "trades"
        self.block_loader = block_loader or BlockLoader(data_dir, db_path)
        self._view_created = False

    def _ensure_view(self) -> None:
        if self._view_created:
            return
        glob_pattern = self._get_parquet_glob(self._trades_dir)
        self.conn.execute(f'''
            CREATE OR REPLACE VIEW trades AS
            SELECT * FROM read_parquet('{glob_pattern}')
        ''')
        self._view_created = True

    def query_trades(
        self,
        *,
        asset_ids: list[str] | None = None,
        min_block: int | None = None,
        max_block: int | None = None,
        limit: int | None = 10000,
    ) -> pl.DataFrame:
        """Query trades with filters."""
        self._ensure_view()

        conditions = []
        params = []

        if asset_ids is not None:
            placeholders = ", ".join(["?" for _ in asset_ids])
            conditions.append(f"(maker_asset_id IN ({placeholders}) OR taker_asset_id IN ({placeholders}))")
            params.extend(asset_ids)
            params.extend(asset_ids)

        if min_block is not None:
            conditions.append("block_number >= ?")
            params.append(min_block)

        if max_block is not None:
            conditions.append("block_number < ?")
            params.append(max_block)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        limit_clause = f"LIMIT {limit}" if limit else ""

        query = f"""
            SELECT * FROM trades
            WHERE {where_clause}
            ORDER BY block_number ASC
            {limit_clause}
        """

        result = self.conn.execute(query, params).pl()
        return result

    def get_trades_for_market(
        self,
        clob_token_ids: list[str],
        min_block: int | None = None,
        max_block: int | None = None,
        limit: int | None = 10000,
    ) -> pl.DataFrame:
        """Get trades for a market by its CLOB token IDs."""
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

        block_numbers = trades_df["block_number"].unique().to_list()
        timestamps = self.block_loader.get_timestamps_batch(block_numbers)
        
        return trades_df.with_columns(
            pl.col("block_number").map_elements(
                lambda bn: timestamps.get(bn),
                return_dtype=pl.Datetime("us", "UTC")
            ).alias("block_timestamp")
        )
