"""Category-aware market and trade loading from parquet files."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import duckdb

from .category_index import CategoryIndex, CategoryType, SubcategoryType


@dataclass
class Market:
    """Basic market information."""
    id: str
    question: str
    slug: str
    category: str | None = None
    subcategory: str | None = None
    volume: float = 0.0
    liquidity: float = 0.0
    outcomes: list[str] = field(default_factory=lambda: ["Yes", "No"])


@dataclass
class Trade:
    """A single trade record."""
    market_id: str
    timestamp: datetime
    price: float
    size: float
    side: str  # "buy" or "sell"
    outcome_index: int = 0


class CategoryAwareMarketLoader:
    """Load markets filtered by LLM-assigned category.

    Combines market parquet files with category index for filtered queries.
    """

    def __init__(
        self,
        markets_dir: str | Path,
        trades_dir: str | Path | None = None,
        db_path: str | Path | None = None,
    ):
        """Initialize loader with data paths.

        Args:
            markets_dir: Directory containing market parquet files
            trades_dir: Directory containing trade parquet files (optional)
            db_path: Path to DuckDB with market_categories table (optional)
        """
        self.markets_dir = Path(markets_dir)
        self.trades_dir = Path(trades_dir) if trades_dir else None
        self.db_path = Path(db_path) if db_path else None

        self._conn: duckdb.DuckDBPyConnection | None = None
        self._category_index: CategoryIndex | None = None

        # Cache parquet file lists
        self._market_files: list[str] | None = None
        self._trade_files: list[str] | None = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Lazy DuckDB connection for parquet queries."""
        if self._conn is None:
            self._conn = duckdb.connect()
        return self._conn

    @property
    def category_index(self) -> CategoryIndex | None:
        """Lazy category index connection."""
        if self._category_index is None and self.db_path:
            self._category_index = CategoryIndex(self.db_path)
        return self._category_index

    def close(self) -> None:
        """Close all connections."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._category_index is not None:
            self._category_index.close()
            self._category_index = None

    def __enter__(self) -> CategoryAwareMarketLoader:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @property
    def market_files(self) -> list[str]:
        """Get list of market parquet files."""
        if self._market_files is None:
            self._market_files = [
                str(f) for f in self.markets_dir.glob("*.parquet")
                if not f.name.startswith("._")
            ]
        return self._market_files

    @property
    def trade_files(self) -> list[str]:
        """Get list of trade parquet files."""
        if self._trade_files is None and self.trades_dir:
            self._trade_files = [
                str(f) for f in self.trades_dir.glob("*.parquet")
                if not f.name.startswith("._")
            ]
        return self._trade_files or []

    def query_by_category(
        self,
        category: CategoryType,
        subcategory: SubcategoryType | None = None,
        min_volume: float = 0,
        limit: int = 1000,
    ) -> list[Market]:
        """Get markets in a category with full details.

        Args:
            category: Main category (politics, sports, crypto, etc.)
            subcategory: Optional subcategory filter
            min_volume: Minimum volume filter
            limit: Maximum results

        Returns:
            List of Market objects with category info
        """
        if not self.category_index:
            raise ValueError("Category index not available - provide db_path")

        # Get market IDs from category index
        market_ids = self.category_index.query_by_category(
            category=category,
            subcategory=subcategory,
            limit=limit * 2,  # Get extra for volume filter
        )

        if not market_ids:
            return []

        # Join with market parquet for full details
        placeholders = ",".join(f"'{mid}'" for mid in market_ids)
        query = f"""
            SELECT
                m.id,
                m.question,
                m.slug,
                m.volume,
                m.liquidity
            FROM read_parquet({self.market_files}) m
            WHERE m.id IN ({placeholders})
            {"AND COALESCE(m.volume, 0) >= " + str(min_volume) if min_volume > 0 else ""}
            ORDER BY COALESCE(m.volume, 0) DESC
            LIMIT {limit}
        """

        results = self.conn.execute(query).fetchall()

        # Get categories for results
        result_ids = [r[0] for r in results]
        categories = self.category_index.get_categories_batch(result_ids)

        markets = []
        for r in results:
            cat = categories.get(r[0])
            markets.append(Market(
                id=r[0],
                question=r[1] or "",
                slug=r[2] or "",
                volume=float(r[3] or 0),
                liquidity=float(r[4] or 0),
                category=cat.category if cat else None,
                subcategory=cat.subcategory if cat else None,
            ))

        return markets

    def get_market(self, market_id: str) -> Market | None:
        """Get a single market by ID.

        Args:
            market_id: Market identifier

        Returns:
            Market object or None if not found
        """
        result = self.conn.execute(f"""
            SELECT id, question, slug, volume, liquidity
            FROM read_parquet({self.market_files})
            WHERE id = ?
            LIMIT 1
        """, [market_id]).fetchone()

        if result is None:
            return None

        cat = None
        if self.category_index:
            cat = self.category_index.get_category(market_id)

        return Market(
            id=result[0],
            question=result[1] or "",
            slug=result[2] or "",
            volume=float(result[3] or 0),
            liquidity=float(result[4] or 0),
            category=cat.category if cat else None,
            subcategory=cat.subcategory if cat else None,
        )

    def get_markets_batch(self, market_ids: list[str]) -> list[Market]:
        """Get multiple markets by ID.

        Args:
            market_ids: List of market identifiers

        Returns:
            List of Market objects (only found markets)
        """
        if not market_ids:
            return []

        placeholders = ",".join(f"'{mid}'" for mid in market_ids)
        results = self.conn.execute(f"""
            SELECT id, question, slug, volume, liquidity
            FROM read_parquet({self.market_files})
            WHERE id IN ({placeholders})
        """).fetchall()

        categories = {}
        if self.category_index:
            categories = self.category_index.get_categories_batch(market_ids)

        markets = []
        for r in results:
            cat = categories.get(r[0])
            markets.append(Market(
                id=r[0],
                question=r[1] or "",
                slug=r[2] or "",
                volume=float(r[3] or 0),
                liquidity=float(r[4] or 0),
                category=cat.category if cat else None,
                subcategory=cat.subcategory if cat else None,
            ))

        return markets

    def get_trades(
        self,
        market_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100000,
    ) -> list[Trade]:
        """Get trades for specific markets.

        Args:
            market_ids: Market IDs to get trades for
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum trades to return

        Returns:
            List of Trade objects sorted by timestamp
        """
        if not self.trade_files:
            raise ValueError("Trade directory not configured")

        if not market_ids:
            return []

        placeholders = ",".join(f"'{mid}'" for mid in market_ids)

        time_filter = ""
        if start_time:
            time_filter += f" AND timestamp >= '{start_time.isoformat()}'"
        if end_time:
            time_filter += f" AND timestamp <= '{end_time.isoformat()}'"

        query = f"""
            SELECT
                market_id,
                timestamp,
                price,
                size,
                side,
                outcome_index
            FROM read_parquet({self.trade_files})
            WHERE market_id IN ({placeholders})
            {time_filter}
            ORDER BY timestamp
            LIMIT {limit}
        """

        results = self.conn.execute(query).fetchall()

        return [
            Trade(
                market_id=r[0],
                timestamp=r[1],
                price=float(r[2]),
                size=float(r[3]),
                side=r[4] or "buy",
                outcome_index=int(r[5] or 0),
            )
            for r in results
        ]

    def stream_trades(
        self,
        market_ids: list[str],
        batch_size: int = 10000,
    ) -> Iterator[list[Trade]]:
        """Stream trades in batches for large datasets.

        Args:
            market_ids: Market IDs to get trades for
            batch_size: Number of trades per batch

        Yields:
            Batches of Trade objects
        """
        if not self.trade_files:
            raise ValueError("Trade directory not configured")

        if not market_ids:
            return

        placeholders = ",".join(f"'{mid}'" for mid in market_ids)

        query = f"""
            SELECT
                market_id,
                timestamp,
                price,
                size,
                side,
                outcome_index
            FROM read_parquet({self.trade_files})
            WHERE market_id IN ({placeholders})
            ORDER BY timestamp
        """

        # Use DuckDB's fetch_many for streaming
        result = self.conn.execute(query)

        while True:
            batch = result.fetchmany(batch_size)
            if not batch:
                break

            trades = [
                Trade(
                    market_id=r[0],
                    timestamp=r[1],
                    price=float(r[2]),
                    size=float(r[3]),
                    side=r[4] or "buy",
                    outcome_index=int(r[5] or 0),
                )
                for r in batch
            ]
            yield trades

    def get_all_market_ids(self) -> list[str]:
        """Get all market IDs from parquet files.

        Returns:
            List of all market IDs
        """
        results = self.conn.execute(f"""
            SELECT DISTINCT id FROM read_parquet({self.market_files})
        """).fetchall()
        return [r[0] for r in results]

    def count_markets(self) -> int:
        """Get total number of markets."""
        return self.conn.execute(f"""
            SELECT COUNT(*) FROM read_parquet({self.market_files})
        """).fetchone()[0]

    def category_summary(self) -> dict[str, dict]:
        """Get summary statistics by category.

        Returns:
            Dict with category stats including count, volume, etc.
        """
        if not self.category_index:
            raise ValueError("Category index not available")

        counts = self.category_index.count_by_category()
        total = self.category_index.total_categorized()

        return {
            "total_categorized": total,
            "by_category": counts,
        }
