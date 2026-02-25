"""Category index for querying market classifications from DuckDB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import duckdb


CategoryType = Literal[
    "politics", "sports", "crypto", "entertainment",
    "weather", "finance", "science", "other"
]

SubcategoryType = Literal[
    "us-election", "us-congress", "international", "policy",
    "nba", "nfl", "mlb", "nhl", "soccer", "mma", "other",
    "bitcoin", "ethereum", "altcoins", "defi",
    "awards", "tv", "movies", "celebrities",
    "fed", "markets", "economics",
    "temperature", "storms", "climate",
    "space", "research", "tech", "ai",
    "misc",
]


@dataclass
class MarketCategory:
    """A categorized market record."""
    market_id: str
    category: str
    subcategory: str
    confidence: float = 1.0


class CategoryIndex:
    """Query interface for market_categories table in DuckDB."""

    def __init__(self, db_path: str | Path = "/root/combarbbot/polymarket.db"):
        self.db_path = Path(db_path)
        self._conn: duckdb.DuckDBPyConnection | None = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path), read_only=True)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "CategoryIndex":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def get_category(self, market_id: str) -> MarketCategory | None:
        result = self.conn.execute(
            "SELECT market_id, category, subcategory, confidence "
            "FROM market_categories WHERE market_id = ?",
            [market_id]
        ).fetchone()
        if result is None:
            return None
        return MarketCategory(market_id=result[0], category=result[1], 
                            subcategory=result[2], confidence=result[3] or 1.0)

    def get_categories_batch(self, market_ids: list[str]) -> dict[str, MarketCategory]:
        if not market_ids:
            return {}
        placeholders = ",".join("?" * len(market_ids))
        results = self.conn.execute(
            f"SELECT market_id, category, subcategory, confidence "
            f"FROM market_categories WHERE market_id IN ({placeholders})",
            market_ids
        ).fetchall()
        return {r[0]: MarketCategory(market_id=r[0], category=r[1], 
                subcategory=r[2], confidence=r[3] or 1.0) for r in results}

    def query_by_category(self, category: CategoryType, subcategory: SubcategoryType | None = None,
                         limit: int = 1000, offset: int = 0) -> list[str]:
        if subcategory:
            results = self.conn.execute(
                "SELECT market_id FROM market_categories WHERE category = ? AND subcategory = ? "
                "ORDER BY market_id LIMIT ? OFFSET ?",
                [category, subcategory, limit, offset]
            ).fetchall()
        else:
            results = self.conn.execute(
                "SELECT market_id FROM market_categories WHERE category = ? "
                "ORDER BY market_id LIMIT ? OFFSET ?",
                [category, limit, offset]
            ).fetchall()
        return [r[0] for r in results]

    def count_by_category(self) -> dict[str, int]:
        results = self.conn.execute(
            "SELECT category, COUNT(*) as cnt FROM market_categories GROUP BY category ORDER BY cnt DESC"
        ).fetchall()
        return {r[0]: r[1] for r in results}

    def total_categorized(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM market_categories").fetchone()[0]
