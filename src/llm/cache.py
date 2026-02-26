"""RelationshipGraph Cache (LLM-05).

DuckDB-backed cache for RelationshipGraph to avoid redundant LLM calls.
Cache key is a hash of sorted market IDs.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

from .schema import (
    MarketCluster,
    MarketRelationship,
    RelationshipGraph,
)

logger = logging.getLogger(__name__)

# Default cache TTL
DEFAULT_TTL_HOURS = 24


class RelationshipCache:
    """Cache RelationshipGraph by market set hash.
    
    Uses DuckDB for persistence. Cache key is a SHA256 hash of sorted
    market IDs, ensuring same markets always hit the same cache entry.
    """

    def __init__(self, db_path: str | Path = "polymarket.db"):
        """Initialize cache with DuckDB backend.
        
        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = Path(db_path)
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._init_table()
        logger.debug("[CACHE] RelationshipCache initialized: db=%s", db_path)

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Lazy connection getter."""
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
        return self._conn

    def _init_table(self) -> None:
        """Create cache table if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS relationship_cache (
                market_hash TEXT PRIMARY KEY,
                market_ids TEXT,
                graph_json TEXT,
                created_at TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)
        logger.debug("[CACHE] Table relationship_cache ensured")

    def _market_set_hash(self, market_ids: list[str]) -> str:
        """Compute stable hash of sorted market IDs.
        
        Args:
            market_ids: List of market IDs
            
        Returns:
            16-character hex hash
        """
        sorted_ids = ",".join(sorted(market_ids))
        return hashlib.sha256(sorted_ids.encode()).hexdigest()[:16]

    def get(self, market_ids: list[str]) -> RelationshipGraph | None:
        """Retrieve cached graph if exists and not stale.
        
        Args:
            market_ids: List of market IDs to look up
            
        Returns:
            Cached RelationshipGraph or None if not found/expired
        """
        if not market_ids:
            return None
            
        market_hash = self._market_set_hash(market_ids)
        now = datetime.now()
        
        result = self.conn.execute("""
            SELECT graph_json FROM relationship_cache
            WHERE market_hash = ?
            AND expires_at > ?
        """, [market_hash, now]).fetchone()
        
        if result is None:
            logger.debug("[CACHE] Miss for hash %s (%d markets)", market_hash, len(market_ids))
            return None
        
        try:
            graph_data = json.loads(result[0])
            graph = self._deserialize_graph(graph_data)
            logger.info("[CACHE] Hit for hash %s (%d markets)", market_hash, len(market_ids))
            return graph
        except Exception as e:
            logger.warning("[CACHE] Failed to deserialize cached graph: %s", e)
            # Remove corrupted entry
            self.conn.execute("DELETE FROM relationship_cache WHERE market_hash = ?", [market_hash])
            return None

    def set(
        self,
        market_ids: list[str],
        graph: RelationshipGraph,
        ttl_hours: int = DEFAULT_TTL_HOURS,
    ) -> None:
        """Store graph with expiry.
        
        Args:
            market_ids: List of market IDs (used as cache key)
            graph: RelationshipGraph to cache
            ttl_hours: Time-to-live in hours
        """
        if not market_ids:
            return
            
        market_hash = self._market_set_hash(market_ids)
        now = datetime.now()
        expires_at = now + timedelta(hours=ttl_hours)
        
        graph_json = json.dumps(self._serialize_graph(graph))
        market_ids_json = json.dumps(sorted(market_ids))
        
        # Upsert
        self.conn.execute("""
            INSERT INTO relationship_cache (market_hash, market_ids, graph_json, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (market_hash) DO UPDATE SET
                market_ids = excluded.market_ids,
                graph_json = excluded.graph_json,
                created_at = excluded.created_at,
                expires_at = excluded.expires_at
        """, [market_hash, market_ids_json, graph_json, now, expires_at])
        
        logger.info("[CACHE] Stored graph for hash %s (%d markets, expires in %dh)", 
                    market_hash, len(market_ids), ttl_hours)

    def invalidate(self, market_ids: list[str]) -> None:
        """Invalidate cache entry for given market IDs.
        
        Args:
            market_ids: List of market IDs to invalidate
        """
        if not market_ids:
            return
            
        market_hash = self._market_set_hash(market_ids)
        self.conn.execute("DELETE FROM relationship_cache WHERE market_hash = ?", [market_hash])
        logger.debug("[CACHE] Invalidated hash %s", market_hash)

    def clear_expired(self) -> int:
        """Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        now = datetime.now()
        result = self.conn.execute("""
            DELETE FROM relationship_cache WHERE expires_at <= ?
        """, [now])
        count = result.fetchone()[0] if result else 0
        logger.info("[CACHE] Cleared %d expired entries", count)
        return count

    def clear_all(self) -> None:
        """Clear entire cache."""
        self.conn.execute("DELETE FROM relationship_cache")
        logger.info("[CACHE] Cleared all entries")

    def stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dict with total_entries, expired_entries, total_size_kb
        """
        now = datetime.now()
        
        total = self.conn.execute("SELECT COUNT(*) FROM relationship_cache").fetchone()[0]
        expired = self.conn.execute(
            "SELECT COUNT(*) FROM relationship_cache WHERE expires_at <= ?", 
            [now]
        ).fetchone()[0]
        
        # Approximate size
        size_result = self.conn.execute("""
            SELECT SUM(LENGTH(graph_json)) FROM relationship_cache
        """).fetchone()[0] or 0
        
        return {
            "total_entries": total,
            "expired_entries": expired,
            "valid_entries": total - expired,
            "total_size_kb": size_result / 1024,
        }

    def _serialize_graph(self, graph: RelationshipGraph) -> dict:
        """Serialize RelationshipGraph to JSON-compatible dict."""
        return {
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "theme": c.theme,
                    "market_ids": c.market_ids,
                    "is_partition": c.is_partition,
                    "relationships": [
                        {
                            "type": r.type,
                            "from_market": r.from_market,
                            "to_market": r.to_market,
                            "confidence": r.confidence,
                            "reasoning": r.reasoning,
                        }
                        for r in c.relationships
                    ],
                }
                for c in graph.clusters
            ],
            "model_used": graph.model_used,
            "generated_at": graph.generated_at.isoformat(),
        }

    def _deserialize_graph(self, data: dict) -> RelationshipGraph:
        """Deserialize dict to RelationshipGraph."""
        clusters = []
        for c in data.get("clusters", []):
            relationships = [
                MarketRelationship(
                    type=r["type"],
                    from_market=r["from_market"],
                    to_market=r.get("to_market"),
                    confidence=r.get("confidence", 0.9),
                    reasoning=r.get("reasoning"),
                )
                for r in c.get("relationships", [])
            ]
            clusters.append(MarketCluster(
                cluster_id=c["cluster_id"],
                theme=c["theme"],
                market_ids=c["market_ids"],
                is_partition=c.get("is_partition", False),
                relationships=relationships,
            ))
        
        return RelationshipGraph(
            clusters=clusters,
            model_used=data.get("model_used"),
        )

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("[CACHE] Connection closed")


# Module-level cache instance for convenience
_default_cache: RelationshipCache | None = None


def get_cache(db_path: str | Path = "polymarket.db") -> RelationshipCache:
    """Get or create default cache instance.
    
    Args:
        db_path: Path to DuckDB database
        
    Returns:
        RelationshipCache instance
    """
    global _default_cache
    if _default_cache is None:
        _default_cache = RelationshipCache(db_path)
    return _default_cache
