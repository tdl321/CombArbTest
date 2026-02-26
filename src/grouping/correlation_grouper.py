"""Correlation-based market grouper for rebalancing arbitrage.

Groups markets by price correlation using historical data from
MarketDataSource.get_time_series().
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np

from src.core.types import (
    GroupingType,
    MarketGroup,
    MarketMeta,
)
from src.core.protocols import MarketDataSource

logger = logging.getLogger(__name__)


class CorrelationGrouper:
    """Group markets by price correlation for rebalancing arbitrage.

    Markets are grouped if their price time series are sufficiently
    correlated, suggesting they should move together.
    """

    grouping_type = GroupingType.CORRELATION

    def __init__(
        self,
        min_correlation: float = 0.5,
        lookback_days: int = 30,
        min_group_size: int = 2,
        max_group_size: int = 10,
    ):
        self.min_correlation = min_correlation
        self.lookback_days = lookback_days
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size

    def group(
        self,
        markets: list[MarketMeta],
        data_source: MarketDataSource | None = None,
    ) -> list[MarketGroup]:
        if data_source is None:
            return []  # Need price data for correlation

        # Get time series for all markets
        end = datetime.now()
        start = end - timedelta(days=self.lookback_days)

        series = {}
        for m in markets:
            ts = data_source.get_time_series(
                m.id, start=start, end=end, interval_minutes=60
            )
            if ts and len(ts.points) > 24:  # Need at least 24 hours
                series[m.id] = ts.yes_prices

        if len(series) < 2:
            return []

        # Build correlation matrix
        market_ids = list(series.keys())
        n = len(market_ids)

        # Align time series to common length
        min_len = min(len(v) for v in series.values())
        if min_len < 2:
            return []

        matrix = np.array([series[mid][:min_len] for mid in market_ids])

        # Handle constant columns (zero variance)
        std = np.std(matrix, axis=1)
        valid_mask = std > 1e-8
        if sum(valid_mask) < 2:
            return []

        valid_ids = [mid for mid, v in zip(market_ids, valid_mask) if v]
        valid_matrix = matrix[valid_mask]

        corr = np.corrcoef(valid_matrix)

        # Cluster by correlation (simple single-linkage)
        groups = self._cluster_by_correlation(
            valid_ids, corr, self.min_correlation
        )

        # Convert to MarketGroup objects
        result = []
        for i, group_ids in enumerate(groups):
            if len(group_ids) < self.min_group_size:
                continue

            group_ids = group_ids[:self.max_group_size]

            # Compute average intra-cluster correlation
            indices = [valid_ids.index(mid) for mid in group_ids]
            avg_corr = float(np.mean([
                corr[a][b]
                for a in indices for b in indices if a != b
            ])) if len(indices) > 1 else 0.0

            result.append(MarketGroup(
                group_id="corr_group_" + str(i),
                name="Correlation cluster #" + str(i) + " (" + str(len(group_ids)) + " markets)",
                market_ids=group_ids,
                group_type=GroupingType.CORRELATION,
                constraints=[],  # No logical constraints, just correlation
                metadata={
                    "avg_correlation": avg_corr,
                },
            ))

        logger.info(
            "[GROUPER] Correlation grouping: %d markets -> %d groups",
            len(markets), len(result),
        )
        return result

    def _cluster_by_correlation(
        self,
        market_ids: list[str],
        corr_matrix: np.ndarray,
        threshold: float,
    ) -> list[list[str]]:
        """Simple single-linkage clustering by correlation."""
        n = len(market_ids)
        visited = set()
        clusters = []

        for i in range(n):
            if i in visited:
                continue

            cluster = [market_ids[i]]
            visited.add(i)
            queue = [i]

            while queue:
                current = queue.pop(0)
                for j in range(n):
                    if j not in visited and corr_matrix[current][j] >= threshold:
                        cluster.append(market_ids[j])
                        visited.add(j)
                        queue.append(j)

            clusters.append(cluster)

        return clusters
