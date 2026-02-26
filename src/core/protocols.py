"""Protocol definitions for the modular pipeline.

These define the contracts that data providers, groupers, strategies,
and backtest evaluators must implement.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterator, Protocol, runtime_checkable

from .types import (
    DataRequirements,
    ExecutedTrade,
    GroupingType,
    GroupSnapshot,
    MarketGroup,
    MarketMeta,
    MarketSnapshot,
    MarketTimeSeries,
    Opportunity,
    PricePoint,
    StrategyConfig,
)


# ─────────────────────────────────────────────────────────
# Data Source Protocol
# ─────────────────────────────────────────────────────────

@runtime_checkable
class MarketDataSource(Protocol):
    """Interface for loading market data from any source.

    Implementations:
        - ParquetMarketSource: current DuckDB/parquet files
        - APIMarketSource: live Polymarket CLOB API
        - MockMarketSource: synthetic data for testing
    """

    def get_market(self, market_id: str) -> MarketMeta | None:
        """Get metadata for a single market."""
        ...

    def get_markets(
        self,
        market_ids: list[str] | None = None,
        category: str | None = None,
        min_volume: float | None = None,
        limit: int | None = None,
    ) -> list[MarketMeta]:
        """Query markets with filters."""
        ...

    def get_snapshot(
        self,
        market_id: str,
        at_time: datetime | None = None,
    ) -> MarketSnapshot | None:
        """Get point-in-time market snapshot.

        If at_time is None, returns latest available.
        """
        ...

    def get_snapshots(
        self,
        market_ids: list[str],
        at_time: datetime | None = None,
    ) -> dict[str, MarketSnapshot]:
        """Get snapshots for multiple markets at the same time."""
        ...

    def get_time_series(
        self,
        market_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
        interval_minutes: int = 60,
    ) -> MarketTimeSeries:
        """Get historical price time series for a market.

        Args:
            market_id: Market identifier
            start: Start of time range
            end: End of time range
            interval_minutes: Aggregation interval (e.g., 60 = hourly bars)

        Returns:
            MarketTimeSeries with price points at the requested interval
        """
        ...

    def iter_snapshots(
        self,
        market_ids: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Iterator[GroupSnapshot]:
        """Iterate over synchronized snapshots across markets.

        Yields GroupSnapshot objects in chronological order, one per
        tick/interval where any market has activity.
        """
        ...

    def close(self) -> None:
        """Release resources."""
        ...


# ─────────────────────────────────────────────────────────
# Market Grouper Protocol
# ─────────────────────────────────────────────────────────

@runtime_checkable
class MarketGrouper(Protocol):
    """Interface for grouping markets into analyzable clusters.

    Different strategies need different grouping logic:
    - Partition arb needs exhaustive+exclusive groups
    - Statistical arb needs correlated price groups
    - Cross-market arb needs same-event groups across platforms
    """

    grouping_type: GroupingType

    def group(
        self,
        markets: list[MarketMeta],
        data_source: MarketDataSource | None = None,
    ) -> list[MarketGroup]:
        """Group markets according to this grouper's logic.

        Args:
            markets: List of market metadata to group
            data_source: Optional data source for price-based grouping

        Returns:
            List of MarketGroup objects
        """
        ...


# ─────────────────────────────────────────────────────────
# Strategy Protocol
# ─────────────────────────────────────────────────────────

@runtime_checkable
class ArbitrageStrategy(Protocol):
    """Core abstraction: a pluggable arbitrage strategy.

    Every strategy implements this protocol. The backtest engine
    calls these methods in order:
        1. detect() - find opportunities in current state
        2. size_trades() - determine position sizes
        3. validate() - pre-execution checks
        4. update_positions() - handle open positions (for held strategies)

    Strategies declare what data and grouping they need via properties.
    """

    @property
    def name(self) -> str:
        """Unique strategy identifier."""
        ...

    @property
    def required_grouping(self) -> GroupingType:
        """What kind of market grouping this strategy needs."""
        ...

    @property
    def data_requirements(self) -> DataRequirements:
        """What data this strategy requires at each evaluation."""
        ...

    def detect(
        self,
        group: MarketGroup,
        snapshot: GroupSnapshot,
        history: dict[str, MarketTimeSeries] | None = None,
    ) -> list[Opportunity]:
        """Detect arbitrage opportunities in a market group.

        Called once per group per evaluation time step.

        Args:
            group: The market group with constraints
            snapshot: Current prices for all markets in group
            history: Optional price history (if strategy needs it)

        Returns:
            List of detected opportunities (may be empty)
        """
        ...

    def size_trades(
        self,
        opportunities: list[Opportunity],
        portfolio_state: dict[str, Any] | None = None,
    ) -> list[Opportunity]:
        """Size the trades for detected opportunities.

        May filter out opportunities that are too small after sizing.

        Args:
            opportunities: Raw detected opportunities
            portfolio_state: Current portfolio (for position limits)

        Returns:
            Sized opportunities (subset of input)
        """
        ...

    def validate(
        self,
        opportunities: list[Opportunity],
        snapshot: GroupSnapshot,
    ) -> list[Opportunity]:
        """Pre-execution validation.

        Checks that opportunities are still valid given current prices.
        Filters out stale or invalid opportunities.
        """
        ...

    def update_positions(
        self,
        open_trades: list[ExecutedTrade],
        snapshot: GroupSnapshot,
        history: dict[str, MarketTimeSeries] | None = None,
    ) -> list[Opportunity]:
        """Manage open positions (for held-position strategies).

        Called each time step for strategies that hold positions.
        Returns new opportunities for position adjustments or exits.

        For instant-arb strategies, this is a no-op returning [].
        """
        ...


# ─────────────────────────────────────────────────────────
# Backtest Engine Protocol
# ─────────────────────────────────────────────────────────

@runtime_checkable
class BacktestEvaluator(Protocol):
    """Interface for different backtest evaluation modes.

    Implementations:
        - SinglePointEvaluator: for instant-arb (current behavior)
        - TimeSteppedSimulator: for held positions (rebalancing, stat arb)
    """

    def evaluate(
        self,
        strategy: ArbitrageStrategy,
        groups: list[MarketGroup],
        data_source: MarketDataSource,
        config: StrategyConfig,
    ) -> Any:
        """Run the backtest and return results."""
        ...


# ─────────────────────────────────────────────────────────
# Report Plugin Protocol
# ─────────────────────────────────────────────────────────

@runtime_checkable
class ReportPlugin(Protocol):
    """Interface for strategy-specific report components."""

    def generate_text(self, result: Any) -> str:
        """Generate strategy-specific text report section."""
        ...

    def generate_visualizations(
        self,
        result: Any,
        output_dir: str,
    ) -> list[str]:
        """Generate strategy-specific visualization files."""
        ...
