"""Partition arbitrage strategy: outcomes that should sum to 1.0 don't.

This is the simplest strategy: purely algebraic check.
No solver needed. No history needed.

Wraps the existing check_partition() logic from
src/backtest/constraint_checker.py as an ArbitrageStrategy.
"""

from __future__ import annotations

from src.core.types import (
    DataRequirements,
    GroupingType,
    GroupSnapshot,
    MarketGroup,
    MarketTimeSeries,
    Opportunity,
    ExecutedTrade,
    StrategyConfig,
    TradeDirection,
    TradeLeg,
    TradeType,
)
from src.strategies.registry import register_strategy


@register_strategy("partition_arb")
class PartitionArbitrage:
    """Partition arbitrage: outcomes that should sum to 1.0 don't.

    This is the simplest strategy: purely algebraic check.
    No solver needed. No history needed.
    """

    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig(
            strategy_name="partition_arb",
            min_profit_threshold=0.001,
            fee_per_leg=0.01,
        )

    @property
    def name(self) -> str:
        return "partition_arb"

    @property
    def required_grouping(self) -> GroupingType:
        return GroupingType.PARTITION

    @property
    def data_requirements(self) -> DataRequirements:
        return DataRequirements(
            needs_snapshots=True,
            needs_time_series=False,
            min_markets_per_group=3,
        )

    def detect(
        self,
        group: MarketGroup,
        snapshot: GroupSnapshot,
        history: dict[str, MarketTimeSeries] | None = None,
    ) -> list[Opportunity]:
        if not group.is_partition or group.size < 3:
            return []

        prices = snapshot.yes_prices
        total = sum(prices.values())

        violation = abs(total - 1.0)
        if violation < self.config.min_profit_threshold:
            return []

        # Determine direction
        if total < 1.0:
            direction = TradeDirection.BUY
            profit = 1.0 - total
        else:
            direction = TradeDirection.SELL
            profit = total - 1.0

        legs = [
            TradeLeg(
                market_id=mid,
                direction=direction,
                target_price=prices[mid],
            )
            for mid in group.market_ids
            if mid in prices
        ]

        return [Opportunity(
            strategy_name=self.name,
            group_id=group.group_id,
            timestamp=snapshot.timestamp,
            legs=legs,
            expected_profit=profit,
            confidence=1.0,  # Algebraically certain
            trade_type=TradeType.INSTANT_ARB,
            metadata={
                "partition_sum": total,
                "direction": "BUY_ALL" if total < 1.0 else "SELL_ALL",
                "violation": violation,
            },
        )]

    def size_trades(self, opportunities, portfolio_state=None):
        return opportunities  # Full Kelly for partition arb

    def validate(self, opportunities, snapshot):
        # Re-check that prices haven't moved
        valid = []
        for opp in opportunities:
            prices = snapshot.yes_prices
            total = sum(prices.get(mid, 0.5) for mid in opp.market_ids)
            if abs(total - 1.0) > self.config.min_profit_threshold:
                valid.append(opp)
        return valid

    def update_positions(self, open_trades, snapshot, history=None):
        return []  # Instant arb, no position management
