"""Combinatorial arbitrage strategy: Frank-Wolfe optimization over marginal polytope.

Uses the existing optimizer module (src/optimizer/) as a tool.
Wraps find_marginal_arbitrage() as an ArbitrageStrategy.
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


@register_strategy("combinatorial_arb")
class CombinatorialArbitrage:
    """Frank-Wolfe optimization over the marginal polytope.

    Uses the existing optimizer module as a tool.
    """

    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig(
            strategy_name="combinatorial_arb",
            min_profit_threshold=0.001,
            fee_per_leg=0.01,
            extra={
                "max_iterations": 100,
                "tolerance": 1e-4,
                "step_mode": "adaptive",
                "kl_threshold": 0.01,
            },
        )

    @property
    def name(self) -> str:
        return "combinatorial_arb"

    @property
    def required_grouping(self) -> GroupingType:
        return GroupingType.SEMANTIC

    @property
    def data_requirements(self) -> DataRequirements:
        return DataRequirements(
            needs_snapshots=True,
            needs_time_series=False,
            min_markets_per_group=2,
        )

    def detect(self, group, snapshot, history=None):
        from src.optimizer.frank_wolfe import find_marginal_arbitrage
        from src.optimizer.schema import (
            OptimizationConfig, RelationshipGraph, MarketCluster,
            MarketRelationship,
        )

        prices = snapshot.yes_prices
        list_prices = {mid: [p, 1.0 - p] for mid, p in prices.items()}

        # Convert MarketGroup constraints to RelationshipGraph
        relationships = [
            MarketRelationship(
                type=c.type.value,
                from_market=c.from_market,
                to_market=c.to_market,
                confidence=c.confidence,
            )
            for c in group.constraints
        ]

        graph = RelationshipGraph(clusters=[
            MarketCluster(
                cluster_id=group.group_id,
                theme=group.name,
                market_ids=group.market_ids,
                is_partition=group.is_partition,
                relationships=relationships,
            )
        ])

        opt_config = OptimizationConfig(
            max_iterations=self.config.extra.get("max_iterations", 100),
            tolerance=self.config.extra.get("tolerance", 1e-4),
            step_mode=self.config.extra.get("step_mode", "adaptive"),
        )

        result = find_marginal_arbitrage(
            market_prices=list_prices,
            relationships=graph,
            config=opt_config,
        )

        kl_threshold = self.config.extra.get("kl_threshold", 0.01)
        if result.kl_divergence < kl_threshold:
            return []

        # Convert solver result to Opportunity
        coherent = {mid: ps[0] for mid, ps in result.coherent_market_prices.items()}
        legs = []
        for mid in group.market_ids:
            if mid in prices and mid in coherent:
                diff = coherent[mid] - prices[mid]
                if abs(diff) > 0.005:
                    direction = TradeDirection.BUY if diff > 0 else TradeDirection.SELL
                    legs.append(TradeLeg(
                        market_id=mid,
                        direction=direction,
                        target_price=prices[mid],
                    ))

        if not legs:
            return []

        return [Opportunity(
            strategy_name=self.name,
            group_id=group.group_id,
            timestamp=snapshot.timestamp,
            legs=legs,
            expected_profit=result.kl_divergence,
            confidence=min(1.0, 1.0 - result.duality_gap / max(result.kl_divergence, 1e-10)),
            trade_type=TradeType.INSTANT_ARB,
            metadata={
                "kl_divergence": result.kl_divergence,
                "duality_gap": result.duality_gap,
                "converged": result.converged,
                "iterations": result.iterations,
                "coherent_prices": coherent,
            },
        )]

    def size_trades(self, opportunities, portfolio_state=None):
        return opportunities

    def validate(self, opportunities, snapshot):
        return opportunities  # Solver already validated

    def update_positions(self, open_trades, snapshot, history=None):
        return []
