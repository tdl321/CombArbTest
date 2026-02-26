"""Rebalancing arbitrage strategy: capture temporary mispricings from price drift.

Exploits temporary mispricings between related markets caused by price movements.
When market A moves, related market B may lag, creating a window where
buying/selling B captures the spread.

Unlike instant arbitrage, this requires:
1. Historical context: a baseline "fair" relationship between markets
2. Drift detection: identifying when current prices deviate from the baseline
3. Position management: entering, holding, and exiting positions over time
4. Mean-reversion assumption: prices will revert to the fair relationship
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from src.core.types import (
    DataRequirements,
    ExecutedTrade,
    GroupingType,
    GroupSnapshot,
    MarketGroup,
    MarketTimeSeries,
    Opportunity,
    StrategyConfig,
    TradeDirection,
    TradeLeg,
    TradeType,
)
from src.strategies.registry import register_strategy

logger = logging.getLogger(__name__)


@dataclass
class RebalanceSignal:
    """Internal signal for a rebalancing opportunity."""
    group_id: str
    timestamp: datetime
    baseline_prices: dict[str, float]
    current_prices: dict[str, float]
    deviations: dict[str, float]
    trigger_market: str
    mean_reversion_score: float
    expected_holding_hours: float


@register_strategy("rebalancing_arb")
class RebalancingArbitrage:
    """Rebalancing arbitrage: capture temporary mispricings from price drift.

    Configuration (in StrategyConfig.extra):
        lookback_hours: int = 168        # 7 days for baseline
        deviation_threshold: float = 0.05 # 5% deviation to trigger
        max_holding_hours: float = 48    # Max hold before forced exit
        reversion_alpha: float = 0.3     # EMA alpha for reversion speed
        min_volume_ratio: float = 0.5    # Min volume vs avg to confirm
    """

    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig(
            strategy_name="rebalancing_arb",
            min_profit_threshold=0.01,
            fee_per_leg=0.01,
            extra={
                "lookback_hours": 168,
                "deviation_threshold": 0.05,
                "max_holding_hours": 48,
                "reversion_alpha": 0.3,
                "min_volume_ratio": 0.5,
            },
        )
        # Track per-group baselines
        self._baselines: dict[str, dict[str, float]] = {}
        self._baseline_ages: dict[str, datetime] = {}

    @property
    def name(self) -> str:
        return "rebalancing_arb"

    @property
    def required_grouping(self) -> GroupingType:
        return GroupingType.CORRELATION

    @property
    def data_requirements(self) -> DataRequirements:
        return DataRequirements(
            needs_snapshots=True,
            needs_time_series=True,
            lookback_periods=self.config.extra.get("lookback_hours", 168),
            needs_volume=True,
            min_markets_per_group=2,
        )

    def detect(
        self,
        group: MarketGroup,
        snapshot: GroupSnapshot,
        history: dict[str, MarketTimeSeries] | None = None,
    ) -> list[Opportunity]:
        if history is None:
            return []

        threshold = self.config.extra.get("deviation_threshold", 0.05)
        current_prices = snapshot.yes_prices

        # Step 1: Compute or update baseline
        baseline = self._compute_baseline(group, history)
        if baseline is None:
            return []

        # Step 2: Compute deviations
        deviations = {}
        for mid in group.market_ids:
            if mid in current_prices and mid in baseline:
                deviations[mid] = current_prices[mid] - baseline[mid]

        if not deviations:
            return []

        # Step 3: Check if deviation is significant
        max_dev = max(abs(d) for d in deviations.values())
        if max_dev < threshold:
            return []

        # Step 4: Identify trigger market (largest absolute move)
        trigger = max(deviations, key=lambda m: abs(deviations[m]))

        # Step 5: Compute mean-reversion score from historical behavior
        reversion_score = self._compute_reversion_score(
            group, history, deviations
        )

        # Step 6: Build trades
        #   - Markets that moved UP relative to baseline: SELL (overweight)
        #   - Markets that moved DOWN relative to baseline: BUY (underweight)
        legs = []
        for mid, dev in deviations.items():
            if abs(dev) > threshold * 0.5:
                direction = TradeDirection.SELL if dev > 0 else TradeDirection.BUY
                legs.append(TradeLeg(
                    market_id=mid,
                    direction=direction,
                    target_price=current_prices[mid],
                    size=abs(dev),
                ))

        if len(legs) < 2:
            return []

        expected_holding = self.config.extra.get("max_holding_hours", 48) * (1 - reversion_score)

        return [Opportunity(
            strategy_name=self.name,
            group_id=group.group_id,
            timestamp=snapshot.timestamp,
            legs=legs,
            expected_profit=max_dev * reversion_score,
            confidence=reversion_score,
            trade_type=TradeType.HELD_POSITION,
            hold_duration=expected_holding,
            metadata={
                "baseline_prices": baseline,
                "deviations": deviations,
                "trigger_market": trigger,
                "max_deviation": max_dev,
                "reversion_score": reversion_score,
            },
        )]

    def _compute_baseline(
        self,
        group: MarketGroup,
        history: dict[str, MarketTimeSeries],
    ) -> dict[str, float] | None:
        """Compute rolling average prices as baseline."""
        lookback = self.config.extra.get("lookback_hours", 168)
        baseline = {}

        for mid in group.market_ids:
            ts = history.get(mid)
            if ts is None or len(ts.points) < lookback // 4:
                return None  # Insufficient data

            # Use last N price points for rolling average
            recent = ts.latest(lookback)
            prices = [
                p.prices.get(ts.market.outcomes[0], 0.5) for p in recent
            ]
            baseline[mid] = sum(prices) / len(prices)

        self._baselines[group.group_id] = baseline
        return baseline

    def _compute_reversion_score(
        self,
        group: MarketGroup,
        history: dict[str, MarketTimeSeries],
        current_deviations: dict[str, float],
    ) -> float:
        """Estimate probability of mean reversion.

        Uses deviation magnitude as a heuristic proxy.
        Large deviations are MORE likely to revert (overreaction)
        but also more risky, so cap the score.
        """
        max_dev = max(abs(d) for d in current_deviations.values())

        if max_dev > 0.20:
            return 0.8  # Very large deviation, likely reverts
        elif max_dev > 0.10:
            return 0.6
        elif max_dev > 0.05:
            return 0.4
        else:
            return 0.2  # Small deviation, might persist

    def update_positions(
        self,
        open_trades: list[ExecutedTrade],
        snapshot: GroupSnapshot,
        history: dict[str, MarketTimeSeries] | None = None,
    ) -> list[Opportunity]:
        """Manage open rebalancing positions.

        Exit conditions:
        1. Prices reverted to baseline (take profit)
        2. Max holding time exceeded (time stop)
        3. Deviation increased (stop loss)
        """
        max_hold = self.config.extra.get("max_holding_hours", 48)
        exit_signals = []

        for trade in open_trades:
            if not trade.is_open:
                continue

            group_id = trade.opportunity.group_id
            baseline = self._baselines.get(group_id)
            if baseline is None:
                continue

            # Check time stop
            hold_hours = (snapshot.timestamp - trade.entry_time).total_seconds() / 3600
            if hold_hours >= max_hold:
                exit_signals.append(self._create_exit_opportunity(
                    trade, snapshot, reason="time_stop"
                ))
                continue

            # Check profit target (deviation reverted)
            current_prices = snapshot.yes_prices
            current_devs = {
                mid: current_prices.get(mid, 0.5) - baseline.get(mid, 0.5)
                for mid in trade.opportunity.market_ids
            }
            entry_devs = trade.opportunity.metadata.get("deviations", {})

            # If deviations have reversed sign or shrunk significantly
            reversion_pct = self._compute_reversion_pct(entry_devs, current_devs)
            if reversion_pct > 0.7:
                exit_signals.append(self._create_exit_opportunity(
                    trade, snapshot, reason="take_profit"
                ))
                continue

            # Check stop loss (deviation worsened by 50%+)
            if reversion_pct < -0.5:
                exit_signals.append(self._create_exit_opportunity(
                    trade, snapshot, reason="stop_loss"
                ))

        return exit_signals

    def _compute_reversion_pct(
        self,
        entry_devs: dict[str, float],
        current_devs: dict[str, float],
    ) -> float:
        """Compute what fraction of the original deviation has reverted.

        Returns:
            1.0 = fully reverted, 0.0 = unchanged, negative = worsened
        """
        if not entry_devs:
            return 0.0

        total_entry = sum(abs(d) for d in entry_devs.values())
        if total_entry < 1e-6:
            return 0.0

        total_current = sum(abs(current_devs.get(m, 0.0)) for m in entry_devs)
        return 1.0 - (total_current / total_entry)

    def _create_exit_opportunity(
        self,
        trade: ExecutedTrade,
        snapshot: GroupSnapshot,
        reason: str,
    ) -> Opportunity:
        """Create an exit opportunity (reverse all legs)."""
        reverse_legs = []
        for leg in trade.opportunity.legs:
            reverse_direction = (
                TradeDirection.SELL if leg.direction == TradeDirection.BUY
                else TradeDirection.BUY
            )
            reverse_legs.append(TradeLeg(
                market_id=leg.market_id,
                direction=reverse_direction,
                target_price=snapshot.yes_prices.get(leg.market_id, 0.5),
                size=leg.size,
            ))

        return Opportunity(
            strategy_name=self.name,
            group_id=trade.opportunity.group_id,
            timestamp=snapshot.timestamp,
            legs=reverse_legs,
            expected_profit=trade.unrealized_pnl,
            confidence=1.0,
            trade_type=TradeType.REBALANCE,
            metadata={"exit_reason": reason},
        )

    def size_trades(self, opportunities, portfolio_state=None):
        return opportunities

    def validate(self, opportunities, snapshot):
        return opportunities
