"""Refactored backtest engine supporting both instant and held strategies.

The engine dispatches to the appropriate evaluator based on whether
the strategy produces INSTANT_ARB or HELD_POSITION trade types.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Iterator

from pydantic import BaseModel, Field

from src.core.types import (
    DataRequirements,
    ExecutedTrade,
    GroupSnapshot,
    MarketGroup,
    MarketTimeSeries,
    Opportunity,
    StrategyConfig,
    TradeDirection,
    TradeType,
)
from src.core.protocols import ArbitrageStrategy, MarketDataSource

logger = logging.getLogger(__name__)


class BacktestResult(BaseModel):
    """Universal backtest result container."""
    strategy_name: str
    start_date: datetime
    end_date: datetime

    # Opportunities
    total_opportunities: int
    opportunities: list[Opportunity] = Field(default_factory=list)

    # Trades
    total_trades: int = 0
    executed_trades: list[ExecutedTrade] = Field(default_factory=list)

    # P&L
    gross_pnl: float = 0.0
    total_fees: float = 0.0
    net_pnl: float = 0.0

    # Risk
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0

    # Equity curve (for time-stepped strategies)
    equity_curve: list[tuple[datetime, float]] = Field(default_factory=list)


def get_evaluator(strategy: ArbitrageStrategy):
    """Select the appropriate evaluator for a strategy.

    Instant-arb strategies get SinglePointEvaluator.
    Held-position strategies get TimeSteppedSimulator.
    """
    reqs = strategy.data_requirements
    if reqs.needs_time_series:
        return TimeSteppedSimulator()
    return SinglePointEvaluator()


class SinglePointEvaluator:
    """Evaluator for instant-arbitrage strategies.

    This is equivalent to the current WalkForwardSimulator behavior:
    iterate through snapshots, detect opportunities, compute instant P&L.
    """

    def evaluate(
        self,
        strategy: ArbitrageStrategy,
        groups: list[MarketGroup],
        data_source: MarketDataSource,
        config: StrategyConfig | None = None,
    ) -> BacktestResult:
        config = config or StrategyConfig(strategy_name=strategy.name)
        all_opportunities = []
        all_trades = []
        cumulative_pnl = 0.0
        peak_pnl = 0.0
        max_drawdown = 0.0
        equity_curve = []
        wins = 0
        losses = 0

        for group in groups:
            market_ids = group.market_ids
            logger.info(
                "[ENGINE] Evaluating group %s (%d markets) with strategy %s",
                group.group_id, len(market_ids), strategy.name,
            )

            for group_snapshot in data_source.iter_snapshots(market_ids):
                # Detect
                opportunities = strategy.detect(group, group_snapshot)
                if not opportunities:
                    continue

                # Size
                sized = strategy.size_trades(opportunities)

                # Validate
                validated = strategy.validate(sized, group_snapshot)

                for opp in validated:
                    all_opportunities.append(opp)

                    # Compute instant P&L
                    fee_total = opp.num_legs * config.fee_per_leg
                    net = opp.expected_profit - fee_total

                    trade = ExecutedTrade(
                        opportunity=opp,
                        entry_time=opp.timestamp,
                        exit_time=opp.timestamp,  # Instant
                        entry_prices=group_snapshot.yes_prices,
                        exit_prices=group_snapshot.yes_prices,
                        fees_paid=fee_total,
                        realized_pnl=opp.expected_profit,
                    )
                    all_trades.append(trade)

                    cumulative_pnl += net
                    peak_pnl = max(peak_pnl, cumulative_pnl)
                    max_drawdown = max(max_drawdown, peak_pnl - cumulative_pnl)
                    equity_curve.append((opp.timestamp, cumulative_pnl))

                    if net > 0:
                        wins += 1
                    else:
                        losses += 1

        total_trades = len(all_trades)
        gross = sum(t.realized_pnl or 0 for t in all_trades)
        fees = sum(t.fees_paid for t in all_trades)

        return BacktestResult(
            strategy_name=strategy.name,
            start_date=equity_curve[0][0] if equity_curve else datetime.now(),
            end_date=equity_curve[-1][0] if equity_curve else datetime.now(),
            total_opportunities=len(all_opportunities),
            opportunities=all_opportunities,
            total_trades=total_trades,
            executed_trades=all_trades,
            gross_pnl=gross,
            total_fees=fees,
            net_pnl=gross - fees,
            max_drawdown=max_drawdown,
            win_rate=wins / max(1, wins + losses),
            equity_curve=equity_curve,
        )


class TimeSteppedSimulator:
    """Evaluator for held-position strategies (rebalancing, stat arb).

    Steps through time at fixed intervals, manages open positions,
    computes mark-to-market P&L at each step.
    """

    def __init__(self, step_minutes: int = 60):
        self.step_minutes = step_minutes

    def evaluate(
        self,
        strategy: ArbitrageStrategy,
        groups: list[MarketGroup],
        data_source: MarketDataSource,
        config: StrategyConfig | None = None,
    ) -> BacktestResult:
        config = config or StrategyConfig(strategy_name=strategy.name)

        # Position tracker
        open_trades: list[ExecutedTrade] = []
        closed_trades: list[ExecutedTrade] = []
        all_opportunities: list[Opportunity] = []

        cumulative_pnl = 0.0
        peak_pnl = 0.0
        max_drawdown = 0.0
        equity_curve: list[tuple[datetime, float]] = []
        wins = 0
        losses = 0

        for group in groups:
            market_ids = group.market_ids
            lookback = strategy.data_requirements.lookback_periods

            logger.info(
                "[ENGINE] Time-stepping group %s (%d markets) with strategy %s",
                group.group_id, len(market_ids), strategy.name,
            )

            for group_snapshot in data_source.iter_snapshots(market_ids):
                ts = group_snapshot.timestamp

                # Load history window for strategies that need it
                history = None
                if strategy.data_requirements.needs_time_series:
                    history = {}
                    for mid in market_ids:
                        series = data_source.get_time_series(
                            mid,
                            start=ts - timedelta(hours=lookback),
                            end=ts,
                            interval_minutes=self.step_minutes,
                        )
                        if series:
                            history[mid] = series

                # Step 1: Update existing positions (mark-to-market)
                for trade in open_trades:
                    self._mark_to_market(trade, group_snapshot)

                # Step 2: Check for exit signals on open positions
                exit_opps = strategy.update_positions(
                    open_trades, group_snapshot, history
                )

                for exit_opp in exit_opps:
                    matching = self._find_matching_trade(exit_opp, open_trades)
                    if matching:
                        self._close_trade(matching, group_snapshot)
                        open_trades.remove(matching)
                        closed_trades.append(matching)

                        net = matching.total_pnl
                        cumulative_pnl += net
                        if net > 0:
                            wins += 1
                        else:
                            losses += 1

                # Step 3: Detect new opportunities
                new_opps = strategy.detect(group, group_snapshot, history)
                sized = strategy.size_trades(new_opps)
                validated = strategy.validate(sized, group_snapshot)

                for opp in validated:
                    all_opportunities.append(opp)

                    if opp.trade_type == TradeType.INSTANT_ARB:
                        fee = opp.num_legs * config.fee_per_leg
                        trade = ExecutedTrade(
                            opportunity=opp,
                            entry_time=ts,
                            exit_time=ts,
                            entry_prices=group_snapshot.yes_prices,
                            exit_prices=group_snapshot.yes_prices,
                            fees_paid=fee,
                            realized_pnl=opp.expected_profit,
                        )
                        closed_trades.append(trade)
                        cumulative_pnl += opp.expected_profit - fee
                        wins += 1
                    else:
                        fee = opp.num_legs * config.fee_per_leg
                        trade = ExecutedTrade(
                            opportunity=opp,
                            entry_time=ts,
                            entry_prices=dict(group_snapshot.yes_prices),
                            fees_paid=fee,
                        )
                        open_trades.append(trade)

                # Track equity
                unrealized = sum(t.unrealized_pnl for t in open_trades)
                total_equity = cumulative_pnl + unrealized
                peak_pnl = max(peak_pnl, total_equity)
                max_drawdown = max(max_drawdown, peak_pnl - total_equity)
                equity_curve.append((ts, total_equity))

        # Force-close remaining open positions at last known prices
        for trade in open_trades:
            trade.exit_time = equity_curve[-1][0] if equity_curve else datetime.now()
            trade.realized_pnl = trade.unrealized_pnl
            closed_trades.append(trade)
            cumulative_pnl += trade.total_pnl

        all_trades = closed_trades
        gross = sum(t.realized_pnl or 0 for t in all_trades)
        fees = sum(t.fees_paid for t in all_trades)

        return BacktestResult(
            strategy_name=strategy.name,
            start_date=equity_curve[0][0] if equity_curve else datetime.now(),
            end_date=equity_curve[-1][0] if equity_curve else datetime.now(),
            total_opportunities=len(all_opportunities),
            opportunities=all_opportunities,
            total_trades=len(all_trades),
            executed_trades=all_trades,
            gross_pnl=gross,
            total_fees=fees,
            net_pnl=gross - fees,
            max_drawdown=max_drawdown,
            win_rate=wins / max(1, wins + losses),
            equity_curve=equity_curve,
        )

    def _mark_to_market(
        self,
        trade: ExecutedTrade,
        snapshot: GroupSnapshot,
    ) -> None:
        """Update unrealized P&L based on current prices."""
        if not trade.is_open:
            return

        unrealized = 0.0
        for leg in trade.opportunity.legs:
            entry_price = trade.entry_prices.get(leg.market_id, 0.5)
            current_price = snapshot.yes_prices.get(leg.market_id, entry_price)

            if leg.direction == TradeDirection.BUY:
                unrealized += (current_price - entry_price) * leg.size
            elif leg.direction == TradeDirection.SELL:
                unrealized += (entry_price - current_price) * leg.size

        trade.unrealized_pnl = unrealized

    def _close_trade(
        self,
        trade: ExecutedTrade,
        snapshot: GroupSnapshot,
    ) -> None:
        """Close a position and compute realized P&L."""
        trade.exit_time = snapshot.timestamp
        trade.exit_prices = dict(snapshot.yes_prices)
        trade.realized_pnl = trade.unrealized_pnl

    def _find_matching_trade(
        self,
        exit_opp: Opportunity,
        open_trades: list[ExecutedTrade],
    ) -> ExecutedTrade | None:
        """Find the open trade that an exit opportunity refers to."""
        for trade in open_trades:
            if trade.opportunity.group_id == exit_opp.group_id:
                return trade
        return None
