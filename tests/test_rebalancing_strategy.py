"""Unit tests for the rebalancing arbitrage strategy."""

import pytest
from datetime import datetime, timedelta

from src.core.types import (
    DataRequirements,
    GroupingType,
    GroupSnapshot,
    MarketGroup,
    MarketMeta,
    MarketSnapshot,
    MarketTimeSeries,
    Opportunity,
    ExecutedTrade,
    PricePoint,
    StrategyConfig,
    TradeDirection,
    TradeLeg,
    TradeType,
)
from src.strategies.rebalancing_arb import RebalancingArbitrage


def _make_market(mid: str, question: str = "Test?") -> MarketMeta:
    return MarketMeta(
        id=mid, question=question, slug=mid, outcomes=["Yes", "No"]
    )


def _make_snapshot(
    group_id: str,
    prices: dict[str, float],
    ts: datetime | None = None,
) -> GroupSnapshot:
    ts = ts or datetime(2025, 1, 1)
    snapshots = {}
    for mid, price in prices.items():
        market = _make_market(mid)
        snapshots[mid] = MarketSnapshot(
            market=market,
            price_point=PricePoint(
                timestamp=ts,
                prices={"Yes": price, "No": 1.0 - price},
            ),
        )
    return GroupSnapshot(
        group_id=group_id,
        timestamp=ts,
        snapshots=snapshots,
    )


def _make_history(
    market: MarketMeta,
    prices: list[float],
    start: datetime | None = None,
) -> MarketTimeSeries:
    start = start or datetime(2024, 12, 1)
    points = [
        PricePoint(
            timestamp=start + timedelta(hours=i),
            prices={"Yes": p, "No": 1.0 - p},
        )
        for i, p in enumerate(prices)
    ]
    return MarketTimeSeries(market=market, points=points)


def _make_group(market_ids: list[str], group_id: str = "test_group") -> MarketGroup:
    return MarketGroup(
        group_id=group_id,
        name="Test Group",
        market_ids=market_ids,
        group_type=GroupingType.CORRELATION,
    )


class TestRebalancingStrategy:
    """Tests for RebalancingArbitrage strategy."""

    def test_init_default_config(self):
        strategy = RebalancingArbitrage()
        assert strategy.name == "rebalancing_arb"
        assert strategy.required_grouping == GroupingType.CORRELATION
        assert strategy.data_requirements.needs_time_series is True
        assert strategy.data_requirements.needs_volume is True

    def test_init_custom_config(self):
        config = StrategyConfig(
            strategy_name="rebalancing_arb",
            min_profit_threshold=0.02,
            extra={"deviation_threshold": 0.10},
        )
        strategy = RebalancingArbitrage(config=config)
        assert strategy.config.extra["deviation_threshold"] == 0.10

    def test_detect_no_history_returns_empty(self):
        strategy = RebalancingArbitrage()
        group = _make_group(["A", "B"])
        snapshot = _make_snapshot("test_group", {"A": 0.5, "B": 0.5})
        result = strategy.detect(group, snapshot, history=None)
        assert result == []

    def test_detect_no_deviation_returns_empty(self):
        strategy = RebalancingArbitrage()
        group = _make_group(["A", "B"])

        # Baseline and current prices are the same
        snapshot = _make_snapshot("test_group", {"A": 0.50, "B": 0.50})
        history = {
            "A": _make_history(_make_market("A"), [0.50] * 200),
            "B": _make_history(_make_market("B"), [0.50] * 200),
        }

        result = strategy.detect(group, snapshot, history=history)
        assert result == []

    def test_detect_significant_deviation(self):
        """When prices deviate significantly from baseline, detect opportunity."""
        strategy = RebalancingArbitrage(config=StrategyConfig(
            strategy_name="rebalancing_arb",
            extra={"deviation_threshold": 0.05, "lookback_hours": 168},
        ))
        group = _make_group(["A", "B"])

        # Baseline: A=0.50, B=0.50
        # Current: A=0.60, B=0.40 (10% deviation each)
        ts = datetime(2025, 1, 15)
        snapshot = _make_snapshot("test_group", {"A": 0.60, "B": 0.40}, ts=ts)
        history = {
            "A": _make_history(_make_market("A"), [0.50] * 200),
            "B": _make_history(_make_market("B"), [0.50] * 200),
        }

        result = strategy.detect(group, snapshot, history=history)
        assert len(result) == 1

        opp = result[0]
        assert opp.strategy_name == "rebalancing_arb"
        assert opp.trade_type == TradeType.HELD_POSITION
        assert opp.confidence > 0
        assert opp.hold_duration is not None
        assert len(opp.legs) >= 2

        # A moved up -> SELL, B moved down -> BUY
        leg_a = next(l for l in opp.legs if l.market_id == "A")
        leg_b = next(l for l in opp.legs if l.market_id == "B")
        assert leg_a.direction == TradeDirection.SELL
        assert leg_b.direction == TradeDirection.BUY

    def test_detect_insufficient_history(self):
        """With insufficient history, returns empty."""
        strategy = RebalancingArbitrage()
        group = _make_group(["A", "B"])
        snapshot = _make_snapshot("test_group", {"A": 0.60, "B": 0.40})
        history = {
            "A": _make_history(_make_market("A"), [0.50] * 5),  # Too short
            "B": _make_history(_make_market("B"), [0.50] * 5),
        }

        result = strategy.detect(group, snapshot, history=history)
        assert result == []

    def test_size_trades_passthrough(self):
        strategy = RebalancingArbitrage()
        opps = [Opportunity(
            strategy_name="rebalancing_arb",
            group_id="test",
            timestamp=datetime.now(),
            legs=[],
            expected_profit=0.1,
        )]
        assert strategy.size_trades(opps) == opps

    def test_validate_passthrough(self):
        strategy = RebalancingArbitrage()
        snapshot = _make_snapshot("test", {"A": 0.5})
        opps = [Opportunity(
            strategy_name="rebalancing_arb",
            group_id="test",
            timestamp=datetime.now(),
            legs=[],
            expected_profit=0.1,
        )]
        assert strategy.validate(opps, snapshot) == opps


class TestPositionManagement:
    """Tests for rebalancing position exit logic."""

    def test_time_stop_exit(self):
        """Positions held too long should trigger time stop."""
        strategy = RebalancingArbitrage(config=StrategyConfig(
            strategy_name="rebalancing_arb",
            extra={"max_holding_hours": 24},
        ))

        # Set up baseline
        strategy._baselines["test_group"] = {"A": 0.50, "B": 0.50}

        entry_time = datetime(2025, 1, 1, 0, 0)
        exit_time = datetime(2025, 1, 2, 1, 0)  # 25 hours later

        trade = ExecutedTrade(
            opportunity=Opportunity(
                strategy_name="rebalancing_arb",
                group_id="test_group",
                timestamp=entry_time,
                legs=[
                    TradeLeg(market_id="A", direction=TradeDirection.SELL, target_price=0.6),
                    TradeLeg(market_id="B", direction=TradeDirection.BUY, target_price=0.4),
                ],
                expected_profit=0.05,
                trade_type=TradeType.HELD_POSITION,
                metadata={"deviations": {"A": 0.10, "B": -0.10}},
            ),
            entry_time=entry_time,
            entry_prices={"A": 0.60, "B": 0.40},
        )

        snapshot = _make_snapshot("test_group", {"A": 0.55, "B": 0.45}, ts=exit_time)
        exits = strategy.update_positions([trade], snapshot)

        assert len(exits) == 1
        assert exits[0].metadata.get("exit_reason") == "time_stop"

    def test_take_profit_exit(self):
        """Positions that reverted should trigger take profit."""
        strategy = RebalancingArbitrage(config=StrategyConfig(
            strategy_name="rebalancing_arb",
            extra={"max_holding_hours": 48},
        ))

        strategy._baselines["test_group"] = {"A": 0.50, "B": 0.50}

        entry_time = datetime(2025, 1, 1, 0, 0)
        check_time = datetime(2025, 1, 1, 6, 0)  # 6 hours later

        trade = ExecutedTrade(
            opportunity=Opportunity(
                strategy_name="rebalancing_arb",
                group_id="test_group",
                timestamp=entry_time,
                legs=[
                    TradeLeg(market_id="A", direction=TradeDirection.SELL, target_price=0.6),
                    TradeLeg(market_id="B", direction=TradeDirection.BUY, target_price=0.4),
                ],
                expected_profit=0.05,
                trade_type=TradeType.HELD_POSITION,
                metadata={"deviations": {"A": 0.10, "B": -0.10}},
            ),
            entry_time=entry_time,
            entry_prices={"A": 0.60, "B": 0.40},
        )

        # Prices reverted close to baseline
        snapshot = _make_snapshot("test_group", {"A": 0.51, "B": 0.49}, ts=check_time)
        exits = strategy.update_positions([trade], snapshot)

        assert len(exits) == 1
        assert exits[0].metadata.get("exit_reason") == "take_profit"

    def test_update_positions_no_open_trades(self):
        strategy = RebalancingArbitrage()
        snapshot = _make_snapshot("test", {"A": 0.5})
        exits = strategy.update_positions([], snapshot)
        assert exits == []


class TestRebalancingRegistration:
    """Test strategy registration."""

    def test_registered(self):
        import src.strategies.rebalancing_arb  # noqa: F401
        from src.strategies.registry import list_strategies
        assert "rebalancing_arb" in list_strategies()

    def test_get_strategy(self):
        import src.strategies.rebalancing_arb  # noqa: F401
        from src.strategies.registry import get_strategy
        s = get_strategy("rebalancing_arb")
        assert s.name == "rebalancing_arb"

    def test_protocol_conformance(self):
        from src.core.protocols import ArbitrageStrategy
        s = RebalancingArbitrage()
        assert isinstance(s, ArbitrageStrategy)
