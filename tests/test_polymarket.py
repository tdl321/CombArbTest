"""Tests for the Polymarket live data pipeline.

Covers:
- API type parsing (PolymarketMarket, PolymarketEvent)
- GammaClient with mocked HTTP
- ClobClient with mocked HTTP
- Mapping to core types
- RelationshipInferrer (negRisk auto-inference)
- DatasetBuilder integration
- LiveDataset.to_solver_input() format
- LiveDataset.run_solver() executes
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.data.polymarket.types import (
    PolymarketEvent,
    PolymarketMarket,
    PolymarketOrderBook,
    PolymarketPriceHistory,
    _parse_json_or_list,
)
from src.data.polymarket.config import PolymarketConfig
from src.data.polymarket.gamma_client import GammaClient
from src.data.polymarket.clob_client import ClobClient
from src.data.polymarket.mapping import (
    to_market_meta,
    to_price_point,
    to_market_snapshot,
    to_market_time_series,
    to_group_snapshot,
)
from src.data.polymarket.relationship_inference import RelationshipInferrer
from src.data.polymarket.dataset import DatasetSpec, DatasetBuilder, LiveDataset
from src.core.types import (
    ConstraintType,
    GroupingType,
    MarketGroup,
    MarketStatus,
)
from src.optimizer.schema import (
    MarketCluster,
    MarketRelationship,
    RelationshipGraph,
)


# ═══════════════════════════════════════════════════════════
# Fixtures: mock API responses
# ═══════════════════════════════════════════════════════════

def _make_market_response(
    condition_id: str = "0xabc123",
    question: str = "Will Max Verstappen win?",
    slug: str = "will-max-verstappen-win",
    outcomes: list[str] | None = None,
    outcome_prices: list[float] | None = None,
    clob_token_ids: list[str] | None = None,
    volume: float = 100000.0,
    liquidity: float = 50000.0,
    active: bool = True,
    closed: bool = False,
    neg_risk: bool = False,
    group_item_title: str | None = None,
) -> dict:
    """Build a mock Gamma API market response."""
    return {
        "conditionId": condition_id,
        "question": question,
        "slug": slug,
        "outcomes": json.dumps(outcomes or ["Yes", "No"]),
        "outcomePrices": json.dumps(outcome_prices or [0.65, 0.35]),
        "clobTokenIds": json.dumps(
            clob_token_ids or ["token_yes_" + condition_id, "token_no_" + condition_id]
        ),
        "volume": str(volume),
        "liquidity": str(liquidity),
        "active": active,
        "closed": closed,
        "acceptingOrders": True,
        "description": "Test market",
        "negRisk": neg_risk,
        "groupItemTitle": group_item_title,
        "category": "sports",
    }


def _make_event_response(
    event_id: str = "evt_001",
    title: str = "F1 Drivers Championship 2026",
    slug: str = "f1-drivers-championship-2026",
    neg_risk: bool = True,
    num_markets: int = 3,
) -> dict:
    """Build a mock Gamma API event response."""
    drivers = [
        ("Max Verstappen", 0.35),
        ("Lewis Hamilton", 0.12),
        ("Lando Norris", 0.18),
        ("Charles Leclerc", 0.15),
        ("Oscar Piastri", 0.10),
    ]
    markets = []
    for i in range(min(num_markets, len(drivers))):
        name, price = drivers[i]
        markets.append(_make_market_response(
            condition_id=f"cond_{i:03d}",
            question=f"Will {name} win the F1 Championship?",
            slug=f"will-{name.lower().replace(' ', '-')}-win",
            outcome_prices=[price, 1.0 - price],
            neg_risk=neg_risk,
            group_item_title=name,
        ))

    return {
        "id": event_id,
        "title": title,
        "slug": slug,
        "description": "Test event",
        "markets": markets,
        "negRisk": neg_risk,
        "category": "sports",
        "volume": "500000",
        "liquidity": "200000",
        "active": True,
        "closed": False,
    }


# ═══════════════════════════════════════════════════════════
# Tests: Type Parsing
# ═══════════════════════════════════════════════════════════

class TestTypeParsing:
    """Test raw API type parsing."""

    def test_parse_json_or_list_string(self):
        assert _parse_json_or_list('["Yes", "No"]') == ["Yes", "No"]

    def test_parse_json_or_list_already_list(self):
        assert _parse_json_or_list(["Yes", "No"]) == ["Yes", "No"]

    def test_parse_json_or_list_invalid(self):
        assert _parse_json_or_list("not json") == []

    def test_parse_json_or_list_none(self):
        assert _parse_json_or_list(None) == []

    def test_market_from_api_response(self):
        data = _make_market_response(
            condition_id="0xtest",
            question="Will it rain?",
            outcome_prices=[0.7, 0.3],
        )
        market = PolymarketMarket.from_api_response(data)

        assert market.condition_id == "0xtest"
        assert market.question == "Will it rain?"
        assert market.outcomes == ["Yes", "No"]
        assert market.yes_price == 0.7
        assert market.no_price == 0.3
        assert market.yes_token_id == "token_yes_0xtest"
        assert market.no_token_id == "token_no_0xtest"
        assert market.volume == 100000.0
        assert market.is_active is True

    def test_market_inactive(self):
        data = _make_market_response(closed=True)
        market = PolymarketMarket.from_api_response(data)
        assert market.is_active is False

    def test_event_from_api_response(self):
        data = _make_event_response(num_markets=3, neg_risk=True)
        event = PolymarketEvent.from_api_response(data)

        assert event.event_id == "evt_001"
        assert event.title == "F1 Drivers Championship 2026"
        assert event.neg_risk is True
        assert len(event.markets) == 3
        assert len(event.active_markets) == 3

    def test_event_active_markets_filter(self):
        data = _make_event_response(num_markets=3)
        data["markets"][1]["closed"] = True
        event = PolymarketEvent.from_api_response(data)

        assert len(event.markets) == 3
        assert len(event.active_markets) == 2

    def test_orderbook_parsing(self):
        data = {
            "bids": [
                {"price": "0.65", "size": "100"},
                {"price": "0.64", "size": "200"},
            ],
            "asks": [
                {"price": "0.67", "size": "150"},
                {"price": "0.68", "size": "300"},
            ],
        }
        book = PolymarketOrderBook.from_api_response("tok_1", data)
        assert book.best_bid == 0.65
        assert book.best_ask == 0.67
        assert book.midpoint == pytest.approx(0.66)
        assert book.spread == pytest.approx(0.02)

    def test_price_history_parsing(self):
        data = {
            "history": [
                {"t": 1700000000, "p": 0.55},
                {"t": 1700003600, "p": 0.60},
                {"t": 1700007200, "p": 0.58},
            ]
        }
        ph = PolymarketPriceHistory.from_api_response("tok_1", data)
        assert len(ph.points) == 3
        assert ph.points[0].price == 0.55
        assert ph.points[1].price == 0.60


# ═══════════════════════════════════════════════════════════
# Tests: Mapping to Core Types
# ═══════════════════════════════════════════════════════════

class TestMapping:
    """Test conversion from Polymarket types to core types."""

    def test_to_market_meta(self):
        data = _make_market_response(
            condition_id="cond_001",
            question="Will Max win?",
            slug="max-win",
            volume=200000,
        )
        pm = PolymarketMarket.from_api_response(data)
        meta = to_market_meta(pm)

        assert meta.id == "cond_001"
        assert meta.question == "Will Max win?"
        assert meta.slug == "max-win"
        assert meta.outcomes == ["Yes", "No"]
        assert meta.volume == 200000.0
        assert meta.status == MarketStatus.ACTIVE

    def test_to_market_meta_closed(self):
        data = _make_market_response(closed=True)
        pm = PolymarketMarket.from_api_response(data)
        meta = to_market_meta(pm)
        assert meta.status == MarketStatus.CLOSED

    def test_to_price_point(self):
        data = _make_market_response(outcome_prices=[0.72, 0.28])
        pm = PolymarketMarket.from_api_response(data)
        pp = to_price_point(pm)

        assert pp.prices["Yes"] == 0.72
        assert pp.prices["No"] == 0.28
        assert pp.total == pytest.approx(1.0)

    def test_to_market_snapshot(self):
        data = _make_market_response(outcome_prices=[0.65, 0.35])
        pm = PolymarketMarket.from_api_response(data)
        snap = to_market_snapshot(pm)

        assert snap.yes_price == 0.65
        assert snap.market.id == pm.condition_id

    def test_to_market_time_series(self):
        data = _make_market_response()
        pm = PolymarketMarket.from_api_response(data)

        history_data = {
            "history": [
                {"t": 1700000000, "p": 0.55},
                {"t": 1700003600, "p": 0.60},
            ]
        }
        history = PolymarketPriceHistory.from_api_response(pm.yes_token_id, history_data)
        ts = to_market_time_series(pm, history)

        assert len(ts.points) == 2
        assert ts.yes_prices[0] == 0.55
        assert ts.yes_prices[1] == 0.60

    def test_to_group_snapshot(self):
        event_data = _make_event_response(num_markets=2)
        event = PolymarketEvent.from_api_response(event_data)
        markets = {m.condition_id: m for m in event.markets}

        group = to_group_snapshot(event, markets)
        assert group.group_id == event.event_id
        assert len(group.snapshots) == 2


# ═══════════════════════════════════════════════════════════
# Tests: Relationship Inference
# ═══════════════════════════════════════════════════════════

class TestRelationshipInferrer:
    """Test constraint inference from event metadata."""

    def test_neg_risk_true_generates_mutex(self):
        """negRisk=True should generate MUTUALLY_EXCLUSIVE for all pairs."""
        data = _make_event_response(num_markets=3, neg_risk=True)
        event = PolymarketEvent.from_api_response(data)
        inferrer = RelationshipInferrer()

        constraints, is_partition = inferrer.infer_from_event(event)

        assert is_partition is True

        mutex = [c for c in constraints if c.type == ConstraintType.MUTUALLY_EXCLUSIVE]
        exhaustive = [c for c in constraints if c.type == ConstraintType.EXHAUSTIVE]

        # 3 markets -> 3 mutex pairs
        assert len(mutex) == 3
        # 3 exhaustive constraints (one per market)
        assert len(exhaustive) == 3

        # All mutex constraints have confidence=1.0
        for c in mutex:
            assert c.confidence == 1.0

    def test_neg_risk_false_no_constraints(self):
        """negRisk=False should produce no automatic constraints."""
        data = _make_event_response(num_markets=3, neg_risk=False)
        event = PolymarketEvent.from_api_response(data)
        inferrer = RelationshipInferrer()

        constraints, is_partition = inferrer.infer_from_event(event)

        assert is_partition is False
        assert len(constraints) == 0

    def test_single_market_no_constraints(self):
        """Single-market event should produce no constraints even with negRisk."""
        data = _make_event_response(num_markets=1, neg_risk=True)
        event = PolymarketEvent.from_api_response(data)
        inferrer = RelationshipInferrer()

        constraints, is_partition = inferrer.infer_from_event(event)

        assert is_partition is False
        assert len(constraints) == 0

    def test_build_market_group(self):
        """build_market_group produces correct MarketGroup."""
        data = _make_event_response(num_markets=4, neg_risk=True)
        event = PolymarketEvent.from_api_response(data)
        inferrer = RelationshipInferrer()

        group = inferrer.build_market_group(event)

        assert group.name == event.title
        assert group.is_partition is True
        assert group.group_type == GroupingType.PARTITION
        assert len(group.market_ids) == 4
        # 4 markets -> 6 mutex pairs + 4 exhaustive = 10 constraints
        assert len(group.constraints) == 10

    def test_build_relationship_graph(self):
        """build_relationship_graph produces solver-compatible graph."""
        data = _make_event_response(num_markets=3, neg_risk=True)
        event = PolymarketEvent.from_api_response(data)
        inferrer = RelationshipInferrer()

        graph = inferrer.build_relationship_graph(event)

        assert len(graph.clusters) == 1
        cluster = graph.clusters[0]
        assert cluster.is_partition is True
        assert len(cluster.market_ids) == 3
        assert len(cluster.relationships) == 3  # 3 mutex pairs (exhaustive filtered for solver safety)


# ═══════════════════════════════════════════════════════════
# Tests: GammaClient (mocked HTTP)
# ═══════════════════════════════════════════════════════════

class TestGammaClient:
    """Test GammaClient with mocked HTTP responses."""

    def test_get_event_by_slug(self):
        event_data = _make_event_response()
        client = GammaClient()

        with patch.object(client, "_request", return_value=[event_data]):
            event = client.get_event_by_slug("f1-drivers-championship-2026")

        assert event is not None
        assert event.title == "F1 Drivers Championship 2026"
        assert len(event.markets) == 3

    def test_get_event_by_slug_not_found(self):
        client = GammaClient()

        with patch.object(client, "_request", return_value=[]):
            event = client.get_event_by_slug("nonexistent")

        assert event is None

    def test_get_event_by_id(self):
        event_data = _make_event_response(event_id="evt_42")
        client = GammaClient()

        with patch.object(client, "_request", return_value=event_data):
            event = client.get_event_by_id("evt_42")

        assert event is not None
        assert event.event_id == "evt_42"

    def test_search_events(self):
        events = [
            _make_event_response(event_id="e1", title="F1 Race 1"),
            _make_event_response(event_id="e2", title="F1 Race 2"),
        ]
        client = GammaClient()

        with patch.object(client, "_request", return_value=events):
            results = client.search_events(query="F1")

        assert len(results) == 2

    def test_get_market_by_id(self):
        market_data = _make_market_response(condition_id="cond_99")
        client = GammaClient()

        with patch.object(client, "_request", return_value=market_data):
            market = client.get_market_by_id("cond_99")

        assert market is not None
        assert market.condition_id == "cond_99"


# ═══════════════════════════════════════════════════════════
# Tests: ClobClient (mocked HTTP)
# ═══════════════════════════════════════════════════════════

class TestClobClient:
    """Test ClobClient with mocked HTTP responses."""

    def test_get_midpoint(self):
        client = ClobClient()

        with patch.object(client, "_request", return_value={"mid": "0.6500"}):
            mid = client.get_midpoint("token_123")

        assert mid == pytest.approx(0.65)

    def test_get_midpoint_missing(self):
        client = ClobClient()

        with patch.object(client, "_request", return_value={}):
            mid = client.get_midpoint("token_123")

        assert mid is None

    def test_get_price(self):
        client = ClobClient()

        with patch.object(client, "_request", return_value={"price": "0.70"}):
            price = client.get_price("token_123", "buy")

        assert price == pytest.approx(0.70)

    def test_get_orderbook(self):
        client = ClobClient()
        book_data = {
            "bids": [{"price": "0.65", "size": "100"}],
            "asks": [{"price": "0.67", "size": "150"}],
        }

        with patch.object(client, "_request", return_value=book_data):
            book = client.get_orderbook("token_123")

        assert book is not None
        assert book.best_bid == 0.65
        assert book.best_ask == 0.67

    def test_get_price_history(self):
        client = ClobClient()
        history_data = {
            "history": [
                {"t": 1700000000, "p": 0.55},
                {"t": 1700003600, "p": 0.60},
            ]
        }

        with patch.object(client, "_request", return_value=history_data):
            history = client.get_price_history("token_123", interval="1d")

        assert history is not None
        assert len(history.points) == 2

    def test_get_midpoints_batch(self):
        client = ClobClient()

        with patch.object(client, "get_midpoint", side_effect=[0.65, 0.45]):
            results = client.get_midpoints_batch(["tok_a", "tok_b"])

        assert results == {"tok_a": 0.65, "tok_b": 0.45}


# ═══════════════════════════════════════════════════════════
# Tests: DatasetBuilder Integration
# ═══════════════════════════════════════════════════════════

class TestDatasetBuilder:
    """Integration test for DatasetBuilder with mock API."""

    def _make_builder(self) -> tuple[DatasetBuilder, GammaClient, ClobClient]:
        gamma = GammaClient()
        clob = ClobClient()
        builder = DatasetBuilder(gamma=gamma, clob=clob)
        return builder, gamma, clob

    def test_build_from_event_slug(self):
        """Build a dataset from an event slug with negRisk=True."""
        builder, gamma, clob = self._make_builder()
        event_data = _make_event_response(num_markets=3, neg_risk=True)

        with patch.object(gamma, "get_event_by_slug") as mock_get:
            mock_get.return_value = PolymarketEvent.from_api_response(event_data)

            # Mock CLOB midpoints (no-op if we skip refresh)
            with patch.object(clob, "get_midpoint", return_value=None):
                spec = DatasetSpec(
                    name="Test F1",
                    event_slugs=["f1-drivers-championship-2026"],
                    refresh_prices_from_clob=False,
                )
                dataset = builder.build(spec)

        assert dataset.name == "Test F1"
        assert len(dataset.events) == 1
        assert len(dataset.markets) == 3
        assert dataset.market_group.is_partition is True
        assert len(dataset.market_group.constraints) > 0

    def test_build_with_clob_refresh(self):
        """Build a dataset that refreshes prices from CLOB."""
        builder, gamma, clob = self._make_builder()
        event_data = _make_event_response(num_markets=2, neg_risk=True)

        with patch.object(gamma, "get_event_by_slug") as mock_get:
            mock_get.return_value = PolymarketEvent.from_api_response(event_data)

            with patch.object(clob, "get_midpoint", return_value=0.42):
                spec = DatasetSpec(
                    name="Test",
                    event_slugs=["test"],
                    refresh_prices_from_clob=True,
                )
                dataset = builder.build(spec)

        # All markets should have CLOB-refreshed prices
        for market in dataset.markets.values():
            assert market.yes_price == pytest.approx(0.42)

    def test_build_with_volume_filter(self):
        """Markets below min_volume should be filtered out."""
        builder, gamma, clob = self._make_builder()
        event_data = _make_event_response(num_markets=3)
        # Set one market to low volume
        event_data["markets"][2]["volume"] = "10"

        with patch.object(gamma, "get_event_by_slug") as mock_get:
            mock_get.return_value = PolymarketEvent.from_api_response(event_data)
            with patch.object(clob, "get_midpoint", return_value=None):
                spec = DatasetSpec(
                    name="Test",
                    event_slugs=["test"],
                    min_volume=1000,
                    refresh_prices_from_clob=False,
                )
                dataset = builder.build(spec)

        assert len(dataset.markets) == 2  # One filtered out


# ═══════════════════════════════════════════════════════════
# Tests: LiveDataset solver integration
# ═══════════════════════════════════════════════════════════

class TestLiveDataset:
    """Test LiveDataset.to_solver_input() and run_solver()."""

    def _make_dataset(self, num_markets: int = 3) -> LiveDataset:
        """Build a test LiveDataset."""
        event_data = _make_event_response(num_markets=num_markets, neg_risk=True)
        event = PolymarketEvent.from_api_response(event_data)
        markets = {m.condition_id: m for m in event.markets}

        inferrer = RelationshipInferrer()
        market_ids = list(markets.keys())
        group = inferrer.build_market_group(event, market_ids)
        graph = inferrer.build_relationship_graph(event, market_ids)

        return LiveDataset(
            name="Test",
            events=[event],
            markets=markets,
            market_group=group,
            relationships=graph,
        )

    def test_to_solver_input_format(self):
        """to_solver_input returns correct format for find_marginal_arbitrage."""
        dataset = self._make_dataset(num_markets=3)
        prices, graph, outcomes = dataset.to_solver_input()

        # Should have 3 markets
        assert len(prices) == 3
        assert len(outcomes) == 3

        # Each market should have [yes_price, no_price]
        for mid, p in prices.items():
            assert len(p) == 2
            assert 0.0 <= p[0] <= 1.0
            assert p[1] == pytest.approx(1.0 - p[0])

        # Outcomes should be ["Yes", "No"]
        for mid, o in outcomes.items():
            assert o == ["Yes", "No"]

        # Graph should be a RelationshipGraph
        assert isinstance(graph, RelationshipGraph)
        assert len(graph.clusters) == 1

    def test_to_solver_input_prices_match_markets(self):
        """Solver input prices should match market prices."""
        dataset = self._make_dataset(num_markets=2)
        prices, _, _ = dataset.to_solver_input()

        for mid, p in prices.items():
            market = dataset.markets[mid]
            assert p[0] == pytest.approx(market.yes_price)

    def test_run_solver_executes(self):
        """run_solver should execute without error."""
        dataset = self._make_dataset(num_markets=3)
        result = dataset.run_solver()

        assert result is not None
        assert result.kl_divergence >= 0.0
        assert result.converged is True or result.iterations > 0

    def test_summary(self):
        """summary() should return a readable string."""
        dataset = self._make_dataset(num_markets=3)
        s = dataset.summary()

        assert "Dataset: Test" in s
        assert "Events: 1" in s
        assert "Markets:" in s
        assert "Sum(Yes prices):" in s
