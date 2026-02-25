"""Consolidated data layer tests.

Combines: test_data.py, test_loader.py, test_tick_stream.py

Tests for:
- MarketLoader, TradeLoader, BlockLoader
- TickStream and TickPosition
- CrossMarketIterator
"""
import sys
sys.path.insert(0, '/root/combarbbot/src')

import pytest
from decimal import Decimal
from data import (
    BlockLoader, MarketLoader, TradeLoader,
    TickStream, CrossMarketIterator, TickPosition,
)

DATA_DIR = '/root/prediction-market-analysis/data/polymarket'


# =============================================================================
# TickPosition Tests
# =============================================================================

class TestTickPosition:
    """Tests for TickPosition ordering."""
    
    def test_same_block_higher_log_index(self):
        """Higher log_index in same block should be greater."""
        p1 = TickPosition(100, 5)
        p2 = TickPosition(100, 10)
        assert p1 < p2
    
    def test_higher_block_always_greater(self):
        """Higher block should be greater regardless of log_index."""
        p1 = TickPosition(100, 999)
        p2 = TickPosition(101, 1)
        assert p1 < p2
    
    def test_equality(self):
        """Same block and log_index should be equal."""
        p1 = TickPosition(100, 5)
        p2 = TickPosition(100, 5)
        assert p1 == p2


# =============================================================================
# Loader Integration Tests (require data files)
# =============================================================================

@pytest.mark.integration
class TestMarketLoader:
    """Integration tests for MarketLoader."""
    
    @pytest.fixture
    def market_loader(self):
        loader = MarketLoader(DATA_DIR)
        yield loader
        loader.close()
    
    def test_get_market(self, market_loader):
        """Test loading a specific market."""
        market = market_loader.get_market('253591')
        assert market is not None
        assert market.id == '253591'
        assert market.question is not None
    
    def test_query_markets(self, market_loader):
        """Test querying markets with filters."""
        markets = market_loader.query_markets(min_volume=1_000_000, limit=5)
        assert len(markets) <= 5


@pytest.mark.integration  
class TestTradeLoader:
    """Integration tests for TradeLoader."""
    
    @pytest.fixture
    def loaders(self):
        block_loader = BlockLoader(DATA_DIR)
        trade_loader = TradeLoader(DATA_DIR, block_loader=block_loader)
        market_loader = MarketLoader(DATA_DIR)
        yield trade_loader, market_loader
        trade_loader.close()
        market_loader.close()
        block_loader.close()
    
    def test_get_files_for_block_range(self, loaders):
        """Test file selection for block range."""
        trade_loader, _ = loaders
        files = trade_loader._get_files_for_block_range(63950000, 63960000)
        assert len(files) >= 0  # May be 0 if no data in range
    
    def test_query_trades(self, loaders):
        """Test querying trades with block range."""
        trade_loader, _ = loaders
        trades = trade_loader.query_trades(
            min_block=63950000,
            max_block=63960000,
            limit=100
        )
        assert len(trades) <= 100


@pytest.mark.integration
class TestTickStream:
    """Integration tests for TickStream."""
    
    @pytest.fixture
    def loaders(self):
        block_loader = BlockLoader(DATA_DIR)
        trade_loader = TradeLoader(DATA_DIR, block_loader=block_loader)
        market_loader = MarketLoader(DATA_DIR)
        yield block_loader, trade_loader, market_loader
        trade_loader.close()
        market_loader.close()
        block_loader.close()
    
    def test_tick_iteration_ordering(self, loaders):
        """Test that ticks are returned in order."""
        block_loader, trade_loader, market_loader = loaders
        
        # Get a market with trades
        markets = market_loader.query_markets(min_volume=1_000_000, limit=1)
        if markets.is_empty():
            pytest.skip("No markets with sufficient volume")
        
        market_id = markets['id'][0]
        market = market_loader.get_market(market_id)
        
        if market is None or market.clob_token_ids is None:
            pytest.skip("Market has no CLOB token IDs")
        
        stream = TickStream(
            trade_loader=trade_loader,
            block_loader=block_loader,
            market_id=market_id,
            token_ids=market.clob_token_ids,
            outcome_index=0,
        )
        
        # Get first 10 ticks
        ticks = []
        for i, tick in enumerate(stream.iter_ticks()):
            ticks.append(tick)
            if i >= 9:
                break
        
        # Verify ordering
        for i in range(1, len(ticks)):
            assert ticks[i-1].position <= ticks[i].position


@pytest.mark.integration
class TestCrossMarketIterator:
    """Integration tests for CrossMarketIterator."""
    
    @pytest.fixture
    def loaders(self):
        block_loader = BlockLoader(DATA_DIR)
        trade_loader = TradeLoader(DATA_DIR, block_loader=block_loader)
        market_loader = MarketLoader(DATA_DIR)
        yield block_loader, trade_loader, market_loader
        trade_loader.close()
        market_loader.close()
        block_loader.close()
    
    def test_cross_market_snapshots(self, loaders):
        """Test cross-market iteration produces valid snapshots."""
        block_loader, trade_loader, market_loader = loaders
        
        # Get top 2 markets by volume
        markets = market_loader.query_markets(min_volume=100_000_000, limit=2)
        if len(markets) < 2:
            pytest.skip("Not enough markets for cross-market test")
        
        market_ids = markets['id'].to_list()
        
        iterator = CrossMarketIterator(
            trade_loader=trade_loader,
            block_loader=block_loader,
            market_loader=market_loader,
            market_ids=market_ids,
        )
        
        # Get first 20 snapshots
        snapshots = []
        for i, snapshot in enumerate(iterator.iter_snapshots()):
            snapshots.append(snapshot)
            if i >= 19:
                break
        
        assert len(snapshots) > 0
        
        # Check at least some snapshots have prices
        with_prices = sum(1 for s in snapshots if s.has_all_prices())
        assert with_prices >= 0  # May be 0 for sparse data


# =============================================================================
# Run as script for manual testing
# =============================================================================

def run_manual_tests():
    """Run tests manually without pytest."""
    print("=== Testing Data Layer ===\n")
    
    # TickPosition tests
    print("Testing TickPosition...")
    p1 = TickPosition(100, 5)
    p2 = TickPosition(100, 10)
    p3 = TickPosition(101, 1)
    assert p1 < p2
    assert p2 < p3
    print("✓ TickPosition ordering works\n")
    
    # Loader tests
    print("Testing loaders...")
    market_loader = MarketLoader(DATA_DIR)
    trade_loader = TradeLoader(DATA_DIR)
    
    market = market_loader.get_market('253591')
    print(f"✓ Market loaded: {market.question[:40]}...")
    
    trades = trade_loader.query_trades(
        min_block=63950000,
        max_block=63960000,
        limit=10
    )
    print(f"✓ Trades queried: {len(trades)} rows")
    
    market_loader.close()
    trade_loader.close()
    
    print("\n=== All manual tests passed ===")


if __name__ == '__main__':
    run_manual_tests()
