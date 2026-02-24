"""Test tick-level data layer."""
import sys
sys.path.insert(0, '/root/combarbbot/src')

from decimal import Decimal
from data import (
    BlockLoader, MarketLoader, TradeLoader,
    TickStream, CrossMarketIterator, TickPosition,
)

DATA_DIR = '/root/prediction-market-analysis/data/polymarket'

def test_tick_position_ordering():
    """Test TickPosition comparison."""
    p1 = TickPosition(100, 5)
    p2 = TickPosition(100, 10)
    p3 = TickPosition(101, 1)
    
    assert p1 < p2, "Same block, higher log_index should be greater"
    assert p2 < p3, "Higher block should be greater regardless of log_index"
    assert p1 < p3
    print("✓ TickPosition ordering works")


def test_tick_stream():
    """Test TickStream iteration."""
    block_loader = BlockLoader(DATA_DIR)
    trade_loader = TradeLoader(DATA_DIR, block_loader=block_loader)
    market_loader = MarketLoader(DATA_DIR)
    
    # Get a market with trades
    markets = market_loader.query_markets(min_volume=1_000_000, limit=1)
    if markets.is_empty():
        print("⚠ No markets found with sufficient volume")
        return
    
    market_id = markets['id'][0]
    market = market_loader.get_market(market_id)
    
    if market is None or market.clob_token_ids is None:
        print(f"⚠ Market {market_id} has no CLOB token IDs")
        return
    
    print(f"Testing with market: {market.question[:50]}...")
    print(f"Token IDs: {market.clob_token_ids}")
    
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
    
    print(f"\nFound {len(ticks)} ticks")
    
    if ticks:
        print("\nFirst 5 ticks:")
        for tick in ticks[:5]:
            print(f"  block={tick.block_number} log={tick.log_index} "
                  f"side={tick.side} price={tick.price:.4f} size={tick.size:.2f}")
        
        # Verify ordering
        for i in range(1, len(ticks)):
            assert ticks[i-1].position <= ticks[i].position, "Ticks not in order!"
        print("✓ Ticks are properly ordered")
    
    block_loader.close()
    trade_loader.close()
    market_loader.close()


def test_cross_market_iterator():
    """Test CrossMarketIterator."""
    block_loader = BlockLoader(DATA_DIR)
    trade_loader = TradeLoader(DATA_DIR, block_loader=block_loader)
    market_loader = MarketLoader(DATA_DIR)
    
    # Get top 2 markets by volume
    markets = market_loader.query_markets(min_volume=100_000_000, limit=2)
    if len(markets) < 2:
        print("⚠ Not enough markets for cross-market test")
        return
    
    market_ids = markets['id'].to_list()
    print(f"Testing cross-market with {len(market_ids)} markets")
    
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
    
    print(f"\nFound {len(snapshots)} snapshots")
    
    if snapshots:
        print("\nSample snapshots:")
        for snap in snapshots[:3]:
            prices = snap.get_prices()
            price_str = ", ".join(f"{mid[:8]}={p:.4f}" if p else f"{mid[:8]}=None" 
                                   for mid, p in prices.items())
            print(f"  pos=({snap.position.block_number}, {snap.position.log_index}) prices=[{price_str}]")
        
        # Check how many snapshots have all prices
        with_prices = sum(1 for s in snapshots if s.has_all_prices())
        print(f"\n✓ {with_prices}/{len(snapshots)} snapshots have all prices")
    
    block_loader.close()
    trade_loader.close()
    market_loader.close()


if __name__ == '__main__':
    print("=== Testing Tick-Level Data Layer ===\n")
    test_tick_position_ordering()
    print()
    test_tick_stream()
    print()
    test_cross_market_iterator()
    print("\n=== All tests passed ===")
