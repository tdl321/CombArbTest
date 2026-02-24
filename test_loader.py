import sys
sys.path.insert(0, 'src')

from data import MarketLoader, TradeLoader, BlockLoader

data_dir = '/root/prediction-market-analysis/data/polymarket'

print('1. Testing MarketLoader...')
market_loader = MarketLoader(data_dir)
market = market_loader.get_market('253591')
print(f'   Market: {market.question[:40]}...')

print('2. Testing TradeLoader with block range filter...')
trade_loader = TradeLoader(data_dir)

# Get files for a specific block range
files = trade_loader._get_files_for_block_range(63950000, 63960000)
print(f'   Files matching block range: {len(files)}')

# Query trades
trades = trade_loader.query_trades(
    min_block=63950000,
    max_block=63960000,
    limit=100
)
print(f'   Trades in range: {len(trades)}')

print('3. Testing market-specific trades...')
market_trades = trade_loader.get_trades_for_market(
    market.clob_token_ids,
    min_block=63950000,
    max_block=63960000,
)
print(f'   Market trades: {len(market_trades)}')

if not market_trades.is_empty():
    print(f'   Sample: {market_trades.head(1)}')

print()
print('All tests passed!')
