import sys
sys.path.insert(0, 'src')

from data import MarketLoader, TradeLoader, BlockLoader

data_dir = '/root/prediction-market-analysis/data/polymarket'
market_loader = MarketLoader(data_dir)
trade_loader = TradeLoader(data_dir)
block_loader = BlockLoader(data_dir)

print('=== DATA-02: Trade Loading ===')
market = market_loader.get_market('253591')
print(f'Market: {market.question[:50]}')

market_trades = trade_loader.get_trades_for_market(
    market.clob_token_ids,
    min_block=63900000,
    max_block=63960000,
)
print(f'Trades: {len(market_trades)}')

if not market_trades.is_empty():
    enriched = trade_loader.enrich_with_timestamps(market_trades)
    print()
    print('Sample:')
    print(enriched.select(['block_number', 'block_timestamp', 'maker_amount']).head(3))
    print()
    print('DATA-02 passed!')
