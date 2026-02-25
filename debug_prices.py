"""Debug prices from market data."""
import sys
sys.path.insert(0, "/root/combarbbot")

from src.data import MarketLoader, TradeLoader, BlockLoader, CrossMarketIterator

DATA_DIR = "/root/prediction-market-analysis/data/polymarket"

market_loader = MarketLoader(DATA_DIR)
trade_loader = TradeLoader(DATA_DIR)
block_loader = BlockLoader(DATA_DIR)

pres_market_ids = ["253591", "253597", "253642", "253641", "253595", "253609"]

valid_ids = []
for mid in pres_market_ids:
    m = market_loader.get_market(mid)
    if m and m.clob_token_ids:
        valid_ids.append(mid)

iterator = CrossMarketIterator(
    trade_loader=trade_loader,
    block_loader=block_loader,
    market_loader=market_loader,
    market_ids=valid_ids,
)

# Count snapshots with all 6 prices
full_price_count = 0
partial_counts = {i: 0 for i in range(7)}
first_full = None
MAX_TICKS = 5000

print("Scanning snapshots for price coverage...")
for i, snapshot in enumerate(iterator.iter_snapshots()):
    if i >= MAX_TICKS:
        break
    prices = snapshot.get_prices()
    valid_count = sum(1 for p in prices.values() if p is not None)
    partial_counts[valid_count] += 1
    
    if valid_count == 6:
        full_price_count += 1
        if first_full is None:
            first_full = i
            float_prices = {k: float(v) if v else None for k, v in prices.items()}
            total = sum(p for p in float_prices.values() if p is not None)
            print(f"\nFirst full snapshot at tick {i}:")
            for mid, p in float_prices.items():
                print(f"  {mid}: {p:.4f}")
            print(f"  SUM: {total:.4f} (deviation from 1: {total - 1:.4f})")

print(f"\nPrice coverage distribution ({MAX_TICKS} ticks):")
for count, num in sorted(partial_counts.items()):
    pct = num / (MAX_TICKS/100)
    print(f"  {count}/6 prices: {num} snapshots ({pct:.1f}%)")
    
print(f"\nSnapshots with all 6 prices: {full_price_count}")
