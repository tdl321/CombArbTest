"""Run backtest with fixed partition constraint detection."""
import sys
sys.path.insert(0, "/root/combarbbot")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from src.data import MarketLoader
from src.backtest import run_backtest, print_report
from src.optimizer.schema import RelationshipGraph, MarketCluster, MarketRelationship

DATA_DIR = "/root/prediction-market-analysis/data/polymarket"

print("=" * 70)
print("PARTITION CONSTRAINT BACKTEST (BUG FIXED)")
print("=" * 70)

market_loader = MarketLoader(DATA_DIR)

# Presidential election markets
pres_market_ids = ["253591", "253597", "253642", "253641", "253595", "253609"]

print("\nLoading Presidential Election markets...")
valid_ids = []
for mid in pres_market_ids:
    m = market_loader.get_market(mid)
    if m and m.clob_token_ids:
        valid_ids.append(mid)
        print(f"  {mid}: {m.question[:50]}...")

# Create partition cluster with is_partition=True
cluster = MarketCluster(
    cluster_id="pres-2024",
    market_ids=valid_ids,
    relationships=[],  # No relationships needed - is_partition flag is enough
    is_partition=True,  # This now triggers partition detection!
)

graph = RelationshipGraph(clusters=[cluster])

print(f"\nPartition cluster: {len(valid_ids)} markets with is_partition=True")

# Run backtest
print("\n" + "=" * 70)
print("RUNNING BACKTEST")
print("=" * 70)

report = run_backtest(
    market_ids=valid_ids,
    relationship_graph=graph,
    data_dir=DATA_DIR,
    kl_threshold=0.001,
    transaction_cost=0.015,
    max_ticks=20000,
)

print_report(report)

# Show top opportunities
if report.opportunities:
    print("\n" + "=" * 70)
    print(f"TOP 5 OPPORTUNITIES (of {len(report.opportunities)} total)")
    print("=" * 70)
    for i, opp in enumerate(report.opportunities[:5], 1):
        print(f"\n#{i}: Block {opp.block_number}")
        print(f"  Constraint: {opp.trade.constraint_type}")
        print(f"  Locked Profit: ${opp.trade.locked_profit:.4f}")
        print(f"  Net Profit: ${opp.net_profit():.4f}")
        print(f"  Direction: {opp.trade.description}")
else:
    print("\nNo opportunities found - something is still wrong")
