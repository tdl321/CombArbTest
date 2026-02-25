"""Deep election backtest with implication chains."""
import sys
sys.path.insert(0, "/root/combarbbot")
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.backtest import run_backtest, print_report
from src.optimizer.schema import RelationshipGraph, MarketCluster, MarketRelationship

# Election markets with implication relationships
TRUMP_WINS = '253591'      # Trump wins 2024 election
HARRIS_WINS = '253597'     # Harris wins 2024 election
REP_WINS_PA = '255152'     # Republican wins Pennsylvania
DEM_WINS_PA = '255151'     # Democrat wins Pennsylvania

print('='*70)
print('DEEP ELECTION BACKTEST: Multi-Market Implication Chains')
print('='*70)
print()
print('Markets:')
print('  253591: Trump wins 2024 election')
print('  253597: Harris wins 2024 election')  
print('  255152: Republican wins Pennsylvania')
print('  255151: Democrat wins Pennsylvania')
print()
print('Implication Constraints:')
print('  RepPA -> Trump: If Rep wins PA, Trump wins national')
print('  DemPA -> Harris: If Dem wins PA, Harris wins national')
print()

graph = RelationshipGraph(clusters=[
    MarketCluster(
        cluster_id='election-chain',
        market_ids=[TRUMP_WINS, HARRIS_WINS, REP_WINS_PA, DEM_WINS_PA],
        relationships=[
            MarketRelationship(
                type='implies',
                from_market=REP_WINS_PA,
                to_market=TRUMP_WINS,
                confidence=1.0,
            ),
            MarketRelationship(
                type='implies',
                from_market=DEM_WINS_PA,
                to_market=HARRIS_WINS,
                confidence=1.0,
            ),
        ],
        is_partition=False,
    )
])

print('Running backtest with 100,000 ticks, threshold=0.001...')
print()

report = run_backtest(
    market_ids=[TRUMP_WINS, HARRIS_WINS, REP_WINS_PA, DEM_WINS_PA],
    relationship_graph=graph,
    max_ticks=100000,
    kl_threshold=0.001,  # 0.1% threshold - very sensitive
    progress_interval=10000,
)
print()
print_report(report)

# Show detailed opportunities if found
if report.opportunities:
    print()
    print('='*70)
    print(f'TOP 20 ARBITRAGE OPPORTUNITIES')
    print('='*70)
    
    sorted_opps = sorted(report.opportunities, key=lambda x: x.kl_divergence, reverse=True)
    
    for i, opp in enumerate(sorted_opps[:20]):
        print(f'\n--- Opportunity {i+1} ---')
        print(f'  KL Divergence: {opp.kl_divergence:.6f}')
        print(f'  Block: {opp.position[0] if opp.position else "N/A"}')
        print(f'  Gross: ${opp.theoretical_profit:.4f}')
        print(f'  Prices:')
        for mid, price in opp.market_prices.items():
            coherent = opp.coherent_prices.get(mid, price)
            market_name = {
                TRUMP_WINS: 'Trump',
                HARRIS_WINS: 'Harris',
                REP_WINS_PA: 'RepPA',
                DEM_WINS_PA: 'DemPA',
            }.get(mid, mid[:8])
            print(f'    {market_name}: {price:.3f} -> {coherent:.3f}')
