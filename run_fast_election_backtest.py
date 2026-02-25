"""Fast election backtest with tuned optimizer config."""
import sys
sys.path.insert(0, "/root/combarbbot")
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

from src.backtest import run_backtest, print_report
from src.optimizer.schema import RelationshipGraph, MarketCluster, MarketRelationship, OptimizationConfig

# Election markets
TRUMP_WINS = '253591'
HARRIS_WINS = '253597'
REP_WINS_PA = '255152'
DEM_WINS_PA = '255151'

print('='*70)
print('FAST ELECTION BACKTEST: Tuned Optimizer')
print('='*70)

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

# Fast config: fewer iterations, looser tolerance
fast_config = OptimizationConfig(
    max_iterations=100,  # Instead of 1000
    tolerance=1e-3,      # Instead of 1e-6
)

print(f'Config: max_iter={fast_config.max_iterations}, tol={fast_config.tolerance}')
print('Running backtest with 50,000 ticks...')

report = run_backtest(
    market_ids=[TRUMP_WINS, HARRIS_WINS, REP_WINS_PA, DEM_WINS_PA],
    relationship_graph=graph,
    max_ticks=50000,
    kl_threshold=0.001,
    progress_interval=5000,
)

print_report(report)

if report.opportunities:
    print(f'\nFound {len(report.opportunities)} opportunities!')
    sorted_opps = sorted(report.opportunities, key=lambda x: x.kl_divergence, reverse=True)[:10]
    for i, opp in enumerate(sorted_opps):
        print(f'{i+1}. KL={opp.kl_divergence:.4f} Profit=${opp.theoretical_profit:.2f}')
