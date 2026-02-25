"""Combinatorial Arbitrage Backtest - TRUE implication chains."""
import sys
sys.path.insert(0, "/root/combarbbot")
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from src.backtest import run_backtest, print_report
from src.optimizer.schema import RelationshipGraph, MarketCluster, MarketRelationship

# REAL combinatorial markets
TRUMP_WINS = '253591'      # Trump wins 2024 election
REP_WINS_PA = '255152'     # Republican wins Pennsylvania

print('='*60)
print('COMBINATORIAL BACKTEST: Implication Chain')
print('='*60)
print()
print('Markets:')
print('  A (253591): Trump wins 2024 election')
print('  B (255152): Republican wins Pennsylvania')
print()
print('Constraint: B -> A (Rep wins PA implies Trump wins)')
print('  This is TRUE combinatorial - not simple mutex!')
print()

graph = RelationshipGraph(clusters=[
    MarketCluster(
        cluster_id='pa-national-chain',
        market_ids=[TRUMP_WINS, REP_WINS_PA],
        relationships=[
            MarketRelationship(
                type='implies',
                from_market=REP_WINS_PA,  # B
                to_market=TRUMP_WINS,      # A
                confidence=1.0,
            )
        ],
        is_partition=False,
    )
])

report = run_backtest(
    market_ids=[TRUMP_WINS, REP_WINS_PA],
    relationship_graph=graph,
    max_ticks=30,
    kl_threshold=0.005,
)
print_report(report)
