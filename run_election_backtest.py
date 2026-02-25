#!/usr/bin/env python3
"""Deep backtest for 2024 election markets with implication chains."""

import sys
sys.path.insert(0, '/root/combarbbot')

import duckdb
from datetime import datetime, timedelta
from collections import defaultdict
import json

# Data paths
TRADES_DIR = '/root/prediction-market-analysis/data/kalshi/trades/*.parquet'

# Markets to analyze with implication chains
# RepPA → Trump: If Republican wins PA, Trump wins presidency
# DemPA → Harris: If Democrat wins PA, Harris wins presidency  
# Also GA (Georgia) with same logic
MARKETS = {
    'TRUMP': 'PRES-2024-DJT',
    'HARRIS': 'PRES-2024-KH',
    'REP_PA': 'PRESPARTYPA-24-R',
    'DEM_PA': 'PRESPARTYPA-24-D',
    'REP_GA': 'PRESPARTYGA-24-R',
    'DEM_GA': 'PRESPARTYGA-24-D',
}

# Implication chains: (from, to) means from→to (from implies to)
IMPLICATIONS = [
    ('REP_PA', 'TRUMP'),   # If Rep wins PA → Trump wins
    ('DEM_PA', 'HARRIS'),  # If Dem wins PA → Harris wins
    ('REP_GA', 'TRUMP'),   # If Rep wins GA → Trump wins
    ('DEM_GA', 'HARRIS'),  # If Dem wins GA → Harris wins
]

def load_trades(conn):
    """Load all election market trades."""
    tickers = list(MARKETS.values())
    placeholders = ','.join(f"'{t}'" for t in tickers)
    
    query = f"""
        SELECT ticker, created_time, yes_price, count
        FROM read_parquet(['{TRADES_DIR}'])
        WHERE ticker IN ({placeholders})
        ORDER BY created_time
    """
    
    print(f"Loading trades for {len(tickers)} markets...")
    trades = conn.execute(query).fetchall()
    print(f"Loaded {len(trades):,} trades")
    return trades


def create_snapshots(trades, interval_minutes=60):
    """Create price snapshots at regular intervals."""
    # Build ticker → market name mapping
    ticker_to_name = {v: k for k, v in MARKETS.items()}
    
    # Track current prices and create snapshots
    current_prices = {}
    snapshots = []
    
    if not trades:
        return snapshots
    
    # Start at first trade time, rounded to hour
    first_time = trades[0][1].replace(minute=0, second=0, microsecond=0)
    interval = timedelta(minutes=interval_minutes)
    current_bucket = first_time
    
    trade_idx = 0
    last_snapshot_time = None
    
    while trade_idx < len(trades):
        bucket_end = current_bucket + interval
        
        # Process all trades in this bucket
        while trade_idx < len(trades) and trades[trade_idx][1] < bucket_end:
            ticker, ts, price, count = trades[trade_idx]
            name = ticker_to_name.get(ticker)
            if name and price is not None:
                current_prices[name] = float(price) / 100.0  # Convert cents to probability
            trade_idx += 1
        
        # Save snapshot if we have prices
        if len(current_prices) >= 2:  # Need at least 2 markets
            snapshots.append({
                'timestamp': current_bucket,
                'prices': dict(current_prices)
            })
        
        current_bucket = bucket_end
        
        # Skip empty periods
        if trade_idx < len(trades):
            next_trade_time = trades[trade_idx][1]
            while current_bucket + interval < next_trade_time:
                current_bucket += interval
    
    return snapshots


def check_implication_violation(prices, from_market, to_market):
    """Check if from→to implication is violated.
    
    If from→to, then P(from) <= P(to) must hold.
    Violation: P(from) > P(to) means we can profit.
    """
    if from_market not in prices or to_market not in prices:
        return None
    
    p_from = prices[from_market]
    p_to = prices[to_market]
    
    violation = p_from - p_to
    if violation > 0.001:  # 0.1% threshold
        return {
            'from': from_market,
            'to': to_market,
            'p_from': p_from,
            'p_to': p_to,
            'violation': violation,
            'profit_pct': violation * 100
        }
    return None


def run_backtest(snapshots, threshold=0.01):
    """Run backtest and find arbitrage opportunities."""
    opportunities = []
    
    for snap in snapshots:
        prices = snap['prices']
        timestamp = snap['timestamp']
        
        for from_m, to_m in IMPLICATIONS:
            violation = check_implication_violation(prices, from_m, to_m)
            if violation and violation['violation'] >= threshold:
                opportunities.append({
                    'timestamp': timestamp,
                    **violation,
                    'all_prices': prices
                })
    
    return opportunities


def analyze_results(opportunities):
    """Analyze and summarize arbitrage opportunities."""
    if not opportunities:
        print("\nNo arbitrage opportunities found above threshold.")
        return
    
    print(f"\n{'='*60}")
    print(f"ARBITRAGE OPPORTUNITIES FOUND: {len(opportunities)}")
    print(f"{'='*60}\n")
    
    # Group by implication chain
    by_chain = defaultdict(list)
    for opp in opportunities:
        key = f"{opp['from']}→{opp['to']}"
        by_chain[key].append(opp)
    
    for chain, opps in sorted(by_chain.items(), key=lambda x: -len(x[1])):
        violations = [o['violation'] for o in opps]
        avg_v = sum(violations) / len(violations)
        max_v = max(violations)
        
        print(f"\n{chain}: {len(opps)} opportunities")
        print(f"  Avg violation: {avg_v*100:.2f}%")
        print(f"  Max violation: {max_v*100:.2f}%")
        
        # Show top 5 opportunities
        top = sorted(opps, key=lambda x: -x['violation'])[:5]
        print(f"  Top opportunities:")
        for o in top:
            ts = o['timestamp'].strftime('%Y-%m-%d %H:%M')
            print(f"    {ts}: {o['from']}={o['p_from']:.1%} > {o['to']}={o['p_to']:.1%} | Profit: {o['profit_pct']:.1f}%")
    
    # Summary stats
    all_violations = [o['violation'] for o in opportunities]
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total opportunities: {len(opportunities)}")
    print(f"Average profit potential: {sum(all_violations)/len(all_violations)*100:.2f}%")
    print(f"Maximum profit potential: {max(all_violations)*100:.2f}%")
    
    # Time distribution
    times = [o['timestamp'] for o in opportunities]
    print(f"Time range: {min(times)} to {max(times)}")
    
    return opportunities


def main():
    print("="*60)
    print("DEEP BACKTEST: 2024 Election Arbitrage Detection")
    print("="*60)
    print(f"\nMarkets: {list(MARKETS.keys())}")
    print(f"Implications: {IMPLICATIONS}")
    print()
    
    conn = duckdb.connect()
    
    # Load trades
    trades = load_trades(conn)
    if not trades:
        print("No trades found!")
        return
    
    # Create hourly snapshots
    print("\nCreating hourly price snapshots...")
    snapshots = create_snapshots(trades, interval_minutes=60)
    print(f"Created {len(snapshots)} snapshots")
    
    # Also create 15-minute snapshots for finer resolution
    print("\nCreating 15-minute price snapshots...")
    snapshots_15m = create_snapshots(trades, interval_minutes=15)
    print(f"Created {len(snapshots_15m)} snapshots")
    
    # Run backtest with different thresholds
    for threshold in [0.01, 0.02, 0.05]:  # 1%, 2%, 5%
        print(f"\n{'#'*60}")
        print(f"THRESHOLD: {threshold*100:.0f}%")
        print(f"{'#'*60}")
        
        opportunities = run_backtest(snapshots_15m, threshold=threshold)
        analyze_results(opportunities)
    
    conn.close()
    print("\nBacktest complete!")


if __name__ == '__main__':
    main()
