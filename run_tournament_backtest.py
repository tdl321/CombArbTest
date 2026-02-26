#!/usr/bin/env python3
"""Tournament-specific backtest for sports partition markets.

Standalone version that doesn't require the full backtest module.
"""

import sys
sys.path.insert(0, "/root/combarbbot")

import argparse
import logging
import json
import glob
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import duckdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "/root/prediction-market-analysis/data/polymarket"


@dataclass
class TournamentMarket:
    """A market that's part of a tournament partition."""
    market_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    volume: float
    end_date: Optional[datetime] = None
    closed: bool = False


@dataclass
class Tournament:
    """A tournament partition (mutually exclusive outcomes)."""
    name: str
    markets: list[TournamentMarket]
    
    @property
    def all_token_ids(self) -> list[str]:
        tokens = []
        for m in self.markets:
            tokens.append(m.yes_token_id)
            tokens.append(m.no_token_id)
        return tokens


@dataclass
class PartitionTick:
    """A price snapshot for all markets in a partition."""
    timestamp: datetime
    block_number: int
    prices: dict[str, float]
    partition_sum: float
    
    @property
    def violation(self) -> float:
        return abs(self.partition_sum - 1.0)


@dataclass
class TournamentBacktestResult:
    """Results from a tournament backtest."""
    tournament: Tournament
    ticks: list[PartitionTick]
    violations: list[PartitionTick]
    
    @property
    def min_sum(self) -> float:
        return min(t.partition_sum for t in self.ticks) if self.ticks else 0.0
    
    @property
    def max_sum(self) -> float:
        return max(t.partition_sum for t in self.ticks) if self.ticks else 0.0
    
    @property
    def avg_sum(self) -> float:
        return sum(t.partition_sum for t in self.ticks) / len(self.ticks) if self.ticks else 0.0
    
    @property
    def total_profit(self) -> float:
        return sum(v.violation for v in self.violations)


# Pre-defined tournaments
KNOWN_TOURNAMENTS = {
    "nba-mvp-2023-24": Tournament(
        name="NBA MVP 2023-24",
        markets=[
            TournamentMarket(
                market_id="500334",
                question="Will Luka Doncic win the 2023-24 NBA MVP?",
                yes_token_id="59142792435409296006151867236844804416169458214931096817238930512418420167705",
                no_token_id="9954333019185024376178629637224124315267626795800410439102371672117401535100",
                volume=198380,
            ),
            TournamentMarket(
                market_id="500332",
                question="Will Nikola Jokic win the 2023-24 NBA MVP?",
                yes_token_id="54185570520315123143574021744262228127280133590162961349055274706697137884202",
                no_token_id="51983709602471097643449718644005474958698139280907153982946676738710811168329",
                volume=136594,
            ),
            TournamentMarket(
                market_id="500333",
                question="Will Shai Gilgeous-Alexander win the 2023-24 NBA MVP?",
                yes_token_id="88586068497344261946583629458191787994732926164797543051008907601591537452744",
                no_token_id="40316406143905190163577472682455096944049904839276779271633701019560246072593",
                volume=122811,
            ),
            TournamentMarket(
                market_id="500336",
                question="Will Jayson Tatum win the 2023-24 NBA MVP?",
                yes_token_id="27660163373076269165226948560953005319649928690845268572528611445359346608634",
                no_token_id="98602612519958663144470884557859411904514251671642749404699562385518218276737",
                volume=43708,
            ),
            TournamentMarket(
                market_id="500335",
                question="Will Giannis Antetokounmpo win the 2023-24 NBA MVP?",
                yes_token_id="46460166632291713003130759836638610022497015297760563574419034249263171978308",
                no_token_id="70101639333220373995958615982024801878915687192687916435031444175868495699508",
                volume=41208,
            ),
            TournamentMarket(
                market_id="500337",
                question="Will another player win the 2023-24 NBA MVP?",
                yes_token_id="42590503078799437262704245085332014840264422346317840684744189168655649355795",
                no_token_id="3178053238041626061744657923485820430124069097655966977060875552696368586966",
                volume=43096,
            ),
        ],
    ),
}


def run_tournament_backtest(
    tournament: Tournament,
    data_dir: str = DEFAULT_DATA_DIR,
    violation_threshold: float = 0.01,
    interval_minutes: int = 60,
) -> TournamentBacktestResult:
    """Run backtest on a tournament partition."""
    trades_files = [
        f for f in glob.glob(f"{data_dir}/trades/*.parquet")
        if not f.split("/")[-1].startswith("._")
    ]
    
    blocks_files = [
        f for f in glob.glob(f"{data_dir}/blocks/*.parquet")
        if not f.split("/")[-1].startswith("._")
    ]
    
    conn = duckdb.connect()
    
    # Build token -> market mapping
    token_to_market = {}
    for m in tournament.markets:
        token_to_market[m.yes_token_id] = (m.market_id, "yes")
        token_to_market[m.no_token_id] = (m.market_id, "no")
    
    all_tokens = tournament.all_token_ids
    token_placeholders = ",".join(f"'{t}'" for t in all_tokens)
    
    logger.info("Querying trades for %d markets (%d tokens)...", 
                len(tournament.markets), len(all_tokens))
    
    query = f"""
        SELECT 
            t.block_number,
            t.maker_asset_id,
            t.taker_asset_id,
            t.maker_amount,
            t.taker_amount,
            b.timestamp
        FROM read_parquet({trades_files}) t
        LEFT JOIN read_parquet({blocks_files}) b ON t.block_number = b.block_number
        WHERE t.maker_asset_id IN ({token_placeholders})
           OR t.taker_asset_id IN ({token_placeholders})
        ORDER BY t.block_number
    """
    
    trades = conn.execute(query).fetchall()
    conn.close()
    
    logger.info("Found %d trades for tournament %s", len(trades), tournament.name)
    
    if not trades:
        return TournamentBacktestResult(tournament=tournament, ticks=[], violations=[])
    
    # Initialize prices at 50% (equal probability)
    current_prices = {m.market_id: 0.5 for m in tournament.markets}
    ticks = []
    
    interval = timedelta(minutes=interval_minutes)
    last_snapshot_time = None
    
    for trade in trades:
        block_num, maker_asset, taker_asset, maker_amt, taker_amt, ts_str = trade
        
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00").replace("+00:00", ""))
            except:
                continue
        else:
            continue
        
        # Calculate implied price from trade
        if maker_amt and taker_amt and (maker_amt + taker_amt) > 0:
            if maker_asset in token_to_market:
                mid, side = token_to_market[maker_asset]
                price = taker_amt / (maker_amt + taker_amt)
                if side == "yes":
                    current_prices[mid] = max(0.001, min(0.999, price))
            elif taker_asset in token_to_market:
                mid, side = token_to_market[taker_asset]
                price = maker_amt / (maker_amt + taker_amt)
                if side == "yes":
                    current_prices[mid] = max(0.001, min(0.999, price))
        
        # Create snapshot at intervals
        if last_snapshot_time is None or ts - last_snapshot_time >= interval:
            partition_sum = sum(current_prices.values())
            tick = PartitionTick(
                timestamp=ts,
                block_number=block_num,
                prices=dict(current_prices),
                partition_sum=partition_sum,
            )
            ticks.append(tick)
            last_snapshot_time = ts
    
    violations = [t for t in ticks if t.violation >= violation_threshold]
    
    return TournamentBacktestResult(
        tournament=tournament,
        ticks=ticks,
        violations=violations,
    )


def format_report(result: TournamentBacktestResult) -> str:
    """Format backtest result as text."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"Tournament: {result.tournament.name}")
    lines.append("=" * 70)
    
    lines.append(f"\nMarkets ({len(result.tournament.markets)}):")
    for m in result.tournament.markets:
        lines.append(f"  - {m.question}")
        lines.append(f"    ID: {m.market_id}, Volume: ${m.volume:,.0f}")
    
    if result.ticks:
        start = min(t.timestamp for t in result.ticks)
        end = max(t.timestamp for t in result.ticks)
        lines.append(f"\nTimeline: {start} to {end}")
        lines.append(f"Snapshots: {len(result.ticks)}")
    else:
        lines.append("\nNo trade data found!")
        return "\n".join(lines)
    
    lines.append(f"\nPartition Sum Analysis:")
    lines.append(f"  Min sum: {result.min_sum:.4f}")
    lines.append(f"  Max sum: {result.max_sum:.4f}")
    lines.append(f"  Avg sum: {result.avg_sum:.4f}")
    
    lines.append(f"\nViolations (|sum - 1| > threshold): {len(result.violations)}")
    if result.violations:
        lines.append("\nTop 10 violations:")
        top = sorted(result.violations, key=lambda v: -v.violation)[:10]
        for v in top:
            direction = "OVER" if v.partition_sum > 1 else "UNDER"
            lines.append(f"  {v.timestamp}: sum={v.partition_sum:.4f} ({direction})")
            # Show individual prices
            for mid, price in v.prices.items():
                lines.append(f"    {mid}: {price:.4f}")
    
    lines.append(f"\nArbitrage Opportunities: {len(result.violations)}")
    lines.append(f"Total Potential Profit: ${result.total_profit:.4f} per $1 wagered")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Tournament backtest")
    parser.add_argument("--tournament", type=str, default="nba-mvp-2023-24", 
                        help="Tournament key")
    parser.add_argument("--threshold", type=float, default=0.01, 
                        help="Violation threshold")
    parser.add_argument("--interval", type=int, default=60, 
                        help="Snapshot interval (minutes)")
    args = parser.parse_args()

    print("=" * 70)
    print("SPORTS TOURNAMENT ARBITRAGE BACKTEST")
    print("=" * 70)
    print(f"\nAvailable tournaments: {list(KNOWN_TOURNAMENTS.keys())}")

    if args.tournament not in KNOWN_TOURNAMENTS:
        print(f"Unknown tournament: {args.tournament}")
        return
    
    tournament = KNOWN_TOURNAMENTS[args.tournament]
    print(f"\nRunning backtest: {tournament.name}")
    print(f"Markets: {len(tournament.markets)}")
    print(f"Threshold: {args.threshold}")
    print(f"Interval: {args.interval} minutes")
    
    result = run_tournament_backtest(
        tournament=tournament,
        violation_threshold=args.threshold,
        interval_minutes=args.interval,
    )
    
    print("\n" + format_report(result))
    print("\nBacktest complete!")


if __name__ == "__main__":
    main()
