"""Tournament-specific backtest for sports partition markets.

Sports tournaments have clean partition constraints:
- Championship winner markets: exactly 1 team wins - sum P(teams) = 1
- Bracket outcomes: mutually exclusive advancement
- Game results: Win/Lose/Draw partitions
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Generator, Optional
import json
import glob

import duckdb

from src.llm import RelationshipGraph, MarketCluster, MarketRelationship

logger = logging.getLogger(__name__)

from src.config import DEFAULT_DATA_DIR


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
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    @property
    def market_ids(self) -> list[str]:
        return [m.market_id for m in self.markets]
    
    @property
    def yes_token_ids(self) -> list[str]:
        return [m.yes_token_id for m in self.markets]
    
    @property
    def all_token_ids(self) -> list[str]:
        tokens = []
        for m in self.markets:
            tokens.append(m.yes_token_id)
            tokens.append(m.no_token_id)
        return tokens

    def to_relationship_graph(self) -> RelationshipGraph:
        """Convert tournament to a RelationshipGraph with partition constraints."""
        relationships = []
        for i, m1 in enumerate(self.markets):
            for m2 in self.markets[i+1:]:
                relationships.append(MarketRelationship(
                    type="mutually_exclusive",
                    from_market=m1.market_id,
                    to_market=m2.market_id,
                    confidence=1.0,
                ))
        
        cluster = MarketCluster(
            cluster_id=self.name.lower().replace(" ", "-"),
            theme=self.name,
            market_ids=self.market_ids,
            is_partition=True,
            relationships=relationships,
        )
        
        return RelationshipGraph(clusters=[cluster])


@dataclass
class PartitionTick:
    """A price snapshot for all markets in a partition."""
    timestamp: datetime
    block_number: int
    prices: dict[str, float]
    partition_sum: float
    
    @property
    def violation(self) -> float:
        """Amount of partition violation (should be 0 for valid partition)."""
        return abs(self.partition_sum - 1.0)
    
    @property
    def arbitrage_profit(self) -> float:
        """Profit from arbitrage if partition is violated."""
        return self.violation


@dataclass
class TournamentBacktestResult:
    """Results from a tournament backtest."""
    tournament: Tournament
    ticks: list[PartitionTick]
    violations: list[PartitionTick]
    
    @property
    def min_sum(self) -> float:
        if not self.ticks:
            return 0.0
        return min(t.partition_sum for t in self.ticks)
    
    @property
    def max_sum(self) -> float:
        if not self.ticks:
            return 0.0
        return max(t.partition_sum for t in self.ticks)
    
    @property
    def avg_sum(self) -> float:
        if not self.ticks:
            return 0.0
        return sum(t.partition_sum for t in self.ticks) / len(self.ticks)
    
    @property
    def total_profit(self) -> float:
        """Total potential profit from all violations."""
        return sum(v.arbitrage_profit for v in self.violations)


def discover_tournaments(
    db_path: str = "/root/combarbbot/polymarket.db",
    data_dir: str = DEFAULT_DATA_DIR,
    subcategory: str = "nba",
    search_pattern: str = "%mvp%",
) -> list[Tournament]:
    """Discover tournament markets from categorized data."""
    conn = duckdb.connect(db_path)
    
    market_ids = conn.execute(
        "SELECT market_id FROM market_categories WHERE category='sports' AND subcategory=?",
        [subcategory]
    ).fetchall()
    market_ids = [r[0] for r in market_ids]
    
    if not market_ids:
        logger.warning("No markets found for subcategory=%s", subcategory)
        return []
    
    parquet_files = [
        f for f in glob.glob(f"{data_dir}/markets/*.parquet")
        if not f.split("/")[-1].startswith("._")
    ]
    
    placeholders = ",".join(f"'{mid}'" for mid in market_ids)
    query = f"""
        SELECT id, question, clob_token_ids, volume, end_date, closed
        FROM read_parquet({parquet_files})
        WHERE id IN ({placeholders})
        AND LOWER(question) LIKE '{search_pattern}'
        ORDER BY end_date DESC, volume DESC
    """
    
    results = conn.execute(query).fetchall()
    conn.close()
    
    tournaments_by_date = {}
    for r in results:
        market_id, question, tokens_json, volume, end_date, closed = r
        tokens = json.loads(tokens_json) if tokens_json else []
        
        if len(tokens) < 2:
            continue
        
        key = end_date.strftime("%Y-%m") if end_date else "unknown"
        
        if key not in tournaments_by_date:
            tournaments_by_date[key] = []
        
        tournaments_by_date[key].append(TournamentMarket(
            market_id=market_id,
            question=question,
            yes_token_id=tokens[0],
            no_token_id=tokens[1],
            volume=volume or 0,
            end_date=end_date,
            closed=closed,
        ))
    
    tournaments = []
    for key, markets in tournaments_by_date.items():
        if len(markets) >= 2:
            name = markets[0].question.split("win")[0].strip() if "win" in markets[0].question.lower() else key
            name = name.replace("Will", "").strip()
            
            tournaments.append(Tournament(
                name=f"{subcategory.upper()} {name} ({key})",
                markets=markets,
                end_date=markets[0].end_date,
            ))
    
    return tournaments


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
    
    token_to_market = {}
    for m in tournament.markets:
        token_to_market[m.yes_token_id] = (m.market_id, "yes")
        token_to_market[m.no_token_id] = (m.market_id, "no")
    
    all_tokens = tournament.all_token_ids
    token_placeholders = ",".join(f"'{t}'" for t in all_tokens)
    
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
        
        if maker_amt and taker_amt and (maker_amt + taker_amt) > 0:
            if maker_asset in token_to_market:
                mid, side = token_to_market[maker_asset]
                price = taker_amt / (maker_amt + taker_amt)
                if side == "yes":
                    current_prices[mid] = price
            elif taker_asset in token_to_market:
                mid, side = token_to_market[taker_asset]
                price = maker_amt / (maker_amt + taker_amt)
                if side == "yes":
                    current_prices[mid] = price
        
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


def format_tournament_report(result: TournamentBacktestResult) -> str:
    """Format a tournament backtest result as text."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"Tournament: {result.tournament.name}")
    lines.append("=" * 60)
    
    lines.append(f"\nMarkets ({len(result.tournament.markets)}):")
    for m in result.tournament.markets:
        lines.append(f"  - {m.question[:60]}...")
        lines.append(f"    ID: {m.market_id}, Volume: {m.volume:,.0f}")
    
    if result.ticks:
        start = min(t.timestamp for t in result.ticks)
        end = max(t.timestamp for t in result.ticks)
        lines.append(f"\nTimeline: {start} to {end}")
        lines.append(f"Snapshots: {len(result.ticks)}")
    
    lines.append(f"\nPartition Sum Analysis:")
    lines.append(f"  Min sum: {result.min_sum:.4f}")
    lines.append(f"  Max sum: {result.max_sum:.4f}")
    lines.append(f"  Avg sum: {result.avg_sum:.4f}")
    
    lines.append(f"\nViolations (|sum - 1| > threshold): {len(result.violations)}")
    if result.violations:
        lines.append("Top 10 violations:")
        top_violations = sorted(result.violations, key=lambda v: -v.violation)[:10]
        for v in top_violations:
            direction = "OVER" if v.partition_sum > 1 else "UNDER"
            lines.append(f"  {v.timestamp}: sum={v.partition_sum:.4f} ({direction}) profit={v.arbitrage_profit:.4f}")
    
    lines.append(f"\nArbitrage Opportunities: {len(result.violations)}")
    lines.append(f"Total Potential Profit: ${result.total_profit:.4f} per $1 wagered")
    
    return "\n".join(lines)
