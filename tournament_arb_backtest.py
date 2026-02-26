#!/usr/bin/env python3
"""Tournament Combinatorial Arbitrage Backtest.

Runs the Frank-Wolfe optimizer on tournament partition markets to detect
genuine combinatorial arbitrage opportunities.
"""

import sys
sys.path.insert(0, "/root/combarbbot")

import logging
import glob
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict

import duckdb
import numpy as np

from src.optimizer.frank_wolfe import find_marginal_arbitrage
from src.optimizer.schema import OptimizationConfig, RelationshipGraph, MarketCluster, MarketRelationship

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "/root/prediction-market-analysis/data/polymarket"


@dataclass
class TournamentMarket:
    market_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    volume: float


@dataclass
class ArbitrageOpportunity:
    timestamp: datetime
    block_number: int
    market_prices: dict[str, float]
    coherent_prices: dict[str, float]
    kl_divergence: float
    trade_directions: dict[str, str]  # BUY/SELL
    expected_profit: float


def build_partition_graph(market_ids: list[str], tournament_name: str) -> RelationshipGraph:
    """Build a RelationshipGraph with partition constraints.
    
    For a partition (exactly one outcome), all pairs are mutually exclusive.
    """
    relationships = []
    
    # Add mutually_exclusive for all pairs
    for i, m1 in enumerate(market_ids):
        for m2 in market_ids[i+1:]:
            relationships.append(MarketRelationship(
                type="mutually_exclusive",
                from_market=m1,
                to_market=m2,
                confidence=1.0,
            ))
    
    cluster = MarketCluster(
        cluster_id=tournament_name.lower().replace(" ", "-"),
        theme=tournament_name,
        market_ids=market_ids,
        is_partition=True,
        relationships=relationships,
    )
    
    return RelationshipGraph(clusters=[cluster])


def run_tournament_arb_backtest(
    markets: list[TournamentMarket],
    tournament_name: str,
    data_dir: str = DEFAULT_DATA_DIR,
    kl_threshold: float = 0.001,
    interval_minutes: int = 15,
    max_ticks: int = 10000,
) -> list[ArbitrageOpportunity]:
    """Run combinatorial arbitrage backtest on tournament markets.
    
    Uses Frank-Wolfe optimizer to find coherent prices and detect arbitrage.
    """
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
    for m in markets:
        token_to_market[m.yes_token_id] = (m.market_id, "yes")
        token_to_market[m.no_token_id] = (m.market_id, "no")
    
    all_tokens = []
    for m in markets:
        all_tokens.extend([m.yes_token_id, m.no_token_id])
    
    token_placeholders = ",".join(f"'{t}'" for t in all_tokens)
    
    logger.info("Loading trades for %d markets...", len(markets))
    
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
    
    logger.info("Found %d trades", len(trades))
    
    if not trades:
        return []
    
    # Build relationship graph with partition constraints
    market_ids = [m.market_id for m in markets]
    relationship_graph = build_partition_graph(market_ids, tournament_name)
    
    logger.info("Built partition graph with %d mutually_exclusive constraints",
                len(relationship_graph.clusters[0].relationships))
    
    # Optimizer config
    config = OptimizationConfig(
        max_iterations=100,
        tolerance=1e-6,
        step_mode="adaptive",
    )
    
    # Initialize prices at uniform
    n_markets = len(markets)
    current_prices = {m.market_id: 1.0 / n_markets for m in markets}
    
    opportunities = []
    interval = timedelta(minutes=interval_minutes)
    last_snapshot_time = None
    tick_count = 0
    
    logger.info("Running backtest with KL threshold %.4f...", kl_threshold)
    
    for trade in trades:
        block_num, maker_asset, taker_asset, maker_amt, taker_amt, ts_str = trade
        
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00").replace("+00:00", ""))
            except:
                continue
        else:
            continue
        
        # Update price from trade
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
        
        # Run solver at intervals
        if last_snapshot_time is None or ts - last_snapshot_time >= interval:
            tick_count += 1
            
            if tick_count > max_ticks:
                break
            
            # Format prices for solver: dict[str, list[float]]
            list_prices = {mid: [p, 1.0 - p] for mid, p in current_prices.items()}
            
            try:
                # Run Frank-Wolfe optimizer
                result = find_marginal_arbitrage(
                    market_prices=list_prices,
                    relationships=relationship_graph,
                    config=config,
                )
                
                # Check for arbitrage
                if result.kl_divergence >= kl_threshold:
                    # Extract coherent YES prices
                    coherent = {mid: prices[0] for mid, prices in result.coherent_market_prices.items()}
                    
                    # Determine trade directions
                    directions = {}
                    expected_profit = 0.0
                    
                    for mid in market_ids:
                        market_p = current_prices[mid]
                        coherent_p = coherent.get(mid, market_p)
                        diff = coherent_p - market_p
                        
                        if diff > 0.005:  # Market underpriced
                            directions[mid] = "BUY"
                            expected_profit += diff
                        elif diff < -0.005:  # Market overpriced
                            directions[mid] = "SELL"
                            expected_profit += abs(diff)
                        else:
                            directions[mid] = "HOLD"
                    
                    opp = ArbitrageOpportunity(
                        timestamp=ts,
                        block_number=block_num,
                        market_prices=dict(current_prices),
                        coherent_prices=coherent,
                        kl_divergence=result.kl_divergence,
                        trade_directions=directions,
                        expected_profit=expected_profit,
                    )
                    opportunities.append(opp)
                    
            except Exception as e:
                logger.warning("Solver error at tick %d: %s", tick_count, e)
            
            last_snapshot_time = ts
            
            if tick_count % 100 == 0:
                logger.info("Processed %d ticks, found %d opportunities", 
                           tick_count, len(opportunities))
    
    return opportunities


def format_backtest_report(
    opportunities: list[ArbitrageOpportunity],
    markets: list[TournamentMarket],
    tournament_name: str,
) -> str:
    """Format backtest results."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"COMBINATORIAL ARBITRAGE BACKTEST: {tournament_name}")
    lines.append("=" * 70)
    
    # Market summary
    lines.append(f"\nMarkets ({len(markets)}):")
    for m in markets[:10]:
        lines.append(f"  - {m.question[:55]}...")
    if len(markets) > 10:
        lines.append(f"  ... and {len(markets) - 10} more")
    
    # Opportunity summary
    lines.append(f"\nArbitrage Opportunities Found: {len(opportunities)}")
    
    if opportunities:
        kls = [o.kl_divergence for o in opportunities]
        profits = [o.expected_profit for o in opportunities]
        
        lines.append(f"\nKL Divergence Stats:")
        lines.append(f"  Min: {min(kls):.6f}")
        lines.append(f"  Max: {max(kls):.6f}")
        lines.append(f"  Avg: {np.mean(kls):.6f}")
        
        lines.append(f"\nExpected Profit Stats:")
        lines.append(f"  Min: ${min(profits):.4f}")
        lines.append(f"  Max: ${max(profits):.4f}")
        lines.append(f"  Total: ${sum(profits):.4f}")
        
        # Timeline
        start = min(o.timestamp for o in opportunities)
        end = max(o.timestamp for o in opportunities)
        lines.append(f"\nTimeline: {start} to {end}")
        
        # Top opportunities
        lines.append(f"\nTop 10 Opportunities by KL Divergence:")
        top = sorted(opportunities, key=lambda x: -x.kl_divergence)[:10]
        
        # Build market name lookup
        market_names = {m.market_id: m.question.split("Will the ")[1].split(" win")[0] 
                       if "Will the " in m.question else m.market_id 
                       for m in markets}
        
        for i, opp in enumerate(top):
            lines.append(f"\n  [{i+1}] {opp.timestamp} | KL={opp.kl_divergence:.6f} | Profit=${opp.expected_profit:.4f}")
            
            # Show mispriced markets
            for mid, direction in opp.trade_directions.items():
                if direction != "HOLD":
                    name = market_names.get(mid, mid)[:20]
                    mp = opp.market_prices[mid]
                    cp = opp.coherent_prices.get(mid, mp)
                    lines.append(f"      {direction} {name}: {mp:.3f} -> {cp:.3f}")
    
    return "\n".join(lines)


# =============================================================================
# TOURNAMENT DEFINITIONS
# =============================================================================

NBA_MVP_2023_24 = [
    TournamentMarket(market_id="500334", question="Will Luka Doncic win the 2023-24 NBA MVP?",
        yes_token_id="59142792435409296006151867236844804416169458214931096817238930512418420167705",
        no_token_id="9954333019185024376178629637224124315267626795800410439102371672117401535100", volume=198380),
    TournamentMarket(market_id="500332", question="Will Nikola Jokic win the 2023-24 NBA MVP?",
        yes_token_id="54185570520315123143574021744262228127280133590162961349055274706697137884202",
        no_token_id="51983709602471097643449718644005474958698139280907153982946676738710811168329", volume=136594),
    TournamentMarket(market_id="500333", question="Will Shai Gilgeous-Alexander win the 2023-24 NBA MVP?",
        yes_token_id="88586068497344261946583629458191787994732926164797543051008907601591537452744",
        no_token_id="40316406143905190163577472682455096944049904839276779271633701019560246072593", volume=122811),
    TournamentMarket(market_id="500336", question="Will Jayson Tatum win the 2023-24 NBA MVP?",
        yes_token_id="27660163373076269165226948560953005319649928690845268572528611445359346608634",
        no_token_id="98602612519958663144470884557859411904514251671642749404699562385518218276737", volume=43708),
    TournamentMarket(market_id="500335", question="Will Giannis Antetokounmpo win the 2023-24 NBA MVP?",
        yes_token_id="46460166632291713003130759836638610022497015297760563574419034249263171978308",
        no_token_id="70101639333220373995958615982024801878915687192687916435031444175868495699508", volume=41208),
    TournamentMarket(market_id="500337", question="Will another player win the 2023-24 NBA MVP?",
        yes_token_id="42590503078799437262704245085332014840264422346317840684744189168655649355795",
        no_token_id="3178053238041626061744657923485820430124069097655966977060875552696368586966", volume=43096),
]

# Super Bowl 2025 - Top 10 teams by volume (to keep solver fast)
SUPER_BOWL_2025_TOP10 = [
    TournamentMarket(market_id="503313", question="Will the Chiefs win Super Bowl 2025?",
        yes_token_id="22535833765723427929773245088435042776045949943240943368073750664192788269527",
        no_token_id="42871158580795323243941598536325925204919765750879891842520258699472618139885", volume=16403932),
    TournamentMarket(market_id="503324", question="Will the 49ers win Super Bowl 2025?",
        yes_token_id="57404941070480647064900845338248984784706447590708819584066371103229440035635",
        no_token_id="105887143603794106345878722758219961934527916904279993939903253209841411968927", volume=11810114),
    TournamentMarket(market_id="503322", question="Will the Eagles win Super Bowl 2025?",
        yes_token_id="110222417228270638383974743746762302792556220380554556504458115620557107501861",
        no_token_id="34527047802979125804174050325432167077742263137042415948357405372259768531455", volume=11829712),
    TournamentMarket(market_id="503307", question="Will the Lions win Super Bowl 2025?",
        yes_token_id="51052158557761079600060821198344572146234634299666295420235183322360518805559",
        no_token_id="24457538192307362260130901215385681960887087239978986650052790651700145579385", volume=12167588),
    TournamentMarket(market_id="503299", question="Will the Ravens win Super Bowl 2025?",
        yes_token_id="65899342545197974464674790375677332783836387984276968232607993321656475710546",
        no_token_id="44043530156416941552422118596418820865632961462273897699074006193128135935861", volume=6399020),
    TournamentMarket(market_id="503300", question="Will the Bills win Super Bowl 2025?",
        yes_token_id="10543796747987526217726719445503113036676541789761379932363198740436075720933",
        no_token_id="44496525088677969212608424691084899842806265405266245973131576352260105857324", volume=8862284),
    TournamentMarket(market_id="503310", question="Will the Texans win Super Bowl 2025?",
        yes_token_id="40946145547892120835388934032378411687415301148304670567000395360529369472824",
        no_token_id="13767388689589800926602299149240507927219262519162147884681600566873095986070", volume=33300682),
    TournamentMarket(market_id="503309", question="Will the Packers win Super Bowl 2025?",
        yes_token_id="7689215271552383133483004508832984274573710042218778913722716721310897959707",
        no_token_id="89002498769528768644726791543970005724643616817519213760138963002436160049197", volume=7565957),
    TournamentMarket(market_id="503328", question="Will the Commanders win Super Bowl 2025?",
        yes_token_id="16571355100327104454501190118689337347118543324830329817599252230759175472329",
        no_token_id="7958598477921247868125626874893923014274724309877711676804172989905474637451", volume=30528260),
    TournamentMarket(market_id="503305", question="Will the Cowboys win Super Bowl 2025?",
        yes_token_id="91353054216890740335748868776987613119107738900235010213982466959971119689491",
        no_token_id="8059405312579604783993514945805451199107762053048515732379629084368237423245", volume=12217168),
]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tournament", type=str, default="nba-mvp",
                       choices=["nba-mvp", "super-bowl"],
                       help="Tournament to backtest")
    parser.add_argument("--kl-threshold", type=float, default=0.001)
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--max-ticks", type=int, default=5000)
    args = parser.parse_args()
    
    if args.tournament == "nba-mvp":
        markets = NBA_MVP_2023_24
        name = "NBA MVP 2023-24"
    else:
        markets = SUPER_BOWL_2025_TOP10
        name = "Super Bowl 2025 (Top 10)"
    
    print("=" * 70)
    print(f"RUNNING COMBINATORIAL ARBITRAGE BACKTEST: {name}")
    print("=" * 70)
    print(f"Markets: {len(markets)}")
    print(f"KL Threshold: {args.kl_threshold}")
    print(f"Interval: {args.interval} minutes")
    print(f"Max Ticks: {args.max_ticks}")
    print()
    
    opportunities = run_tournament_arb_backtest(
        markets=markets,
        tournament_name=name,
        kl_threshold=args.kl_threshold,
        interval_minutes=args.interval,
        max_ticks=args.max_ticks,
    )
    
    print("\n" + format_backtest_report(opportunities, markets, name))
