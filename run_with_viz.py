#!/usr/bin/env python3
"""Tournament backtest with simplex visualizations."""

import sys
sys.path.insert(0, "/root/combarbbot")

import logging
import glob
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

import numpy as np
import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Imports
from src.optimizer.frank_wolfe import find_marginal_arbitrage
from src.optimizer.schema import OptimizationConfig, RelationshipGraph, MarketCluster, MarketRelationship

# Try to import visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from src.visualization.simplex import SimplexProjector
    HAS_VIZ = True
except ImportError as e:
    logger.warning("Visualization unavailable: %s", e)
    HAS_VIZ = False

DEFAULT_DATA_DIR = "/root/prediction-market-analysis/data/polymarket"


@dataclass
class TournamentMarket:
    market_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    volume: float
    short_name: str = ""


@dataclass
class ArbitrageOpportunity:
    timestamp: datetime
    block_number: int
    market_prices: dict[str, float]
    coherent_prices: dict[str, float]
    kl_divergence: float
    trade_directions: dict[str, str]
    expected_profit: float


def build_partition_graph(market_ids: list[str], tournament_name: str) -> RelationshipGraph:
    relationships = []
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


def generate_simplex_plot(
    opp: ArbitrageOpportunity,
    markets: list[TournamentMarket],
    output_dir: Path,
    index: int,
) -> Optional[Path]:
    """Generate simplex visualization for an arbitrage opportunity."""
    if not HAS_VIZ:
        return None
    
    n = len(markets)
    labels = [m.short_name for m in markets]
    
    projector = SimplexProjector(n, labels)
    
    # Get prices as arrays (in market order)
    market_prices = np.array([opp.market_prices.get(m.market_id, 0) for m in markets])
    coherent_prices = np.array([opp.coherent_prices.get(m.market_id, 0) for m in markets])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw simplex boundary
    vertices = projector.vertices
    poly = Polygon(vertices, closed=True, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(poly)
    
    # Label vertices
    for i, (v, label) in enumerate(zip(vertices, labels)):
        offset = v - vertices.mean(axis=0)
        offset = offset / np.linalg.norm(offset) * 0.15
        ax.annotate(label, v + offset, fontsize=10, ha='center', va='center')
    
    # Project points
    market_2d = projector.to_2d(market_prices)
    coherent_2d = projector.to_2d(coherent_prices)
    
    # Plot points
    ax.scatter(*market_2d, c='red', s=200, zorder=5, label=f'Market (sum={market_prices.sum():.3f})')
    ax.scatter(*coherent_2d, c='green', s=200, zorder=5, label=f'Coherent (sum={coherent_prices.sum():.3f})')
    
    # Draw arrow from market to coherent
    ax.annotate('', xy=coherent_2d, xytext=market_2d,
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(f'Arbitrage #{index}: KL={opp.kl_divergence:.4f}, Profit=${opp.expected_profit:.4f}\n{opp.timestamp}')
    ax.axis('off')
    
    output_path = output_dir / f"simplex_{index:02d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def run_backtest_with_viz(
    markets: list[TournamentMarket],
    tournament_name: str,
    output_dir: str = "/root/combarbbot/viz_output",
    kl_threshold: float = 0.001,
    interval_minutes: int = 30,
    max_ticks: int = 200,
    max_viz: int = 10,
) -> list[ArbitrageOpportunity]:
    """Run backtest and generate visualizations."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    data_dir = DEFAULT_DATA_DIR
    trades_files = [f for f in glob.glob(f"{data_dir}/trades/*.parquet") if not f.split("/")[-1].startswith("._")]
    blocks_files = [f for f in glob.glob(f"{data_dir}/blocks/*.parquet") if not f.split("/")[-1].startswith("._")]
    
    conn = duckdb.connect()
    
    token_to_market = {}
    for m in markets:
        token_to_market[m.yes_token_id] = (m.market_id, "yes")
        token_to_market[m.no_token_id] = (m.market_id, "no")
    
    all_tokens = []
    for m in markets:
        all_tokens.extend([m.yes_token_id, m.no_token_id])
    
    token_placeholders = ",".join(f"'{t}'" for t in all_tokens)
    
    logger.info("Loading trades...")
    query = f"""
        SELECT t.block_number, t.maker_asset_id, t.taker_asset_id, t.maker_amount, t.taker_amount, b.timestamp
        FROM read_parquet({trades_files}) t
        LEFT JOIN read_parquet({blocks_files}) b ON t.block_number = b.block_number
        WHERE t.maker_asset_id IN ({token_placeholders}) OR t.taker_asset_id IN ({token_placeholders})
        ORDER BY t.block_number
    """
    trades = conn.execute(query).fetchall()
    conn.close()
    
    logger.info("Found %d trades", len(trades))
    
    market_ids = [m.market_id for m in markets]
    relationship_graph = build_partition_graph(market_ids, tournament_name)
    config = OptimizationConfig(max_iterations=100, tolerance=1e-6, step_mode="adaptive")
    
    n_markets = len(markets)
    current_prices = {m.market_id: 1.0 / n_markets for m in markets}
    
    opportunities = []
    interval = timedelta(minutes=interval_minutes)
    last_snapshot_time = None
    tick_count = 0
    
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
                    current_prices[mid] = max(0.001, min(0.999, price))
            elif taker_asset in token_to_market:
                mid, side = token_to_market[taker_asset]
                price = maker_amt / (maker_amt + taker_amt)
                if side == "yes":
                    current_prices[mid] = max(0.001, min(0.999, price))
        
        if last_snapshot_time is None or ts - last_snapshot_time >= interval:
            tick_count += 1
            if tick_count > max_ticks:
                break
            
            list_prices = {mid: [p, 1.0 - p] for mid, p in current_prices.items()}
            
            try:
                result = find_marginal_arbitrage(market_prices=list_prices, relationships=relationship_graph, config=config)
                
                if result.kl_divergence >= kl_threshold:
                    coherent = {mid: prices[0] for mid, prices in result.coherent_market_prices.items()}
                    
                    directions = {}
                    expected_profit = 0.0
                    for mid in market_ids:
                        market_p = current_prices[mid]
                        coherent_p = coherent.get(mid, market_p)
                        diff = coherent_p - market_p
                        if diff > 0.005:
                            directions[mid] = "BUY"
                            expected_profit += diff
                        elif diff < -0.005:
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
                logger.warning("Solver error: %s", e)
            
            last_snapshot_time = ts
            
            if tick_count % 50 == 0:
                logger.info("Tick %d, %d opportunities", tick_count, len(opportunities))
    
    # Generate visualizations for top opportunities
    if HAS_VIZ and opportunities:
        logger.info("Generating %d simplex visualizations...", min(max_viz, len(opportunities)))
        top_opps = sorted(opportunities, key=lambda x: -x.kl_divergence)[:max_viz]
        
        for i, opp in enumerate(top_opps):
            viz_path = generate_simplex_plot(opp, markets, output_path, i + 1)
            if viz_path:
                logger.info("Saved: %s", viz_path)
    
    return opportunities


# NBA MVP 2023-24 markets
NBA_MVP = [
    TournamentMarket(market_id="500334", question="Will Luka Doncic win the 2023-24 NBA MVP?",
        yes_token_id="59142792435409296006151867236844804416169458214931096817238930512418420167705",
        no_token_id="9954333019185024376178629637224124315267626795800410439102371672117401535100",
        volume=198380, short_name="Doncic"),
    TournamentMarket(market_id="500332", question="Will Nikola Jokic win the 2023-24 NBA MVP?",
        yes_token_id="54185570520315123143574021744262228127280133590162961349055274706697137884202",
        no_token_id="51983709602471097643449718644005474958698139280907153982946676738710811168329",
        volume=136594, short_name="Jokic"),
    TournamentMarket(market_id="500333", question="Will Shai Gilgeous-Alexander win the 2023-24 NBA MVP?",
        yes_token_id="88586068497344261946583629458191787994732926164797543051008907601591537452744",
        no_token_id="40316406143905190163577472682455096944049904839276779271633701019560246072593",
        volume=122811, short_name="SGA"),
    TournamentMarket(market_id="500336", question="Will Jayson Tatum win the 2023-24 NBA MVP?",
        yes_token_id="27660163373076269165226948560953005319649928690845268572528611445359346608634",
        no_token_id="98602612519958663144470884557859411904514251671642749404699562385518218276737",
        volume=43708, short_name="Tatum"),
    TournamentMarket(market_id="500335", question="Will Giannis Antetokounmpo win the 2023-24 NBA MVP?",
        yes_token_id="46460166632291713003130759836638610022497015297760563574419034249263171978308",
        no_token_id="70101639333220373995958615982024801878915687192687916435031444175868495699508",
        volume=41208, short_name="Giannis"),
    TournamentMarket(market_id="500337", question="Will another player win the 2023-24 NBA MVP?",
        yes_token_id="42590503078799437262704245085332014840264422346317840684744189168655649355795",
        no_token_id="3178053238041626061744657923485820430124069097655966977060875552696368586966",
        volume=43096, short_name="Other"),
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="/root/combarbbot/viz_output")
    parser.add_argument("--max-ticks", type=int, default=200)
    parser.add_argument("--max-viz", type=int, default=10)
    args = parser.parse_args()
    
    print("=" * 70)
    print("NBA MVP 2023-24 COMBINATORIAL ARBITRAGE WITH VISUALIZATIONS")
    print("=" * 70)
    
    opps = run_backtest_with_viz(
        markets=NBA_MVP,
        tournament_name="NBA MVP 2023-24",
        output_dir=args.output_dir,
        max_ticks=args.max_ticks,
        max_viz=args.max_viz,
    )
    
    print(f"\nTotal opportunities: {len(opps)}")
    print(f"Visualizations saved to: {args.output_dir}")
