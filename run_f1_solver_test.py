#!/usr/bin/env python3
"""F1 2026 Combinatorial Arbitrage Test.

Fetches live F1 drivers + constructors markets, discovers cross-event
logical constraints via LLM, runs the solver, and shows arbitrage.

This demonstrates the REAL value of the solver: detecting mispricing
from logical relationships across events (implies, prerequisites)
where constraint violations aren't obvious from a simple sum check.

Usage:
    python3 run_f1_solver_test.py
"""

import sys
import logging
import os

sys.path.insert(0, "/root/combarbbot")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt

from src.data.polymarket.dataset import DatasetBuilder, DatasetSpec, LiveDataset
from src.llm.clustering import ComplexConstraintExtractor
from src.llm.schema import MarketInfo, MarketRelationship
from src.optimizer.schema import MarketCluster, RelationshipGraph
from src.optimizer.frank_wolfe import find_marginal_arbitrage, find_arbitrage
from src.optimizer.schema import OptimizationConfig, MarginalArbitrageResult
from src.arbitrage.extractor import ArbitrageExtractor
from src.visualization.bregman_plot import (
    compute_bregman_analysis,
    plot_bregman_dual_panel,
    plot_single_cluster_summary,
    BregmanAnalysis,
)


# ─── Configuration ──────────────────────────────────────────────

EVENTS = [
    {
        "name": "F1 2026 Drivers' Championship",
        "slugs": ["2026-f1-drivers-champion"],
    },
    {
        "name": "F1 2026 Constructors' Championship",
        "slugs": ["f1-constructors-champion"],
    },
]

SOLVER_CONFIG = OptimizationConfig(
    max_iterations=5000,
    tolerance=1e-6,
    step_mode="line_search",
)

FEE_PER_LEG = 0.01
OUTPUT_DIR = "/root/combarbbot/output"

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("f1_test")
logger.setLevel(logging.INFO)


# ─── Display helpers ────────────────────────────────────────────

def print_header(text: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def print_section(text: str) -> None:
    print(f"\n--- {text} ---")


def print_market_table(name: str, markets: dict, market_ids: list[str]) -> None:
    """Print a formatted table of markets with prices."""
    print(f"\n  {name}")
    print(f"  {'#':<4} {'Market':<40} {'Yes':>8} {'No':>8}")
    print(f"  {'─'*4} {'─'*40} {'─'*8} {'─'*8}")

    yes_sum = 0.0
    for i, mid in enumerate(market_ids, 1):
        market = markets.get(mid)
        if not market:
            continue
        label = (market.group_item_title or market.question)[:40]
        p_yes = market.yes_price
        p_no = market.no_price
        yes_sum += p_yes
        print(f"  {i:<4} {label:<40} {p_yes:>8.4f} {p_no:>8.4f}")

    print(f"  {'─'*4} {'─'*40} {'─'*8}")
    print(f"  {'':4} {'SUM':<40} {yes_sum:>8.4f}")


def print_constraint_table(constraints: list[MarketRelationship], all_markets: dict) -> None:
    """Print discovered constraints with market labels."""
    for i, c in enumerate(constraints, 1):
        from_label = _market_label(c.from_market, all_markets)
        to_label = _market_label(c.to_market, all_markets)
        arrow = "→" if c.type in ("implies", "prerequisite") else "↔"
        print(f"  {i:>3}. {from_label} {arrow} {to_label}")
        print(f"       type={c.type}, confidence={c.confidence:.2f}")
        if c.reasoning:
            print(f"       reason: {c.reasoning[:80]}")


def _market_label(mid: str | None, markets: dict) -> str:
    """Get a short label for a market ID."""
    if not mid:
        return "?"
    market = markets.get(mid)
    if market:
        return (market.group_item_title or market.question)[:30]
    return mid[:20] + "..."


# ─── Visualization helpers ──────────────────────────────────────

def build_bregman_for_event(
    name: str,
    market_ids: list[str],
    all_markets: dict,
    result: MarginalArbitrageResult,
) -> BregmanAnalysis:
    """Build a BregmanAnalysis for one event from solver results."""
    outcomes = []
    market_prices_dict = {}
    coherent_prices_dict = {}

    for mid in market_ids:
        market = all_markets.get(mid)
        if not market or mid not in result.market_prices:
            continue
        label = (market.group_item_title or market.question)[:30]
        outcomes.append(label)
        market_prices_dict[label] = result.market_prices[mid][0]
        coherent_prices_dict[label] = result.coherent_market_prices[mid][0]

    return compute_bregman_analysis(
        cluster_id=name[:20],
        question=name,
        outcomes=outcomes,
        market_prices=market_prices_dict,
        coherent_prices=coherent_prices_dict,
        kl_divergence=result.kl_divergence,
        iterations=result.iterations,
    )


def build_bregman_for_cross_event(
    cross_constraints: list[MarketRelationship],
    all_markets: dict,
    result: MarginalArbitrageResult,
) -> BregmanAnalysis | None:
    """Build a BregmanAnalysis focused on cross-event implies pairs."""
    # Collect unique markets involved in cross-event constraints
    involved_mids = []
    seen = set()
    for c in cross_constraints:
        if c.type in ("implies", "prerequisite"):
            for mid in [c.from_market, c.to_market]:
                if mid and mid not in seen:
                    involved_mids.append(mid)
                    seen.add(mid)

    if not involved_mids:
        return None

    outcomes = []
    market_prices_dict = {}
    coherent_prices_dict = {}

    for mid in involved_mids:
        market = all_markets.get(mid)
        if not market or mid not in result.market_prices:
            continue
        label = (market.group_item_title or market.question)[:30]
        outcomes.append(label)
        market_prices_dict[label] = result.market_prices[mid][0]
        coherent_prices_dict[label] = result.coherent_market_prices[mid][0]

    if len(outcomes) < 2:
        return None

    return compute_bregman_analysis(
        cluster_id="Cross-Event",
        question="Driver → Constructor Implies Constraints",
        outcomes=outcomes,
        market_prices=market_prices_dict,
        coherent_prices=coherent_prices_dict,
        kl_divergence=result.kl_divergence,
        iterations=result.iterations,
    )


def generate_visualizations(
    drivers_ids: list[str],
    constructors_ids: list[str],
    cross_event: list[MarketRelationship],
    all_markets: dict,
    result: MarginalArbitrageResult,
) -> list[str]:
    """Generate all Bregman visualizations and return saved file paths."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved = []

    # 1. Drivers championship — 3-panel summary
    drivers_analysis = build_bregman_for_event(
        "F1 2026 Drivers' Championship",
        drivers_ids, all_markets, result,
    )
    path = os.path.join(OUTPUT_DIR, "f1_drivers_bregman.png")
    fig = plot_single_cluster_summary(drivers_analysis, figsize=(20, 7), save_path=path)
    plt.close(fig)
    saved.append(path)

    # 2. Constructors championship — 3-panel summary
    constructors_analysis = build_bregman_for_event(
        "F1 2026 Constructors' Championship",
        constructors_ids, all_markets, result,
    )
    path = os.path.join(OUTPUT_DIR, "f1_constructors_bregman.png")
    fig = plot_single_cluster_summary(constructors_analysis, figsize=(18, 6), save_path=path)
    plt.close(fig)
    saved.append(path)

    # 3. Cross-event implies — focused dual panel on just the linked markets
    cross_analysis = build_bregman_for_cross_event(cross_event, all_markets, result)
    if cross_analysis:
        path = os.path.join(OUTPUT_DIR, "f1_cross_event_bregman.png")
        fig = plot_bregman_dual_panel(cross_analysis, figsize=(16, 6), save_path=path)
        plt.close(fig)
        saved.append(path)

    return saved


# ─── Main pipeline ──────────────────────────────────────────────

def main() -> int:
    print_header("F1 2026 Combinatorial Arbitrage Test")
    print("  Cross-event arbitrage: drivers x constructors championship")
    print("  Using LLM to discover logical constraints between events")

    builder = DatasetBuilder()

    # ── Step 1: Fetch both F1 events ────────────────────────────
    print_section("Step 1: Fetching live data from Polymarket")

    datasets: list[LiveDataset] = []
    all_markets: dict = {}  # condition_id -> PolymarketMarket
    all_market_ids: list[str] = []

    for event_cfg in EVENTS:
        spec = DatasetSpec(
            name=event_cfg["name"],
            event_slugs=event_cfg["slugs"],
            refresh_prices_from_clob=True,
        )
        try:
            dataset = builder.build(spec)
        except Exception as e:
            print(f"  ERROR fetching {event_cfg['name']}: {e}")
            logger.exception("Dataset build failed")
            return 1

        n = len(dataset.market_group.market_ids)
        print(f"  {event_cfg['name']}: {n} markets loaded")
        datasets.append(dataset)

        # Collect all markets
        for mid in dataset.market_group.market_ids:
            if mid in dataset.markets:
                all_markets[mid] = dataset.markets[mid]
                all_market_ids.append(mid)

    if len(datasets) < 2:
        print("  ERROR: Need both drivers and constructors events")
        return 1

    drivers_dataset = datasets[0]
    constructors_dataset = datasets[1]
    drivers_ids = drivers_dataset.market_group.market_ids
    constructors_ids = constructors_dataset.market_group.market_ids

    print(f"\n  Total: {len(all_market_ids)} markets "
          f"({len(drivers_ids)} drivers + {len(constructors_ids)} constructors)")

    # ── Step 2: Display market prices ───────────────────────────
    print_section("Step 2: Market Prices")
    print_market_table("Drivers Championship", all_markets, drivers_ids)
    print_market_table("Constructors Championship", all_markets, constructors_ids)

    # ── Step 3: Convert to MarketInfo for LLM ───────────────────
    print_section("Step 3: LLM Cross-Event Constraint Discovery")

    market_infos = []
    for mid in all_market_ids:
        market = all_markets[mid]
        market_infos.append(MarketInfo(
            id=mid,
            question=market.group_item_title or market.question,
            outcomes=market.outcomes if market.outcomes else ["Yes", "No"],
        ))

    print(f"  Sending {len(market_infos)} markets to LLM for constraint extraction...")

    # ── Step 4: Discover cross-event constraints via LLM ────────
    extractor = ComplexConstraintExtractor(min_confidence=0.7)

    try:
        cross_constraints = extractor.extract_constraints(
            markets=market_infos,
            theme="F1 2026 Championship: drivers and constructors relationships",
            competition="f1",
        )
    except Exception as e:
        print(f"  ERROR from LLM: {e}")
        logger.exception("LLM constraint extraction failed")
        return 1

    # ── Step 5: Display discovered constraints ──────────────────
    print_section("Step 4: LLM-Discovered Cross-Event Constraints")

    # Separate cross-event from within-event constraints
    drivers_set = set(drivers_ids)
    constructors_set = set(constructors_ids)

    cross_event = []
    within_event = []
    for c in cross_constraints:
        from_in_d = c.from_market in drivers_set
        to_in_d = c.to_market in drivers_set if c.to_market else False
        from_in_c = c.from_market in constructors_set
        to_in_c = c.to_market in constructors_set if c.to_market else False

        if (from_in_d and to_in_c) or (from_in_c and to_in_d):
            cross_event.append(c)
        else:
            within_event.append(c)

    print(f"\n  Cross-event constraints (driver <-> constructor): {len(cross_event)}")
    if cross_event:
        print_constraint_table(cross_event, all_markets)

    print(f"\n  Within-event constraints: {len(within_event)}")
    if within_event:
        print_constraint_table(within_event[:5], all_markets)  # Show first 5
        if len(within_event) > 5:
            print(f"       ... and {len(within_event) - 5} more")

    if not cross_event:
        print("\n  WARNING: No cross-event constraints discovered.")
        print("  The LLM didn't find driver->constructor implications.")
        print("  This may happen if market names aren't clear enough.")

    # ── Step 6: Build combined RelationshipGraph ────────────────
    print_section("Step 5: Building Combined Constraint Graph")

    # Get within-event constraints from the existing datasets
    all_relationships = []

    # Add existing within-event relationships from each dataset
    for dataset in datasets:
        for cluster in dataset.relationships.clusters:
            all_relationships.extend(cluster.relationships)

    # Add LLM-discovered cross-event constraints
    all_relationships.extend(cross_constraints)

    # Build single combined cluster with all markets and all constraints
    combined_cluster = MarketCluster(
        cluster_id="f1-combined",
        theme="F1 2026 Drivers + Constructors Combined",
        market_ids=all_market_ids,
        relationships=all_relationships,
        is_partition=False,  # Cross-event: NOT a simple partition
    )

    combined_graph = RelationshipGraph(clusters=[combined_cluster])

    # Count constraint types
    from collections import Counter
    type_counts = Counter(r.type for r in all_relationships)
    print(f"  Markets: {len(all_market_ids)}")
    print(f"  Total constraints: {len(all_relationships)}")
    for ctype, count in sorted(type_counts.items()):
        print(f"    {ctype}: {count}")

    # ── Step 7: Run solver ──────────────────────────────────────
    print_section("Step 6: Running Frank-Wolfe Solver")

    # Build market prices in list format for the solver
    market_prices = {}
    for mid in all_market_ids:
        market = all_markets[mid]
        prices = list(market.outcome_prices)
        if len(prices) < 2:
            prices = [market.yes_price, 1.0 - market.yes_price]
        market_prices[mid] = prices

    try:
        result = find_marginal_arbitrage(
            market_prices=market_prices,
            relationships=combined_graph,
            config=SOLVER_CONFIG,
        )
    except Exception as e:
        print(f"  ERROR running solver: {e}")
        logger.exception("Solver failed")
        return 1

    print(f"  Converged:         {result.converged}")
    print(f"  Iterations:        {result.iterations}")
    print(f"  KL divergence:     {result.kl_divergence:.6f}")
    print(f"  Duality gap:       {result.duality_gap:.2e}")
    print(f"  Guaranteed profit: {result.guaranteed_profit:.6f}")
    print(f"  Has arbitrage:     {result.has_arbitrage()}")

    # ── Step 8: Show price adjustments ──────────────────────────
    print_section("Step 7: Price Adjustments (Market -> Coherent)")

    print(f"  {'Market':<35} {'Market':>8} {'Coherent':>8} {'Adj':>8} {'Dir':<12}")
    print(f"  {'---'*12} {'---'*3} {'---'*3} {'---'*3} {'---'*4}")

    for mid in all_market_ids:
        market = all_markets.get(mid)
        if not market or mid not in result.market_prices:
            continue

        label = _market_label(mid, all_markets)[:35]
        p_market = result.market_prices[mid][0]
        p_coherent = result.coherent_market_prices[mid][0]
        adj = p_coherent - p_market
        direction = "OVERPRICED" if adj < -0.001 else ("UNDERPRICED" if adj > 0.001 else "fair")

        print(f"  {label:<35} {p_market:>8.4f} {p_coherent:>8.4f} {adj:>+8.4f} {direction:<12}")

    # ── Step 9: Check cross-event constraint violations ─────────
    print_section("Step 8: Cross-Event Constraint Violations")

    violation_count = 0
    for c in cross_event:
        from_label = _market_label(c.from_market, all_markets)
        to_label = _market_label(c.to_market, all_markets)
        p_from = all_markets[c.from_market].yes_price if c.from_market in all_markets else 0.0
        p_to = all_markets[c.to_market].yes_price if c.to_market in all_markets else 0.0

        if c.type in ("implies", "prerequisite"):
            # For implies: P(from) should be <= P(to)
            if p_from > p_to:
                profit = p_from - p_to
                net = profit - (2 * FEE_PER_LEG)
                violation_count += 1
                print(f"  VIOLATION: {from_label} -> {to_label}")
                print(f"    P(from)={p_from:.4f} > P(to)={p_to:.4f}")
                print(f"    Gross profit: {profit:.4f}")
                print(f"    Net profit:   {net:+.4f} ({'PROFITABLE' if net > 0 else 'fees exceed'})")
                print(f"    Trade: SELL {from_label} @ {p_from:.4f}, BUY {to_label} @ {p_to:.4f}")
            else:
                print(f"  OK: {from_label} -> {to_label}")
                print(f"    P(from)={p_from:.4f} <= P(to)={p_to:.4f}")

        elif c.type == "mutually_exclusive":
            total = p_from + p_to
            if total > 1.0:
                profit = total - 1.0
                net = profit - (2 * FEE_PER_LEG)
                violation_count += 1
                print(f"  VIOLATION: {from_label} <-> {to_label} (mutex)")
                print(f"    P(from)+P(to)={total:.4f} > 1.0")
                print(f"    Gross profit: {profit:.4f}")
                print(f"    Net profit:   {net:+.4f}")
            else:
                print(f"  OK: {from_label} <-> {to_label} (mutex, sum={total:.4f})")

    if not cross_event:
        print("  (No cross-event constraints to check)")

    # ── Step 10: Extract executable trades via ArbitrageExtractor ─
    print_section("Step 9: Executable Trades (via Solver)")

    # Use the legacy path for ArbitrageExtractor compatibility
    single_prices = {mid: p[0] for mid, p in market_prices.items()}

    try:
        legacy_result = find_arbitrage(
            market_prices=single_prices,
            relationships=combined_graph,
            config=SOLVER_CONFIG,
        )

        trade_extractor = ArbitrageExtractor(
            min_profit_threshold=0.001,
            fee_per_leg=FEE_PER_LEG,
        )
        trades = trade_extractor.extract_trades(legacy_result)
    except Exception as e:
        print(f"  ERROR extracting trades: {e}")
        logger.exception("Trade extraction failed")
        trades = []

    if trades:
        for i, trade in enumerate(trades, 1):
            print(f"\n  Trade #{i}: {trade.constraint_type}")
            print(f"    Description: {trade.description}")
            print(f"    Legs: {trade.num_legs}")
            print(f"    Locked profit: {trade.locked_profit:.4f}")
            print(f"    Net profit:    {trade.net_profit(FEE_PER_LEG):.4f}")
            print(f"    Positions:")
            for mid, direction in trade.positions.items():
                price = trade.market_prices.get(mid, 0)
                label = _market_label(mid, all_markets)
                print(f"      {direction:>4} {label} @ {price:.4f}")
    else:
        print("  No pairwise trades exceed the profit threshold.")
        if violation_count > 0:
            print("  (But cross-event violations were found above -- check manually)")

    # ── Step 11: Generate visualizations ────────────────────────
    print_section("Step 10: Bregman Visualizations")

    try:
        saved_plots = generate_visualizations(
            drivers_ids=drivers_ids,
            constructors_ids=constructors_ids,
            cross_event=cross_event,
            all_markets=all_markets,
            result=result,
        )
        for path in saved_plots:
            print(f"  Saved: {path}")
    except Exception as e:
        print(f"  ERROR generating visualizations: {e}")
        logger.exception("Visualization failed")

    # ── Summary ─────────────────────────────────────────────────
    print_header("Summary")
    print(f"  Events:              {len(datasets)}")
    print(f"  Total markets:       {len(all_market_ids)} ({len(drivers_ids)} drivers + {len(constructors_ids)} constructors)")
    print(f"  LLM constraints:     {len(cross_constraints)} total ({len(cross_event)} cross-event)")
    print(f"  Solver constraints:  {len(all_relationships)}")
    print(f"  KL divergence:       {result.kl_divergence:.6f}")
    print(f"  Converged:           {result.converged}")
    print(f"  Cross-event violations: {violation_count}")
    print(f"  Executable trades:   {len(trades)}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
