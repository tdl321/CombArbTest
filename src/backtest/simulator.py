"""Walk-Forward Simulation Module (BT-01, BT-02).

Iterates through historical data chronologically using tick-level data,
detects arbitrage opportunities at each time step.

Streamlined for 3+ market partition arbitrage:
- Partition constraints (exhaustive + exclusive) checked algebraically
- Fallback to Frank-Wolfe solver for complex multi-market constraints
- Simple 2-market constraints have been removed

REFACTORED: Now uses ArbitrageExtractor for correct profit calculation
based on violation magnitude, not price-convergence model.
"""

import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Iterator, Optional
import numpy as np

from src.data import (
    MarketLoader,
    TradeLoader,
    BlockLoader,
    CrossMarketIterator,
    CrossMarketSnapshot,
    Market,
)
from src.optimizer import (
    find_arbitrage,
    ArbitrageResult,
    RelationshipGraph,
    MarketCluster,
    OptimizationConfig,
)
from src.llm import MarketInfo
from src.visualization.schema import ArbitrageSignal
from src.arbitrage.extractor import ArbitrageExtractor, ArbitrageTrade

from .schema import (
    ArbitrageOpportunity,
    BacktestConfig,
    SimulationState,
)
from .constraint_checker import (
    PartitionViolation,
    check_partition,
    is_partition_constraint,
    get_partition_market_ids,
    compute_partition_coherent_prices,
    compute_partition_trades,
    format_partition_violation,
)

logger = logging.getLogger(__name__)

# Minimum markets required for combinatorial arbitrage
MIN_PARTITION_MARKETS = 3


class WalkForwardSimulator:
    """Walk-forward simulation engine for arbitrage detection."""

    def __init__(
        self,
        market_loader: MarketLoader,
        trade_loader: TradeLoader,
        block_loader: BlockLoader,
    ):
        self.market_loader = market_loader
        self.trade_loader = trade_loader
        self.block_loader = block_loader
        logger.debug("[SIM] WalkForwardSimulator initialized")

    def _should_process_cluster(self, cluster: MarketCluster) -> bool:
        """Only process clusters with 3+ markets for combinatorial arbitrage.
        
        Binary (2-market) clusters are skipped as they don't represent
        true combinatorial arbitrage opportunities.
        """
        return len(cluster.market_ids) >= MIN_PARTITION_MARKETS

    def run(
        self,
        market_ids: list[str],
        relationship_graph: RelationshipGraph,
        config: BacktestConfig,
    ) -> Iterator[ArbitrageOpportunity]:
        """Run walk-forward simulation.
        
        Args:
            market_ids: List of market IDs to analyze
            relationship_graph: Graph of market relationships
            config: Backtest configuration
            
        Yields:
            ArbitrageOpportunity objects as they are detected
        """
        logger.info(
            "[SIM] Starting simulation: %d markets, %d clusters, violation_threshold=%.4f",
            len(market_ids),
            len(relationship_graph.clusters),
            config.kl_threshold
        )
        start_time = time.time()

        cluster_lookup = self._build_cluster_lookup(relationship_graph)
        logger.debug("[SIM] Built cluster lookup: %d market->cluster mappings", len(cluster_lookup))

        iterator = CrossMarketIterator(
            trade_loader=self.trade_loader,
            block_loader=self.block_loader,
            market_loader=self.market_loader,
            market_ids=market_ids,
        )

        state = SimulationState()
        last_progress_time: Optional[datetime] = None
        clusters_checked = 0
        partition_checks = 0
        solver_checks = 0

        for snapshot in iterator.iter_snapshots(
            start_block=config.start_block,
            end_block=config.end_block,
        ):
            state.ticks_processed += 1
            state.current_block = snapshot.position.block_number
            state.current_log_index = snapshot.position.log_index
            state.current_timestamp = snapshot.timestamp

            if config.progress_interval and state.ticks_processed % config.progress_interval == 0:
                self._report_progress(state, last_progress_time)
                last_progress_time = datetime.now()

            if config.max_ticks and state.ticks_processed >= config.max_ticks:
                logger.info("[SIM] Reached max ticks (%d), stopping", config.max_ticks)
                break

            if not snapshot.has_all_prices():
                continue

            prices = snapshot.get_prices()

            float_prices = {
                mid: float(p) if p is not None else 0.5
                for mid, p in prices.items()
            }

            for cluster in relationship_graph.clusters:
                # Skip clusters with fewer than 3 markets
                # For complex constraints, we allow 2+ markets
                # Only require 3+ for is_partition=True clusters
                if getattr(cluster, 'is_partition', False):
                    if not self._should_process_cluster(cluster):
                        continue
                elif len(cluster.market_ids) < 2:
                    continue

                logger.debug("[SIM] Processing cluster: %s (%d markets, is_partition=%s, relationships=%d)",
                            cluster.cluster_id[:20], len(cluster.market_ids),
                            getattr(cluster, 'is_partition', False), len(cluster.relationships))

                cluster_prices = {
                    mid: float_prices[mid]
                    for mid in cluster.market_ids
                    if mid in float_prices
                }

                min_markets = MIN_PARTITION_MARKETS if getattr(cluster, 'is_partition', False) else 2
                if len(cluster_prices) < min_markets:
                    logger.debug("[SIM] Skipping cluster %s: only %d/%d prices available",
                                cluster.cluster_id[:12], len(cluster_prices), len(cluster.market_ids))
                    continue

                if not cluster.relationships and not getattr(cluster, "is_partition", False):
                    continue

                clusters_checked += 1

                # Convert to dict to avoid llm.schema vs optimizer.schema type conflict
                cluster_graph = RelationshipGraph(
                    clusters=[cluster.model_dump()],
                )

                opportunity = self._check_arbitrage(
                    snapshot=snapshot,
                    cluster=cluster,
                    prices=cluster_prices,
                    graph=cluster_graph,
                    config=config,
                )

                if opportunity is not None:
                    state.opportunities_found += 1
                    net_pnl = opportunity.net_profit(config.transaction_cost)
                    state.update_pnl(net_pnl)
                    
                    logger.debug(
                        "[SIM] Opportunity found: block=%d, cluster=%s, type=%s, net=%.4f",
                        snapshot.position.block_number,
                        cluster.cluster_id[:12],
                        opportunity.trade.constraint_type,
                        net_pnl
                    )
                    yield opportunity

        elapsed = time.time() - start_time
        logger.info(
            "[SIM] Simulation complete: %d ticks, %d opportunities, %.3fs",
            state.ticks_processed,
            state.opportunities_found,
            elapsed
        )
        logger.info(
            "[SIM] Final cumulative PnL: %.4f, max drawdown: %.4f",
            state.cumulative_pnl,
            state.max_drawdown
        )
        logger.debug(
            "[SIM] Stats: clusters_checked=%d, ticks_per_sec=%.1f",
            clusters_checked,
            state.ticks_processed / elapsed if elapsed > 0 else 0
        )

    def run_signals(
        self,
        market_ids: list[str],
        relationship_graph: RelationshipGraph,
        config: BacktestConfig,
    ) -> Iterator[ArbitrageSignal]:
        """Run simulation yielding pure arbitrage signals (no PnL).
        
        Args:
            market_ids: List of market IDs to analyze
            relationship_graph: Graph of market relationships
            config: Backtest configuration
            
        Yields:
            ArbitrageSignal objects as they are detected
        """
        logger.info(
            "[SIM] Starting signal-only simulation: %d markets, %d clusters",
            len(market_ids),
            len(relationship_graph.clusters)
        )
        start_time = time.time()

        iterator = CrossMarketIterator(
            trade_loader=self.trade_loader,
            block_loader=self.block_loader,
            market_loader=self.market_loader,
            market_ids=market_ids,
        )

        state = SimulationState()
        last_progress_time: Optional[datetime] = None

        for snapshot in iterator.iter_snapshots(
            start_block=config.start_block,
            end_block=config.end_block,
        ):
            state.ticks_processed += 1
            state.current_block = snapshot.position.block_number
            state.current_log_index = snapshot.position.log_index
            state.current_timestamp = snapshot.timestamp

            if config.progress_interval and state.ticks_processed % config.progress_interval == 0:
                self._report_progress(state, last_progress_time)
                last_progress_time = datetime.now()

            if config.max_ticks and state.ticks_processed >= config.max_ticks:
                logger.info("[SIM] Reached max ticks (%d), stopping", config.max_ticks)
                break

            if not snapshot.has_all_prices():
                continue

            prices = snapshot.get_prices()

            float_prices = {
                mid: float(p) if p is not None else 0.5
                for mid, p in prices.items()
            }

            for cluster in relationship_graph.clusters:
                # Skip clusters with fewer than 3 markets
                # For complex constraints, we allow 2+ markets
                # Only require 3+ for is_partition=True clusters
                if getattr(cluster, 'is_partition', False):
                    if not self._should_process_cluster(cluster):
                        continue
                elif len(cluster.market_ids) < 2:
                    continue

                logger.debug("[SIM] Processing cluster: %s (%d markets, is_partition=%s, relationships=%d)",
                            cluster.cluster_id[:20], len(cluster.market_ids),
                            getattr(cluster, 'is_partition', False), len(cluster.relationships))

                cluster_prices = {
                    mid: float_prices[mid]
                    for mid in cluster.market_ids
                    if mid in float_prices
                }

                min_markets = MIN_PARTITION_MARKETS if getattr(cluster, 'is_partition', False) else 2
                if len(cluster_prices) < min_markets:
                    logger.debug("[SIM] Skipping cluster %s: only %d/%d prices available",
                                cluster.cluster_id[:12], len(cluster_prices), len(cluster.market_ids))
                    continue

                if not cluster.relationships and not getattr(cluster, "is_partition", False):
                    continue

                # Convert to dict to avoid llm.schema vs optimizer.schema type conflict
                cluster_graph = RelationshipGraph(
                    clusters=[cluster.model_dump()],
                )

                signal = self._check_signal(
                    snapshot=snapshot,
                    cluster=cluster,
                    prices=cluster_prices,
                    graph=cluster_graph,
                    config=config,
                )

                if signal is not None:
                    state.opportunities_found += 1
                    logger.debug(
                        "[SIM] Signal found: block=%d, cluster=%s, kl=%.4f",
                        snapshot.position.block_number,
                        cluster.cluster_id[:12],
                        signal.kl_divergence
                    )
                    yield signal

        elapsed = time.time() - start_time
        logger.info(
            "[SIM] Signal simulation complete: %d ticks, %d signals, %.3fs",
            state.ticks_processed,
            state.opportunities_found,
            elapsed
        )

    def _check_signal(
        self,
        snapshot: CrossMarketSnapshot,
        cluster: MarketCluster,
        prices: dict[str, float],
        graph: RelationshipGraph,
        config: BacktestConfig,
    ) -> Optional[ArbitrageSignal]:
        """Check for arbitrage signal in a cluster (no PnL calculation).
        
        Two-stage detection:
        1. Check if this is a partition constraint (fast, algebraic)
        2. Fallback to solver for complex constraints
        """
        try:
            # Stage 1: Check partition constraints (algebraic, fast)
            if is_partition_constraint(cluster):
                partition_ids = get_partition_market_ids(cluster)
                violation = check_partition(partition_ids, prices)
                
                if violation:
                    logger.info(
                        "[SIM] Partition signal: cluster=%s, sum=%.4f, %s",
                        cluster.cluster_id[:12],
                        violation.total,
                        violation.direction
                    )
                    return self._create_partition_signal(
                        snapshot=snapshot,
                        cluster=cluster,
                        violation=violation,
                        config=config,
                    )
            
            # Stage 2: Fallback to solver for non-partition complex constraints
            return self._check_signal_with_solver(
                snapshot=snapshot,
                cluster=cluster,
                prices=prices,
                graph=graph,
                config=config,
            )

        except Exception as e:
            logger.warning(
                "[SIM] Error checking signal for cluster %s: %s",
                cluster.cluster_id[:12],
                e
            )
            logger.debug("[SIM] Signal check error details", exc_info=True)
            return None

    def _create_partition_signal(
        self,
        snapshot: CrossMarketSnapshot,
        cluster: MarketCluster,
        violation: PartitionViolation,
        config: BacktestConfig,
    ) -> Optional[ArbitrageSignal]:
        """Create an arbitrage signal from a partition violation."""
        # Calculate coherent prices (normalized to sum to 1)
        coherent_prices = violation.coherent_prices
        prices = violation.prices
        
        # KL divergence for partition is related to the sum deviation
        kl_divergence = abs(violation.violation_amount)
        
        if kl_divergence < config.kl_threshold:
            logger.debug(
                "[SIM] Partition signal below threshold: kl=%.4f < %.4f",
                kl_divergence,
                config.kl_threshold
            )
            return None
        
        # Calculate edge magnitude (L2 distance)
        markets = list(prices.keys())
        market_vec = np.array([prices[m] for m in markets])
        coherent_vec = np.array([coherent_prices[m] for m in markets])
        edge_magnitude = float(np.linalg.norm(market_vec - coherent_vec))
        
        # Compute direction for each market
        direction = {
            m: coherent_prices[m] - prices[m]
            for m in markets
        }
        
        # Violation description
        violation_desc = "Partition sum=%.4f, %s by %.4f" % (
            violation.total,
            violation.direction,
            abs(violation.violation_amount)
        )
        
        return ArbitrageSignal(
            timestamp=snapshot.timestamp or datetime.now(),
            cluster_id=cluster.cluster_id,
            markets=markets,
            constraint_type="partition",
            market_prices=prices,
            coherent_prices=coherent_prices,
            edge_magnitude=edge_magnitude,
            kl_divergence=kl_divergence,
            direction=direction,
            constraint_violation=violation_desc,
            block_number=snapshot.position.block_number,
            detection_method="partition",
        )

    def _check_signal_with_solver(
        self,
        snapshot: CrossMarketSnapshot,
        cluster: MarketCluster,
        prices: dict[str, float],
        graph: RelationshipGraph,
        config: BacktestConfig,
    ) -> Optional[ArbitrageSignal]:
        """Check for arbitrage signal using the Frank-Wolfe solver."""
        logger.debug("[SIM] SOLVER SIGNAL PATH: cluster=%s (%d markets)",
                    cluster.cluster_id[:20], len(prices))

        # Use adaptive solver for 9-10x speedup
        opt_config = OptimizationConfig(
            step_mode=config.solver_mode,
            max_iterations=100,
            tolerance=1e-4,
        )
        result = find_arbitrage(
            market_prices=prices,
            relationships=graph,
            config=opt_config,
        )

        if result.kl_divergence < config.kl_threshold:
            return None

        markets = list(prices.keys())
        market_vec = np.array([prices[m] for m in markets])
        coherent_vec = np.array([result.coherent_prices[m] for m in markets])
        edge_magnitude = float(np.linalg.norm(market_vec - coherent_vec))
        
        direction = {
            m: result.coherent_prices[m] - prices[m]
            for m in markets
        }
        
        constraint_type = _infer_constraint_type(cluster.relationships)
        violation_desc = _format_violation_result(result)

        logger.debug(
            "[SIM] Solver signal: cluster=%s, kl=%.4f, type=%s",
            cluster.cluster_id[:12],
            result.kl_divergence,
            constraint_type
        )

        return ArbitrageSignal(
            timestamp=snapshot.timestamp or datetime.now(),
            cluster_id=cluster.cluster_id,
            markets=markets,
            constraint_type=constraint_type,
            market_prices=prices,
            coherent_prices=result.coherent_prices,
            edge_magnitude=edge_magnitude,
            kl_divergence=result.kl_divergence,
            direction=direction,
            constraint_violation=violation_desc,
            block_number=snapshot.position.block_number,
            detection_method="solver",
        )

    def _build_cluster_lookup(
        self,
        graph: RelationshipGraph,
    ) -> dict[str, MarketCluster]:
        """Build lookup from market_id to cluster."""
        lookup = {}
        for cluster in graph.clusters:
            for market_id in cluster.market_ids:
                lookup[market_id] = cluster
        return lookup

    def _extract_arbitrage(
        self,
        result: ArbitrageResult,
        snapshot: CrossMarketSnapshot,
        cluster: MarketCluster,
        config: BacktestConfig,
    ) -> Optional[ArbitrageOpportunity]:
        """Extract arbitrage trade from solver result using ArbitrageExtractor.
        
        This is the NEW approach that calculates profit correctly based on
        violation magnitude, not price-convergence.
        """
        extractor = ArbitrageExtractor(
            min_profit_threshold=config.min_profit,
            fee_per_leg=config.transaction_cost,
        )
        
        trades = extractor.extract_trades(result)
        
        if not trades:
            logger.debug(
                "[SIM] No trades extracted: cluster=%s, violations=%d",
                cluster.cluster_id[:12],
                len(result.constraints_violated)
            )
            return None
        
        # Take best trade by locked_profit
        best_trade = max(trades, key=lambda t: t.locked_profit)
        
        # Check profitability threshold after fees
        net = best_trade.net_profit(config.transaction_cost)
        if net < config.min_profit:
            logger.debug(
                "[SIM] Trade below min_profit: net=%.4f < %.4f",
                net,
                config.min_profit
            )
            return None
        
        logger.debug(
            "[SIM] Extracted arbitrage: type=%s, locked=%.4f, net=%.4f, legs=%d",
            best_trade.constraint_type,
            best_trade.locked_profit,
            net,
            best_trade.num_legs
        )
        
        return ArbitrageOpportunity(
            timestamp=snapshot.timestamp or datetime.now(),
            block_number=snapshot.position.block_number,
            cluster_id=cluster.cluster_id,
            trade=best_trade,
            solver_result=result,
            detection_method="solver",
        )

    def _check_arbitrage(
        self,
        snapshot: CrossMarketSnapshot,
        cluster: MarketCluster,
        prices: dict[str, float],
        graph: RelationshipGraph,
        config: BacktestConfig,
    ) -> Optional[ArbitrageOpportunity]:
        """Check for arbitrage opportunity in a cluster.
        
        Two-stage detection:
        1. Check partition constraints (fast, algebraic)
        2. Fallback to solver for complex constraints
        
        REFACTORED: Now uses _extract_arbitrage with ArbitrageExtractor
        for correct profit calculation.
        """
        try:
            # Stage 1: Check partition constraints (algebraic, fast)
            if is_partition_constraint(cluster):
                partition_ids = get_partition_market_ids(cluster)
                partition_violation = check_partition(partition_ids, prices)
                
                if partition_violation:
                    logger.debug(
                        "[SIM] Partition detected: cluster=%s, sum=%.4f, %s",
                        cluster.cluster_id[:12],
                        partition_violation.total,
                        partition_violation.direction
                    )
                    return self._create_partition_opportunity(
                        snapshot=snapshot,
                        cluster=cluster,
                        violation=partition_violation,
                        config=config,
                    )
            
            # Stage 2: Fallback to solver for non-partition constraints
            return self._check_arbitrage_with_solver(
                snapshot=snapshot,
                cluster=cluster,
                prices=prices,
                graph=graph,
                config=config,
            )

        except Exception as e:
            logger.warning(
                "[SIM] Error checking arbitrage for cluster %s: %s",
                cluster.cluster_id[:12],
                e
            )
            logger.debug("[SIM] Arbitrage check error details", exc_info=True)
            return None

    def _create_partition_opportunity(
        self,
        snapshot: CrossMarketSnapshot,
        cluster: MarketCluster,
        violation: PartitionViolation,
        config: BacktestConfig,
    ) -> Optional[ArbitrageOpportunity]:
        """Create an arbitrage opportunity from a partition violation.
        
        REFACTORED: Now creates ArbitrageTrade directly from the violation.
        The profit is the violation magnitude (|1 - sum|), not price-convergence.
        """
        # Determine positions based on violation direction
        if violation.direction == "underpriced":
            # Sum < 1: Buy all -> pay sum, receive 1.0
            positions = {m: "BUY" for m in violation.prices}
            direction_word = "BUY ALL"
        else:
            # Sum > 1: Sell all -> receive sum, pay 1.0
            positions = {m: "SELL" for m in violation.prices}
            direction_word = "SELL ALL"
        
        # Locked profit = violation magnitude
        locked_profit = abs(violation.violation_amount)
        
        # Create the ArbitrageTrade
        trade = ArbitrageTrade(
            constraint_type="partition",
            positions=positions,
            violation_amount=locked_profit,
            locked_profit=locked_profit,
            market_prices=violation.prices,
            description="Partition arb: %s at sum=%.4f, profit=%.4f" % (
                direction_word, violation.total, locked_profit
            ),
        )
        
        # Check profitability after fees
        net = trade.net_profit(config.transaction_cost)
        if net < config.min_profit:
            logger.debug(
                "[SIM] Partition trade below threshold: net=%.4f < %.4f",
                net,
                config.min_profit
            )
            return None
        
        logger.info(
            "[SIM] Partition arbitrage: cluster=%s, sum=%.4f, %s, locked=%.4f, net=%.4f",
            cluster.cluster_id[:12],
            violation.total,
            violation.direction,
            trade.locked_profit,
            net
        )
        
        return ArbitrageOpportunity(
            timestamp=snapshot.timestamp or datetime.now(),
            block_number=snapshot.position.block_number,
            cluster_id=cluster.cluster_id,
            trade=trade,
            solver_result=None,  # No solver used for partition
            detection_method="partition",
        )

    def _create_partition_opportunity_v2(
        self,
        snapshot: CrossMarketSnapshot,
        cluster: MarketCluster,
        violation: PartitionViolation,
        config: BacktestConfig,
    ) -> Optional[ArbitrageOpportunity]:
        """Create opportunity from partition violation using ArbitrageTrade logic.
        
        This creates the correct hedged positions where profit = |1 - sum(prices)|.
        """
        from src.arbitrage.extractor import ArbitrageTrade
        
        # Determine direction
        if violation.direction == "underpriced":
            positions = {m: "BUY" for m in violation.prices}
            direction_word = "BUY ALL"
        else:
            positions = {m: "SELL" for m in violation.prices}
            direction_word = "SELL ALL"
        
        locked_profit = abs(violation.violation_amount)
        
        trade = ArbitrageTrade(
            constraint_type="partition",
            positions=positions,
            violation_amount=locked_profit,
            locked_profit=locked_profit,
            market_prices=dict(violation.prices),
            description="Partition arb: {} at {:.4f}".format(direction_word, violation.total),
        )
        
        # Check profitability after fees
        net_profit = trade.net_profit(config.transaction_cost)
        if net_profit < config.min_profit:
            logger.debug(
                "[SIM] Partition v2 below threshold: net=%.4f < %.4f",
                net_profit,
                config.min_profit
            )
            return None
        
        logger.info(
            "[SIM] Partition v2 arbitrage: cluster=%s, sum=%.4f, %s, locked=%.4f, net=%.4f",
            cluster.cluster_id[:12],
            violation.total,
            violation.direction,
            locked_profit,
            net_profit
        )
        
        return ArbitrageOpportunity(
            timestamp=snapshot.timestamp or datetime.now(),
            block_number=snapshot.position.block_number,
            cluster_id=cluster.cluster_id,
            trade=trade,
            solver_result=None,
            detection_method="partition_v2",
        )


    def _check_arbitrage_with_solver(
        self,
        snapshot: CrossMarketSnapshot,
        cluster: MarketCluster,
        prices: dict[str, float],
        graph: RelationshipGraph,
        config: BacktestConfig,
    ) -> Optional[ArbitrageOpportunity]:
        """Check for arbitrage using the Frank-Wolfe solver (for complex constraints).
        
        REFACTORED: Now uses _extract_arbitrage with ArbitrageExtractor
        for correct profit calculation based on violation magnitude.
        """
        logger.info("[SIM] SOLVER PATH: Calling Frank-Wolfe for cluster=%s (%d markets)",
                   cluster.cluster_id[:20], len(prices))
        logger.debug("[SIM] Solver input prices: %s", {k: f"{v:.4f}" for k, v in prices.items()})

        # Use adaptive solver for 9-10x speedup
        opt_config = OptimizationConfig(
            step_mode=config.solver_mode,
            max_iterations=100,
            tolerance=1e-4,
        )
        result = find_arbitrage(
            market_prices=prices,
            relationships=graph,
            config=opt_config,
        )

        logger.info("[SIM] SOLVER RESULT: kl=%.6f, converged=%s, iters=%d",
                   result.kl_divergence, result.converged, result.iterations)
        logger.debug("[SIM] Coherent prices: %s", {k: f"{v:.4f}" for k, v in result.coherent_prices.items()})

        if result.kl_divergence < config.kl_threshold:
            logger.debug("[SIM] Below threshold: kl=%.6f < %.6f", result.kl_divergence, config.kl_threshold)
            return None

        logger.debug(
            "[SIM] Solver found divergence: cluster=%s, kl=%.4f, violations=%d",
            cluster.cluster_id[:12],
            result.kl_divergence,
            len(result.constraints_violated)
        )

        # Use the new extraction method
        return self._extract_arbitrage(
            result=result,
            snapshot=snapshot,
            cluster=cluster,
            config=config,
        )

    def _report_progress(
        self,
        state: SimulationState,
        last_time: Optional[datetime],
    ) -> None:
        """Report simulation progress."""
        ts_str = state.current_timestamp.isoformat() if state.current_timestamp else "N/A"

        logger.info(
            "[SIM] Progress: %d ticks | Block %d | Opps: %d | PnL: %.4f | Time: %s",
            state.ticks_processed,
            state.current_block,
            state.opportunities_found,
            state.cumulative_pnl,
            ts_str
        )


def _infer_constraint_type(relationships: list) -> str:
    """Infer primary constraint type from relationships."""
    types = [r.type for r in relationships]
    if "exhaustive" in types:
        if "mutually_exclusive" in types:
            return "partition"
        return "exhaustive"
    if types:
        return types[0]
    return "unknown"


def _format_violation_result(result) -> str:
    """Format constraint violation from ArbitrageResult as human-readable string."""
    if result.constraints_violated:
        return "; ".join(cv.description for cv in result.constraints_violated[:2])
    return "Violation: %.4f" % result.kl_divergence


def run_simulation(
    market_ids: list[str],
    relationship_graph: RelationshipGraph,
    market_loader: MarketLoader,
    trade_loader: TradeLoader,
    block_loader: BlockLoader,
    config: BacktestConfig,
) -> list[ArbitrageOpportunity]:
    """Convenience function to run a simulation."""
    logger.info(
        "[SIM] run_simulation: %d markets, blocks %s-%s",
        len(market_ids),
        config.start_block,
        config.end_block
    )
    simulator = WalkForwardSimulator(
        market_loader=market_loader,
        trade_loader=trade_loader,
        block_loader=block_loader,
    )

    return list(simulator.run(
        market_ids=market_ids,
        relationship_graph=relationship_graph,
        config=config,
    ))


def run_signal_simulation(
    market_ids: list[str],
    relationship_graph: RelationshipGraph,
    market_loader: MarketLoader,
    trade_loader: TradeLoader,
    block_loader: BlockLoader,
    config: BacktestConfig,
) -> list[ArbitrageSignal]:
    """Convenience function to run a signal-only simulation."""
    logger.info(
        "[SIM] run_signal_simulation: %d markets, blocks %s-%s",
        len(market_ids),
        config.start_block,
        config.end_block
    )
    simulator = WalkForwardSimulator(
        market_loader=market_loader,
        trade_loader=trade_loader,
        block_loader=block_loader,
    )

    return list(simulator.run_signals(
        market_ids=market_ids,
        relationship_graph=relationship_graph,
        config=config,
    ))


def run_backtest_with_report(
    market_ids: list[str],
    relationship_graph: RelationshipGraph,
    market_loader: MarketLoader,
    trade_loader: TradeLoader,
    block_loader: BlockLoader,
    config: BacktestConfig,
    output_dir: str,
) -> "BacktestOutput":
    """Run complete backtest with report generation.
    
    This is the main entry point for running a backtest that produces
    a complete report with visualizations.
    
    Args:
        market_ids: List of market IDs to analyze
        relationship_graph: Graph of market relationships/constraints
        market_loader: Loader for market data
        trade_loader: Loader for trade history
        block_loader: Loader for block data
        config: Backtest configuration
        output_dir: Directory to save reports and visualizations
        
    Returns:
        BacktestOutput with complete results
    """
    from .schema import BacktestOutput
    from .report_generator import generate_full_report
    from ..arbitrage.extractor import ArbitrageTrade
    
    logger.info("[BACKTEST] Starting backtest with report generation")
    logger.info(
        "[BACKTEST] Config: markets=%d, violation_threshold=%.4f, tx_cost=%.4f, output=%s",
        len(market_ids),
        config.kl_threshold,
        config.transaction_cost,
        output_dir
    )
    
    start_time = time.time()
    
    # Run simulation
    simulator = WalkForwardSimulator(market_loader, trade_loader, block_loader)
    opportunities = list(simulator.run(market_ids, relationship_graph, config))
    
    sim_elapsed = time.time() - start_time
    logger.info(
        "[BACKTEST] Simulation complete: %d opportunities in %.3fs",
        len(opportunities),
        sim_elapsed
    )
    
    # Extract trades for report
    # Opportunities now have .trade (ArbitrageTrade) directly
    trades = []
    for opp in opportunities:
        # opp.trade is already an ArbitrageTrade
        trades.append(opp.trade)
    
    # Determine period
    if opportunities:
        period_start = min(o.timestamp for o in opportunities)
        period_end = max(o.timestamp for o in opportunities)
        logger.debug(
            "[BACKTEST] Period: %s to %s",
            period_start.isoformat(),
            period_end.isoformat()
        )
    else:
        period_start = None
        period_end = None
        logger.debug("[BACKTEST] No opportunities, period undefined")
    
    # Generate report
    logger.info("[BACKTEST] Generating report...")
    report, report_text, viz_files = generate_full_report(
        trades=trades,
        output_dir=output_dir,
        period_start=period_start,
        period_end=period_end,
        markets_analyzed=len(market_ids),
        clusters_monitored=len(relationship_graph.clusters),
        fee_per_leg=config.transaction_cost,
    )
    
    # Construct output
    from pathlib import Path
    from .schema import BacktestReport
    
    report_text_path = str(Path(output_dir) / "backtest_report.txt")
    
    # Convert ArbitrageBacktestReport to BacktestReport for schema compatibility
    backtest_report = BacktestReport(
        start_date=report.period_start or datetime.now(),
        end_date=report.period_end or datetime.now(),
        duration_hours=((report.period_end - report.period_start).total_seconds() / 3600
                       if report.period_start and report.period_end else 0.0),
        markets_analyzed=report.markets_analyzed,
        clusters_found=report.clusters_monitored,
        total_opportunities=report.total_opportunities,
        opportunities_per_hour=(report.total_opportunities / max(1, 
            (report.period_end - report.period_start).total_seconds() / 3600)
            if report.period_start and report.period_end else 0.0),
        gross_pnl=report.total_locked_profit,
        net_pnl=report.total_net_profit,
        transaction_costs=report.total_fees,
        win_count=report.profitable_after_fees,
        loss_count=report.unprofitable_after_fees,
        win_rate=report.profitable_after_fees / max(1, report.total_opportunities),
        avg_violation=report.avg_profit_per_opp,
        by_constraint_type={k: v.count for k, v in report.by_constraint_type.items()},
        min_profit_threshold=config.min_profit,
        transaction_cost_rate=config.transaction_cost,
    )
    
    total_elapsed = time.time() - start_time
    logger.info(
        "[BACKTEST] Complete: %d opps, net_pnl=%.4f, %d viz files, %.3fs total",
        len(opportunities),
        report.total_net_profit,
        len(viz_files),
        total_elapsed
    )
    
    return BacktestOutput(
        run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
        run_timestamp=datetime.now(),
        config=config,
        report=backtest_report,
        opportunities=opportunities,
        report_text_path=report_text_path,
        visualization_paths=viz_files,
    )
