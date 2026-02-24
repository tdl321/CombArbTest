"""Walk-Forward Simulation Module (BT-01, BT-02).

Iterates through historical data chronologically using tick-level data,
detects arbitrage opportunities at each time step.
"""

from datetime import datetime
from decimal import Decimal
from typing import Iterator, Optional
import logging

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
)
from src.llm import MarketInfo

from .schema import (
    ArbitrageOpportunity,
    BacktestConfig,
    SimulationState,
)
from .pnl import calculate_opportunity_pnl

logger = logging.getLogger(__name__)


class WalkForwardSimulator:
    """Walk-forward simulation engine for arbitrage detection.
    
    Iterates through historical tick data chronologically and detects
    arbitrage opportunities using cached LLM relationship graphs.
    
    Usage:
        simulator = WalkForwardSimulator(
            market_loader=market_loader,
            trade_loader=trade_loader,
            block_loader=block_loader,
        )
        
        # Run simulation with cached relationship graph
        opportunities = list(simulator.run(
            market_ids=["market1", "market2"],
            relationship_graph=graph,
            config=BacktestConfig(...),
        ))
    """
    
    def __init__(
        self,
        market_loader: MarketLoader,
        trade_loader: TradeLoader,
        block_loader: BlockLoader,
    ):
        """Initialize the simulator.
        
        Args:
            market_loader: Loader for market metadata
            trade_loader: Loader for trade history
            block_loader: Loader for block timestamps
        """
        self.market_loader = market_loader
        self.trade_loader = trade_loader
        self.block_loader = block_loader
    
    def run(
        self,
        market_ids: list[str],
        relationship_graph: RelationshipGraph,
        config: BacktestConfig,
    ) -> Iterator[ArbitrageOpportunity]:
        """Run walk-forward simulation.
        
        Iterates through tick data and yields arbitrage opportunities.
        Uses the relationship graph cached from LLM (do not re-call per tick).
        
        Args:
            market_ids: List of market IDs to simulate
            relationship_graph: Pre-computed relationship graph from LLM
            config: Backtest configuration
            
        Yields:
            ArbitrageOpportunity for each detected opportunity
        """
        logger.info(f"Starting simulation with {len(market_ids)} markets")
        
        # Build cluster lookup for fast access
        cluster_lookup = self._build_cluster_lookup(relationship_graph)
        
        # Create cross-market iterator
        iterator = CrossMarketIterator(
            trade_loader=self.trade_loader,
            block_loader=self.block_loader,
            market_loader=self.market_loader,
            market_ids=market_ids,
        )
        
        # Track simulation state
        state = SimulationState()
        last_progress_time: Optional[datetime] = None
        
        # Iterate through snapshots
        for snapshot in iterator.iter_snapshots(
            start_block=config.start_block,
            end_block=config.end_block,
        ):
            state.ticks_processed += 1
            state.current_block = snapshot.position.block_number
            state.current_log_index = snapshot.position.log_index
            state.current_timestamp = snapshot.timestamp
            
            # Progress reporting
            if config.progress_interval and state.ticks_processed % config.progress_interval == 0:
                self._report_progress(state, last_progress_time)
                last_progress_time = datetime.now()
            
            # Check max ticks
            if config.max_ticks and state.ticks_processed >= config.max_ticks:
                logger.info(f"Reached max ticks ({config.max_ticks}), stopping")
                break
            
            # Skip if not all prices available
            if not snapshot.has_all_prices():
                continue
            
            # Get current prices
            prices = snapshot.get_prices()
            
            # Convert Decimal prices to float
            float_prices = {
                mid: float(p) if p is not None else 0.5
                for mid, p in prices.items()
            }
            
            # Check each cluster for arbitrage
            for cluster in relationship_graph.clusters:
                # Get prices for markets in this cluster
                cluster_prices = {
                    mid: float_prices[mid]
                    for mid in cluster.market_ids
                    if mid in float_prices
                }
                
                if len(cluster_prices) < 2:
                    # Need at least 2 markets in cluster
                    continue
                
                # Skip if no relationships defined
                if not cluster.relationships:
                    continue
                
                # Create a sub-graph for just this cluster
                cluster_graph = RelationshipGraph(
                    clusters=[cluster],
                )
                
                # Run arbitrage detection
                opportunity = self._check_arbitrage(
                    snapshot=snapshot,
                    cluster=cluster,
                    prices=cluster_prices,
                    graph=cluster_graph,
                    config=config,
                )
                
                if opportunity is not None:
                    state.opportunities_found += 1
                    state.update_pnl(opportunity.net_profit)
                    yield opportunity
        
        logger.info(
            f"Simulation complete: processed {state.ticks_processed} ticks, "
            f"found {state.opportunities_found} opportunities"
        )
    
    def _build_cluster_lookup(
        self,
        graph: RelationshipGraph,
    ) -> dict[str, MarketCluster]:
        """Build lookup from market_id to cluster.
        
        Args:
            graph: Relationship graph
            
        Returns:
            Dict mapping market_id to its cluster
        """
        lookup = {}
        for cluster in graph.clusters:
            for market_id in cluster.market_ids:
                lookup[market_id] = cluster
        return lookup
    
    def _check_arbitrage(
        self,
        snapshot: CrossMarketSnapshot,
        cluster: MarketCluster,
        prices: dict[str, float],
        graph: RelationshipGraph,
        config: BacktestConfig,
    ) -> Optional[ArbitrageOpportunity]:
        """Check for arbitrage opportunity in a cluster.
        
        Args:
            snapshot: Current market snapshot
            cluster: The cluster to check
            prices: Current prices for cluster markets
            graph: Relationship graph for the cluster
            config: Backtest configuration
            
        Returns:
            ArbitrageOpportunity if found, None otherwise
        """
        try:
            # Run optimizer to find arbitrage-free prices
            result = find_arbitrage(
                market_prices=prices,
                relationships=graph,
            )
            
            # Check if KL divergence exceeds threshold
            if result.kl_divergence < config.kl_threshold:
                return None
            
            # Calculate PnL
            gross_profit, net_profit, trade_directions = calculate_opportunity_pnl(
                market_prices=prices,
                coherent_prices=result.coherent_prices,
                kl_divergence=result.kl_divergence,
                transaction_cost_rate=config.transaction_cost,
            )
            
            # Check minimum profit threshold
            if gross_profit < config.min_profit:
                return None
            
            # Get constraint violation descriptions
            constraint_descriptions = [
                cv.description for cv in result.constraints_violated
            ]
            
            return ArbitrageOpportunity(
                timestamp=snapshot.timestamp or datetime.now(),
                position=(snapshot.position.block_number, snapshot.position.log_index),
                cluster_id=cluster.cluster_id,
                market_prices=prices,
                coherent_prices=result.coherent_prices,
                kl_divergence=result.kl_divergence,
                constraints_violated=constraint_descriptions,
                theoretical_profit=gross_profit,
                net_profit=net_profit,
                trade_direction=trade_directions,
            )
            
        except Exception as e:
            logger.warning(f"Error checking arbitrage for cluster {cluster.cluster_id}: {e}")
            return None
    
    def _report_progress(
        self,
        state: SimulationState,
        last_time: Optional[datetime],
    ) -> None:
        """Report simulation progress.
        
        Args:
            state: Current simulation state
            last_time: Time of last progress report
        """
        ticks_per_sec = 0.0
        if last_time:
            elapsed = (datetime.now() - last_time).total_seconds()
            if elapsed > 0:
                # Use progress interval since that is ticks since last report
                ticks_per_sec = state.ticks_processed / elapsed  # Approximate
        
        ts_str = state.current_timestamp.isoformat() if state.current_timestamp else "N/A"
        
        logger.info(
            f"Progress: {state.ticks_processed:,} ticks | "
            f"Block {state.current_block:,} | "
            f"Opps: {state.opportunities_found} | "
            f"PnL: ${state.cumulative_pnl:.4f} | "
            f"Time: {ts_str}"
        )


def run_simulation(
    market_ids: list[str],
    relationship_graph: RelationshipGraph,
    market_loader: MarketLoader,
    trade_loader: TradeLoader,
    block_loader: BlockLoader,
    config: BacktestConfig,
) -> list[ArbitrageOpportunity]:
    """Convenience function to run a simulation.
    
    Args:
        market_ids: Markets to simulate
        relationship_graph: Pre-computed relationship graph
        market_loader: Market data loader
        trade_loader: Trade data loader
        block_loader: Block data loader
        config: Backtest configuration
        
    Returns:
        List of detected opportunities
    """
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
