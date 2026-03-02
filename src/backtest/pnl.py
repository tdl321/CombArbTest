"""PnL Calculation Module (BT-03).

Calculates profit from arbitrage opportunities.

The correct model uses violation magnitude as profit (ArbitrageExtractor).

Usage:
    from src.arbitrage.extractor import ArbitrageExtractor
    extractor = ArbitrageExtractor(min_profit_threshold=0.001, fee_per_leg=0.01)
    trades = extractor.extract_trades(solver_result)
    if trades:
        best = max(trades, key=lambda t: t.locked_profit)
        gross = best.locked_profit
        net = best.net_profit(fee_per_leg=0.01)
        positions = best.positions  # dict of market_id -> 'BUY'/'SELL'
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def apply_transaction_costs(
    gross_profit: float,
    num_trades: int,
    transaction_cost_rate: float = 0.015,
) -> float:
    """Apply transaction costs to gross profit.
    
    Note: Consider using ArbitrageTrade.net_profit(fee_per_leg) which
    handles fees correctly for hedged arbitrage positions.
    """
    total_cost = num_trades * transaction_cost_rate
    net_profit = gross_profit - total_cost

    logger.debug(
        "[PNL] Transaction costs: %.6f for %d trades, net profit: %.6f",
        total_cost,
        num_trades,
        net_profit
    )
    return net_profit


def calculate_arbitrage_pnl(
    market_prices: dict[str, float],
    constraints_violated: list,
    fee_per_leg: float = 0.01,
    stake_size: float = 1.0,
) -> tuple[float, float, dict[str, str]]:
    """Calculate PnL from actual arbitrage (constraint violations).
    
    This is the CORRECT approach: profit = violation magnitude x stake
    Not: profit = sum of price adjustments toward coherent prices
    
    Args:
        market_prices: Current market prices
        constraints_violated: List of constraint violations from solver
        fee_per_leg: Transaction fee per leg
        stake_size: Size of position
        
    Returns:
        gross_profit: Profit before fees
        net_profit: Profit after fees  
        trade_directions: market_id -> BUY/SELL
    """
    from ..arbitrage.extractor import ArbitrageExtractor
    from ..optimizer.schema import ArbitrageResult
    
    logger.debug(
        "[PNL] calculate_arbitrage_pnl: %d markets, %d violations, fee=%.4f",
        len(market_prices),
        len(constraints_violated),
        fee_per_leg
    )
    
    # Build a minimal ArbitrageResult to use the extractor
    result = ArbitrageResult(
        market_prices=market_prices,
        coherent_prices=market_prices,  # Not used by extractor
        kl_divergence=0.0,
        constraints_violated=constraints_violated,
        converged=True,
        iterations=0,
    )
    
    extractor = ArbitrageExtractor(
        min_profit_threshold=0.0,
        fee_per_leg=fee_per_leg,
    )
    
    trades = extractor.extract_trades(result)
    
    if not trades:
        logger.debug("[PNL] No trades extracted from violations")
        return 0.0, 0.0, {}
    
    # Take the best trade (highest profit)
    best_trade = max(trades, key=lambda t: t.locked_profit)
    
    gross_profit = best_trade.locked_profit * stake_size
    net_profit = best_trade.net_profit(fee_per_leg) * stake_size
    
    logger.debug(
        "[PNL] Arbitrage PnL: type=%s, gross=%.4f, net=%.4f, legs=%d",
        best_trade.constraint_type,
        gross_profit,
        net_profit,
        best_trade.num_legs
    )
    
    return gross_profit, net_profit, best_trade.positions


def calculate_implies_pnl(
    p_from: float,  # P(A) - the implying event
    p_to: float,    # P(B) - the implied event (A -> B)
    fee_per_leg: float = 0.01,
    stake_size: float = 1.0,
) -> tuple[bool, float, float]:
    """Check implication arbitrage.
    
    If A -> B but P(A) > P(B), we have arbitrage:
    - Sell A, Buy B
    - Locked profit = P(A) - P(B) in all valid states
    
    Args:
        p_from: Probability of implying event A
        p_to: Probability of implied event B
        fee_per_leg: Transaction fee per leg
        stake_size: Size of position
        
    Returns:
        has_arb: Whether arbitrage exists
        gross_profit: Before fees
        net_profit: After fees
    """
    total_fees = 2 * fee_per_leg  # Two legs
    
    logger.debug(
        "[PNL] Implies check: P(A)=%.4f, P(B)=%.4f, fees=%.4f",
        p_from,
        p_to,
        total_fees
    )
    
    if p_from > p_to + total_fees:
        gross = (p_from - p_to) * stake_size
        net = gross - (total_fees * stake_size)
        logger.debug(
            "[PNL] Implies arb found: gross=%.4f, net=%.4f",
            gross,
            net
        )
        return True, gross, net
    
    logger.debug("[PNL] No implies arbitrage: spread=%.4f < fees", p_from - p_to)
    return False, 0.0, 0.0
