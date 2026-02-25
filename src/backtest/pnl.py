"""PnL Calculation Module (BT-03).

Calculates theoretical profit from arbitrage opportunities.

DEPRECATION NOTICE:
The following functions are deprecated and will be removed in a future version:
- calculate_theoretical_profit() -> Use ArbitrageExtractor.extract_trades()
- calculate_kl_profit() -> Use ArbitrageExtractor.extract_trades()
- calculate_opportunity_pnl() -> Use ArbitrageExtractor.extract_trades()

The old functions use an incorrect profit model (sum of price adjustments).
The correct model uses violation magnitude as profit (ArbitrageExtractor).

Migration Guide:
    # OLD (deprecated)
    gross, net, dirs = calculate_opportunity_pnl(market_prices, coherent_prices, kl)
    
    # NEW (correct)
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
import warnings
from decimal import Decimal
from typing import Optional

from .schema import ArbitrageOpportunity

logger = logging.getLogger(__name__)


# ============================================================================
# DEPRECATED: Old price-convergence model (incorrect)
# ============================================================================

def calculate_theoretical_profit(
    market_prices: dict[str, float],
    coherent_prices: dict[str, float],
    kl_divergence: float,
    stake_size: float = 1.0,
) -> tuple[float, dict[str, str]]:
    """Calculate theoretical profit from price divergence.
    
    .. deprecated::
        This function uses an incorrect profit model (sum of |coherent - market|).
        Use ArbitrageExtractor.extract_trades() instead, which calculates profit
        as the violation magnitude (locked arbitrage profit).
        
    Migration:
        from src.arbitrage.extractor import ArbitrageExtractor
        extractor = ArbitrageExtractor()
        trades = extractor.extract_trades(solver_result)
        best_trade = max(trades, key=lambda t: t.locked_profit)
        profit = best_trade.locked_profit
        positions = best_trade.positions
    """
    warnings.warn(
        "calculate_theoretical_profit is deprecated. "
        "Use ArbitrageExtractor.extract_trades() for correct arbitrage profit calculation. "
        "See module docstring for migration guide.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    logger.warning(
        "[PNL] DEPRECATED: calculate_theoretical_profit called with %d markets",
        len(market_prices)
    )
    logger.debug(
        "[PNL] Calculating theoretical profit: markets=%d, kl=%.4f, stake=%.2f",
        len(market_prices),
        kl_divergence,
        stake_size
    )

    profit = 0.0
    trade_directions: dict[str, str] = {}

    for market_id in market_prices:
        market_price = market_prices[market_id]
        coherent_price = coherent_prices.get(market_id, market_price)

        adjustment = coherent_price - market_price

        if abs(adjustment) < 1e-6:
            continue

        if adjustment > 0:
            trade_directions[market_id] = "BUY"
            profit += adjustment * stake_size
        else:
            trade_directions[market_id] = "SELL"
            profit += abs(adjustment) * stake_size

    logger.debug(
        "[PNL] Theoretical profit: %.6f from %d trades (DEPRECATED MODEL)",
        profit,
        len(trade_directions)
    )
    return profit, trade_directions


def calculate_kl_profit(
    market_prices: dict[str, float],
    coherent_prices: dict[str, float],
    kl_divergence: float,
    stake_size: float = 1.0,
) -> float:
    """Alternative profit calculation based on KL divergence.
    
    .. deprecated::
        This function uses KL divergence as a proxy for profit, which is incorrect.
        Use ArbitrageExtractor.extract_trades() instead.
    """
    warnings.warn(
        "calculate_kl_profit is deprecated. "
        "Use ArbitrageExtractor.extract_trades() for correct arbitrage profit calculation.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning(
        "[PNL] DEPRECATED: calculate_kl_profit called with kl=%.4f",
        kl_divergence
    )
    return kl_divergence * stake_size


def apply_transaction_costs(
    gross_profit: float,
    num_trades: int,
    transaction_cost_rate: float = 0.015,
) -> float:
    """Apply transaction costs to gross profit.
    
    Note: This function is still valid for applying costs to any gross profit.
    However, consider using ArbitrageTrade.net_profit(fee_per_leg) which
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


def calculate_opportunity_pnl(
    market_prices: dict[str, float],
    coherent_prices: dict[str, float],
    kl_divergence: float,
    transaction_cost_rate: float = 0.015,
    stake_size: float = 1.0,
) -> tuple[float, float, dict[str, str]]:
    """Calculate full PnL for an arbitrage opportunity.
    
    .. deprecated::
        This function uses an incorrect profit model. The correct approach:
        
        from src.arbitrage.extractor import ArbitrageExtractor
        extractor = ArbitrageExtractor(fee_per_leg=0.01)
        trades = extractor.extract_trades(solver_result)
        if trades:
            best = max(trades, key=lambda t: t.locked_profit)
            gross = best.locked_profit
            net = best.net_profit(fee_per_leg=0.01)
            positions = best.positions
    """
    warnings.warn(
        "calculate_opportunity_pnl is deprecated. "
        "Use ArbitrageExtractor.extract_trades() for correct arbitrage profit. "
        "See module docstring for migration guide.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    logger.warning(
        "[PNL] DEPRECATED: calculate_opportunity_pnl called with %d markets, kl=%.4f",
        len(market_prices),
        kl_divergence
    )
    
    # Suppress nested deprecation warning for backward compatibility
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        gross_profit, trade_directions = calculate_theoretical_profit(
            market_prices=market_prices,
            coherent_prices=coherent_prices,
            kl_divergence=kl_divergence,
            stake_size=stake_size,
        )

    num_trades = len(trade_directions)

    net_profit = apply_transaction_costs(
        gross_profit=gross_profit,
        num_trades=num_trades,
        transaction_cost_rate=transaction_cost_rate,
    )

    logger.debug(
        "[PNL] Opportunity PnL (DEPRECATED): gross=%.4f, net=%.4f, trades=%d",
        gross_profit,
        net_profit,
        num_trades
    )

    return gross_profit, net_profit, trade_directions


class PnLTracker:
    """Track cumulative PnL over a backtest.
    
    .. deprecated::
        This class uses the old ArbitrageOpportunity schema with theoretical_profit.
        When the schema is updated to use ArbitrageTrade, this class should be
        updated to use trade.locked_profit and trade.net_profit() instead.
    """

    def __init__(self, transaction_cost_rate: float = 0.015):
        warnings.warn(
            "PnLTracker uses deprecated ArbitrageOpportunity.theoretical_profit. "
            "Will be updated when schema migration is complete.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.transaction_cost_rate = transaction_cost_rate

        self.gross_pnl = 0.0
        self.net_pnl = 0.0
        self.total_costs = 0.0
        self.num_trades = 0

        self.win_count = 0
        self.loss_count = 0

        self.peak_pnl = 0.0
        self.max_drawdown = 0.0

        self.returns: list[float] = []

        logger.info(
            "[PNL] PnLTracker initialized (DEPRECATED): tx_cost_rate=%.4f",
            transaction_cost_rate
        )

    def record_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> None:
        """Record an arbitrage opportunity."""
        self.gross_pnl += opportunity.theoretical_profit
        self.net_pnl += opportunity.net_profit
        self.total_costs += (opportunity.theoretical_profit - opportunity.net_profit)
        self.num_trades += len(opportunity.trade_direction)

        if opportunity.net_profit > 0:
            self.win_count += 1
        elif opportunity.net_profit < 0:
            self.loss_count += 1

        if self.net_pnl > self.peak_pnl:
            self.peak_pnl = self.net_pnl

        drawdown = self.peak_pnl - self.net_pnl
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        self.returns.append(opportunity.net_profit)

        logger.debug(
            "[PNL] Recorded: net=%.4f, cumulative=%.4f, drawdown=%.4f",
            opportunity.net_profit,
            self.net_pnl,
            drawdown
        )

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.0

    @property
    def avg_return(self) -> float:
        """Calculate average return per opportunity."""
        if not self.returns:
            return 0.0
        return sum(self.returns) / len(self.returns)

    def get_summary(self) -> dict:
        """Get summary statistics."""
        summary = {
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "total_costs": self.total_costs,
            "num_trades": self.num_trades,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "avg_return": self.avg_return,
        }
        logger.info(
            "[PNL] Summary: gross=%.4f, net=%.4f, win_rate=%.2f%%, max_dd=%.4f, trades=%d",
            self.gross_pnl,
            self.net_pnl,
            self.win_rate * 100,
            self.max_drawdown,
            self.num_trades
        )
        return summary


# ============================================================================
# NEW: Correct arbitrage-based PnL calculation
# ============================================================================

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
    from ..arbitrage.extractor import ArbitrageExtractor, ArbitrageTrade
    from ..optimizer.schema import ArbitrageResult
    
    logger.debug(
        "[PNL] calculate_arbitrage_pnl: %d markets, %d violations, fee=%.4f",
        len(market_prices),
        len(constraints_violated),
        fee_per_leg
    )
    
    # Build a minimal ArbitrageResult to use the extractor
    # (In practice, you'd pass the full result)
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


def calculate_partition_pnl(
    outcome_prices: list[float],
    fee_per_leg: float = 0.01,
    stake_size: float = 1.0,
) -> tuple[str | None, float, float]:
    """Simple partition arbitrage check.
    
    For a set of mutually exclusive, exhaustive outcomes:
    - If sum < 1: BUY ALL -> profit = (1 - sum) per unit
    - If sum > 1: SELL ALL -> profit = (sum - 1) per unit
    
    Args:
        outcome_prices: List of outcome prices
        fee_per_leg: Transaction fee per leg
        stake_size: Size of position
        
    Returns:
        direction: "BUY_ALL", "SELL_ALL", or None
        gross_profit: Before fees
        net_profit: After fees
    """
    total = sum(outcome_prices)
    num_legs = len(outcome_prices)
    total_fees = num_legs * fee_per_leg
    
    logger.debug(
        "[PNL] Partition check: %d outcomes, sum=%.4f, fees=%.4f",
        num_legs,
        total,
        total_fees
    )
    
    if total < 1.0 - total_fees:
        gross = (1.0 - total) * stake_size
        net = gross - (total_fees * stake_size)
        logger.debug(
            "[PNL] Partition arb: BUY_ALL, gross=%.4f, net=%.4f",
            gross,
            net
        )
        return "BUY_ALL", gross, net
    
    elif total > 1.0 + total_fees:
        gross = (total - 1.0) * stake_size
        net = gross - (total_fees * stake_size)
        logger.debug(
            "[PNL] Partition arb: SELL_ALL, gross=%.4f, net=%.4f",
            gross,
            net
        )
        return "SELL_ALL", gross, net
    
    logger.debug("[PNL] No partition arbitrage: sum=%.4f within fee buffer", total)
    return None, 0.0, 0.0


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
