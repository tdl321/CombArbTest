"""PnL Calculation Module (BT-03).

Calculates theoretical profit from arbitrage opportunities.
"""

from decimal import Decimal
from typing import Optional
import logging

from .schema import ArbitrageOpportunity

logger = logging.getLogger(__name__)


def calculate_theoretical_profit(
    market_prices: dict[str, float],
    coherent_prices: dict[str, float],
    kl_divergence: float,
    stake_size: float = 1.0,
) -> tuple[float, dict[str, str]]:
    """Calculate theoretical profit from price divergence.
    
    The profit comes from the difference between market prices and
    coherent (arbitrage-free) prices. We buy underpriced assets and
    sell overpriced assets.
    
    Args:
        market_prices: Current market prices {market_id: price}
        coherent_prices: Arbitrage-free prices {market_id: price}
        kl_divergence: KL divergence between the two price distributions
        stake_size: Amount to stake (default $1)
        
    Returns:
        Tuple of (profit, trade_directions)
        - profit: Theoretical profit in stake units
        - trade_directions: Dict of {market_id: "BUY"/"SELL"}
    """
    profit = 0.0
    trade_directions: dict[str, str] = {}
    
    for market_id in market_prices:
        market_price = market_prices[market_id]
        coherent_price = coherent_prices.get(market_id, market_price)
        
        # Price adjustment needed
        adjustment = coherent_price - market_price
        
        if abs(adjustment) < 1e-6:
            # No meaningful price difference
            continue
        
        if adjustment > 0:
            # Market is underpriced vs coherent -> BUY
            # Expected profit: coherent - market (what we pay)
            trade_directions[market_id] = "BUY"
            profit += adjustment * stake_size
        else:
            # Market is overpriced vs coherent -> SELL
            # Expected profit: market - coherent (what we receive)
            trade_directions[market_id] = "SELL"
            profit += abs(adjustment) * stake_size
    
    return profit, trade_directions


def calculate_kl_profit(
    market_prices: dict[str, float],
    coherent_prices: dict[str, float],
    kl_divergence: float,
    stake_size: float = 1.0,
) -> float:
    """Alternative profit calculation based on KL divergence.
    
    KL divergence represents the "information inefficiency" in the market.
    We can approximate expected profit as proportional to KL divergence.
    
    This is a simplified model - actual profit depends on how prices
    converge and market making fees.
    
    Args:
        market_prices: Current market prices
        coherent_prices: Arbitrage-free prices
        kl_divergence: KL(market || coherent)
        stake_size: Amount to stake
        
    Returns:
        Estimated profit from KL-based arbitrage
    """
    # KL divergence in nats; profit is roughly proportional
    # Scale factor based on typical price ranges
    return kl_divergence * stake_size


def apply_transaction_costs(
    gross_profit: float,
    num_trades: int,
    transaction_cost_rate: float = 0.015,
) -> float:
    """Apply transaction costs to gross profit.
    
    Transaction costs typically include:
    - Trading fees (taker/maker fees)
    - Spread costs
    - Slippage
    
    Args:
        gross_profit: Profit before costs
        num_trades: Number of trades executed
        transaction_cost_rate: Cost as fraction (default 1.5% round-trip)
        
    Returns:
        Net profit after costs
    """
    # Each arbitrage involves buying and selling, so we pay costs on both sides
    total_cost = num_trades * transaction_cost_rate
    return gross_profit - total_cost


def calculate_opportunity_pnl(
    market_prices: dict[str, float],
    coherent_prices: dict[str, float],
    kl_divergence: float,
    transaction_cost_rate: float = 0.015,
    stake_size: float = 1.0,
) -> tuple[float, float, dict[str, str]]:
    """Calculate full PnL for an arbitrage opportunity.
    
    Args:
        market_prices: Current market prices
        coherent_prices: Arbitrage-free prices
        kl_divergence: KL divergence
        transaction_cost_rate: Transaction cost rate
        stake_size: Amount to stake per market
        
    Returns:
        Tuple of (gross_profit, net_profit, trade_directions)
    """
    # Calculate gross profit
    gross_profit, trade_directions = calculate_theoretical_profit(
        market_prices=market_prices,
        coherent_prices=coherent_prices,
        kl_divergence=kl_divergence,
        stake_size=stake_size,
    )
    
    # Count trades (markets where we have non-trivial positions)
    num_trades = len(trade_directions)
    
    # Apply costs
    net_profit = apply_transaction_costs(
        gross_profit=gross_profit,
        num_trades=num_trades,
        transaction_cost_rate=transaction_cost_rate,
    )
    
    return gross_profit, net_profit, trade_directions


class PnLTracker:
    """Track cumulative PnL over a backtest.
    
    Tracks gross/net profit, win rate, and drawdown.
    """
    
    def __init__(self, transaction_cost_rate: float = 0.015):
        """Initialize tracker.
        
        Args:
            transaction_cost_rate: Transaction cost rate to apply
        """
        self.transaction_cost_rate = transaction_cost_rate
        
        # Cumulative metrics
        self.gross_pnl = 0.0
        self.net_pnl = 0.0
        self.total_costs = 0.0
        self.num_trades = 0
        
        # Win/loss tracking
        self.win_count = 0
        self.loss_count = 0
        
        # Drawdown tracking
        self.peak_pnl = 0.0
        self.max_drawdown = 0.0
        
        # For Sharpe ratio
        self.returns: list[float] = []
    
    def record_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> None:
        """Record an arbitrage opportunity.
        
        Args:
            opportunity: The opportunity to record
        """
        self.gross_pnl += opportunity.theoretical_profit
        self.net_pnl += opportunity.net_profit
        self.total_costs += (opportunity.theoretical_profit - opportunity.net_profit)
        self.num_trades += len(opportunity.trade_direction)
        
        # Track wins/losses (based on net profit)
        if opportunity.net_profit > 0:
            self.win_count += 1
        elif opportunity.net_profit < 0:
            self.loss_count += 1
        
        # Update peak and drawdown
        if self.net_pnl > self.peak_pnl:
            self.peak_pnl = self.net_pnl
        
        drawdown = self.peak_pnl - self.net_pnl
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        # Track return for Sharpe
        self.returns.append(opportunity.net_profit)
    
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
        return {
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
