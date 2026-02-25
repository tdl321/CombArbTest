"""Arbitrage Trade Extractor.

Converts constraint violations into executable arbitrage trades.
The key insight: profit comes from the VIOLATION MAGNITUDE, not price convergence.

For each constraint type:
- Partition (sum=1): Buy all if sum<1, sell all if sum>1
- Binary (YES+NO=1): Same as partition
- Implies (A→B): If P(A)>P(B), sell A + buy B
- Mutex (A+B≤1): If sum>1, sell both
"""

import logging
from dataclasses import dataclass, field
from typing import Literal

from ..optimizer.schema import ArbitrageResult, ConstraintViolation

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageTrade:
    """A single executable arbitrage trade.
    
    Represents the complete hedged position for one constraint violation.
    """
    constraint_type: str
    positions: dict[str, Literal["BUY", "SELL"]]  # market_id -> direction
    violation_amount: float  # The constraint violation magnitude
    locked_profit: float  # Guaranteed profit at settlement (before fees)
    market_prices: dict[str, float]  # Prices at detection
    description: str
    
    @property
    def num_legs(self) -> int:
        return len(self.positions)
    
    def net_profit(self, fee_per_leg: float = 0.01) -> float:
        """Profit after transaction fees."""
        total_fees = self.num_legs * fee_per_leg
        return self.locked_profit - total_fees


@dataclass 
class ArbitrageExtractor:
    """Extracts executable arbitrage trades from solver results."""
    
    min_profit_threshold: float = 0.001  # Minimum profit to consider
    fee_per_leg: float = 0.01  # Transaction fee per market leg
    
    def extract_trades(self, result: ArbitrageResult) -> list[ArbitrageTrade]:
        """Extract all arbitrage trades from a solver result.
        
        Args:
            result: ArbitrageResult from the optimizer
            
        Returns:
            List of profitable ArbitrageTrade objects
        """
        logger.debug(
            "[ARB] Extracting trades: %d violations, %d markets, kl=%.4f",
            len(result.constraints_violated),
            len(result.market_prices),
            result.kl_divergence
        )
        
        trades = []
        
        for violation in result.constraints_violated:
            logger.debug(
                "[ARB] Processing violation: type=%s, from=%s, to=%s",
                violation.constraint_type,
                violation.from_market,
                violation.to_market
            )
            
            trade = self._violation_to_trade(
                violation=violation,
                market_prices=result.market_prices,
            )
            
            if trade:
                net = trade.net_profit(self.fee_per_leg)
                if net >= self.min_profit_threshold:
                    logger.debug(
                        "[ARB] Trade accepted: type=%s, locked=%.4f, net=%.4f, legs=%d",
                        trade.constraint_type,
                        trade.locked_profit,
                        net,
                        trade.num_legs
                    )
                    trades.append(trade)
                else:
                    logger.debug(
                        "[ARB] Trade rejected (below threshold): type=%s, net=%.4f < %.4f",
                        trade.constraint_type,
                        net,
                        self.min_profit_threshold
                    )
        
        # Also check for partition-level arbitrage across all markets
        partition_trade = self._check_partition_arbitrage(result.market_prices)
        if partition_trade:
            net = partition_trade.net_profit(self.fee_per_leg)
            if net >= self.min_profit_threshold:
                logger.debug(
                    "[ARB] Partition trade found: locked=%.4f, net=%.4f",
                    partition_trade.locked_profit,
                    net
                )
                trades.append(partition_trade)
        
        if trades:
            logger.info(
                "[ARB] Extracted %d trades from %d violations, best_profit=%.4f",
                len(trades),
                len(result.constraints_violated),
                max(t.locked_profit for t in trades)
            )
        else:
            logger.debug(
                "[ARB] No profitable trades from %d violations",
                len(result.constraints_violated)
            )
        
        return trades
    
    def _violation_to_trade(
        self,
        violation: ConstraintViolation,
        market_prices: dict[str, float],
    ) -> ArbitrageTrade | None:
        """Convert a single constraint violation into a trade."""
        
        ctype = violation.constraint_type.lower()
        
        if "binary" in ctype:
            return self._handle_binary_violation(violation, market_prices)
        elif "implies" in ctype or "prerequisite" in ctype:
            return self._handle_implies_violation(violation, market_prices)
        elif "mutex" in ctype:
            return self._handle_mutex_violation(violation, market_prices)
        elif "partition" in ctype:
            return self._handle_partition_violation(violation, market_prices)
        else:
            logger.warning("[ARB] Unknown constraint type: %s", ctype)
            return None
    
    def _handle_binary_violation(
        self,
        violation: ConstraintViolation,
        market_prices: dict[str, float],
    ) -> ArbitrageTrade | None:
        """Handle YES + NO = 1 violation.
        
        If sum < 1: Buy both → pay sum, receive 1.0
        If sum > 1: Sell both → receive sum, pay 1.0
        """
        yes_market = violation.from_market
        no_market = violation.to_market
        
        if not no_market or yes_market not in market_prices or no_market not in market_prices:
            logger.debug(
                "[ARB] Binary violation skipped: missing markets (from=%s, to=%s)",
                yes_market, no_market
            )
            return None
        
        p_yes = market_prices[yes_market]
        p_no = market_prices[no_market]
        total = p_yes + p_no
        
        if total < 1.0:
            # Buy both
            profit = 1.0 - total
            positions = {yes_market: "BUY", no_market: "BUY"}
            desc = "Binary arb: buy both at %.4f, receive 1.0" % total
            direction = "underpriced"
        elif total > 1.0:
            # Sell both
            profit = total - 1.0
            positions = {yes_market: "SELL", no_market: "SELL"}
            desc = "Binary arb: sell both at %.4f, pay 1.0" % total
            direction = "overpriced"
        else:
            return None
        
        logger.debug(
            "[ARB] Binary: %s+%s=%.4f (%s), profit=%.4f",
            yes_market[:8], no_market[:8], total, direction, profit
        )
        
        return ArbitrageTrade(
            constraint_type="binary",
            positions=positions,
            violation_amount=abs(total - 1.0),
            locked_profit=profit,
            market_prices={yes_market: p_yes, no_market: p_no},
            description=desc,
        )
    
    def _handle_implies_violation(
        self,
        violation: ConstraintViolation,
        market_prices: dict[str, float],
    ) -> ArbitrageTrade | None:
        """Handle A → B violation where P(A) > P(B).
        
        Trade: Sell A, Buy B
        
        Outcomes:
        - A happens (B must too): lose on A, win on B → net = P(A) - P(B) + (1-P(A)) - (1-P(B)) = 0? 
        
        Wait, let me recalculate:
        - Sell A at P(A): if A happens, pay 1-P(A); if not, keep P(A)
        - Buy B at P(B): if B happens, get 1-P(B); if not, lose P(B)
        
        Case 1: A yes, B yes (A→B so valid)
          Sell A: receive P(A), pay out 1.0 → net = P(A) - 1
          Buy B: pay P(B), receive 1.0 → net = 1 - P(B)
          Total: P(A) - 1 + 1 - P(B) = P(A) - P(B) ✓
        
        Case 2: A no, B yes
          Sell A: receive P(A), pay 0 → net = P(A)
          Buy B: pay P(B), receive 1.0 → net = 1 - P(B)
          Total: P(A) + 1 - P(B) ✓
        
        Case 3: A no, B no
          Sell A: receive P(A), pay 0 → net = P(A)
          Buy B: pay P(B), receive 0 → net = -P(B)
          Total: P(A) - P(B) ✓
        
        Minimum profit = P(A) - P(B) when A→B is violated
        """
        from_market = violation.from_market  # A (the implying event)
        to_market = violation.to_market  # B (the implied event)
        
        if not to_market or from_market not in market_prices or to_market not in market_prices:
            logger.debug(
                "[ARB] Implies violation skipped: missing markets (from=%s, to=%s)",
                from_market, to_market
            )
            return None
        
        p_from = market_prices[from_market]
        p_to = market_prices[to_market]
        
        # Violation means P(from) > P(to)
        if p_from <= p_to:
            logger.debug(
                "[ARB] Implies: no violation P(%s)=%.4f <= P(%s)=%.4f",
                from_market[:8], p_from, to_market[:8], p_to
            )
            return None
        
        profit = p_from - p_to
        
        logger.debug(
            "[ARB] Implies: %s→%s violated, P(from)=%.4f > P(to)=%.4f, profit=%.4f",
            from_market[:8], to_market[:8], p_from, p_to, profit
        )
        
        return ArbitrageTrade(
            constraint_type="implies",
            positions={from_market: "SELL", to_market: "BUY"},
            violation_amount=profit,
            locked_profit=profit,
            market_prices={from_market: p_from, to_market: p_to},
            description="Implies arb: %s→%s, sell@%.3f buy@%.3f" % (
                from_market[:12], to_market[:12], p_from, p_to
            ),
        )
    
    def _handle_mutex_violation(
        self,
        violation: ConstraintViolation,
        market_prices: dict[str, float],
    ) -> ArbitrageTrade | None:
        """Handle P(A) + P(B) ≤ 1 violation.
        
        If sum > 1: Sell both
        
        Outcomes (A and B mutually exclusive):
        - A yes, B no: pay 1 on A, keep B premium
        - A no, B yes: keep A premium, pay 1 on B  
        - A no, B no: keep both premiums
        
        Worst case: one happens, pay 1, received sum > 1
        Profit = sum - 1
        """
        market_a = violation.from_market
        market_b = violation.to_market
        
        if not market_b or market_a not in market_prices or market_b not in market_prices:
            logger.debug(
                "[ARB] Mutex violation skipped: missing markets (from=%s, to=%s)",
                market_a, market_b
            )
            return None
        
        p_a = market_prices[market_a]
        p_b = market_prices[market_b]
        total = p_a + p_b
        
        if total <= 1.0:
            logger.debug(
                "[ARB] Mutex: no violation P(%s)+P(%s)=%.4f <= 1.0",
                market_a[:8], market_b[:8], total
            )
            return None
        
        profit = total - 1.0
        
        logger.debug(
            "[ARB] Mutex: %s+%s=%.4f > 1.0, profit=%.4f",
            market_a[:8], market_b[:8], total, profit
        )
        
        return ArbitrageTrade(
            constraint_type="mutex",
            positions={market_a: "SELL", market_b: "SELL"},
            violation_amount=profit,
            locked_profit=profit,
            market_prices={market_a: p_a, market_b: p_b},
            description="Mutex arb: sell both at %.4f, max payout 1.0" % total,
        )
    
    def _check_partition_arbitrage(
        self,
        market_prices: dict[str, float],
    ) -> ArbitrageTrade | None:
        """Check if all markets form a partition with sum ≠ 1.
        
        This is the classic "buy/sell the field" arbitrage.
        """
        if len(market_prices) < 2:
            return None
        
        total = sum(market_prices.values())
        
        # Need meaningful deviation (not just floating point noise)
        if abs(total - 1.0) < 0.001:
            return None
        
        if total < 1.0:
            profit = 1.0 - total
            positions = {m: "BUY" for m in market_prices}
            desc = "Partition arb: buy all at %.4f, receive 1.0" % total
            direction = "underpriced"
        else:
            profit = total - 1.0
            positions = {m: "SELL" for m in market_prices}
            desc = "Partition arb: sell all at %.4f, pay 1.0" % total
            direction = "overpriced"
        
        logger.debug(
            "[ARB] Partition check: %d markets, sum=%.4f (%s), profit=%.4f",
            len(market_prices), total, direction, profit
        )
        
        return ArbitrageTrade(
            constraint_type="partition",
            positions=positions,
            violation_amount=abs(total - 1.0),
            locked_profit=profit,
            market_prices=dict(market_prices),
            description=desc,
        )
    
    def _handle_partition_violation(
        self,
        violation: ConstraintViolation,
        market_prices: dict[str, float],
    ) -> ArbitrageTrade | None:
        """Handle explicit partition constraint violation."""
        # This would need the full list of markets in the partition
        # For now, delegate to _check_partition_arbitrage
        return None


def extract_arbitrage_from_result(
    result: ArbitrageResult,
    min_profit: float = 0.001,
    fee_per_leg: float = 0.01,
) -> list[ArbitrageTrade]:
    """Convenience function to extract trades from solver result."""
    logger.debug(
        "[ARB] extract_arbitrage_from_result: min_profit=%.4f, fee=%.4f",
        min_profit, fee_per_leg
    )
    extractor = ArbitrageExtractor(
        min_profit_threshold=min_profit,
        fee_per_leg=fee_per_leg,
    )
    return extractor.extract_trades(result)
