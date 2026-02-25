# Strategy Gap Analysis: Current Implementation vs Reference

## Executive Summary

The current implementation has solid **detection** infrastructure but lacks critical **execution** components. The system can identify arbitrage opportunities but cannot profitably exploit them in production.

| Component | Reference | Current | Gap Severity |
|-----------|-----------|---------|--------------|
| Optimization | Integer Programming (Gurobi) | Linear Programming (HiGHS) | **MEDIUM** |
| Price Data | Real-time WebSocket + VWAP | Historical parquet snapshots | **CRITICAL** |
| Execution | Parallel atomic orders | None (backtest only) | **CRITICAL** |
| Liquidity | Order book depth modeling | Assumed infinite | **HIGH** |
| Position Sizing | Modified Kelly + depth cap | Fixed stake | **HIGH** |
| Slippage | VWAP-based estimation | None | **HIGH** |
| Latency | <30ms decision-to-mempool | N/A (backtest) | **CRITICAL** |

---

## Gap 1: Linear Programming vs Integer Programming

### Current Implementation
```python
# src/optimizer/lmo.py - Uses HiGHS LP solver
h = highspy.Highs()
h.addVars(self.n, self.constraints.lb, self.constraints.ub)
# Continuous variables only
```

### Reference Requirement
The reference specifies integer programming for the Linear Minimization Oracle:
```
zₜ = argmin over z in Z of ∇F(μₜ)·z
Where Z = {z ∈ {0,1}^I : A^T × z ≥ b}  # Binary constraints
```

### Impact
- **Low for partition constraints**: Partitions have closed-form solutions (normalization)
- **Medium for complex dependencies**: Non-partition constraints (implies, AND, OR) may need binary variables
- **Current workaround**: The partition fast-path handles most cases algebraically

### Implementation
```python
# Option 1: Add Gurobi for complex constraints
import gurobipy as gp

def solve_ip(self, g, constraints):
    model = gp.Model()
    z = model.addVars(self.n, vtype=gp.GRB.BINARY)
    model.setObjective(gp.quicksum(g[i] * z[i] for i in range(self.n)))
    # Add constraints...
    model.optimize()
    return [z[i].X for i in range(self.n)]

# Option 2: Keep LP for partitions, add IP fallback for complex cases
```

---

## Gap 2: No Real-Time Price Feed (CRITICAL)

### Current Implementation
```python
# src/data/loader.py - Historical parquet files only
class TradeLoader(DataLoader):
    def query_trades(self, asset_ids, min_block, max_block):
        # Queries historical data from parquet
```

### Reference Requirement
```
WebSocket connection to Polymarket CLOB API
  └─ Order book updates (price/volume changes)
  └─ Trade execution feed (fills happening)
  └─ Market creation/settlement events
```

### Implementation
Create `src/data/realtime.py`:
```python
import asyncio
import websockets
import json
from dataclasses import dataclass
from typing import Callable, Optional
from collections import defaultdict

@dataclass
class OrderBookLevel:
    price: float
    size: float
    
@dataclass 
class OrderBook:
    bids: list[OrderBookLevel]  # Sorted descending by price
    asks: list[OrderBookLevel]  # Sorted ascending by price
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None
    
    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    def get_vwap(self, side: str, size: float) -> Optional[float]:
        """Calculate VWAP for executing a given size."""
        levels = self.asks if side == "BUY" else self.bids
        total_cost = 0.0
        remaining = size
        
        for level in levels:
            fill = min(remaining, level.size)
            total_cost += fill * level.price
            remaining -= fill
            if remaining <= 0:
                break
        
        if remaining > 0:
            return None  # Insufficient liquidity
        return total_cost / size

class PolymarketWebSocket:
    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    
    def __init__(self):
        self.order_books: dict[str, OrderBook] = {}
        self.callbacks: list[Callable] = []
        self._ws = None
        
    async def connect(self, token_ids: list[str]):
        async with websockets.connect(self.WS_URL) as ws:
            self._ws = ws
            
            # Subscribe to markets
            for token_id in token_ids:
                await ws.send(json.dumps({
                    "type": "subscribe",
                    "channel": "book",
                    "market": token_id
                }))
            
            async for message in ws:
                data = json.loads(message)
                self._handle_message(data)
    
    def _handle_message(self, data: dict):
        if data.get("type") == "book":
            token_id = data["market"]
            self.order_books[token_id] = OrderBook(
                bids=[OrderBookLevel(float(b["price"]), float(b["size"])) 
                      for b in data.get("bids", [])],
                asks=[OrderBookLevel(float(a["price"]), float(a["size"])) 
                      for a in data.get("asks", [])]
            )
            for callback in self.callbacks:
                callback(token_id, self.order_books[token_id])
    
    def on_update(self, callback: Callable):
        self.callbacks.append(callback)
```

---

## Gap 3: No VWAP-Based Price Analysis (HIGH)

### Current Implementation
```python
# src/backtest/simulator.py - Uses point prices
prices = snapshot.get_prices()  # Single price per market
float_prices = {mid: float(p) for mid, p in prices.items()}
```

### Reference Requirement
```
VWAP = Σ(priceᵢ × volumeᵢ) / Σ(volumeᵢ)

For each block on Polygon (~2 seconds):
  Calculate VWAP_yes from all YES trades in that block
  Calculate VWAP_no from all NO trades in that block
```

### Implementation
Add to `src/data/price_series.py`:
```python
import polars as pl
from dataclasses import dataclass

@dataclass
class VWAPSnapshot:
    token_id: str
    block_number: int
    vwap: float
    volume: float
    trade_count: int

def compute_block_vwap(trades_df: pl.DataFrame, block_number: int) -> dict[str, VWAPSnapshot]:
    """Compute VWAP for each token in a specific block."""
    block_trades = trades_df.filter(pl.col("block_number") == block_number)
    
    vwaps = {}
    for token_id in block_trades["maker_asset_id"].unique().to_list():
        token_trades = block_trades.filter(
            (pl.col("maker_asset_id") == token_id) | 
            (pl.col("taker_asset_id") == token_id)
        )
        
        if len(token_trades) == 0:
            continue
            
        # Calculate VWAP
        total_value = (token_trades["price"] * token_trades["size"]).sum()
        total_volume = token_trades["size"].sum()
        
        vwaps[token_id] = VWAPSnapshot(
            token_id=token_id,
            block_number=block_number,
            vwap=total_value / total_volume if total_volume > 0 else 0,
            volume=total_volume,
            trade_count=len(token_trades)
        )
    
    return vwaps

def check_vwap_arbitrage(
    vwap_yes: float,
    vwap_no: float,
    threshold: float = 0.02
) -> tuple[bool, float]:
    """Check if VWAP sum deviates from 1.0."""
    total = vwap_yes + vwap_no
    deviation = abs(total - 1.0)
    return deviation > threshold, deviation
```

---

## Gap 4: No Order Book Depth / Liquidity Modeling (HIGH)

### Current Implementation
```python
# src/backtest/pnl.py - Assumes infinite liquidity
def calculate_theoretical_profit(market_prices, coherent_prices, ...):
    for market_id in market_prices:
        adjustment = coherent_price - market_price
        profit += adjustment * stake_size  # No liquidity check
```

### Reference Requirement
```
profit = (price deviation) × min(volume across all required positions)
```

### Implementation
Create `src/execution/liquidity.py`:
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class LiquidityAnalysis:
    max_executable_size: float
    expected_slippage: float
    vwap_cost: float
    depth_at_price: dict[str, float]  # market_id -> available volume

def analyze_execution_liquidity(
    order_books: dict[str, OrderBook],
    target_prices: dict[str, float],
    directions: dict[str, str],  # "BUY" or "SELL"
) -> LiquidityAnalysis:
    """Analyze how much can be executed at what cost."""
    
    available_sizes = []
    
    for market_id, direction in directions.items():
        book = order_books.get(market_id)
        if not book:
            return LiquidityAnalysis(0, float('inf'), float('inf'), {})
        
        # Calculate available volume up to target price
        target = target_prices[market_id]
        levels = book.asks if direction == "BUY" else book.bids
        
        available = 0.0
        for level in levels:
            if direction == "BUY" and level.price <= target:
                available += level.size
            elif direction == "SELL" and level.price >= target:
                available += level.size
        
        available_sizes.append(available)
    
    # Minimum across all markets determines max executable
    max_size = min(available_sizes) if available_sizes else 0
    
    # Calculate expected slippage at this size
    slippages = []
    for market_id, direction in directions.items():
        book = order_books[market_id]
        mid = book.mid_price or target_prices[market_id]
        vwap = book.get_vwap(direction, max_size)
        if vwap:
            slippage = abs(vwap - mid) / mid
            slippages.append(slippage)
    
    avg_slippage = sum(slippages) / len(slippages) if slippages else 0
    
    return LiquidityAnalysis(
        max_executable_size=max_size,
        expected_slippage=avg_slippage,
        vwap_cost=sum(slippages),  # Total slippage cost
        depth_at_price={m: s for m, s in zip(directions.keys(), available_sizes)}
    )
```

---

## Gap 5: No Position Sizing Logic (HIGH)

### Current Implementation
```python
# src/backtest/pnl.py
stake_size: float = 1.0  # Fixed stake
```

### Reference Requirement
```
f* = (b×p - q) / b × √p

Where:
- b = arbitrage profit percentage
- p = probability of full execution
- q = 1 - p

Cap at 50% of order book depth.
```

### Implementation
Create `src/execution/sizing.py`:
```python
import math
from dataclasses import dataclass

@dataclass
class PositionSize:
    optimal_size: float
    capped_size: float
    kelly_fraction: float
    execution_probability: float

def calculate_position_size(
    profit_rate: float,           # Expected profit as fraction
    execution_probability: float, # P(all legs fill)
    max_order_book_depth: float,  # Minimum liquidity across markets
    capital: float,               # Available capital
    max_depth_fraction: float = 0.5,  # Don't take >50% of book
) -> PositionSize:
    """Modified Kelly criterion with execution risk and depth cap."""
    
    b = profit_rate
    p = execution_probability
    q = 1 - p
    
    # Modified Kelly: f* = (b*p - q) / b * sqrt(p)
    if b <= 0 or p <= 0:
        return PositionSize(0, 0, 0, p)
    
    kelly = ((b * p - q) / b) * math.sqrt(p)
    kelly = max(0, kelly)  # Never negative
    
    optimal_size = kelly * capital
    
    # Cap at fraction of order book depth
    depth_cap = max_order_book_depth * max_depth_fraction
    capped_size = min(optimal_size, depth_cap)
    
    return PositionSize(
        optimal_size=optimal_size,
        capped_size=capped_size,
        kelly_fraction=kelly,
        execution_probability=p
    )

def estimate_execution_probability(
    required_sizes: dict[str, float],
    available_depths: dict[str, float],
) -> float:
    """Estimate probability all legs execute based on depth."""
    if not required_sizes:
        return 0.0
    
    # Simple model: P(fill) = min(1, depth/required) for each leg
    # P(all fill) = product of individual probabilities
    prob = 1.0
    for market_id, required in required_sizes.items():
        available = available_depths.get(market_id, 0)
        leg_prob = min(1.0, available / required) if required > 0 else 1.0
        prob *= leg_prob
    
    return prob
```

---

## Gap 6: No Execution Layer (CRITICAL)

### Current Implementation
- Backtest only, no live trading capability

### Reference Requirement
```
Sophisticated arbitrage system:
  WebSocket price feed:          <5ms
  Decision computation:          <10ms
  Direct RPC submission:         ~15ms
  Parallel execution:            ~10ms
  Total:                         ~2,040ms
```

### Implementation
Create `src/execution/executor.py`:
```python
import asyncio
import time
from dataclasses import dataclass
from typing import Optional
from web3 import Web3
from eth_account import Account

@dataclass
class ExecutionResult:
    success: bool
    tx_hashes: list[str]
    fill_prices: dict[str, float]
    total_cost: float
    execution_time_ms: float
    error: Optional[str] = None

@dataclass
class ExecutionConfig:
    min_profit_threshold: float = 0.05  # /bin/zsh.05 minimum
    max_slippage: float = 0.02          # 2% max slippage
    gas_price_gwei: int = 30
    parallel_submission: bool = True

class ArbitrageExecutor:
    POLYMARKET_CTF = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
    
    def __init__(
        self, 
        web3: Web3,
        private_key: str,
        config: ExecutionConfig = None
    ):
        self.w3 = web3
        self.account = Account.from_key(private_key)
        self.config = config or ExecutionConfig()
        
    async def execute_arbitrage(
        self,
        trades: dict[str, tuple[str, float]],  # market_id -> (direction, size)
        order_books: dict[str, OrderBook],
    ) -> ExecutionResult:
        """Execute arbitrage with parallel order submission."""
        start = time.time()
        
        # Pre-execution validation
        validation = self._validate_execution(trades, order_books)
        if not validation.valid:
            return ExecutionResult(
                success=False,
                tx_hashes=[],
                fill_prices={},
                total_cost=0,
                execution_time_ms=(time.time() - start) * 1000,
                error=validation.error
            )
        
        # Build transactions
        txs = self._build_transactions(trades, order_books)
        
        # Submit in parallel
        if self.config.parallel_submission:
            results = await asyncio.gather(*[
                self._submit_tx(tx) for tx in txs
            ])
        else:
            results = [await self._submit_tx(tx) for tx in txs]
        
        # Check all succeeded
        success = all(r.success for r in results)
        
        return ExecutionResult(
            success=success,
            tx_hashes=[r.tx_hash for r in results if r.tx_hash],
            fill_prices={r.market_id: r.fill_price for r in results},
            total_cost=sum(r.cost for r in results),
            execution_time_ms=(time.time() - start) * 1000,
            error=None if success else "Partial fill"
        )
    
    def _validate_execution(self, trades, order_books) -> 'ValidationResult':
        """Pre-submission validation."""
        # Check liquidity sufficient
        for market_id, (direction, size) in trades.items():
            book = order_books.get(market_id)
            if not book:
                return ValidationResult(False, f"No order book for {market_id}")
            
            vwap = book.get_vwap(direction, size)
            if vwap is None:
                return ValidationResult(False, f"Insufficient liquidity for {market_id}")
            
            # Check slippage
            mid = book.mid_price
            slippage = abs(vwap - mid) / mid if mid else float('inf')
            if slippage > self.config.max_slippage:
                return ValidationResult(False, f"Slippage too high for {market_id}: {slippage:.2%}")
        
        # Check profit still viable after slippage
        # ... (calculate net profit with VWAP prices)
        
        return ValidationResult(True, None)

@dataclass
class ValidationResult:
    valid: bool
    error: Optional[str]
```

---

## Gap 7: No Gas-Aware Cost Model

### Current Implementation
```python
# src/backtest/pnl.py
transaction_cost_rate: float = 0.015  # Fixed 1.5%
```

### Reference Requirement
```
Gas fees on 4-leg strategy: ~/bin/zsh.02
/bin/zsh.08 profit → 25% goes to gas
/bin/zsh.03 profit → 67% goes to gas
```

### Implementation
Update `src/backtest/pnl.py`:
```python
from dataclasses import dataclass

@dataclass
class GasEstimate:
    gas_units: int
    gas_price_gwei: float
    cost_eth: float
    cost_usd: float

def estimate_gas_cost(
    num_trades: int,
    gas_price_gwei: float = 30,
    eth_price_usd: float = 2500,
    gas_per_trade: int = 150_000,  # Polygon CLOB order
) -> GasEstimate:
    """Estimate gas cost for multi-leg execution."""
    total_gas = num_trades * gas_per_trade
    cost_eth = (total_gas * gas_price_gwei) / 1e9
    cost_usd = cost_eth * eth_price_usd
    
    return GasEstimate(
        gas_units=total_gas,
        gas_price_gwei=gas_price_gwei,
        cost_eth=cost_eth,
        cost_usd=cost_usd
    )

def calculate_net_profit_with_gas(
    gross_profit: float,
    num_trades: int,
    gas_price_gwei: float = 30,
) -> tuple[float, float]:
    """Calculate net profit accounting for gas."""
    gas = estimate_gas_cost(num_trades, gas_price_gwei)
    net = gross_profit - gas.cost_usd
    gas_fraction = gas.cost_usd / gross_profit if gross_profit > 0 else 1.0
    return net, gas_fraction
```

---

## Implementation Priority

### Phase 1: Core Execution (Week 1-2)
1. **Real-time WebSocket feed** - Cannot trade without live prices
2. **Order book tracking** - Need depth for sizing
3. **Basic execution layer** - Submit orders to CLOB

### Phase 2: Optimization (Week 3-4)
4. **VWAP analysis** - Better price signals
5. **Liquidity modeling** - Know what's executable
6. **Position sizing** - Kelly criterion + depth cap

### Phase 3: Production (Week 5-6)
7. **Parallel execution** - Atomic multi-leg trades
8. **Gas optimization** - Dynamic gas pricing
9. **Monitoring/alerts** - Drawdown, execution rate

### Phase 4: Enhancement (Week 7+)
10. **Integer programming** - For complex constraints
11. **Latency optimization** - Direct RPC, connection pooling
12. **Copy-trade detection** - Avoid being front-run

---

## Quick Wins

### 1. Add Minimum Profit Filter
```python
# In simulator.py _check_arbitrage
MIN_PROFIT_USD = 0.05
if net_profit < MIN_PROFIT_USD:
    return None  # Skip unprofitable opportunities
```

### 2. Add Depth Awareness to Backtest
```python
# In pnl.py calculate_opportunity_pnl
def calculate_opportunity_pnl(..., max_executable_size: float = None):
    if max_executable_size:
        stake_size = min(stake_size, max_executable_size)
```

### 3. Track Execution Feasibility
```python
@dataclass
class ArbitrageOpportunity:
    # ... existing fields ...
    estimated_executable_size: float = 0.0
    liquidity_limited: bool = False
```
