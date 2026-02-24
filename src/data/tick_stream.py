"""Tick-level data layer for arbitrage detection.

This module provides tick-level (trade-by-trade) data access that preserves
the precise ordering needed for cross-market arbitrage detection.

Key concepts:
- TickStream: Iterator over trades with (block_number, log_index) ordering
- MarketStateSnapshot: Point-in-time state of a market (best bid/ask implied)
- CrossMarketIterator: Synchronized iteration across multiple markets
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Iterator, Sequence

import polars as pl

from .loader import BlockLoader, MarketLoader, TradeLoader


@dataclass(frozen=True, slots=True)
class TickPosition:
    """Unique position in the tick stream.
    
    (block_number, log_index) provides deterministic ordering.
    All trades at the same position happened atomically (same tx or same block instant).
    """
    block_number: int
    log_index: int
    
    def __lt__(self, other: 'TickPosition') -> bool:
        if self.block_number != other.block_number:
            return self.block_number < other.block_number
        return self.log_index < other.log_index
    
    def __le__(self, other: 'TickPosition') -> bool:
        return self == other or self < other


@dataclass(frozen=True, slots=True)
class Tick:
    """A single trade event with computed price.
    
    Preserves raw data while adding computed fields for analysis.
    """
    position: TickPosition
    transaction_hash: str
    token_id: str  # The outcome token being traded
    side: str  # 'BUY' or 'SELL' (from taker's perspective)
    price: Decimal  # USDC per outcome token
    size: Decimal  # Tokens traded
    usdc_volume: Decimal  # USDC exchanged
    maker: str
    taker: str
    timestamp: datetime | None = None  # Joined from blocks table
    
    @property
    def block_number(self) -> int:
        return self.position.block_number
    
    @property
    def log_index(self) -> int:
        return self.position.log_index


@dataclass
class MarketStateSnapshot:
    """Point-in-time state of a market derived from recent trades.
    
    For arbitrage detection, we care about:
    - Last traded price (most recent signal)
    - Price direction (momentum)
    - Time since last trade (staleness)
    """
    market_id: str
    token_id: str
    outcome_index: int
    position: TickPosition
    timestamp: datetime | None
    
    # Derived from recent trades
    last_price: Decimal | None = None
    prev_price: Decimal | None = None  # For momentum calculation
    trade_count_window: int = 0  # Trades in lookback window
    
    @property
    def price_change(self) -> Decimal | None:
        """Price change from previous trade."""
        if self.last_price is None or self.prev_price is None:
            return None
        return self.last_price - self.prev_price
    
    @property
    def is_stale(self) -> bool:
        """Whether this state is based on old data (no recent trades)."""
        return self.trade_count_window == 0


@dataclass
class CrossMarketSnapshot:
    """Synchronized snapshot of multiple related markets at a tick position.
    
    Used for detecting arbitrage opportunities across markets.
    """
    position: TickPosition
    timestamp: datetime | None
    states: dict[str, MarketStateSnapshot]  # market_id -> state
    
    def get_prices(self) -> dict[str, Decimal | None]:
        """Get last traded prices for all markets."""
        return {mid: state.last_price for mid, state in self.states.items()}
    
    def has_all_prices(self) -> bool:
        """Check if all markets have valid prices."""
        return all(state.last_price is not None for state in self.states.values())


class TickStream:
    """Iterator over trades for a market, maintaining block/log ordering.
    
    Usage:
        stream = TickStream(trade_loader, block_loader, market)
        for tick in stream.iter_ticks(start_block, end_block):
            print(f"{tick.position}: {tick.price}")
    """
    
    def __init__(
        self,
        trade_loader: TradeLoader,
        block_loader: BlockLoader,
        market_id: str,
        token_ids: list[str],
        outcome_index: int = 0,
    ):
        self.trade_loader = trade_loader
        self.block_loader = block_loader
        self.market_id = market_id
        self.token_ids = token_ids
        self.target_token_id = token_ids[outcome_index]
        self.outcome_index = outcome_index
    
    def iter_ticks(
        self,
        start_block: int | None = None,
        end_block: int | None = None,
        batch_size: int = 10000,
    ) -> Iterator[Tick]:
        """Iterate over ticks in (block, log_index) order.
        
        Yields ticks lazily in batches to manage memory.
        """
        current_block = start_block
        
        while True:
            # Fetch batch of trades
            trades_df = self.trade_loader.query_trades(
                asset_ids=[self.target_token_id],
                min_block=current_block,
                max_block=end_block,
                limit=batch_size,
            )
            
            if trades_df.is_empty():
                break
            
            # Enrich with timestamps
            trades_df = self.trade_loader.enrich_with_timestamps(trades_df)
            
            # Sort by position
            trades_df = trades_df.sort(['block_number', 'log_index'])
            
            # Convert to ticks
            for row in trades_df.iter_rows(named=True):
                tick = self._row_to_tick(row)
                if tick is not None:
                    yield tick
            
            # Move to next batch
            last_block = trades_df['block_number'].max()
            if last_block == current_block:
                # All trades in this batch are from same block, need to handle pagination
                # This is a simplification - real impl would track log_index too
                break
            current_block = last_block
            
            # Check if we've reached the end
            if end_block is not None and current_block >= end_block:
                break
    
    def _row_to_tick(self, row: dict) -> Tick | None:
        """Convert a trade row to a Tick."""
        maker_asset = str(row['maker_asset_id'])
        taker_asset = str(row['taker_asset_id'])
        maker_amount = Decimal(row['maker_amount']) / Decimal(1_000_000)
        taker_amount = Decimal(row['taker_amount']) / Decimal(1_000_000)
        
        # Determine side and price
        # maker_asset_id == '0' means maker provides USDC (taker is buying tokens)
        # taker_asset_id == '0' means taker provides USDC (taker is selling tokens)
        if maker_asset == '0':
            # Taker buys tokens: pays USDC, receives tokens
            side = 'BUY'
            price = maker_amount / taker_amount if taker_amount > 0 else None
            size = taker_amount
            usdc_volume = maker_amount
            token_id = taker_asset
        elif taker_asset == '0':
            # Taker sells tokens: provides tokens, receives USDC
            side = 'SELL'
            price = taker_amount / maker_amount if maker_amount > 0 else None
            size = maker_amount
            usdc_volume = taker_amount
            token_id = maker_asset
        else:
            # Token-to-token trade (rare) - skip for now
            return None
        
        if price is None:
            return None
        
        # Filter to target token
        if token_id != self.target_token_id:
            return None
        
        return Tick(
            position=TickPosition(
                block_number=row['block_number'],
                log_index=row['log_index'],
            ),
            transaction_hash=row['transaction_hash'],
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            usdc_volume=usdc_volume,
            maker=row['maker'],
            taker=row['taker'],
            timestamp=row.get('block_timestamp'),
        )


class CrossMarketIterator:
    """Synchronized iteration across multiple markets for arbitrage detection.
    
    This is the key class for arbitrage detection. It maintains state for
    multiple markets and yields snapshots at each tick position where
    any market has activity.
    
    Usage:
        iterator = CrossMarketIterator(trade_loader, block_loader, markets)
        for snapshot in iterator.iter_snapshots(start_block, end_block):
            if snapshot.has_all_prices():
                prices = snapshot.get_prices()
                # Check for arbitrage...
    """
    
    def __init__(
        self,
        trade_loader: TradeLoader,
        block_loader: BlockLoader,
        market_loader: MarketLoader,
        market_ids: list[str],
        outcome_indices: dict[str, int] | None = None,  # market_id -> outcome_index
    ):
        self.trade_loader = trade_loader
        self.block_loader = block_loader
        self.market_loader = market_loader
        self.market_ids = market_ids
        self.outcome_indices = outcome_indices or {mid: 0 for mid in market_ids}
        
        # Initialize streams
        self._streams: dict[str, TickStream] = {}
        self._token_ids: dict[str, str] = {}  # market_id -> target_token_id
        self._init_streams()
    
    def _init_streams(self) -> None:
        """Initialize tick streams for each market."""
        for market_id in self.market_ids:
            market = self.market_loader.get_market(market_id)
            if market is None or market.clob_token_ids is None:
                continue
            
            outcome_idx = self.outcome_indices.get(market_id, 0)
            if outcome_idx >= len(market.clob_token_ids):
                continue
            
            self._token_ids[market_id] = market.clob_token_ids[outcome_idx]
            self._streams[market_id] = TickStream(
                trade_loader=self.trade_loader,
                block_loader=self.block_loader,
                market_id=market_id,
                token_ids=market.clob_token_ids,
                outcome_index=outcome_idx,
            )
    
    def iter_snapshots(
        self,
        start_block: int | None = None,
        end_block: int | None = None,
        batch_size: int = 5000,
    ) -> Iterator[CrossMarketSnapshot]:
        """Iterate over cross-market snapshots in tick order.
        
        Yields a snapshot at each tick position where any market has a trade.
        The snapshot contains the current state of ALL markets at that position.
        """
        if not self._streams:
            return
        
        # Collect all token IDs for query
        all_token_ids = list(self._token_ids.values())
        
        # Fetch all trades for all markets
        trades_df = self.trade_loader.query_trades(
            asset_ids=all_token_ids,
            min_block=start_block,
            max_block=end_block,
            limit=batch_size,
        )
        
        if trades_df.is_empty():
            return
        
        # Enrich with timestamps
        trades_df = self.trade_loader.enrich_with_timestamps(trades_df)
        
        # Sort by position
        trades_df = trades_df.sort(['block_number', 'log_index'])
        
        # Track current state for each market
        states: dict[str, MarketStateSnapshot] = {}
        for market_id in self.market_ids:
            if market_id in self._token_ids:
                states[market_id] = MarketStateSnapshot(
                    market_id=market_id,
                    token_id=self._token_ids[market_id],
                    outcome_index=self.outcome_indices.get(market_id, 0),
                    position=TickPosition(0, 0),
                    timestamp=None,
                )
        
        # Iterate through trades and yield snapshots
        for row in trades_df.iter_rows(named=True):
            position = TickPosition(row['block_number'], row['log_index'])
            timestamp = row.get('block_timestamp')
            
            # Determine which market this trade is for
            maker_asset = str(row['maker_asset_id'])
            taker_asset = str(row['taker_asset_id'])
            
            # Find matching market
            trade_token = None
            for market_id, token_id in self._token_ids.items():
                if token_id == maker_asset or token_id == taker_asset:
                    trade_token = token_id
                    break
            
            if trade_token is None:
                continue
            
            # Calculate price
            maker_amount = Decimal(row['maker_amount']) / Decimal(1_000_000)
            taker_amount = Decimal(row['taker_amount']) / Decimal(1_000_000)
            
            if maker_asset == '0' and taker_amount > 0:
                price = maker_amount / taker_amount
            elif taker_asset == '0' and maker_amount > 0:
                price = taker_amount / maker_amount
            else:
                continue
            
            # Update state for the affected market
            for market_id, token_id in self._token_ids.items():
                if token_id == trade_token:
                    old_state = states[market_id]
                    states[market_id] = MarketStateSnapshot(
                        market_id=market_id,
                        token_id=token_id,
                        outcome_index=self.outcome_indices.get(market_id, 0),
                        position=position,
                        timestamp=timestamp,
                        last_price=price,
                        prev_price=old_state.last_price,
                        trade_count_window=old_state.trade_count_window + 1,
                    )
                    break
            
            # Yield snapshot with current states
            yield CrossMarketSnapshot(
                position=position,
                timestamp=timestamp,
                states=dict(states),  # Copy to prevent mutation
            )


def detect_price_divergence(
    snapshot: CrossMarketSnapshot,
    threshold: Decimal = Decimal('0.02'),  # 2% threshold
) -> dict | None:
    """Detect if prices in snapshot diverge beyond threshold.
    
    For binary markets, YES + NO should sum to ~1.0.
    For related markets, check for mispricing based on logical constraints.
    
    Returns dict with divergence details if found, None otherwise.
    """
    if not snapshot.has_all_prices():
        return None
    
    prices = snapshot.get_prices()
    
    # Simple example: check if any price is obviously mispriced (< 0.01 or > 0.99)
    for market_id, price in prices.items():
        if price is not None:
            if price < Decimal('0.01') or price > Decimal('0.99'):
                return {
                    'type': 'extreme_price',
                    'market_id': market_id,
                    'price': price,
                    'position': snapshot.position,
                    'timestamp': snapshot.timestamp,
                }
    
    # More sophisticated divergence detection would go here
    # (e.g., checking if related markets violate logical constraints)
    
    return None
