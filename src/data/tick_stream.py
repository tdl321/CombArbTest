"""Tick-level data layer for arbitrage detection.

This module provides tick-level (trade-by-trade) data access that preserves
the precise ordering needed for cross-market arbitrage detection.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Iterator, Sequence

import polars as pl

from .loader import BlockLoader, MarketLoader, TradeLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TickPosition:
    """Unique position in the tick stream."""
    block_number: int
    log_index: int
    
    def __lt__(self, other: "TickPosition") -> bool:
        if self.block_number != other.block_number:
            return self.block_number < other.block_number
        return self.log_index < other.log_index
    
    def __le__(self, other: "TickPosition") -> bool:
        return self == other or self < other


@dataclass(frozen=True, slots=True)
class Tick:
    """A single trade event with computed price."""
    position: TickPosition
    transaction_hash: str
    token_id: str
    side: str
    price: Decimal
    size: Decimal
    usdc_volume: Decimal
    maker: str
    taker: str
    timestamp: datetime | None = None
    
    @property
    def block_number(self) -> int:
        return self.position.block_number
    
    @property
    def log_index(self) -> int:
        return self.position.log_index


@dataclass
class MarketStateSnapshot:
    """Point-in-time state of a market derived from recent trades."""
    market_id: str
    token_id: str
    outcome_index: int
    position: TickPosition
    timestamp: datetime | None
    last_price: Decimal | None = None
    prev_price: Decimal | None = None
    trade_count_window: int = 0
    
    @property
    def price_change(self) -> Decimal | None:
        if self.last_price is None or self.prev_price is None:
            return None
        return self.last_price - self.prev_price
    
    @property
    def is_stale(self) -> bool:
        return self.trade_count_window == 0


@dataclass
class CrossMarketSnapshot:
    """Synchronized snapshot of multiple related markets at a tick position."""
    position: TickPosition
    timestamp: datetime | None
    states: dict[str, MarketStateSnapshot]
    
    def get_prices(self) -> dict[str, Decimal | None]:
        return {mid: state.last_price for mid, state in self.states.items()}
    
    def has_all_prices(self) -> bool:
        return all(state.last_price is not None for state in self.states.values())


class TickStream:
    """Iterator over trades for a market, maintaining block/log ordering."""
    
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
        logger.debug("[DATA] TickStream initialized for market %s, token %s", market_id, self.target_token_id)
    
    def iter_ticks(
        self,
        start_block: int | None = None,
        end_block: int | None = None,
        batch_size: int = 10000,
    ) -> Iterator[Tick]:
        """Iterate over ticks in (block, log_index) order."""
        logger.info("[DATA] Starting tick iteration: start_block=%s, end_block=%s", start_block, end_block)
        tick_count = 0
        current_block = start_block
        
        while True:
            trades_df = self.trade_loader.query_trades(
                asset_ids=[self.target_token_id],
                min_block=current_block,
                max_block=end_block,
                limit=batch_size,
            )
            
            if trades_df.is_empty():
                break
            
            trades_df = self.trade_loader.enrich_with_timestamps(trades_df)
            trades_df = trades_df.sort(["block_number", "log_index"])
            
            for row in trades_df.iter_rows(named=True):
                tick = self._row_to_tick(row)
                if tick is not None:
                    tick_count += 1
                    yield tick
            
            last_block = trades_df["block_number"].max()
            if last_block == current_block:
                break
            current_block = last_block
            
            if end_block is not None and current_block >= end_block:
                break
        
        logger.info("[DATA] Tick iteration complete: %d ticks yielded", tick_count)
    
    def _row_to_tick(self, row: dict) -> Tick | None:
        """Convert a trade row to a Tick."""
        maker_asset = str(row["maker_asset_id"])
        taker_asset = str(row["taker_asset_id"])
        maker_amount = Decimal(row["maker_amount"]) / Decimal(1_000_000)
        taker_amount = Decimal(row["taker_amount"]) / Decimal(1_000_000)
        
        if maker_asset == "0":
            side = "BUY"
            price = maker_amount / taker_amount if taker_amount > 0 else None
            size = taker_amount
            usdc_volume = maker_amount
            token_id = taker_asset
        elif taker_asset == "0":
            side = "SELL"
            price = taker_amount / maker_amount if maker_amount > 0 else None
            size = maker_amount
            usdc_volume = taker_amount
            token_id = maker_asset
        else:
            return None
        
        if price is None:
            return None
        
        if token_id != self.target_token_id:
            return None
        
        return Tick(
            position=TickPosition(
                block_number=row["block_number"],
                log_index=row["log_index"],
            ),
            transaction_hash=row["transaction_hash"],
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            usdc_volume=usdc_volume,
            maker=row["maker"],
            taker=row["taker"],
            timestamp=row.get("block_timestamp"),
        )


class CrossMarketIterator:
    """Synchronized iteration across multiple markets for arbitrage detection."""
    
    def __init__(
        self,
        trade_loader: TradeLoader,
        block_loader: BlockLoader,
        market_loader: MarketLoader,
        market_ids: list[str],
        outcome_indices: dict[str, int] | None = None,
    ):
        self.trade_loader = trade_loader
        self.block_loader = block_loader
        self.market_loader = market_loader
        self.market_ids = market_ids
        self.outcome_indices = outcome_indices or {mid: 0 for mid in market_ids}
        
        self._streams: dict[str, TickStream] = {}
        self._token_ids: dict[str, str] = {}
        self._init_streams()
        logger.info("[DATA] CrossMarketIterator initialized with %d markets", len(market_ids))
    
    def _init_streams(self) -> None:
        """Initialize tick streams for each market."""
        for market_id in self.market_ids:
            market = self.market_loader.get_market(market_id)
            if market is None or market.clob_token_ids is None:
                logger.warning("[DATA] Skipping market %s: no data or token IDs", market_id)
                continue
            
            outcome_idx = self.outcome_indices.get(market_id, 0)
            if outcome_idx >= len(market.clob_token_ids):
                logger.warning("[DATA] Skipping market %s: invalid outcome index %d", market_id, outcome_idx)
                continue
            
            self._token_ids[market_id] = market.clob_token_ids[outcome_idx]
            self._streams[market_id] = TickStream(
                trade_loader=self.trade_loader,
                block_loader=self.block_loader,
                market_id=market_id,
                token_ids=market.clob_token_ids,
                outcome_index=outcome_idx,
            )
        logger.debug("[DATA] Initialized %d market streams", len(self._streams))
    
    def iter_snapshots(
        self,
        start_block: int | None = None,
        end_block: int | None = None,
        batch_size: int = 5000,
    ) -> Iterator[CrossMarketSnapshot]:
        """Iterate over cross-market snapshots in tick order."""
        if not self._streams:
            logger.warning("[DATA] No streams available for iteration")
            return
        
        logger.info("[DATA] Starting cross-market iteration: %d markets, blocks %s to %s",
                    len(self._streams), start_block, end_block)
        start_time = time.time()
        snapshot_count = 0
        
        all_token_ids = list(self._token_ids.values())
        
        trades_df = self.trade_loader.query_trades(
            asset_ids=all_token_ids,
            min_block=start_block,
            max_block=end_block,
            limit=batch_size,
        )
        
        if trades_df.is_empty():
            logger.warning("[DATA] No trades found in block range")
            return
        
        trades_df = self.trade_loader.enrich_with_timestamps(trades_df)
        trades_df = trades_df.sort(["block_number", "log_index"])
        
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
        
        for row in trades_df.iter_rows(named=True):
            position = TickPosition(row["block_number"], row["log_index"])
            timestamp = row.get("block_timestamp")
            
            maker_asset = str(row["maker_asset_id"])
            taker_asset = str(row["taker_asset_id"])
            
            trade_token = None
            for market_id, token_id in self._token_ids.items():
                if token_id == maker_asset or token_id == taker_asset:
                    trade_token = token_id
                    break
            
            if trade_token is None:
                continue
            
            maker_amount = Decimal(row["maker_amount"]) / Decimal(1_000_000)
            taker_amount = Decimal(row["taker_amount"]) / Decimal(1_000_000)
            
            if maker_asset == "0" and taker_amount > 0:
                price = maker_amount / taker_amount
            elif taker_asset == "0" and maker_amount > 0:
                price = taker_amount / maker_amount
            else:
                continue
            
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
            
            snapshot_count += 1
            yield CrossMarketSnapshot(
                position=position,
                timestamp=timestamp,
                states=dict(states),
            )
        
        elapsed = time.time() - start_time
        logger.info("[DATA] Cross-market iteration complete: %d snapshots in %.3fs", snapshot_count, elapsed)


def detect_price_divergence(
    snapshot: CrossMarketSnapshot,
    threshold: Decimal = Decimal("0.02"),
) -> dict | None:
    """Detect if prices in snapshot diverge beyond threshold."""
    if not snapshot.has_all_prices():
        return None
    
    prices = snapshot.get_prices()
    
    for market_id, price in prices.items():
        if price is not None:
            if price < Decimal("0.01") or price > Decimal("0.99"):
                logger.debug("[DATA] Extreme price detected: market=%s, price=%s", market_id, price)
                return {
                    "type": "extreme_price",
                    "market_id": market_id,
                    "price": price,
                    "position": snapshot.position,
                    "timestamp": snapshot.timestamp,
                }
    
    return None
