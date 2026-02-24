"""Data models for Polymarket market and trade data."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MarketStatus(str, Enum):
    """Market status enum."""
    ACTIVE = "active"
    CLOSED = "closed"


class Market(BaseModel):
    """Polymarket market metadata."""
    id: str
    condition_id: str
    question: str
    slug: str
    outcomes: list[str]  # Parsed from JSON string
    outcome_prices: list[float]  # Parsed from JSON string
    clob_token_ids: list[str] | None = None  # Parsed from JSON string
    volume: float
    liquidity: float
    active: bool
    closed: bool
    end_date: datetime | None = None
    created_at: datetime | None = None
    market_maker_address: str | None = None
    fetched_at: datetime | None = None

    @property
    def status(self) -> MarketStatus:
        """Derive status from active/closed flags."""
        if self.closed:
            return MarketStatus.CLOSED
        return MarketStatus.ACTIVE


class Trade(BaseModel):
    """Polymarket trade from OrderFilled event."""
    block_number: int
    transaction_hash: str
    log_index: int
    order_hash: str
    maker: str
    taker: str
    maker_asset_id: int
    taker_asset_id: int
    maker_amount: int  # 6 decimals for USDC
    taker_amount: int  # 6 decimals
    fee: int  # 6 decimals
    timestamp: datetime | None = None
    fetched_at: datetime | None = None
    contract: str = "CTF Exchange"

    @property
    def price(self) -> float | None:
        """Calculate implied price from amounts.
        
        Price = maker_amount / taker_amount when buying outcome tokens.
        Returns None if calculation not possible.
        """
        if self.taker_amount == 0:
            return None
        # When buying outcome tokens: you pay USDC (maker_amount) for tokens (taker_amount)
        # Price is cost per token = maker_amount / taker_amount
        return self.maker_amount / self.taker_amount


class BlockTimestamp(BaseModel):
    """Block number to timestamp mapping."""
    block_number: int
    timestamp: datetime
