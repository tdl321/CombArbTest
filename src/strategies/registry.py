"""Strategy plugin registry.

Strategies register themselves here. The backtest engine
discovers and instantiates strategies through this registry.
"""

from __future__ import annotations

import logging
from typing import Any, Type

from src.core.protocols import ArbitrageStrategy
from src.core.types import StrategyConfig

logger = logging.getLogger(__name__)

_STRATEGY_REGISTRY: dict[str, Type[ArbitrageStrategy]] = {}


def register_strategy(name: str):
    """Decorator to register a strategy class.

    Usage:
        @register_strategy("partition_arb")
        class PartitionArbitrage:
            ...
    """
    def decorator(cls):
        if name in _STRATEGY_REGISTRY:
            logger.warning("Overwriting strategy registration: %s", name)
        _STRATEGY_REGISTRY[name] = cls
        logger.info("Registered strategy: %s -> %s", name, cls.__name__)
        return cls
    return decorator


def get_strategy(name: str, config: StrategyConfig | None = None) -> ArbitrageStrategy:
    """Instantiate a registered strategy by name."""
    if name not in _STRATEGY_REGISTRY:
        available = list(_STRATEGY_REGISTRY.keys())
        raise KeyError(f"Strategy '{name}' not found. Available: {available}")

    cls = _STRATEGY_REGISTRY[name]
    if config is not None:
        return cls(config=config)
    return cls()


def list_strategies() -> list[str]:
    """List all registered strategy names."""
    return list(_STRATEGY_REGISTRY.keys())


def get_all_strategies(
    configs: dict[str, StrategyConfig] | None = None,
) -> list[ArbitrageStrategy]:
    """Instantiate all registered strategies."""
    configs = configs or {}
    strategies = []
    for name in _STRATEGY_REGISTRY:
        config = configs.get(name)
        strategies.append(get_strategy(name, config))
    return strategies
