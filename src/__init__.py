# combarbbot src package
"""Combinatorial arbitrage detection for prediction markets."""

def __getattr__(name):
    """Lazy import optimizer functions."""
    if name in ("find_arbitrage", "find_marginal_arbitrage", "detect_arbitrage_simple"):
        from .optimizer import find_arbitrage, find_marginal_arbitrage, detect_arbitrage_simple
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["find_arbitrage", "find_marginal_arbitrage", "detect_arbitrage_simple"]
