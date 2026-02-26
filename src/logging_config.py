"""Centralized Logging Configuration for combarbbot.

This module provides standardized logging setup across all components.
Each component uses a consistent prefix format: [PREFIX] message

Prefixes by module:
- [DATA] - Data loading operations (loader, tick_stream)
- [LLM] - LLM client operations
- [CLUSTER] - Overall clustering pipeline
- [STAGE1] - Stage 1: Semantic clustering
- [STAGE2] - Stage 2: Constraint extraction
- [OPT] - Overall optimization pipeline
- [FW] - Frank-Wolfe algorithm
- [LMO] - Linear Minimization Oracle
- [KL] - KL divergence calculations
- [BACKTEST] - Backtest runner
- [SIM] - Walk-forward simulation
- [PNL] - PnL calculations
- [REPORT] - Report generation
- [ARB] - Arbitrage trade extraction

Usage:
    from src.logging_config import setup_logging, get_logger

    # At application startup
    setup_logging(level=logging.INFO)

    # In each module (use standard logging.getLogger)
    import logging
    logger = logging.getLogger(__name__)
    logger.info("[DATA] Loading markets")
"""

import logging
import sys
import time
from datetime import datetime
from typing import Optional


class PrefixFormatter(logging.Formatter):
    """Custom formatter that includes timestamp and formats messages."""

    DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = False,
        datefmt: str = None,
    ):
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self._datefmt = datefmt if datefmt else self.DEFAULT_DATEFMT
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        parts = []

        if self.include_timestamp:
            timestamp = datetime.fromtimestamp(record.created).strftime(self._datefmt)
            parts.append(timestamp)

        if self.include_level:
            parts.append("[%s]" % record.levelname)

        # Check for prefix in extra or in the message itself
        prefix = getattr(record, "prefix", "")
        if prefix:
            parts.append("[%s]" % prefix)

        parts.append(record.getMessage())

        message = " ".join(parts)

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            message = message + "\n" + record.exc_text

        return message


class PrefixLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that automatically adds prefix to all messages."""

    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra["prefix"] = self.extra.get("prefix", "")
        kwargs["extra"] = extra
        return msg, kwargs


MODULE_PREFIXES = {
    "src.data": "DATA",
    "src.data.loader": "DATA",
    "src.data.tick_stream": "DATA",
    "src.data.models": "DATA",
    "src.llm": "LLM",
    "src.llm.client": "LLM",
    "src.llm.clustering": "CLUSTER",
    "src.llm.extractor": "EXTRACT",
    "src.llm.schema": "LLM",
    "src.optimizer": "OPT",
    "src.optimizer.frank_wolfe": "FW",
    "src.optimizer.lmo": "LMO",
    "src.optimizer.divergence": "KL",
    "src.optimizer.schema": "OPT",
    "src.arbitrage": "ARB",
    "src.arbitrage.extractor": "ARB",
    "src.backtest": "BACKTEST",
    "src.backtest.runner": "BACKTEST",
    "src.backtest.simulator": "SIM",
    "src.backtest.pnl": "PNL",
    "src.backtest.report": "REPORT",
    "src.backtest.report_generator": "REPORT",
    "src.backtest.schema": "BACKTEST",
    "src.config": "CONFIG",
}


def get_prefix_for_module(module_name: str) -> str:
    """Get the appropriate prefix for a module name."""
    if module_name in MODULE_PREFIXES:
        return MODULE_PREFIXES[module_name]

    parts = module_name.split(".")
    for i in range(len(parts), 0, -1):
        parent = ".".join(parts[:i])
        if parent in MODULE_PREFIXES:
            return MODULE_PREFIXES[parent]

    return parts[-1].upper() if parts else "LOG"


def get_logger(name_or_prefix: str) -> PrefixLoggerAdapter:
    """Get a logger with the specified prefix.

    Args:
        name_or_prefix: Either a module name (e.g., "src.data.loader")
                       or a prefix (e.g., "DATA")

    Returns:
        PrefixLoggerAdapter with appropriate prefix
    """
    if "." in name_or_prefix:
        module_name = name_or_prefix
        prefix = get_prefix_for_module(module_name)
    else:
        prefix = name_or_prefix
        module_name = "combarbbot.%s" % prefix.lower()

    logger = logging.getLogger(module_name)
    return PrefixLoggerAdapter(logger, {"prefix": prefix})


def setup_logging(
    level: int = logging.INFO,
    include_timestamp: bool = True,
    include_level: bool = False,
    stream=None,
    log_file: Optional[str] = None,
) -> None:
    """Configure logging for the entire application.

    This sets up the root logger and all module loggers to use
    our custom formatter with prefixes.
    """
    formatter = PrefixFormatter(
        include_timestamp=include_timestamp,
        include_level=include_level,
    )

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers from root
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add stream handler to root
    stream_handler = logging.StreamHandler(stream or sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    root_logger.addHandler(stream_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

    # Configure all our module loggers
    for module_name in MODULE_PREFIXES.keys():
        module_logger = logging.getLogger(module_name)
        module_logger.setLevel(level)
        # Let messages propagate to root
        module_logger.propagate = True

    # Also configure combarbbot namespace
    combarbbot_logger = logging.getLogger("combarbbot")
    combarbbot_logger.setLevel(level)
    combarbbot_logger.propagate = True


def setup_module_logger(module_name: str, level: int = logging.INFO) -> PrefixLoggerAdapter:
    """Setup and return a logger for a specific module."""
    prefix = get_prefix_for_module(module_name)
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    return PrefixLoggerAdapter(logger, {"prefix": prefix})


class TimingContext:
    """Context manager for timing operations with logging."""

    def __init__(self, logger: PrefixLoggerAdapter, operation: str, level: int = logging.DEBUG):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(self.level, "%s starting...", self.operation)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed = time.time() - self.start_time
            if exc_type:
                self.logger.log(self.level, "%s failed after %.3fs", self.operation, elapsed)
            else:
                self.logger.log(self.level, "%s completed in %.3fs", self.operation, elapsed)
        return False


def timed(logger: PrefixLoggerAdapter, operation: str, level: int = logging.DEBUG) -> TimingContext:
    """Create a timing context for an operation."""
    return TimingContext(logger, operation, level)
