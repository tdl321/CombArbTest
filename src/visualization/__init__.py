from .simplex import SimplexProjector
from .signal_plot import plot_arbitrage_signal, plot_signal_batch
from .schema import ArbitrageSignal
from .bregman_plot import (
    BregmanAnalysis,
    compute_bregman_analysis,
    signal_to_bregman,
    plot_bregman_dual_panel,
    plot_bregman_report,
    plot_single_cluster_summary,
)

__all__ = [
    "SimplexProjector",
    "ArbitrageSignal",
    "plot_arbitrage_signal",
    "plot_signal_batch",
    "BregmanAnalysis",
    "compute_bregman_analysis",
    "signal_to_bregman",
    "plot_bregman_dual_panel",
    "plot_bregman_report",
    "plot_single_cluster_summary",
]
