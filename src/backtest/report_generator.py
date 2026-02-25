"""Standardized Backtest Report Generator.

Generates comprehensive reports with:
1. Summary statistics (correct arbitrage metrics)
2. Breakdown by constraint type
3. Top opportunities
4. Simplex visualizations
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..arbitrage.extractor import ArbitrageTrade
from ..visualization.bregman_plot import BregmanAnalysis, plot_bregman_dual_panel

logger = logging.getLogger(__name__)


@dataclass
class ConstraintTypeStats:
    """Statistics for a single constraint type."""
    constraint_type: str
    count: int = 0
    total_locked_profit: float = 0.0
    total_net_profit: float = 0.0
    avg_edge_pct: float = 0.0
    max_edge_pct: float = 0.0
    min_edge_pct: float = float("inf")
    
    def add_trade(self, trade: ArbitrageTrade, fee_per_leg: float = 0.01):
        self.count += 1
        self.total_locked_profit += trade.locked_profit
        self.total_net_profit += trade.net_profit(fee_per_leg)
        
        edge_pct = trade.violation_amount * 100
        self.max_edge_pct = max(self.max_edge_pct, edge_pct)
        self.min_edge_pct = min(self.min_edge_pct, edge_pct)
        self.avg_edge_pct = (
            (self.avg_edge_pct * (self.count - 1) + edge_pct) / self.count
        )


@dataclass
class EdgeDistribution:
    """Distribution of edge sizes."""
    bucket_lt_1pct: int = 0
    bucket_1_2pct: int = 0
    bucket_2_5pct: int = 0
    bucket_5_10pct: int = 0
    bucket_gt_10pct: int = 0
    
    profit_lt_1pct: float = 0.0
    profit_1_2pct: float = 0.0
    profit_2_5pct: float = 0.0
    profit_5_10pct: float = 0.0
    profit_gt_10pct: float = 0.0
    
    def add_trade(self, trade: ArbitrageTrade, fee_per_leg: float = 0.01):
        edge_pct = trade.violation_amount * 100
        net = trade.net_profit(fee_per_leg)
        
        if edge_pct < 1:
            self.bucket_lt_1pct += 1
            self.profit_lt_1pct += net
        elif edge_pct < 2:
            self.bucket_1_2pct += 1
            self.profit_1_2pct += net
        elif edge_pct < 5:
            self.bucket_2_5pct += 1
            self.profit_2_5pct += net
        elif edge_pct < 10:
            self.bucket_5_10pct += 1
            self.profit_5_10pct += net
        else:
            self.bucket_gt_10pct += 1
            self.profit_gt_10pct += net


@dataclass
class ArbitrageBacktestReport:
    """Complete arbitrage backtest report."""
    generated_at: datetime = field(default_factory=datetime.now)
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    
    markets_analyzed: int = 0
    clusters_monitored: int = 0
    
    total_opportunities: int = 0
    profitable_after_fees: int = 0
    unprofitable_after_fees: int = 0
    
    total_locked_profit: float = 0.0
    total_fees: float = 0.0
    total_net_profit: float = 0.0
    avg_profit_per_opp: float = 0.0
    
    by_constraint_type: dict = field(default_factory=dict)
    edge_distribution: EdgeDistribution = field(default_factory=EdgeDistribution)
    
    fee_per_leg: float = 0.01
    assumed_fill_rate: float = 1.0
    assumed_slippage: float = 0.0
    
    top_opportunities: list = field(default_factory=list)
    all_trades: list = field(default_factory=list)
    
    def add_trade(self, trade: ArbitrageTrade):
        self.total_opportunities += 1
        self.total_locked_profit += trade.locked_profit
        
        net = trade.net_profit(self.fee_per_leg)
        fees = trade.locked_profit - net
        
        self.total_fees += fees
        self.total_net_profit += net
        
        if net > 0:
            self.profitable_after_fees += 1
        else:
            self.unprofitable_after_fees += 1
        
        ctype = trade.constraint_type
        if ctype not in self.by_constraint_type:
            self.by_constraint_type[ctype] = ConstraintTypeStats(constraint_type=ctype)
        self.by_constraint_type[ctype].add_trade(trade, self.fee_per_leg)
        
        self.edge_distribution.add_trade(trade, self.fee_per_leg)
        self.all_trades.append(trade)
    
    def finalize(self, keep_top_n: int = 10):
        if self.total_opportunities > 0:
            self.avg_profit_per_opp = self.total_net_profit / self.total_opportunities
        
        self.all_trades.sort(key=lambda t: t.locked_profit, reverse=True)
        self.top_opportunities = self.all_trades[:keep_top_n]
        
        logger.debug(
            "[REPORT] Finalized report: %d opps, %.4f net, %d constraint types",
            self.total_opportunities,
            self.total_net_profit,
            len(self.by_constraint_type)
        )


def generate_report_text(report: ArbitrageBacktestReport) -> str:
    """Generate human-readable text report."""
    logger.debug("[REPORT] Generating text report for %d opportunities", report.total_opportunities)
    start_time = time.time()
    
    lines = []
    sep = "=" * 74
    
    lines.append(sep)
    lines.append("                     ARBITRAGE BACKTEST REPORT")
    lines.append(sep)
    lines.append("")
    
    ps = report.period_start.strftime("%Y-%m-%d") if report.period_start else "N/A"
    pe = report.period_end.strftime("%Y-%m-%d") if report.period_end else "N/A"
    lines.append("Period: {} -> {}".format(ps, pe))
    lines.append("Markets Analyzed: {}".format(report.markets_analyzed))
    lines.append("Clusters Monitored: {}".format(report.clusters_monitored))
    lines.append("")
    
    lines.append("-" * 74)
    lines.append("                         OPPORTUNITY SUMMARY")
    lines.append("-" * 74)
    lines.append("")
    lines.append("Total Opportunities Detected:     {}".format(report.total_opportunities))
    
    for ctype, stats in sorted(report.by_constraint_type.items(), 
                                key=lambda x: x[1].count, reverse=True):
        pct = (stats.count / report.total_opportunities * 100) if report.total_opportunities > 0 else 0
        lines.append("  - {:20} {:6}  ({:5.1f}%)".format(ctype.capitalize(), stats.count, pct))
    
    lines.append("")
    profitable_pct = (report.profitable_after_fees / report.total_opportunities * 100) if report.total_opportunities > 0 else 0
    lines.append("Profitable After Fees:            {:6}  ({:5.1f}%)".format(report.profitable_after_fees, profitable_pct))
    lines.append("Unprofitable After Fees:          {:6}  (fees exceeded edge)".format(report.unprofitable_after_fees))
    lines.append("")
    
    lines.append("-" * 74)
    lines.append("                           PROFIT SUMMARY")
    lines.append("-" * 74)
    lines.append("")
    
    avg_locked = report.total_locked_profit / report.total_opportunities if report.total_opportunities > 0 else 0
    avg_fees = report.total_fees / report.total_opportunities if report.total_opportunities > 0 else 0
    avg_net = report.total_net_profit / report.total_opportunities if report.total_opportunities > 0 else 0
    
    lines.append("Locked Profit (gross):        ${:.4f}/opp    ${:.2f} total".format(avg_locked, report.total_locked_profit))
    lines.append("Transaction Fees:             ${:.4f}/opp    ${:.2f} total".format(avg_fees, report.total_fees))
    lines.append("Net Profit:                   ${:.4f}/opp    ${:.2f} total".format(avg_net, report.total_net_profit))
    lines.append("")
    
    lines.append("By Constraint Type:")
    for ctype, stats in sorted(report.by_constraint_type.items(),
                                key=lambda x: x[1].total_net_profit, reverse=True):
        pct_of_total = (stats.total_net_profit / report.total_net_profit * 100) if report.total_net_profit > 0 else 0
        lines.append("  - {:12} ${:8.2f} net  ({:5.1f}%)   avg edge: {:.1f}%".format(
            ctype.capitalize(), stats.total_net_profit, pct_of_total, stats.avg_edge_pct))
    lines.append("")
    
    lines.append("-" * 74)
    lines.append("                        EDGE SIZE DISTRIBUTION")
    lines.append("-" * 74)
    lines.append("")
    ed = report.edge_distribution
    lines.append("  < 1%:          {:5} opps   ${:8.2f} net".format(ed.bucket_lt_1pct, ed.profit_lt_1pct))
    lines.append("  1% - 2%:       {:5} opps   ${:8.2f} net".format(ed.bucket_1_2pct, ed.profit_1_2pct))
    lines.append("  2% - 5%:       {:5} opps   ${:8.2f} net".format(ed.bucket_2_5pct, ed.profit_2_5pct))
    lines.append("  5% - 10%:      {:5} opps   ${:8.2f} net".format(ed.bucket_5_10pct, ed.profit_5_10pct))
    lines.append("  > 10%:         {:5} opps   ${:8.2f} net  (rare)".format(ed.bucket_gt_10pct, ed.profit_gt_10pct))
    lines.append("")
    
    lines.append("-" * 74)
    lines.append("                          TOP OPPORTUNITIES")
    lines.append("-" * 74)
    lines.append("")
    
    for i, trade in enumerate(report.top_opportunities[:5], 1):
        net = trade.net_profit(report.fee_per_leg)
        edge_pct = trade.violation_amount * 100
        positions_str = ", ".join("{}:{}".format(m, d) for m, d in list(trade.positions.items())[:3])
        
        lines.append("#{:d}  {:12}  +${:.4f} net  edge:{:.1f}%".format(i, trade.constraint_type.upper(), net, edge_pct))
        lines.append("    {}".format(trade.description[:65]))
        lines.append("    Positions: {}".format(positions_str))
        lines.append("")
    
    lines.append(sep)
    
    elapsed = time.time() - start_time
    logger.debug("[REPORT] Text report generated in %.3fs", elapsed)
    
    return "\n".join(lines)


def generate_simplex_visualizations(
    report: ArbitrageBacktestReport,
    output_dir: str,
    max_plots: int = 10,
) -> list:
    """Generate both Bregman dual-panel AND simplex geometry plots.
    
    For each top opportunity:
    1. Bregman dual-panel (existing) - shows projection analysis
    2. Simplex geometry plot (new) - shows market vs coherent on simplex
    
    Also generates a summary grid overview of all top opportunities.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    logger.info(
        "[REPORT] Starting visualization generation: %d opportunities, max_plots=%d",
        len(report.top_opportunities),
        max_plots
    )
    start_time = time.time()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.debug("[REPORT] Output directory: %s", output_path)
    
    saved_files = []
    
    for i, trade in enumerate(report.top_opportunities[:max_plots], 1):
        logger.debug(
            "[REPORT] Generating plots for opportunity #%d: type=%s, profit=%.4f",
            i, trade.constraint_type, trade.locked_profit
        )
        
        # 1. Bregman dual-panel (existing)
        bregman_path = _generate_bregman_plot(trade, i, output_path, report.fee_per_leg)
        if bregman_path:
            saved_files.append(bregman_path)
        
        # 2. Simplex geometry plot (new)
        simplex_path = _generate_simplex_plot(trade, i, output_path, report.fee_per_leg)
        if simplex_path:
            saved_files.append(simplex_path)
    
    # 3. Summary grid of all top opportunities
    if report.top_opportunities:
        logger.debug("[REPORT] Generating summary grid")
        grid_path = _generate_summary_grid(report.top_opportunities, output_path, report.fee_per_leg)
        if grid_path:
            saved_files.append(grid_path)
    
    elapsed = time.time() - start_time
    logger.info(
        "[REPORT] Visualization complete: %d files generated in %.3fs",
        len(saved_files),
        elapsed
    )
    
    return saved_files


def _generate_bregman_plot(
    trade: ArbitrageTrade,
    index: int,
    output_path: Path,
    fee_per_leg: float = 0.01,
) -> Optional[str]:
    """Generate Bregman dual-panel visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    try:
        markets = list(trade.market_prices.keys())
        total = sum(trade.market_prices.values())
        coherent = {m: p / total for m, p in trade.market_prices.items()}
        
        analysis = BregmanAnalysis(
            cluster_id="Opportunity #{}".format(index),
            question=trade.description,
            outcomes=markets,
            observed=[trade.market_prices[m] for m in markets],
            projected=[coherent[m] for m in markets],
            sum_p=total,
            overround=(total - 1.0) * 100,
            kl_divergence=trade.violation_amount,
        )
        
        save_path = output_path / "bregman_{:02d}_{}.png".format(index, trade.constraint_type)
        fig = plot_bregman_dual_panel(analysis, save_path=str(save_path))
        plt.close(fig)
        
        logger.info("[REPORT] Saved Bregman plot: %s", save_path.name)
        return str(save_path)
    except Exception as e:
        logger.warning("[REPORT] Failed to generate Bregman plot #%d: %s", index, e)
        logger.debug("[REPORT] Bregman plot error details", exc_info=True)
        return None


def _generate_simplex_plot(
    trade: ArbitrageTrade,
    index: int,
    output_path: Path,
    fee_per_leg: float = 0.01,
) -> Optional[str]:
    """Generate simplex geometry visualization showing market vs coherent.
    
    Uses the signal_plot module to show:
    - Simplex boundary (feasible region)
    - Market prices (red X, possibly outside)
    - Coherent prices (green dot, on simplex)
    - Edge vector (blue arrow)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    try:
        from ..visualization.signal_plot import plot_arbitrage_signal
        from ..visualization.schema import ArbitrageSignal
        
        # Convert trade to signal format
        markets = list(trade.market_prices.keys())
        total = sum(trade.market_prices.values())
        
        # Coherent prices normalized to sum to 1
        coherent = {m: p / total for m, p in trade.market_prices.items()}
        
        # Calculate direction for each market
        direction = {m: coherent[m] - trade.market_prices[m] for m in markets}
        
        # Create signal for visualization
        signal = ArbitrageSignal(
            timestamp=datetime.now(),
            cluster_id="Opportunity #{}: {}".format(index, trade.constraint_type.upper()),
            markets=markets,
            constraint_type=trade.constraint_type,
            market_prices=trade.market_prices,
            coherent_prices=coherent,
            edge_magnitude=trade.violation_amount,
            kl_divergence=trade.violation_amount,
            direction=direction,
            constraint_violation=trade.description,
            block_number=0,
        )
        
        save_path = output_path / "simplex_{:02d}_{}.png".format(index, trade.constraint_type)
        fig = plot_arbitrage_signal(signal, save_path=str(save_path))
        plt.close(fig)
        
        logger.info("[REPORT] Saved simplex plot: %s", save_path.name)
        return str(save_path)
    except Exception as e:
        logger.warning("[REPORT] Failed to generate simplex plot #%d: %s", index, e)
        logger.debug("[REPORT] Simplex plot error details", exc_info=True)
        return None


def _generate_summary_grid(
    top_opportunities: list,
    output_path: Path,
    fee_per_leg: float = 0.01,
) -> Optional[str]:
    """Generate a summary grid showing all top opportunities.
    
    Creates a multi-panel figure with mini simplex plots for quick overview.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    
    try:
        from ..visualization.simplex import SimplexProjector
        
        n_opps = len(top_opportunities)
        if n_opps == 0:
            logger.debug("[REPORT] No opportunities for summary grid")
            return None
        
        # Calculate grid dimensions
        cols = min(4, n_opps)
        rows = (n_opps + cols - 1) // cols
        
        logger.debug("[REPORT] Creating summary grid: %dx%d for %d opportunities", rows, cols, n_opps)
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if n_opps == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, trade in enumerate(top_opportunities):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            _plot_mini_simplex(ax, trade, idx + 1, fee_per_leg)
        
        # Hide unused axes
        for idx in range(n_opps, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis("off")
        
        plt.suptitle("Top {} Arbitrage Opportunities Overview".format(n_opps), fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        save_path = output_path / "summary_grid.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logger.info("[REPORT] Saved summary grid: %s", save_path.name)
        return str(save_path)
    except Exception as e:
        logger.warning("[REPORT] Failed to generate summary grid: %s", e)
        logger.debug("[REPORT] Summary grid error details", exc_info=True)
        return None


def _plot_mini_simplex(ax, trade: ArbitrageTrade, index: int, fee_per_leg: float = 0.01):
    """Plot a mini simplex visualization for the summary grid."""
    import numpy as np
    from matplotlib.patches import Polygon
    
    try:
        from ..visualization.simplex import SimplexProjector
        
        markets = list(trade.market_prices.keys())
        n = len(markets)
        
        proj = SimplexProjector(n)
        
        # Draw simplex
        vertices = proj.vertices
        simplex = Polygon(vertices, closed=True, fill=True,
                          facecolor="lightgreen", edgecolor="darkgreen",
                          alpha=0.2, linewidth=1)
        ax.add_patch(simplex)
        
        # Get price vectors
        market_vec = np.array([trade.market_prices[m] for m in markets])
        total = sum(trade.market_prices.values())
        coherent_vec = market_vec / total
        
        # Project to 2D
        if abs(total - 1.0) > 0.01:
            market_2d = proj.to_2d(market_vec / market_vec.sum()) * total
        else:
            market_2d = proj.to_2d(market_vec)
        coherent_2d = proj.to_2d(coherent_vec)
        
        # Plot points
        ax.scatter(*market_2d, c="red", s=60, marker="X", zorder=10)
        ax.scatter(*coherent_2d, c="green", s=60, marker="o", zorder=10)
        
        # Draw arrow
        ax.annotate("", xy=coherent_2d, xytext=market_2d,
                    arrowprops=dict(arrowstyle="->", color="blue", lw=1.5), zorder=9)
        
        # Set limits
        margin = 0.2
        ax.set_xlim(vertices[:, 0].min() - margin, vertices[:, 0].max() + margin)
        ax.set_ylim(vertices[:, 1].min() - margin, vertices[:, 1].max() + margin)
        
        # Title with profit info
        net_profit = trade.net_profit(fee_per_leg)
        edge_pct = trade.violation_amount * 100
        title_str = "#{} {} +${:.3f}\nedge:{:.1f}%".format(
            index, trade.constraint_type[:8], net_profit, edge_pct)
        ax.set_title(title_str, fontsize=9)
        
        ax.set_aspect("equal")
        ax.axis("off")
    except Exception as e:
        logger.debug("[REPORT] Mini simplex error for #%d: %s", index, e)
        ax.text(0.5, 0.5, "Error", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("#{} Error".format(index), fontsize=9)
        ax.axis("off")


def generate_full_report(
    trades: list,
    output_dir: str,
    period_start=None,
    period_end=None,
    markets_analyzed: int = 0,
    clusters_monitored: int = 0,
    fee_per_leg: float = 0.01,
):
    """Generate complete backtest report with text and visualizations.
    
    Args:
        trades: List of ArbitrageTrade objects
        output_dir: Directory to save reports
        period_start: Start of backtest period
        period_end: End of backtest period
        markets_analyzed: Number of markets analyzed
        clusters_monitored: Number of clusters monitored
        fee_per_leg: Transaction fee per leg
        
    Returns:
        Tuple of (report, report_text, viz_files)
    """
    logger.info(
        "[REPORT] Generating full report: %d trades, output=%s",
        len(trades),
        output_dir
    )
    start_time = time.time()
    
    report = ArbitrageBacktestReport(
        period_start=period_start,
        period_end=period_end,
        markets_analyzed=markets_analyzed,
        clusters_monitored=clusters_monitored,
        fee_per_leg=fee_per_leg,
    )
    
    logger.debug("[REPORT] Adding %d trades to report", len(trades))
    for trade in trades:
        report.add_trade(trade)
    
    report.finalize(keep_top_n=10)
    
    logger.debug(
        "[REPORT] Report stats: total_opps=%d, profitable=%d, net_pnl=%.4f",
        report.total_opportunities,
        report.profitable_after_fees,
        report.total_net_profit
    )
    
    report_text = generate_report_text(report)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    text_file = output_path / "backtest_report.txt"
    with open(text_file, "w") as f:
        f.write(report_text)
    logger.info("[REPORT] Saved text report: %s", text_file)
    
    viz_files = generate_simplex_visualizations(report, output_dir)
    
    elapsed = time.time() - start_time
    logger.info(
        "[REPORT] Full report complete in %.3fs: text + %d visualizations",
        elapsed,
        len(viz_files)
    )
    
    return report, report_text, viz_files
