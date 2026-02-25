"""Bregman Projection visualization for arbitrage analysis.

This module provides dual-panel visualizations showing:
1. Price correction bar charts (delta between projected and observed prices)
2. Simplex geometry radar charts (visual comparison of market vs fair prices)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from .schema import ArbitrageSignal


@dataclass
class BregmanAnalysis:
    """Analysis results for Bregman projection visualization.

    Attributes:
        cluster_id: Unique identifier for the market cluster
        question: The market question being analyzed
        outcomes: List of outcome labels in the cluster
        observed: Market prices for each outcome
        projected: Coherent/fair prices after Bregman projection
        sum_p: Sum of observed probabilities (should be 1.0 for fair market)
        overround: (sum_p - 1) * 100 -- positive means overpriced, negative means underpriced
        kl_divergence: KL divergence between observed and projected distributions
        iterations: Number of solver iterations to reach convergence
    """
    cluster_id: str
    question: str
    outcomes: List[str]
    observed: List[float]
    projected: List[float]
    sum_p: float
    overround: float
    kl_divergence: float
    iterations: int = 0


def compute_bregman_analysis(
    cluster_id: str,
    question: str,
    outcomes: List[str],
    market_prices: dict[str, float],
    coherent_prices: dict[str, float],
    kl_divergence: float = 0.0,
    iterations: int = 0,
) -> BregmanAnalysis:
    """Compute Bregman analysis from market and coherent prices.

    Args:
        cluster_id: Unique identifier for the cluster
        question: The market question
        outcomes: List of outcome names
        market_prices: Dict mapping outcome -> observed market price
        coherent_prices: Dict mapping outcome -> projected coherent price
        kl_divergence: Pre-computed KL divergence (optional)
        iterations: Number of solver iterations (optional)

    Returns:
        BregmanAnalysis object with computed metrics
    """
    observed = [market_prices.get(o, 0.0) for o in outcomes]
    projected = [coherent_prices.get(o, 0.0) for o in outcomes]
    sum_p = sum(observed)
    overround = (sum_p - 1.0) * 100

    return BregmanAnalysis(
        cluster_id=cluster_id,
        question=question,
        outcomes=outcomes,
        observed=observed,
        projected=projected,
        sum_p=sum_p,
        overround=overround,
        kl_divergence=kl_divergence,
        iterations=iterations,
    )


def signal_to_bregman(signal: ArbitrageSignal, question: str = "") -> BregmanAnalysis:
    """Convert ArbitrageSignal to BregmanAnalysis for visualization.

    Args:
        signal: ArbitrageSignal object from arbitrage detection
        question: Optional market question (defaults to cluster_id if not provided)

    Returns:
        BregmanAnalysis object ready for visualization
    """
    return compute_bregman_analysis(
        cluster_id=signal.cluster_id,
        question=question or signal.cluster_id,
        outcomes=signal.markets,
        market_prices=signal.market_prices,
        coherent_prices=signal.coherent_prices,
        kl_divergence=signal.kl_divergence,
    )


def plot_bregman_dual_panel(
    analysis: BregmanAnalysis,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create dual-panel Bregman projection visualization.

    Left panel: Price correction bar chart (Delta = Projected - Observed)
        - Green bars indicate underpriced outcomes (BUY signal)
        - Red bars indicate overpriced outcomes (SELL signal)
        - Annotations show the price transition (observed -> projected)

    Right panel: Simplex geometry radar chart
        - Green filled area shows observed market prices
        - Red dashed outline shows projected coherent prices
        - Gap between them visualizes the mispricing

    Args:
        analysis: BregmanAnalysis object with computed metrics
        figsize: Figure size as (width, height) tuple
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=figsize)

    # Use GridSpec for 1 row, 2 columns with space for title
    gs = fig.add_gridspec(1, 2, wspace=0.3, left=0.08, right=0.95, top=0.85, bottom=0.15)

    # Left panel: Bar chart (regular axes)
    ax_bar = fig.add_subplot(gs[0, 0])
    _plot_correction_bars(ax_bar, analysis)

    # Right panel: Radar chart (polar axes)
    ax_radar = fig.add_subplot(gs[0, 1], projection='polar')
    _plot_simplex_radar(ax_radar, analysis)

    # Build title with statistics
    overround_str = f"+{analysis.overround:.1f}%" if analysis.overround >= 0 else f"{analysis.overround:.1f}%"

    # Truncate question if too long
    question_display = analysis.question[:60] + "..." if len(analysis.question) > 60 else analysis.question

    title = (
        f"{analysis.cluster_id}: {question_display}\n"
        f"[{len(analysis.outcomes)} outcomes, "
        f"\u03A3p={analysis.sum_p:.3f}, "
        f"overround={overround_str}, "
        f"KL={analysis.kl_divergence:.2e}"
    )
    if analysis.iterations > 0:
        title += f", {analysis.iterations} iter"
    title += "]"

    fig.suptitle(title, fontsize=11, fontweight='bold', y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def _plot_correction_bars(ax: plt.Axes, analysis: BregmanAnalysis):
    """Plot price correction bar chart.

    Shows delta (Projected - Observed) for each outcome.
    Green bars = underpriced (buy signal), Red bars = overpriced (sell signal).
    """
    n = len(analysis.outcomes)
    deltas = [(p - o) * 100 for o, p in zip(analysis.observed, analysis.projected)]

    # Colors: green for underpriced (buy), red for overpriced (sell)
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in deltas]

    y_pos = np.arange(n)
    bars = ax.barh(y_pos, deltas, color=colors, edgecolor='black', linewidth=0.5, height=0.7)

    # Labels for outcomes
    truncated_labels = [o[:30] + '...' if len(o) > 30 else o for o in analysis.outcomes]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(truncated_labels, fontsize=9)
    ax.set_xlabel('\u0394 = Projected - Observed (%)', fontsize=10)
    ax.axvline(x=0, color='black', linewidth=1.5)

    # Add value labels on bars showing transition
    max_abs_delta = max(abs(d) for d in deltas) if deltas and max(abs(d) for d in deltas) > 0 else 1
    for i, (bar, delta, obs, proj) in enumerate(zip(bars, deltas, analysis.observed, analysis.projected)):
        label = f'{obs:.1%}\u2192{proj:.1%}'

        # Position label outside the bar
        if delta >= 0:
            x_pos = delta + max_abs_delta * 0.05
            ha = 'left'
        else:
            x_pos = delta - max_abs_delta * 0.05
            ha = 'right'

        ax.annotate(label, (x_pos, i), va='center', ha=ha, fontsize=8, color='#555555')

    # Set symmetric x-limits with some padding
    padding = max_abs_delta * 0.4
    ax.set_xlim(-max_abs_delta - padding, max_abs_delta + padding)

    ax.set_title('Price Correction per Outcome', fontsize=10, fontweight='bold')

    # Add grid for readability
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Legend
    buy_patch = mpatches.Patch(color='#2ecc71', label='Underpriced (BUY)')
    sell_patch = mpatches.Patch(color='#e74c3c', label='Overpriced (SELL)')
    ax.legend(handles=[buy_patch, sell_patch], loc='lower right', fontsize=8)


def _plot_simplex_radar(ax: plt.Axes, analysis: BregmanAnalysis):
    """Plot simplex geometry as radar/spider chart.

    Shows observed prices as green filled area and projected prices as red outline.
    The gap between them visualizes the mispricing direction and magnitude.
    """
    n = len(analysis.outcomes)

    # Compute angles for radar chart (evenly spaced around circle)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    # Close the data loops
    observed = analysis.observed + [analysis.observed[0]]
    projected = analysis.projected + [analysis.projected[0]]

    # Plot observed (market) prices - green fill
    ax.fill(angles, observed, color='#2ecc71', alpha=0.35, label='Observed (Market)')
    ax.plot(angles, observed, color='#27ae60', linewidth=2.5, marker='o', markersize=6)

    # Plot projected (coherent) prices - red outline/fill
    ax.fill(angles, projected, color='#e74c3c', alpha=0.15)
    ax.plot(angles, projected, color='#c0392b', linewidth=2.5, linestyle='--',
            marker='s', markersize=5, label='Projected (Coherent)')

    # Set labels at vertices
    ax.set_xticks(angles[:-1])
    truncated_labels = [o[:15] + '..' if len(o) > 15 else o for o in analysis.outcomes]
    ax.set_xticklabels(truncated_labels, fontsize=8)

    # Set radial limits based on data
    max_val = max(max(analysis.observed), max(analysis.projected)) * 1.1
    ax.set_ylim(0, max(max_val, 0.5))

    # Add overround annotation at bottom
    overround_str = f"+{analysis.overround:.1f}%" if analysis.overround >= 0 else f"{analysis.overround:.1f}%"
    ax.annotate(
        f'\u03A3p={analysis.sum_p:.3f} ({overround_str})',
        xy=(0.5, -0.12), xycoords='axes fraction',
        ha='center', fontsize=9, fontweight='bold'
    )

    ax.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.15, 1.15))
    ax.set_title('Simplex Geometry', fontsize=10, fontweight='bold', pad=15)


def plot_bregman_report(
    analyses: List[BregmanAnalysis],
    save_path: Optional[str] = None,
    max_per_page: int = 4,
) -> List[plt.Figure]:
    """Generate multi-page Bregman report for multiple clusters.

    Creates a paginated report with dual-panel visualizations for each cluster.

    Args:
        analyses: List of BregmanAnalysis objects to visualize
        save_path: Optional base path for saving figures (adds _page1, _page2, etc.)
        max_per_page: Maximum number of clusters per page

    Returns:
        List of matplotlib Figure objects
    """
    figures = []

    for page_num, i in enumerate(range(0, len(analyses), max_per_page)):
        batch = analyses[i:i + max_per_page]
        n_rows = len(batch)

        # Create figure with subplots for each row
        fig = plt.figure(figsize=(14, 5 * n_rows))

        for row, analysis in enumerate(batch):
            # Calculate vertical positions for this row
            row_top = 1.0 - (row / n_rows)
            row_bottom = 1.0 - ((row + 1) / n_rows)

            # Add some padding
            row_top -= 0.02
            row_bottom += 0.03

            # Create GridSpec for this row
            gs_row = fig.add_gridspec(
                1, 2,
                wspace=0.3,
                left=0.08, right=0.95,
                top=row_top,
                bottom=row_bottom
            )

            # Left panel: Bar chart
            ax_bar = fig.add_subplot(gs_row[0, 0])
            _plot_correction_bars(ax_bar, analysis)

            # Right panel: Radar chart
            ax_radar = fig.add_subplot(gs_row[0, 1], projection='polar')
            _plot_simplex_radar(ax_radar, analysis)

            # Add row title
            overround_str = f"+{analysis.overround:.1f}%" if analysis.overround >= 0 else f"{analysis.overround:.1f}%"
            question_short = analysis.question[:40] + "..." if len(analysis.question) > 40 else analysis.question
            row_title = f"{analysis.cluster_id}: {question_short}"
            ax_bar.set_title(
                f"{row_title}\n[{len(analysis.outcomes)} outcomes, \u03A3p={analysis.sum_p:.3f}]",
                fontsize=9, fontweight='bold', loc='left'
            )

        fig.suptitle('Bregman Projection - Polymarket Mispricing Analysis',
                    fontsize=14, fontweight='bold', y=0.995)

        figures.append(fig)

        if save_path:
            page_path = save_path.replace('.png', f'_page{page_num + 1}.png')
            fig.savefig(page_path, dpi=150, bbox_inches='tight')

    return figures


def plot_single_cluster_summary(
    analysis: BregmanAnalysis,
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create a comprehensive single-cluster summary with three panels.

    Adds a third panel showing the trade recommendations table.

    Args:
        analysis: BregmanAnalysis object
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, wspace=0.3, left=0.05, right=0.95, top=0.85, bottom=0.15)

    # Left panel: Bar chart
    ax_bar = fig.add_subplot(gs[0, 0])
    _plot_correction_bars(ax_bar, analysis)

    # Middle panel: Radar chart
    ax_radar = fig.add_subplot(gs[0, 1], projection='polar')
    _plot_simplex_radar(ax_radar, analysis)

    # Right panel: Trade recommendations table
    ax_table = fig.add_subplot(gs[0, 2])
    _plot_trade_table(ax_table, analysis)

    # Title
    overround_str = f"+{analysis.overround:.1f}%" if analysis.overround >= 0 else f"{analysis.overround:.1f}%"
    question_display = analysis.question[:55] + "..." if len(analysis.question) > 55 else analysis.question

    fig.suptitle(
        f"{analysis.cluster_id}: {question_display}\n"
        f"[{len(analysis.outcomes)} outcomes, \u03A3p={analysis.sum_p:.3f}, "
        f"overround={overround_str}, KL={analysis.kl_divergence:.2e}]",
        fontsize=11, fontweight='bold', y=0.98
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def _plot_trade_table(ax: plt.Axes, analysis: BregmanAnalysis):
    """Plot trade recommendations table."""
    ax.axis('off')

    # Prepare table data
    table_data = []
    for i, outcome in enumerate(analysis.outcomes):
        obs = analysis.observed[i]
        proj = analysis.projected[i]
        delta = (proj - obs) * 100
        action = "BUY" if delta > 0 else "SELL"
        edge = abs(delta)

        table_data.append([
            outcome[:25] + '...' if len(outcome) > 25 else outcome,
            f'{obs:.1%}',
            f'{proj:.1%}',
            f'{delta:+.1f}%',
            action,
            f'{edge:.1f}%'
        ])

    # Sort by edge magnitude (descending)
    table_data.sort(key=lambda x: float(x[5].rstrip('%')), reverse=True)

    # Column headers
    columns = ['Outcome', 'Market', 'Fair', '\u0394', 'Action', 'Edge']

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.35, 0.12, 0.12, 0.13, 0.13, 0.13]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)

    # Color code action column
    for i, row in enumerate(table_data):
        action = row[4]
        cell = table[(i + 1, 4)]  # +1 for header row
        if action == "BUY":
            cell.set_facecolor('#d5f5e3')
        else:
            cell.set_facecolor('#fadbd8')

    # Style header
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Trade Recommendations', fontsize=10, fontweight='bold', pad=10)
