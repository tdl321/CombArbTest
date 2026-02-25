"""Plot arbitrage signals on simplex."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Optional, List
from .simplex import SimplexProjector
from .schema import ArbitrageSignal

def plot_arbitrage_signal(
    signal: ArbitrageSignal,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 10),
) -> plt.Figure:
    """
    Visualize single arbitrage signal on simplex.
    
    Shows:
    - Simplex boundary (feasible region)
    - Market prices (red X, possibly outside)
    - Coherent prices (green dot, on simplex)
    - Edge vector (blue arrow)
    """
    markets = signal.markets
    n = len(markets)
    proj = SimplexProjector(n, markets)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw simplex
    _draw_simplex(ax, proj)
    
    # Get price vectors
    market_vec = np.array([signal.market_prices[m] for m in markets])
    coherent_vec = np.array([signal.coherent_prices[m] for m in markets])
    
    # Project to 2D
    # For market prices, we need special handling if sum != 1
    market_sum = market_vec.sum()
    if abs(market_sum - 1.0) > 0.01:
        # Scale to show position relative to simplex
        market_2d = proj.to_2d(market_vec / market_vec.sum()) * market_sum
    else:
        market_2d = proj.to_2d(market_vec)
    
    coherent_2d = proj.to_2d(coherent_vec)
    
    # Plot market point (red X)
    ax.scatter(*market_2d, c='red', s=200, marker='X', 
               label=f'Market (sum={market_sum:.3f})', zorder=10)
    
    # Plot coherent point (green dot)
    ax.scatter(*coherent_2d, c='green', s=200, marker='o',
               label='Coherent (sum=1.0)', zorder=10)
    
    # Draw edge arrow
    ax.annotate('', xy=coherent_2d, xytext=market_2d,
                arrowprops=dict(arrowstyle='->', color='blue', lw=3),
                zorder=9)
    
    # Edge magnitude label
    mid = (market_2d + coherent_2d) / 2
    ax.annotate(f'Edge: {signal.edge_magnitude:.4f}\nKL: {signal.kl_divergence:.4f}',
                mid, fontsize=12, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Title and labels
    ax.set_title(f'{signal.cluster_id}\n{signal.constraint_violation}', fontsize=14)
    ax.legend(loc='upper right')
    
    # Add market labels at vertices
    for i, (label, vertex) in enumerate(zip(markets, proj.vertices)):
        price_m = signal.market_prices[label]
        price_c = signal.coherent_prices[label]
        direction = signal.direction[label]
        action = "BUY" if direction > 0 else "SELL"
        ax.annotate(f'{label[:20]}\nM:{price_m:.2%}\nC:{price_c:.2%}\n{action}',
                    vertex, fontsize=9, ha='center', va='bottom')
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def _draw_simplex(ax: plt.Axes, proj: SimplexProjector, alpha: float = 0.2):
    """Draw simplex as shaded region."""
    from matplotlib.patches import Polygon
    
    vertices = proj.vertices
    
    # Fill simplex
    simplex = Polygon(vertices, closed=True, fill=True,
                      facecolor='lightgreen', edgecolor='darkgreen',
                      alpha=alpha, linewidth=2, label='Feasible Region')
    ax.add_patch(simplex)
    
    # Extend axes to show points outside
    margin = 0.3
    ax.set_xlim(vertices[:, 0].min() - margin, vertices[:, 0].max() + margin)
    ax.set_ylim(vertices[:, 1].min() - margin, vertices[:, 1].max() + margin)


def plot_signal_batch(
    signals: List[ArbitrageSignal],
    max_plots: int = 9,
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """Plot multiple signals in a grid."""
    n_plots = min(len(signals), max_plots)
    cols = 3
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for i, (ax, signal) in enumerate(zip(axes, signals[:n_plots])):
        _plot_signal_mini(ax, signal)
    
    # Hide unused axes
    for ax in axes[n_plots:]:
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{save_dir}/signals_batch.png', dpi=150, bbox_inches='tight')
    
    return fig


def _plot_signal_mini(ax: plt.Axes, signal: ArbitrageSignal):
    """Mini version for grid plots."""
    markets = signal.markets
    n = len(markets)
    proj = SimplexProjector(n)
    
    _draw_simplex(ax, proj, alpha=0.15)
    
    market_vec = np.array([signal.market_prices[m] for m in markets])
    coherent_vec = np.array([signal.coherent_prices[m] for m in markets])
    
    market_sum = market_vec.sum()
    if abs(market_sum - 1.0) > 0.01:
        market_2d = proj.to_2d(market_vec / market_vec.sum()) * market_sum
    else:
        market_2d = proj.to_2d(market_vec)
    coherent_2d = proj.to_2d(coherent_vec)
    
    ax.scatter(*market_2d, c='red', s=80, marker='X', zorder=10)
    ax.scatter(*coherent_2d, c='green', s=80, marker='o', zorder=10)
    ax.annotate('', xy=coherent_2d, xytext=market_2d,
                arrowprops=dict(arrowstyle='->', color='blue', lw=2), zorder=9)
    
    ax.set_title(f'{signal.cluster_id[:25]}\nEdge: {signal.edge_magnitude:.4f}', fontsize=10)
    ax.set_aspect('equal')
    ax.axis('off')
