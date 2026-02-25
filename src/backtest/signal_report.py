"""Generate signal-only reports proving arbitrage exists."""
from typing import List
from datetime import datetime
from collections import Counter
from src.visualization.schema import ArbitrageSignal

def generate_signal_report(signals: List[ArbitrageSignal]) -> str:
    """Generate text report of arbitrage signals."""
    if not signals:
        return "No arbitrage signals detected."
    
    # Statistics
    by_type = Counter(s.constraint_type for s in signals)
    by_cluster = Counter(s.cluster_id for s in signals)
    
    # Sort by edge magnitude
    top_signals = sorted(signals, key=lambda s: s.edge_magnitude, reverse=True)[:10]
    
    lines = [
        "=" * 70,
        "ARBITRAGE SIGNAL REPORT",
        "=" * 70,
        "",
        f"Period: {min(s.timestamp for s in signals)} to {max(s.timestamp for s in signals)}",
        f"Total Signals: {len(signals)}",
        "",
        "By Constraint Type:",
    ]
    
    for ctype, count in by_type.most_common():
        pct = count / len(signals) * 100
        lines.append(f"  - {ctype}: {count} ({pct:.1f}%)")
    
    lines.extend([
        "",
        "Top 10 Signals by Edge Magnitude:",
        "-" * 70,
    ])
    
    for i, sig in enumerate(top_signals, 1):
        lines.append(
            f"{i:2}. {sig.timestamp} | {sig.cluster_id[:20]:20} | "
            f"Edge: {sig.edge_magnitude:.4f} | {sig.constraint_violation[:30]}"
        )
    
    lines.extend([
        "",
        "=" * 70,
        f"PROOF: {len(signals)} instances where market prices violated logical constraints.",
        "=" * 70,
    ])
    
    return "\n".join(lines)
