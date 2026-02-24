"""Report Generation Module (BT-04).

Generates summary statistics and formatted reports from backtest results.
"""

from datetime import datetime, timedelta
from typing import Optional
import statistics

from .schema import (
    ArbitrageOpportunity,
    BacktestReport,
    ClusterPerformance,
)
from .pnl import PnLTracker


def generate_report(
    opportunities: list[ArbitrageOpportunity],
    start_date: datetime,
    end_date: datetime,
    markets_analyzed: int,
    clusters_found: int,
    cluster_themes: dict[str, str],  # cluster_id -> theme
    cluster_market_ids: dict[str, list[str]],  # cluster_id -> market_ids
    kl_threshold: float = 0.01,
    transaction_cost_rate: float = 0.015,
    store_all_opportunities: bool = True,
) -> BacktestReport:
    """Generate a complete backtest report.
    
    Args:
        opportunities: List of detected opportunities
        start_date: Backtest start time
        end_date: Backtest end time
        markets_analyzed: Number of markets in the backtest
        clusters_found: Number of clusters identified
        cluster_themes: Mapping of cluster_id to theme description
        cluster_market_ids: Mapping of cluster_id to market IDs
        kl_threshold: KL threshold used
        transaction_cost_rate: Transaction cost rate used
        store_all_opportunities: Whether to include all opportunities in report
        
    Returns:
        BacktestReport with all statistics
    """
    # Calculate duration
    duration = end_date - start_date
    duration_hours = duration.total_seconds() / 3600.0
    
    # Aggregate metrics
    total_opportunities = len(opportunities)
    opportunities_per_hour = total_opportunities / duration_hours if duration_hours > 0 else 0
    
    # PnL metrics
    gross_pnl = sum(o.theoretical_profit for o in opportunities)
    net_pnl = sum(o.net_profit for o in opportunities)
    transaction_costs = gross_pnl - net_pnl
    
    # Win/loss
    win_count = sum(1 for o in opportunities if o.net_profit > 0)
    loss_count = sum(1 for o in opportunities if o.net_profit < 0)
    win_rate = win_count / total_opportunities if total_opportunities > 0 else 0
    
    # KL divergence stats
    if opportunities:
        kl_values = [o.kl_divergence for o in opportunities]
        avg_kl = statistics.mean(kl_values)
        max_kl = max(kl_values)
        min_kl = min(kl_values)
    else:
        avg_kl = max_kl = min_kl = 0.0
    
    # Calculate max drawdown
    max_drawdown = calculate_max_drawdown(opportunities)
    
    # Find best/worst trades
    best_trade = None
    worst_trade = None
    if opportunities:
        sorted_by_net = sorted(opportunities, key=lambda x: x.net_profit)
        worst_trade = sorted_by_net[0]
        best_trade = sorted_by_net[-1]
    
    # Calculate cluster-level performance
    cluster_performance = calculate_cluster_performance(
        opportunities=opportunities,
        cluster_themes=cluster_themes,
        cluster_market_ids=cluster_market_ids,
    )
    
    return BacktestReport(
        start_date=start_date,
        end_date=end_date,
        duration_hours=duration_hours,
        markets_analyzed=markets_analyzed,
        clusters_found=clusters_found,
        total_opportunities=total_opportunities,
        opportunities_per_hour=opportunities_per_hour,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        transaction_costs=transaction_costs,
        win_count=win_count,
        loss_count=loss_count,
        win_rate=win_rate,
        avg_kl_divergence=avg_kl,
        max_kl_divergence=max_kl,
        min_kl_divergence=min_kl,
        max_drawdown=max_drawdown,
        best_trade=best_trade,
        worst_trade=worst_trade,
        cluster_performance=cluster_performance,
        opportunities=opportunities if store_all_opportunities else [],
        kl_threshold=kl_threshold,
        transaction_cost_rate=transaction_cost_rate,
    )


def calculate_max_drawdown(opportunities: list[ArbitrageOpportunity]) -> float:
    """Calculate maximum drawdown from cumulative PnL.
    
    Args:
        opportunities: List of opportunities in chronological order
        
    Returns:
        Maximum drawdown value
    """
    if not opportunities:
        return 0.0
    
    cumulative_pnl = 0.0
    peak_pnl = 0.0
    max_drawdown = 0.0
    
    for opp in opportunities:
        cumulative_pnl += opp.net_profit
        if cumulative_pnl > peak_pnl:
            peak_pnl = cumulative_pnl
        
        drawdown = peak_pnl - cumulative_pnl
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    return max_drawdown


def calculate_cluster_performance(
    opportunities: list[ArbitrageOpportunity],
    cluster_themes: dict[str, str],
    cluster_market_ids: dict[str, list[str]],
) -> list[ClusterPerformance]:
    """Calculate performance metrics for each cluster.
    
    Args:
        opportunities: All opportunities
        cluster_themes: Mapping of cluster_id to theme
        cluster_market_ids: Mapping of cluster_id to market IDs
        
    Returns:
        List of ClusterPerformance objects
    """
    # Group opportunities by cluster
    by_cluster: dict[str, list[ArbitrageOpportunity]] = {}
    for opp in opportunities:
        if opp.cluster_id not in by_cluster:
            by_cluster[opp.cluster_id] = []
        by_cluster[opp.cluster_id].append(opp)
    
    # Calculate performance for each cluster
    performance_list = []
    for cluster_id, cluster_opps in by_cluster.items():
        gross_pnl = sum(o.theoretical_profit for o in cluster_opps)
        net_pnl = sum(o.net_profit for o in cluster_opps)
        win_count = sum(1 for o in cluster_opps if o.net_profit > 0)
        loss_count = sum(1 for o in cluster_opps if o.net_profit < 0)
        
        kl_values = [o.kl_divergence for o in cluster_opps]
        avg_kl = statistics.mean(kl_values) if kl_values else 0.0
        max_kl = max(kl_values) if kl_values else 0.0
        
        performance_list.append(ClusterPerformance(
            cluster_id=cluster_id,
            theme=cluster_themes.get(cluster_id, "Unknown"),
            market_ids=cluster_market_ids.get(cluster_id, []),
            num_opportunities=len(cluster_opps),
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            win_count=win_count,
            loss_count=loss_count,
            avg_kl_divergence=avg_kl,
            max_kl_divergence=max_kl,
        ))
    
    # Sort by net PnL descending
    performance_list.sort(key=lambda x: x.net_pnl, reverse=True)
    
    return performance_list


def format_report(report: BacktestReport) -> str:
    """Format a backtest report as a human-readable string.
    
    Args:
        report: The BacktestReport to format
        
    Returns:
        Formatted string report
    """
    lines = [
        "=" * 60,
        "BACKTEST REPORT",
        "=" * 60,
        "",
        "Time Range",
        "-" * 40,
        f"  Start:    {report.start_date.isoformat()}",
        f"  End:      {report.end_date.isoformat()}",
        f"  Duration: {report.duration_hours:.1f} hours",
        "",
        "Market Coverage",
        "-" * 40,
        f"  Markets Analyzed: {report.markets_analyzed}",
        f"  Clusters Found:   {report.clusters_found}",
        "",
        "Opportunities",
        "-" * 40,
        f"  Total Found:    {report.total_opportunities}",
        f"  Per Hour:       {report.opportunities_per_hour:.2f}",
        "",
        "PnL Summary (per $1 stake per market)",
        "-" * 40,
        f"  Gross PnL:          ${report.gross_pnl:,.4f}",
        f"  Transaction Costs:  ${report.transaction_costs:,.4f}",
        f"  Net PnL:            ${report.net_pnl:,.4f}",
        "",
        "Win/Loss Analysis",
        "-" * 40,
        f"  Wins:       {report.win_count}",
        f"  Losses:     {report.loss_count}",
        f"  Win Rate:   {report.win_rate * 100:.1f}%",
        "",
        "KL Divergence Statistics",
        "-" * 40,
        f"  Average:  {report.avg_kl_divergence:.6f}",
        f"  Maximum:  {report.max_kl_divergence:.6f}",
        f"  Minimum:  {report.min_kl_divergence:.6f}",
        "",
        "Risk Metrics",
        "-" * 40,
        f"  Max Drawdown:  ${report.max_drawdown:,.4f}",
    ]
    
    # Best/Worst trades
    if report.best_trade:
        lines.extend([
            "",
            "Best Trade",
            "-" * 40,
            f"  Cluster:    {report.best_trade.cluster_id}",
            f"  Net Profit: ${report.best_trade.net_profit:,.4f}",
            f"  KL Div:     {report.best_trade.kl_divergence:.6f}",
            f"  Time:       {report.best_trade.timestamp.isoformat()}",
        ])
    
    if report.worst_trade:
        lines.extend([
            "",
            "Worst Trade",
            "-" * 40,
            f"  Cluster:    {report.worst_trade.cluster_id}",
            f"  Net Profit: ${report.worst_trade.net_profit:,.4f}",
            f"  KL Div:     {report.worst_trade.kl_divergence:.6f}",
            f"  Time:       {report.worst_trade.timestamp.isoformat()}",
        ])
    
    # Cluster performance
    if report.cluster_performance:
        lines.extend([
            "",
            "Cluster Performance",
            "-" * 40,
        ])
        for cp in report.cluster_performance[:5]:  # Top 5 clusters
            lines.append(
                f"  {cp.cluster_id[:20]:<20} | "
                f"Opps: {cp.num_opportunities:>4} | "
                f"Net: ${cp.net_pnl:>8.4f} | "
                f"Win: {cp.win_rate * 100:>5.1f}%"
            )
    
    lines.extend([
        "",
        "Configuration",
        "-" * 40,
        f"  KL Threshold:      {report.kl_threshold}",
        f"  Transaction Cost:  {report.transaction_cost_rate * 100:.1f}%",
        "",
        "=" * 60,
    ])
    
    return "\n".join(lines)


def report_to_dict(report: BacktestReport) -> dict:
    """Convert report to a dictionary for JSON serialization.
    
    Args:
        report: The BacktestReport to convert
        
    Returns:
        Dictionary representation
    """
    return {
        "time_range": {
            "start_date": report.start_date.isoformat(),
            "end_date": report.end_date.isoformat(),
            "duration_hours": report.duration_hours,
        },
        "coverage": {
            "markets_analyzed": report.markets_analyzed,
            "clusters_found": report.clusters_found,
        },
        "opportunities": {
            "total": report.total_opportunities,
            "per_hour": report.opportunities_per_hour,
        },
        "pnl": {
            "gross": report.gross_pnl,
            "net": report.net_pnl,
            "transaction_costs": report.transaction_costs,
        },
        "win_loss": {
            "wins": report.win_count,
            "losses": report.loss_count,
            "win_rate": report.win_rate,
        },
        "kl_divergence": {
            "average": report.avg_kl_divergence,
            "maximum": report.max_kl_divergence,
            "minimum": report.min_kl_divergence,
        },
        "risk": {
            "max_drawdown": report.max_drawdown,
        },
        "config": {
            "kl_threshold": report.kl_threshold,
            "transaction_cost_rate": report.transaction_cost_rate,
        },
    }
