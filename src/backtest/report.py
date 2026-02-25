"""Report Generation Module (BT-04).

Generates summary statistics and formatted reports from backtest results.
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Optional

from .schema import (
    ArbitrageOpportunity,
    BacktestReport,
    ClusterPerformance,
)
from .pnl import PnLTracker

logger = logging.getLogger(__name__)


def generate_report(
    opportunities: list[ArbitrageOpportunity],
    start_date: datetime,
    end_date: datetime,
    markets_analyzed: int,
    clusters_found: int,
    cluster_themes: dict[str, str],
    cluster_market_ids: dict[str, list[str]],
    kl_threshold: float = 0.01,
    transaction_cost_rate: float = 0.015,
    store_all_opportunities: bool = True,
) -> BacktestReport:
    """Generate a complete backtest report."""
    logger.info("[REPORT] Generating report: %d opportunities, %d markets, %d clusters",
                len(opportunities), markets_analyzed, clusters_found)

    duration = end_date - start_date
    duration_hours = duration.total_seconds() / 3600.0

    total_opportunities = len(opportunities)
    opportunities_per_hour = total_opportunities / duration_hours if duration_hours > 0 else 0

    gross_pnl = sum(o.theoretical_profit for o in opportunities)
    net_pnl = sum(o.net_profit() for o in opportunities)
    transaction_costs = gross_pnl - net_pnl

    win_count = sum(1 for o in opportunities if o.net_profit() > 0)
    loss_count = sum(1 for o in opportunities if o.net_profit() < 0)
    win_rate = win_count / total_opportunities if total_opportunities > 0 else 0

    if opportunities:
        kl_values = [o.trade.violation_amount for o in opportunities]
        avg_kl = statistics.mean(kl_values)
        max_kl = max(kl_values)
        min_kl = min(kl_values)
    else:
        avg_kl = max_kl = min_kl = 0.0

    max_drawdown = calculate_max_drawdown(opportunities)

    best_trade = None
    worst_trade = None
    if opportunities:
        sorted_by_net = sorted(opportunities, key=lambda x: x.net_profit())
        worst_trade = sorted_by_net[0]
        best_trade = sorted_by_net[-1]

    cluster_performance = calculate_cluster_performance(
        opportunities=opportunities,
        cluster_themes=cluster_themes,
        cluster_market_ids=cluster_market_ids,
    )

    logger.info("[REPORT] Report metrics: gross_pnl=%.4f, net_pnl=%.4f, win_rate=%.2f%%, max_dd=%.4f",
                gross_pnl, net_pnl, win_rate * 100, max_drawdown)

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
    """Calculate maximum drawdown from cumulative PnL."""
    if not opportunities:
        return 0.0

    cumulative_pnl = 0.0
    peak_pnl = 0.0
    max_drawdown = 0.0

    for opp in opportunities:
        cumulative_pnl += opp.net_profit()
        if cumulative_pnl > peak_pnl:
            peak_pnl = cumulative_pnl

        drawdown = peak_pnl - cumulative_pnl
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    logger.debug("[REPORT] Max drawdown calculated: %.4f", max_drawdown)
    return max_drawdown


def calculate_cluster_performance(
    opportunities: list[ArbitrageOpportunity],
    cluster_themes: dict[str, str],
    cluster_market_ids: dict[str, list[str]],
) -> list[ClusterPerformance]:
    """Calculate performance metrics for each cluster."""
    logger.debug("[REPORT] Calculating cluster performance")

    by_cluster: dict[str, list[ArbitrageOpportunity]] = {}
    for opp in opportunities:
        if opp.cluster_id not in by_cluster:
            by_cluster[opp.cluster_id] = []
        by_cluster[opp.cluster_id].append(opp)

    performance_list = []
    for cluster_id, cluster_opps in by_cluster.items():
        gross_pnl = sum(o.locked_profit for o in cluster_opps)
        net_pnl = sum(o.net_profit() for o in cluster_opps)
        win_count = sum(1 for o in cluster_opps if o.net_profit() > 0)
        loss_count = sum(1 for o in cluster_opps if o.net_profit() < 0)

        kl_values = [o.trade.violation_amount for o in cluster_opps]
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

    performance_list.sort(key=lambda x: x.net_pnl, reverse=True)

    logger.debug("[REPORT] Calculated performance for %d clusters", len(performance_list))
    return performance_list


def format_report(report: BacktestReport) -> str:
    """Format a backtest report as a human-readable string."""
    logger.debug("[REPORT] Formatting report")

    lines = [
        "=" * 60,
        "BACKTEST REPORT",
        "=" * 60,
        "",
        "Time Range",
        "-" * 40,
        "  Start:    %s" % report.start_date.isoformat(),
        "  End:      %s" % report.end_date.isoformat(),
        "  Duration: %.1f hours" % report.duration_hours,
        "",
        "Market Coverage",
        "-" * 40,
        "  Markets Analyzed: %d" % report.markets_analyzed,
        "  Clusters Found:   %d" % report.clusters_found,
        "",
        "Opportunities",
        "-" * 40,
        "  Total Found:    %d" % report.total_opportunities,
        "  Per Hour:       %.2f" % report.opportunities_per_hour,
        "",
        "PnL Summary (per $1 stake per market)",
        "-" * 40,
        "  Gross PnL:          $%.4f" % report.gross_pnl,
        "  Transaction Costs:  $%.4f" % report.transaction_costs,
        "  Net PnL:            $%.4f" % report.net_pnl,
        "",
        "Win/Loss Analysis",
        "-" * 40,
        "  Wins:       %d" % report.win_count,
        "  Losses:     %d" % report.loss_count,
        "  Win Rate:   %.1f%%" % (report.win_rate * 100),
        "",
        "KL Divergence Statistics",
        "-" * 40,
        "  Average:  %.6f" % report.avg_kl_divergence,
        "  Maximum:  %.6f" % report.max_kl_divergence,
        "  Minimum:  %.6f" % report.min_kl_divergence,
        "",
        "Risk Metrics",
        "-" * 40,
        "  Max Drawdown:  $%.4f" % report.max_drawdown,
    ]

    if report.best_trade:
        lines.extend([
            "",
            "Best Trade",
            "-" * 40,
            "  Cluster:    %s" % report.best_trade.cluster_id,
            "  Net Profit: $%.4f" % report.best_trade.net_profit(),
            "  KL Div:     %.6f" % report.best_trade.trade.violation_amount,
            "  Time:       %s" % report.best_trade.timestamp.isoformat(),
        ])

    if report.worst_trade:
        lines.extend([
            "",
            "Worst Trade",
            "-" * 40,
            "  Cluster:    %s" % report.worst_trade.cluster_id,
            "  Net Profit: $%.4f" % report.worst_trade.net_profit(),
            "  KL Div:     %.6f" % report.worst_trade.trade.violation_amount,
            "  Time:       %s" % report.worst_trade.timestamp.isoformat(),
        ])

    if report.cluster_performance:
        lines.extend([
            "",
            "Cluster Performance",
            "-" * 40,
        ])
        for cp in report.cluster_performance[:5]:
            lines.append(
                "  %-20s | Opps: %4d | Net: $%8.4f | Win: %5.1f%%" %
                (cp.cluster_id[:20], cp.num_opportunities, cp.net_pnl, cp.win_rate * 100)
            )

    lines.extend([
        "",
        "Configuration",
        "-" * 40,
        "  KL Threshold:      %s" % report.kl_threshold,
        "  Transaction Cost:  %.1f%%" % (report.transaction_cost_rate * 100),
        "",
        "=" * 60,
    ])

    return "\n".join(lines)


def report_to_dict(report: BacktestReport) -> dict:
    """Convert report to a dictionary for JSON serialization."""
    logger.debug("[REPORT] Converting report to dict")

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
