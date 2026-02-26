"""Run backtest using market categories instead of hardcoded IDs."""
import sys
sys.path.insert(0, "/root/combarbbot")

from src.backtest import run_backtest, print_report

print("=" * 70)
print("CATEGORY BACKTEST: Politics")
print("=" * 70)

report = run_backtest(
    category="politics",
    kl_threshold=0.001,
    max_ticks=10000,
)
print_report(report)
