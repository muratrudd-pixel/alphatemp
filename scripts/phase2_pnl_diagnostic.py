"""Phase 2 P&L diagnostic — tracks trajectory, not a kill signal.

Runs the strategy backtester against a Phase 2 bias correction candidate
and reports fee-adjusted P&L. At Phase 2, P&L is diagnostic only because
static std still leaks into tail brackets (fixed in Phase 3).

Usage:
    python scripts/phase2_pnl_diagnostic.py
    python scripts/phase2_pnl_diagnostic.py --model emos
    python scripts/phase2_pnl_diagnostic.py --model xgboost_qr --start 2024-01-01
"""

import argparse
from datetime import date
from typing import Dict

from loguru import logger

from core.db import DEFAULT_DB_PATH
from services.strategy_backtester import BacktestConfig, StrategyBacktester


# Registry of Phase 2 candidate model functions
MODEL_REGISTRY = {
    "ols": ("services.backtester", "wf_regression_full"),
    "ols_no_spinup": ("services.backtester", "wf_regression_full_no_spinup"),
    "emos": ("services.phase2_emos", "emos_model_fn"),
    "xgboost_qr": ("services.phase2_xgboost", "xgboost_model_fn"),
    "xgboost_ext": ("services.phase2_xgboost", "xgboost_extended_model_fn"),
    "cross_hour": ("services.backtester", "wf_regression_cross_hour"),
}


def _load_model_fn(name):
    """Import and return a model function by registry name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            "Unknown model '{}'. Available: {}".format(name, list(MODEL_REGISTRY.keys()))
        )
    module_path, fn_name = MODEL_REGISTRY[name]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, fn_name)


def run_pnl_diagnostic(model_name="ols", db_path=DEFAULT_DB_PATH,
                        start=None, end=None):
    # type: (str, str, date, date) -> Dict
    """Run strategy backtest with a Phase 2 candidate."""
    start = start or date(2023, 6, 1)
    end = end or date(2026, 2, 1)

    model_fn = _load_model_fn(model_name)

    config = BacktestConfig(
        starting_capital=100.0,
        burn_in_days=90,
        fixed_bet_size=1,
        # min_displacement is now ignored — displacement check is dynamic
        # fee-adjusted in the trading loop: model_prob > ask + fee_hurdle
        start_date=start,
        end_date=end,
    )

    logger.info("Running P&L diagnostic for '{}' ({} to {})...",
                model_name, start, end)

    bt = StrategyBacktester(db_path=db_path, config=config)
    report = bt.run(model_fn)

    _print_report(model_name, report, config.starting_capital)
    return report


def _print_report(model_name, report, starting_capital):
    # type: (str, Dict, float) -> None
    """Print formatted P&L diagnostic."""
    pnl_dollars = report["total_pnl_dollars"]
    final = starting_capital + pnl_dollars
    return_pct = report["total_return_pct"]

    print("\n" + "=" * 55)
    print("PHASE 2 P&L DIAGNOSTIC: {}".format(model_name))
    print("=" * 55)
    print("Total P&L:      ${:.2f}".format(pnl_dollars))
    print("Final Capital:  ${:.2f}".format(final))
    print("Return:         {:.1f}%".format(return_pct))
    print("Win Rate:       {:.1%}".format(report["win_rate"]))
    print("Trades:         {}".format(report["total_trades"]))
    print("Profit Factor:  {:.2f}".format(report["profit_factor"]))
    print("Max Drawdown:   {:.1%}".format(report["max_drawdown"]))
    print("Sharpe:         {:.2f}".format(report["sharpe_ratio"]))

    # Edge by hour of day
    edge_by_hour = report.get("edge_by_hour", {})
    if edge_by_hour:
        print("\nEdge by ET hour:")
        for hour in sorted(edge_by_hour.keys()):
            e = edge_by_hour[hour]
            print("  {:02d} ET: {:>4d} triggers, avg disp {:.3f}, PnL ${:.2f}".format(
                hour, e.get("count", 0),
                e.get("avg_displacement", 0),
                e.get("total_pnl", 0) / 100.0,
            ))

    # Bootstrap CI
    bootstrap = report.get("bootstrap", {})
    if bootstrap:
        print("\nBootstrap ({} iterations):".format(
            bootstrap.get("iterations", "?")))
        print("  Mean P&L:  ${:.2f}".format(bootstrap.get("mean_pnl", 0) / 100.0))
        print("  95% CI:    [${:.2f}, ${:.2f}]".format(
            bootstrap.get("ci_lower", 0) / 100.0,
            bootstrap.get("ci_upper", 0) / 100.0,
        ))

    print("-" * 55)
    print("NOTE: P&L is diagnostic only at Phase 2.")
    print("Static std persists until Phase 3 (dynamic uncertainty).")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 P&L diagnostic")
    parser.add_argument("--model", default="ols",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Which Phase 2 candidate to evaluate")
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s),
                        default=None)
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s),
                        default=None)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    run_pnl_diagnostic(args.model, args.db, args.start, args.end)


if __name__ == "__main__":
    main()
