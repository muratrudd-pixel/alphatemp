"""Run the Kalshi strategy backtester and print results.

Usage:
    python scripts/run_strategy_backtest.py
    python scripts/run_strategy_backtest.py --capital 50 --min-displacement 0.15
    python scripts/run_strategy_backtest.py --start 2025-01-01 --end 2025-06-01
"""

import argparse
import sys
from datetime import date, datetime

from loguru import logger

from services.strategy_backtester import (
    BacktestConfig,
    StrategyBacktester,
)
from services.backtester import (
    uniform_model, walk_forward_model,
    wf_regression_full, wf_regression_full_no_spinup,
    wf_regression_cross_hour,
)


def parse_args():
    p = argparse.ArgumentParser(description="Kalshi Strategy Backtester")
    p.add_argument("--db", default="data/alphatemp.duckdb", help="DuckDB path")
    p.add_argument("--capital", type=float, default=100.0, help="Starting capital ($)")
    p.add_argument("--start", type=str, default="2024-11-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    p.add_argument("--burn-in", type=int, default=90, help="Burn-in days")
    p.add_argument("--min-displacement", type=float, default=0.12, help="Minimum displacement")
    p.add_argument("--min-prob", type=float, default=0.05, help="Minimum model probability")
    p.add_argument("--max-spread", type=float, default=10.0, help="Maximum spread (cents)")
    p.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap iterations")
    p.add_argument("--model", type=str, default="uniform",
                    help="Model: uniform, walkforward, ols, ols_no_spinup, emos, xgboost, cross_hour")
    return p.parse_args()


def print_report(report):
    """Pretty-print the backtest report."""
    print("\n" + "=" * 60)
    print("  KALSHI STRATEGY BACKTEST REPORT")
    print("=" * 60)

    print("\n-- Aggregate Metrics --")
    print("  Total trades:      {}".format(report.get("total_trades", 0)))
    print("  Win rate:          {:.1%}".format(report.get("win_rate", 0)))
    print("  Total P&L:         {:.1f}c (${:.2f})".format(
        report.get("total_pnl", 0), report.get("total_pnl_dollars", 0)))
    print("  Return:            {:.1f}%".format(report.get("total_return_pct", 0)))
    print("  Sharpe ratio:      {:.2f}".format(report.get("sharpe_ratio", 0)))
    print("  Max drawdown:      {:.1f}c".format(report.get("max_drawdown", 0)))
    print("  Profit factor:     {:.2f}".format(report.get("profit_factor", 0)))
    print("  Avg win:           {:.1f}c".format(report.get("avg_win", 0)))
    print("  Avg loss:          {:.1f}c".format(report.get("avg_loss", 0)))
    print("  Total fees:        {:.1f}c".format(report.get("total_fees", 0)))
    print("  Final bankroll:    ${:.2f}".format(report.get("final_bankroll", 0)))

    bs = report.get("bootstrap", {})
    if bs:
        print("\n-- Bootstrap Confidence (95%) --")
        ci = bs.get("pnl_ci_95", (0, 0))
        print("  P&L CI:            [{:.1f}c, {:.1f}c]".format(ci[0], ci[1]))
        print("  Prob profitable:   {:.1%}".format(bs.get("prob_profitable", 0)))

    edge = report.get("edge_by_hour", {})
    if edge:
        print("\n-- Edge by ET Hour --")
        print("  {:>4}  {:>10}  {:>10}  {:>8}  {:>5}".format(
            "Hour", "Model Br.", "Mkt Br.", "Edge", "N"))
        for h in sorted(edge.keys()):
            e = edge[h]
            print("  {:>4}  {:>10.4f}  {:>10.4f}  {:>+8.4f}  {:>5}".format(
                h, e["model_brier"], e["market_brier"], e["edge"], e["count"]))

    missed = report.get("missed_due_to_capital", 0)
    if missed > 0:
        print("\n-- Capital Constraints --")
        print("  Missed signals (capital):  {}".format(missed))

    print("\n" + "=" * 60)


def main():
    args = parse_args()

    config = BacktestConfig(
        starting_capital=args.capital,
        start_date=date.fromisoformat(args.start),
        end_date=date.fromisoformat(args.end) if args.end else None,
        burn_in_days=args.burn_in,
        min_displacement=args.min_displacement,
        min_model_prob=args.min_prob,
        max_spread_cents=args.max_spread,
        bootstrap_iterations=args.bootstrap,
    )

    models = {
        "uniform": uniform_model,
        "walkforward": walk_forward_model,
        "ols": wf_regression_full,
        "ols_no_spinup": wf_regression_full_no_spinup,
        "cross_hour": wf_regression_cross_hour,
    }
    # Lazy imports for optional candidates
    if args.model == "emos":
        from services.phase2_emos import emos_model_fn
        models["emos"] = emos_model_fn
    elif args.model == "xgboost":
        from services.phase2_xgboost import xgboost_model_fn
        models["xgboost"] = xgboost_model_fn
    model_fn = models.get(args.model, uniform_model)

    logger.info("Running strategy backtest: {} -> {}, capital=${}, model={}",
                config.start_date, config.end_date or "latest",
                config.starting_capital, args.model)

    bt = StrategyBacktester(db_path=args.db, config=config)
    report = bt.run(model_fn=model_fn)

    print_report(report)


if __name__ == "__main__":
    main()
