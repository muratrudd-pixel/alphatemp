"""Sweep min_model_prob to find optimal threshold.

Runs the strategy backtester across multiple min_model_prob values
and prints a comparison table.

Usage:
    python scripts/sweep_min_model_prob.py
    python scripts/sweep_min_model_prob.py --model multimodel --db data/alphatemp.duckdb
"""

import argparse
import sys
from datetime import date

from loguru import logger

from services.strategy_backtester import (
    BacktestConfig,
    StrategyBacktester,
)
from services.backtester import (
    uniform_model,
    walk_forward_model,
    wf_regression_full,
    wf_regression_full_no_spinup,
    wf_regression_cross_hour,
    wf_multimodel_full,
)


SWEEP_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def parse_args():
    p = argparse.ArgumentParser(description="Sweep min_model_prob threshold")
    p.add_argument("--db", default="data/alphatemp.duckdb", help="DuckDB path")
    p.add_argument("--model", type=str, default="multimodel",
                    help="Model: uniform, walkforward, ols, ols_no_spinup, cross_hour, multimodel")
    p.add_argument("--capital", type=float, default=100.0, help="Starting capital ($)")
    p.add_argument("--start", type=str, default="2024-11-01", help="Start date")
    p.add_argument("--end", type=str, default=None, help="End date")
    return p.parse_args()


def main():
    args = parse_args()

    models = {
        "uniform": uniform_model,
        "walkforward": walk_forward_model,
        "ols": wf_regression_full,
        "ols_no_spinup": wf_regression_full_no_spinup,
        "cross_hour": wf_regression_cross_hour,
        "multimodel": wf_multimodel_full,
    }
    model_fn = models.get(args.model, wf_multimodel_full)

    results = []

    for prob_threshold in SWEEP_VALUES:
        logger.info("Running backtest with min_model_prob={:.2f}", prob_threshold)

        config = BacktestConfig(
            starting_capital=args.capital,
            start_date=date.fromisoformat(args.start),
            end_date=date.fromisoformat(args.end) if args.end else None,
            min_model_prob=prob_threshold,
        )

        bt = StrategyBacktester(db_path=args.db, config=config)
        report = bt.run(model_fn=model_fn)

        results.append({
            "min_prob": prob_threshold,
            "trades": report.get("total_trades", 0),
            "win_rate": report.get("win_rate", 0),
            "pnl_cents": report.get("total_pnl", 0),
            "pnl_dollars": report.get("total_pnl_dollars", 0),
            "sharpe": report.get("sharpe_ratio", 0),
            "max_dd": report.get("max_drawdown", 0),
            "profit_factor": report.get("profit_factor", 0),
            "final_bankroll": report.get("final_bankroll", 0),
        })

    # Print comparison table
    print("\n" + "=" * 90)
    print("  min_model_prob SWEEP RESULTS  (model: {})".format(args.model))
    print("=" * 90)
    print(
        "  {:>8}  {:>6}  {:>8}  {:>10}  {:>8}  {:>7}  {:>7}  {:>10}".format(
            "min_prob", "trades", "win_rate", "pnl ($)", "sharpe",
            "max_dd", "pf", "bankroll",
        )
    )
    print("  " + "-" * 76)

    for r in results:
        print(
            "  {:>8.2f}  {:>6}  {:>7.1%}  {:>10.2f}  {:>8.2f}  {:>6.1f}c  {:>7.2f}  {:>9.2f}".format(
                r["min_prob"],
                r["trades"],
                r["win_rate"],
                r["pnl_dollars"],
                r["sharpe"],
                r["max_dd"],
                r["profit_factor"],
                r["final_bankroll"],
            )
        )

    print("=" * 90)


if __name__ == "__main__":
    main()
