"""Diagnostic backtest — day-by-day model accuracy report.

Usage:
    PYTHONPATH=. python scripts/diagnostic_backtest.py --start 2026-03-12 --end 2026-03-17
"""

import argparse
import math
from datetime import date, timedelta

import duckdb
import numpy as np
from loguru import logger

from services.feature_builder import FeatureBuilder
from services.model import QRModel


def parse_args():
    p = argparse.ArgumentParser(description="Day-by-day model diagnostic")
    p.add_argument("--db", default="data/alphatemp.duckdb")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    return p.parse_args()


def run_diagnostic(db_path, start_date, end_date):
    # type: (str, date, date) -> None
    fb = FeatureBuilder(db_path)
    model = QRModel()

    con = duckdb.connect(db_path, read_only=True)

    # Get actual highs from nws_daily
    actuals = {}
    rows = con.execute("""
        SELECT obs_date, max_temp_f FROM nws_daily
        WHERE station_id = 'KNYC'
          AND obs_date >= ? AND obs_date <= ?
        ORDER BY obs_date
    """, [start_date, end_date]).fetchall()
    for r in rows:
        actuals[r[0]] = r[1]

    # Get paper trading summary per day
    trade_summary = {}
    try:
        trade_rows = con.execute("""
            SELECT event_date::DATE as d,
                   COUNT(*) as trades,
                   SUM(CASE WHEN pnl_cents > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(COALESCE(pnl_cents, 0)) as total_pnl
            FROM paper_positions
            WHERE event_date::DATE >= ? AND event_date::DATE <= ?
              AND status = 'settled'
            GROUP BY 1
        """, [start_date, end_date]).fetchall()
        for r in trade_rows:
            trade_summary[r[0]] = {"trades": r[1], "wins": r[2], "pnl_cents": r[3]}
    except Exception:
        pass  # table may not exist or be empty

    con.close()

    print("\n" + "=" * 80)
    print("  DIAGNOSTIC BACKTEST: {} to {}".format(start_date, end_date))
    print("=" * 80)
    print("{:>10} {:>8} {:>8} {:>6} {:>8} {:>8} {:>8} {:>10}".format(
        "Date", "Predict", "Actual", "Error", "Trades", "Wins", "PnL($)", "RunHour"))
    print("-" * 80)

    current = start_date
    total_abs_error = 0.0
    days_counted = 0

    while current <= end_date:
        actual = actuals.get(current)
        if actual is None:
            current += timedelta(days=1)
            continue

        # Run model at update_hour=12 (midday — representative)
        train_result = fb.get_training_data(current, 12)
        if train_result is None:
            print("{:>10} {:>8} {:>8} {:>6} — insufficient training data".format(
                current.isoformat(), "N/A", actual, "N/A"))
            current += timedelta(days=1)
            continue

        X_train, y_train, _, run_hour = train_result
        date_key = current.isoformat()
        coefficients = model.fit(X_train, y_train, run_hour=run_hour, date_key=date_key)
        if coefficients is None:
            current += timedelta(days=1)
            continue

        feat_result = fb.build_features(current, 12)
        if feat_result is None:
            current += timedelta(days=1)
            continue

        features, fcst_high, _ = feat_result
        bracket_probs = model.predict_bracket_probs(
            features, fcst_high, run_hour=run_hour, date_key=date_key
        )

        # Predicted high = weighted sum of bracket centers
        if bracket_probs:
            predicted = sum(k * p for k, p in bracket_probs.items())
        else:
            predicted = fcst_high

        error = predicted - actual
        total_abs_error += abs(error)
        days_counted += 1

        ts = trade_summary.get(current, {"trades": 0, "wins": 0, "pnl_cents": 0})
        pnl_dollars = ts["pnl_cents"] / 100.0

        print("{:>10} {:>8.1f} {:>8.1f} {:>+6.1f} {:>8} {:>8} {:>8.2f} {:>10}z".format(
            current.isoformat(), predicted, actual, error,
            ts["trades"], ts["wins"], pnl_dollars, run_hour))

        current += timedelta(days=1)

    if days_counted > 0:
        mae = total_abs_error / days_counted
        print("-" * 80)
        print("  MAE: {:.2f}F over {} days".format(mae, days_counted))
    print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    run_diagnostic(args.db, start, end)
