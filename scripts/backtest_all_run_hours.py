"""Backtest: all 24 HRRR run hours vs current 4-run baseline.

Compares:
  A) RUN_HOURS = [0, 6, 12, 18]  (current baseline)
  B) RUN_HOURS = list(range(24))  (every HRRR run hour)

Same model_fn, same update_hours, same eval window.
Prints Brier, hit rate, and P&L for each so you can compare directly.

Usage:
    cd ~/Projects/alphatemp/alphatemp
    PYTHONPATH=. venv/bin/python scripts/backtest_all_run_hours.py
"""

import os
import sys
import time
from datetime import date, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.backtester import Backtester
from autoresearch.run_experiment import compute_displacement_pnl

# Import model_fn fresh
import importlib
import autoresearch.experiment as _exp_mod
importlib.reload(_exp_mod)
from autoresearch.experiment import model_fn, _coeff_cache

# ── Config ─────────────────────────────────────────────────────────────────

EVAL_DAYS = 90
UPDATE_HOURS_ET = [0, 4, 8, 12, 16, 20]
DB_PATH = "data/alphatemp.duckdb"

CONFIGS = [
    # Baseline already measured: Brier=0.6667, Hit=44.6%, P&L=14248c
    # ("4-run baseline [0,6,12,18]", [0, 6, 12, 18]),
    ("24-run (every hour)",        list(range(24))),
]


def run_backtest(label, run_hours):
    # type: (str, list) -> None
    """Run backtest with given run_hours and print results."""
    _coeff_cache.clear()

    bt = Backtester(db_path=DB_PATH, city="NYC")
    eval_start = date.today() - timedelta(days=EVAL_DAYS)

    print("\n" + "=" * 60)
    print("  %s" % label)
    print("  run_hours = %s" % run_hours)
    print("=" * 60)

    start = time.time()
    try:
        result = bt.run(
            model_fn=model_fn,
            start_date=eval_start,
            run_hours=run_hours,
            update_hours_et=UPDATE_HOURS_ET,
        )
    except Exception as e:
        print("  ERROR: %s" % e)
        return

    elapsed = time.time() - start
    pnl_cents, bets_placed = compute_displacement_pnl(result.run_results)

    print("  Brier:    %.4f" % result.mean_brier)
    print("  Hit rate: %.4f" % result.top1_hit_rate)
    print("  P&L:      %.0fc (%d bets)" % (pnl_cents, bets_placed))
    print("  Evals:    %d" % result.total_evaluations)
    print("  Time:     %.1fs" % elapsed)

    # Per-run-hour breakdown using RunResult.run_hour field
    if result.run_results:
        from collections import defaultdict
        rh_briers = defaultdict(list)  # type: dict
        rh_hits = defaultdict(lambda: [0, 0])  # type: dict  # [hits, total]
        for r in result.run_results:
            rh = r.run_hour
            if r.brier_score is not None:
                rh_briers[rh].append(r.brier_score)
            rh_hits[rh][1] += 1
            if r.hit:
                rh_hits[rh][0] += 1

        if rh_briers:
            print("\n  Per-run-hour breakdown:")
            print("    %4s  %8s  %8s  %5s" % ("RH", "Brier", "HitRate", "n"))
            print("    %4s  %8s  %8s  %5s" % ("----", "--------", "--------", "-----"))
            for rh in sorted(rh_briers.keys()):
                scores = rh_briers[rh]
                avg_b = sum(scores) / len(scores)
                hits, total = rh_hits[rh]
                hit_pct = hits / total if total > 0 else 0.0
                print("    %02dz   %.4f    %.1f%%    %d" % (rh, avg_b, hit_pct * 100, len(scores)))


def main():
    print("AlphaTemp — HRRR Run Hour Comparison Backtest")
    print("Eval window: last %d days" % EVAL_DAYS)
    print("Update hours ET: %s" % UPDATE_HOURS_ET)

    for label, run_hours in CONFIGS:
        run_backtest(label, run_hours)

    print("\n" + "=" * 60)
    print("Done. Compare Brier scores above — lower is better.")
    print("=" * 60)


if __name__ == "__main__":
    main()
