#!/usr/bin/env python3
"""Phase 2B Analysis — compare observation-based models against Phase 2 baseline.

Runs Phase 2B ablation variants across all update hours and reports:
  - Brier by (model, run_hour, update_hour)
  - Improvement curve: how each update_hour improves over no-obs baseline
  - Decision gate: <1% kill, 1-2% discuss, >2% keep

Usage: ./venv/bin/python scripts/phase2b_analysis.py
"""

import sys
from datetime import date

sys.path.insert(0, ".")

from loguru import logger
from services.backtester import (
    Backtester,
    wf_regression_full,
    wf_phase2b_full,
    wf_phase2b_instant,
    wf_phase2b_cumul,
    wf_phase2b_runmax,
    wf_phase2b_slope,
)

DB_PATH = "data/alphatemp.duckdb"
UPDATE_HOURS = list(range(8, 19))  # 08 through 18 ET


def main():
    bt = Backtester(db_path=DB_PATH, city="NYC")

    # --- Phase 2 baseline (no update hours — single eval per run_hour) ---
    print("=" * 90)
    print("Phase 2 Baseline (no observations)")
    print("=" * 90)
    p2_result = bt.run(wf_regression_full)
    bt.print_summary(p2_result)
    p2_brier = p2_result.mean_brier

    # --- Phase 2B variants ---
    models = [
        ("wf_phase2b_full",    wf_phase2b_full,    "All 4 features"),
        ("wf_phase2b_instant", wf_phase2b_instant,  "temp_div only"),
        ("wf_phase2b_cumul",   wf_phase2b_cumul,    "cumul_div only"),
        ("wf_phase2b_runmax",  wf_phase2b_runmax,   "running_max only"),
        ("wf_phase2b_slope",   wf_phase2b_slope,    "slope_div only"),
    ]

    all_results = {}  # name -> BacktestResult

    for model_name, model_fn, desc in models:
        print("\n" + "=" * 90)
        print(f"{model_name} — {desc}")
        print("=" * 90)

        result = bt.run(model_fn, update_hours_et=UPDATE_HOURS)
        all_results[model_name] = result

        # Overall
        pct_change = (result.mean_brier - p2_brier) / p2_brier * 100
        print(f"  Overall Brier: {result.mean_brier:.4f}  "
              f"(vs P2 {p2_brier:.4f}, {pct_change:+.1f}%)")
        print(f"  Evaluations: {result.total_evaluations}, Skipped: {result.skipped}")

        # Per update hour
        by_update = getattr(result, 'by_update_hour', {})
        if by_update:
            print(f"\n  {'Hour':>4s}  {'N':>5s}  {'Brier':>7s}  {'vs P2':>7s}  {'Hit%':>6s}")
            print(f"  {'----':>4s}  {'-----':>5s}  {'-------':>7s}  {'-------':>7s}  {'------':>6s}")
            for uhr in sorted(by_update.keys()):
                stats = by_update[uhr]
                pct = (stats['mean_brier'] - p2_brier) / p2_brier * 100
                print(f"  {uhr:4d}  {stats['count']:5d}  {stats['mean_brier']:7.4f}  "
                      f"{pct:+6.1f}%  {stats['top1_hit_rate']:5.1%}")

    # --- Summary comparison table ---
    print("\n" + "=" * 90)
    print("SUMMARY — Phase 2B vs Phase 2 Baseline")
    print("=" * 90)
    print(f"\n  Phase 2 baseline Brier: {p2_brier:.4f}")
    print(f"\n  {'Model':<22s}  {'Brier':>7s}  {'vs P2':>7s}  {'Gate':>8s}")
    print(f"  {'-'*22}  {'-------':>7s}  {'-------':>7s}  {'--------':>8s}")

    for model_name, _, desc in models:
        r = all_results[model_name]
        pct = (r.mean_brier - p2_brier) / p2_brier * 100
        if pct < -2.0:
            gate = "KEEP"
        elif pct < -1.0:
            gate = "DISCUSS"
        else:
            gate = "KILL"
        print(f"  {model_name:<22s}  {r.mean_brier:7.4f}  {pct:+6.1f}%  {gate:>8s}")

    # --- Improvement curve for best model ---
    print("\n" + "=" * 90)
    print("IMPROVEMENT CURVE — wf_phase2b_full by update hour")
    print("=" * 90)
    best = all_results.get("wf_phase2b_full")
    if best:
        by_update = getattr(best, 'by_update_hour', {})
        by_run = best.by_run_hour
        print(f"\n  {'Hour':>4s}  {'Brier':>7s}  {'Δ vs P2':>9s}  {'Δ%':>7s}")
        print(f"  {'----':>4s}  {'-------':>7s}  {'---------':>9s}  {'-------':>7s}")
        for uhr in sorted(by_update.keys()):
            b = by_update[uhr]['mean_brier']
            delta = b - p2_brier
            pct = delta / p2_brier * 100
            print(f"  {uhr:4d}  {b:7.4f}  {delta:+8.4f}  {pct:+6.1f}%")

    print("\n" + "=" * 90)
    print("Done.")
    print("=" * 90)


if __name__ == "__main__":
    main()
