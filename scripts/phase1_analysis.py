#!/usr/bin/env python3
"""Phase 1 Analysis — Compare walk-forward bias correction models.

Runs all model variants against historical data and produces a comparison table.
Also outputs per-run-hour bias/std snapshots to verify expanding window behavior.

Usage:
    cd ~/Projects/alphatemp/alphatemp
    python scripts/phase1_analysis.py
"""

import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("/scripts", ""))

from datetime import date

import duckdb
from loguru import logger

from services.backtester import (
    Backtester,
    uniform_model,
    bias_corrected_model,
    walk_forward_model,
    walk_forward_t_model,
    _walk_forward_bias_query,
    WALK_FORWARD_MIN_DAYS,
)

DB_PATH = "data/alphatemp.duckdb"


def run_all_models():
    """Run all 4 model variants and collect results."""
    models = [
        ("uniform", uniform_model),
        ("bias_corrected", bias_corrected_model),
        ("walk_forward", walk_forward_model),
        ("walk_forward_t", walk_forward_t_model),
    ]

    results = {}
    for name, fn in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running model: {name}")
        logger.info(f"{'='*60}")
        bt = Backtester(db_path=DB_PATH, city="NYC")
        result = bt.run(fn)
        bt.print_summary(result)
        results[name] = result

    return results


def print_comparison_table(results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("PHASE 1 — MODEL COMPARISON")
    print("=" * 90)

    header = (
        f"{'Model':<22} | {'Brier':>7} | {'Top-1':>6} | {'Evals':>6} | "
        f"{'00z':>7} | {'06z':>7} | {'12z':>7} | {'18z':>7}"
    )
    print(header)
    print("-" * 90)

    for name, result in results.items():
        hour_scores = {}
        for h in [0, 6, 12, 18]:
            if h in result.by_run_hour:
                hour_scores[h] = f"{result.by_run_hour[h]['mean_brier']:.4f}"
            else:
                hour_scores[h] = "  n/a  "

        print(
            f"{name:<22} | {result.mean_brier:7.4f} | "
            f"{result.top1_hit_rate:5.1%} | {result.total_evaluations:6d} | "
            f"{hour_scores[0]:>7} | {hour_scores[6]:>7} | "
            f"{hour_scores[12]:>7} | {hour_scores[18]:>7}"
        )

    print("=" * 90)

    # Improvement table
    if "uniform" in results and len(results) > 1:
        baseline_brier = results["uniform"].mean_brier
        print(f"\nImprovement vs uniform baseline (Brier {baseline_brier:.4f}):")
        for name, result in results.items():
            if name == "uniform":
                continue
            if baseline_brier > 0:
                pct = (1.0 - result.mean_brier / baseline_brier) * 100
                print(f"  {name:<22}: {pct:+.1f}%")


def print_walk_forward_diagnostics():
    """Show per-run-hour bias/std at different points in time."""
    print("\n" + "=" * 90)
    print("WALK-FORWARD DIAGNOSTICS — Per-Run-Hour Bias at Sample Dates")
    print("=" * 90)

    con = duckdb.connect(DB_PATH, read_only=True)

    # Pick sample dates spread across the dataset
    sample_dates = [
        date(2022, 7, 1),   # ~6 months in (if data starts ~2021)
        date(2023, 1, 1),   # ~1 year in
        date(2024, 1, 1),   # ~2 years in
        date(2025, 1, 1),   # ~3 years in
        date(2025, 12, 1),  # Late in dataset
    ]

    header = f"{'Date':<12} | {'Hour':>4} | {'N':>5} | {'Mean Bias':>10} | {'Std':>7} | {'Kurtosis':>10}"
    print(header)
    print("-" * 65)

    for sample_date in sample_dates:
        for hour in [0, 6, 12, 18]:
            stats = _walk_forward_bias_query(con, hour, "KNYC", sample_date)
            if stats is None:
                print(f"{sample_date} | {hour:02d}z  |   <{WALK_FORWARD_MIN_DAYS}  |        n/a |     n/a |        n/a")
            else:
                mean_bias, std_error, n, kurtosis = stats
                kurt_str = f"{kurtosis:+.2f}" if kurtosis is not None else "n/a"
                std_str = f"{std_error:.3f}" if std_error is not None else "n/a"
                print(
                    f"{sample_date} | {hour:02d}z  | {int(n):5d} | "
                    f"{mean_bias:+10.3f} | {std_str:>7} | {kurt_str:>10}"
                )
        print("-" * 65)

    con.close()


def print_evaluation_counts(results):
    """Show how many evaluations each model skipped."""
    print("\n" + "=" * 90)
    print("EVALUATION COUNTS")
    print("=" * 90)
    for name, result in results.items():
        total_possible = result.total_evaluations + result.skipped
        skip_pct = result.skipped / total_possible * 100 if total_possible > 0 else 0
        print(
            f"  {name:<22}: {result.total_evaluations:,} evaluated, "
            f"{result.skipped:,} skipped ({skip_pct:.1f}%)"
        )
        if name.startswith("walk_forward"):
            print(
                f"    (Skips include <{WALK_FORWARD_MIN_DAYS}-day training window)"
            )


if __name__ == "__main__":
    logger.info("Phase 1 Analysis — Walk-Forward Bias Correction")
    logger.info(f"Database: {DB_PATH}")

    # Run diagnostics first (fast, helps verify data)
    print_walk_forward_diagnostics()

    # Run all models
    results = run_all_models()

    # Print comparison
    print_comparison_table(results)
    print_evaluation_counts(results)

    print("\n--- Phase 1 analysis complete ---")
