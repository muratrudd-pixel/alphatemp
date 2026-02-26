#!/usr/bin/env python3
"""Phase 2 Analysis — Compare regression models vs Phase 1 baseline.

Runs all Phase 2 variants (feature-conditioned walk-forward regression) plus
the Phase 1 walk-forward baseline. Outputs comparison table with Brier scores
overall and per run hour, plus regression coefficient diagnostics.

Usage:
    cd ~/Projects/alphatemp/alphatemp
    python scripts/phase2_analysis.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("/scripts", ""))

from datetime import date

import duckdb
import numpy as np
from loguru import logger
from scipy.linalg import lstsq

from services.backtester import (
    Backtester,
    walk_forward_model,
    wf_regression_full,
    wf_regression_fcst,
    wf_regression_month,
    wf_regression_delta,
    _walk_forward_regression_data,
    _encode_month,
    _FEATURE_SETS,
    WALK_FORWARD_MIN_DAYS,
)

DB_PATH = "data/alphatemp.duckdb"


def run_all_models():
    """Run Phase 1 baseline and all Phase 2 regression variants."""
    models = [
        ("walk_forward (P1)", walk_forward_model),
        ("wf_regression_full", wf_regression_full),
        ("wf_regression_fcst", wf_regression_fcst),
        ("wf_regression_month", wf_regression_month),
        ("wf_regression_delta", wf_regression_delta),
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
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("PHASE 2 — MODEL COMPARISON (vs Phase 1 Baseline)")
    print("=" * 100)

    header = (
        f"{'Model':<24} | {'Brier':>7} | {'vs P1':>7} | {'Evals':>6} | "
        f"{'00z':>7} | {'06z':>7} | {'12z':>7} | {'18z':>7}"
    )
    print(header)
    print("-" * 100)

    baseline_brier = None
    for name, result in results.items():
        if baseline_brier is None:
            baseline_brier = result.mean_brier

        hour_scores = {}
        for h in [0, 6, 12, 18]:
            if h in result.by_run_hour:
                hour_scores[h] = f"{result.by_run_hour[h]['mean_brier']:.4f}"
            else:
                hour_scores[h] = "  n/a  "

        if name == "walk_forward (P1)":
            vs_p1 = "  base "
        elif baseline_brier and baseline_brier > 0:
            pct = (1.0 - result.mean_brier / baseline_brier) * 100
            vs_p1 = f"{pct:+.1f}%"
        else:
            vs_p1 = "  n/a  "

        print(
            f"{name:<24} | {result.mean_brier:7.4f} | {vs_p1:>7} | "
            f"{result.total_evaluations:6d} | "
            f"{hour_scores[0]:>7} | {hour_scores[6]:>7} | "
            f"{hour_scores[12]:>7} | {hour_scores[18]:>7}"
        )

    print("=" * 100)


def print_regression_diagnostics():
    """Show regression coefficients at sample dates to verify expanding window."""
    print("\n" + "=" * 100)
    print("REGRESSION DIAGNOSTICS — Coefficients at Sample Dates (Full Model)")
    print("=" * 100)

    con = duckdb.connect(DB_PATH, read_only=True)
    feature_indices = _FEATURE_SETS["wf_regression_full"]
    feature_names = ["intercept", "fcst_high", "sin(month)", "cos(month)", "delta_temp"]

    sample_dates = [
        date(2022, 7, 1),
        date(2023, 1, 1),
        date(2024, 1, 1),
        date(2025, 1, 1),
        date(2025, 12, 1),
    ]

    header = f"{'Date':<12} | {'Hour':>4} | {'N':>5} | "
    header += " | ".join(f"{name:>12}" for name in feature_names)
    header += f" | {'Resid Std':>9}"
    print(header)
    print("-" * (len(header) + 5))

    for sample_date in sample_dates:
        for hour in [0, 12]:  # Just show 00z and 12z to keep output manageable
            rows = _walk_forward_regression_data(con, hour, "KNYC", sample_date)
            if rows is None:
                print(f"{sample_date} | {hour:02d}z  | <{WALK_FORWARD_MIN_DAYS:3d}  | {'n/a':>12}" * 6)
                continue

            n = len(rows)
            # Fit the full model to get coefficients
            n_features = len(feature_indices)
            A = np.empty((n, 1 + n_features), dtype=np.float64)
            y = np.empty(n, dtype=np.float64)

            for i, (error, fcst_high, month, delta_temp) in enumerate(rows):
                from services.backtester import _encode_month
                sin_m, cos_m = _encode_month(month)
                all_features = [fcst_high, sin_m, cos_m, delta_temp]
                A[i, 0] = 1.0
                for j, idx in enumerate(feature_indices):
                    A[i, 1 + j] = all_features[idx]
                y[i] = error

            result = lstsq(A, y)
            coeffs = result[0]
            residuals = y - A @ coeffs
            resid_std = float(np.std(residuals, ddof=1 + n_features))

            coeff_strs = [f"{c:+12.4f}" for c in coeffs]
            print(
                f"{sample_date} | {hour:02d}z  | {n:5d} | "
                + " | ".join(coeff_strs)
                + f" | {resid_std:9.4f}"
            )
        print("-" * (len(header) + 5))

    con.close()


def print_decision_gate(results):
    """Print the go/no-go decision based on improvement thresholds."""
    print("\n" + "=" * 100)
    print("DECISION GATE")
    print("=" * 100)

    baseline_name = "walk_forward (P1)"
    if baseline_name not in results:
        print("ERROR: Phase 1 baseline not found in results.")
        return

    baseline_brier = results[baseline_name].mean_brier

    best_name = None
    best_improvement = 0.0

    for name, result in results.items():
        if name == baseline_name:
            continue
        pct = (1.0 - result.mean_brier / baseline_brier) * 100 if baseline_brier > 0 else 0
        print(f"  {name:<24}: Brier={result.mean_brier:.4f}, improvement={pct:+.1f}%")
        if pct > best_improvement:
            best_improvement = pct
            best_name = name

    print()
    if best_improvement > 2.0:
        print(f"VERDICT: CLEAR WIN. Best model: {best_name} ({best_improvement:+.1f}%)")
        print("Proceed with this model.")
    elif best_improvement > 1.0:
        print(f"VERDICT: MARGINAL. Best model: {best_name} ({best_improvement:+.1f}%)")
        print("Worth discussing — complexity vs improvement tradeoff.")
    else:
        print(f"VERDICT: NOT WORTH IT. Best improvement: {best_improvement:+.1f}%")
        print("Phase 2 regression adds complexity without sufficient improvement.")
        print("Consider skipping to Phase 2B (intra-day observation updates).")


if __name__ == "__main__":
    logger.info("Phase 2 Analysis — Feature-Conditioned Walk-Forward Regression")
    logger.info(f"Database: {DB_PATH}")

    # Run diagnostics first
    print_regression_diagnostics()

    # Run all models
    results = run_all_models()

    # Comparison table
    print_comparison_table(results)

    # Decision gate
    print_decision_gate(results)

    print("\n--- Phase 2 analysis complete ---")
