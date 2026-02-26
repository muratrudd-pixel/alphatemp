#!/usr/bin/env python3
"""Phase 3 Analysis — Multi-model comparison and ensemble evaluation.

Runs:
  Part 1: Per-model Phase 1 (walk-forward bias) — GFS and ECMWF vs HRRR baseline
  Part 2: Per-model Phase 2 (regression ablation) per model
  Part 3: Error correlation between models at run_hour=0
  Part 4: Equal-weight ensemble evaluation at all run hours

Usage:
    cd ~/Projects/alphatemp/alphatemp
    PYTHONPATH=. python scripts/phase3_analysis.py
    PYTHONPATH=. python scripts/phase3_analysis.py --part 1   # run only Part 1
    PYTHONPATH=. python scripts/phase3_analysis.py --part 3   # run only Part 3
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("/scripts", ""))

import argparse
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
from loguru import logger

from services.backtester import (
    Backtester,
    BacktestDataProvider,
    RUN_HOURS,
    _strip_tz,
    _walk_forward_raw,
    wf_bias_hrrr,
    wf_bias_gfs,
    wf_bias_ecmwf,
    wf_regression_full,
    wf_regression_fcst,
    wf_regression_month,
    wf_regression_delta,
    wf_regression_full_gfs,
    wf_regression_fcst_gfs,
    wf_regression_month_gfs,
    wf_regression_delta_gfs,
    wf_regression_full_ecmwf,
    wf_regression_fcst_ecmwf,
    wf_regression_month_ecmwf,
    wf_regression_delta_ecmwf,
    _compute_bracket_probs_from_dist,
    compute_brier_score_1f,
    HRRR_AVAILABILITY_LAG_HOURS,
)
from services.ensemble import combine_mixture_brackets

DB_PATH = "data/alphatemp.duckdb"


# ===================================================================
# Part 1: Per-model Phase 1 (walk-forward bias)
# ===================================================================

def part1_phase1_per_model():
    """Run Phase 1 walk-forward bias model for each weather model."""
    print("\n" + "=" * 100)
    print("PART 1: Per-Model Phase 1 (Walk-Forward Bias)")
    print("=" * 100)

    models = {
        "hrrr": {"fn": wf_bias_hrrr, "run_hours": [0, 6, 12, 18], "model_name": "hrrr"},
        "gfs":  {"fn": wf_bias_gfs,  "run_hours": [0],            "model_name": "gfs"},
        "ecmwf": {"fn": wf_bias_ecmwf, "run_hours": [0],          "model_name": "ecmwf"},
    }

    results = {}
    for label, cfg in models.items():
        logger.info(f"\nRunning Phase 1 for {label.upper()} (run_hours={cfg['run_hours']})...")
        bt = Backtester(db_path=DB_PATH, city="NYC")
        result = bt.run(
            cfg["fn"],
            run_hours=cfg["run_hours"],
            model_name=cfg["model_name"],
        )
        results[label] = result

    # Print comparison table
    print("\n" + "-" * 80)
    print(f"{'Model':<12} | {'Brier':>7} | {'Evals':>6} | {'Skipped':>7} | {'00z':>7}")
    print("-" * 80)
    for label, result in results.items():
        hour_0 = result.by_run_hour.get(0, {}).get("mean_brier", float("nan"))
        print(
            f"{label.upper():<12} | {result.mean_brier:7.4f} | "
            f"{result.total_evaluations:6d} | {result.skipped:7d} | "
            f"{hour_0:7.4f}"
        )
    print("-" * 80)

    return results


# ===================================================================
# Part 2: Per-model Phase 2 (regression ablation)
# ===================================================================

def part2_regression_ablation():
    """Run regression ablation for each model and compare vs Phase 1."""
    print("\n" + "=" * 100)
    print("PART 2: Per-Model Phase 2 Regression Ablation")
    print("=" * 100)

    model_configs = {
        "hrrr": {
            "run_hours": [0, 6, 12, 18],
            "model_name": "hrrr",
            "phase1_fn": wf_bias_hrrr,
            "ablation": {
                "full":  wf_regression_full,
                "fcst":  wf_regression_fcst,
                "month": wf_regression_month,
                "delta": wf_regression_delta,
            },
        },
        "gfs": {
            "run_hours": [0],
            "model_name": "gfs",
            "phase1_fn": wf_bias_gfs,
            "ablation": {
                "full":  wf_regression_full_gfs,
                "fcst":  wf_regression_fcst_gfs,
                "month": wf_regression_month_gfs,
                "delta": wf_regression_delta_gfs,
            },
        },
        "ecmwf": {
            "run_hours": [0],
            "model_name": "ecmwf",
            "phase1_fn": wf_bias_ecmwf,
            "ablation": {
                "full":  wf_regression_full_ecmwf,
                "fcst":  wf_regression_fcst_ecmwf,
                "month": wf_regression_month_ecmwf,
                "delta": wf_regression_delta_ecmwf,
            },
        },
    }

    all_results = {}  # {model: {variant: BacktestResult}}
    champions = {}    # {model: (variant_name, BacktestResult)}

    for model_label, cfg in model_configs.items():
        print(f"\n{'='*60}")
        print(f"  {model_label.upper()} Regression Ablation")
        print(f"{'='*60}")

        model_results = {}
        bt = Backtester(db_path=DB_PATH, city="NYC")

        # Phase 1 baseline
        logger.info(f"Running Phase 1 baseline for {model_label.upper()}...")
        p1_result = bt.run(
            cfg["phase1_fn"],
            run_hours=cfg["run_hours"],
            model_name=cfg["model_name"],
        )
        model_results["phase1"] = p1_result

        # Regression variants
        for variant_name, fn in cfg["ablation"].items():
            logger.info(f"Running {model_label.upper()} regression_{variant_name}...")
            result = bt.run(
                fn,
                run_hours=cfg["run_hours"],
                model_name=cfg["model_name"],
            )
            model_results[variant_name] = result

        all_results[model_label] = model_results

        # Print table for this model
        p1_brier = p1_result.mean_brier
        print(f"\n{model_label.upper()} Results:")
        print(f"{'Variant':<20} | {'Brier':>7} | {'vs P1':>7} | {'Evals':>6} | {'00z':>7}")
        print("-" * 70)

        best_name = "phase1"
        best_brier = p1_brier

        for variant_name, result in model_results.items():
            hour_0 = result.by_run_hour.get(0, {}).get("mean_brier", float("nan"))
            if variant_name == "phase1":
                vs = "  base "
            else:
                pct = (1.0 - result.mean_brier / p1_brier) * 100 if p1_brier > 0 else 0
                vs = f"{pct:+.1f}%"

            print(
                f"{variant_name:<20} | {result.mean_brier:7.4f} | "
                f"{vs:>7} | {result.total_evaluations:6d} | {hour_0:7.4f}"
            )

            if result.mean_brier < best_brier:
                best_brier = result.mean_brier
                best_name = variant_name

        print("-" * 70)
        print(f"  Champion: {best_name} (Brier {best_brier:.4f})")

        gate = best_brier < p1_brier
        print(f"  GATE: {'PASS' if gate else 'FAIL'} — regression {'beats' if gate else 'does NOT beat'} Phase 1")

        champions[model_label] = (best_name, model_results[best_name])

    return all_results, champions


# ===================================================================
# Part 3: Error correlation between models
# ===================================================================

def part3_error_correlation():
    """Compute error correlation between HRRR, GFS, and ECMWF at run_hour=0."""
    print("\n" + "=" * 100)
    print("PART 3: Error Correlation Between Models (run_hour=0)")
    print("=" * 100)

    con = duckdb.connect(DB_PATH, read_only=True)

    # Get all settlement dates
    dates = con.execute("""
        SELECT obs_date, max_temp_f
        FROM nws_daily
        WHERE station_id = 'KNYC'
        AND max_temp_f IS NOT NULL
        ORDER BY obs_date
    """).fetchall()

    models_cfg = [
        ("hrrr", "hrrr"),
        ("gfs", "gfs"),
        ("ecmwf", "ecmwf"),
    ]

    # Collect per-date residuals from Phase 1 .raw
    residuals = {m: {} for m, _ in models_cfg}  # {model: {date: residual}}

    for obs_date, actual_high in dates:
        model_run_utc = datetime(
            obs_date.year, obs_date.month, obs_date.day,
            0, 0, tzinfo=timezone.utc,
        )
        ref_time = model_run_utc + timedelta(hours=HRRR_AVAILABILITY_LAG_HOURS)

        for model_label, model_name in models_cfg:
            provider = BacktestDataProvider(
                db_path=DB_PATH,
                station_id="KNYC",
                model_run=model_run_utc,
                ref_time=ref_time,
                connection=con,
                model_name=model_name,
            )
            raw = _walk_forward_raw(provider, ref_time, model_name=model_name)
            if raw is not None:
                center, std = raw
                residual = center - actual_high
                residuals[model_label][obs_date] = residual

    con.close()

    # Find overlapping dates
    all_models = [m for m, _ in models_cfg]
    overlap_dates = sorted(
        set.intersection(*(set(residuals[m].keys()) for m in all_models))
    )
    n_overlap = len(overlap_dates)
    print(f"\nOverlapping dates with all 3 models: {n_overlap}")
    if n_overlap > 0:
        print(f"  Range: {overlap_dates[0]} to {overlap_dates[-1]}")

    if n_overlap < 30:
        print("ERROR: Not enough overlapping dates for correlation analysis.")
        return None

    # Build residual arrays
    arrays = {}
    for m in all_models:
        arrays[m] = np.array([residuals[m][d] for d in overlap_dates])

    # Print summary stats
    print(f"\n{'Model':<8} | {'Mean Resid':>10} | {'Std':>7} | {'N':>5}")
    print("-" * 40)
    for m in all_models:
        print(f"{m.upper():<8} | {np.mean(arrays[m]):+10.3f} | {np.std(arrays[m]):7.3f} | {len(arrays[m]):5d}")

    # Correlation matrix
    print("\n--- Error Correlation Matrix ---")
    print(f"{'':>8}", end="")
    for m in all_models:
        print(f" | {m.upper():>8}", end="")
    print()
    print("-" * 40)

    correlations = {}
    for i, m1 in enumerate(all_models):
        print(f"{m1.upper():<8}", end="")
        for j, m2 in enumerate(all_models):
            corr = np.corrcoef(arrays[m1], arrays[m2])[0, 1]
            correlations[(m1, m2)] = corr
            print(f" | {corr:8.4f}", end="")
        print()

    # Gate check
    pairs = [("hrrr", "gfs"), ("hrrr", "ecmwf"), ("gfs", "ecmwf")]
    print("\n--- Gate Check: At least one pair with correlation < 0.7 ---")
    gate_pass = False
    for m1, m2 in pairs:
        corr = correlations[(m1, m2)]
        status = "< 0.7" if corr < 0.7 else ">= 0.7"
        print(f"  {m1.upper()}-{m2.upper()}: {corr:.4f} ({status})")
        if corr < 0.7:
            gate_pass = True

    print(f"\n  GATE: {'PASS' if gate_pass else 'FAIL'} — "
          f"{'at least one pair has low correlation' if gate_pass else 'ALL pairs highly correlated'}")

    return correlations


# ===================================================================
# Part 4: Equal-weight ensemble evaluation
# ===================================================================

def part4_ensemble_evaluation(champions=None):
    """Evaluate equal-weight ensemble at all HRRR run hours.

    For each date and run_hour:
      - HRRR: provider with model_run=date+run_hour, model_name='hrrr'
      - GFS: provider with model_run=date+00z, model_name='gfs'
      - ECMWF: provider with model_run=date+00z, model_name='ecmwf'
    Calls .raw on each champion, combines via mixture, scores.
    """
    print("\n" + "=" * 100)
    print("PART 4: Equal-Weight Ensemble Evaluation")
    print("=" * 100)

    # Determine champion model functions per model
    # Default to regression_full if no champions provided
    if champions is None:
        hrrr_fn = wf_regression_full
        gfs_fn = wf_regression_full_gfs
        ecmwf_fn = wf_regression_full_ecmwf
        print("  Using default champions: regression_full per model")
    else:
        hrrr_champ_name = champions.get("hrrr", ("full", None))[0]
        gfs_champ_name = champions.get("gfs", ("full", None))[0]
        ecmwf_champ_name = champions.get("ecmwf", ("full", None))[0]

        # Map champion names back to model functions
        hrrr_map = {
            "phase1": wf_bias_hrrr, "full": wf_regression_full,
            "fcst": wf_regression_fcst, "month": wf_regression_month,
            "delta": wf_regression_delta,
        }
        gfs_map = {
            "phase1": wf_bias_gfs, "full": wf_regression_full_gfs,
            "fcst": wf_regression_fcst_gfs, "month": wf_regression_month_gfs,
            "delta": wf_regression_delta_gfs,
        }
        ecmwf_map = {
            "phase1": wf_bias_ecmwf, "full": wf_regression_full_ecmwf,
            "fcst": wf_regression_fcst_ecmwf, "month": wf_regression_month_ecmwf,
            "delta": wf_regression_delta_ecmwf,
        }

        hrrr_fn = hrrr_map.get(hrrr_champ_name, wf_regression_full)
        gfs_fn = gfs_map.get(gfs_champ_name, wf_regression_full_gfs)
        ecmwf_fn = ecmwf_map.get(ecmwf_champ_name, wf_regression_full_ecmwf)
        print(f"  Champions: HRRR={hrrr_champ_name}, GFS={gfs_champ_name}, ECMWF={ecmwf_champ_name}")

    con = duckdb.connect(DB_PATH, read_only=True)

    # Get settlement dates
    dates = con.execute("""
        SELECT obs_date, max_temp_f
        FROM nws_daily
        WHERE station_id = 'KNYC'
        AND max_temp_f IS NOT NULL
        ORDER BY obs_date
    """).fetchall()

    run_hours = [0, 6, 12, 18]

    # Track results per run_hour
    ensemble_briers = {h: [] for h in run_hours}  # {hour: [brier_score]}
    hrrr_briers = {h: [] for h in run_hours}
    ensemble_all = []
    hrrr_all = []
    dates_processed = 0

    for i, (obs_date, actual_high) in enumerate(dates):
        for hour in run_hours:
            hrrr_run_utc = datetime(
                obs_date.year, obs_date.month, obs_date.day,
                hour, 0, tzinfo=timezone.utc,
            )
            hrrr_ref_time = hrrr_run_utc + timedelta(hours=HRRR_AVAILABILITY_LAG_HOURS)

            # GFS/ECMWF always use midnight model_run
            global_run_utc = datetime(
                obs_date.year, obs_date.month, obs_date.day,
                0, 0, tzinfo=timezone.utc,
            )
            global_ref_time = global_run_utc + timedelta(hours=HRRR_AVAILABILITY_LAG_HOURS)

            # HRRR provider
            hrrr_provider = BacktestDataProvider(
                db_path=DB_PATH,
                station_id="KNYC",
                model_run=hrrr_run_utc,
                ref_time=hrrr_ref_time,
                connection=con,
                model_name="hrrr",
            )

            # HRRR raw prediction
            hrrr_raw = hrrr_fn.raw(hrrr_provider, hrrr_ref_time)
            if hrrr_raw is None:
                continue

            # Also compute HRRR-only Brier for comparison
            from scipy.stats import norm
            hrrr_brackets = _compute_bracket_probs_from_dist(
                norm(0, 1), hrrr_raw[0], hrrr_raw[1]
            )
            hrrr_brier = compute_brier_score_1f(hrrr_brackets, actual_high)
            hrrr_briers[hour].append(hrrr_brier)
            hrrr_all.append(hrrr_brier)

            # GFS provider
            gfs_provider = BacktestDataProvider(
                db_path=DB_PATH,
                station_id="KNYC",
                model_run=global_run_utc,
                ref_time=global_ref_time,
                connection=con,
                model_name="gfs",
            )

            # ECMWF provider
            ecmwf_provider = BacktestDataProvider(
                db_path=DB_PATH,
                station_id="KNYC",
                model_run=global_run_utc,
                ref_time=global_ref_time,
                connection=con,
                model_name="ecmwf",
            )

            # Collect raw predictions for ensemble
            predictions = [hrrr_raw]
            weights = [1.0]

            gfs_raw = gfs_fn.raw(gfs_provider, global_ref_time)
            if gfs_raw is not None:
                predictions.append(gfs_raw)
                weights.append(1.0)

            ecmwf_raw = ecmwf_fn.raw(ecmwf_provider, global_ref_time)
            if ecmwf_raw is not None:
                predictions.append(ecmwf_raw)
                weights.append(1.0)

            # Normalize weights
            total_w = sum(weights)
            weights = [w / total_w for w in weights]

            # Combine ensemble
            ensemble_brackets = combine_mixture_brackets(predictions, weights)
            if not ensemble_brackets:
                continue

            ensemble_brier = compute_brier_score_1f(ensemble_brackets, actual_high)
            ensemble_briers[hour].append(ensemble_brier)
            ensemble_all.append(ensemble_brier)

        dates_processed += 1
        if (i + 1) % 200 == 0:
            logger.info(f"  Processed {i + 1}/{len(dates)} days...")

    con.close()

    # Print results
    print(f"\n{'Run Hour':<10} | {'Ens Brier':>10} | {'HRRR Brier':>11} | {'Delta':>7} | {'N':>6}")
    print("-" * 60)

    for hour in run_hours:
        if ensemble_briers[hour]:
            ens_mean = np.mean(ensemble_briers[hour])
            hrrr_mean = np.mean(hrrr_briers[hour])
            pct = (1.0 - ens_mean / hrrr_mean) * 100 if hrrr_mean > 0 else 0
            n = len(ensemble_briers[hour])
            print(f"{hour:02d}z       | {ens_mean:10.4f} | {hrrr_mean:11.4f} | {pct:+6.1f}% | {n:6d}")
        else:
            print(f"{hour:02d}z       | {'n/a':>10} | {'n/a':>11} | {'n/a':>7} | {0:6d}")

    print("-" * 60)

    if ensemble_all and hrrr_all:
        ens_overall = np.mean(ensemble_all)
        hrrr_overall = np.mean(hrrr_all)
        pct_overall = (1.0 - ens_overall / hrrr_overall) * 100 if hrrr_overall > 0 else 0

        print(f"{'Overall':<10} | {ens_overall:10.4f} | {hrrr_overall:11.4f} | {pct_overall:+6.1f}% | {len(ensemble_all):6d}")
        print("-" * 60)

        gate = ens_overall < hrrr_overall
        print(f"\n  ENSEMBLE BRIER: {ens_overall:.4f} vs BEST SINGLE MODEL (HRRR): {hrrr_overall:.4f}")
        print(f"  GATE: {'PASS' if gate else 'FAIL'} — ensemble {'beats' if gate else 'does NOT beat'} HRRR alone")
    else:
        print("\n  ERROR: No ensemble evaluations produced.")

    return ensemble_all, hrrr_all


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3 Multi-Model Analysis")
    parser.add_argument("--part", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Run only a specific part (1-4). Default: run all.")
    args = parser.parse_args()

    champions = None

    if args.part is None or args.part == 1:
        part1_phase1_per_model()

    if args.part is None or args.part == 2:
        _, champions = part2_regression_ablation()

    if args.part is None or args.part == 3:
        part3_error_correlation()

    if args.part is None or args.part == 4:
        part4_ensemble_evaluation(champions)

    print("\n" + "=" * 100)
    print("Phase 3 Analysis Complete")
    print("=" * 100)


if __name__ == "__main__":
    main()
