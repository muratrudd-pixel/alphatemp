#!/usr/bin/env python3
"""Phase 3B Analysis — Per-model Phase 2B + Ensemble with Phase 2B + Learned Weights.

Conditional steps (unlocked after Phase 3 gates passed).

  Part 5: Per-model Phase 2B ablation (GFS/ECMWF at run_hour=0, update_hours 14-18 ET)
  Part 6: Ensemble with Phase 2B at update hours 14-18 ET
  Part 7: Learned weights (inverse-Brier weighting)

Usage:
    cd ~/Projects/alphatemp/alphatemp
    PYTHONPATH=. python scripts/phase3b_analysis.py
    PYTHONPATH=. python scripts/phase3b_analysis.py --part 5
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
from scipy.stats import norm

from services.backtester import (
    Backtester,
    BacktestDataProvider,
    RUN_HOURS,
    _strip_tz,
    _compute_bracket_probs_from_dist,
    compute_brier_score_1f,
    HRRR_AVAILABILITY_LAG_HOURS,
    # Phase 2 regression champions (all models use 'full')
    wf_regression_full,
    wf_regression_full_gfs,
    wf_regression_full_ecmwf,
    # Phase 2B variants — HRRR
    wf_phase2b_full,
    wf_phase2b_instant,
    wf_phase2b_cumul,
    wf_phase2b_runmax,
    wf_phase2b_slope,
    # Phase 2B variants — GFS
    wf_phase2b_full_gfs,
    wf_phase2b_instant_gfs,
    wf_phase2b_cumul_gfs,
    wf_phase2b_runmax_gfs,
    wf_phase2b_slope_gfs,
    # Phase 2B variants — ECMWF
    wf_phase2b_full_ecmwf,
    wf_phase2b_instant_ecmwf,
    wf_phase2b_cumul_ecmwf,
    wf_phase2b_runmax_ecmwf,
    wf_phase2b_slope_ecmwf,
)
from services.ensemble import combine_mixture_brackets

DB_PATH = "data/alphatemp.duckdb"

# HRRR Phase 2B analysis showed crossover at 14 ET — don't use obs before then
UPDATE_HOURS_ET = list(range(14, 19))  # 14, 15, 16, 17, 18 ET


# ===================================================================
# Part 5: Per-model Phase 2B ablation
# ===================================================================

def part5_phase2b_per_model():
    """Run Phase 2B ablation for GFS and ECMWF (HRRR results already known)."""
    print("\n" + "=" * 100)
    print("PART 5: Per-Model Phase 2B Ablation (update_hours 14-18 ET)")
    print("=" * 100)

    model_configs = {
        "hrrr": {
            "run_hours": [0, 6, 12, 18],
            "model_name": "hrrr",
            "p2_baseline": wf_regression_full,
            "ablation": [
                ("full",    wf_phase2b_full,    "All 4 features"),
                ("instant", wf_phase2b_instant, "temp_div only"),
                ("cumul",   wf_phase2b_cumul,   "cumul_div only"),
                ("runmax",  wf_phase2b_runmax,  "running_max only"),
                ("slope",   wf_phase2b_slope,   "slope_div only"),
            ],
        },
        "gfs": {
            "run_hours": [0],
            "model_name": "gfs",
            "p2_baseline": wf_regression_full_gfs,
            "ablation": [
                ("full",    wf_phase2b_full_gfs,    "All 4 features"),
                ("instant", wf_phase2b_instant_gfs, "temp_div only"),
                ("cumul",   wf_phase2b_cumul_gfs,   "cumul_div only"),
                ("runmax",  wf_phase2b_runmax_gfs,  "running_max only"),
                ("slope",   wf_phase2b_slope_gfs,   "slope_div only"),
            ],
        },
        "ecmwf": {
            "run_hours": [0],
            "model_name": "ecmwf",
            "p2_baseline": wf_regression_full_ecmwf,
            "ablation": [
                ("full",    wf_phase2b_full_ecmwf,    "All 4 features"),
                ("instant", wf_phase2b_instant_ecmwf, "temp_div only"),
                ("cumul",   wf_phase2b_cumul_ecmwf,   "cumul_div only"),
                ("runmax",  wf_phase2b_runmax_ecmwf,  "running_max only"),
                ("slope",   wf_phase2b_slope_ecmwf,   "slope_div only"),
            ],
        },
    }

    all_results = {}   # {model: {variant: BacktestResult}}
    p2_baselines = {}  # {model: BacktestResult}
    champions = {}     # {model: (variant_name, model_fn, BacktestResult)}

    for model_label, cfg in model_configs.items():
        print(f"\n{'='*60}")
        print(f"  {model_label.upper()} Phase 2B Ablation")
        print(f"{'='*60}")

        bt = Backtester(db_path=DB_PATH, city="NYC")

        # Phase 2 baseline (no update hours)
        logger.info(f"Running Phase 2 baseline for {model_label.upper()}...")
        p2_result = bt.run(
            cfg["p2_baseline"],
            run_hours=cfg["run_hours"],
            model_name=cfg["model_name"],
        )
        p2_baselines[model_label] = p2_result
        p2_brier = p2_result.mean_brier

        model_results = {}
        best_name = None
        best_brier = p2_brier
        best_fn = cfg["p2_baseline"]

        for variant_name, fn, desc in cfg["ablation"]:
            logger.info(f"Running {model_label.upper()} phase2b_{variant_name} ({desc})...")
            result = bt.run(
                fn,
                run_hours=cfg["run_hours"],
                model_name=cfg["model_name"],
                update_hours_et=UPDATE_HOURS_ET,
            )
            model_results[variant_name] = result

            if result.mean_brier < best_brier:
                best_brier = result.mean_brier
                best_name = variant_name
                best_fn = fn

        all_results[model_label] = model_results

        # Print results
        print(f"\n{model_label.upper()} Phase 2B Results (Phase 2 baseline: {p2_brier:.4f}):")
        print(f"{'Variant':<20} | {'Brier':>7} | {'vs P2':>7} | {'Evals':>6} | {'Gate':>8}")
        print("-" * 65)

        for variant_name, fn, desc in cfg["ablation"]:
            result = model_results[variant_name]
            pct = (1.0 - result.mean_brier / p2_brier) * 100 if p2_brier > 0 else 0
            if pct > 2.0:
                gate = "KEEP"
            elif pct > 1.0:
                gate = "DISCUSS"
            else:
                gate = "MARGINAL"
            print(
                f"p2b_{variant_name:<15} | {result.mean_brier:7.4f} | "
                f"{pct:+6.1f}% | {result.total_evaluations:6d} | {gate:>8}"
            )

        print("-" * 65)
        if best_name:
            print(f"  Champion: p2b_{best_name} (Brier {best_brier:.4f}, "
                  f"{(1.0 - best_brier / p2_brier) * 100:+.1f}% vs P2)")
            champions[model_label] = (best_name, best_fn, model_results[best_name])
        else:
            print(f"  No Phase 2B variant beat Phase 2 baseline.")
            champions[model_label] = (None, cfg["p2_baseline"], p2_result)

        # Improvement curve for best model
        best_result = model_results.get(best_name or "full")
        if best_result:
            by_update = getattr(best_result, 'by_update_hour', {})
            if by_update:
                print(f"\n  Improvement curve ({best_name or 'full'}):")
                print(f"  {'Hour':>4s}  {'Brier':>7s}  {'vs P2':>7s}")
                print(f"  {'----':>4s}  {'-------':>7s}  {'-------':>7s}")
                for uhr in sorted(by_update.keys()):
                    b = by_update[uhr]['mean_brier']
                    pct = (1.0 - b / p2_brier) * 100
                    print(f"  {uhr:4d}  {b:7.4f}  {pct:+6.1f}%")

    return all_results, p2_baselines, champions


# ===================================================================
# Part 6: Ensemble with Phase 2B
# ===================================================================

def part6_ensemble_with_phase2b(champions=None):
    """Evaluate ensemble with Phase 2B at update hours 14-18 ET.

    For each date and run_hour, at each update_hour:
      - HRRR: Phase 2B champion at model_run=date+run_hour
      - GFS: Phase 2B champion at model_run=date+00z
      - ECMWF: Phase 2B champion at model_run=date+00z
    """
    print("\n" + "=" * 100)
    print("PART 6: Ensemble with Phase 2B (update_hours 14-18 ET)")
    print("=" * 100)

    # Determine champion model functions
    if champions is None:
        hrrr_fn = wf_phase2b_full
        gfs_fn = wf_phase2b_full_gfs
        ecmwf_fn = wf_phase2b_full_ecmwf
        print("  Using default champions: phase2b_full per model")
    else:
        hrrr_info = champions.get("hrrr", (None, wf_phase2b_full, None))
        gfs_info = champions.get("gfs", (None, wf_phase2b_full_gfs, None))
        ecmwf_info = champions.get("ecmwf", (None, wf_phase2b_full_ecmwf, None))
        hrrr_fn = hrrr_info[1]
        gfs_fn = gfs_info[1]
        ecmwf_fn = ecmwf_info[1]
        print(f"  Champions: HRRR={hrrr_info[0]}, GFS={gfs_info[0]}, ECMWF={ecmwf_info[0]}")

    # Also get Phase 2 (no obs) functions for comparison
    hrrr_p2_fn = wf_regression_full
    gfs_p2_fn = wf_regression_full_gfs
    ecmwf_p2_fn = wf_regression_full_ecmwf

    con = duckdb.connect(DB_PATH, read_only=True)

    dates = con.execute("""
        SELECT obs_date, max_temp_f
        FROM nws_daily
        WHERE station_id = 'KNYC'
        AND max_temp_f IS NOT NULL
        ORDER BY obs_date
    """).fetchall()

    run_hours = [0, 6, 12, 18]

    # Results: {update_hour: {run_hour: [brier]}}
    ens_p2b_by_update = {uhr: {h: [] for h in run_hours} for uhr in UPDATE_HOURS_ET}
    ens_p2_by_update = {uhr: {h: [] for h in run_hours} for uhr in UPDATE_HOURS_ET}
    hrrr_p2b_by_update = {uhr: {h: [] for h in run_hours} for uhr in UPDATE_HOURS_ET}

    from zoneinfo import ZoneInfo
    _et_tz = ZoneInfo("America/New_York")

    for i, (obs_date, actual_high) in enumerate(dates):
        for hour in run_hours:
            hrrr_run_utc = datetime(
                obs_date.year, obs_date.month, obs_date.day,
                hour, 0, tzinfo=timezone.utc,
            )
            global_run_utc = datetime(
                obs_date.year, obs_date.month, obs_date.day,
                0, 0, tzinfo=timezone.utc,
            )

            for uhr in UPDATE_HOURS_ET:
                # Compute ref_time for this update hour
                ref_et = datetime(
                    obs_date.year, obs_date.month, obs_date.day,
                    uhr, 0, tzinfo=_et_tz,
                )
                ref_utc = ref_et.astimezone(timezone.utc)

                # Skip if HRRR forecast wouldn't be available yet
                earliest = hrrr_run_utc + timedelta(hours=HRRR_AVAILABILITY_LAG_HOURS)
                if ref_utc < earliest:
                    continue

                # HRRR provider + Phase 2B raw
                hrrr_provider = BacktestDataProvider(
                    db_path=DB_PATH, station_id="KNYC",
                    model_run=hrrr_run_utc, ref_time=ref_utc,
                    connection=con, model_name="hrrr",
                )
                hrrr_raw = hrrr_fn.raw(hrrr_provider, ref_utc)
                if hrrr_raw is None:
                    continue

                # HRRR-only Brier (for comparison)
                hrrr_brackets = _compute_bracket_probs_from_dist(
                    norm(0, 1), hrrr_raw[0], hrrr_raw[1]
                )
                hrrr_brier = compute_brier_score_1f(hrrr_brackets, actual_high)
                hrrr_p2b_by_update[uhr][hour].append(hrrr_brier)

                # GFS provider (midnight run, same ref_time for obs cutoff)
                gfs_provider = BacktestDataProvider(
                    db_path=DB_PATH, station_id="KNYC",
                    model_run=global_run_utc, ref_time=ref_utc,
                    connection=con, model_name="gfs",
                )
                # ECMWF provider
                ecmwf_provider = BacktestDataProvider(
                    db_path=DB_PATH, station_id="KNYC",
                    model_run=global_run_utc, ref_time=ref_utc,
                    connection=con, model_name="ecmwf",
                )

                # --- Phase 2B ensemble ---
                predictions = [hrrr_raw]
                weights = [1.0]

                gfs_raw = gfs_fn.raw(gfs_provider, ref_utc)
                if gfs_raw is not None:
                    predictions.append(gfs_raw)
                    weights.append(1.0)

                ecmwf_raw = ecmwf_fn.raw(ecmwf_provider, ref_utc)
                if ecmwf_raw is not None:
                    predictions.append(ecmwf_raw)
                    weights.append(1.0)

                total_w = sum(weights)
                weights = [w / total_w for w in weights]

                ens_brackets = combine_mixture_brackets(predictions, weights)
                if ens_brackets:
                    ens_brier = compute_brier_score_1f(ens_brackets, actual_high)
                    ens_p2b_by_update[uhr][hour].append(ens_brier)

                # --- Phase 2 (no obs) ensemble for comparison ---
                p2_preds = []
                p2_weights = []

                hrrr_p2_raw = hrrr_p2_fn.raw(hrrr_provider, ref_utc)
                if hrrr_p2_raw is not None:
                    p2_preds.append(hrrr_p2_raw)
                    p2_weights.append(1.0)

                gfs_p2_raw = gfs_p2_fn.raw(gfs_provider, ref_utc)
                if gfs_p2_raw is not None:
                    p2_preds.append(gfs_p2_raw)
                    p2_weights.append(1.0)

                ecmwf_p2_raw = ecmwf_p2_fn.raw(ecmwf_provider, ref_utc)
                if ecmwf_p2_raw is not None:
                    p2_preds.append(ecmwf_p2_raw)
                    p2_weights.append(1.0)

                if p2_preds:
                    total_p2w = sum(p2_weights)
                    p2_weights = [w / total_p2w for w in p2_weights]
                    p2_brackets = combine_mixture_brackets(p2_preds, p2_weights)
                    if p2_brackets:
                        ens_p2_by_update[uhr][hour].append(
                            compute_brier_score_1f(p2_brackets, actual_high)
                        )

        if (i + 1) % 200 == 0:
            logger.info(f"  Processed {i + 1}/{len(dates)} days...")

    con.close()

    # Print results
    print(f"\n{'Update':>6} | {'Ens+P2B':>8} | {'Ens P2':>8} | {'HRRR P2B':>9} | {'Ens+P2B vs Ens P2':>17} | {'N':>6}")
    print("-" * 75)

    for uhr in UPDATE_HOURS_ET:
        ens_p2b_all = []
        ens_p2_all = []
        hrrr_all = []
        for h in run_hours:
            ens_p2b_all.extend(ens_p2b_by_update[uhr][h])
            ens_p2_all.extend(ens_p2_by_update[uhr][h])
            hrrr_all.extend(hrrr_p2b_by_update[uhr][h])

        if ens_p2b_all and ens_p2_all:
            ens_p2b_mean = np.mean(ens_p2b_all)
            ens_p2_mean = np.mean(ens_p2_all)
            hrrr_mean = np.mean(hrrr_all) if hrrr_all else float('nan')
            pct = (1.0 - ens_p2b_mean / ens_p2_mean) * 100
            print(f"{uhr:4d}ET | {ens_p2b_mean:8.4f} | {ens_p2_mean:8.4f} | "
                  f"{hrrr_mean:9.4f} | {pct:+16.1f}% | {len(ens_p2b_all):6d}")
        else:
            print(f"{uhr:4d}ET | {'n/a':>8} | {'n/a':>8} | {'n/a':>9} | {'n/a':>17} | {0:6d}")

    # Overall
    all_p2b = []
    all_p2 = []
    all_hrrr = []
    for uhr in UPDATE_HOURS_ET:
        for h in run_hours:
            all_p2b.extend(ens_p2b_by_update[uhr][h])
            all_p2.extend(ens_p2_by_update[uhr][h])
            all_hrrr.extend(hrrr_p2b_by_update[uhr][h])

    if all_p2b and all_p2:
        p2b_overall = np.mean(all_p2b)
        p2_overall = np.mean(all_p2)
        hrrr_overall = np.mean(all_hrrr) if all_hrrr else float('nan')
        pct_overall = (1.0 - p2b_overall / p2_overall) * 100

        print("-" * 75)
        print(f"{'ALL':>6} | {p2b_overall:8.4f} | {p2_overall:8.4f} | "
              f"{hrrr_overall:9.4f} | {pct_overall:+16.1f}% | {len(all_p2b):6d}")

        print(f"\n  ENSEMBLE+P2B: {p2b_overall:.4f} vs ENSEMBLE P2: {p2_overall:.4f}")
        gate = p2b_overall < p2_overall
        print(f"  GATE: {'PASS' if gate else 'FAIL'} — "
              f"Phase 2B {'improves' if gate else 'does NOT improve'} the ensemble")

    return ens_p2b_by_update, ens_p2_by_update


# ===================================================================
# Part 7: Learned Weights (inverse-Brier)
# ===================================================================

def part7_learned_weights():
    """Test inverse-Brier weighting vs equal weights.

    Uses Phase 2 regression_full results (from Phase 3 analysis) to compute
    per-model Brier at run_hour=0, then assigns inverse-Brier weights.
    """
    print("\n" + "=" * 100)
    print("PART 7: Learned Weights (Inverse-Brier)")
    print("=" * 100)

    # First, get per-model Brier at run_hour=0 from Phase 2 regression_full
    bt = Backtester(db_path=DB_PATH, city="NYC")

    models_at_0z = [
        ("hrrr", wf_regression_full, "hrrr"),
        ("gfs", wf_regression_full_gfs, "gfs"),
        ("ecmwf", wf_regression_full_ecmwf, "ecmwf"),
    ]

    model_briers = {}
    for label, fn, model_name in models_at_0z:
        result = bt.run(fn, run_hours=[0], model_name=model_name)
        model_briers[label] = result.mean_brier
        logger.info(f"  {label.upper()} at 00z: Brier = {result.mean_brier:.4f}")

    # Compute inverse-Brier weights
    inv_briers = {m: 1.0 / b for m, b in model_briers.items()}
    total_inv = sum(inv_briers.values())
    learned_weights = {m: v / total_inv for m, v in inv_briers.items()}

    print(f"\nPer-model Brier at 00z:")
    for m in ["hrrr", "gfs", "ecmwf"]:
        print(f"  {m.upper()}: {model_briers[m]:.4f}")

    print(f"\nEqual weights:          HRRR={1/3:.3f}, GFS={1/3:.3f}, ECMWF={1/3:.3f}")
    print(f"Inverse-Brier weights:  HRRR={learned_weights['hrrr']:.3f}, "
          f"GFS={learned_weights['gfs']:.3f}, ECMWF={learned_weights['ecmwf']:.3f}")

    # Now run ensemble comparison: equal vs learned weights
    con = duckdb.connect(DB_PATH, read_only=True)

    dates = con.execute("""
        SELECT obs_date, max_temp_f
        FROM nws_daily
        WHERE station_id = 'KNYC'
        AND max_temp_f IS NOT NULL
        ORDER BY obs_date
    """).fetchall()

    run_hours = [0, 6, 12, 18]
    equal_briers = {h: [] for h in run_hours}
    learned_briers = {h: [] for h in run_hours}

    hrrr_fn = wf_regression_full
    gfs_fn = wf_regression_full_gfs
    ecmwf_fn = wf_regression_full_ecmwf

    w_equal = [1/3, 1/3, 1/3]
    w_learned = [learned_weights["hrrr"], learned_weights["gfs"], learned_weights["ecmwf"]]

    for i, (obs_date, actual_high) in enumerate(dates):
        for hour in run_hours:
            hrrr_run_utc = datetime(
                obs_date.year, obs_date.month, obs_date.day,
                hour, 0, tzinfo=timezone.utc,
            )
            hrrr_ref = hrrr_run_utc + timedelta(hours=HRRR_AVAILABILITY_LAG_HOURS)

            global_run_utc = datetime(
                obs_date.year, obs_date.month, obs_date.day,
                0, 0, tzinfo=timezone.utc,
            )
            global_ref = global_run_utc + timedelta(hours=HRRR_AVAILABILITY_LAG_HOURS)

            hrrr_provider = BacktestDataProvider(
                db_path=DB_PATH, station_id="KNYC",
                model_run=hrrr_run_utc, ref_time=hrrr_ref,
                connection=con, model_name="hrrr",
            )
            gfs_provider = BacktestDataProvider(
                db_path=DB_PATH, station_id="KNYC",
                model_run=global_run_utc, ref_time=global_ref,
                connection=con, model_name="gfs",
            )
            ecmwf_provider = BacktestDataProvider(
                db_path=DB_PATH, station_id="KNYC",
                model_run=global_run_utc, ref_time=global_ref,
                connection=con, model_name="ecmwf",
            )

            # Collect raw predictions
            raws = []
            hrrr_raw = hrrr_fn.raw(hrrr_provider, hrrr_ref)
            if hrrr_raw is not None:
                raws.append(("hrrr", hrrr_raw))

            gfs_raw = gfs_fn.raw(gfs_provider, global_ref)
            if gfs_raw is not None:
                raws.append(("gfs", gfs_raw))

            ecmwf_raw = ecmwf_fn.raw(ecmwf_provider, global_ref)
            if ecmwf_raw is not None:
                raws.append(("ecmwf", ecmwf_raw))

            if not raws:
                continue

            # Equal weight ensemble
            preds = [r for _, r in raws]
            n = len(preds)
            eq_w = [1.0 / n] * n
            eq_brackets = combine_mixture_brackets(preds, eq_w)
            if eq_brackets:
                equal_briers[hour].append(compute_brier_score_1f(eq_brackets, actual_high))

            # Learned weight ensemble
            lw_weights = []
            for label, _ in raws:
                lw_weights.append(learned_weights[label])
            total_lw = sum(lw_weights)
            lw_weights = [w / total_lw for w in lw_weights]

            lw_brackets = combine_mixture_brackets(preds, lw_weights)
            if lw_brackets:
                learned_briers[hour].append(compute_brier_score_1f(lw_brackets, actual_high))

        if (i + 1) % 200 == 0:
            logger.info(f"  Processed {i + 1}/{len(dates)} days...")

    con.close()

    # Print comparison
    print(f"\n{'Run Hour':<10} | {'Equal Wt':>9} | {'Learned Wt':>10} | {'Delta':>7} | {'N':>6}")
    print("-" * 55)

    for hour in run_hours:
        if equal_briers[hour] and learned_briers[hour]:
            eq = np.mean(equal_briers[hour])
            lw = np.mean(learned_briers[hour])
            pct = (1.0 - lw / eq) * 100
            print(f"{hour:02d}z       | {eq:9.4f} | {lw:10.4f} | {pct:+6.1f}% | {len(equal_briers[hour]):6d}")

    # Overall
    all_eq = []
    all_lw = []
    for h in run_hours:
        all_eq.extend(equal_briers[h])
        all_lw.extend(learned_briers[h])

    if all_eq and all_lw:
        eq_overall = np.mean(all_eq)
        lw_overall = np.mean(all_lw)
        pct_overall = (1.0 - lw_overall / eq_overall) * 100

        print("-" * 55)
        print(f"{'Overall':<10} | {eq_overall:9.4f} | {lw_overall:10.4f} | {pct_overall:+6.1f}% | {len(all_eq):6d}")

        print(f"\n  LEARNED WEIGHTS: {lw_overall:.4f} vs EQUAL WEIGHTS: {eq_overall:.4f}")
        if lw_overall < eq_overall:
            print(f"  RESULT: Learned weights win by {pct_overall:+.1f}% — use inverse-Brier weights")
        else:
            print(f"  RESULT: Equal weights win — keep equal weights")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3B Analysis — Conditional Steps")
    parser.add_argument("--part", type=int, choices=[5, 6, 7], default=None,
                        help="Run only a specific part (5-7). Default: run all.")
    args = parser.parse_args()

    champions = None

    if args.part is None or args.part == 5:
        _, _, champions = part5_phase2b_per_model()

    if args.part is None or args.part == 6:
        part6_ensemble_with_phase2b(champions)

    if args.part is None or args.part == 7:
        part7_learned_weights()

    print("\n" + "=" * 100)
    print("Phase 3B Analysis Complete")
    print("=" * 100)


if __name__ == "__main__":
    main()
