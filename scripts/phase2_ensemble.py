"""Phase 2 Task 7: Multi-model ensemble comparison.

Tests whether combining HRRR + GFS + ECMWF via latest-run ensemble
with age-based decay weights beats single-model HRRR OLS.

Usage:
    PYTHONPATH=. python scripts/phase2_ensemble.py
    PYTHONPATH=. python scripts/phase2_ensemble.py --start 2023-06-01 --end 2025-12-31
"""

import argparse
from datetime import date
from typing import Dict, List, Optional

from loguru import logger

from core.db import DEFAULT_DB_PATH
from services.backtester import (
    Backtester,
    RUN_HOURS,
    wf_regression_full,
    wf_regression_full_gfs,
    wf_regression_full_ecmwf,
    wf_multimodel_full,
    wf_multimodel_hrrr_gfs,
)
from services.ensemble import make_latest_run_ensemble_fn, make_learned_weight_ensemble_fn


def run_ensemble_comparison(db_path=DEFAULT_DB_PATH, start=None, end=None, run_hours=None):
    # type: (str, Optional[date], Optional[date], Optional[List[int]]) -> Dict
    """Run HRRR baseline + GFS/ECMWF per-model + ensemble variants."""
    start = start or date(2023, 1, 1)
    end = end or date(2026, 2, 1)
    hours = run_hours or RUN_HOURS

    bt = Backtester(db_path=db_path)
    results = {}

    # 1. HRRR OLS baseline (all 24 hours)
    logger.info("Running HRRR OLS baseline (all hours)...")
    results["hrrr_ols"] = bt.run(wf_regression_full, start, end, run_hours=hours)

    # 2. GFS OLS (synoptic hours only — where GFS data exists)
    gfs_hours = [h for h in hours if h in [0, 6, 12, 18]]
    if gfs_hours:
        logger.info("Running GFS OLS (hours {})...".format(gfs_hours))
        try:
            results["gfs_ols"] = bt.run(
                wf_regression_full_gfs, start, end,
                run_hours=gfs_hours, model_name='gfs',
            )
        except Exception as e:
            logger.error("GFS OLS failed: {}", e)

    # 3. ECMWF OLS (00z only — where ECMWF open data exists)
    ecmwf_hours = [h for h in hours if h in [0]]
    if ecmwf_hours:
        logger.info("Running ECMWF OLS (hours {})...".format(ecmwf_hours))
        try:
            results["ecmwf_ols"] = bt.run(
                wf_regression_full_ecmwf, start, end,
                run_hours=ecmwf_hours, model_name='ecmwf',
            )
        except Exception as e:
            logger.error("ECMWF OLS failed: {}", e)

    # 4-6. Ensemble variants with different decay rates
    for lam in [0.05, 0.1, 0.2]:
        label = "ensemble_l{:.2f}".format(lam)
        logger.info("Running {} (all hours)...".format(label))
        try:
            ensemble_fn = make_latest_run_ensemble_fn([
                (wf_regression_full, 'hrrr'),
                (wf_regression_full_gfs, 'gfs'),
                (wf_regression_full_ecmwf, 'ecmwf'),
            ], decay_lambda=lam)
            results[label] = bt.run(ensemble_fn, start, end, run_hours=hours)
        except Exception as e:
            logger.error("{} failed: {}", label, e)

    # 7. Level 3: Multi-model OLS stacking (HRRR + GFS + ECMWF)
    logger.info("Running multimodel_full (HRRR+GFS+ECMWF OLS stacking, all hours)...")
    try:
        results["multimodel_full"] = bt.run(
            wf_multimodel_full, start, end, run_hours=hours,
        )
    except Exception as e:
        logger.error("multimodel_full failed: {}", e)

    # 8. Level 3: Multi-model OLS stacking (HRRR + GFS only)
    logger.info("Running multimodel_hg (HRRR+GFS OLS stacking, all hours)...")
    try:
        results["multimodel_hg"] = bt.run(
            wf_multimodel_hrrr_gfs, start, end, run_hours=hours,
        )
    except Exception as e:
        logger.error("multimodel_hg failed: {}", e)

    # 9. Level 2: Walk-forward learned mixture weights
    # DISABLED — too slow without .raw() caching (~hours for full backtest).
    # TODO: precompute all model .raw() predictions per run_hour, then grid search.
    # See task #6.
    # logger.info("Running learned_weights (walk-forward per-hour, all hours)...")
    # learned_fn = make_learned_weight_ensemble_fn([
    #     (wf_regression_full, 'hrrr'),
    #     (wf_regression_full_gfs, 'gfs'),
    #     (wf_regression_full_ecmwf, 'ecmwf'),
    # ], lookback_days=365, reoptimize_every=30)
    # results["learned_weights"] = bt.run(learned_fn, start, end, run_hours=hours)

    _print_comparison(results)
    _print_per_hour_breakdown(results)

    return results


def _print_comparison(results):
    # type: (Dict) -> None
    """Print summary comparison table."""
    baseline = results.get("hrrr_ols")
    baseline_brier = baseline.mean_brier if baseline else 1.0

    print("\n" + "=" * 75)
    print("PHASE 2 TASK 7: MULTI-MODEL ENSEMBLE COMPARISON")
    print("=" * 75)
    print("{:<30} {:>8} {:>10} {:>9} {:>9}".format(
        "Model", "Brier", "vs HRRR", "N Evals", "Hit%"))
    print("-" * 75)

    for name, result in sorted(results.items(), key=lambda x: x[1].mean_brier):
        delta = (baseline_brier - result.mean_brier) / baseline_brier * 100
        sign = "+" if delta >= 0 else ""
        n = len(result.run_results)
        hits = sum(1 for r in result.run_results if r.hit)
        hit_pct = 100.0 * hits / n if n > 0 else 0
        print("{:<30} {:>8.4f} {:>9}{:.1f}% {:>9} {:>8.1f}%".format(
            name, result.mean_brier, sign, delta, n, hit_pct))

    print("-" * 75)

    best_name = min(results, key=lambda k: results[k].mean_brier)
    best_brier = results[best_name].mean_brier
    improvement = (baseline_brier - best_brier) / baseline_brier * 100

    print("CHAMPION: {} (Brier {:.4f})".format(best_name, best_brier))
    print("Improvement over HRRR OLS: {:.2f}%".format(improvement))

    if best_name != "hrrr_ols" and improvement > 0:
        print("\nMULTI-MODEL WINS — record optimal configuration")
    else:
        print("\nHRRR OLS REMAINS CHAMPION — multi-model provides no value")


def _print_per_hour_breakdown(results):
    # type: (Dict) -> None
    """Show per-hour Brier for baseline vs best ensemble, highlighting where ensemble helps."""
    baseline = results.get("hrrr_ols")
    if baseline is None:
        return

    # Find best non-baseline model
    ensemble_names = [k for k in results if k != "hrrr_ols"]
    if not ensemble_names:
        return

    best_ens_name = min(ensemble_names, key=lambda k: results[k].mean_brier)
    best_ens = results[best_ens_name]

    print("\n" + "=" * 75)
    print("PER-HOUR BREAKDOWN: HRRR OLS vs {}".format(best_ens_name))
    print("=" * 75)
    print("{:>5} {:>12} {:>12} {:>10} {:>10} {:>10}".format(
        "Hour", "HRRR Brier", "Ens Brier", "Delta", "HRRR n", "Ens n"))
    print("-" * 75)

    all_hours = sorted(set(list(baseline.by_run_hour.keys()) +
                           list(best_ens.by_run_hour.keys())))

    helps = 0
    hurts = 0
    for hour in all_hours:
        hrrr_m = baseline.by_run_hour.get(hour, {})
        ens_m = best_ens.by_run_hour.get(hour, {})
        hrrr_b = hrrr_m.get('mean_brier', float('nan'))
        ens_b = ens_m.get('mean_brier', float('nan'))
        hrrr_n = hrrr_m.get('count', 0)
        ens_n = ens_m.get('count', 0)

        delta = hrrr_b - ens_b  # positive = ensemble better
        marker = ""
        if delta > 0.001:
            marker = " <-- helps"
            helps += 1
        elif delta < -0.001:
            marker = " <-- hurts"
            hurts += 1

        print("{:>4}z {:>12.4f} {:>12.4f} {:>+10.4f} {:>10} {:>10}{}".format(
            hour, hrrr_b, ens_b, delta, hrrr_n, ens_n, marker))

    print("-" * 75)
    print("Ensemble helps at {} hours, hurts at {} hours".format(helps, hurts))


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 Task 7: Multi-model ensemble comparison")
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--run-hours", default=None,
                        help="Comma-separated run hours (default: all 24)")
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    run_hours = None
    if args.run_hours:
        run_hours = [int(h) for h in args.run_hours.split(",")]

    run_ensemble_comparison(args.db, args.start, args.end, run_hours)


if __name__ == "__main__":
    main()
