"""Head-to-head comparison of Phase 2 bias correction candidates.

Runs all candidates on identical date ranges and compares:
- Mean Brier score
- Per-hour Brier breakdown
- Top-1 hit rates

Usage:
    python scripts/phase2_compare.py
    python scripts/phase2_compare.py --start 2023-06-01 --end 2025-12-31
"""

import argparse
from datetime import date
from typing import Dict, Optional, List

from loguru import logger

from core.db import DEFAULT_DB_PATH
from services.backtester import Backtester, RUN_HOURS


def compare_candidates(db_path=DEFAULT_DB_PATH, start=None, end=None, run_hours=None):
    # type: (str, Optional[date], Optional[date], Optional[List[int]]) -> Dict
    """Run all candidates and compare."""
    start = start or date(2023, 1, 1)
    end = end or date(2026, 2, 1)
    hours = run_hours or RUN_HOURS

    bt = Backtester(db_path=db_path)
    candidates = {}

    # Candidate A: OLS (24h baseline)
    from services.backtester import wf_regression_full
    logger.info("Running Candidate A: OLS (24h baseline)...")
    candidates["ols"] = bt.run(wf_regression_full, start, end, run_hours=hours)

    # Candidate A': OLS (no spin-up)
    from services.backtester import wf_regression_full_no_spinup
    logger.info("Running Candidate A': OLS (no spin-up)...")
    candidates["ols_no_spinup"] = bt.run(wf_regression_full_no_spinup, start, end, run_hours=hours)

    # Candidate B: EMOS
    try:
        from services.phase2_emos import emos_model_fn
        logger.info("Running Candidate B: EMOS...")
        candidates["emos"] = bt.run(emos_model_fn, start, end, run_hours=hours)
    except Exception as e:
        logger.error("EMOS failed: {}", e)

    # Candidate C: XGBoost QR (base features)
    try:
        from services.phase2_xgboost import xgboost_model_fn
        logger.info("Running Candidate C: XGBoost QR...")
        candidates["xgboost_qr"] = bt.run(xgboost_model_fn, start, end, run_hours=hours)
    except Exception as e:
        logger.error("XGBoost failed: {}", e)

    # Candidate C': XGBoost QR (extended weather features)
    try:
        from services.phase2_xgboost import xgboost_extended_model_fn
        logger.info("Running Candidate C': XGBoost QR (extended features)...")
        candidates["xgboost_ext"] = bt.run(xgboost_extended_model_fn, start, end, run_hours=hours)
    except Exception as e:
        logger.error("XGBoost extended failed: {}", e)

    # Candidate D: Cross-hour
    from services.backtester import wf_regression_cross_hour
    logger.info("Running Candidate D: Cross-hour OLS...")
    candidates["cross_hour"] = bt.run(wf_regression_cross_hour, start, end, run_hours=hours)

    _print_comparison(candidates)

    # Per-model breakdown for the winner
    baseline_brier = candidates["ols"].mean_brier
    best_name = min(candidates, key=lambda k: candidates[k].mean_brier)
    best_brier = candidates[best_name].mean_brier
    improvement = (baseline_brier - best_brier) / baseline_brier * 100

    if improvement >= 2.0:
        logger.info("Running per-model breakdown for winner: {}...", best_name)
        _per_model_breakdown(bt, start, end, hours)

    return candidates


def _print_comparison(candidates):
    # type: (Dict) -> None
    """Print formatted comparison table."""
    baseline = candidates.get("ols")
    baseline_brier = baseline.mean_brier if baseline else 1.0

    print("\n" + "=" * 70)
    print("PHASE 2 HEAD-TO-HEAD COMPARISON")
    print("=" * 70)
    print("{:<30} {:>8} {:>9} {:>9}".format(
        "Candidate", "Brier", "vs Base", "N Evals"))
    print("-" * 70)

    for name, result in sorted(candidates.items(), key=lambda x: x[1].mean_brier):
        delta = (baseline_brier - result.mean_brier) / baseline_brier * 100
        sign = "+" if delta >= 0 else ""
        print("{:<30} {:>8.4f} {:>8}{:.1f}% {:>9}".format(
            name, result.mean_brier, sign, delta, len(result.run_results)))

    print("-" * 70)

    best_name = min(candidates, key=lambda k: candidates[k].mean_brier)
    best_brier = candidates[best_name].mean_brier
    improvement = (baseline_brier - best_brier) / baseline_brier * 100

    print("CHAMPION: {} (Brier {:.4f})".format(best_name, best_brier))
    print("Improvement over OLS: {:.1f}%".format(improvement))

    if improvement >= 2.0:
        print("\nPHASE 2 GATE: PASSED (>= 2% Brier improvement)")
    else:
        print("\nPHASE 2 GATE: PENDING (< 2% improvement)")

    # Per-hour breakdown for champion
    champion = candidates[best_name]
    print("\n{} per-hour Brier:".format(best_name))
    for hour, metrics in sorted(champion.by_run_hour.items()):
        print("  {:02d}z: {:.4f} (n={})".format(
            hour, metrics['mean_brier'], metrics['count']))


def _per_model_breakdown(bt, start, end, run_hours):
    # type: (Backtester, date, date, List[int]) -> None
    """Run winner on GFS/ECMWF variants for per-model bias quality."""
    from services.backtester import wf_regression_full_gfs, wf_regression_full_ecmwf

    # GFS only runs at synoptic hours
    gfs_hours = [h for h in run_hours if h in [0, 6, 12, 18]]
    ecmwf_hours = [h for h in run_hours if h in [0, 6, 12, 18]]

    print("\n--- Per-Model Breakdown ---")
    try:
        logger.info("Running GFS variant...")
        gfs_result = bt.run(wf_regression_full_gfs, start, end,
                            run_hours=gfs_hours, model_name='gfs')
        print("GFS OLS: Brier {:.4f} (n={})".format(
            gfs_result.mean_brier, len(gfs_result.run_results)))
    except Exception as e:
        logger.error("GFS failed: {}", e)

    try:
        logger.info("Running ECMWF variant...")
        ecmwf_result = bt.run(wf_regression_full_ecmwf, start, end,
                              run_hours=ecmwf_hours, model_name='ecmwf')
        print("ECMWF OLS: Brier {:.4f} (n={})".format(
            ecmwf_result.mean_brier, len(ecmwf_result.run_results)))
    except Exception as e:
        logger.error("ECMWF failed: {}", e)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 head-to-head comparison")
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--run-hours", default=None,
                        help="Comma-separated run hours (default: all 24)")
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    run_hours = None
    if args.run_hours:
        run_hours = [int(h) for h in args.run_hours.split(",")]

    compare_candidates(args.db, args.start, args.end, run_hours)


if __name__ == "__main__":
    main()
