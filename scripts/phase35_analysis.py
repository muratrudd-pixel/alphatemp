"""Phase 3.5: Extended variable ablation analysis.

Tests each extended weather variable as an additional regression feature
for GFS and ECMWF models. Compares against Phase 2 baseline (fcst_high + month).

Usage:
    PYTHONPATH=. python scripts/phase35_analysis.py [--part N]

Parts:
    1 = Per-variable ablation (each extended variable individually)
    2 = Best combination testing
    3 = Ensemble re-evaluation with extended features
"""

import argparse
from datetime import date

from loguru import logger

from services.backtester import (
    Backtester,
    _make_regression_model,
    _FEATURE_SETS,
    _EXTENDED_FEATURE_SETS,
)

DB_PATH = "data/alphatemp.duckdb"
STATION_ID = "KNYC"


def part1_per_variable_ablation():
    """Test each extended variable individually for GFS and ECMWF."""
    print()
    print("=" * 100)
    print("PART 1: Per-Variable Extended Feature Ablation")
    print("=" * 100)

    bt = Backtester(DB_PATH, "NYC")

    for model_name in ["gfs", "ecmwf"]:
        print()
        print("-" * 80)
        print("Model: {}".format(model_name.upper()))
        print("-" * 80)

        # Baseline: Phase 2 champion (fcst_high + month, no extended)
        baseline_fn = _make_regression_model(
            "baseline_{}".format(model_name),
            [0, 1, 2],  # fcst_high + sin_month + cos_month
            model_name=model_name,
            extended=False,
        )
        baseline_result = bt.run(baseline_fn, run_hours=[0], model_name=model_name)
        baseline_brier = baseline_result.mean_brier

        # Full Phase 2 (with delta_temp) for reference
        full_fn = _make_regression_model(
            "full_{}".format(model_name),
            [0, 1, 2, 3],
            model_name=model_name,
            extended=False,
        )
        full_result = bt.run(full_fn, run_hours=[0], model_name=model_name)
        full_brier = full_result.mean_brier

        print()
        print("  {:<30s} {:>10s} {:>10s} {:>8s}".format(
            "Feature Set", "Brier(1F)", "vs Base", "N"))
        print("  " + "-" * 62)
        print("  {:<30s} {:>10.4f} {:>10s} {:>8d}".format(
            "baseline (fcst+month)", baseline_brier, "---", baseline_result.total_evaluations))
        print("  {:<30s} {:>10.4f} {:>10s} {:>8d}".format(
            "full_p2 (fcst+month+delta)", full_brier,
            "{:+.1f}%".format((baseline_brier - full_brier) / baseline_brier * 100),
            full_result.total_evaluations))

        # Test each extended variable
        ext_results = []
        for ext_name, indices in sorted(_EXTENDED_FEATURE_SETS.items()):
            fn = _make_regression_model(
                "{}_{}".format(ext_name, model_name),
                indices,
                model_name=model_name,
                extended=True,
            )
            result = bt.run(fn, run_hours=[0], model_name=model_name)
            brier = result.mean_brier
            delta_pct = (baseline_brier - brier) / baseline_brier * 100
            ext_results.append((ext_name, brier, delta_pct, result.total_evaluations))
            print("  {:<30s} {:>10.4f} {:>10s} {:>8d}".format(
                ext_name, brier,
                "{:+.1f}%".format(delta_pct),
                result.total_evaluations))

        # Sort by improvement
        ext_results.sort(key=lambda x: -x[2])
        print()
        print("  Ranked by improvement:")
        survivors = []
        for name, brier, delta, n in ext_results:
            gate = "PASS" if delta > 2.0 else "FAIL"
            print("    {:>+6.1f}%  {:<25s} ({})".format(delta, name, gate))
            if delta > 2.0:
                survivors.append(name)

        print()
        if survivors:
            print("  SURVIVORS (>2% improvement): {}".format(", ".join(survivors)))
        else:
            print("  NO SURVIVORS — no variable beat 2% gate")


def part2_best_combination():
    """Test combinations of surviving extended variables."""
    print()
    print("=" * 100)
    print("PART 2: Best Combination of Extended Variables")
    print("=" * 100)

    bt = Backtester(DB_PATH, "NYC")

    for model_name in ["gfs", "ecmwf"]:
        print()
        print("-" * 80)
        print("Model: {}".format(model_name.upper()))
        print("-" * 80)

        # Baseline
        baseline_fn = _make_regression_model(
            "baseline_{}".format(model_name),
            [0, 1, 2],
            model_name=model_name,
            extended=False,
        )
        baseline_result = bt.run(baseline_fn, run_hours=[0], model_name=model_name)
        baseline_brier = baseline_result.mean_brier

        # Test combinations: all extended, top survivors, etc.
        combos = {
            # All variables that passed signal gate for both models
            "all_shared": [0, 1, 2, 4, 5, 6, 7, 9, 11],  # dewpoint, humidity, wind, pressure, precip, dewdep
            # Top-4 by signal strength (across both models)
            "top4": [0, 1, 2, 5, 6, 7, 11],  # humidity, wind, pressure, dewdep
            # Top-3
            "top3": [0, 1, 2, 5, 7, 11],  # humidity, pressure, dewdep
            # Full P2 + all extended
            "full_plus_all": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            # Full P2 + top-3
            "full_plus_top3": [0, 1, 2, 3, 5, 7, 11],
        }

        if model_name == "gfs":
            # GFS-specific: all 11 variables passed signal
            combos["gfs_all"] = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
            # GFS top by r: wind_dir(0.220), dewdep(0.214), humidity(-0.196), cloud(-0.183), radiation(0.182)
            combos["gfs_top5"] = [0, 1, 2, 5, 8, 10, 11, 6]  # humidity, cloud, radiation, dewdep, wind

        print()
        print("  {:<35s} {:>10s} {:>10s} {:>8s}".format(
            "Combination", "Brier(1F)", "vs Base", "N"))
        print("  " + "-" * 67)
        print("  {:<35s} {:>10.4f} {:>10s} {:>8d}".format(
            "baseline (fcst+month)", baseline_brier, "---", baseline_result.total_evaluations))

        for combo_name, indices in sorted(combos.items()):
            fn = _make_regression_model(
                "combo_{}_{}".format(combo_name, model_name),
                indices,
                model_name=model_name,
                extended=True,
            )
            result = bt.run(fn, run_hours=[0], model_name=model_name)
            brier = result.mean_brier
            delta_pct = (baseline_brier - brier) / baseline_brier * 100
            print("  {:<35s} {:>10.4f} {:>10s} {:>8d}".format(
                combo_name, brier,
                "{:+.1f}%".format(delta_pct),
                result.total_evaluations))


def part3_ensemble_reevaluation():
    """Re-evaluate ensemble with best extended feature sets."""
    print()
    print("=" * 100)
    print("PART 3: Ensemble Re-evaluation with Extended Features")
    print("=" * 100)
    print()
    print("  (Run after Parts 1-2 identify champion feature sets per model)")
    print("  (Use scripts/phase3_analysis.py Part 4 pattern with updated model functions)")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3.5: Extended variable ablation"
    )
    parser.add_argument(
        "--part", type=int, default=0,
        help="Run specific part (1, 2, or 3). 0 = run all.",
    )
    args = parser.parse_args()

    parts = {
        1: part1_per_variable_ablation,
        2: part2_best_combination,
        3: part3_ensemble_reevaluation,
    }

    if args.part == 0:
        for p in [1, 2]:
            parts[p]()
    elif args.part in parts:
        parts[args.part]()
    else:
        print("Unknown part: {}".format(args.part))


if __name__ == "__main__":
    main()
