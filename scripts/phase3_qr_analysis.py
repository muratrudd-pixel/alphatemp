#!/usr/bin/env python3
"""Phase 3: Quantile Regression — Ablation Analysis.

Compares 3 QR variants against the Phase 2 champion (multimodel_full).

Variants:
  - multimodel_full: Phase 2 champion — multi-model OLS stacking (Brier 0.6511)
  - qr_base:         QR with update_hour + fcst_high + sin/cos month
  - qr_full:         QR with all 6 features (+ divergence)
  - qr_multimodel:   QR with 8 features (+ GFS/ECMWF highs)

Gate: >2% Brier improvement over multimodel_full AND P&L trajectory improvement.

Usage:
  PYTHONPATH=. python scripts/phase3_qr_analysis.py          # Full run (all hours, all dates)
  PYTHONPATH=. python scripts/phase3_qr_analysis.py --quick   # Sampled hours, last 2 years
"""

import argparse
import sys
import time
from datetime import date, timedelta

sys.path.insert(0, ".")

from loguru import logger
from services.backtester import (
    Backtester,
    wf_multimodel_full,
    wf_qr_base,
    wf_qr_full,
    wf_qr_multimodel,
    wf_qr_multimodel_v2,
)

DB_PATH = "data/alphatemp.duckdb"

# Full 0-23 ET cycle — strategy backtest showed edge at 21-23z
UPDATE_HOURS_FULL = list(range(0, 24))

# Quick mode: sample 6 representative hours across the cycle
UPDATE_HOURS_QUICK = [0, 4, 8, 12, 16, 20]

# Run hours: which HRRR model runs to evaluate
# Quick mode uses 4 representative runs to cut LP solve count by 6x
RUN_HOURS_FULL = list(range(24))
RUN_HOURS_QUICK = [0, 12]

VARIANTS = [
    ("multimodel_full", wf_multimodel_full),
    ("qr_base", wf_qr_base),
    ("qr_full", wf_qr_full),
    ("qr_multimodel", wf_qr_multimodel),
    ("qr_multimodel_v2", wf_qr_multimodel_v2),
]


def main():
    parser = argparse.ArgumentParser(description="Phase 3 QR Ablation Analysis")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 6 sampled hours, last 2 years")
    args = parser.parse_args()

    if args.quick:
        update_hours = UPDATE_HOURS_QUICK
        run_hours = RUN_HOURS_QUICK
        start_date = date.today() - timedelta(days=730)
        mode_label = "QUICK (6 update hrs, 4 run hrs, 2 years)"
    else:
        update_hours = UPDATE_HOURS_FULL
        run_hours = RUN_HOURS_FULL
        start_date = None  # all available data
        mode_label = "FULL (24 update hrs, 24 run hrs, all dates)"

    bt = Backtester(db_path=DB_PATH, city="NYC")

    print("=" * 80)
    print("Phase 3: Quantile Regression — Ablation Analysis")
    print("=" * 80)
    print("  Mode:          {}".format(mode_label))
    print("  Update hours:  {}".format(update_hours))
    print("  Run hours:     {}".format(run_hours))
    print("  Variants:      {}".format(len(VARIANTS)))
    print("  Solver:        scipy.optimize.linprog (HiGHS)")
    print("  CDF tails:     Exponential decay past q05/q95")
    print("  Physical floor: running_max hard-clamp")
    print("  Training:      Cross-hour (all 24 update hours per run_hour)")
    print("  Window:        Rolling 365-day")
    print()

    all_results = {}
    total_start = time.time()

    for i, (vname, model_fn) in enumerate(VARIANTS):
        print("[{}/{}] Running {}...".format(i + 1, len(VARIANTS), vname))

        t0 = time.time()
        result = bt.run(
            model_fn,
            update_hours_et=update_hours,
            run_hours=run_hours,
            start_date=start_date,
        )
        elapsed = time.time() - t0
        all_results[vname] = result
        print("  Done: mean_brier={:.4f}  evals={}  ({:.0f}s)".format(
            result.mean_brier, result.total_evaluations, elapsed
        ))
        print()

    # --- Latest-run mode: simulate production (freshest forecast only) ---
    print("[latest-run] Running qr_multimodel with latest_run=True...")
    t0 = time.time()
    result_lr = bt.run(
        wf_qr_multimodel,
        update_hours_et=update_hours,
        start_date=start_date,
        latest_run=True,
    )
    elapsed = time.time() - t0
    all_results["qr_multimodel_latest"] = result_lr
    print("  Done: mean_brier={:.4f}  evals={}  ({:.0f}s)".format(
        result_lr.mean_brier, result_lr.total_evaluations, elapsed
    ))
    print()

    total_elapsed = time.time() - total_start
    print("All variants complete. Total time: {:.0f}s ({:.1f} min)".format(
        total_elapsed, total_elapsed / 60
    ))

    variant_names = [v[0] for v in VARIANTS] + ["qr_multimodel_latest"]

    # --- Section 1: Per-update-hour Brier ---
    print()
    print("=" * 80)
    print("SECTION 1: Per-Update-Hour Brier Scores")
    print("=" * 80)

    header = "{:>6s}".format("Hr ET")
    for vname in variant_names:
        header += " {:>16s}".format(vname[:16])
    print(header)
    print("-" * len(header))

    for h in update_hours:
        row = "{:>6d}".format(h)
        for vname in variant_names:
            by_update = getattr(all_results[vname], 'by_update_hour', {})
            if h in by_update:
                brier = by_update[h]["mean_brier"]
                row += " {:>16.4f}".format(brier)
            else:
                row += " {:>16s}".format("N/A")
        print(row)

    # --- Section 2: Improvement vs baseline (%) ---
    print()
    print("=" * 80)
    print("SECTION 2: Improvement vs multimodel_full (%) — positive = better")
    print("=" * 80)

    base_by_update = getattr(all_results["multimodel_full"], 'by_update_hour', {})

    header2 = "{:>6s}".format("Hr ET")
    for vname in variant_names:
        if vname == "multimodel_full":
            continue
        header2 += " {:>16s}".format(vname[:16])
    print(header2)
    print("-" * len(header2))

    for h in update_hours:
        row = "{:>6d}".format(h)
        base_b = base_by_update.get(h, {}).get("mean_brier", 0)
        for vname in variant_names:
            if vname == "multimodel_full":
                continue
            by_update = getattr(all_results[vname], 'by_update_hour', {})
            var_b = by_update.get(h, {}).get("mean_brier", 0)
            if base_b > 0:
                pct = (base_b - var_b) / base_b * 100
                row += " {:>+15.2f}%".format(pct)
            else:
                row += " {:>16s}".format("N/A")
        print(row)

    # --- Section 3: Overall summary + gate check ---
    print()
    print("=" * 80)
    print("SECTION 3: Overall Summary + Gate Check")
    print("=" * 80)

    base_brier = all_results["multimodel_full"].mean_brier
    gate_brier = base_brier * 0.98  # 2% improvement threshold
    print()
    print("  Baseline Brier:  {:.4f} (multimodel_full)".format(base_brier))
    print("  Gate threshold:  {:.4f} (2% improvement)".format(gate_brier))
    print()
    print("  {:<20s} {:>8s} {:>9s} {:>10s}".format(
        "Variant", "Brier", "Improv", "Verdict"
    ))
    print("  " + "-" * 50)

    for vname in variant_names:
        brier = all_results[vname].mean_brier
        if vname == "multimodel_full":
            print("  {:<20s} {:>8.4f} {:>9s} {:>10s}".format(
                vname, brier, "---", "BASE"
            ))
        else:
            improv = (base_brier - brier) / base_brier * 100
            if improv > 2.0:
                verdict = "PASS"
            elif improv > 1.0:
                verdict = "DISCUSS"
            elif improv > 0:
                verdict = "MARGINAL"
            else:
                verdict = "KILL"
            print("  {:<20s} {:>8.4f} {:>+8.2f}% {:>10s}".format(
                vname, brier, improv, verdict
            ))

    # --- Section 4: Tradeable window focus (13-16 ET) ---
    print()
    print("=" * 80)
    print("SECTION 4: Tradeable Window (13-16 ET)")
    print("=" * 80)

    tradeable_hours = [h for h in [13, 14, 15, 16] if h in update_hours]
    if tradeable_hours:
        base_trade_avg = 0.0
        print()
        print("  {:<20s} {:>10s} {:>10s}".format("Variant", "Avg Brier", "vs Base"))
        print("  " + "-" * 43)

        for vname in variant_names:
            by_update = getattr(all_results[vname], 'by_update_hour', {})
            trade_briers = [
                by_update[h]["mean_brier"]
                for h in tradeable_hours
                if h in by_update
            ]
            if trade_briers:
                avg = sum(trade_briers) / len(trade_briers)
                if vname == "multimodel_full":
                    base_trade_avg = avg
                    print("  {:<20s} {:>10.4f} {:>10s}".format(vname, avg, "BASE"))
                else:
                    if base_trade_avg > 0:
                        pct = (base_trade_avg - avg) / base_trade_avg * 100
                        print("  {:<20s} {:>10.4f} {:>+9.2f}%".format(vname, avg, pct))
    else:
        print("  (tradeable hours not in sampled set)")

    # --- Section 5: Overnight edge (21-23 ET / 0-3 ET) ---
    print()
    print("=" * 80)
    print("SECTION 5: Overnight Edge (21-23 ET & 0-3 ET)")
    print("=" * 80)

    overnight_hours = [h for h in [21, 22, 23, 0, 1, 2, 3] if h in update_hours]
    if overnight_hours:
        base_overnight_avg = 0.0
        print()
        print("  Hours: {}".format(overnight_hours))
        print()
        print("  {:<20s} {:>10s} {:>10s}".format("Variant", "Avg Brier", "vs Base"))
        print("  " + "-" * 43)

        for vname in variant_names:
            by_update = getattr(all_results[vname], 'by_update_hour', {})
            overnight_briers = [
                by_update[h]["mean_brier"]
                for h in overnight_hours
                if h in by_update
            ]
            if overnight_briers:
                avg = sum(overnight_briers) / len(overnight_briers)
                if vname == "multimodel_full":
                    base_overnight_avg = avg
                    print("  {:<20s} {:>10.4f} {:>10s}".format(vname, avg, "BASE"))
                else:
                    if base_overnight_avg > 0:
                        pct = (base_overnight_avg - avg) / base_overnight_avg * 100
                        print("  {:<20s} {:>10.4f} {:>+9.2f}%".format(vname, avg, pct))
    else:
        print("  (overnight hours not in sampled set)")

    # --- Section 6: Early/Mid day (4-12 ET) — market dominates here ---
    print()
    print("=" * 80)
    print("SECTION 6: Mid-Day (4-12 ET) — Market Dominates")
    print("=" * 80)

    midday_hours = [h for h in range(4, 13) if h in update_hours]
    if midday_hours:
        base_midday_avg = 0.0
        print()
        print("  {:<20s} {:>10s} {:>10s}".format("Variant", "Avg Brier", "vs Base"))
        print("  " + "-" * 43)

        for vname in variant_names:
            by_update = getattr(all_results[vname], 'by_update_hour', {})
            midday_briers = [
                by_update[h]["mean_brier"]
                for h in midday_hours
                if h in by_update
            ]
            if midday_briers:
                avg = sum(midday_briers) / len(midday_briers)
                if vname == "multimodel_full":
                    base_midday_avg = avg
                    print("  {:<20s} {:>10.4f} {:>10s}".format(vname, avg, "BASE"))
                else:
                    if base_midday_avg > 0:
                        pct = (base_midday_avg - avg) / base_midday_avg * 100
                        print("  {:<20s} {:>10.4f} {:>+9.2f}%".format(vname, avg, pct))
    else:
        print("  (mid-day hours not in sampled set)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
