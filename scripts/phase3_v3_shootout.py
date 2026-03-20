#!/usr/bin/env python3
"""Phase 3: v3 Shootout — Slim ablation to decide if cumulative_divergence earns its keep.

Compares only the variants needed for the v2 vs v3 decision:
  - qr_multimodel_v2: current best QR (10 features)
  - qr_multimodel_v3: v2 + cumulative_divergence (11 features)
  - qr_mm_v3_latest:  v3 with latest-run mode (production config)

Uses single run_hour [12] to cut LP solves in half.
Quick mode: 6 update hours, last 2 years.

Usage:
  PYTHONPATH=. python scripts/phase3_v3_shootout.py
"""

import json
import os
import sys
import time
from datetime import date, timedelta

sys.path.insert(0, ".")

from services.backtester import (
    Backtester,
    wf_qr_multimodel_v2,
    wf_qr_multimodel_v3,
)

DB_PATH = "data/alphatemp.duckdb"
UPDATE_HOURS = [0, 4, 8, 12, 16, 20]
RUN_HOURS = [12]
START_DATE = date.today() - timedelta(days=730)

RESULTS_DIR = "data/qr_results"


def _save_result(vname, result, elapsed, results_dir):
    entry = {
        "variant": vname,
        "mean_brier": result.mean_brier,
        "top1_hit_rate": result.top1_hit_rate,
        "total_evaluations": result.total_evaluations,
        "total_days": result.total_days,
        "skipped": result.skipped,
        "elapsed_seconds": round(elapsed, 1),
        "by_update_hour": getattr(result, 'by_update_hour', {}),
        "by_run_hour": result.by_run_hour,
        "timestamp": date.today().isoformat(),
    }
    path = os.path.join(results_dir, "{}.json".format(vname))
    with open(path, "w") as f:
        json.dump(entry, f, indent=2)
    print("  Saved: {}".format(path))


def main():
    results_dir = os.path.join(RESULTS_DIR, "{}_v3_shootout".format(
        date.today().isoformat()
    ))
    os.makedirs(results_dir, exist_ok=True)

    bt = Backtester(db_path=DB_PATH, city="NYC")

    print("=" * 70)
    print("Phase 3: v3 Shootout — cumulative_divergence decision")
    print("=" * 70)
    print("  Update hours:  {}".format(UPDATE_HOURS))
    print("  Run hours:     {}".format(RUN_HOURS))
    print("  Start date:    {}".format(START_DATE))
    print("  Variants:      3 (v2, v3, v3+latest)")
    print()

    # Known results from earlier this session (same --quick params)
    print("  Prior results (verified this session):")
    print("    multimodel_full: 0.6275")
    print("    qr_base:         0.7180")
    print("    qr_full:         0.6960")
    print()

    all_results = {}
    total_start = time.time()

    # --- Variant 1: v2 (the bar) ---
    print("[1/3] Running qr_multimodel_v2...")
    t0 = time.time()
    result = bt.run(
        wf_qr_multimodel_v2,
        update_hours_et=UPDATE_HOURS,
        run_hours=RUN_HOURS,
        start_date=START_DATE,
    )
    elapsed = time.time() - t0
    all_results["qr_multimodel_v2"] = result
    print("  Done: mean_brier={:.4f}  evals={}  ({:.0f}s)".format(
        result.mean_brier, result.total_evaluations, elapsed
    ))
    _save_result("qr_multimodel_v2", result, elapsed, results_dir)
    print()

    # --- Variant 2: v3 (v2 + cumulative_divergence) ---
    print("[2/3] Running qr_multimodel_v3...")
    t0 = time.time()
    result = bt.run(
        wf_qr_multimodel_v3,
        update_hours_et=UPDATE_HOURS,
        run_hours=RUN_HOURS,
        start_date=START_DATE,
    )
    elapsed = time.time() - t0
    all_results["qr_multimodel_v3"] = result
    print("  Done: mean_brier={:.4f}  evals={}  ({:.0f}s)".format(
        result.mean_brier, result.total_evaluations, elapsed
    ))
    _save_result("qr_multimodel_v3", result, elapsed, results_dir)
    print()

    # --- Variant 3: v3 + latest-run (production mode) ---
    print("[3/3] Running qr_multimodel_v3 with latest_run=True...")
    t0 = time.time()
    result = bt.run(
        wf_qr_multimodel_v3,
        update_hours_et=UPDATE_HOURS,
        start_date=START_DATE,
        latest_run=True,
    )
    elapsed = time.time() - t0
    all_results["qr_mm_v3_latest"] = result
    print("  Done: mean_brier={:.4f}  evals={}  ({:.0f}s)".format(
        result.mean_brier, result.total_evaluations, elapsed
    ))
    _save_result("qr_mm_v3_latest", result, elapsed, results_dir)
    print()

    total_elapsed = time.time() - total_start
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    base_brier = 0.6275  # multimodel_full from earlier this session

    print("  {:<22s} {:>8s} {:>10s} {:>8s}".format(
        "Variant", "Brier", "vs base", "Verdict"
    ))
    print("  " + "-" * 52)
    print("  {:<22s} {:>8.4f} {:>10s} {:>8s}".format(
        "multimodel_full", base_brier, "---", "BASE"
    ))

    for vname in ["qr_multimodel_v2", "qr_multimodel_v3", "qr_mm_v3_latest"]:
        r = all_results[vname]
        improv = (base_brier - r.mean_brier) / base_brier * 100
        if improv > 2.0:
            verdict = "PASS"
        elif improv > 0:
            verdict = "MARGINAL"
        else:
            verdict = "KILL"
        print("  {:<22s} {:>8.4f} {:>+9.2f}% {:>8s}".format(
            vname, r.mean_brier, improv, verdict
        ))

    # v2 vs v3 head-to-head
    v2_brier = all_results["qr_multimodel_v2"].mean_brier
    v3_brier = all_results["qr_multimodel_v3"].mean_brier
    delta = (v2_brier - v3_brier) / v2_brier * 100
    print()
    print("  v2 vs v3 head-to-head: {:+.3f}% (positive = v3 wins)".format(delta))

    # Per-hour comparison
    print()
    print("  {:<6s} {:>12s} {:>12s} {:>12s}".format(
        "Hr ET", "v2", "v3", "v3_latest"
    ))
    print("  " + "-" * 44)
    for h in UPDATE_HOURS:
        row = "  {:>4d}  ".format(h)
        for vname in ["qr_multimodel_v2", "qr_multimodel_v3", "qr_mm_v3_latest"]:
            by_update = getattr(all_results[vname], 'by_update_hour', {})
            if h in by_update:
                row += " {:>11.4f} ".format(by_update[h]["mean_brier"])
            else:
                row += " {:>11s} ".format("N/A")
        print(row)

    print()
    print("Total time: {:.0f}s ({:.1f} min)".format(
        total_elapsed, total_elapsed / 60
    ))
    print()
    if delta > 0:
        print("RECOMMENDATION: v3 wins. Ship it to EC2 for full run.")
    else:
        print("RECOMMENDATION: v3 adds noise. Kill cumulative_divergence.")


if __name__ == "__main__":
    main()
