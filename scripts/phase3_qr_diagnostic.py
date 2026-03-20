#!/usr/bin/env python3
"""Phase 3 QR Diagnostic — where is qr_multimodel leaving edge on the table?

Wraps qr_multimodel in a diagnostic harness that captures every prediction,
then analyzes:
1. Running max floor: is impossible bracket mass being eliminated?
2. Spread collapse: does distribution narrow through the day?
3. Bracket concentration: how many brackets hold 90% of mass?
4. Worst misses: what do the biggest Brier failures look like?
5. Late-day running max proximity to settlement
"""

import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, ".")

import numpy as np
from loguru import logger
from zoneinfo import ZoneInfo

from services.backtester import (
    Backtester,
    _ensure_level1,
    _ET,
    wf_qr_multimodel,
)

import duckdb

DB_PATH = "data/alphatemp.duckdb"
RUN_HOURS = [0, 12]
UPDATE_HOURS = [0, 4, 8, 12, 16, 20]

# Collector: accumulates per-prediction diagnostics
_diag_records = []


def _make_diagnostic_wrapper(inner_fn):
    """Wrap a model function to capture predictions + context for diagnostics."""

    def wrapper(provider, ref_time):
        probs = inner_fn(provider, ref_time)
        if probs is None:
            return None

        # Extract context from provider
        con = provider._shared_con
        station_id = provider.station_id
        run_hour = provider.model_run.hour
        current_date = provider.model_run.date()

        ref_utc = ref_time if ref_time.tzinfo else ref_time.replace(tzinfo=timezone.utc)
        update_hour_et = ref_utc.astimezone(_ET).hour
        cutoff_ts = ref_utc.timestamp()

        # Get observations to compute running_max
        l1 = _ensure_level1(con, run_hour, station_id, model_name='hrrr')
        obs_by_date = l1.get("obs_by_date", {})
        day_obs = obs_by_date.get(current_date, [])
        truncated = [(ts, temp) for ts, temp in day_obs if ts <= cutoff_ts]
        running_max = max([t for _, t in truncated]) if len(truncated) >= 2 else None

        _diag_records.append({
            "date": current_date,
            "run_hour": run_hour,
            "update_hour_et": update_hour_et,
            "running_max": running_max,
            "probs": probs,
        })

        return probs

    wrapper.__name__ = inner_fn.__name__
    wrapper.__doc__ = inner_fn.__doc__
    return wrapper


def main():
    global _diag_records
    _diag_records = []

    wrapped = _make_diagnostic_wrapper(wf_qr_multimodel)

    bt = Backtester(db_path=DB_PATH, city="NYC")
    start_date = date.today() - timedelta(days=730)

    print("=" * 80)
    print("Phase 3 QR Diagnostic — qr_multimodel deep dive")
    print("=" * 80)
    print("  Running backtester to capture predictions...")
    print()

    t0 = time.time()
    result = bt.run(
        wrapped,
        update_hours_et=UPDATE_HOURS,
        run_hours=RUN_HOURS,
        start_date=start_date,
    )
    elapsed = time.time() - t0
    print("  Backtest complete: {:.0f}s, {} evals, {} diagnostic records".format(
        elapsed, result.total_evaluations, len(_diag_records)
    ))

    # Get settlement data
    con = duckdb.connect(DB_PATH, read_only=True)
    settlements = con.execute("""
        SELECT obs_date, max_temp_f
        FROM nws_daily
        WHERE station_id = 'KNYC' AND obs_date >= ? AND max_temp_f IS NOT NULL
    """, [start_date]).fetchall()
    settlement_map = {row[0]: row[1] for row in settlements}
    con.close()

    # Aggregate diagnostics per update hour
    stats = defaultdict(lambda: {
        "count": 0,
        "running_max_available": 0,
        "impossible_bracket_mass": [],
        "spread_90": [],
        "brier_scores": [],
        "worst_misses": [],
    })

    for rec in _diag_records:
        uh = rec["update_hour_et"]
        probs = rec["probs"]
        running_max = rec["running_max"]
        obs_date = rec["date"]
        settlement = settlement_map.get(obs_date)
        if settlement is None:
            continue

        s = stats[uh]
        s["count"] += 1

        # 1. Running max floor
        if running_max is not None:
            s["running_max_available"] += 1
            impossible_mass = sum(
                p for k, p in probs.items() if k + 0.5 < running_max
            )
            s["impossible_bracket_mass"].append(impossible_mass)

        # 2. Spread: brackets holding 90% mass
        sorted_probs = sorted(probs.values(), reverse=True)
        cumsum = 0.0
        n_brackets_90 = 0
        for p in sorted_probs:
            cumsum += p
            n_brackets_90 += 1
            if cumsum >= 0.90:
                break
        s["spread_90"].append(n_brackets_90)

        # 3. Brier
        settlement_int = round(settlement)
        brier = sum(
            (p - (1.0 if k == settlement_int else 0.0)) ** 2
            for k, p in probs.items()
        )
        if settlement_int not in probs:
            brier += 1.0
        s["brier_scores"].append(brier)

        peak = max(probs, key=probs.get)
        s["worst_misses"].append((
            brier, obs_date, settlement, peak, running_max,
            settlement - running_max if running_max else None,
        ))

    # --- Reports ---

    print()
    print("=" * 80)
    print("SECTION 1: Running Max Floor — Is It Working?")
    print("=" * 80)
    print()
    print("  Avg Imp Mass = average probability mass assigned to brackets below running_max")
    print("  (should be ~0% if floor is working)")
    print()
    print("  {:>6s} {:>8s} {:>10s} {:>14s}".format(
        "Hr ET", "N", "Has RunMax", "Avg Imp Mass"
    ))
    print("  " + "-" * 42)
    for uh in UPDATE_HOURS:
        s = stats[uh]
        if s["count"] == 0:
            continue
        avg_imp = np.mean(s["impossible_bracket_mass"]) if s["impossible_bracket_mass"] else 0
        print("  {:>6d} {:>8d} {:>10d} {:>13.2f}%".format(
            uh, s["count"], s["running_max_available"], avg_imp * 100
        ))

    print()
    print("=" * 80)
    print("SECTION 2: Spread Collapse — Brackets Holding 90% Mass")
    print("=" * 80)
    print()
    print("  Fewer brackets = tighter distribution = more confident")
    print("  Should decrease from hour 0 → hour 20")
    print()
    print("  {:>6s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "Hr ET", "Mean", "Median", "P25", "P75"
    ))
    print("  " + "-" * 50)
    for uh in UPDATE_HOURS:
        s = stats[uh]
        if not s["spread_90"]:
            continue
        arr = np.array(s["spread_90"])
        print("  {:>6d} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f}".format(
            uh, np.mean(arr), np.median(arr),
            np.percentile(arr, 25), np.percentile(arr, 75)
        ))

    print()
    print("=" * 80)
    print("SECTION 3: Brier Score by Update Hour")
    print("=" * 80)
    print()
    print("  {:>6s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "Hr ET", "Mean", "Median", "P25", "P75"
    ))
    print("  " + "-" * 50)
    for uh in UPDATE_HOURS:
        s = stats[uh]
        if not s["brier_scores"]:
            continue
        arr = np.array(s["brier_scores"])
        print("  {:>6d} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            uh, np.mean(arr), np.median(arr),
            np.percentile(arr, 25), np.percentile(arr, 75)
        ))

    print()
    print("=" * 80)
    print("SECTION 4: Worst 10 Misses (highest Brier)")
    print("=" * 80)
    print()
    all_misses = []
    for uh in UPDATE_HOURS:
        for miss in stats[uh]["worst_misses"]:
            all_misses.append((miss[0], uh, miss[1], miss[2], miss[3], miss[4]))
    all_misses.sort(reverse=True)

    print("  {:>8s} {:>6s} {:>12s} {:>8s} {:>8s} {:>8s}".format(
        "Brier", "Hr ET", "Date", "Settle", "Peak", "RunMax"
    ))
    print("  " + "-" * 55)
    for brier, uh, obs_date, settlement, peak, runmax in all_misses[:10]:
        rm_str = "{:.0f}".format(runmax) if runmax is not None else "N/A"
        print("  {:>8.4f} {:>6d} {:>12s} {:>8.0f} {:>8d} {:>8s}".format(
            brier, uh, str(obs_date), settlement, peak, rm_str
        ))

    print()
    print("=" * 80)
    print("SECTION 5: Late-Day Running Max vs Settlement")
    print("=" * 80)
    print()
    print("  How close is the observed running max to final settlement?")
    print("  Gap = settlement - running_max (positive = high hasn't peaked yet)")
    print()
    for uh in UPDATE_HOURS:
        s = stats[uh]
        gaps = []
        for miss in s["worst_misses"]:
            _, _, settlement, _, runmax, gap = miss
            if gap is not None:
                gaps.append(gap)
        if gaps:
            arr = np.array(gaps)
            print("  Hour {:>2d} ET: mean gap={:>+5.1f}°F, median={:>+5.1f}°F, "
                  "pct<=0={:>5.1f}%, pct<=2={:>5.1f}%".format(
                uh, np.mean(arr), np.median(arr),
                100 * np.mean(arr <= 0), 100 * np.mean(arr <= 2)
            ))

    print()
    print("Done.")


if __name__ == "__main__":
    main()
