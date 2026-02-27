#!/usr/bin/env python3
"""Phase 3.6: Neighbor Station Observations -- HRRR Ablation Analysis.

Runs 6 neighbor variants + baseline on HRRR across 0-18 ET.
Compares Brier scores to find which variant helps and when.

Variants:
  A1/A2 — Raw neighbor obs as additional features / blended curve
  B1/B2 — Station-offset-corrected neighbor obs / blended curve
  C1/C2 — Trend-only extraction from neighbor obs / blended curve

Usage:
    cd ~/Projects/alphatemp/alphatemp
    ../venv/bin/python scripts/phase36_neighbor_analysis.py
"""
import sys
import time

sys.path.insert(0, ".")

from datetime import date
from typing import Dict, List, Optional, Tuple

from services.backtester import (
    Backtester,
    wf_phase2b_full,
    wf_phase2b_neighbor_a1,
    wf_phase2b_neighbor_a2,
    wf_phase2b_neighbor_b1,
    wf_phase2b_neighbor_b2,
    wf_phase2b_neighbor_c1,
    wf_phase2b_neighbor_c2,
)

DB_PATH = "data/alphatemp.duckdb"
RUN_HOURS = [0, 6, 12, 18]
UPDATE_HOURS_ET = list(range(0, 19))
START_DATE = date(2021, 6, 1)   # 90 days after earliest KLGA/KEWR obs (Dec 2020)
END_DATE = date(2026, 2, 25)

# Gate thresholds (percentage improvement vs baseline)
GATE_PASS = 2.0    # >2% = PASS
GATE_DISCUSS = 1.0  # 1-2% = DISCUSS, <1% = KILL

VARIANTS = [
    ("baseline", wf_phase2b_full),
    ("A1_raw_feat", wf_phase2b_neighbor_a1),
    ("A2_raw_blend", wf_phase2b_neighbor_a2),
    ("B1_off_feat", wf_phase2b_neighbor_b1),
    ("B2_off_blend", wf_phase2b_neighbor_b2),
    ("C1_trn_feat", wf_phase2b_neighbor_c1),
    ("C2_trn_blend", wf_phase2b_neighbor_c2),
]

VARIANT_NAMES = [name for name, _ in VARIANTS]
NEIGHBOR_NAMES = [name for name, _ in VARIANTS if name != "baseline"]


def _fmt_brier(val):
    # type: (Optional[float]) -> str
    """Format a Brier score or return dashes if None."""
    if val is None:
        return "   ---  "
    return "{:8.4f}".format(val)


def _fmt_pct(val):
    # type: (Optional[float]) -> str
    """Format a percentage improvement or return dashes if None."""
    if val is None:
        return "   ---  "
    return "{:+7.2f}%".format(val)


def _gate_label(pct):
    # type: (Optional[float]) -> str
    """Return PASS / DISCUSS / KILL based on improvement percentage."""
    if pct is None:
        return "N/A"
    if pct > GATE_PASS:
        return "PASS"
    elif pct > GATE_DISCUSS:
        return "DISCUSS"
    else:
        return "KILL"


def print_section1(all_briers):
    # type: (Dict[str, Dict[int, Optional[float]]]) -> None
    """Section 1: Per-update-hour Brier scores."""
    print()
    print("=" * 90)
    print("SECTION 1: Per-Update-Hour Brier Scores")
    print("=" * 90)
    print()

    # Header
    header = "{:>7s}".format("Hr ET")
    for name in VARIANT_NAMES:
        header += " {:>10s}".format(name[:10])
    print(header)
    print("-" * (7 + 11 * len(VARIANT_NAMES)))

    for uhr in UPDATE_HOURS_ET:
        row = "{:>7d}".format(uhr)
        for name in VARIANT_NAMES:
            val = all_briers[name].get(uhr)
            row += " {:>10s}".format(_fmt_brier(val))
        print(row)


def print_section2(all_briers):
    # type: (Dict[str, Dict[int, Optional[float]]]) -> None
    """Section 2: Improvement vs baseline (percentage)."""
    print()
    print("=" * 90)
    print("SECTION 2: Improvement vs Baseline (%)")
    print("  Positive = variant is better (lower Brier)")
    print("=" * 90)
    print()

    header = "{:>7s}".format("Hr ET")
    for name in NEIGHBOR_NAMES:
        header += " {:>10s}".format(name[:10])
    print(header)
    print("-" * (7 + 11 * len(NEIGHBOR_NAMES)))

    for uhr in UPDATE_HOURS_ET:
        row = "{:>7d}".format(uhr)
        base_val = all_briers["baseline"].get(uhr)
        for name in NEIGHBOR_NAMES:
            var_val = all_briers[name].get(uhr)
            if base_val is not None and var_val is not None and base_val > 0:
                pct = (base_val - var_val) / base_val * 100.0
                row += " {:>10s}".format(_fmt_pct(pct))
            else:
                row += " {:>10s}".format(_fmt_pct(None))
        print(row)


def print_section3(all_briers):
    # type: (Dict[str, Dict[int, Optional[float]]]) -> None
    """Section 3: Crossover analysis.

    For each variant, find the earliest hour where it consistently
    beats baseline (variant Brier < baseline Brier for all remaining hours).
    """
    print()
    print("=" * 90)
    print("SECTION 3: Crossover Analysis")
    print("  Earliest hour where variant beats baseline for all remaining hours")
    print("=" * 90)
    print()

    for name in NEIGHBOR_NAMES:
        # Find the earliest hour where variant < baseline for all hours >= that hour
        crossover = None  # type: Optional[int]
        for start_uhr in UPDATE_HOURS_ET:
            beats_all = True
            for uhr in range(start_uhr, 19):
                base_val = all_briers["baseline"].get(uhr)
                var_val = all_briers[name].get(uhr)
                if base_val is None or var_val is None or var_val >= base_val:
                    beats_all = False
                    break
            if beats_all:
                crossover = start_uhr
                break

        if crossover is not None:
            # Also compute the improvement at crossover
            base_at = all_briers["baseline"].get(crossover)
            var_at = all_briers[name].get(crossover)
            if base_at and var_at and base_at > 0:
                pct = (base_at - var_at) / base_at * 100.0
                print("  {:12s}  crossover at {:2d} ET  ({:+.2f}% at crossover)".format(
                    name, crossover, pct))
            else:
                print("  {:12s}  crossover at {:2d} ET".format(name, crossover))
        else:
            # Check if it ever beats baseline at individual hours
            beats_any = []  # type: List[int]
            for uhr in UPDATE_HOURS_ET:
                base_val = all_briers["baseline"].get(uhr)
                var_val = all_briers[name].get(uhr)
                if base_val is not None and var_val is not None and var_val < base_val:
                    beats_any.append(uhr)
            if beats_any:
                print("  {:12s}  no consistent crossover; beats baseline at hours: {}".format(
                    name, ", ".join(str(h) for h in beats_any)))
            else:
                print("  {:12s}  never beats baseline".format(name))


def print_section4(all_briers):
    # type: (Dict[str, Dict[int, Optional[float]]]) -> None
    """Section 4: Gate check.

    For each variant: best improvement % and at which hour.
    PASS if >2%, DISCUSS if 1-2%, KILL if <1%.
    """
    print()
    print("=" * 90)
    print("SECTION 4: Gate Check")
    print("  PASS: >2% improvement  |  DISCUSS: 1-2%  |  KILL: <1%")
    print("=" * 90)
    print()

    print("  {:12s} {:>8s} {:>8s} {:>10s} {:>10s}".format(
        "Variant", "Best %", "At Hr", "18ET %", "Verdict"))
    print("  " + "-" * 52)

    for name in NEIGHBOR_NAMES:
        best_pct = None  # type: Optional[float]
        best_hour = None  # type: Optional[int]

        for uhr in UPDATE_HOURS_ET:
            base_val = all_briers["baseline"].get(uhr)
            var_val = all_briers[name].get(uhr)
            if base_val is not None and var_val is not None and base_val > 0:
                pct = (base_val - var_val) / base_val * 100.0
                if best_pct is None or pct > best_pct:
                    best_pct = pct
                    best_hour = uhr

        # Also get 18 ET improvement
        base_18 = all_briers["baseline"].get(18)
        var_18 = all_briers[name].get(18)
        pct_18 = None  # type: Optional[float]
        if base_18 is not None and var_18 is not None and base_18 > 0:
            pct_18 = (base_18 - var_18) / base_18 * 100.0

        verdict = _gate_label(best_pct)

        print("  {:12s} {:>8s} {:>8s} {:>10s} {:>10s}".format(
            name,
            _fmt_pct(best_pct).strip() if best_pct is not None else "---",
            "{:d} ET".format(best_hour) if best_hour is not None else "---",
            _fmt_pct(pct_18).strip() if pct_18 is not None else "---",
            verdict,
        ))


def main():
    # type: () -> None
    print("=" * 90)
    print("Phase 3.6: Neighbor Station Observations -- HRRR Ablation Analysis")
    print("=" * 90)
    print()
    print("  Date range:    {} to {}".format(START_DATE, END_DATE))
    print("  Run hours:     {}".format(RUN_HOURS))
    print("  Update hours:  0-18 ET")
    print("  Variants:      {} ({} + baseline)".format(len(VARIANTS), len(NEIGHBOR_NAMES)))
    print()

    bt = Backtester(DB_PATH, "NYC")

    # {variant_name: BacktestResult}
    results = {}  # type: Dict[str, object]
    # {variant_name: {update_hour_et: mean_brier}}
    all_briers = {}  # type: Dict[str, Dict[int, Optional[float]]]

    total_t0 = time.time()

    for idx, (name, model_fn) in enumerate(VARIANTS):
        print()
        print("=" * 60)
        print("[{}/{}] Running {}...".format(idx + 1, len(VARIANTS), name))
        print("=" * 60)

        t0 = time.time()
        result = bt.run(
            model_fn,
            start_date=START_DATE,
            end_date=END_DATE,
            run_hours=RUN_HOURS,
            update_hours_et=UPDATE_HOURS_ET,
        )
        elapsed = time.time() - t0
        results[name] = result

        print("  Done: mean_brier={:.4f}  evals={}  skipped={}  ({:.0f}s)".format(
            result.mean_brier,
            result.total_evaluations,
            result.skipped,
            elapsed,
        ))

        # Extract per-update-hour Brier scores
        by_uhr = getattr(result, "by_update_hour", {})
        brier_by_uhr = {}  # type: Dict[int, Optional[float]]
        for uhr in UPDATE_HOURS_ET:
            entry = by_uhr.get(uhr)
            if entry is not None:
                brier_by_uhr[uhr] = entry["mean_brier"]
            else:
                brier_by_uhr[uhr] = None
        all_briers[name] = brier_by_uhr

    total_elapsed = time.time() - total_t0
    print()
    print("All variants complete. Total time: {:.0f}s ({:.1f} min)".format(
        total_elapsed, total_elapsed / 60.0))

    # -----------------------------------------------------------------------
    # Reports
    # -----------------------------------------------------------------------
    print_section1(all_briers)
    print_section2(all_briers)
    print_section3(all_briers)
    print_section4(all_briers)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 90)
    print("OVERALL SUMMARY")
    print("=" * 90)
    print()
    print("  {:12s} {:>10s} {:>10s} {:>10s}".format(
        "Variant", "Mean Brier", "Hit Rate", "Evals"))
    print("  " + "-" * 44)
    for name in VARIANT_NAMES:
        r = results[name]
        print("  {:12s} {:>10.4f} {:>9.1f}% {:>10d}".format(
            name,
            r.mean_brier,
            r.top1_hit_rate * 100.0,
            r.total_evaluations,
        ))

    print()
    print("Done.")


if __name__ == "__main__":
    main()
