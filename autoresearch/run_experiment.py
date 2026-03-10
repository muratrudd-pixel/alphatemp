"""AlphaTemp Autoresearch — Experiment Harness (LOCKED — do not modify)

Runs experiment.py's model_fn through the backtester, computes a composite
score, logs results to results.tsv, and exits with code 0 (improvement)
or code 1 (no improvement).

Usage: PYTHONPATH=. python autoresearch/run_experiment.py
"""

import json
import os
import signal
import sys
import time
from datetime import date, datetime, timedelta

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.backtester import Backtester

# ── Constants ───────────────────────────────────────────────────────────────

TIMEOUT_SECONDS = 90
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.tsv")
BEST_SCORE_FILE = os.path.join(os.path.dirname(__file__), "best_score.txt")
BASELINE_FILE = os.path.join(os.path.dirname(__file__), "baseline.json")

# Quick-mode backtest parameters
EVAL_DAYS = 90  # Evaluate last N days only (keeps experiments under timeout)
RUN_HOURS = [0, 6, 12, 18]
UPDATE_HOURS_ET = [0, 4, 8, 12, 16, 20]

# Composite weights
W_PNL = 0.45
W_BRIER = 0.35
W_HIT = 0.20


# ── Timeout handler ─────────────────────────────────────────────────────────

def _timeout_handler(signum, frame):
    print("TIMEOUT: Experiment exceeded {} seconds".format(TIMEOUT_SECONDS))
    sys.exit(1)


# ── Scoring ─────────────────────────────────────────────────────────────────

def compute_composite(brier, hit_rate, pnl_cents, baseline):
    # type: (float, float, float, dict) -> float
    """Compute composite score (lower = better).

    Normalizes all components to [0, 1] range using baseline stats,
    then weights them.
    """
    # Brier: already lower = better, typically 0.3-1.5
    brier_min = baseline.get("brier_min", 0.3)
    brier_max = baseline.get("brier_max", 1.5)
    brier_norm = (brier - brier_min) / max(brier_max - brier_min, 0.01)
    brier_norm = max(0.0, min(1.0, brier_norm))

    # Hit rate: higher = better, invert so lower = better
    hit_min = baseline.get("hit_min", 0.0)
    hit_max = baseline.get("hit_max", 0.5)
    hit_norm = 1.0 - (hit_rate - hit_min) / max(hit_max - hit_min, 0.01)
    hit_norm = max(0.0, min(1.0, hit_norm))

    # P&L: higher = better (positive = profit), invert so lower = better
    pnl_min = baseline.get("pnl_min", -10000.0)
    pnl_max = baseline.get("pnl_max", 10000.0)
    pnl_norm = 1.0 - (pnl_cents - pnl_min) / max(pnl_max - pnl_min, 0.01)
    pnl_norm = max(0.0, min(1.0, pnl_norm))

    return W_PNL * pnl_norm + W_BRIER * brier_norm + W_HIT * hit_norm


def compute_simple_pnl(results):
    # type: (list) -> float
    """Simplified P&L: bet $1 at 50c on every Kalshi-scored event.

    Hit = profit (100 - 50 - fee), miss = loss (50 + fee).
    Returns total P&L in cents.
    """
    from services.strategy_backtester import compute_taker_fee

    total_pnl = 0.0
    for r in results:
        if not r.used_kalshi_brackets:
            continue
        entry_price = 50.0  # cents
        qty = 1
        fee = compute_taker_fee(entry_price, qty)
        if r.hit:
            pnl = (100.0 - entry_price) * qty - fee
        else:
            pnl = -(entry_price * qty + fee)
        total_pnl += pnl

    return total_pnl


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # Set timeout
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)

    start_time = time.time()

    # Import experiment (force reload to pick up changes)
    try:
        import importlib
        import autoresearch.experiment as _exp_mod
        importlib.reload(_exp_mod)
        from autoresearch.experiment import model_fn, DESCRIPTION, _coeff_cache
        _coeff_cache.clear()
    except ImportError as e:
        print("ERROR: Failed to import experiment.py: {}".format(e))
        sys.exit(1)
    except Exception as e:
        print("ERROR: experiment.py has a syntax/runtime error: {}".format(e))
        sys.exit(1)

    print("Running experiment: {}".format(DESCRIPTION))

    # Run backtest (limited to recent dates for speed)
    bt = Backtester(db_path="data/alphatemp.duckdb", city="NYC")
    eval_start = date.today() - timedelta(days=EVAL_DAYS)
    try:
        result = bt.run(
            model_fn=model_fn,
            start_date=eval_start,
            run_hours=RUN_HOURS,
            update_hours_et=UPDATE_HOURS_ET,
        )
    except Exception as e:
        print("ERROR: Backtest failed: {}".format(e))
        sys.exit(1)

    elapsed = time.time() - start_time

    # Cancel timeout
    if hasattr(signal, 'SIGALRM'):
        signal.alarm(0)

    # Extract metrics
    brier = result.mean_brier
    hit_rate = result.top1_hit_rate
    total_evals = result.total_evaluations
    pnl_cents = compute_simple_pnl(result.run_results)

    print("  Brier: {:.4f}  Hit rate: {:.4f}  P&L: {:.0f}c  Evals: {}  Time: {:.1f}s".format(
        brier, hit_rate, pnl_cents, total_evals, elapsed,
    ))

    # Load or create baseline
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE) as f:
            baseline = json.load(f)
    else:
        # First run: establish baseline from this experiment
        baseline = {
            "brier_min": brier * 0.5,
            "brier_max": brier * 1.5,
            "hit_min": 0.0,
            "hit_max": max(hit_rate * 2.0, 0.3),
            "pnl_min": min(pnl_cents * 2.0, -5000.0),
            "pnl_max": max(pnl_cents * 2.0, 5000.0),
        }
        with open(BASELINE_FILE, 'w') as f:
            json.dump(baseline, f, indent=2)
        print("  Baseline established and saved.")

    composite = compute_composite(brier, hit_rate, pnl_cents, baseline)
    print("SCORE: {:.6f}".format(composite))

    # Load current best
    if os.path.exists(BEST_SCORE_FILE):
        with open(BEST_SCORE_FILE) as f:
            best = float(f.read().strip())
    else:
        best = float('inf')

    # Log to results.tsv
    header_needed = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, 'a') as f:
        if header_needed:
            f.write("timestamp\tcomposite\tbrier\thit_rate\tpnl_cents\tevals\telapsed_s\tdescription\n")
        f.write("{}\t{:.6f}\t{:.4f}\t{:.4f}\t{:.0f}\t{}\t{:.1f}\t{}\n".format(
            datetime.utcnow().isoformat(),
            composite, brier, hit_rate, pnl_cents,
            total_evals, elapsed, DESCRIPTION,
        ))

    # Check for improvement
    if composite < best:
        print("IMPROVEMENT: {:.6f} -> {:.6f} (delta: {:.6f})".format(
            best, composite, best - composite,
        ))
        with open(BEST_SCORE_FILE, 'w') as f:
            f.write("{:.6f}\n".format(composite))
        sys.exit(0)  # Success — agent should commit
    else:
        print("NO IMPROVEMENT: {:.6f} >= best {:.6f}".format(composite, best))
        sys.exit(1)  # Failure — agent should discard


if __name__ == "__main__":
    main()
