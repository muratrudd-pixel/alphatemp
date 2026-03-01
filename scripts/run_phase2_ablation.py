"""Run Phase 2 bias correction ablation across all candidates.

Usage:
    python scripts/run_phase2_ablation.py --candidate ols
    python scripts/run_phase2_ablation.py --candidate ols --run-hours 0,6,12,18
    python scripts/run_phase2_ablation.py --candidate all
"""

import argparse
import json
from datetime import date
from typing import List, Optional

from loguru import logger

from core.db import DEFAULT_DB_PATH
from services.backtester import Backtester


def run_ablation(
    candidate,  # type: str
    db_path=DEFAULT_DB_PATH,  # type: str
    run_hours=None,  # type: Optional[List[int]]
    start_date=None,  # type: Optional[date]
    end_date=None,  # type: Optional[date]
):
    """Run a single ablation candidate and print results."""
    bt = Backtester(db_path=db_path)

    # Import the appropriate model function
    if candidate == "ols":
        from services.backtester import wf_regression_full as model_fn
    elif candidate == "ols_no_spinup":
        from services.backtester import wf_regression_full_no_spinup as model_fn
    elif candidate == "emos":
        from services.phase2_emos import emos_model_fn as model_fn
    elif candidate == "xgboost":
        from services.phase2_xgboost import xgboost_model_fn as model_fn
    elif candidate == "cross_hour":
        from services.backtester import wf_regression_cross_hour as model_fn
    else:
        raise ValueError("Unknown candidate: {}".format(candidate))

    result = bt.run(
        model_fn=model_fn,
        start_date=start_date or date(2023, 1, 1),
        end_date=end_date or date(2026, 2, 1),
        run_hours=run_hours,
    )

    logger.info("Candidate: {}", candidate)
    logger.info("Run hours: {}", run_hours or "all 24")
    logger.info("Mean Brier: {:.4f}", result.mean_brier)
    logger.info("Top-1 hit: {:.1%}", result.top1_hit_rate)
    logger.info("N results: {}", len(result.run_results))

    # Per-hour breakdown
    logger.info("Per-hour Brier:")
    for hour, metrics in sorted(result.by_run_hour.items()):
        logger.info("  {:02d}z: {:.4f} (n={})", hour, metrics['mean_brier'], metrics['count'])

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 2 ablation runner")
    parser.add_argument("--candidate", required=True,
                        choices=["ols", "ols_no_spinup", "emos", "xgboost",
                                 "cross_hour", "all"])
    parser.add_argument("--run-hours", default=None,
                        help="Comma-separated run hours (default: all 24)")
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    run_hours = None
    if args.run_hours:
        run_hours = [int(h) for h in args.run_hours.split(",")]

    if args.candidate == "all":
        candidates = ["ols", "ols_no_spinup", "emos", "xgboost", "cross_hour"]
    else:
        candidates = [args.candidate]

    for cand in candidates:
        try:
            run_ablation(cand, args.db, run_hours, args.start, args.end)
        except Exception as e:
            logger.error("Candidate {} failed: {}", cand, e)


if __name__ == "__main__":
    main()
