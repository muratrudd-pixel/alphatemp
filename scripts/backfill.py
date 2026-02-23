#!/usr/bin/env python3
"""90-day backfill pipeline — observations, forecasts, bias model, and NWS daily.

Usage:
    python scripts/backfill.py --days 90
    python scripts/backfill.py --days 90 --obs-only
    python scripts/backfill.py --days 90 --fcst-only
    python scripts/backfill.py --days 90 --bias-only
    python scripts/backfill.py --days 90 --nws-only
    python scripts/backfill.py --days 90 --dry-run
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from loguru import logger

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.constants import CITIES, STATION_COORDS
from core.db import init_db, get_connection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaTemp 90-day backfill pipeline")
    parser.add_argument("--days", type=int, default=90, help="Days to backfill (default: 90)")
    parser.add_argument("--obs-only", action="store_true", help="Run observation backfill only")
    parser.add_argument("--fcst-only", action="store_true", help="Run forecast backfill only")
    parser.add_argument("--bias-only", action="store_true", help="Run bias model only")
    parser.add_argument("--nws-only", action="store_true", help="Run NWS daily backfill only")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without hitting APIs")
    parser.add_argument("--db-path", default="data/alphatemp.duckdb", help="Database path")
    return parser.parse_args()


def run_observations(token: str, days: int, db_path: str) -> int:
    from services.backfiller import backfill
    logger.info("=" * 60)
    logger.info("STEP 1: Observation backfill (Synoptic API)")
    logger.info("=" * 60)
    return backfill(token=token, days_back=days, db_path=db_path)


def run_forecasts(days: int, db_path: str) -> int:
    from services.forecast_backfiller import backfill_forecasts
    logger.info("=" * 60)
    logger.info("STEP 2: Forecast backfill (HRRR via Herbie)")
    logger.info("=" * 60)
    return backfill_forecasts(days_back=days, db_path=db_path)


def run_nws_daily(days: int, db_path: str) -> int:
    import asyncio
    from services.nws_fetcher import NWSFetcher
    logger.info("=" * 60)
    logger.info("STEP 4: NWS daily backfill (ACIS)")
    logger.info("=" * 60)
    fetcher = NWSFetcher(db_path=db_path)
    return asyncio.run(fetcher.backfill(days_back=days))


def run_bias_model(db_path: str) -> int:
    from services.bias_model import BiasModel
    logger.info("=" * 60)
    logger.info("STEP 3: Bias model computation")
    logger.info("=" * 60)
    model = BiasModel(db_path=db_path)
    return model.compute_and_store()


def print_quality_report(db_path: str, days: int) -> None:
    logger.info("=" * 60)
    logger.info("DATA QUALITY REPORT")
    logger.info("=" * 60)

    con = get_connection(db_path)
    stations = list(STATION_COORDS.keys())

    # Observation coverage
    logger.info("Observation coverage:")
    for stid in stations:
        row = con.execute(
            "SELECT COUNT(DISTINCT CAST(observed_at AS DATE)) FROM observations WHERE station_id = ?",
            [stid],
        ).fetchone()
        obs_days = row[0] if row else 0
        logger.info(f"  {stid}: {obs_days} days with data")

    # Forecast coverage
    logger.info("Forecast coverage:")
    for stid in stations:
        row = con.execute(
            "SELECT COUNT(DISTINCT model_run) FROM forecasts WHERE station_id = ?",
            [stid],
        ).fetchone()
        fcst_runs = row[0] if row else 0
        logger.info(f"  {stid}: {fcst_runs} 12z runs")

    # Paired days (both forecast + observation)
    logger.info("Paired days (forecast + observation):")
    for stid in stations:
        row = con.execute(
            """SELECT COUNT(*) FROM (
                SELECT DISTINCT CAST(model_run AS DATE) AS d FROM forecasts WHERE station_id = ?
                INTERSECT
                SELECT DISTINCT CAST(observed_at AS DATE) AS d FROM observations WHERE station_id = ?
            )""",
            [stid, stid],
        ).fetchone()
        paired = row[0] if row else 0
        logger.info(f"  {stid}: {paired} paired days")

    # NWS daily coverage
    logger.info("NWS daily coverage:")
    for stid in stations:
        row = con.execute(
            "SELECT COUNT(*) FROM nws_daily WHERE station_id = ?",
            [stid],
        ).fetchone()
        nws_days = row[0] if row else 0
        logger.info(f"  {stid}: {nws_days} days with NWS data")

    # Bias model results
    logger.info("Bias model results:")
    for stid in stations:
        row = con.execute(
            """SELECT mean_bias, std_error, sample_days FROM station_bias
               WHERE station_id = ? ORDER BY calculated_at DESC LIMIT 1""",
            [stid],
        ).fetchone()
        if row and row[0] is not None:
            logger.info(
                f"  {stid}: mean_bias={row[0]:+.2f}F, "
                f"std_error={row[1]:.2f}F, sample_days={row[2]}"
            )
        else:
            logger.info(f"  {stid}: no bias data")

    con.close()


def print_dry_run(days: int) -> None:
    stations = [cfg["settlement"] for cfg in CITIES.values()]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    logger.info("DRY RUN — no API calls will be made")
    logger.info(f"Date range: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    logger.info(f"Stations: {', '.join(stations)}")

    import math
    from services.backfiller import CHUNK_DAYS
    obs_chunks = math.ceil(days / CHUNK_DAYS)
    logger.info(f"Observations: {obs_chunks} Synoptic API calls (~{obs_chunks * 0.5:.0f}s)")
    logger.info(f"Forecasts: {days} Herbie calls (~{days * 2:.0f}s)")
    logger.info("Bias model: pure SQL aggregation (seconds)")


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.dry_run:
        print_dry_run(args.days)
        return

    run_all = not (args.obs_only or args.fcst_only or args.bias_only or args.nws_only)

    # Ensure DB exists
    init_db(args.db_path)

    token = os.environ.get("SYNOPTIC_TOKEN")
    if (run_all or args.obs_only) and not token:
        logger.error("SYNOPTIC_TOKEN not set in environment or .env")
        sys.exit(1)

    obs_rows = 0
    fcst_rows = 0
    bias_count = 0
    nws_rows = 0

    if run_all or args.obs_only:
        obs_rows = run_observations(token, args.days, args.db_path)

    if run_all or args.fcst_only:
        fcst_rows = run_forecasts(args.days, args.db_path)

    if run_all or args.bias_only:
        bias_count = run_bias_model(args.db_path)

    if run_all or args.nws_only:
        nws_rows = run_nws_daily(args.days, args.db_path)

    # Quality report
    print_quality_report(args.db_path, args.days)

    logger.info("=" * 60)
    logger.info(
        f"Pipeline complete: {obs_rows} obs rows, "
        f"{fcst_rows} fcst rows, {bias_count} bias entries, "
        f"{nws_rows} NWS daily rows"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
