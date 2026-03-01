"""Tests for OLS baseline on 24 HRRR run hours."""

import duckdb
import pytest
from datetime import date, datetime, timedelta
from core.db import init_db


def _seed_24h_data(db_path, n_days=120):
    """Seed 120 days x 24 hours of HRRR forecasts + NWS daily."""
    con = duckdb.connect(db_path)
    base = date(2024, 1, 1)
    for d in range(n_days):
        current = base + timedelta(days=d)
        actual_high = 50 + 30 * (d % 365) / 365  # Seasonal pattern
        for hour in range(24):
            model_run = datetime(current.year, current.month, current.day, hour)
            bias = 2.0 + 0.5 * (hour / 24)  # Slight hour-dependent bias
            for fxx in range(1, 19):
                valid_at = model_run + timedelta(hours=fxx)
                temp_f = actual_high + bias + (fxx * 0.1)
                con.execute(
                    "INSERT INTO forecasts "
                    "(station_id, model_run, valid_at, temp_f, temp_c, "
                    "ingested_at, model_name, fxx, is_spinup) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'hrrr', ?, ?)",
                    ['KNYC', model_run, valid_at, temp_f,
                     round((temp_f - 32) * 5 / 9, 2),
                     datetime.now(), fxx, fxx <= 3],
                )
        con.execute(
            "INSERT INTO nws_daily "
            "(station_id, obs_date, max_temp_f, ingested_at) "
            "VALUES ('KNYC', ?, ?, ?)",
            [current, actual_high, datetime.now()],
        )
    con.close()


def test_ols_24h_produces_results(tmp_path):
    """OLS on 24 run hours should produce results for all hours."""
    db_path = str(tmp_path / "test.duckdb")
    init_db(db_path)
    _seed_24h_data(db_path)

    from services.backtester import Backtester, wf_regression_full
    bt = Backtester(db_path=db_path)
    result = bt.run(
        model_fn=wf_regression_full,
        start_date=date(2024, 4, 1),  # After 90-day warm-up
        end_date=date(2024, 4, 30),
        run_hours=list(range(24)),
    )
    assert len(result.run_results) > 0
    assert result.mean_brier < 1.0  # Sanity — not perfect but not broken
    # Should have results for multiple run hours
    hours_seen = {r.run_hour for r in result.run_results}
    assert len(hours_seen) >= 20  # Allow some hours to skip if no data
