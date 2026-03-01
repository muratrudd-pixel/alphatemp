"""Tests for 24-hour RUN_HOURS expansion."""

import duckdb
import pytest
from datetime import date, datetime, timedelta
from core.db import init_db
from services.backtester import Backtester, RUN_HOURS


def test_run_hours_includes_all_24():
    """RUN_HOURS should list all 24 HRRR run hours."""
    assert len(RUN_HOURS) == 24
    assert RUN_HOURS == list(range(24))


def test_backtester_accepts_run_hours_param(tmp_path):
    """Backtester.run() should accept a run_hours parameter."""
    db_path = str(tmp_path / "test.duckdb")
    init_db(db_path)
    bt = Backtester(db_path=db_path)
    # Should not raise — just verify the parameter is accepted
    result = bt.run(
        model_fn=lambda prov, ref: None,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
        run_hours=[0, 12],
    )
    assert result is not None


def test_backtester_filters_by_run_hours(tmp_path):
    """Backtester should only evaluate specified run_hours."""
    db_path = str(tmp_path / "test.duckdb")
    init_db(db_path)

    # Seed forecasts at multiple run hours
    con = duckdb.connect(db_path)
    for hour in [0, 6, 12, 18]:
        model_run = datetime(2024, 6, 15, hour)
        for fxx in range(1, 19):
            valid_at = model_run + timedelta(hours=fxx)
            con.execute(
                "INSERT INTO forecasts (station_id, model_run, valid_at, "
                "temp_f, temp_c, ingested_at, model_name, fxx, is_spinup) "
                "VALUES (?, ?, ?, ?, ?, ?, 'hrrr', ?, ?)",
                ['KNYC', model_run, valid_at, 75.0, 23.9,
                 datetime(2024, 6, 15), fxx, fxx <= 3],
            )
    # Seed NWS daily
    con.execute(
        "INSERT INTO nws_daily (station_id, obs_date, max_temp_f, ingested_at) "
        "VALUES ('KNYC', '2024-06-15', 78.0, '2024-06-16')"
    )
    con.close()

    bt = Backtester(db_path=db_path)
    # Evaluate only 0z and 12z
    result = bt.run(
        model_fn=lambda prov, ref: {k: 1.0 / 31 for k in range(60, 91)},
        start_date=date(2024, 6, 15),
        end_date=date(2024, 6, 15),
        run_hours=[0, 12],
    )
    # Should have results for exactly 2 run hours, not 4
    hours_seen = set()
    for r in result.run_results:
        hours_seen.add(r.run_hour)
    assert hours_seen == {0, 12}
