"""Tests for scripts/backfill_merge.py merge_table function."""

import os
import sys
from datetime import datetime

import duckdb
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.db import init_db
from scripts.backfill_merge import merge_table


# -- helpers --

def _insert_forecast_row(con, station_id, model_run, valid_at, model_name="hrrr",
                         temp_f=72.0, temp_c=22.2, fxx=1, is_spinup=False):
    # type: (...) -> None
    """Insert a single forecast row with all 9 columns."""
    con.execute(
        "INSERT INTO forecasts "
        "(station_id, model_run, valid_at, temp_f, temp_c, ingested_at, model_name, fxx, is_spinup) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [station_id, model_run, valid_at, temp_f, temp_c, datetime.utcnow(),
         model_name, fxx, is_spinup],
    )


def _insert_observation_row(con, station_id, observed_at, temp_f=55.0):
    # type: (...) -> None
    """Insert a single observation row."""
    con.execute(
        "INSERT INTO observations "
        "(station_id, observed_at, temp_f, temp_c_tenth, raw_metar, ingested_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [station_id, observed_at, temp_f, (temp_f - 32) * 5 / 9,
         None, datetime.utcnow()],
    )


def _count(db_path, table):
    # type: (str, str) -> int
    con = duckdb.connect(db_path, read_only=True)
    try:
        return con.execute("SELECT COUNT(*) FROM {}".format(table)).fetchone()[0]
    finally:
        con.close()


# -- fixtures --

@pytest.fixture
def main_db(tmp_path):
    """Create and initialise a main DB."""
    path = str(tmp_path / "main.duckdb")
    init_db(path)
    return path


@pytest.fixture
def temp_db(tmp_path):
    """Create and initialise a temp (backfill) DB."""
    path = str(tmp_path / "temp.duckdb")
    init_db(path)
    return path


# -- tests --

class TestMergeForecasts:
    """Tests for merging the forecasts table."""

    def test_merge_inserts_new_rows(self, main_db, temp_db):
        """Rows in temp DB appear in main DB after merge."""
        con = duckdb.connect(temp_db)
        try:
            _insert_forecast_row(
                con, "KNYC",
                datetime(2024, 6, 1, 12, 0),
                datetime(2024, 6, 1, 13, 0),
            )
            _insert_forecast_row(
                con, "KNYC",
                datetime(2024, 6, 1, 12, 0),
                datetime(2024, 6, 1, 14, 0),
            )
            _insert_forecast_row(
                con, "KNYC",
                datetime(2024, 6, 1, 12, 0),
                datetime(2024, 6, 1, 15, 0),
            )
        finally:
            con.close()

        inserted = merge_table(main_db, temp_db, table="forecasts")

        assert inserted == 3
        assert _count(main_db, "forecasts") == 3

    def test_duplicates_are_skipped(self, main_db, temp_db):
        """Merging the same rows twice does not create duplicates."""
        con = duckdb.connect(temp_db)
        try:
            _insert_forecast_row(
                con, "KNYC",
                datetime(2024, 7, 1, 0, 0),
                datetime(2024, 7, 1, 1, 0),
            )
            _insert_forecast_row(
                con, "KNYC",
                datetime(2024, 7, 1, 0, 0),
                datetime(2024, 7, 1, 2, 0),
            )
        finally:
            con.close()

        first = merge_table(main_db, temp_db, table="forecasts")
        second = merge_table(main_db, temp_db, table="forecasts")

        assert first == 2
        assert second == 0
        assert _count(main_db, "forecasts") == 2

    def test_merge_mixed_new_and_existing(self, main_db, temp_db):
        """Only truly new rows are inserted when some already exist."""
        # Pre-populate main with one row
        con = duckdb.connect(main_db)
        try:
            _insert_forecast_row(
                con, "KNYC",
                datetime(2024, 8, 1, 6, 0),
                datetime(2024, 8, 1, 7, 0),
                model_name="gfs",
            )
        finally:
            con.close()

        # Temp has the same row plus a new one
        con = duckdb.connect(temp_db)
        try:
            _insert_forecast_row(
                con, "KNYC",
                datetime(2024, 8, 1, 6, 0),
                datetime(2024, 8, 1, 7, 0),
                model_name="gfs",
            )
            _insert_forecast_row(
                con, "KNYC",
                datetime(2024, 8, 1, 6, 0),
                datetime(2024, 8, 1, 8, 0),
                model_name="gfs",
            )
        finally:
            con.close()

        inserted = merge_table(main_db, temp_db, table="forecasts")

        assert inserted == 1
        assert _count(main_db, "forecasts") == 2

    def test_different_model_names_not_duplicates(self, main_db, temp_db):
        """Same station/run/valid_at but different model_name = distinct rows."""
        con = duckdb.connect(main_db)
        try:
            _insert_forecast_row(
                con, "KNYC",
                datetime(2024, 9, 1, 0, 0),
                datetime(2024, 9, 1, 1, 0),
                model_name="hrrr",
            )
        finally:
            con.close()

        con = duckdb.connect(temp_db)
        try:
            _insert_forecast_row(
                con, "KNYC",
                datetime(2024, 9, 1, 0, 0),
                datetime(2024, 9, 1, 1, 0),
                model_name="gfs",
            )
        finally:
            con.close()

        inserted = merge_table(main_db, temp_db, table="forecasts")

        assert inserted == 1
        assert _count(main_db, "forecasts") == 2


class TestMergeObservations:
    """Tests for merging the observations table."""

    def test_merge_observations(self, main_db, temp_db):
        """Observations merge correctly into main DB."""
        con = duckdb.connect(temp_db)
        try:
            _insert_observation_row(con, "KNYC", datetime(2024, 6, 1, 12, 0))
            _insert_observation_row(con, "KNYC", datetime(2024, 6, 1, 13, 0))
        finally:
            con.close()

        inserted = merge_table(main_db, temp_db, table="observations")

        assert inserted == 2
        assert _count(main_db, "observations") == 2

    def test_observations_duplicates_skipped(self, main_db, temp_db):
        """Duplicate observation rows are not re-inserted."""
        con = duckdb.connect(temp_db)
        try:
            _insert_observation_row(con, "KJFK", datetime(2024, 6, 15, 18, 0))
        finally:
            con.close()

        first = merge_table(main_db, temp_db, table="observations")
        second = merge_table(main_db, temp_db, table="observations")

        assert first == 1
        assert second == 0
        assert _count(main_db, "observations") == 1

    def test_observations_different_stations_not_duplicates(self, main_db, temp_db):
        """Same timestamp at different stations = distinct rows."""
        ts = datetime(2024, 10, 1, 12, 0)

        con = duckdb.connect(main_db)
        try:
            _insert_observation_row(con, "KNYC", ts)
        finally:
            con.close()

        con = duckdb.connect(temp_db)
        try:
            _insert_observation_row(con, "KJFK", ts)
        finally:
            con.close()

        inserted = merge_table(main_db, temp_db, table="observations")

        assert inserted == 1
        assert _count(main_db, "observations") == 2
