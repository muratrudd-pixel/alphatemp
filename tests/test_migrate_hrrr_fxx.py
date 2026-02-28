"""Tests for scripts/migrate_hrrr_fxx.py — backfill fxx and is_spinup."""

import os
from datetime import datetime, timedelta

import duckdb
import pytest

from core.db import init_db
from scripts.migrate_hrrr_fxx import migrate


@pytest.fixture
def db_path(tmp_path):
    """Create a fresh test DB with schema initialized."""
    path = str(tmp_path / "test.duckdb")
    init_db(path)
    return path


def _insert_forecast(con, station_id, model_run, valid_at, model_name="hrrr",
                     fxx=None, is_spinup=None, temp_f=45.0, temp_c=7.2):
    """Insert a forecast row with explicit fxx/is_spinup (can be None)."""
    con.execute(
        """
        INSERT INTO forecasts
            (station_id, model_run, valid_at, temp_f, temp_c, ingested_at,
             model_name, fxx, is_spinup)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [station_id, model_run, valid_at, temp_f, temp_c,
         datetime.utcnow(), model_name, fxx, is_spinup],
    )


class TestMigrateHrrrFxx:
    """Tests for the fxx/is_spinup backfill migration."""

    def test_basic_backfill(self, db_path):
        """Insert HRRR rows with NULL fxx, run migrate, verify fxx and is_spinup."""
        con = duckdb.connect(db_path)
        model_run = datetime(2026, 2, 27, 12, 0, 0)

        # fxx=1h and fxx=6h
        _insert_forecast(con, "KNYC", model_run, model_run + timedelta(hours=1))
        _insert_forecast(con, "KNYC", model_run, model_run + timedelta(hours=6))
        con.close()

        updated = migrate(db_path)
        assert updated == 2

        con = duckdb.connect(db_path, read_only=True)
        rows = con.execute(
            "SELECT fxx, is_spinup FROM forecasts ORDER BY valid_at"
        ).fetchall()
        con.close()

        assert rows[0] == (1, True)   # 1h -> spinup
        assert rows[1] == (6, False)  # 6h -> not spinup

    def test_spinup_boundary_fxx_3(self, db_path):
        """fxx=3 should be is_spinup=True (boundary case: <= 3)."""
        con = duckdb.connect(db_path)
        model_run = datetime(2026, 2, 27, 12, 0, 0)
        _insert_forecast(con, "KNYC", model_run, model_run + timedelta(hours=3))
        con.close()

        migrate(db_path)

        con = duckdb.connect(db_path, read_only=True)
        row = con.execute("SELECT fxx, is_spinup FROM forecasts").fetchone()
        con.close()

        assert row[0] == 3
        assert row[1] is True

    def test_fxx_1_is_spinup(self, db_path):
        """fxx=1 (valid_at = model_run + 1h) gets is_spinup=True."""
        con = duckdb.connect(db_path)
        model_run = datetime(2026, 2, 27, 6, 0, 0)
        _insert_forecast(con, "KNYC", model_run, model_run + timedelta(hours=1))
        con.close()

        migrate(db_path)

        con = duckdb.connect(db_path, read_only=True)
        row = con.execute("SELECT fxx, is_spinup FROM forecasts").fetchone()
        con.close()

        assert row[0] == 1
        assert row[1] is True

    def test_fxx_4_not_spinup(self, db_path):
        """fxx=4 (valid_at = model_run + 4h) gets is_spinup=False."""
        con = duckdb.connect(db_path)
        model_run = datetime(2026, 2, 27, 6, 0, 0)
        _insert_forecast(con, "KNYC", model_run, model_run + timedelta(hours=4))
        con.close()

        migrate(db_path)

        con = duckdb.connect(db_path, read_only=True)
        row = con.execute("SELECT fxx, is_spinup FROM forecasts").fetchone()
        con.close()

        assert row[0] == 4
        assert row[1] is False

    def test_idempotent(self, db_path):
        """Running migrate twice produces the same results."""
        con = duckdb.connect(db_path)
        model_run = datetime(2026, 2, 27, 12, 0, 0)
        _insert_forecast(con, "KNYC", model_run, model_run + timedelta(hours=2))
        _insert_forecast(con, "KNYC", model_run, model_run + timedelta(hours=8))
        con.close()

        first_run = migrate(db_path)
        assert first_run == 2

        # Second run — nothing to do
        second_run = migrate(db_path)
        assert second_run == 0

        # Values unchanged
        con = duckdb.connect(db_path, read_only=True)
        rows = con.execute(
            "SELECT fxx, is_spinup FROM forecasts ORDER BY valid_at"
        ).fetchall()
        con.close()

        assert rows[0] == (2, True)
        assert rows[1] == (8, False)

    def test_non_hrrr_rows_untouched(self, db_path):
        """GFS rows with NULL fxx should NOT be updated by the migration."""
        con = duckdb.connect(db_path)
        model_run = datetime(2026, 2, 27, 12, 0, 0)

        # One HRRR row, one GFS row — both with NULL fxx
        _insert_forecast(con, "KNYC", model_run, model_run + timedelta(hours=2),
                         model_name="hrrr")
        _insert_forecast(con, "KNYC", model_run, model_run + timedelta(hours=2),
                         model_name="gfs")
        con.close()

        updated = migrate(db_path)
        assert updated == 1  # Only the HRRR row

        con = duckdb.connect(db_path, read_only=True)
        gfs_row = con.execute(
            "SELECT fxx, is_spinup FROM forecasts WHERE model_name = 'gfs'"
        ).fetchone()
        hrrr_row = con.execute(
            "SELECT fxx, is_spinup FROM forecasts WHERE model_name = 'hrrr'"
        ).fetchone()
        con.close()

        # GFS row should still have NULLs
        assert gfs_row[0] is None
        assert gfs_row[1] is None

        # HRRR row should be populated
        assert hrrr_row[0] == 2
        assert hrrr_row[1] is True

    def test_no_rows_returns_zero(self, db_path):
        """Empty table: migrate returns 0 without error."""
        result = migrate(db_path)
        assert result == 0

    def test_already_populated_rows_skipped(self, db_path):
        """Rows that already have fxx set should not be re-processed."""
        con = duckdb.connect(db_path)
        model_run = datetime(2026, 2, 27, 12, 0, 0)

        # Pre-populated row
        _insert_forecast(con, "KNYC", model_run, model_run + timedelta(hours=5),
                         fxx=5, is_spinup=False)
        # NULL row
        _insert_forecast(con, "KNYC", model_run, model_run + timedelta(hours=10))
        con.close()

        updated = migrate(db_path)
        # Updated count includes all rows with fxx IS NOT NULL (both the pre-populated
        # and the newly backfilled)
        assert updated == 2

        con = duckdb.connect(db_path, read_only=True)
        rows = con.execute(
            "SELECT fxx, is_spinup FROM forecasts ORDER BY valid_at"
        ).fetchall()
        con.close()

        assert rows[0] == (5, False)   # Pre-populated — unchanged
        assert rows[1] == (10, False)  # Newly backfilled

    def test_multiple_stations(self, db_path):
        """Migration works across multiple stations."""
        con = duckdb.connect(db_path)
        model_run = datetime(2026, 2, 27, 0, 0, 0)

        _insert_forecast(con, "KNYC", model_run, model_run + timedelta(hours=1))
        _insert_forecast(con, "KLGA", model_run, model_run + timedelta(hours=12))
        _insert_forecast(con, "KEWR", model_run, model_run + timedelta(hours=18))
        con.close()

        updated = migrate(db_path)
        assert updated == 3

        con = duckdb.connect(db_path, read_only=True)
        rows = con.execute(
            "SELECT station_id, fxx, is_spinup FROM forecasts ORDER BY station_id"
        ).fetchall()
        con.close()

        # KEWR: 18h
        assert rows[0] == ("KEWR", 18, False)
        # KLGA: 12h
        assert rows[1] == ("KLGA", 12, False)
        # KNYC: 1h
        assert rows[2] == ("KNYC", 1, True)
