"""DataProvider interface — abstracts data access for live vs backtest modes.

Same code path principle: ProbabilityEngine accepts a DataProvider and doesn't
know whether it's running live or in a historical backtest.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional

import duckdb
from loguru import logger

from core.constants import STATION_COORDS
from core.db import get_connection
from services.bias_model import StationBias


# ---------------------------------------------------------------------------
# Lag constants — used by BacktestDataProvider to simulate real-world delays.
# Phase 0-1: only HRRR_AVAILABILITY_LAG_HOURS is active.
# Phase 2+: observation and partial-forecast lags will matter for drift.
# ---------------------------------------------------------------------------
HRRR_AVAILABILITY_LAG_HOURS = 2      # HRRR data lands ~45-90 min after run
OBS_REPORTING_LAG_MINUTES = 10       # METAR obs → API availability (Phase 2+)
PARTIAL_FORECAST_WINDOW_MINUTES = 30  # fxx trickle-in window (Phase 2+)


def _strip_tz(dt: datetime) -> datetime:
    """Strip timezone info for DuckDB TIMESTAMP columns (stored as naive UTC)."""
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


class DataProvider(ABC):
    """Abstract data provider for ProbabilityEngine.

    Implementations supply data access methods that ProbabilityEngine
    previously did via direct DB queries.
    """

    @abstractmethod
    def get_forecast_high(self, station_id: str) -> Optional[float]:
        """MAX(temp_f) from the relevant model run.

        Live: latest model run in the DB.
        Backtest: the specific model_run being evaluated.
        """
        ...

    @abstractmethod
    def get_bias_stats(self, station_id: str) -> Optional[StationBias]:
        """Latest bias stats for this station.

        Live: latest row from station_bias (cached with TTL).
        Backtest: latest row where calculated_at <= ref_time.
        """
        ...

    @abstractmethod
    def get_drift_score(self, city: str) -> float:
        """Latest drift score for a city.

        Live: latest from drift_signals.
        Backtest: 0.0 (no historical drift signals).
        """
        ...

    @abstractmethod
    def get_recent_drift_scores(self, city: str, limit: int = 10) -> List[float]:
        """Last N drift scores for stability computation.

        Live: last N rows from drift_signals.
        Backtest: empty list (stability_factor defaults to 1.0).
        """
        ...

    @abstractmethod
    def get_recent_forecast_highs(self, station_id: str, limit: int = 3) -> List[float]:
        """MAX(temp_f) from the last N model runs for convergence.

        Live: last N distinct model_run groups.
        Backtest: last N model runs where model_run <= current run.
        """
        ...

    @abstractmethod
    def get_forecast_curve(self, station_id: str) -> List[tuple]:
        """Full HRRR hourly forecast curve: [(valid_at, temp_f), ...].

        Live: curve from the latest model run.
        Backtest: curve from the specific model_run being evaluated.
        """
        ...

    @abstractmethod
    def get_observations_in_range(
        self, station_id: str, start_utc: datetime, end_utc: datetime,
    ) -> List[tuple]:
        """Observations within a UTC time range: [(observed_at, temp_f), ...].

        Backtest: enforces observed_at <= ref_time for walk-forward safety.
        """
        ...


class LiveDataProvider(DataProvider):
    """Provides live data from the latest available DB state.

    Handles bias caching with TTL refresh (migrated from ProbabilityEngine).
    """

    CACHE_TTL_SECONDS = 3600  # 1 hour

    def __init__(self, db_path: str = "data/alphatemp.duckdb"):
        self.db_path = db_path
        self._bias_cache = {}  # type: Dict[str, StationBias]
        self._cache_loaded_at = None  # type: Optional[datetime]

    def _refresh_cache_if_stale(self) -> None:
        now = datetime.now(timezone.utc)
        if self._cache_loaded_at is None or \
           (now - self._cache_loaded_at).total_seconds() >= self.CACHE_TTL_SECONDS:
            self._load_bias_cache()

    def _load_bias_cache(self) -> None:
        con = get_connection(self.db_path)
        for station_id in STATION_COORDS:
            row = con.execute(
                """SELECT mean_bias, std_error, sample_days
                   FROM station_bias
                   WHERE station_id = ?
                   ORDER BY calculated_at DESC LIMIT 1""",
                [station_id],
            ).fetchone()
            if row:
                self._bias_cache[station_id] = StationBias(
                    station_id=station_id,
                    mean_bias=row[0],
                    std_error=row[1],
                    sample_days=row[2],
                )
        con.close()
        self._cache_loaded_at = datetime.now(timezone.utc)
        logger.debug(f"Loaded bias cache for {len(self._bias_cache)} stations")

    def get_forecast_high(self, station_id: str) -> Optional[float]:
        # Settlement-day filter: only fxx whose valid_at falls within
        # the settlement window (05z to 29z = midnight-to-midnight EST)
        con = get_connection(self.db_path)
        row = con.execute(
            """SELECT MAX(f.temp_f) FROM forecasts f
               WHERE f.station_id = ?
               AND f.model_name = 'hrrr'
               AND f.model_run = (
                   SELECT MAX(model_run) FROM forecasts
                   WHERE station_id = ? AND model_name = 'hrrr'
               )
               AND (EXTRACT(HOUR FROM f.model_run) + f.fxx) >= 5
               AND (EXTRACT(HOUR FROM f.model_run) + f.fxx) < 29""",
            [station_id, station_id],
        ).fetchone()
        con.close()
        return row[0] if row and row[0] is not None else None

    def get_bias_stats(self, station_id: str) -> Optional[StationBias]:
        self._refresh_cache_if_stale()
        return self._bias_cache.get(station_id)

    def get_drift_score(self, city: str) -> float:
        con = get_connection(self.db_path)
        row = con.execute(
            """SELECT drift_score FROM drift_signals
               WHERE city = ?
               ORDER BY calculated_at DESC LIMIT 1""",
            [city],
        ).fetchone()
        con.close()
        return row[0] if row else 0.0

    def get_recent_drift_scores(self, city: str, limit: int = 10) -> List[float]:
        con = get_connection(self.db_path)
        rows = con.execute(
            """SELECT drift_score FROM drift_signals
               WHERE city = ?
               ORDER BY calculated_at DESC LIMIT ?""",
            [city, limit],
        ).fetchall()
        con.close()
        return [r[0] for r in rows]

    def get_recent_forecast_highs(self, station_id: str, limit: int = 3) -> List[float]:
        con = get_connection(self.db_path)
        rows = con.execute(
            """SELECT model_run, MAX(temp_f) as fcst_high
               FROM forecasts
               WHERE station_id = ?
               AND model_name = 'hrrr'
               AND (EXTRACT(HOUR FROM model_run) + fxx) >= 5
               AND (EXTRACT(HOUR FROM model_run) + fxx) < 29
               GROUP BY model_run
               ORDER BY model_run DESC
               LIMIT ?""",
            [station_id, limit],
        ).fetchall()
        con.close()
        return [r[1] for r in rows if r[1] is not None]

    def get_forecast_curve(self, station_id: str) -> List[tuple]:
        con = get_connection(self.db_path)
        rows = con.execute(
            """SELECT valid_at, temp_f FROM forecasts
               WHERE station_id = ?
               AND model_name = 'hrrr'
               AND model_run = (
                   SELECT MAX(model_run) FROM forecasts
                   WHERE station_id = ? AND model_name = 'hrrr'
               )
               ORDER BY valid_at""",
            [station_id, station_id],
        ).fetchall()
        con.close()
        return rows

    def get_observations_in_range(
        self, station_id: str, start_utc: datetime, end_utc: datetime,
    ) -> List[tuple]:
        con = get_connection(self.db_path)
        rows = con.execute(
            """SELECT observed_at, temp_f FROM observations
               WHERE station_id = ? AND observed_at BETWEEN ? AND ?
               AND temp_f IS NOT NULL
               ORDER BY observed_at""",
            [station_id, _strip_tz(start_utc), _strip_tz(end_utc)],
        ).fetchall()
        con.close()
        return rows


class BacktestDataProvider(DataProvider):
    """Provides time-fenced historical data for backtesting.

    Anchored to a specific model_run. Simulates what data would have been
    available at ref_time (model_run + HRRR_AVAILABILITY_LAG_HOURS).

    Parameters
    ----------
    db_path : str
        Path to DuckDB database.
    station_id : str
        Station to query forecasts for (e.g., "KNYC").
    model_run : datetime
        The specific HRRR model run being evaluated.
    ref_time : datetime
        Simulated "current time" — typically model_run + 2h.
    connection : duckdb.DuckDBPyConnection, optional
        Shared persistent connection from the Backtester. If provided,
        this provider will NOT close it. If omitted, opens its own.
    """

    def __init__(
        self,
        db_path: str,
        station_id: str,
        model_run: datetime,
        ref_time: datetime,
        connection: Optional[duckdb.DuckDBPyConnection] = None,
        model_name: str = 'hrrr',
    ):
        self.db_path = db_path
        self.station_id = station_id
        self.model_run = _strip_tz(model_run)
        self.ref_time = _strip_tz(ref_time)
        self.model_name = model_name
        self._shared_con = connection
        self._bias_cache = {}  # type: Dict[str, StationBias]
        self._bias_loaded = False

    def _ensure_bias_loaded(self) -> None:
        if self._bias_loaded:
            return
        con = self._shared_con if self._shared_con else get_connection(self.db_path)
        try:
            for sid in STATION_COORDS:
                row = con.execute(
                    """SELECT mean_bias, std_error, sample_days
                       FROM station_bias
                       WHERE station_id = ? AND calculated_at <= ?
                       ORDER BY calculated_at DESC LIMIT 1""",
                    [sid, self.ref_time],
                ).fetchone()
                if row:
                    self._bias_cache[sid] = StationBias(
                        station_id=sid,
                        mean_bias=row[0],
                        std_error=row[1],
                        sample_days=row[2],
                    )
        finally:
            if self._shared_con is None:
                con.close()
        self._bias_loaded = True

    def get_forecast_high(self, station_id: str) -> Optional[float]:
        con = self._shared_con if self._shared_con else get_connection(self.db_path)
        try:
            # Settlement-day filter: only fxx whose valid_at falls within
            # the settlement window (05z to 29z = midnight-to-midnight EST)
            row = con.execute(
                """SELECT MAX(temp_f) FROM forecasts
                   WHERE station_id = ? AND model_run = ?
                   AND model_name = ?
                   AND (EXTRACT(HOUR FROM model_run) + fxx) >= 5
                   AND (EXTRACT(HOUR FROM model_run) + fxx) < 29""",
                [station_id, self.model_run, self.model_name],
            ).fetchone()
            return row[0] if row and row[0] is not None else None
        finally:
            if self._shared_con is None:
                con.close()

    def get_bias_stats(self, station_id: str) -> Optional[StationBias]:
        self._ensure_bias_loaded()
        return self._bias_cache.get(station_id)

    def get_drift_score(self, city: str) -> float:
        return 0.0

    def get_recent_drift_scores(self, city: str, limit: int = 10) -> List[float]:
        return []

    def get_recent_forecast_highs(self, station_id: str, limit: int = 3) -> List[float]:
        con = self._shared_con if self._shared_con else get_connection(self.db_path)
        try:
            rows = con.execute(
                """SELECT model_run, MAX(temp_f) as fcst_high
                   FROM forecasts
                   WHERE station_id = ? AND model_run <= ?
                   AND model_name = ?
                   AND (EXTRACT(HOUR FROM model_run) + fxx) >= 5
                   AND (EXTRACT(HOUR FROM model_run) + fxx) < 29
                   GROUP BY model_run
                   ORDER BY model_run DESC
                   LIMIT ?""",
                [station_id, self.model_run, self.model_name, limit],
            ).fetchall()
            return [r[1] for r in rows if r[1] is not None]
        finally:
            if self._shared_con is None:
                con.close()

    def get_forecast_curve(self, station_id: str) -> List[tuple]:
        con = self._shared_con if self._shared_con else get_connection(self.db_path)
        try:
            return con.execute(
                """SELECT valid_at, temp_f FROM forecasts
                   WHERE station_id = ? AND model_run = ?
                   AND model_name = ?
                   ORDER BY valid_at""",
                [station_id, self.model_run, self.model_name],
            ).fetchall()
        finally:
            if self._shared_con is None:
                con.close()

    def get_observations_in_range(
        self, station_id: str, start_utc: datetime, end_utc: datetime,
    ) -> List[tuple]:
        # Enforce walk-forward: never see obs beyond ref_time
        effective_end = min(_strip_tz(end_utc), self.ref_time)
        con = self._shared_con if self._shared_con else get_connection(self.db_path)
        try:
            return con.execute(
                """SELECT observed_at, temp_f FROM observations
                   WHERE station_id = ? AND observed_at BETWEEN ? AND ?
                   AND temp_f IS NOT NULL
                   ORDER BY observed_at""",
                [station_id, _strip_tz(start_utc), effective_end],
            ).fetchall()
        finally:
            if self._shared_con is None:
                con.close()
