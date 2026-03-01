"""Tests for DataProvider implementations and Backtester."""

import os
from datetime import date, datetime, timedelta, timezone

import duckdb
import pytest

from core.db import init_db, get_connection
from services.bias_model import StationBias
from services.data_provider import LiveDataProvider, BacktestDataProvider
from services.backtester import (
    Backtester,
    KalshiBracket,
    compute_brier_score,
    compute_brier_score_1f,
    map_probs_to_kalshi_brackets,
    uniform_model,
    walk_forward_model,
    walk_forward_t_model,
    _walk_forward_bias_query,
    WALK_FORWARD_MIN_DAYS,
)

TEST_DB = "data/test_backtest.duckdb"


@pytest.fixture
def test_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)
    yield TEST_DB
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def _seed_forecast(con, station_id, model_run_str, temps):
    """Seed forecast temps for a model run (fxx 1..len(temps))."""
    model_run = datetime.fromisoformat(model_run_str)
    for i, temp_f in enumerate(temps):
        fxx = i + 1
        valid_at = model_run + timedelta(hours=fxx)
        temp_c = round((temp_f - 32) * 5 / 9, 2)
        con.execute(
            "INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, fxx, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            [station_id, model_run, valid_at, temp_f, temp_c, fxx],
        )


def _seed_backtest_data(db_path):
    """Seed 3 days of forecasts + nws_daily + bias + kalshi_settlements."""
    con = get_connection(db_path)

    # 3 days of 12z forecasts (temps climb to peak then fall)
    for day_str, peak in [("2025-01-15", 50.0), ("2025-01-16", 55.0), ("2025-01-17", 45.0)]:
        temps = [peak - 5, peak - 3, peak - 1, peak, peak - 1, peak - 3, peak - 5, peak - 7]
        _seed_forecast(con, "KNYC", f"{day_str} 12:00:00", temps)

    # NWS daily actual highs
    for day_str, actual in [("2025-01-15", 48.0), ("2025-01-16", 56.0), ("2025-01-17", 44.0)]:
        con.execute(
            "INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at) "
            "VALUES ('KNYC', ?, ?, NULL, 'test', CURRENT_TIMESTAMP)",
            [day_str, actual],
        )

    # Bias stats (computed before backtest range)
    con.execute(
        "INSERT INTO station_bias (station_id, calculated_at, mean_bias, std_error, sample_days) "
        "VALUES ('KNYC', '2025-01-01 00:00:00', 1.5, 2.0, 50)"
    )

    # Kalshi settlements for Jan 15 (actual high = 48°F)
    # Brackets: ≤45, [46,47], [48,49], [50,51], ≥52
    for ticker, floor, cap, settled in [
        ("KXHIGHNY-25JAN15-T46", None, 46.0, 0),
        ("KXHIGHNY-25JAN15-B46.5", 46.0, 47.0, 0),
        ("KXHIGHNY-25JAN15-B48.5", 48.0, 49.0, 1),  # actual=48, settled YES
        ("KXHIGHNY-25JAN15-B50.5", 50.0, 51.0, 0),
        ("KXHIGHNY-25JAN15-T52", 52.0, None, 0),
    ]:
        con.execute(
            "INSERT INTO kalshi_settlements "
            "(market_ticker, event_ticker, series_ticker, city, measure, event_date, "
            "floor_strike, cap_strike, settled_yes, volume, close_time, ingested_at) "
            "VALUES (?, 'KXHIGHNY-25JAN15', 'KXHIGHNY', 'NYC', 'high', '2025-01-15', "
            "?, ?, ?, 100, '2025-01-16 00:00:00', CURRENT_TIMESTAMP)",
            [ticker, floor, cap, settled],
        )

    con.close()


@pytest.fixture
def seeded_db(test_db):
    _seed_backtest_data(test_db)
    return test_db


# -----------------------------------------------------------------------
# LiveDataProvider tests
# -----------------------------------------------------------------------

def test_live_provider_forecast_high(seeded_db):
    """get_forecast_high returns MAX(temp_f) from latest model run."""
    provider = LiveDataProvider(db_path=seeded_db)
    # Latest run is Jan 17 12z with peak 45°F
    high = provider.get_forecast_high("KNYC")
    assert high == 45.0


def test_live_provider_bias_stats(seeded_db):
    """get_bias_stats returns cached bias from station_bias table."""
    provider = LiveDataProvider(db_path=seeded_db)
    bias = provider.get_bias_stats("KNYC")
    assert bias is not None
    assert bias.mean_bias == 1.5
    assert bias.std_error == 2.0
    assert bias.sample_days == 50


def test_live_provider_no_data(test_db):
    """Returns None when no data exists."""
    provider = LiveDataProvider(db_path=test_db)
    assert provider.get_forecast_high("KNYC") is None
    assert provider.get_bias_stats("KNYC") is None
    assert provider.get_drift_score("NYC") == 0.0
    assert provider.get_recent_drift_scores("NYC") == []
    assert provider.get_recent_forecast_highs("KNYC") == []


# -----------------------------------------------------------------------
# BacktestDataProvider tests
# -----------------------------------------------------------------------

def test_backtest_provider_specific_run(seeded_db):
    """BacktestDataProvider returns data only for its configured model_run."""
    # Create provider for Jan 15 12z (peak 50)
    provider = BacktestDataProvider(
        db_path=seeded_db,
        station_id="KNYC",
        model_run=datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc),
        ref_time=datetime(2025, 1, 15, 14, 0, tzinfo=timezone.utc),
    )
    assert provider.get_forecast_high("KNYC") == 50.0

    # Provider for Jan 16 12z (peak 55)
    provider2 = BacktestDataProvider(
        db_path=seeded_db,
        station_id="KNYC",
        model_run=datetime(2025, 1, 16, 12, 0, tzinfo=timezone.utc),
        ref_time=datetime(2025, 1, 16, 14, 0, tzinfo=timezone.utc),
    )
    assert provider2.get_forecast_high("KNYC") == 55.0


def test_backtest_provider_bias_time_fence(seeded_db):
    """BacktestDataProvider only sees bias calculated before ref_time."""
    # Insert a second bias row AFTER the backtest range
    con = get_connection(seeded_db)
    con.execute(
        "INSERT INTO station_bias (station_id, calculated_at, mean_bias, std_error, sample_days) "
        "VALUES ('KNYC', '2025-02-01 00:00:00', 3.0, 1.0, 100)"
    )
    con.close()

    # Provider with ref_time in January — should only see the Jan bias (1.5)
    provider = BacktestDataProvider(
        db_path=seeded_db,
        station_id="KNYC",
        model_run=datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc),
        ref_time=datetime(2025, 1, 15, 14, 0, tzinfo=timezone.utc),
    )
    bias = provider.get_bias_stats("KNYC")
    assert bias is not None
    assert bias.mean_bias == 1.5

    # Provider with ref_time in February — should see the Feb bias (3.0)
    provider2 = BacktestDataProvider(
        db_path=seeded_db,
        station_id="KNYC",
        model_run=datetime(2025, 2, 1, 12, 0, tzinfo=timezone.utc),
        ref_time=datetime(2025, 2, 1, 14, 0, tzinfo=timezone.utc),
    )
    bias2 = provider2.get_bias_stats("KNYC")
    assert bias2 is not None
    assert bias2.mean_bias == 3.0


def test_backtest_provider_drift_zero(seeded_db):
    """BacktestDataProvider always returns 0 drift."""
    provider = BacktestDataProvider(
        db_path=seeded_db,
        station_id="KNYC",
        model_run=datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc),
        ref_time=datetime(2025, 1, 15, 14, 0, tzinfo=timezone.utc),
    )
    assert provider.get_drift_score("NYC") == 0.0
    assert provider.get_recent_drift_scores("NYC") == []


def test_backtest_provider_recent_highs_fenced(seeded_db):
    """get_recent_forecast_highs only returns runs <= model_run."""
    # Provider anchored to Jan 16 12z — should see Jan 15 and Jan 16 runs
    provider = BacktestDataProvider(
        db_path=seeded_db,
        station_id="KNYC",
        model_run=datetime(2025, 1, 16, 12, 0, tzinfo=timezone.utc),
        ref_time=datetime(2025, 1, 16, 14, 0, tzinfo=timezone.utc),
    )
    highs = provider.get_recent_forecast_highs("KNYC", limit=3)
    # Should be [55.0 (Jan16), 50.0 (Jan15)] — NOT 45.0 (Jan17 is in the future)
    assert len(highs) == 2
    assert highs[0] == 55.0  # Most recent first
    assert highs[1] == 50.0


def test_backtest_provider_shared_connection(seeded_db):
    """BacktestDataProvider works correctly with a shared connection."""
    import duckdb
    shared_con = duckdb.connect(seeded_db, read_only=True)
    try:
        provider = BacktestDataProvider(
            db_path=seeded_db,
            station_id="KNYC",
            model_run=datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc),
            ref_time=datetime(2025, 1, 15, 14, 0, tzinfo=timezone.utc),
            connection=shared_con,
        )
        assert provider.get_forecast_high("KNYC") == 50.0
        bias = provider.get_bias_stats("KNYC")
        assert bias is not None
        assert bias.mean_bias == 1.5
    finally:
        shared_con.close()


# -----------------------------------------------------------------------
# Brier score tests
# -----------------------------------------------------------------------

def test_brier_perfect():
    """Perfect prediction: all probability on the settled bracket."""
    brackets = [
        KalshiBracket(None, 46.0, 0),
        KalshiBracket(46.0, 47.0, 0),
        KalshiBracket(48.0, 49.0, 1),  # settled
        KalshiBracket(50.0, 51.0, 0),
        KalshiBracket(52.0, None, 0),
    ]
    # Model puts all probability on bracket containing 48
    probs_1f = {48: 0.6, 49: 0.4}  # All mass in the settled bracket
    mapped = map_probs_to_kalshi_brackets(probs_1f, brackets)
    brier = compute_brier_score(mapped, brackets)
    assert brier == pytest.approx(0.0, abs=0.01)


def test_brier_uniform():
    """Uniform distribution across Kalshi brackets."""
    brackets = [
        KalshiBracket(None, 46.0, 0),
        KalshiBracket(46.0, 47.0, 0),
        KalshiBracket(48.0, 49.0, 1),  # settled
        KalshiBracket(50.0, 51.0, 0),
        KalshiBracket(52.0, None, 0),
    ]
    # Uniform across many 1°F brackets
    probs_1f = {k: 1.0 / 31 for k in range(35, 66)}
    mapped = map_probs_to_kalshi_brackets(probs_1f, brackets)
    brier = compute_brier_score(mapped, brackets)
    # Should be > 0 (uniform is not perfect)
    assert brier > 0.5


def test_brier_1f_perfect():
    """1°F fallback: perfect prediction."""
    probs = {48: 1.0}
    brier = compute_brier_score_1f(probs, 48.0)
    assert brier == pytest.approx(0.0, abs=0.001)


def test_brier_1f_uniform():
    """1°F fallback: uniform distribution."""
    probs = {k: 1.0 / 31 for k in range(35, 66)}
    brier = compute_brier_score_1f(probs, 50.0)
    assert brier > 0.0


# -----------------------------------------------------------------------
# Kalshi bracket mapping tests
# -----------------------------------------------------------------------

def test_bracket_contains_lower_tail():
    """Lower tail: temp < cap_strike."""
    b = KalshiBracket(None, 46.0, 0)
    assert b.contains(45)
    assert not b.contains(46)
    assert not b.contains(47)


def test_bracket_contains_interior():
    """Interior: floor_strike <= temp <= cap_strike."""
    b = KalshiBracket(48.0, 49.0, 1)
    assert not b.contains(47)
    assert b.contains(48)
    assert b.contains(49)
    assert not b.contains(50)


def test_bracket_contains_upper_tail():
    """Upper tail: temp > floor_strike."""
    b = KalshiBracket(52.0, None, 0)
    assert not b.contains(52)
    assert b.contains(53)
    assert b.contains(60)


def test_map_probs_covers_all_brackets():
    """Mapped probabilities should sum to ~1.0."""
    brackets = [
        KalshiBracket(None, 46.0, 0),
        KalshiBracket(46.0, 47.0, 0),
        KalshiBracket(48.0, 49.0, 1),
        KalshiBracket(50.0, 51.0, 0),
        KalshiBracket(52.0, None, 0),
    ]
    probs_1f = {k: 1.0 / 31 for k in range(35, 66)}
    mapped = map_probs_to_kalshi_brackets(probs_1f, brackets)
    assert sum(mapped) == pytest.approx(1.0, abs=0.01)
    assert len(mapped) == 5


# -----------------------------------------------------------------------
# Backtester integration tests
# -----------------------------------------------------------------------

def test_backtester_uniform_model(seeded_db):
    """Backtester runs end-to-end with uniform model."""
    bt = Backtester(db_path=seeded_db, city="NYC")
    result = bt.run(uniform_model, run_hours=[12])

    # 3 days, 12z only, all should have forecast data
    assert result.total_days == 3
    assert result.total_evaluations == 3
    assert result.skipped == 0
    assert result.mean_brier > 0.0
    # Jan 15 has Kalshi brackets, Jan 16-17 don't
    assert result.kalshi_scored == 1
    assert result.fallback_scored == 2


def test_backtester_skips_missing(seeded_db):
    """Backtester correctly skips dates without forecast data."""
    bt = Backtester(db_path=seeded_db, city="NYC")
    # 00z runs don't exist in seed data
    result = bt.run(uniform_model, run_hours=[0])
    assert result.total_evaluations == 0
    assert result.skipped == 3  # 3 days × 1 hour, all skipped


def test_backtester_per_hour(seeded_db):
    """Backtester produces per-run-hour breakdown."""
    bt = Backtester(db_path=seeded_db, city="NYC")
    result = bt.run(uniform_model, run_hours=[12])
    assert 12 in result.by_run_hour
    assert result.by_run_hour[12]["count"] == 3


def test_engine_with_backtest_provider(seeded_db):
    """ProbabilityEngine works with BacktestDataProvider (same code path)."""
    from services.probability import ProbabilityEngine

    provider = BacktestDataProvider(
        db_path=seeded_db,
        station_id="KNYC",
        model_run=datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc),
        ref_time=datetime(2025, 1, 15, 14, 0, tzinfo=timezone.utc),
    )
    engine = ProbabilityEngine(data_provider=provider)
    forecast = engine.calculate_city("NYC", ref_time=datetime(2025, 1, 15, 14, 0, tzinfo=timezone.utc))

    assert forecast is not None
    assert forecast.city == "NYC"
    # center = fcst_high(50) - bias(1.5) + drift(0) = 48.5
    assert forecast.center == pytest.approx(48.5, abs=0.1)
    total = sum(forecast.bracket_probs.values())
    assert total == pytest.approx(1.0, abs=0.01)


# -----------------------------------------------------------------------
# Walk-forward model tests
# -----------------------------------------------------------------------

WALK_FORWARD_DB = "data/test_walkforward.duckdb"


def _seed_walk_forward_data(db_path, n_days=120):
    """Seed n_days of 12z+00z forecasts + nws_daily for walk-forward testing.

    Forecast bias is intentionally different per run hour:
      - 12z forecasts: peak = actual + 2.0 (positive bias)
      - 00z forecasts: peak = actual + 0.5 (smaller bias)
    This lets us verify per-run-hour differentiation.
    """
    con = get_connection(db_path)
    base_date = date(2024, 1, 1)

    for i in range(n_days):
        obs_date = base_date + timedelta(days=i)
        actual_high = 50.0 + (i % 10)  # Cycles 50-59°F

        # NWS settlement truth
        con.execute(
            "INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at) "
            "VALUES ('KNYC', ?, ?, NULL, 'test', CURRENT_TIMESTAMP)",
            [obs_date.isoformat(), actual_high],
        )

        # 12z forecast: intentional +2.0 bias
        fcst_peak_12z = actual_high + 2.0
        temps_12z = [fcst_peak_12z - 3, fcst_peak_12z - 1, fcst_peak_12z,
                     fcst_peak_12z - 1, fcst_peak_12z - 3]
        model_run_12z = datetime(obs_date.year, obs_date.month, obs_date.day, 12, 0)
        for j, temp_f in enumerate(temps_12z):
            fxx = j + 1  # fxx 1-5; 12+1=13 to 12+5=17, all within settlement window
            valid_at = model_run_12z + timedelta(hours=fxx)
            temp_c = round((temp_f - 32) * 5 / 9, 2)
            con.execute(
                "INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, fxx, ingested_at) "
                "VALUES ('KNYC', ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                [model_run_12z, valid_at, temp_f, temp_c, fxx],
            )

        # 00z forecast: intentional +0.5 bias
        # Use fxx 6-10 so all fall within settlement window (0+6=6 >= 5)
        fcst_peak_00z = actual_high + 0.5
        temps_00z = [fcst_peak_00z - 3, fcst_peak_00z - 1, fcst_peak_00z,
                     fcst_peak_00z - 1, fcst_peak_00z - 3]
        model_run_00z = datetime(obs_date.year, obs_date.month, obs_date.day, 0, 0)
        for j, temp_f in enumerate(temps_00z):
            fxx = j + 6  # fxx 6-10; 0+6=6 to 0+10=10, all within settlement window
            valid_at = model_run_00z + timedelta(hours=fxx)
            temp_c = round((temp_f - 32) * 5 / 9, 2)
            con.execute(
                "INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, fxx, ingested_at) "
                "VALUES ('KNYC', ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                [model_run_00z, valid_at, temp_f, temp_c, fxx],
            )

    con.close()


def _cleanup_db(path):
    """Remove DuckDB file and its WAL."""
    for suffix in ("", ".wal"):
        p = path + suffix if suffix else path
        if os.path.exists(p):
            os.remove(p)


@pytest.fixture
def walk_forward_db():
    """Create a test DB with 120 days of walk-forward-friendly data."""
    _cleanup_db(WALK_FORWARD_DB)
    init_db(WALK_FORWARD_DB)
    _seed_walk_forward_data(WALK_FORWARD_DB, n_days=120)
    yield WALK_FORWARD_DB
    _cleanup_db(WALK_FORWARD_DB)


def test_walk_forward_no_future_data(walk_forward_db):
    """Walk-forward model only uses dates strictly before current_date."""
    con = duckdb.connect(walk_forward_db, read_only=True)

    # Query bias as of day 100 (2024-04-10)
    test_date = date(2024, 4, 10)
    stats = _walk_forward_bias_query(con, 12, "KNYC", test_date)
    assert stats is not None
    n_dates = int(stats[2])

    # Query bias as of day 101 (2024-04-11) — should have exactly 1 more date
    next_date = date(2024, 4, 11)
    stats_next = _walk_forward_bias_query(con, 12, "KNYC", next_date)
    assert stats_next is not None
    assert int(stats_next[2]) == n_dates + 1

    # Query bias as of day 1 — should be None (no prior data)
    first_date = date(2024, 1, 1)
    stats_first = _walk_forward_bias_query(con, 12, "KNYC", first_date)
    assert stats_first is None  # 0 prior dates < min window

    con.close()


def test_walk_forward_min_window(walk_forward_db):
    """Returns None when fewer than WALK_FORWARD_MIN_DAYS of history."""
    con = duckdb.connect(walk_forward_db, read_only=True)

    # Day 89 (2024-03-30) — only 89 prior dates, below 90-day minimum
    too_early = date(2024, 3, 30)
    stats = _walk_forward_bias_query(con, 12, "KNYC", too_early)
    assert stats is None

    # Day 91 (2024-04-01) — 91 prior dates, above minimum
    enough = date(2024, 4, 1)
    stats_ok = _walk_forward_bias_query(con, 12, "KNYC", enough)
    assert stats_ok is not None
    assert int(stats_ok[2]) >= WALK_FORWARD_MIN_DAYS

    con.close()


def test_walk_forward_per_run_hour(walk_forward_db):
    """Different run hours produce different bias values."""
    con = duckdb.connect(walk_forward_db, read_only=True)

    test_date = date(2024, 5, 1)  # Well past the 90-day window

    stats_12z = _walk_forward_bias_query(con, 12, "KNYC", test_date)
    stats_00z = _walk_forward_bias_query(con, 0, "KNYC", test_date)

    assert stats_12z is not None
    assert stats_00z is not None

    # 12z bias should be ~2.0, 00z bias should be ~0.5
    assert abs(stats_12z[0] - 2.0) < 0.3, f"12z bias {stats_12z[0]} not near 2.0"
    assert abs(stats_00z[0] - 0.5) < 0.3, f"00z bias {stats_00z[0]} not near 0.5"

    # They must be different from each other
    assert abs(stats_12z[0] - stats_00z[0]) > 1.0

    con.close()


def test_walk_forward_uses_nws_truth(walk_forward_db):
    """Error is computed against nws_daily.max_temp_f, not observations."""
    con = duckdb.connect(walk_forward_db, read_only=True)

    test_date = date(2024, 5, 1)
    stats = _walk_forward_bias_query(con, 12, "KNYC", test_date)
    assert stats is not None

    # Our seeded data has forecast = actual + 2.0 for 12z,
    # so mean bias should be ~2.0 (forecast - nws_truth)
    mean_bias = stats[0]
    assert abs(mean_bias - 2.0) < 0.3, (
        f"Mean bias {mean_bias} suggests error is NOT computed against NWS truth"
    )

    con.close()


def test_walk_forward_model_returns_probs(walk_forward_db):
    """walk_forward_model returns valid bracket probabilities."""
    con = duckdb.connect(walk_forward_db, read_only=True)
    try:
        # Date within seeded range (day 110) with >90 days prior history
        test_run = datetime(2024, 4, 20, 12, 0, tzinfo=timezone.utc)
        provider = BacktestDataProvider(
            db_path=walk_forward_db,
            station_id="KNYC",
            model_run=test_run,
            ref_time=test_run + timedelta(hours=2),
            connection=con,
        )

        probs = walk_forward_model(provider, test_run)
        assert probs is not None
        assert len(probs) > 0
        assert abs(sum(probs.values()) - 1.0) < 0.01
    finally:
        con.close()


def test_walk_forward_t_model_returns_probs(walk_forward_db):
    """walk_forward_t_model returns valid bracket probabilities."""
    con = duckdb.connect(walk_forward_db, read_only=True)
    try:
        test_run = datetime(2024, 4, 20, 12, 0, tzinfo=timezone.utc)
        provider = BacktestDataProvider(
            db_path=walk_forward_db,
            station_id="KNYC",
            model_run=test_run,
            ref_time=test_run + timedelta(hours=2),
            connection=con,
        )

        probs = walk_forward_t_model(provider, test_run)
        assert probs is not None
        assert len(probs) > 0
        assert abs(sum(probs.values()) - 1.0) < 0.01
    finally:
        con.close()
