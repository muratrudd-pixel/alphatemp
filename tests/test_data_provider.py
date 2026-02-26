"""Tests for DataProvider implementations and Backtester."""

import os
from datetime import date, datetime, timedelta, timezone

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
            "INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            [station_id, model_run, valid_at, temp_f, temp_c],
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
