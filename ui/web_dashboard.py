"""AlphaTemp Web Dashboard — FastAPI backend serving Plotly + Tailwind frontend."""

import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from core.constants import CITIES, STATION_COORDS
from core.db import get_connection, init_db
from core.timezone import et_day_bounds_utc, get_today_et
from services.probability import ProbabilityEngine

app = FastAPI(title="AlphaTemp Command Center")

# Static files — directory lives at project root alongside ui/
_static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Single engine instance — reused across requests
engine = ProbabilityEngine()


@app.on_event("startup")
async def startup():
    """Ensure DB tables exist and load bias cache."""
    init_db()
    try:
        engine.load_bias_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    """Redirect root to Operations tab."""
    return RedirectResponse(url="/operations")


@app.get("/operations")
async def operations_page(request: Request):
    """Serve the Operations tab."""
    return templates.TemplateResponse("operations.html", {"request": request, "active_tab": "operations"})


@app.get("/performance")
async def performance_page(request: Request):
    """Serve the Performance tab."""
    return templates.TemplateResponse("performance.html", {"request": request, "active_tab": "performance"})


@app.get("/review")
async def review_page(request: Request):
    """Serve the Review tab."""
    return templates.TemplateResponse("review.html", {"request": request, "active_tab": "review"})


@app.get("/mobile")
async def mobile_page(request: Request):
    """Serve the Mobile tab."""
    return templates.TemplateResponse("mobile.html", {"request": request, "active_tab": "mobile"})


STALE_THRESHOLD_MINUTES = 30


@app.get("/api/health")
async def health():
    """Health check — DB connectivity + data freshness."""
    now = datetime.now(timezone.utc)
    db_ok = False
    obs_age = None
    fcst_age = None

    try:
        con = get_connection()
        con.execute("SELECT 1")
        db_ok = True

        # Latest observation age
        obs_row = con.execute(
            "SELECT MAX(observed_at) FROM observations"
        ).fetchone()
        if obs_row and obs_row[0] is not None:
            obs_ts = obs_row[0]
            if obs_ts.tzinfo is None:
                obs_ts = obs_ts.replace(tzinfo=timezone.utc)
            obs_age = round((now - obs_ts).total_seconds() / 60, 1)

        # Latest forecast age
        fcst_row = con.execute(
            "SELECT MAX(ingested_at) FROM forecasts WHERE model_name = 'hrrr'"
        ).fetchone()
        if fcst_row and fcst_row[0] is not None:
            fcst_ts = fcst_row[0]
            if fcst_ts.tzinfo is None:
                fcst_ts = fcst_ts.replace(tzinfo=timezone.utc)
            fcst_age = round((now - fcst_ts).total_seconds() / 60, 1)

        con.close()
    except Exception:
        pass

    obs_stale = obs_age is None or obs_age > STALE_THRESHOLD_MINUTES
    fcst_stale = fcst_age is None or fcst_age > STALE_THRESHOLD_MINUTES

    status = "ok"
    if not db_ok:
        status = "degraded"
    elif obs_stale or fcst_stale:
        status = "degraded"

    return {
        "status": status,
        "timestamp": now.isoformat(),
        "db_connected": db_ok,
        "obs_age_minutes": obs_age,
        "fcst_age_minutes": fcst_age,
        "obs_stale": obs_stale,
        "fcst_stale": fcst_stale,
    }


@app.get("/api/kpi-summary")
async def kpi_summary(city: str = "nyc"):
    """Bundled KPI metrics for the persistent header bar."""
    city_upper = city.upper()
    con = get_connection()
    try:
        # System status from health check
        health_data = await health()
        system_status = "red" if not health_data.get("db_connected", True) else (
            "amber" if health_data.get("obs_stale") or health_data.get("fcst_stale") else "green"
        )

        # Model high from probability engine
        try:
            forecast = engine.calculate_city(city_upper, None)
            model_high = forecast.center if forecast else None
        except Exception:
            model_high = None

        # Settlement status from nws_daily
        today_et = get_today_et()
        row = con.execute("""
            SELECT max_temp_f, source FROM nws_daily
            WHERE station_id = 'KNYC' AND obs_date = ?
            ORDER BY CASE source
                WHEN 'NWS_CLI' THEN 3 WHEN 'DSM' THEN 2 ELSE 1
            END DESC LIMIT 1
        """, [today_et]).fetchone()
        settlement = {
            "temp": row[0] if row else None,
            "source": row[1] if row else "pending",
        }

        # Drift
        drift_row = con.execute("""
            SELECT drift_score FROM drift_signals
            WHERE city = ? ORDER BY calculated_at DESC LIMIT 1
        """, [city_upper]).fetchone()
        drift = round(drift_row[0], 1) if drift_row else 0.0

        # Positions P&L
        positions = con.execute("""
            SELECT
                COUNT(*) FILTER (WHERE status = 'open') as open_count,
                COALESCE(SUM(net_pnl) FILTER (WHERE event_date = ?), 0) as day_pnl,
                COALESCE(SUM(net_pnl), 0) as total_pnl
            FROM paper_positions WHERE city = ?
        """, [today_et, city_upper]).fetchone()

        # Market consensus (highest-prob bracket from market)
        bracket_row = con.execute("""
            SELECT floor_strike, cap_strike FROM market_ticks
            WHERE city = ? AND captured_at >= NOW() - INTERVAL '10 minutes'
            ORDER BY yes_bid DESC LIMIT 1
        """, [city_upper]).fetchone()
        market_consensus = "{}-{}F".format(
            int(bracket_row[0]), int(bracket_row[1])
        ) if bracket_row else None

        return {
            "system_status": system_status,
            "model_high": model_high,
            "settlement": settlement,
            "market_consensus": market_consensus,
            "drift": drift,
            "open_positions": positions[0] if positions else 0,
            "day_pnl": round(positions[1], 2) if positions else 0,
            "total_pnl": round(positions[2], 2) if positions else 0,
        }
    finally:
        con.close()


@app.get("/health")
async def health_page(request: Request):
    """Serve the Health page."""
    return templates.TemplateResponse("health.html", {"request": request, "active_tab": "health"})


@app.get("/blotter")
async def blotter_page(request: Request):
    """Serve the Blotter page."""
    return templates.TemplateResponse("blotter.html", {"request": request, "active_tab": "blotter"})


@app.get("/api/observations/{city}")
async def observation_feed(city: str, date: str = None):
    """Unified observation feed — METAR, SPECI, and 6-hour synoptic max entries.

    Returns observations for the given day (midnight–midnight ET), newest first.
    Each entry has a source tag and is_new_high flag.
    """
    city = city.upper()
    if city not in CITIES:
        return {"error": f"Unknown city: {city}"}

    station_id = CITIES[city]["settlement"]

    day_start_utc, day_end_utc = et_day_bounds_utc(date)

    con = get_connection()
    rows = con.execute(
        """SELECT station_id, observed_at, temp_f, six_hr_max_c, raw_metar,
                  ingested_at, ingest_source
            FROM observations
            WHERE station_id = ?
            AND observed_at >= ? AND observed_at < ?
            ORDER BY observed_at ASC
            LIMIT 500""",
        [station_id, day_start_utc, day_end_utc],
    ).fetchall()

    # Build unified feed with source detection
    feed = []
    for sid, obs_at, temp_f, six_hr_max_c, metar, ingested, ingest_src in rows:
        # Report type from raw METAR text
        report_type = "SPECI" if metar and metar.strip().startswith("SPECI") else "METAR"
        # Combine API source + report type (e.g. "AWC SPECI", "Synoptic METAR")
        api_label = (ingest_src or "synoptic").upper()
        if api_label == "AWC":
            source = f"AWC {report_type}"
        elif api_label == "BACKFILL":
            source = f"Backfill {report_type}"
        else:
            source = f"Synoptic {report_type}"

        obs_at_iso = obs_at.isoformat()
        ingested_iso = ingested.isoformat() if ingested else None

        if temp_f is not None:
            feed.append({
                "station_id": sid,
                "observed_at": obs_at_iso,
                "ingested_at": ingested_iso,
                "source": source,
                "temp_f": round(temp_f, 1),
                "obs_window": "snapshot",
                "is_new_high": False,
            })

        if six_hr_max_c is not None:
            six_hr_max_f = round(six_hr_max_c * 9.0 / 5.0 + 32.0, 1)
            feed.append({
                "station_id": sid,
                "observed_at": obs_at_iso,
                "ingested_at": ingested_iso,
                "source": source,
                "temp_f": six_hr_max_f,
                "obs_window": "prior 6hrs",
                "is_new_high": False,
            })

    # NWS CLI daily high — add as its own row if available
    from core.timezone import ET as _ET
    nws_date = date if date else datetime.now(_ET).strftime("%Y-%m-%d")
    nws_row = con.execute(
        "SELECT max_temp_f, ingested_at FROM nws_daily WHERE station_id = ? AND obs_date = ?",
        [station_id, nws_date],
    ).fetchone()
    if nws_row and nws_row[0] is not None:
        feed.append({
            "station_id": station_id,
            "observed_at": f"{nws_date}T17:00:00",  # noon ET = 17:00 UTC
            "ingested_at": nws_row[1].isoformat() if nws_row[1] else None,
            "source": "NWS CLI",
            "temp_f": round(nws_row[0], 1),
            "obs_window": "daily high",
            "is_new_high": False,
        })
        # Re-sort after adding NWS CLI
        feed.sort(key=lambda x: x["observed_at"])

    # Track running high chronologically
    running_high = None
    for entry in feed:
        if running_high is None or entry["temp_f"] > running_high:
            running_high = entry["temp_f"]
            entry["is_new_high"] = True

    # Reverse for display (newest first)
    feed.reverse()

    con.close()
    return {
        "city": city,
        "station_id": station_id,
        "observations": feed,
        "running_high": round(running_high, 1) if running_high is not None else None,
    }


@app.get("/api/forecast-points/{city}")
async def forecast_point_feed(city: str, date: str = None):
    """Forecast run summary — one row per model run showing expected high and deltas.

    Returns runs for the given day (midnight–midnight ET), newest first.
    Each run shows its expected high, the time of that high, and the change
    from the prior run.
    """
    from collections import defaultdict

    city = city.upper()
    if city not in CITIES:
        return {"error": f"Unknown city: {city}"}

    station_id = CITIES[city]["settlement"]

    day_start_utc, day_end_utc = et_day_bounds_utc(date)

    con = get_connection()
    rows = con.execute(
        """SELECT model_run, valid_at, temp_f, ingested_at
           FROM forecasts
           WHERE station_id = ?
           AND valid_at >= ? AND valid_at < ?
           AND model_name = 'hrrr'
           ORDER BY model_run ASC, valid_at ASC""",
        [station_id, day_start_utc, day_end_utc],
    ).fetchall()

    # Group by model_run, find the high for each run
    runs_data = defaultdict(lambda: {"points": [], "ingested_at": None})
    for model_run, valid_at, temp_f, ingested in rows:
        if temp_f is None:
            continue
        entry = runs_data[model_run]
        entry["points"].append((valid_at, temp_f))
        if ingested and entry["ingested_at"] is None:
            entry["ingested_at"] = ingested

    # Build summaries with deltas from prior run
    summaries = []
    prev_high_f = None
    prev_high_at = None
    for model_run in sorted(runs_data.keys()):
        entry = runs_data[model_run]
        if not entry["points"]:
            continue

        # Find the high temp and its valid_at
        best_valid, best_temp = max(entry["points"], key=lambda p: p[1])

        high_f = round(best_temp, 1)
        high_at = best_valid.isoformat()
        ingested = entry["ingested_at"]

        # Compute deltas from prior run
        temp_change = round(high_f - prev_high_f, 1) if prev_high_f is not None else None
        time_change_mins = None
        if prev_high_at is not None:
            delta_secs = (best_valid - prev_high_at_dt).total_seconds()
            time_change_mins = round(delta_secs / 60)

        summaries.append({
            "model_run": model_run.isoformat(),
            "ingested_at": ingested.isoformat() if ingested else None,
            "high_temp_f": high_f,
            "high_valid_at": high_at,
            "temp_change": temp_change,
            "time_change_mins": time_change_mins,
        })

        prev_high_f = high_f
        prev_high_at = high_at
        prev_high_at_dt = best_valid

    # Reverse for display (newest first)
    summaries.reverse()

    con.close()
    latest_run = summaries[0]["model_run"] if summaries else None
    forecast_high = summaries[0]["high_temp_f"] if summaries else None
    return {
        "city": city,
        "station_id": station_id,
        "runs": summaries,
        "latest_run": latest_run,
        "forecast_high": forecast_high,
    }


@app.get("/api/forecast-curve/{city}")
async def forecast_curve(city: str, date: str = None):
    """Return HRRR forecast curve + observations + confidence ribbon bounds.

    Optional date param (YYYY-MM-DD) scopes to a specific day (midnight–midnight ET).
    When date is set: uses the latest model run *for that day*, returns prior runs,
    and fetches full-day observations from the settlement station.
    """
    from collections import defaultdict

    city = city.upper()
    if city not in CITIES:
        return {"error": f"Unknown city: {city}"}

    station_id = CITIES[city]["settlement"]
    con = get_connection()

    # Parse day boundaries (DST-aware via zoneinfo)
    day_start_utc = day_end_utc = None
    if date:
        try:
            datetime.strptime(date, "%Y-%m-%d")  # validate format
            day_start_utc, day_end_utc = et_day_bounds_utc(date)
        except ValueError:
            pass  # bad format — fall through to default behavior

    # Latest model run forecasts — scoped to day if date provided
    if day_start_utc:
        forecasts = con.execute(
            """SELECT valid_at, temp_f, model_run, ingested_at FROM forecasts
               WHERE station_id = ?
               AND model_name = 'hrrr'
               AND model_run = (
                   SELECT MAX(model_run) FROM forecasts
                   WHERE station_id = ? AND valid_at >= ? AND valid_at < ?
                   AND model_name = 'hrrr'
               )
               AND valid_at >= ? AND valid_at < ?
               ORDER BY valid_at""",
            [station_id, station_id, day_start_utc, day_end_utc,
             day_start_utc, day_end_utc],
        ).fetchall()
    else:
        forecasts = con.execute(
            """SELECT valid_at, temp_f, model_run, ingested_at FROM forecasts
               WHERE station_id = ?
               AND model_name = 'hrrr'
               AND model_run = (SELECT MAX(model_run) FROM forecasts WHERE station_id = ? AND model_name = 'hrrr')
               ORDER BY valid_at""",
            [station_id, station_id],
        ).fetchall()

    if not forecasts:
        con.close()
        return {"city": city, "forecasts": [], "observations": [],
                "ribbon": [], "prior_runs": []}

    # Build ribbon bounds using the engine's uncertainty model
    bias = engine.provider.get_bias_stats(station_id)
    historical_std = bias.std_error if bias else 2.0

    drift_scores = engine.provider.get_recent_drift_scores(city)
    stability = engine._compute_stability_from_scores(drift_scores)
    recent_highs = engine.provider.get_recent_forecast_highs(station_id)
    convergence = engine._compute_convergence_from_highs(recent_highs)

    ribbon = []
    forecast_points = []
    now = datetime.now(timezone.utc)
    for valid_at, temp_f, model_run_ts, ingested_ts in forecasts:
        if valid_at.tzinfo is None:
            ref = valid_at.replace(tzinfo=timezone.utc)
        else:
            ref = valid_at
        hours_ahead = (ref - now).total_seconds() / 3600
        if hours_ahead <= 0:
            std_at_hour = 0.01  # near-zero — ribbon collapses for past timestamps
        else:
            lead_factor = min(1.0, (hours_ahead / 18.0) ** 0.5)
            std_at_hour = max(0.3, historical_std * lead_factor * stability * convergence)
        upper = round(temp_f + 1.645 * std_at_hour, 1)
        lower = round(temp_f - 1.645 * std_at_hour, 1)

        ribbon.append({
            "valid_at": valid_at.isoformat(),
            "upper_90": upper,
            "lower_90": lower,
            "std": round(std_at_hour, 2),
        })
        forecast_points.append({
            "valid_at": valid_at.isoformat(),
            "temp_f": round(temp_f, 1),
            "model_run": model_run_ts.isoformat() if model_run_ts else None,
            "ingested_at": ingested_ts.isoformat() if ingested_ts else None,
        })

    # Prior model runs — always fetch (live and date modes)
    prior_runs = []
    if day_start_utc:
        latest_mr = con.execute(
            """SELECT MAX(model_run) FROM forecasts
               WHERE station_id = ? AND valid_at >= ? AND valid_at < ?
               AND model_name = 'hrrr'""",
            [station_id, day_start_utc, day_end_utc],
        ).fetchone()[0]

        if latest_mr:
            prior_rows = con.execute(
                """SELECT model_run, valid_at, temp_f FROM forecasts
                   WHERE station_id = ?
                   AND valid_at >= ? AND valid_at < ?
                   AND model_run != ?
                   AND model_name = 'hrrr'
                   ORDER BY model_run DESC, valid_at ASC""",
                [station_id, day_start_utc, day_end_utc, latest_mr],
            ).fetchall()

            prior_by_run = defaultdict(list)
            for mr, va, tf in prior_rows:
                if tf is not None:
                    prior_by_run[mr].append({
                        "valid_at": va.isoformat(), "temp_f": round(tf, 1),
                    })
            prior_runs = [
                {"model_run": mr.isoformat(), "points": pts}
                for mr, pts in sorted(prior_by_run.items(), reverse=True)
            ][:5]
    else:
        # Live mode: prior runs overlapping the current forecast window
        latest_mr = con.execute(
            "SELECT MAX(model_run) FROM forecasts WHERE station_id = ? AND model_name = 'hrrr'",
            [station_id],
        ).fetchone()[0]
        if latest_mr:
            first_valid = forecasts[0][0]
            prior_rows = con.execute(
                """SELECT model_run, valid_at, temp_f FROM forecasts
                   WHERE station_id = ? AND model_run != ?
                   AND valid_at >= ?
                   AND model_name = 'hrrr'
                   ORDER BY model_run DESC, valid_at ASC""",
                [station_id, latest_mr, first_valid],
            ).fetchall()

            prior_by_run = defaultdict(list)
            for mr, va, tf in prior_rows:
                if tf is not None:
                    prior_by_run[mr].append({
                        "valid_at": va.isoformat(), "temp_f": round(tf, 1),
                    })
            sorted_runs = sorted(prior_by_run.items(), reverse=True)[:5]
            prior_runs = [
                {"model_run": mr.isoformat(), "points": pts}
                for mr, pts in sorted_runs
            ]

    # Observations — settlement station only (include 6-hour synoptic max)
    if day_start_utc:
        observations = con.execute(
            """SELECT observed_at, temp_f, six_hr_max_c, ingested_at FROM observations
               WHERE station_id = ?
               AND observed_at >= ? AND observed_at < ?
               ORDER BY observed_at""",
            [station_id, day_start_utc, day_end_utc],
        ).fetchall()
    else:
        first_valid = forecasts[0][0]
        observations = con.execute(
            """SELECT observed_at, temp_f, six_hr_max_c, ingested_at FROM observations
               WHERE station_id = ?
               AND observed_at >= ?
               ORDER BY observed_at""",
            [station_id, first_valid],
        ).fetchall()

    obs_points = []
    six_hr_maxes = []
    running_high_f = None
    running_high_at = None
    for row in observations:
        observed_at_ts, temp_f, six_hr_max_c, obs_ingested = row
        obs_ingested_iso = obs_ingested.isoformat() if obs_ingested else None
        if temp_f is not None:
            obs_points.append({
                "observed_at": observed_at_ts.isoformat(), "temp_f": round(temp_f, 1),
                "ingested_at": obs_ingested_iso,
            })
            if running_high_f is None or temp_f > running_high_f:
                running_high_f = temp_f
                running_high_at = observed_at_ts.isoformat()

        # 6-hour max may exceed the snapshot temp — track it for running high
        if six_hr_max_c is not None:
            six_hr_max_f = round(six_hr_max_c * 9.0 / 5.0 + 32.0, 1)
            six_hr_maxes.append({
                "observed_at": observed_at_ts.isoformat(), "temp_f": six_hr_max_f,
                "ingested_at": obs_ingested_iso,
            })
            if running_high_f is None or six_hr_max_f > running_high_f:
                running_high_f = six_hr_max_f
                running_high_at = observed_at_ts.isoformat()

    # Neighbor station observations (for overlay toggle)
    neighbors = CITIES[city].get("neighbors", [])
    neighbor_obs = {}
    for nbr_id in neighbors:
        if day_start_utc:
            nbr_rows = con.execute(
                """SELECT observed_at, temp_f, ingested_at FROM observations
                   WHERE station_id = ?
                   AND observed_at >= ? AND observed_at < ?
                   ORDER BY observed_at""",
                [nbr_id, day_start_utc, day_end_utc],
            ).fetchall()
        else:
            nbr_rows = con.execute(
                """SELECT observed_at, temp_f, ingested_at FROM observations
                   WHERE station_id = ?
                   AND observed_at >= ?
                   ORDER BY observed_at""",
                [nbr_id, first_valid],
            ).fetchall()
        neighbor_obs[nbr_id] = [
            {
                "observed_at": r[0].isoformat(),
                "temp_f": round(r[1], 1),
                "ingested_at": r[2].isoformat() if r[2] else None,
            }
            for r in nbr_rows if r[1] is not None
        ]

    # Settlement high: prefer NWS daily (CLI thermometer) over running obs max
    observed_high = None
    observed_high_at = None
    settlement_source = None

    # Determine target date for NWS lookup
    if date:
        nws_date = date
    else:
        from core.timezone import ET as _ET
        nws_date = datetime.now(_ET).strftime("%Y-%m-%d")

    nws_row = con.execute(
        "SELECT max_temp_f FROM nws_daily WHERE station_id = ? AND obs_date = ?",
        [station_id, nws_date],
    ).fetchone()

    if nws_row and nws_row[0] is not None:
        observed_high = round(nws_row[0], 1)
        # Place NWS settlement dot at noon ET for display purposes
        observed_high_at = f"{nws_date}T17:00:00"  # noon ET = 17:00 UTC
        settlement_source = "nws_cli"
    elif running_high_f is not None:
        # Fallback: running max of settlement-station obs (including 6-hour maxes)
        observed_high = round(running_high_f, 1)
        observed_high_at = running_high_at
        settlement_source = "obs_running"

    # Get latest drift for this city
    drift_row = con.execute(
        """SELECT drift_score FROM drift_signals
           WHERE city = ? ORDER BY calculated_at DESC LIMIT 1""",
        [city],
    ).fetchone()
    drift = round(drift_row[0], 2) if drift_row else 0.0

    # Historical bias for this station
    bias_val = round(bias.mean_bias, 2) if bias else 0.0

    # Bias adjustment (same formula as ProbabilityEngine line 212)
    adjustment = round(-bias_val + drift, 2)
    bias_adjusted_points = [
        {"valid_at": p["valid_at"], "temp_f": round(p["temp_f"] + adjustment, 1)}
        for p in forecast_points
    ]

    con.close()
    return {
        "city": city,
        "model_run": latest_mr.isoformat() if latest_mr else None,
        "forecasts": forecast_points,
        "observations": obs_points,
        "six_hr_maxes": six_hr_maxes,
        "ribbon": ribbon,
        "prior_runs": prior_runs,
        "bias_adjusted_forecasts": bias_adjusted_points,
        "adjustment": adjustment,
        "drift": drift,
        "bias": bias_val,
        "now_utc": now.isoformat(),
        "observed_high": observed_high,
        "observed_high_at": observed_high_at,
        "settlement_source": settlement_source,
        "neighbor_obs": neighbor_obs,
    }


@app.get("/api/probability/{city}")
async def probability(city: str):
    """Return CityForecast distribution for a single city."""
    city = city.upper()
    if city not in CITIES:
        return {"error": f"Unknown city: {city}"}

    forecast = engine.calculate_city(city)
    if not forecast:
        return {"city": city, "available": False}

    return {
        "city": forecast.city,
        "center": forecast.center,
        "std": forecast.std,
        "interval_90": list(forecast.interval_90),
        "bracket_probs": forecast.bracket_probs,
    }


@app.get("/api/cities")
async def cities_summary():
    """Return summary stats for all 5 cities."""
    summaries = []
    con = get_connection()

    for city in CITIES:
        forecast = engine.calculate_city(city)

        # Get latest drift
        drift_row = con.execute(
            """SELECT drift_score FROM drift_signals
               WHERE city = ? ORDER BY calculated_at DESC LIMIT 1""",
            [city],
        ).fetchone()
        drift = drift_row[0] if drift_row else 0.0

        if forecast:
            summaries.append({
                "city": city,
                "center": forecast.center,
                "std": forecast.std,
                "drift": round(drift, 2),
                "confidence": round(max(0.0, min(1.0, 1.0 - forecast.std / 3.0)), 2),
                "interval_90": list(forecast.interval_90),
            })
        else:
            summaries.append({
                "city": city,
                "center": None,
                "std": None,
                "drift": round(drift, 2),
                "confidence": None,
                "interval_90": None,
            })

    con.close()
    return {"cities": summaries}


@app.get("/api/market/{city}")
async def market_comparison(city: str, date: str = None):
    """Compare model bracket probabilities against Kalshi market prices.

    Optional date param (YYYY-MM-DD) filters to that day's markets.
    Defaults to today in ET (UTC-5).
    """
    city = city.upper()
    if city not in CITIES:
        return {"error": f"Unknown city: {city}"}

    con = get_connection()

    # Determine which date's markets to show (DST-aware)
    from core.timezone import ET as _ET
    if date:
        try:
            target = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            target = datetime.now(_ET)
    else:
        target = datetime.now(_ET)

    # Kalshi tickers embed dates as YYMMMDD (e.g. 26FEB23)
    kalshi_date = target.strftime("%y%b%d").upper()

    # Get latest market ticks for this city + date, with bracket bounds
    ticks = con.execute(
        """SELECT market_id, yes_bid, yes_ask, last_trade, floor_strike, cap_strike
           FROM market_ticks
           WHERE city = ?
           AND market_id LIKE ?
           AND captured_at = (
               SELECT MAX(captured_at) FROM market_ticks
               WHERE city = ? AND market_id LIKE ?
           )
           ORDER BY COALESCE(floor_strike, -999), COALESCE(cap_strike, 999)""",
        [city, f"%{kalshi_date}%", city, f"%{kalshi_date}%"],
    ).fetchall()

    if not ticks:
        con.close()
        return {"city": city, "available": False, "message": "Market data pending"}

    # Get model probabilities
    forecast = engine.calculate_city(city)
    if not forecast:
        con.close()
        return {"city": city, "available": False, "message": "No forecast available"}

    comparisons = []
    for market_id, yes_bid, yes_ask, last_trade, floor_strike, cap_strike in ticks:
        # Kalshi bracket interpretation:
        #   Bottom tail (floor=None, cap=X): resolves YES if high < X → covers ≤(X-1)
        #   Range (floor=X, cap=Y): resolves YES if X ≤ high ≤ Y
        #   Top tail (cap=None, floor=X): resolves YES if high > X → covers ≥(X+1)
        if floor_strike is None and cap_strike is not None:
            label_bound = int(cap_strike) - 1
            bracket_label = f"≤{label_bound}"
            model_prob = sum(p for k, p in forecast.bracket_probs.items() if k <= label_bound)
        elif cap_strike is None and floor_strike is not None:
            label_bound = int(floor_strike) + 1
            bracket_label = f"≥{label_bound}"
            model_prob = sum(p for k, p in forecast.bracket_probs.items() if k >= label_bound)
        elif floor_strike is not None and cap_strike is not None:
            bracket_label = f"{int(floor_strike)}–{int(cap_strike)}"
            model_prob = sum(p for k, p in forecast.bracket_probs.items()
                            if floor_strike <= k <= cap_strike)
        else:
            continue

        model_prob = round(model_prob, 4)
        market_mid = ((yes_bid or 0) + (yes_ask or 0)) / 2.0 if yes_bid and yes_ask else None
        edge = round(model_prob - market_mid, 4) if market_mid is not None else None

        signal = "neutral"
        if edge is not None:
            if edge > 0.05:
                signal = "alpha"
            elif edge < -0.05:
                signal = "overpriced"

        comparisons.append({
            "bracket": bracket_label,
            "model_prob": model_prob,
            "market_mid": round(market_mid, 4) if market_mid is not None else None,
            "edge": edge,
            "signal": signal,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
        })

    con.close()
    return {"city": city, "available": True, "comparisons": comparisons}


@app.get("/api/brackets/{city}")
async def bracket_spread(city: str, date: str = None):
    """Compare model bracket probabilities vs Kalshi market prices in 2°F buckets.

    Maps 1°F model probs to 2°F Kalshi brackets, merges with latest market ticks,
    and computes edge (model_prob - market_mid) for each bracket.
    """
    city = city.upper()
    if city not in CITIES:
        return {"error": f"Unknown city: {city}"}

    # 1. Get model probabilities — handle engine failures gracefully
    try:
        forecast = engine.calculate_city(city)
    except Exception:
        forecast = None

    model_center = forecast.center if forecast else None
    model_std = forecast.std if forecast else None

    # Map 1°F model probs to 2°F Kalshi brackets: floor = (temp_f // 2) * 2
    model_2f = {}  # type: Dict[tuple, float]
    if forecast and forecast.bracket_probs:
        for temp_f, prob in forecast.bracket_probs.items():
            floor = (temp_f // 2) * 2
            cap = floor + 2
            key = (floor, cap)
            model_2f[key] = model_2f.get(key, 0.0) + prob

    # 2. Get latest Kalshi market ticks for this city + date
    from core.timezone import ET as _ET
    if date:
        try:
            target = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            target = datetime.now(_ET)
    else:
        target = datetime.now(_ET)

    kalshi_date = target.strftime("%y%b%d").upper()
    target_date_str = target.strftime("%Y-%m-%d")

    con = get_connection()
    ticks = con.execute(
        """SELECT market_id, yes_bid, yes_ask, last_trade, floor_strike,
                  cap_strike, volume
           FROM market_ticks
           WHERE city = ?
           AND market_id LIKE ?
           AND captured_at = (
               SELECT MAX(captured_at) FROM market_ticks
               WHERE city = ? AND market_id LIKE ?
           )
           ORDER BY COALESCE(floor_strike, -999), COALESCE(cap_strike, 999)""",
        [city, f"%{kalshi_date}%", city, f"%{kalshi_date}%"],
    ).fetchall()

    # Index market data by (floor, cap) for merging
    market_by_bracket = {}  # type: Dict[tuple, dict]
    for market_id, yes_bid, yes_ask, last_trade, floor_strike, cap_strike, volume in ticks:
        if floor_strike is not None and cap_strike is not None:
            key = (int(floor_strike), int(cap_strike))
            market_by_bracket[key] = {
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "volume": volume or 0,
            }

    con.close()

    # 3. Merge model + market into bracket objects
    all_keys = set(model_2f.keys()) | set(market_by_bracket.keys())
    brackets = []
    total_volume = 0
    spread_sum = 0.0
    spread_count = 0

    for key in sorted(all_keys):
        floor, cap = key
        model_prob = round(model_2f.get(key, 0.0), 4)
        mkt = market_by_bracket.get(key, {})
        yes_bid = mkt.get("yes_bid")
        yes_ask = mkt.get("yes_ask")
        vol = mkt.get("volume", 0)

        market_mid = None
        if yes_bid is not None and yes_ask is not None:
            market_mid = round((yes_bid + yes_ask) / 2.0, 4)

        edge = round(model_prob - market_mid, 4) if market_mid is not None else None

        brackets.append({
            "floor": floor,
            "cap": cap,
            "model_prob": model_prob,
            "market_mid": market_mid,
            "edge": edge,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "volume": vol,
        })

        total_volume += vol
        if yes_bid is not None and yes_ask is not None:
            spread_sum += (yes_ask - yes_bid)
            spread_count += 1

    avg_spread = round(spread_sum / spread_count, 4) if spread_count > 0 else 0.0

    return {
        "city": city,
        "date": target_date_str,
        "brackets": brackets,
        "liquidity": {"total_volume": total_volume, "avg_spread": avg_spread},
        "model_center": model_center,
        "model_std": model_std,
    }


@app.get("/api/positions/{city}")
async def get_positions(city: str, date: str = None):
    """Return active paper positions, near-misses, and P&L for a city.

    Queries paper_positions for open/settled trades on the target date,
    and identifies near-miss brackets (edge 5-10%) from the brackets endpoint.
    """
    city = city.upper()
    if city not in CITIES:
        return {"error": f"Unknown city: {city}"}

    from core.timezone import ET as _ET
    if date:
        try:
            datetime.strptime(date, "%Y-%m-%d")
            target_date = date
        except ValueError:
            target_date = datetime.now(_ET).strftime("%Y-%m-%d")
    else:
        target_date = datetime.now(_ET).strftime("%Y-%m-%d")

    con = get_connection()
    try:
        # Active (open) positions for this city + date
        active_rows = con.execute(
            """SELECT id, bracket_floor, bracket_cap, direction, model_prob,
                      market_price, edge, entry_price, entry_time
               FROM paper_positions
               WHERE city = ? AND event_date = ? AND status = 'open'
               ORDER BY entry_time ASC""",
            [city, target_date],
        ).fetchall()

        active = []
        for row in active_rows:
            pid, floor, cap, direction, model_prob, market_price, edge, entry_price, entry_time = row
            active.append({
                "id": pid,
                "bracket": "{}-{}°F".format(int(floor), int(cap)) if floor is not None and cap is not None else "--",
                "direction": direction,
                "model_prob": round(model_prob, 4) if model_prob is not None else None,
                "market_price": round(market_price, 4) if market_price is not None else None,
                "edge": round(edge, 4) if edge is not None else None,
                "entry_price": round(entry_price, 4) if entry_price is not None else None,
                "entry_time": entry_time.isoformat() if entry_time else None,
            })

        # Daily settled P&L
        daily_row = con.execute(
            """SELECT COALESCE(SUM(net_pnl), 0)
               FROM paper_positions
               WHERE city = ? AND event_date = ? AND status = 'settled'""",
            [city, target_date],
        ).fetchone()
        daily_pnl = round(daily_row[0], 2) if daily_row else 0.0

        # Total settled P&L (all time)
        total_row = con.execute(
            """SELECT COALESCE(SUM(net_pnl), 0)
               FROM paper_positions
               WHERE city = ? AND status = 'settled'""",
            [city],
        ).fetchone()
        total_pnl = round(total_row[0], 2) if total_row else 0.0
    finally:
        con.close()

    # Near-misses: brackets with 5% <= edge < 10% from the brackets endpoint
    near_misses = []
    try:
        brackets_data = await bracket_spread(city, date=target_date)
        threshold = 0.10
        for b in brackets_data.get("brackets", []):
            if b.get("edge") is not None and 0.05 <= b["edge"] < 0.10:
                label = "{}-{}°F".format(b["floor"], b["cap"])
                near_misses.append({
                    "bracket": label,
                    "edge": round(b["edge"], 4),
                    "threshold": threshold,
                })
    except Exception:
        pass  # If brackets fail, just return empty near_misses

    return {
        "city": city,
        "date": target_date,
        "active": active,
        "near_misses": near_misses,
        "daily_pnl": daily_pnl,
        "total_pnl": total_pnl,
    }


@app.get("/api/performance")
async def get_performance(time_range: str = Query("30d", alias="range")):
    """Return P&L data from paper_positions for the Performance tab.

    Query parameter 'range' (aliased to time_range) accepts "7d", "30d", or "all".
    """
    days_map = {"7d": 7, "30d": 30, "all": 9999}
    days = days_map.get(time_range, 30)

    con = get_connection()
    try:
        # Daily P&L, wins, losses, fees, gross for settled positions in range
        # days comes from controlled map — safe to interpolate
        daily_rows = con.execute(
            """SELECT event_date,
                      COALESCE(SUM(net_pnl), 0) AS pnl,
                      COALESCE(SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END), 0) AS wins,
                      COALESCE(SUM(CASE WHEN net_pnl <= 0 THEN 1 ELSE 0 END), 0) AS losses,
                      COALESCE(SUM(fees), 0) AS fees,
                      COALESCE(SUM(gross_pnl), 0) AS gross
               FROM paper_positions
               WHERE status = 'settled'
               AND event_date >= CURRENT_DATE - INTERVAL '{}' DAY
               GROUP BY event_date
               ORDER BY event_date ASC""".format(days),
        ).fetchall()

        # Build daily bars and cumulative P&L
        daily_bars = []
        cumulative_pnl = []
        running = 0.0
        total_wins = 0
        total_losses = 0
        total_gross = 0.0
        total_fees = 0.0
        total_net = 0.0

        for event_date, pnl, wins, losses, fees_val, gross_val in daily_rows:
            date_str = str(event_date)
            daily_bars.append({
                "date": date_str,
                "pnl": round(pnl, 2),
                "wins": int(wins),
                "losses": int(losses),
            })
            running += pnl
            cumulative_pnl.append({
                "date": date_str,
                "pnl": round(running, 2),
            })
            total_wins += int(wins)
            total_losses += int(losses)
            total_gross += gross_val
            total_fees += fees_val
            total_net += pnl

        total_trades = total_wins + total_losses
        win_rate = round(total_wins / total_trades, 4) if total_trades > 0 else 0.0

        # Breakdown by bracket
        bracket_rows = con.execute(
            """SELECT bracket_floor, bracket_cap,
                      COALESCE(SUM(net_pnl), 0) AS pnl,
                      COUNT(*) AS trades,
                      COALESCE(SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END), 0) AS wins
               FROM paper_positions
               WHERE status = 'settled'
               AND event_date >= CURRENT_DATE - INTERVAL '{}' DAY
               GROUP BY bracket_floor, bracket_cap
               ORDER BY bracket_floor ASC""".format(days),
        ).fetchall()

        by_bracket = []
        for floor, cap, pnl, trades, wins in bracket_rows:
            bracket_label = "{}-{}".format(int(floor), int(cap)) if floor is not None and cap is not None else "--"
            wr = round(wins / trades, 4) if trades > 0 else 0.0
            by_bracket.append({
                "bracket": bracket_label,
                "pnl": round(pnl, 2),
                "trades": int(trades),
                "win_rate": wr,
            })

        # Breakdown by edge bucket
        edge_rows = con.execute(
            """SELECT
                   CASE
                       WHEN edge >= 0.15 THEN '>15%'
                       WHEN edge >= 0.10 THEN '10-15%'
                       WHEN edge >= 0.05 THEN '5-10%'
                       ELSE '<5%'
                   END AS bucket,
                   COALESCE(SUM(net_pnl), 0) AS pnl,
                   COUNT(*) AS trades,
                   COALESCE(SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END), 0) AS wins
               FROM paper_positions
               WHERE status = 'settled'
               AND event_date >= CURRENT_DATE - INTERVAL '{}' DAY
               GROUP BY bucket
               ORDER BY bucket DESC""".format(days),
        ).fetchall()

        by_edge = []
        for bucket, pnl, trades, wins in edge_rows:
            wr = round(wins / trades, 4) if trades > 0 else 0.0
            by_edge.append({
                "bucket": bucket,
                "pnl": round(pnl, 2),
                "trades": int(trades),
                "win_rate": wr,
            })
    finally:
        con.close()

    return {
        "range": time_range,
        "cumulative_pnl": cumulative_pnl,
        "daily_bars": daily_bars,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "total_gross": round(total_gross, 2),
        "total_fees": round(total_fees, 2),
        "total_net": round(total_net, 2),
        "by_bracket": by_bracket,
        "by_edge": by_edge,
    }


@app.get("/api/brier-comparison")
async def get_brier_comparison(time_range: str = Query("30d", alias="range")):
    """Brier score comparison — stub until backtester integration is ready."""
    return {
        "range": time_range,
        "by_hour": [],
        "note": "Brier comparison requires backtester integration (in progress on separate worktree)",
    }


@app.get("/api/edge-heatmap")
async def get_edge_heatmap(time_range: str = Query("30d", alias="range")):
    """Edge heatmap — settled positions grouped by bracket and hour (ET)."""
    days_map = {"7d": 7, "30d": 30, "all": 9999}
    days = days_map.get(time_range, 30)

    con = get_connection()
    try:
        cells_rows = con.execute(
            """SELECT bracket_floor, bracket_cap,
                      EXTRACT(HOUR FROM timezone('America/New_York', entry_time)) AS hour_et,
                      AVG(edge) AS avg_edge,
                      COUNT(*) AS trades,
                      COALESCE(SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END), 0) AS wins
               FROM paper_positions
               WHERE status = 'settled'
               AND event_date >= CURRENT_DATE - INTERVAL '{}' DAY
               GROUP BY bracket_floor, bracket_cap, hour_et
               ORDER BY bracket_floor ASC, hour_et ASC""".format(days),
        ).fetchall()

        cells = []
        for floor, cap, hour_et, avg_edge, trades, wins in cells_rows:
            bracket_label = "{}-{}".format(int(floor), int(cap)) if floor is not None and cap is not None else "--"
            wr = round(wins / trades, 4) if trades > 0 else 0.0
            cells.append({
                "bracket": bracket_label,
                "hour_et": int(hour_et) if hour_et is not None else None,
                "avg_edge": round(avg_edge, 4) if avg_edge is not None else None,
                "trades": int(trades),
                "win_rate": wr,
            })
    finally:
        con.close()

    return {
        "range": time_range,
        "cells": cells,
    }


@app.get("/api/market-swings/{city}")
async def market_swings(city: str, date: str = None):
    """Detect material Kalshi price movements (>10c in <2 hours).

    Scans market_ticks for the target date, groups by bracket, and finds
    the largest mid-price swing per bracket within any 2-hour window.
    """
    from collections import defaultdict

    city = city.upper()
    if city not in CITIES:
        return {"error": f"Unknown city: {city}"}

    from core.timezone import ET as _ET
    if date:
        try:
            datetime.strptime(date, "%Y-%m-%d")
            target_date = date
        except ValueError:
            target_date = datetime.now(_ET).strftime("%Y-%m-%d")
    else:
        target_date = datetime.now(_ET).strftime("%Y-%m-%d")

    day_start_utc, day_end_utc = et_day_bounds_utc(target_date)

    con = get_connection()
    try:
        rows = con.execute(
            """SELECT floor_strike, cap_strike, yes_bid, yes_ask, captured_at
               FROM market_ticks
               WHERE city = ?
               AND captured_at >= ? AND captured_at < ?
               AND floor_strike IS NOT NULL AND cap_strike IS NOT NULL
               ORDER BY floor_strike, cap_strike, captured_at ASC""",
            [city, day_start_utc, day_end_utc],
        ).fetchall()
    finally:
        con.close()

    # Group ticks by bracket
    bracket_ticks = defaultdict(list)  # type: dict
    for floor_strike, cap_strike, yes_bid, yes_ask, captured_at in rows:
        # Compute mid-price, handling NULL bid/ask
        if yes_bid is not None and yes_ask is not None:
            mid = (yes_bid + yes_ask) / 2.0
        elif yes_bid is not None:
            mid = yes_bid
        elif yes_ask is not None:
            mid = yes_ask
        else:
            continue  # Skip ticks with no price data

        key = (int(floor_strike), int(cap_strike))
        bracket_ticks[key].append({"mid": mid, "time": captured_at})

    # Find largest swing per bracket within 2-hour windows
    swings = []
    two_hours_secs = 2 * 3600

    for (floor, cap), ticks in bracket_ticks.items():
        if len(ticks) < 2:
            continue

        best_swing = None
        best_abs_change = 0.0

        for i in range(len(ticks)):
            for j in range(i + 1, len(ticks)):
                t_i = ticks[i]["time"]
                t_j = ticks[j]["time"]
                # Ensure both timestamps are comparable
                if hasattr(t_i, 'timestamp') and hasattr(t_j, 'timestamp'):
                    delta_secs = abs((t_j - t_i).total_seconds())
                else:
                    continue

                if delta_secs > two_hours_secs:
                    break  # Ticks are sorted by time, so no need to check further

                change = ticks[j]["mid"] - ticks[i]["mid"]
                abs_change = abs(change)

                if abs_change > 0.10 and abs_change > best_abs_change:
                    best_abs_change = abs_change
                    best_swing = {
                        "bracket": "{}-{}°F".format(floor, cap),
                        "from_price": round(ticks[i]["mid"], 4),
                        "to_price": round(ticks[j]["mid"], 4),
                        "change": round(change, 4),
                        "from_time": t_i.isoformat() if hasattr(t_i, 'isoformat') else str(t_i),
                        "to_time": t_j.isoformat() if hasattr(t_j, 'isoformat') else str(t_j),
                    }

        if best_swing:
            swings.append(best_swing)

    # Sort by absolute change descending, top 10
    swings.sort(key=lambda s: abs(s["change"]), reverse=True)
    swings = swings[:10]

    return {
        "city": city,
        "date": target_date,
        "swings": swings,
    }


# ---------------------------------------------------------------------------
# Review (Model Autopsy) helpers
# ---------------------------------------------------------------------------


def _categorize_incident(pos_row, settlement_temp):
    # type: (dict, float) -> str
    """Classify why a position lost based on settlement temp vs bracket.

    Categories:
    - tail_bracket_underweight: settlement >4F away from bracket range
    - threshold_too_conservative: model and market were close (edge < 5%)
    - model_miss: generic model error (default)
    - unknown: no settlement data available
    """
    if settlement_temp is None:
        return "unknown"

    bracket_floor = pos_row.get("bracket_floor")
    bracket_cap = pos_row.get("bracket_cap")

    if bracket_floor is not None and bracket_cap is not None:
        # Distance from settlement to nearest bracket edge
        if settlement_temp > bracket_cap:
            distance = settlement_temp - bracket_cap
        elif settlement_temp < bracket_floor:
            distance = bracket_floor - settlement_temp
        else:
            distance = 0
        if distance > 4:
            return "tail_bracket_underweight"

    model_prob = pos_row.get("model_prob") or 0
    market_price = pos_row.get("market_price") or 0
    if abs(model_prob - market_price) < 0.05:
        return "threshold_too_conservative"

    return "model_miss"


def _narrate_incident(incident_type, pos_row, settlement_temp):
    # type: (str, dict, float) -> str
    """Generate a short human-readable narrative for an incident."""
    direction = pos_row.get("direction", "?")
    bracket_floor = pos_row.get("bracket_floor")
    bracket_cap = pos_row.get("bracket_cap")
    entry_price = pos_row.get("entry_price")

    bracket_label = "{}-{}°F".format(
        int(bracket_floor), int(bracket_cap)
    ) if bracket_floor is not None and bracket_cap is not None else "?"

    entry_cents = "{}¢".format(int(round(entry_price * 100))) if entry_price is not None else "?¢"

    if incident_type == "lost_bet":
        # Determine if bracket settled YES or NO
        settled = "NO"
        if settlement_temp is not None and bracket_floor is not None and bracket_cap is not None:
            if bracket_floor <= settlement_temp <= bracket_cap:
                settled = "YES"
        temp_str = "{}°F".format(int(round(settlement_temp))) if settlement_temp is not None else "unknown"
        return "Bet {} on {} at {}. Bracket settled {}. Settlement temp: {}.".format(
            direction, bracket_label, entry_cents, settled, temp_str
        )

    return "Lost position on {}.".format(bracket_label)


# ---------------------------------------------------------------------------
# Review (Model Autopsy) endpoints
# ---------------------------------------------------------------------------


@app.get("/api/review/incidents")
async def get_review_incidents(time_range: str = Query("30d", alias="range"),
                                filter_type: str = Query("all", alias="filter")):
    """Return incident cards for the Review (Model Autopsy) tab.

    Compares model predictions vs settlements to identify lost bets
    and categorize what went wrong.
    """
    days_map = {"7d": 7, "30d": 30, "all": 9999}
    days = days_map.get(time_range, 30)

    con = get_connection()
    try:
        # Settled positions with net_pnl < 0 in date range (lost bets)
        pos_rows = con.execute(
            """SELECT event_date, bracket_floor, bracket_cap, direction,
                      model_prob, market_price, edge, entry_price, net_pnl
               FROM paper_positions
               WHERE status = 'settled'
               AND net_pnl < 0
               AND event_date >= CURRENT_DATE - INTERVAL '{}' DAY
               ORDER BY event_date DESC""".format(days),
        ).fetchall()

        # Get settlement temps from nws_daily for KNYC
        # Collect unique dates from positions
        dates = list(set(str(row[0]) for row in pos_rows))
        settlement_temps = {}  # type: dict
        if dates:
            nws_rows = con.execute(
                """SELECT obs_date, max_temp_f FROM nws_daily
                   WHERE station_id = 'KNYC'
                   AND obs_date IN ({})""".format(
                    ", ".join("'{}'".format(d) for d in dates)
                ),
            ).fetchall()
            for obs_date, max_temp_f in nws_rows:
                settlement_temps[str(obs_date)] = max_temp_f
    finally:
        con.close()

    # Build incident cards
    incidents = []
    # Track max abs pnl for severity normalization
    max_abs_pnl = max((abs(row[8]) for row in pos_rows), default=1.0)
    if max_abs_pnl == 0:
        max_abs_pnl = 1.0

    for row in pos_rows:
        event_date, bracket_floor, bracket_cap, direction, model_prob, \
            market_price, edge, entry_price, net_pnl = row

        date_str = str(event_date)
        settlement_temp = settlement_temps.get(date_str)

        pos_dict = {
            "bracket_floor": bracket_floor,
            "bracket_cap": bracket_cap,
            "direction": direction,
            "model_prob": model_prob,
            "market_price": market_price,
            "edge": edge,
            "entry_price": entry_price,
        }

        category = _categorize_incident(pos_dict, settlement_temp)
        # Severity: normalized abs(pnl) clamped to [0, 1]
        severity = round(min(1.0, abs(net_pnl) / max_abs_pnl), 2)

        incident_type = "lost_bet"
        narrative = _narrate_incident(incident_type, pos_dict, settlement_temp)

        bracket_label = "{}-{}°F".format(
            int(bracket_floor), int(bracket_cap)
        ) if bracket_floor is not None and bracket_cap is not None else "--"

        incidents.append({
            "date": date_str,
            "type": incident_type,
            "severity": severity,
            "bracket": bracket_label,
            "direction": direction,
            "model_prob": round(model_prob, 4) if model_prob is not None else None,
            "market_price": round(market_price, 4) if market_price is not None else None,
            "edge": round(edge, 4) if edge is not None else None,
            "net_pnl": round(net_pnl, 2),
            "settlement_temp": settlement_temp,
            "category": category,
            "narrative": narrative,
        })

    # Apply filter
    if filter_type == "worst":
        incidents = [i for i in incidents if i["severity"] >= 0.5]
    elif filter_type == "lost":
        incidents = [i for i in incidents if i["type"] == "lost_bet"]
    elif filter_type == "missed":
        incidents = [i for i in incidents if i["type"] == "missed_edge"]
    # "all" — no filtering

    # Sort by severity descending, limit to 50
    incidents.sort(key=lambda i: i["severity"], reverse=True)
    incidents = incidents[:50]

    return {
        "range": time_range,
        "filter": filter_type,
        "incidents": incidents,
    }


@app.get("/api/review/patterns")
async def get_review_patterns(time_range: str = Query("30d", alias="range")):
    """Aggregate failure patterns from review incidents.

    Groups incidents by category, sums P&L, and maps each to a suggested action.
    """
    # Reuse the incidents endpoint internally
    incidents_resp = await get_review_incidents(time_range=time_range, filter_type="all")
    incidents = incidents_resp["incidents"]

    # Known remedies per category
    remedies = {
        "slow_drift_response": "Consider dynamic std that widens when obs drift > 2°F",
        "tail_bracket_underweight": "Review tail bracket calibration (model underweights >2\u03c3)",
        "threshold_too_conservative": "Backtest threshold at lower values (e.g., 8% vs 10%)",
        "stale_pricing": "Add stale-price detection to trigger model re-evaluation",
        "model_miss": "Review model accuracy for these conditions in backtester",
        "unknown": "Insufficient settlement data to categorize — check NWS ingestion",
    }

    # Group by category
    grouped = {}  # type: dict
    for inc in incidents:
        cat = inc["category"]
        if cat not in grouped:
            grouped[cat] = {"count": 0, "total_pnl": 0.0}
        grouped[cat]["count"] += 1
        grouped[cat]["total_pnl"] += inc["net_pnl"]

    patterns = []
    for cat, agg in grouped.items():
        patterns.append({
            "category": cat,
            "count": agg["count"],
            "total_pnl": round(agg["total_pnl"], 2),
            "suggested_action": remedies.get(cat, "Investigate manually"),
        })

    # Sort by total_pnl ascending (worst first)
    patterns.sort(key=lambda p: p["total_pnl"])

    return {
        "range": time_range,
        "patterns": patterns,
    }
