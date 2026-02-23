"""AlphaTemp Web Dashboard — FastAPI backend serving Plotly + Tailwind frontend."""

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from core.constants import CITIES, STATION_COORDS
from core.db import get_connection, init_db
from core.timezone import et_day_bounds_utc
from services.probability import ProbabilityEngine

app = FastAPI(title="AlphaTemp Command Center")

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


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})


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
            "SELECT MAX(ingested_at) FROM forecasts"
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


@app.get("/api/observations/{city}")
async def observation_feed(city: str, date: str = None):
    """Raw observation feed for settlement + neighbor stations.

    Returns observations for the given day (midnight–midnight ET), newest first.
    Defaults to today in ET.  Running high is settlement-station only (Kalshi settles
    on the settlement ICAO, not neighbors).
    """
    city = city.upper()
    if city not in CITIES:
        return {"error": f"Unknown city: {city}"}

    station_id = CITIES[city]["settlement"]
    all_stations = [station_id] + CITIES[city].get("neighbors", [])
    placeholders = ", ".join("?" for _ in all_stations)

    day_start_utc, day_end_utc = et_day_bounds_utc(date)

    con = get_connection()
    rows = con.execute(
        f"""SELECT station_id, observed_at, temp_f, temp_c_tenth, raw_metar, ingested_at
            FROM observations
            WHERE station_id IN ({placeholders})
            AND observed_at >= ? AND observed_at < ?
            ORDER BY observed_at DESC
            LIMIT 500""",
        [*all_stations, day_start_utc, day_end_utc],
    ).fetchall()

    observations = []
    running_high = None
    running_high_station = None
    for sid, obs_at, temp_f, temp_c, metar, ingested in rows:
        if temp_f is not None:
            observations.append({
                "station_id": sid,
                "observed_at": obs_at.isoformat(),
                "temp_f": round(temp_f, 1),
                "temp_c_tenth": round(temp_c, 1) if temp_c is not None else None,
                "raw_metar": metar,
                "ingested_at": ingested.isoformat() if ingested else None,
            })
            # Running high: settlement station only (Kalshi settles on settlement ICAO)
            if sid == station_id and (running_high is None or temp_f > running_high):
                running_high = temp_f
                running_high_station = sid

    con.close()
    return {
        "city": city,
        "station_id": station_id,
        "observations": observations,
        "running_high": round(running_high, 1) if running_high is not None else None,
        "running_high_station": running_high_station,
    }


@app.get("/api/forecast-points/{city}")
async def forecast_point_feed(city: str, date: str = None):
    """Forecast point feed — individual forecast temps grouped by model run.

    Returns points for the given day (midnight–midnight ET), newest model run first.
    Defaults to today in ET.
    """
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
           ORDER BY model_run DESC, valid_at ASC""",
        [station_id, day_start_utc, day_end_utc],
    ).fetchall()

    points = []
    latest_run = None
    forecast_high = None
    for model_run, valid_at, temp_f, ingested in rows:
        if temp_f is None:
            continue
        if latest_run is None:
            latest_run = model_run
        lead_hours = max(0, round((valid_at - model_run).total_seconds() / 3600))
        points.append({
            "model_run": model_run.isoformat(),
            "valid_at": valid_at.isoformat(),
            "temp_f": round(temp_f, 1),
            "lead_hours": lead_hours,
            "ingested_at": ingested.isoformat() if ingested else None,
        })
        if forecast_high is None or temp_f > forecast_high:
            forecast_high = temp_f

    con.close()
    return {
        "city": city,
        "station_id": station_id,
        "points": points,
        "latest_run": latest_run.isoformat() if latest_run else None,
        "forecast_high": round(forecast_high, 1) if forecast_high is not None else None,
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
            """SELECT valid_at, temp_f FROM forecasts
               WHERE station_id = ?
               AND model_run = (
                   SELECT MAX(model_run) FROM forecasts
                   WHERE station_id = ? AND valid_at >= ? AND valid_at < ?
               )
               AND valid_at >= ? AND valid_at < ?
               ORDER BY valid_at""",
            [station_id, station_id, day_start_utc, day_end_utc,
             day_start_utc, day_end_utc],
        ).fetchall()
    else:
        forecasts = con.execute(
            """SELECT valid_at, temp_f FROM forecasts
               WHERE station_id = ?
               AND model_run = (SELECT MAX(model_run) FROM forecasts WHERE station_id = ?)
               ORDER BY valid_at""",
            [station_id, station_id],
        ).fetchall()

    if not forecasts:
        con.close()
        return {"city": city, "forecasts": [], "observations": [],
                "ribbon": [], "prior_runs": []}

    # Build ribbon bounds using the engine's uncertainty model
    bias = engine._bias_cache.get(station_id)
    historical_std = bias.std_error if bias else 2.0

    stability = engine._compute_stability_factor(con, city)
    convergence = engine._compute_convergence_factor(con, station_id)

    ribbon = []
    forecast_points = []
    now = datetime.now(timezone.utc)
    for valid_at, temp_f in forecasts:
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
        })

    # Prior model runs (only when date is provided)
    prior_runs = []
    if day_start_utc:
        latest_mr = con.execute(
            """SELECT MAX(model_run) FROM forecasts
               WHERE station_id = ? AND valid_at >= ? AND valid_at < ?""",
            [station_id, day_start_utc, day_end_utc],
        ).fetchone()[0]

        if latest_mr:
            prior_rows = con.execute(
                """SELECT model_run, valid_at, temp_f FROM forecasts
                   WHERE station_id = ?
                   AND valid_at >= ? AND valid_at < ?
                   AND model_run != ?
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
            ]

    # Observations — settlement + neighbor stations for denser coverage
    all_stations = [station_id] + CITIES[city].get("neighbors", [])
    obs_placeholders = ", ".join("?" for _ in all_stations)
    if day_start_utc:
        observations = con.execute(
            f"""SELECT station_id, observed_at, temp_f FROM observations
               WHERE station_id IN ({obs_placeholders})
               AND observed_at >= ? AND observed_at < ?
               ORDER BY observed_at""",
            [*all_stations, day_start_utc, day_end_utc],
        ).fetchall()
    else:
        first_valid = forecasts[0][0]
        observations = con.execute(
            f"""SELECT station_id, observed_at, temp_f FROM observations
               WHERE station_id IN ({obs_placeholders})
               AND observed_at >= ?
               ORDER BY observed_at""",
            [*all_stations, first_valid],
        ).fetchall()

    obs_points = [
        {"station_id": row[0], "observed_at": row[1].isoformat(),
         "temp_f": round(row[2], 1)}
        for row in observations if row[2] is not None
    ]

    # Get latest drift for this city
    drift_row = con.execute(
        """SELECT drift_score FROM drift_signals
           WHERE city = ? ORDER BY calculated_at DESC LIMIT 1""",
        [city],
    ).fetchone()
    drift = round(drift_row[0], 2) if drift_row else 0.0

    # Historical bias for this station
    bias_val = round(bias.mean_bias, 2) if bias else 0.0

    con.close()
    return {
        "city": city,
        "forecasts": forecast_points,
        "observations": obs_points,
        "ribbon": ribbon,
        "prior_runs": prior_runs,
        "drift": drift,
        "bias": bias_val,
        "now_utc": now.isoformat(),
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
