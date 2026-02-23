"""AlphaTemp Web Dashboard — FastAPI backend serving Plotly + Tailwind frontend."""

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from core.constants import CITIES, STATION_COORDS
from core.db import get_connection, init_db
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


@app.get("/api/health")
async def health():
    """Basic health check — verifies DB connectivity."""
    try:
        con = get_connection()
        con.execute("SELECT 1")
        con.close()
        db_ok = True
    except Exception:
        db_ok = False

    return {
        "status": "ok" if db_ok else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "db_connected": db_ok,
    }


@app.get("/api/forecast-curve/{city}")
async def forecast_curve(city: str):
    """Return HRRR forecast curve + observations + confidence ribbon bounds."""
    city = city.upper()
    if city not in CITIES:
        return {"error": f"Unknown city: {city}"}

    station_id = CITIES[city]["settlement"]
    con = get_connection()

    # Latest model run forecasts
    forecasts = con.execute(
        """SELECT valid_at, temp_f FROM forecasts
           WHERE station_id = ?
           AND model_run = (SELECT MAX(model_run) FROM forecasts WHERE station_id = ?)
           ORDER BY valid_at""",
        [station_id, station_id],
    ).fetchall()

    if not forecasts:
        con.close()
        return {"city": city, "forecasts": [], "observations": [], "ribbon": []}

    # Build ribbon bounds using the engine's uncertainty model
    bias = engine._bias_cache.get(station_id)
    historical_std = bias.std_error if bias else 2.0

    stability = engine._compute_stability_factor(con, city)
    convergence = engine._compute_convergence_factor(con, station_id)

    ribbon = []
    forecast_points = []
    for valid_at, temp_f in forecasts:
        if valid_at.tzinfo is None:
            ref = valid_at.replace(tzinfo=timezone.utc)
        else:
            ref = valid_at
        time_factor = engine._compute_time_factor(ref)
        std_at_hour = max(0.3, historical_std * time_factor * stability * convergence)
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

    # Recent observations for this station
    first_valid = forecasts[0][0]
    observations = con.execute(
        """SELECT observed_at, temp_f FROM observations
           WHERE station_id = ? AND observed_at >= ?
           ORDER BY observed_at""",
        [station_id, first_valid],
    ).fetchall()

    obs_points = [
        {"observed_at": row[0].isoformat(), "temp_f": round(row[1], 1)}
        for row in observations if row[1] is not None
    ]

    con.close()
    return {
        "city": city,
        "forecasts": forecast_points,
        "observations": obs_points,
        "ribbon": ribbon,
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
                "confidence": round(1.0 - (forecast.std / 3.0), 2),  # Normalized 0-1
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
async def market_comparison(city: str):
    """Compare model bracket probabilities against Kalshi market prices."""
    city = city.upper()
    if city not in CITIES:
        return {"error": f"Unknown city: {city}"}

    con = get_connection()

    # Check if market_ticks has any data for this city
    count_row = con.execute(
        "SELECT COUNT(*) FROM market_ticks WHERE city = ?", [city]
    ).fetchone()

    if not count_row or count_row[0] == 0:
        con.close()
        return {"city": city, "available": False, "message": "Market data pending"}

    # Get latest market ticks
    ticks = con.execute(
        """SELECT market_id, yes_bid, yes_ask, last_trade
           FROM market_ticks
           WHERE city = ?
           AND captured_at = (SELECT MAX(captured_at) FROM market_ticks WHERE city = ?)
           ORDER BY market_id""",
        [city, city],
    ).fetchall()

    # Get model probabilities
    forecast = engine.calculate_city(city)
    if not forecast:
        con.close()
        return {"city": city, "available": False, "message": "No forecast available"}

    comparisons = []
    for market_id, yes_bid, yes_ask, last_trade in ticks:
        # Extract bracket from market_id (e.g., "KTEMP-NYC-24FEB26-B75" -> 75)
        try:
            bracket = int(market_id.split("-B")[-1])
        except (ValueError, IndexError):
            continue

        model_prob = forecast.bracket_probs.get(bracket, 0.0)
        market_mid = ((yes_bid or 0) + (yes_ask or 0)) / 2.0 if yes_bid and yes_ask else None
        edge = round(model_prob - market_mid, 4) if market_mid is not None else None

        signal = "neutral"
        if edge is not None:
            if edge > 0.05:
                signal = "alpha"
            elif edge < -0.05:
                signal = "overpriced"

        comparisons.append({
            "bracket": bracket,
            "model_prob": round(model_prob, 4),
            "market_mid": round(market_mid, 4) if market_mid is not None else None,
            "edge": edge,
            "signal": signal,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
        })

    con.close()
    return {"city": city, "available": True, "comparisons": comparisons}
