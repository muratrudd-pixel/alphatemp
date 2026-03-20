"""AlphaTemp Web Dashboard — FastAPI backend serving Plotly + Tailwind frontend."""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import json as _json
from typing import Dict, Optional, Tuple

from core.constants import CITIES, STATION_COORDS
from core.db import get_connection, init_db
from core.timezone import et_day_bounds_utc, get_today_et

app = FastAPI(title="AlphaTemp Command Center")

# Static files — directory lives at project root alongside ui/
_static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.on_event("startup")
async def startup():
    """Ensure DB tables exist."""
    init_db()


# ---------------------------------------------------------------------------
# Helpers — QR model state from DB
# ---------------------------------------------------------------------------


def _get_model_state(con, city, target_date):
    # type: (...) -> Optional[Tuple[Dict[str, float], float, int]]
    """Read latest bracket_probs from model_state table.

    Returns (bracket_probs_dict, fcst_high, update_hour) or None.
    """
    row = con.execute(
        """SELECT bracket_probs, fcst_high, update_hour FROM model_state
           WHERE city = ? AND target_date = ?""",
        [city, target_date],
    ).fetchone()
    if not row or not row[0]:
        return None
    try:
        probs = _json.loads(row[0])  # type: Dict[str, float]
        # Keys are stringified ints — convert back
        probs_int = {int(k): float(v) for k, v in probs.items()}
        return probs_int, float(row[1]) if row[1] else None, row[2]
    except Exception:
        return None


def _model_expected_value(bracket_probs):
    # type: (Dict[int, float]) -> Optional[float]
    """Compute median temperature from bracket probability distribution.

    Walks the sorted distribution until cumulative probability crosses 0.5.
    Interpolates within the median bracket for a smooth value.
    """
    if not bracket_probs:
        return None
    sorted_floors = sorted(bracket_probs.keys())
    cumulative = 0.0
    for floor in sorted_floors:
        prev_cum = cumulative
        cumulative += bracket_probs[floor]
        if cumulative >= 0.5:
            # Interpolate within this bracket
            frac = (0.5 - prev_cum) / bracket_probs[floor] if bracket_probs[floor] > 0 else 0.5
            return round(floor + frac, 1)
    # Fallback: return center of last bracket
    return round(sorted_floors[-1] + 0.5, 1)


def _model_percentile(bracket_probs, pct):
    # type: (Dict[int, float], float) -> Optional[float]
    """Compute a given percentile from bracket probability distribution."""
    if not bracket_probs:
        return None
    sorted_floors = sorted(bracket_probs.keys())
    cumulative = 0.0
    for floor in sorted_floors:
        prev_cum = cumulative
        cumulative += bracket_probs[floor]
        if cumulative >= pct:
            frac = (pct - prev_cum) / bracket_probs[floor] if bracket_probs[floor] > 0 else 0.5
            return round(floor + frac, 1)
    return round(sorted_floors[-1] + 0.5, 1)


def _get_bias_from_db(con, station_id):
    # type: (...) -> Tuple[float, float]
    """Get latest mean_bias and std_error from station_bias table.

    Returns (mean_bias, std_error) or (0.0, 2.0) if no data.
    """
    row = con.execute(
        """SELECT mean_bias, std_error FROM station_bias
           WHERE station_id = ?
           ORDER BY calculated_at DESC LIMIT 1""",
        [station_id],
    ).fetchone()
    if row:
        return (row[0] or 0.0, row[1] or 2.0)
    return (0.0, 2.0)


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


@app.get("/review")
async def review_page(request: Request):
    """Serve the Review tab."""
    return templates.TemplateResponse("review.html", {"request": request, "active_tab": "review"})


@app.get("/mobile")
async def mobile_page(request: Request):
    """Serve the Mobile tab."""
    return templates.TemplateResponse("mobile.html", {"request": request, "active_tab": "mobile"})


STALE_THRESHOLD_MINUTES = 30


@app.get("/api/nws-cli/{city}")
async def nws_cli_report(city: str, date: str = None):
    """Return raw NWS CLI report text for the given date."""
    city = city.upper()
    if city not in CITIES:
        return {"error": f"Unknown city: {city}"}

    station_id = CITIES[city]["settlement"]
    from core.timezone import ET as _ET
    target_date = date if date else datetime.now(_ET).strftime("%Y-%m-%d")

    con = get_connection()
    row = con.execute(
        "SELECT raw_text, max_temp_f, source, ingested_at FROM nws_daily WHERE station_id = ? AND obs_date = ?",
        [station_id, target_date],
    ).fetchone()
    con.close()

    if not row or not row[0]:
        return {"city": city, "date": target_date, "raw_text": None, "source": row[2] if row else None}

    return {
        "city": city,
        "date": target_date,
        "raw_text": row[0],
        "max_temp_f": row[1],
        "source": row[2],
        "ingested_at": row[3].isoformat() if row[3] else None,
    }


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


@app.get("/api/health/detailed")
async def health_detailed():
    """Comprehensive system health for the /health page."""
    from core.heartbeat import get_all_heartbeats
    from core.constants import (
        POLL_INTERVAL_SECONDS,
        FORECAST_POLL_INTERVAL_SECONDS,
        NWS_CLI_POLL_INTERVAL_SECONDS,
        MARKET_POLL_INTERVAL_SECONDS,
        DRIFT_POLL_INTERVAL_SECONDS,
    )

    con = get_connection()
    try:
        # Data freshness — last record per source
        freshness = {}
        freshness_queries = {
            "observations_synoptic": "SELECT MAX(ingested_at) FROM observations WHERE ingest_source = 'synoptic'",
            "observations_awc": "SELECT MAX(ingested_at) FROM observations WHERE ingest_source IN ('awc', 'iem')",
            "forecasts_hrrr": "SELECT MAX(ingested_at) FROM forecasts WHERE model_name = 'hrrr'",
            "market_ticks": "SELECT MAX(captured_at) FROM market_ticks",
            "drift_signals": "SELECT MAX(calculated_at) FROM drift_signals",
            "nws_cli": "SELECT MAX(ingested_at) FROM nws_daily WHERE source = 'NWS_CLI'",
            "nws_dsm": "SELECT MAX(ingested_at) FROM nws_daily WHERE source = 'DSM'",
        }
        now_utc = datetime.now(timezone.utc)
        for key, query in freshness_queries.items():
            row = con.execute(query).fetchone()
            ts = row[0] if row else None
            age = None
            if ts:
                if ts.tzinfo is None:
                    age = round((now_utc.replace(tzinfo=None) - ts).total_seconds() / 60, 1)
                else:
                    age = round((now_utc - ts).total_seconds() / 60, 1)
            freshness[key] = {
                "last_seen": ts.isoformat() if ts else None,
                "age_minutes": age,
            }

        # Pipeline heartbeats
        pipelines = get_all_heartbeats()

        # Database stats
        table_stats = {}
        for table in ["observations", "forecasts", "market_ticks", "drift_signals",
                       "nws_daily", "paper_positions", "kalshi_settlements",
                       "kalshi_candlesticks", "kalshi_trades"]:
            try:
                row = con.execute("SELECT COUNT(*) FROM {}".format(table)).fetchone()
                table_stats[table] = {"rows": row[0]}
            except Exception:
                table_stats[table] = {"rows": 0}

        try:
            size_row = con.execute("CALL pragma_database_size()").fetchone()
            db_size = size_row[4] if size_row else "unknown"  # human-readable size string
        except Exception:
            db_size = "unknown"

        # Config (static values from constants)
        config = {
            "settlement_station": "KNYC",
            "neighbor_stations": ["KLGA", "KEWR", "KJFK"],
            "polling_intervals": {
                "observations": POLL_INTERVAL_SECONDS,
                "forecasts": FORECAST_POLL_INTERVAL_SECONDS,
                "market": MARKET_POLL_INTERVAL_SECONDS,
                "drift": DRIFT_POLL_INTERVAL_SECONDS,
                "nws_cli": NWS_CLI_POLL_INTERVAL_SECONDS,
            },
            "stale_threshold_min": STALE_THRESHOLD_MINUTES,
            "model": "HRRR",
            "bias_ttl_min": 60,
        }

        return {
            "freshness": freshness,
            "pipelines": pipelines,
            "database": {"tables": table_stats, "size": db_size},
            "config": config,
        }
    finally:
        con.close()


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

        # Model high from QR model (via model_state table)
        today_et = get_today_et()
        model_state = _get_model_state(con, city_upper, today_et)
        if model_state:
            bracket_probs, fcst_high, _ = model_state
            model_high = _model_expected_value(bracket_probs)
        else:
            model_high = None

        # Settlement status from nws_daily
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
        ) if bracket_row and bracket_row[0] is not None and bracket_row[1] is not None else None

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

    # Coverage check: run must span the peak heating window
    # A run needs points before 14z (morning) AND after 20z (afternoon)
    # to be a meaningful daily high forecast
    target_date_str = date or get_today_et()
    target_14z = datetime.fromisoformat(target_date_str + "T14:00:00")
    target_20z = datetime.fromisoformat(target_date_str + "T20:00:00")

    con = get_connection()
    rows = con.execute(
        """SELECT model_run, valid_at, temp_f, ingested_at, model_name
           FROM forecasts
           WHERE station_id = ?
           AND valid_at >= ? AND valid_at < ?
           ORDER BY model_name ASC, model_run ASC, valid_at ASC""",
        [station_id, day_start_utc, day_end_utc],
    ).fetchall()

    # Group by (model_run, model_name)
    runs_data = defaultdict(lambda: {"points": [], "ingested_at": None})
    for model_run, valid_at, temp_f, ingested, model_name in rows:
        if temp_f is None:
            continue
        key = (model_run, model_name)
        entry = runs_data[key]
        entry["points"].append((valid_at, temp_f))
        if ingested and entry["ingested_at"] is None:
            entry["ingested_at"] = ingested

    # Build summaries with per-model deltas
    summaries = []
    prev_high_by_model = {}  # type: Dict[str, Tuple[float, datetime]]
    for (model_run, model_name) in sorted(runs_data.keys()):
        entry = runs_data[(model_run, model_name)]
        if not entry["points"]:
            continue

        # Find the high temp and its valid_at
        best_valid, best_temp = max(entry["points"], key=lambda p: p[1])

        high_f = round(best_temp, 1)
        high_at = best_valid.isoformat()
        ingested = entry["ingested_at"]

        # Coverage: run must span the peak heating window (before 14z AND after 20z)
        min_valid = min(p[0] for p in entry["points"])
        max_valid = max(p[0] for p in entry["points"])
        coverage = "full" if min_valid <= target_14z and max_valid >= target_20z else "partial"

        # Per-model delta from prior run of the same model
        temp_change = None
        time_change_mins = None
        if model_name in prev_high_by_model:
            prev_f, prev_dt = prev_high_by_model[model_name]
            temp_change = round(high_f - prev_f, 1)
            delta_secs = (best_valid - prev_dt).total_seconds()
            time_change_mins = round(delta_secs / 60)

        summaries.append({
            "model_run": model_run.isoformat(),
            "model_name": model_name,
            "ingested_at": ingested.isoformat() if ingested else None,
            "high_temp_f": high_f,
            "high_valid_at": high_at,
            "temp_change": temp_change,
            "time_change_mins": time_change_mins,
            "coverage": coverage,
        })

        prev_high_by_model[model_name] = (high_f, best_valid)

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
    """Return HRRR forecast curve + observations + bias-adjusted forecast.

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
        # Still fetch observations even when no forecasts available
        obs_points = []
        running_high_f = None
        settlement_marker = None
        if day_start_utc:
            obs_rows = con.execute(
                """SELECT observed_at, temp_f FROM observations
                   WHERE station_id = ? AND observed_at >= ? AND observed_at < ?
                   ORDER BY observed_at""",
                [station_id, day_start_utc, day_end_utc],
            ).fetchall()
            for observed_at_ts, temp_f in obs_rows:
                if temp_f is not None:
                    obs_points.append({
                        "observed_at": observed_at_ts.isoformat(),
                        "temp_f": round(temp_f, 1),
                    })
                    if running_high_f is None or temp_f > running_high_f:
                        running_high_f = temp_f
        con.close()
        return {"city": city, "forecasts": [], "observations": obs_points,
                "prior_runs": [],
                "running_high": running_high_f,
                "settlement_marker": settlement_marker}

    # Build forecast points (ribbon removed — old CI engine replaced by QR model)
    bias_val_raw, _ = _get_bias_from_db(con, station_id)

    forecast_points = []
    for valid_at, temp_f, model_run_ts, ingested_ts in forecasts:
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
    bias_val = round(bias_val_raw, 2)

    # Drift-adjusted forecast: -bias + drift
    adjustment = round(-bias_val + drift, 2)
    bias_adjusted_points = [
        {"valid_at": p["valid_at"], "temp_f": round(p["temp_f"] + adjustment, 1)}
        for p in forecast_points
    ]

    con.close()
    now = datetime.now(timezone.utc)
    return {
        "city": city,
        "model_run": latest_mr.isoformat() if latest_mr else None,
        "forecasts": forecast_points,
        "observations": obs_points,
        "six_hr_maxes": six_hr_maxes,
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
        "model_band": _get_model_band(city, date or get_today_et()),
    }


def _get_model_band(city, target_date):
    # type: (str, str) -> Optional[Dict]
    """Get model median and 25th/75th percentile for the chart overlay."""
    con = get_connection()
    try:
        result = _get_model_state(con, city, target_date)
        if not result:
            return None
        bracket_probs, fcst_high, _ = result
        median = _model_expected_value(bracket_probs)
        p25 = _model_percentile(bracket_probs, 0.25)
        p75 = _model_percentile(bracket_probs, 0.75)
        return {"median": median, "p25": p25, "p75": p75}
    finally:
        con.close()


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

    # Get model probabilities from QR model state
    target_date_str = target.strftime("%Y-%m-%d")
    model_result = _get_model_state(con, city, target_date_str)
    if not model_result:
        con.close()
        return {"city": city, "available": False, "message": "No model forecast available"}
    bracket_probs, _, _ = model_result

    comparisons = []
    for market_id, yes_bid, yes_ask, last_trade, floor_strike, cap_strike in ticks:
        # Kalshi bracket interpretation:
        #   Bottom tail (floor=None, cap=X): resolves YES if high < X -> covers <=(X-1)
        #   Range (floor=X, cap=Y): resolves YES if X <= high <= Y
        #   Top tail (cap=None, floor=X): resolves YES if high > X -> covers >=(X+1)
        if floor_strike is None and cap_strike is not None:
            label_bound = int(cap_strike) - 1
            bracket_label = f"<={label_bound}"
            model_prob = sum(p for k, p in bracket_probs.items() if k <= label_bound)
        elif cap_strike is None and floor_strike is not None:
            label_bound = int(floor_strike) + 1
            bracket_label = f">={label_bound}"
            model_prob = sum(p for k, p in bracket_probs.items() if k >= label_bound)
        elif floor_strike is not None and cap_strike is not None:
            bracket_label = "{}-{}".format(int(floor_strike), int(cap_strike))
            model_prob = sum(p for k, p in bracket_probs.items()
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

    # 1. Get model probabilities from QR model state
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

    model_center = None
    model_2f = {}  # type: Dict[tuple, float]

    model_result = _get_model_state(con, city, target_date_str)
    bracket_probs = None
    if model_result:
        bracket_probs, fcst_high, _ = model_result
        model_center = _model_expected_value(bracket_probs)

    # 2. Get latest Kalshi market ticks for this city + date
    captured_at_row = con.execute(
        """SELECT MAX(captured_at) FROM market_ticks
           WHERE city = ? AND market_id LIKE ?""",
        [city, f"%{kalshi_date}%"],
    ).fetchone()
    captured_at_iso = captured_at_row[0].isoformat() if captured_at_row and captured_at_row[0] else None

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

    # Index market data by (floor, cap) for merging — include tail brackets
    market_by_bracket = {}  # type: Dict[tuple, dict]
    for market_id, yes_bid, yes_ask, last_trade, floor_strike, cap_strike, volume in ticks:
        # Bottom tail: floor=None, cap=X → "below X"
        # Range: floor=X, cap=Y → "X to Y"
        # Top tail: floor=X, cap=None → "above X"
        f = int(floor_strike) if floor_strike is not None else None
        c = int(cap_strike) if cap_strike is not None else None
        key = (f, c)
        market_by_bracket[key] = {
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "volume": volume or 0,
        }

    con.close()

    # Map 1°F model probs to actual Kalshi bracket boundaries
    if bracket_probs:
        for key in market_by_bracket:
            floor, cap = key
            if floor is not None and cap is not None:
                model_2f[key] = sum(p for k, p in bracket_probs.items() if floor <= k <= cap)
            elif floor is None and cap is not None:
                model_2f[key] = sum(p for k, p in bracket_probs.items() if k < cap)
            elif cap is None and floor is not None:
                model_2f[key] = sum(p for k, p in bracket_probs.items() if k > floor)

    # 3. Merge model + market into bracket objects
    # When model is disabled, only show market brackets
    if model_2f:
        all_keys = set(model_2f.keys()) | set(market_by_bracket.keys())
    else:
        all_keys = set(market_by_bracket.keys())

    # Sort: bottom tail first (floor=None), then ranges by floor, then top tail (cap=None)
    def bracket_sort_key(k):
        f, c = k
        return (f if f is not None else -9999, c if c is not None else 9999)

    brackets = []
    total_volume = 0
    spread_sum = 0.0
    spread_count = 0

    for key in sorted(all_keys, key=bracket_sort_key):
        floor, cap = key
        has_model = key in model_2f
        model_prob = round(model_2f[key], 4) if has_model else None
        mkt = market_by_bracket.get(key, {})
        yes_bid = mkt.get("yes_bid")
        yes_ask = mkt.get("yes_ask")
        vol = mkt.get("volume", 0)

        market_mid = None
        if yes_bid is not None and yes_ask is not None:
            market_mid = round((yes_bid + yes_ask) / 2.0, 4)

        edge = round(model_prob - market_mid, 4) if (model_prob is not None and market_mid is not None) else None

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
        "captured_at": captured_at_iso,
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
                "bracket": "{}-{}°F".format(int(floor), int(cap)) if floor is not None else "--",
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
                if b["floor"] is None:
                    label = "≤{}°F".format(b["cap"])
                elif b["cap"] is None:
                    label = "≥{}°F".format(b["floor"])
                else:
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
        int(bracket_floor), int(bracket_floor) + 1
    ) if bracket_floor is not None else "?"

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
    and categorize what went wrong.  Also identifies missed_edge cases
    where a bracket settled YES but no position was taken despite the
    model having predicted edge > 5%.
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

        # Get settlement data from nws_daily for KNYC (temp + source)
        # Collect all dates we might need: from lost positions + missed edge window
        nws_rows = con.execute(
            """SELECT obs_date, max_temp_f, source FROM nws_daily
               WHERE station_id = 'KNYC'
               AND obs_date >= CURRENT_DATE - INTERVAL '{}' DAY""".format(days),
        ).fetchall()
        settlement_data = {}  # type: dict
        for obs_date, max_temp_f, source in nws_rows:
            settlement_data[str(obs_date)] = {
                "temp": max_temp_f,
                "source": source,
            }

        # --- Missed edge: brackets that settled YES but we had no position ---
        # Get all brackets we traded (any status) so we can exclude them
        traded_keys = set()  # type: set
        all_pos_rows = con.execute(
            """SELECT event_date, bracket_floor, bracket_cap
               FROM paper_positions
               WHERE event_date >= CURRENT_DATE - INTERVAL '{}' DAY""".format(days),
        ).fetchall()
        for ev_date, bf, bc in all_pos_rows:
            traded_keys.add((str(ev_date), int(bf) if bf is not None else None,
                             int(bc) if bc is not None else None))

        # Get daily best market prices per bracket from market_ticks
        # Average daily midpoint per bracket
        missed_rows = con.execute(
            """SELECT
                   mt.city,
                   CAST(mt.captured_at AS DATE) AS tick_date,
                   CAST(mt.floor_strike AS INTEGER) AS bracket_floor,
                   CAST(mt.cap_strike AS INTEGER) AS bracket_cap,
                   AVG((mt.yes_bid + mt.yes_ask) / 2.0) AS market_mid
               FROM market_ticks mt
               WHERE mt.city = 'NYC'
               AND mt.floor_strike IS NOT NULL
               AND mt.cap_strike IS NOT NULL
               AND mt.yes_bid IS NOT NULL
               AND mt.yes_ask IS NOT NULL
               AND CAST(mt.captured_at AS DATE) >= CURRENT_DATE - INTERVAL '{}' DAY
               GROUP BY mt.city,
                        CAST(mt.captured_at AS DATE),
                        CAST(mt.floor_strike AS INTEGER),
                        CAST(mt.cap_strike AS INTEGER)""".format(days),
        ).fetchall()
    finally:
        con.close()

    # Build missed_edge incidents
    missed_edge_incidents = []
    for _, tick_date, bracket_floor, cap_strike, market_mid in missed_rows:
        date_str = str(tick_date)
        sd = settlement_data.get(date_str)
        if sd is None or sd["temp"] is None:
            continue
        settlement_temp = sd["temp"]
        # Settlement bracket is [floor, floor + BRACKET_WIDTH) — exclusive cap
        # cap_strike from Kalshi is floor + 1, but settlement uses floor + 2
        bracket_cap = bracket_floor + 2
        # Did this bracket settle YES?
        if not (bracket_floor <= settlement_temp < bracket_cap):
            continue
        # Did we already trade this bracket?
        if (date_str, bracket_floor, bracket_cap) in traded_keys:
            continue
        # Edge: model predicted YES probability > market_mid by > 5%
        # Since it settled YES (value = 1.0), the true probability was high.
        # The "missed edge" is 1.0 - market_mid (hindsight edge).
        # Only include if market_mid < 0.95 (meaningful opportunity)
        hindsight_edge = 1.0 - market_mid if market_mid is not None else 0
        if hindsight_edge < 0.05:
            continue

        bracket_label = "{}-{}°F".format(bracket_floor, bracket_cap)
        # Estimate severity by hindsight edge (higher edge = bigger miss)
        severity = round(min(1.0, hindsight_edge), 2)

        narrative = ("Bracket {} settled YES at {}°F. Market was at {}¢ — "
                     "could have bought for ~{}¢ profit per contract.").format(
            bracket_label,
            int(round(settlement_temp)),
            int(round(market_mid * 100)),
            int(round(hindsight_edge * 100)),
        )

        missed_edge_incidents.append({
            "date": date_str,
            "type": "missed_edge",
            "severity": severity,
            "bracket": bracket_label,
            "direction": "BUY YES",
            "model_prob": None,
            "market_price": round(market_mid, 4) if market_mid else None,
            "edge": round(hindsight_edge, 4),
            "net_pnl": 0.0,
            "pnl": 0.0,
            "settlement_temp": settlement_temp,
            "settlement_source": sd.get("source"),
            "category": "missed_edge",
            "narrative": narrative,
        })

    # Build lost_bet incident cards
    incidents = []
    # Track max abs pnl for severity normalization
    max_abs_pnl = max((abs(row[8]) for row in pos_rows), default=1.0)
    if max_abs_pnl == 0:
        max_abs_pnl = 1.0

    for row in pos_rows:
        event_date, bracket_floor, bracket_cap, direction, model_prob, \
            market_price, edge, entry_price, net_pnl = row

        date_str = str(event_date)
        sd = settlement_data.get(date_str)
        settlement_temp = sd.get("temp") if sd else None
        settlement_source = sd.get("source") if sd else None

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
            "pnl": round(net_pnl, 2),
            "settlement_temp": settlement_temp,
            "settlement_source": settlement_source,
            "category": category,
            "narrative": narrative,
        })

    # Merge lost_bet + missed_edge incidents
    incidents.extend(missed_edge_incidents)

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
        "missed_edge": "Bracket settled YES at low market price — review entry threshold or sizing",
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


# ---------------------------------------------------------------------------
# Blotter — bundled countdown + brackets + positions
# ---------------------------------------------------------------------------


@app.get("/api/blotter/calendar/{city}")
async def blotter_calendar(city: str, range: str = "30d"):
    """Daily P&L summary for the calendar widget."""
    city_upper = city.upper()
    if city_upper not in CITIES:
        return {"error": f"Unknown city: {city}"}

    days_back = {"7d": 7, "30d": 30, "90d": 90, "all": 365}.get(range, 30)
    from core.timezone import ET as _ET
    today = datetime.now(_ET).date()
    start_date = today - timedelta(days=days_back)

    con = get_connection()
    try:
        rows = con.execute("""
            SELECT event_date,
                   COUNT(*) as total,
                   SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN net_pnl <= 0 AND status != 'open' THEN 1 ELSE 0 END) as losses,
                   SUM(COALESCE(CASE WHEN status = 'open' THEN unrealized_pnl ELSE net_pnl END, 0)) as net_pnl,
                   SUM(CASE WHEN status != 'open' THEN ABS(COALESCE(entry_price, 0) * COALESCE(contracts, 1)) ELSE 0 END) as wagered
            FROM paper_positions
            WHERE city = ? AND event_date >= ?
            GROUP BY event_date
            ORDER BY event_date
        """, [city_upper, start_date.isoformat()]).fetchall()

        days = []
        total_trades = 0
        total_wins = 0
        total_losses = 0
        total_pnl = 0.0
        total_wagered = 0.0

        for r in rows:
            net = round(r[4], 2)
            days.append({
                "date": r[0].isoformat() if hasattr(r[0], 'isoformat') else str(r[0]),
                "trades": r[1],
                "wins": r[2],
                "losses": r[3],
                "net_pnl": net,
            })
            total_trades += r[1]
            total_wins += r[2]
            total_losses += r[3]
            total_pnl += net
            total_wagered += r[5] or 0

        return {
            "days": days,
            "summary": {
                "total_trades": total_trades,
                "wins": total_wins,
                "losses": total_losses,
                "win_rate": round(total_wins / max(total_trades, 1), 3),
                "total_wagered": round(total_wagered / 100, 2),
                "total_profit": round(total_pnl, 2),
                "roi": round(total_pnl / max(total_wagered / 100, 0.01), 3),
            },
        }
    finally:
        con.close()


@app.get("/api/blotter/{city}")
async def blotter_data(city: str, date: str = None):
    """Bundled data for the blotter page: countdown + brackets + positions."""
    city_upper = city.upper()
    target_date = date or get_today_et()
    con = get_connection()
    try:
        # --- Countdown ---
        model_high = None
        model_bracket = None
        model_bracket_prob = None
        model_result = _get_model_state(con, city_upper, target_date)
        if model_result:
            bracket_probs, fcst_high, _ = model_result
            model_high = _model_expected_value(bracket_probs)
            if model_high is not None:
                floor_2f = int((model_high // 2) * 2)
                # Sum probabilities for the 2degF bracket containing the EV
                model_bracket_prob = round(
                    sum(p for k, p in bracket_probs.items()
                        if floor_2f <= k < floor_2f + 2), 3
                )
                model_bracket = "{}-{}".format(floor_2f, floor_2f + 1)

        # Running obs max
        obs_row = con.execute("""
            SELECT MAX(temp_f), MAX(observed_at)
            FROM observations
            WHERE station_id = 'KNYC'
              AND observed_at >= ?::DATE
              AND observed_at < ?::DATE + INTERVAL '1 day'
        """, [target_date, target_date]).fetchone()
        running_obs_max = obs_row[0] if obs_row else None
        obs_max_time = obs_row[1].isoformat() if obs_row and obs_row[1] else None

        # Settlement source
        settle_row = con.execute("""
            SELECT max_temp_f, source FROM nws_daily
            WHERE station_id = 'KNYC' AND obs_date = ?
            ORDER BY CASE source
                WHEN 'NWS_CLI' THEN 3 WHEN 'DSM' THEN 2 ELSE 1
            END DESC LIMIT 1
        """, [target_date]).fetchone()

        # Market close time
        close_row = con.execute("""
            SELECT close_time FROM kalshi_settlements
            WHERE city = ? AND event_date = ?
            LIMIT 1
        """, [city_upper, target_date]).fetchone()

        countdown = {
            "event_date": target_date,
            "model_high": model_high,
            "model_bracket": model_bracket,
            "model_bracket_prob": model_bracket_prob,
            "running_obs_max": running_obs_max,
            "obs_max_time": obs_max_time,
            "settlement": {
                "temp": settle_row[0] if settle_row else None,
                "source": settle_row[1] if settle_row else "pending",
            },
            "close_time": close_row[0].isoformat() if close_row and close_row[0] else None,
        }

        # --- Brackets (reuse existing endpoint) ---
        brackets_resp = await bracket_spread(city, target_date)

        # --- Positions ---
        positions = con.execute("""
            SELECT bracket_floor, bracket_cap, direction, contracts,
                   entry_price, unrealized_pnl, net_pnl, status, exit_reason,
                   gross_pnl, fees, exit_price, model_prob, edge,
                   entry_time, exit_time
            FROM paper_positions
            WHERE city = ? AND event_date = ?
            ORDER BY bracket_floor
        """, [city_upper, target_date]).fetchall()

        position_list = []
        for p in positions:
            position_list.append({
                "bracket_floor": p[0], "bracket_cap": p[1],
                "direction": p[2], "contracts": p[3],
                "entry_price": p[4],
                "unrealized_pnl": p[5],
                "net_pnl": p[6],
                "status": p[7], "exit_reason": p[8],
                "gross_pnl": p[9], "fees": p[10],
                "exit_price": p[11],
                "model_prob": p[12], "edge": p[13],
                "entry_time": p[14].isoformat() if p[14] else None,
                "exit_time": p[15].isoformat() if p[15] else None,
            })

        open_positions = [p for p in positions if p[7] == "open"]
        closed_positions = [pos for pos in position_list if pos["status"] != "open"]
        open_count = len(open_positions)
        capital_locked = sum((p[4] * p[3] / 100.0) for p in open_positions if p[4] and p[3])
        max_loss = capital_locked  # worst case = lose all locked capital
        day_pnl = sum(
            (p[5] if p[7] == "open" else (p[6] or 0))
            for p in positions
        )

        return {
            "countdown": countdown,
            "brackets": brackets_resp,
            "positions": position_list,
            "closed_positions": closed_positions,
            "positions_summary": {
                "open_count": open_count,
                "capital_locked": round(capital_locked, 2),
                "max_loss": round(max_loss, 2),
                "day_pnl": round(day_pnl, 2),
            },
        }
    finally:
        con.close()
