#!/usr/bin/env python3
"""Day Microscope — Minute-Level Model vs Market Comparison.

Zooms into hand-picked days and watches minute-by-minute how the market's
probability distribution evolves vs our model's — identifying exactly when
and where the market incorporates information we're missing.

Usage:
    cd ~/Projects/alphatemp/alphatemp
    PYTHONPATH=. venv/bin/python scripts/day_microscope.py
"""

import re
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd

sys.path.insert(0, ".")

from services.backtester import (
    Backtester,
    KalshiBracket,
    compute_brier_score,
    map_probs_to_kalshi_brackets,
    wf_qr_multimodel_v2,
)
from core.timezone import ET

# ── Configuration (easy to swap) ────────────────────────────────────────
TARGET_DATES = [
    date(2025, 1, 9),   # Actual high 33°F — mid-range
    date(2025, 1, 16),  # Actual high 30°F — cold
    date(2025, 1, 24),  # Actual high 33°F — mid-range
]

DB_PATH = "data/alphatemp.duckdb"
# start_date/end_date control which settlement dates get evaluated.
# The QR model trains on ALL historical data prior to each evaluation date
# (1900+ curves cached regardless of these bounds).
START_DATE = date(2025, 1, 1)   # Start just before target dates
END_DATE = date(2025, 1, 31)    # End just after — keeps runtime ~3 min
UPDATE_HOURS_ET = list(range(24))  # Every hour — full production simulation
STATIONS = ["KNYC", "KLGA", "KEWR"]

# Divergence threshold (percentage points)
DIVERGENCE_THRESHOLD_PP = 10.0
# Forward-fill staleness threshold (minutes)
STALENESS_MINUTES = 5


# ═══════════════════════════════════════════════════════════════════════
# Phase A: Capture model distributions
# ═══════════════════════════════════════════════════════════════════════

def make_capturing_wrapper(inner_fn, target_dates, store):
    # type: (...) -> callable
    """Wrap a model function to intercept bracket probs for target dates.

    Non-invasive — intercepts the return value without modifying backtester
    or RunResult. The store dict maps (date, hour_et) -> Dict[int, float].
    """
    target_set = set(target_dates)

    def wrapper(provider, ref_time):
        result = inner_fn(provider, ref_time)
        if result:
            # ref_time is timezone-aware UTC (backtester.py line 2896)
            et_dt = ref_time.astimezone(ET)
            sdate = et_dt.date()
            if sdate in target_set:
                store[(sdate, et_dt.hour)] = dict(result)
        return result

    wrapper.__name__ = inner_fn.__name__
    return wrapper


def run_backtest_with_capture(target_dates):
    # type: (List[date]) -> Tuple[Dict, object]
    """Run the full backtester, capturing model distributions for target dates."""
    captured = {}  # type: Dict[Tuple[date, int], Dict[int, float]]

    wrapped_fn = make_capturing_wrapper(
        wf_qr_multimodel_v2, target_dates, captured,
    )

    bt = Backtester(db_path=DB_PATH, city="NYC")
    result = bt.run(
        wrapped_fn,
        update_hours_et=UPDATE_HOURS_ET,
        latest_run=True,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    return captured, result


# ═══════════════════════════════════════════════════════════════════════
# Phase B: Load market + observation data
# ═══════════════════════════════════════════════════════════════════════

def _fix_null_strikes(brackets_df):
    # type: (pd.DataFrame) -> pd.DataFrame
    """Infer floor/cap strikes from ticker name when DB values are NULL.

    Ticker format: KXHIGHNY-25JAN09-B30.5 -> interior bracket, floor=30, cap=31
    The B-prefix midpoint convention: B{x}.5 -> floor=x, cap=x+1.
    """
    for i, row in brackets_df.iterrows():
        if pd.isna(row["floor_strike"]) and pd.isna(row["cap_strike"]):
            ticker = row["market_ticker"]
            suffix = ticker.split("-")[-1]  # e.g. "B30.5"
            m = re.match(r"B(\d+)\.5$", suffix)
            if m:
                mid = int(m.group(1))
                brackets_df.at[i, "floor_strike"] = float(mid)
                brackets_df.at[i, "cap_strike"] = float(mid + 1)
    return brackets_df


def load_brackets(con, target_date):
    # type: (duckdb.DuckDBPyConnection, date) -> Tuple[pd.DataFrame, List[KalshiBracket]]
    """Load settlement brackets for a date, fixing NULL strikes from ticker names."""
    df = con.execute("""
        SELECT market_ticker, floor_strike, cap_strike, settled_yes
        FROM kalshi_settlements
        WHERE event_date = ? AND city = 'NYC' AND measure = 'high'
        ORDER BY floor_strike NULLS FIRST
    """, [target_date]).df()

    if df.empty:
        return df, []

    df = _fix_null_strikes(df)

    # Filter any remaining brackets where both strikes are still NULL
    mask = ~(df["floor_strike"].isna() & df["cap_strike"].isna())
    df = df[mask].reset_index(drop=True)

    # Re-sort: lower tail (floor=NaN) first, then by floor_strike, upper tail last
    def sort_key(row):
        f, c = row["floor_strike"], row["cap_strike"]
        if pd.isna(f):
            return (-1e9,)  # lower tail first
        if pd.isna(c):
            return (1e9,)   # upper tail last
        return (f,)
    df["_sort"] = df.apply(sort_key, axis=1)
    df = df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)

    brackets = []
    for _, row in df.iterrows():
        brackets.append(KalshiBracket(
            floor_strike=row["floor_strike"] if pd.notna(row["floor_strike"]) else None,
            cap_strike=row["cap_strike"] if pd.notna(row["cap_strike"]) else None,
            settled_yes=int(row["settled_yes"]),
        ))
    return df, brackets


def bracket_label(floor_strike, cap_strike):
    # type: (Optional[float], Optional[float]) -> str
    """Build human-readable bracket label from floor/cap strikes."""
    if floor_strike is None or (isinstance(floor_strike, float) and pd.isna(floor_strike)):
        return "<{:.0f}".format(cap_strike)
    if cap_strike is None or (isinstance(cap_strike, float) and pd.isna(cap_strike)):
        return ">{:.0f}".format(floor_strike)
    return "{:.0f}-{:.0f}".format(floor_strike, cap_strike)


def load_candles_for_date(con, tickers, target_date):
    # type: (duckdb.DuckDBPyConnection, List[str], date) -> pd.DataFrame
    """Load all minute-level candlestick data for settlement day.

    Candlestick timestamps are ET (naive) — no timezone conversion needed.
    Settlement day = midnight ET to 23:59 ET.
    """
    df = con.execute("""
        SELECT
            market_ticker,
            end_period_ts,
            yes_bid_close,
            yes_ask_close,
            price_close,
            volume
        FROM kalshi_candlesticks
        WHERE market_ticker IN (SELECT UNNEST(?::VARCHAR[]))
          AND CAST(end_period_ts AS DATE) = ?
        ORDER BY end_period_ts
    """, [tickers, target_date]).df()
    return df


def build_market_timeline(candles_df, tickers, target_date):
    # type: (pd.DataFrame, List[str], date) -> Dict[pd.Timestamp, Dict[str, dict]]
    """Build forward-filled minute-by-minute market state.

    Returns: minute_ts -> { market_ticker -> {mid, yes_ask, is_stale, mins_stale} }

    Samples at every minute where a trade occurred, plus hourly and half-hourly
    boundaries for comparison with the model.
    """
    if candles_df.empty:
        return {}

    # Build per-ticker sorted candle lists
    ticker_candles = {}  # type: Dict[str, List[Tuple]]
    for ticker in tickers:
        mask = candles_df["market_ticker"] == ticker
        tc = candles_df[mask].sort_values("end_period_ts")
        ticker_candles[ticker] = [
            (pd.Timestamp(row["end_period_ts"]), row["yes_bid_close"], row["yes_ask_close"])
            for _, row in tc.iterrows()
        ]

    # Collect all timestamps we want to sample
    sample_times = set()
    # Every actual candle timestamp
    for ts in candles_df["end_period_ts"]:
        sample_times.add(pd.Timestamp(ts))
    # Hourly and half-hourly boundaries
    for h in range(24):
        for m in (0, 30):
            sample_times.add(pd.Timestamp(
                datetime(target_date.year, target_date.month, target_date.day, h, m)
            ))

    sorted_samples = sorted(sample_times)

    # Pointer-based forward-fill — O(samples + candles) per ticker
    pointers = {t: -1 for t in tickers}  # type: Dict[str, int]
    result = {}

    for ts in sorted_samples:
        minute_data = {}

        for ticker in tickers:
            candles = ticker_candles[ticker]
            ptr = pointers[ticker]
            # Advance pointer to the last candle at or before this timestamp
            while ptr + 1 < len(candles) and candles[ptr + 1][0] <= ts:
                ptr += 1
            pointers[ticker] = ptr

            if ptr >= 0:
                candle_ts, bid, ask = candles[ptr]
                bid = bid if bid and not pd.isna(bid) else 0
                ask = ask if ask and not pd.isna(ask) else 0
                mid = (bid + ask) / 2.0
                mins_stale = int((ts - candle_ts).total_seconds() / 60)
                minute_data[ticker] = {
                    "mid": mid,
                    "yes_ask": ask,
                    "is_stale": mins_stale > STALENESS_MINUTES,
                    "mins_stale": mins_stale,
                }

        if minute_data:
            result[ts] = minute_data

    return result


def load_observations(con, target_date):
    # type: (duckdb.DuckDBPyConnection, date) -> pd.DataFrame
    """Load KNYC + KLGA + KEWR observations for a date.

    observed_at is UTC (naive) in the DB. We return the raw UTC timestamps;
    conversion to ET happens at display time.
    """
    et_midnight = datetime(
        target_date.year, target_date.month, target_date.day,
        0, 0, tzinfo=ET,
    )
    utc_start = et_midnight.astimezone(timezone.utc).replace(tzinfo=None)
    utc_end = utc_start + timedelta(hours=24)

    return con.execute("""
        SELECT station_id, observed_at, temp_f
        FROM observations
        WHERE station_id IN ('KNYC', 'KLGA', 'KEWR')
          AND observed_at >= ? AND observed_at < ?
          AND temp_f IS NOT NULL
        ORDER BY observed_at
    """, [utc_start, utc_end]).df()


def load_actual_high(con, target_date):
    # type: (duckdb.DuckDBPyConnection, date) -> Optional[float]
    """Get the official NWS daily high for a date."""
    row = con.execute("""
        SELECT max_temp_f FROM nws_daily
        WHERE obs_date = ? AND station_id = 'KNYC'
    """, [target_date]).fetchone()
    return row[0] if row else None


# ═══════════════════════════════════════════════════════════════════════
# Phase C: Analysis output
# ═══════════════════════════════════════════════════════════════════════

def _get_model_mapped(captured, target_date, hour, kalshi_brackets):
    # type: (Dict, date, int, List[KalshiBracket]) -> Optional[List[float]]
    """Get model probs mapped to Kalshi brackets for a (date, hour), or None."""
    key = (target_date, hour)
    if key not in captured:
        return None
    return map_probs_to_kalshi_brackets(captured[key], kalshi_brackets)


def _get_market_at_hour(market_timeline, tickers, target_date, hour):
    # type: (Dict, List[str], date, int) -> Optional[Dict[str, dict]]
    """Get the market state at the top of the given hour, or nearest after."""
    hour_start = pd.Timestamp(
        datetime(target_date.year, target_date.month, target_date.day, hour, 0)
    )
    hour_end = hour_start + pd.Timedelta(hours=1)

    for ts in sorted(market_timeline.keys()):
        if ts < hour_start:
            continue
        if ts >= hour_end:
            break
        mkt = market_timeline[ts]
        if all(t in mkt for t in tickers):
            return mkt
    return None


def print_probability_timeline(captured, market_timeline, brackets_df,
                                kalshi_brackets, tickers, labels, target_date):
    """Section 1: Probability timeline — model (step) vs market (30-min sample)."""
    print("\n  1. PROBABILITY TIMELINE (M=model, K=market mid, *=model updated)")
    print("  " + "-" * 72)

    # Header
    header = "  {:>5s}  ".format("Hour")
    for lbl in labels:
        header += "{:>11s} ".format(lbl)
    print(header)
    print("  " + "-" * (7 + 12 * len(labels)))

    for h in range(24):
        # Model line
        mapped = _get_model_mapped(captured, target_date, h, kalshi_brackets)
        if mapped:
            line = "  {:>2d}:00".format(h)
            for p in mapped:
                line += "  {:>5.1f}% M  ".format(p * 100)
            line += " *"
        else:
            line = "  {:>2d}:00".format(h)
            for _ in labels:
                line += "  {:>9s}   ".format("---")

        # Market line (use :30 snapshot)
        mkt_ts = pd.Timestamp(
            datetime(target_date.year, target_date.month, target_date.day, h, 30)
        )
        mkt_line = "  {:>5s}".format("")
        has_mkt = False
        if mkt_ts in market_timeline:
            mkt_data = market_timeline[mkt_ts]
            for ticker in tickers:
                if ticker in mkt_data:
                    mid = mkt_data[ticker]["mid"]
                    stale = "s" if mkt_data[ticker]["is_stale"] else " "
                    mkt_line += "  {:>5.1f}% K{:s} ".format(mid, stale)
                    has_mkt = True
                else:
                    mkt_line += "  {:>9s}   ".format("---")

        print(line)
        if has_mkt:
            print(mkt_line)


def find_divergence_events(captured, market_timeline, kalshi_brackets,
                            tickers, labels_map, target_date, top_n=10):
    # type: (...) -> List[dict]
    """Section 2: Material divergence events where |model - market| > threshold."""
    print("\n  2. MATERIAL DIVERGENCE EVENTS (|model - market| > {}pp)".format(
        int(DIVERGENCE_THRESHOLD_PP),
    ))
    print("  " + "-" * 72)

    # Build model step function: latest eval at or before each hour
    model_by_hour = {}  # type: Dict[int, Dict[str, float]]
    for h in range(24):
        mapped = _get_model_mapped(captured, target_date, h, kalshi_brackets)
        if mapped:
            model_by_hour[h] = dict(zip(tickers, mapped))

    events = []
    for minute_ts in sorted(market_timeline.keys()):
        ts_dt = minute_ts.to_pydatetime()
        hour = ts_dt.hour

        # Find latest model eval at or before this hour
        latest_model_hour = None
        for mh in sorted(model_by_hour.keys()):
            if mh <= hour:
                latest_model_hour = mh
        if latest_model_hour is None:
            continue

        model_probs = model_by_hour[latest_model_hour]
        mkt_data = market_timeline[minute_ts]

        for ticker in tickers:
            if ticker not in mkt_data or ticker not in model_probs:
                continue

            model_p = model_probs[ticker]
            market_p = mkt_data[ticker]["mid"] / 100.0
            gap_pp = abs(model_p - market_p) * 100

            if gap_pp > DIVERGENCE_THRESHOLD_PP:
                events.append({
                    "minute": minute_ts,
                    "bracket": labels_map[ticker],
                    "ticker": ticker,
                    "model_p": model_p,
                    "market_p": market_p,
                    "gap_pp": gap_pp,
                    "direction": "model > mkt" if model_p > market_p else "mkt > model",
                    "is_stale": mkt_data[ticker]["is_stale"],
                    "model_hour": latest_model_hour,
                })

    events.sort(key=lambda e: e["gap_pp"], reverse=True)

    if not events:
        print("  No divergence events > {}pp found.".format(int(DIVERGENCE_THRESHOLD_PP)))
        return events

    print("  {:>12s}  {:>8s}  {:>7s}  {:>7s}  {:>6s}  {:>5s}  {:>13s}".format(
        "Minute ET", "Bracket", "Model", "Market", "Gap", "Stale", "Direction",
    ))
    print("  " + "-" * 68)

    for e in events[:top_n]:
        stale_flag = "YES" if e["is_stale"] else "no"
        ts_str = str(e["minute"])[11:19]  # HH:MM:SS
        print("  {:>12s}  {:>8s}  {:>6.1f}%  {:>6.1f}%  {:>5.1f}pp  {:>5s}  {:>13s}".format(
            ts_str,
            e["bracket"],
            e["model_p"] * 100,
            e["market_p"] * 100,
            e["gap_pp"],
            stale_flag,
            e["direction"],
        ))

    total = len(events)
    live = sum(1 for e in events if not e["is_stale"])
    stale = total - live
    print("\n  Total: {} events ({} live, {} stale)".format(total, live, stale))

    return events


def print_brier_comparison(captured, market_timeline, kalshi_brackets,
                            tickers, target_date):
    # type: (...) -> List[Tuple[int, float, float]]
    """Section 3: Brier comparison at each model update hour."""
    print("\n  3. BRIER COMPARISON (model vs market, normalized probs)")
    print("  " + "-" * 72)

    print("  {:>5s}  {:>12s}  {:>12s}  {:>8s}".format(
        "Hour", "Model Brier", "Mkt Brier", "Gap",
    ))
    print("  " + "-" * 43)

    pairs = []  # type: List[Tuple[int, float, float]]

    for h in range(24):
        mapped = _get_model_mapped(captured, target_date, h, kalshi_brackets)
        if mapped is None:
            continue

        model_brier = compute_brier_score(mapped, kalshi_brackets)

        # Market Brier — use normalized midpoints at this hour
        mkt_data = _get_market_at_hour(market_timeline, tickers, target_date, h)
        if mkt_data is None:
            print("  {:>2d}:00  {:>12.4f}  {:>12s}  {:>8s}".format(
                h, model_brier, "---", "---",
            ))
            continue

        mkt_probs = [mkt_data[t]["mid"] / 100.0 for t in tickers]
        # Normalize midpoint probs to sum to 1.0 for apples-to-apples Brier
        total = sum(mkt_probs)
        if total > 0:
            mkt_probs = [p / total for p in mkt_probs]

        market_brier = compute_brier_score(mkt_probs, kalshi_brackets)
        gap = model_brier - market_brier

        print("  {:>2d}:00  {:>12.4f}  {:>12.4f}  {:>+8.4f}".format(
            h, model_brier, market_brier, gap,
        ))
        pairs.append((h, model_brier, market_brier))

    if pairs:
        avg_model = sum(p[1] for p in pairs) / len(pairs)
        avg_market = sum(p[2] for p in pairs) / len(pairs)
        avg_gap = avg_model - avg_market
        print("  " + "-" * 43)
        print("  {:>5s}  {:>12.4f}  {:>12.4f}  {:>+8.4f}".format(
            "AVG", avg_model, avg_market, avg_gap,
        ))
    print("  (positive gap = model worse than market)")

    return pairs


def print_executable_edge(captured, market_timeline, kalshi_brackets,
                           tickers, labels, target_date):
    # type: (...) -> Tuple[int, int]
    """Section 4: Executable edge using raw yes_ask (unnormalized)."""
    print("\n  4. EXECUTABLE EDGE (model_prob - yes_ask/100, per bracket)")
    print("  " + "-" * 72)

    # Header
    header = "  {:>5s}".format("Hour")
    for lbl in labels:
        header += "  {:>9s}".format(lbl)
    print(header)
    print("  " + "-" * (5 + 11 * len(labels)))

    positive_edges = 0
    total_edges = 0

    for h in range(24):
        mapped = _get_model_mapped(captured, target_date, h, kalshi_brackets)
        if mapped is None:
            continue

        mkt_data = _get_market_at_hour(market_timeline, tickers, target_date, h)

        line = "  {:>2d}:00".format(h)
        if mkt_data:
            for i, ticker in enumerate(tickers):
                if ticker in mkt_data:
                    ask_prob = mkt_data[ticker]["yes_ask"] / 100.0
                    edge = mapped[i] - ask_prob
                    total_edges += 1
                    if edge > 0:
                        positive_edges += 1
                    line += "  {:>+8.1f}pp".format(edge * 100)
                else:
                    line += "  {:>9s}".format("---")
        else:
            for _ in labels:
                line += "  {:>9s}".format("---")

        print(line)

    print("\n  Positive edges: {}/{} bracket-hours".format(positive_edges, total_edges))
    return positive_edges, total_edges


def print_observation_timeline(obs_df, target_date, actual_high):
    """Section 5: Observation events through the day."""
    print("\n  5. OBSERVATION TIMELINE")
    print("  " + "-" * 72)
    print("  Official NWS high: {}{}F".format(
        int(actual_high) if actual_high else "?", chr(176),
    ))
    print()

    # Group observations by ET hour and station
    hour_data = defaultdict(dict)  # type: Dict[int, Dict[str, float]]
    for _, row in obs_df.iterrows():
        utc_ts = row["observed_at"]
        if isinstance(utc_ts, pd.Timestamp):
            utc_ts = utc_ts.to_pydatetime()
        # Convert UTC naive to ET
        utc_aware = utc_ts.replace(tzinfo=timezone.utc)
        et_ts = utc_aware.astimezone(ET)
        hour_data[et_ts.hour][row["station_id"]] = row["temp_f"]

    # Print header
    print("  {:>8s}  {:>6s} {:>4s}  {:>6s} {:>4s}  {:>6s} {:>4s}".format(
        "Time ET", "KNYC", "max", "KLGA", "max", "KEWR", "max",
    ))
    print("  " + "-" * 52)

    running_max = {}  # type: Dict[str, float]
    peak_hour = {}  # type: Dict[str, int]

    for h in range(24):
        if h not in hour_data:
            continue

        parts = ["  {:>5d}:51".format(h)]
        for s in STATIONS:
            if s in hour_data[h]:
                temp = hour_data[h][s]
                if s not in running_max or temp > running_max[s]:
                    running_max[s] = temp
                    peak_hour[s] = h
                parts.append("  {:>4.0f}{} {:>3.0f}{}".format(
                    temp, chr(176), running_max[s], chr(176),
                ))
            else:
                parts.append("  {:>11s}".format("---"))

        print("".join(parts))

    print()
    for s in STATIONS:
        if s in running_max:
            print("  {} peak: {:.0f}{}F at {}:51 ET".format(
                s, running_max[s], chr(176), peak_hour[s],
            ))

    return peak_hour


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def analyze_one_day(captured, con, target_date):
    # type: (Dict, duckdb.DuckDBPyConnection, date) -> dict
    """Run all 5 analysis sections for a single day. Returns summary stats."""
    actual_high = load_actual_high(con, target_date)

    print("=" * 72)
    print("DATE: {} -- Actual High: {}{}F".format(
        target_date, int(actual_high) if actual_high else "?", chr(176),
    ))
    print("=" * 72)

    # Load brackets
    brackets_df, kalshi_brackets = load_brackets(con, target_date)
    if not kalshi_brackets:
        print("  No valid settlement brackets found -- skipping.")
        return {}

    tickers = list(brackets_df["market_ticker"])
    labels = [
        bracket_label(row["floor_strike"], row["cap_strike"])
        for _, row in brackets_df.iterrows()
    ]
    labels_map = dict(zip(tickers, labels))

    # Count captured hours
    captured_hours = sorted(h for (d, h) in captured if d == target_date)
    print("  Brackets: {}".format(", ".join(labels)))
    print("  Model captured: {} hours ({})".format(
        len(captured_hours),
        ", ".join(str(h) for h in captured_hours) if captured_hours else "none",
    ))

    # Check if winning bracket is present
    winning = [kb for kb in kalshi_brackets if kb.settled_yes]
    if not winning:
        print("  WARNING: No bracket settled YES -- Brier scores will be incorrect.")

    # Load market data
    candles_df = load_candles_for_date(con, tickers, target_date)
    print("  Candles on settlement day: {} rows".format(len(candles_df)))

    market_timeline = build_market_timeline(candles_df, tickers, target_date)
    print("  Market timeline samples: {}".format(len(market_timeline)))

    # Load observations
    obs_df = load_observations(con, target_date)
    print("  Observation rows: {}".format(len(obs_df)))

    # Section 1: Probability timeline
    print_probability_timeline(
        captured, market_timeline, brackets_df,
        kalshi_brackets, tickers, labels, target_date,
    )

    # Section 2: Divergence events
    events = find_divergence_events(
        captured, market_timeline, kalshi_brackets,
        tickers, labels_map, target_date,
    )

    # Section 3: Brier comparison
    brier_pairs = print_brier_comparison(
        captured, market_timeline, kalshi_brackets,
        tickers, target_date,
    )

    # Section 4: Executable edge
    pos_edges, total_edges = print_executable_edge(
        captured, market_timeline, kalshi_brackets,
        tickers, labels, target_date,
    )

    # Section 5: Observation timeline
    peak_hours = print_observation_timeline(obs_df, target_date, actual_high)

    print()

    return {
        "events": events,
        "brier_pairs": brier_pairs,
        "pos_edges": pos_edges,
        "total_edges": total_edges,
        "peak_hours": peak_hours,
        "actual_high": actual_high,
    }


def print_cross_day_summary(all_stats):
    # type: (List[dict]) -> None
    """Print aggregate summary across all analyzed days."""
    print("=" * 72)
    print("CROSS-DAY SUMMARY")
    print("=" * 72)

    # ── Average Brier by hour ────────────────────────────────────────
    all_pairs = []
    for stats in all_stats:
        all_pairs.extend(stats.get("brier_pairs", []))

    if all_pairs:
        by_hour = defaultdict(list)  # type: Dict[int, List[Tuple[float, float]]]
        for h, mb, mkb in all_pairs:
            by_hour[h].append((mb, mkb))

        print("\n  Average Brier by hour (across {} days):".format(len(all_stats)))
        print("  {:>5s}  {:>12s}  {:>12s}  {:>8s}  {:>3s}".format(
            "Hour", "Model", "Market", "Gap", "N",
        ))
        print("  " + "-" * 47)
        for h in sorted(by_hour.keys()):
            pairs = by_hour[h]
            avg_m = sum(p[0] for p in pairs) / len(pairs)
            avg_mk = sum(p[1] for p in pairs) / len(pairs)
            gap = avg_m - avg_mk
            print("  {:>2d}:00  {:>12.4f}  {:>12.4f}  {:>+8.4f}  {:>3d}".format(
                h, avg_m, avg_mk, gap, len(pairs),
            ))

        overall_m = sum(p[1] for p in all_pairs) / len(all_pairs)
        overall_mk = sum(p[2] for p in all_pairs) / len(all_pairs)
        overall_gap = overall_m - overall_mk
        print("  " + "-" * 47)
        print("  {:>5s}  {:>12.4f}  {:>12.4f}  {:>+8.4f}  {:>3d}".format(
            "ALL", overall_m, overall_mk, overall_gap, len(all_pairs),
        ))
    else:
        print("\n  No Brier comparison data available.")

    # ── Divergence summary ───────────────────────────────────────────
    all_events = []
    for stats in all_stats:
        all_events.extend(stats.get("events", []))
    total_div = len(all_events)
    live_div = sum(1 for e in all_events if not e["is_stale"])
    stale_div = total_div - live_div
    print("\n  Divergence events (>{}pp): {} total ({} live, {} stale)".format(
        int(DIVERGENCE_THRESHOLD_PP), total_div, live_div, stale_div,
    ))

    # ── Peak observation timing ──────────────────────────────────────
    peak_hours_list = [s.get("peak_hours", {}) for s in all_stats]
    if any(ph.get("KNYC") is not None for ph in peak_hours_list):
        knyc_peaks = [ph["KNYC"] for ph in peak_hours_list if "KNYC" in ph]
        avg_peak = sum(knyc_peaks) / len(knyc_peaks)
        print("\n  KNYC peak hour (avg across {} days): {:.1f}:51 ET".format(
            len(knyc_peaks), avg_peak,
        ))

    # ── Executable edge summary ──────────────────────────────────────
    total_pos = sum(s.get("pos_edges", 0) for s in all_stats)
    total_all = sum(s.get("total_edges", 0) for s in all_stats)
    print("\n  Executable edge: {}/{} bracket-hours showed positive edge after spread".format(
        total_pos, total_all,
    ))
    if total_all > 0:
        print("  ({:.1f}% of bracket-hours)".format(total_pos / total_all * 100))

    print()


def main():
    print("=" * 72)
    print("DAY MICROSCOPE -- Minute-Level Model vs Market Comparison")
    print("=" * 72)
    print("  Target dates:    {}".format(", ".join(str(d) for d in TARGET_DATES)))
    print("  Update hours:    0-23 ET (every hour)")
    print("  Eval window:     {} to {}".format(START_DATE, END_DATE))
    print()

    # ══ Phase A: Capture model distributions ═════════════════════════
    print("[Phase A] Running backtester to capture model distributions...")
    print("  (Evaluating {} to {})".format(START_DATE, END_DATE))
    t0 = time.time()
    captured, bt_result = run_backtest_with_capture(TARGET_DATES)
    elapsed = time.time() - t0

    print("  Done in {:.0f}s -- {} total evals, {} captured distributions".format(
        elapsed, bt_result.total_evaluations, len(captured),
    ))

    for d in TARGET_DATES:
        hours = sorted(h for (dd, h) in captured if dd == d)
        print("  {}: {} hours ({})".format(
            d, len(hours),
            ", ".join(str(h) for h in hours) if hours else "none",
        ))
    print()

    # ══ Phase B + C: Per-day analysis ════════════════════════════════
    con = duckdb.connect(DB_PATH, read_only=True)
    all_stats = []

    try:
        for target_date in TARGET_DATES:
            stats = analyze_one_day(captured, con, target_date)
            if stats:
                all_stats.append(stats)
    finally:
        con.close()

    # ══ Cross-day summary ════════════════════════════════════════════
    if all_stats:
        print_cross_day_summary(all_stats)

    print("Done.")


if __name__ == "__main__":
    main()
