#!/usr/bin/env python3
"""Jan 9 Information Flow Diagnostic.

Cross-references HRRR forecast highs, GFS/ECMWF, observations, and
minute-level market data to understand why the market priced the winning
bracket at 65% by noon while our model was at 30%.

Usage:
    PYTHONPATH=. venv/bin/python scripts/jan9_diagnostic.py
"""

import duckdb
import pandas as pd
from datetime import date

DB_PATH = "data/alphatemp.duckdb"
TARGET_DATE = date(2025, 1, 9)
WINNING_TICKER = "KXHIGHNY-25JAN09-B32.5"  # 32-33 bracket (actual high = 33F)


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():
    con = duckdb.connect(DB_PATH, read_only=True)

    # ── 1. HRRR forecast highs by run hour ──────────────────────────────
    section("1. HRRR FORECAST HIGHS BY RUN HOUR (Jan 9)")
    print("  Shows MAX(temp_f) per HRRR run, filtered to settlement window")
    print("  Settlement window: (run_hour + fxx) >= 5 AND < 29")
    print("  Extended runs (00z/06z/12z/18z) have 48h horizon")
    print()

    hrrr = con.execute("""
        SELECT
            EXTRACT(HOUR FROM model_run) AS run_hour,
            COUNT(*) AS n_fxx,
            MIN(fxx) AS min_fxx,
            MAX(fxx) AS max_fxx,
            MAX(temp_f) AS fcst_high_f,
            ROUND(AVG(temp_f), 1) AS avg_f
        FROM forecasts
        WHERE station_id = 'KNYC'
          AND model_name = 'hrrr'
          AND CAST(model_run AS DATE) = '2025-01-09'
          AND (EXTRACT(HOUR FROM model_run) + fxx) >= 5
          AND (EXTRACT(HOUR FROM model_run) + fxx) < 29
        GROUP BY 1
        ORDER BY 1
    """).df()

    print("  {:<8s} {:>8s} {:>8s} {:>10s} {:>10s} {:>8s}".format(
        "Run (z)", "N fxx", "fxx range", "Fcst High", "Avg F", "Extended?"
    ))
    print("  " + "-" * 60)
    for _, row in hrrr.iterrows():
        rh = int(row["run_hour"])
        extended = "YES" if rh in (0, 6, 12, 18) else ""
        fxx_range = "{}-{}".format(int(row["min_fxx"]), int(row["max_fxx"]))
        print("  {:>4d}z    {:>5d}    {:>8s}    {:>7.1f}F    {:>7.1f}F    {}".format(
            rh, int(row["n_fxx"]), fxx_range, row["fcst_high_f"], row["avg_f"], extended
        ))

    # ── 1b. Drill into 12z specifically ─────────────────────────────────
    print()
    print("  --- 12z run detail (every fxx) ---")
    detail_12z = con.execute("""
        SELECT
            fxx,
            temp_f,
            (12 + fxx) AS valid_hour_utc
        FROM forecasts
        WHERE station_id = 'KNYC'
          AND model_name = 'hrrr'
          AND CAST(model_run AS DATE) = '2025-01-09'
          AND EXTRACT(HOUR FROM model_run) = 12
        ORDER BY fxx
    """).df()

    print("  {:<6s} {:>8s} {:>12s} {:>10s}".format(
        "fxx", "Temp F", "Valid UTC", "In Window?"
    ))
    for _, row in detail_12z.iterrows():
        fxx = int(row["fxx"])
        valid_h = int(row["valid_hour_utc"])
        in_window = "YES" if 5 <= valid_h < 29 else "no"
        # Convert to ET for readability
        et_hour = valid_h - 5  # EST offset
        et_str = "{}:00 ET".format(et_hour) if et_hour >= 0 else "prev day"
        print("  {:>4d}   {:>7.1f}F   {:>4d}z ({:>8s})   {}".format(
            fxx, row["temp_f"], valid_h, et_str, in_window
        ))

    # ── 2. GFS and ECMWF forecast highs ─────────────────────────────────
    section("2. GFS + ECMWF FORECAST HIGHS (Jan 9)")

    for model in ["gfs", "ecmwf"]:
        model_data = con.execute("""
            SELECT
                EXTRACT(HOUR FROM model_run) AS run_hour,
                CAST(model_run AS DATE) AS run_date,
                COUNT(*) AS n_fxx,
                MAX(temp_f) AS fcst_high_f,
                ROUND(AVG(temp_f), 1) AS avg_f
            FROM forecasts
            WHERE station_id = 'KNYC'
              AND model_name = ?
              AND CAST(model_run AS DATE) BETWEEN '2025-01-08' AND '2025-01-09'
              AND (EXTRACT(HOUR FROM model_run) + fxx) >= 5
              AND (EXTRACT(HOUR FROM model_run) + fxx) < 29
            GROUP BY 1, 2
            ORDER BY 2, 1
        """, [model]).df()

        print()
        print("  {} forecasts:".format(model.upper()))
        if model_data.empty:
            print("    (no data)")
        else:
            print("  {:<10s} {:<8s} {:>5s} {:>10s} {:>8s}".format(
                "Run Date", "Run Hz", "N", "Fcst High", "Avg F"
            ))
            for _, row in model_data.iterrows():
                print("  {:<10s} {:>4d}z    {:>5d} {:>8.1f}F   {:>7.1f}F".format(
                    str(row["run_date"]), int(row["run_hour"]),
                    int(row["n_fxx"]), row["fcst_high_f"], row["avg_f"],
                ))

    # ── 3. KNYC observations through the day ────────────────────────────
    section("3. KNYC OBSERVATIONS (Jan 9)")
    print("  Timestamps are UTC in DB, showing ET conversion")
    print()

    obs = con.execute("""
        SELECT
            observed_at,
            temp_f
        FROM observations
        WHERE station_id = 'KNYC'
          AND CAST(observed_at AS DATE) = '2025-01-09'
        ORDER BY observed_at
    """).df()

    running_max = None
    print("  {:<20s} {:>8s} {:>10s}".format(
        "Time (UTC → ET)", "Temp F", "Run Max"
    ))
    print("  " + "-" * 42)
    for _, row in obs.iterrows():
        ts = row["observed_at"]
        # UTC to ET (EST = UTC-5 in January)
        utc_str = str(ts)[11:16]
        et_hour = ts.hour - 5
        if et_hour < 0:
            et_hour += 24
            et_str = "prev {:02d}:{:02d}".format(et_hour, ts.minute)
        else:
            et_str = "{:02d}:{:02d} ET".format(et_hour, ts.minute)

        temp = row["temp_f"]
        if pd.isna(temp):
            continue
        if running_max is None or temp > running_max:
            running_max = temp
            marker = " ← NEW MAX"
        else:
            marker = ""

        print("  {} ({}z)  {:>6.1f}F  {:>6.1f}F{}".format(
            et_str, utc_str, temp, running_max, marker
        ))

    # ── 4. Market price evolution for winning bracket ───────────────────
    section("4. MARKET PRICES — WINNING BRACKET ({})".format(WINNING_TICKER))
    print("  Showing price evolution through Jan 9")
    print()

    candles = con.execute("""
        SELECT
            end_period_ts,
            yes_bid_close,
            yes_ask_close,
            price_close,
            volume
        FROM kalshi_candlesticks
        WHERE market_ticker = ?
          AND CAST(end_period_ts AS DATE) = '2025-01-09'
        ORDER BY end_period_ts
    """, [WINNING_TICKER]).df()

    if candles.empty:
        print("  (no candlestick data for this ticker on Jan 9)")
    else:
        # Sample every 30 min for readability, plus show the first jump
        print("  {:<12s} {:>6s} {:>6s} {:>8s} {:>6s}".format(
            "Time ET", "Bid", "Ask", "Mid", "Vol"
        ))
        print("  " + "-" * 45)
        prev_mid = None
        for _, row in candles.iterrows():
            ts = row["end_period_ts"]
            hour = ts.hour
            minute = ts.minute

            bid = row["yes_bid_close"] if pd.notna(row["yes_bid_close"]) else 0
            ask = row["yes_ask_close"] if pd.notna(row["yes_ask_close"]) else 0
            mid = (bid + ask) / 2.0
            vol = int(row["volume"]) if pd.notna(row["volume"]) else 0

            # Show: every 30 min, OR if price moved >5c from last shown
            show = (minute % 30 == 0) or (prev_mid is not None and abs(mid - prev_mid) > 5)
            # Always show first and last
            show = show or prev_mid is None

            if show:
                jump = ""
                if prev_mid is not None and abs(mid - prev_mid) > 5:
                    jump = " ← JUMP {:+.0f}c".format(mid - prev_mid)
                print("  {:>02d}:{:02d} ET    {:>4.0f}c  {:>4.0f}c   {:>5.1f}c  {:>5d}{}".format(
                    hour, minute, bid, ask, mid, vol, jump
                ))
                prev_mid = mid

    # ── 5. All bracket prices at key moments ────────────────────────────
    section("5. ALL BRACKET PRICES AT KEY MOMENTS")
    print("  Snapshots at 09:00, 10:00, 12:00, 14:00, 16:00 ET")
    print()

    for snapshot_hour in [9, 10, 12, 14, 16]:
        bracket_snap = con.execute("""
            WITH last_candle AS (
                SELECT
                    c.market_ticker,
                    s.floor_strike,
                    s.cap_strike,
                    s.settled_yes,
                    c.yes_bid_close,
                    c.yes_ask_close,
                    c.price_close,
                    ROW_NUMBER() OVER (
                        PARTITION BY c.market_ticker
                        ORDER BY c.end_period_ts DESC
                    ) AS rn
                FROM kalshi_candlesticks c
                JOIN kalshi_settlements s ON s.market_ticker = c.market_ticker
                WHERE s.event_date = '2025-01-09'
                  AND s.city = 'NYC'
                  AND s.measure = 'high'
                  AND CAST(c.end_period_ts AS DATE) = '2025-01-09'
                  AND EXTRACT(HOUR FROM c.end_period_ts) <= ?
            )
            SELECT
                market_ticker,
                floor_strike,
                cap_strike,
                settled_yes,
                yes_bid_close,
                yes_ask_close,
                COALESCE(price_close, (yes_bid_close + yes_ask_close) / 2.0) AS mid
            FROM last_candle
            WHERE rn = 1
            ORDER BY floor_strike NULLS FIRST
        """, [snapshot_hour]).df()

        print("  --- {:02d}:00 ET ---".format(snapshot_hour))
        if bracket_snap.empty:
            print("    (no data)")
        else:
            total_ask = 0
            for _, row in bracket_snap.iterrows():
                floor_s = row["floor_strike"]
                cap_s = row["cap_strike"]
                if pd.isna(floor_s):
                    label = "<{:.0f}".format(cap_s)
                elif pd.isna(cap_s):
                    label = ">{:.0f}".format(floor_s)
                else:
                    label = "{:.0f}-{:.0f}".format(floor_s, cap_s)
                won = "WIN" if row["settled_yes"] == 1 else ""
                bid = row["yes_bid_close"] if pd.notna(row["yes_bid_close"]) else 0
                ask = row["yes_ask_close"] if pd.notna(row["yes_ask_close"]) else 0
                mid = row["mid"] if pd.notna(row["mid"]) else 0
                total_ask += ask
                print("    {:<8s}  bid {:>3.0f}c  ask {:>3.0f}c  mid {:>5.1f}c  {}".format(
                    label, bid, ask, mid, won
                ))
            print("    Sum of asks: {:.0f}c (vig = {:.0f}c)".format(total_ask, total_ask - 100))
        print()

    # ── 6. Timeline cross-reference ─────────────────────────────────────
    section("6. TIMELINE: WHAT HAPPENED WHEN")
    print()
    print("  Combining HRRR availability, obs, and market moves")
    print()

    # HRRR availability: model_run hour + ~2h latency
    print("  {:<12s}  {:<50s}".format("Time ET", "Event"))
    print("  " + "-" * 64)

    events = []

    # HRRR runs (approximate availability = run_hour + 2h)
    for _, row in hrrr.iterrows():
        rh = int(row["run_hour"])
        avail_et = rh - 5 + 2  # UTC to ET + 2h latency
        if avail_et < 0:
            avail_et += 24
        events.append((avail_et, 0, "HRRR {:02d}z available — fcst high {:.1f}F ({} fxx)".format(
            rh, row["fcst_high_f"], int(row["n_fxx"])
        )))

    # Observations
    for _, row in obs.iterrows():
        et_hour = row["observed_at"].hour - 5
        et_min = row["observed_at"].minute
        if et_hour < 0:
            et_hour += 24
        events.append((et_hour, et_min, "KNYC obs: {:.1f}F".format(row["temp_f"])))

    # Market jumps (>5c moves on winning bracket)
    prev_mid = None
    for _, row in candles.iterrows():
        bid = row["yes_bid_close"] if pd.notna(row["yes_bid_close"]) else 0
        ask = row["yes_ask_close"] if pd.notna(row["yes_ask_close"]) else 0
        mid = (bid + ask) / 2.0
        if prev_mid is not None and abs(mid - prev_mid) > 5:
            events.append((row["end_period_ts"].hour, row["end_period_ts"].minute,
                          "MARKET {}: {:+.0f}c → {:.0f}c mid".format(
                              WINNING_TICKER.split("-")[-1], mid - prev_mid, mid
                          )))
        prev_mid = mid

    # Sort and print
    events.sort(key=lambda x: (x[0], x[1]))
    for et_h, et_m, desc in events:
        if 4 <= et_h <= 22:  # only show daytime
            print("  {:02d}:{:02d} ET    {}".format(et_h, et_m, desc))

    con.close()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
