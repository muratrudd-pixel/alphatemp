"""Mesonet ↔ KNYC correlation analysis.

Compares 5 NYC Mesonet stations against KNYC (Central Park) observations
to determine which stations best predict KNYC temperature trajectory.

Analyses:
1. Hourly temperature correlation (Pearson r) per station
2. Daily high correlation per station
3. Trend correlation — does a 30-min slope at Mesonet predict KNYC direction?
4. Lead/lag analysis — do Mesonet stations lead KNYC by any offset?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "alphatemp.duckdb"
MESONET_DIR = Path(__file__).parent.parent / "data" / "mesonet"
STATIONS = ["BKNYRD", "BXVNST", "MHCHEL", "MHLSQR", "QNASTO"]


def load_knyc():
    """Load KNYC obs from DuckDB, return DataFrame with UTC timestamps."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT observed_at AS ts, temp_f
        FROM observations
        WHERE station_id = 'KNYC' AND temp_f IS NOT NULL
        ORDER BY observed_at
    """).fetchdf()
    con.close()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def load_mesonet(station):
    """Load one Mesonet CSV, convert ET timestamps to UTC."""
    path = MESONET_DIR / f"{station}.csv"
    df = pd.read_csv(path)
    df.columns = ["station", "ts", "temp_f"]
    df = df.dropna(subset=["temp_f"])
    # Timestamps are like "2020-10-27 00:00:00 EDT" or "... EST"
    # EDT = UTC-4, EST = UTC-5. Parse manually since pandas chokes on these.
    ts_str = df["ts"].str.strip()
    is_edt = ts_str.str.endswith("EDT")
    # Strip the timezone abbreviation, parse as naive
    naive = pd.to_datetime(ts_str.str.rsplit(" ", n=1).str[0])
    # Convert to UTC: EDT -> subtract 4h, EST -> subtract 5h (add offset to get UTC)
    utc_ts = naive.copy()
    utc_ts[is_edt] = naive[is_edt] + pd.Timedelta(hours=4)
    utc_ts[~is_edt] = naive[~is_edt] + pd.Timedelta(hours=5)
    df["ts"] = utc_ts.dt.tz_localize("UTC")
    return df[["ts", "temp_f"]]


def hourly_correlation(knyc, mesonet_stations):
    """Correlate hourly temperatures: round both to nearest hour, inner join."""
    print("\n" + "=" * 60)
    print("1. HOURLY TEMPERATURE CORRELATION (Pearson r)")
    print("=" * 60)

    # Round KNYC to nearest hour
    knyc_h = knyc.copy()
    knyc_h["hour"] = knyc_h["ts"].dt.round("h")
    # Take the obs closest to the hour mark (deduplicate)
    knyc_h = knyc_h.sort_values("ts").groupby("hour").last().reset_index()
    knyc_h = knyc_h[["hour", "temp_f"]].rename(columns={"temp_f": "knyc_f"})

    results = []
    for name, mdf in mesonet_stations.items():
        mh = mdf.copy()
        mh["hour"] = mh["ts"].dt.round("h")
        mh = mh.sort_values("ts").groupby("hour").last().reset_index()
        mh = mh[["hour", "temp_f"]].rename(columns={"temp_f": f"{name}_f"})

        merged = knyc_h.merge(mh, on="hour", how="inner")
        if len(merged) < 100:
            print(f"  {name}: insufficient overlap ({len(merged)} rows)")
            continue

        r = merged["knyc_f"].corr(merged[f"{name}_f"])
        mae = (merged["knyc_f"] - merged[f"{name}_f"]).abs().mean()
        bias = (merged[f"{name}_f"] - merged["knyc_f"]).mean()
        results.append((name, r, mae, bias, len(merged)))

    results.sort(key=lambda x: -x[1])
    print(f"\n  {'Station':<10} {'r':>8} {'MAE':>8} {'Bias':>8} {'N hours':>10}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for name, r, mae, bias, n in results:
        print(f"  {name:<10} {r:>8.4f} {mae:>7.1f}°F {bias:>+7.1f}°F {n:>10,}")
    return results


def daily_high_correlation(knyc, mesonet_stations):
    """Correlate daily high temperatures."""
    print("\n" + "=" * 60)
    print("2. DAILY HIGH CORRELATION")
    print("=" * 60)

    # KNYC daily high (5 AM - 5 AM UTC = midnight-midnight ET roughly)
    knyc_d = knyc.copy()
    knyc_d["date"] = knyc_d["ts"].dt.tz_convert("US/Eastern").dt.date
    knyc_daily = knyc_d.groupby("date")["temp_f"].max().reset_index()
    knyc_daily.columns = ["date", "knyc_high"]

    results = []
    for name, mdf in mesonet_stations.items():
        md = mdf.copy()
        md["date"] = md["ts"].dt.tz_convert("US/Eastern").dt.date
        m_daily = md.groupby("date")["temp_f"].max().reset_index()
        m_daily.columns = ["date", f"{name}_high"]

        merged = knyc_daily.merge(m_daily, on="date", how="inner")
        if len(merged) < 100:
            print(f"  {name}: insufficient overlap ({len(merged)} days)")
            continue

        r = merged["knyc_high"].corr(merged[f"{name}_high"])
        mae = (merged["knyc_high"] - merged[f"{name}_high"]).abs().mean()
        bias = (merged[f"{name}_high"] - merged["knyc_high"]).mean()
        results.append((name, r, mae, bias, len(merged)))

    results.sort(key=lambda x: -x[1])
    print(f"\n  {'Station':<10} {'r':>8} {'MAE':>8} {'Bias':>8} {'N days':>10}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for name, r, mae, bias, n in results:
        print(f"  {name:<10} {r:>8.4f} {mae:>7.1f}°F {bias:>+7.1f}°F {n:>10,}")
    return results


def trend_correlation(knyc, mesonet_stations):
    """Does a 30-min Mesonet trend predict KNYC's next-hour direction?

    For each hour H:
    - Mesonet trend: temp at H:00 minus temp at H-0:30 (30-min delta)
    - KNYC direction: temp at H+1:00 minus temp at H:00 (next hour change)
    Correlate these across all hours.
    """
    print("\n" + "=" * 60)
    print("3. TREND PREDICTION (30-min Mesonet slope → KNYC next-hour change)")
    print("=" * 60)

    # KNYC hourly changes
    knyc_h = knyc.copy()
    knyc_h["hour"] = knyc_h["ts"].dt.round("h")
    knyc_h = knyc_h.sort_values("ts").groupby("hour").last().reset_index()
    knyc_h = knyc_h[["hour", "temp_f"]].rename(columns={"temp_f": "knyc_f"})
    knyc_h["knyc_next"] = knyc_h["knyc_f"].shift(-1)
    knyc_h["knyc_delta"] = knyc_h["knyc_next"] - knyc_h["knyc_f"]
    knyc_h = knyc_h.dropna(subset=["knyc_delta"])

    results = []
    for name, mdf in mesonet_stations.items():
        # Resample to 30-min marks
        mh = mdf.copy()
        mh["t30"] = mh["ts"].dt.round("30min")
        mh = mh.sort_values("ts").groupby("t30").last().reset_index()

        # Compute 30-min trend ending at each hour
        mh_hour = mh[mh["t30"].dt.minute == 0].copy()
        mh_half = mh[mh["t30"].dt.minute == 30].copy()
        mh_half["hour"] = mh_half["t30"] + pd.Timedelta(minutes=30)
        mh_half = mh_half[["hour", "temp_f"]].rename(columns={"temp_f": "temp_30ago"})
        mh_hour = mh_hour.rename(columns={"t30": "hour"})[["hour", "temp_f"]]

        trend = mh_hour.merge(mh_half, on="hour", how="inner")
        trend["mesonet_trend"] = trend["temp_f"] - trend["temp_30ago"]

        merged = knyc_h.merge(trend[["hour", "mesonet_trend"]], on="hour", how="inner")
        if len(merged) < 100:
            print(f"  {name}: insufficient overlap ({len(merged)} rows)")
            continue

        r = merged["knyc_delta"].corr(merged["mesonet_trend"])
        # Also check: does sign match? (both rising or both falling)
        sign_match = ((merged["knyc_delta"] > 0) == (merged["mesonet_trend"] > 0)).mean()
        results.append((name, r, sign_match, len(merged)))

    results.sort(key=lambda x: -x[1])
    print(f"\n  {'Station':<10} {'r':>8} {'Sign match':>12} {'N':>10}")
    print(f"  {'-'*10} {'-'*8} {'-'*12} {'-'*10}")
    for name, r, sm, n in results:
        print(f"  {name:<10} {r:>8.4f} {sm:>11.1%} {n:>10,}")
    return results


def lead_lag_analysis(knyc, mesonet_stations):
    """Check if Mesonet temps lead KNYC by 5, 10, 15, ... 60 min.

    For each lag offset, shift Mesonet back by that amount and correlate
    with KNYC. If correlation peaks at lag > 0, Mesonet leads.
    """
    print("\n" + "=" * 60)
    print("4. LEAD/LAG ANALYSIS (does Mesonet lead KNYC?)")
    print("=" * 60)

    # Resample KNYC to 5-min grid (forward-fill across gaps)
    knyc_5m = knyc.set_index("ts").resample("5min").last().dropna()
    knyc_5m = knyc_5m.rename(columns={"temp_f": "knyc_f"})

    lags = [0, 5, 10, 15, 20, 25, 30, 45, 60]  # minutes

    for name, mdf in mesonet_stations.items():
        m5 = mdf.set_index("ts").resample("5min").last().dropna()
        m5 = m5.rename(columns={"temp_f": f"{name}_f"})

        best_lag = 0
        best_r = -1
        lag_results = []
        for lag in lags:
            shifted = m5.copy()
            shifted.index = shifted.index + pd.Timedelta(minutes=lag)
            merged = knyc_5m.join(shifted, how="inner")
            if len(merged) < 1000:
                continue
            r = merged["knyc_f"].corr(merged[f"{name}_f"])
            lag_results.append((lag, r, len(merged)))
            if r > best_r:
                best_r = r
                best_lag = lag

        print(f"\n  {name} (best lag: {best_lag} min, r={best_r:.4f}):")
        print(f"    {'Lag':>6} {'r':>8} {'N':>10}")
        for lag, r, n in lag_results:
            marker = " <-- best" if lag == best_lag else ""
            print(f"    {lag:>4}m {r:>8.4f} {n:>10,}{marker}")


def seasonal_correlation(knyc, mesonet_stations):
    """Check if correlation varies by season (important for model features)."""
    print("\n" + "=" * 60)
    print("5. SEASONAL DAILY HIGH CORRELATION")
    print("=" * 60)

    knyc_d = knyc.copy()
    knyc_d["date"] = knyc_d["ts"].dt.tz_convert("US/Eastern").dt.date
    knyc_daily = knyc_d.groupby("date")["temp_f"].max().reset_index()
    knyc_daily.columns = ["date", "knyc_high"]
    knyc_daily["date"] = pd.to_datetime(knyc_daily["date"])
    knyc_daily["month"] = knyc_daily["date"].dt.month
    knyc_daily["season"] = knyc_daily["month"].map(
        {12: "Winter", 1: "Winter", 2: "Winter",
         3: "Spring", 4: "Spring", 5: "Spring",
         6: "Summer", 7: "Summer", 8: "Summer",
         9: "Fall", 10: "Fall", 11: "Fall"}
    )

    for name, mdf in mesonet_stations.items():
        md = mdf.copy()
        md["date"] = md["ts"].dt.tz_convert("US/Eastern").dt.date
        m_daily = md.groupby("date")["temp_f"].max().reset_index()
        m_daily.columns = ["date", f"{name}_high"]
        m_daily["date"] = pd.to_datetime(m_daily["date"])

        merged = knyc_daily.merge(m_daily, on="date", how="inner")
        if len(merged) < 100:
            continue

        print(f"\n  {name}:")
        print(f"    {'Season':<10} {'r':>8} {'MAE':>8} {'Bias':>8} {'N':>6}")
        for season in ["Winter", "Spring", "Summer", "Fall"]:
            subset = merged[merged["season"] == season]
            if len(subset) < 30:
                continue
            r = subset["knyc_high"].corr(subset[f"{name}_high"])
            mae = (subset["knyc_high"] - subset[f"{name}_high"]).abs().mean()
            bias = (subset[f"{name}_high"] - subset["knyc_high"]).mean()
            print(f"    {season:<10} {r:>8.4f} {mae:>7.1f}°F {bias:>+7.1f}°F {len(subset):>6}")


if __name__ == "__main__":
    print("Loading KNYC observations from DuckDB...")
    knyc = load_knyc()
    print(f"  {len(knyc):,} rows, {knyc['ts'].min()} to {knyc['ts'].max()}")

    print("\nLoading Mesonet CSVs...")
    mesonet = {}
    for stn in STATIONS:
        df = load_mesonet(stn)
        mesonet[stn] = df
        print(f"  {stn}: {len(df):,} rows, {df['ts'].min()} to {df['ts'].max()}")

    hourly_correlation(knyc, mesonet)
    daily_high_correlation(knyc, mesonet)
    trend_correlation(knyc, mesonet)
    lead_lag_analysis(knyc, mesonet)
    seasonal_correlation(knyc, mesonet)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
