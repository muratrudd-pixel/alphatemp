#!/usr/bin/env python3
"""Apples-to-Apples: Model vs Market Brier Comparison.

Scores our v2 model and the Kalshi market on the **exact same brackets**
for the **exact same (date, update_hour)** pairs.  Only dates where every
bracket has candlestick data are included (full coverage filter).

Usage:
    PYTHONPATH=. venv/bin/python scripts/brier_comparison.py
"""

import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta

import duckdb
import pandas as pd

sys.path.insert(0, ".")

from services.backtester import Backtester, wf_qr_multimodel_v2

DB_PATH = "data/alphatemp.duckdb"
UPDATE_HOURS = [12, 16, 20]
RUN_HOURS = [12]
START_DATE = date.today() - timedelta(days=730)


def load_settlements(con):
    # type: (duckdb.DuckDBPyConnection) -> pd.DataFrame
    """Load all NYC high settlements with bracket info."""
    return con.execute("""
        SELECT market_ticker, event_date, floor_strike, cap_strike, settled_yes
        FROM kalshi_settlements
        WHERE city = 'NYC' AND measure = 'high'
        ORDER BY event_date, floor_strike NULLS FIRST
    """).df()


def load_last_candle_per_hour(con):
    # type: (duckdb.DuckDBPyConnection) -> pd.DataFrame
    """For each (market_ticker, hour), get the last candle in that hour.

    Candlestick timestamps are stored in ET (naive).  We extract the hour
    directly — no UTC→ET conversion needed.
    """
    return con.execute("""
        WITH ranked AS (
            SELECT
                market_ticker,
                CAST(end_period_ts AS DATE) AS candle_date,
                EXTRACT(HOUR FROM end_period_ts) AS hour_et,
                yes_bid_close,
                yes_ask_close,
                price_close,
                ROW_NUMBER() OVER (
                    PARTITION BY market_ticker,
                                 CAST(end_period_ts AS DATE),
                                 EXTRACT(HOUR FROM end_period_ts)
                    ORDER BY end_period_ts DESC
                ) AS rn
            FROM kalshi_candlesticks
        )
        SELECT
            market_ticker,
            candle_date,
            CAST(hour_et AS INTEGER) AS hour_et,
            COALESCE(price_close, (yes_bid_close + yes_ask_close) / 2.0) AS price_cents
        FROM ranked
        WHERE rn = 1
    """).df()


def compute_market_brier(settlements_df, candles_df, target_date, hour_et):
    # type: (pd.DataFrame, pd.DataFrame, date, int) -> float | None
    """Compute market Brier for one (date, hour) pair.

    Returns None if any bracket is missing candle data (full coverage filter).
    """
    brackets = settlements_df[settlements_df["event_date"] == target_date]
    if brackets.empty:
        return None

    tickers = set(brackets["market_ticker"])
    matched = candles_df[
        (candles_df["market_ticker"].isin(tickers))
        & (candles_df["candle_date"] == target_date)
        & (candles_df["hour_et"] == hour_et)
    ]

    # Full coverage filter: every bracket must have a candle
    if set(matched["market_ticker"]) != tickers:
        return None

    # Merge to get settled_yes alongside market price
    merged = matched.merge(
        brackets[["market_ticker", "settled_yes"]],
        on="market_ticker",
        how="inner",
    )

    # Brier: sum of (market_prob - outcome)^2 across brackets
    market_prob = merged["price_cents"] / 100.0
    outcome = merged["settled_yes"].astype(float)
    brier = ((market_prob - outcome) ** 2).sum()
    return float(brier)


def main():
    print("=" * 70)
    print("APPLES-TO-APPLES: Model vs Market Brier Comparison")
    print("=" * 70)
    print("  Update hours ET: {}".format(UPDATE_HOURS))
    print("  Run hours:       {}".format(RUN_HOURS))
    print("  Start date:      {}".format(START_DATE))
    print()

    # ── Step 1: Run v2 backtester ─────────────────────────────────────
    print("[1/4] Running v2 backtester...")
    t0 = time.time()
    bt = Backtester(db_path=DB_PATH, city="NYC")
    result = bt.run(
        wf_qr_multimodel_v2,
        update_hours_et=UPDATE_HOURS,
        run_hours=RUN_HOURS,
        start_date=START_DATE,
    )
    elapsed = time.time() - t0
    print("  Done in {:.0f}s — {} evals, mean Brier {:.4f}".format(
        elapsed, result.total_evaluations, result.mean_brier,
    ))
    print()

    # ── Step 2: Split by scoring method ───────────────────────────────
    kalshi_evals = [r for r in result.run_results if r.used_kalshi_brackets]
    fallback_evals = [r for r in result.run_results if not r.used_kalshi_brackets]

    kalshi_brier = (
        sum(r.brier_score for r in kalshi_evals) / len(kalshi_evals)
        if kalshi_evals else 0.0
    )
    fallback_brier = (
        sum(r.brier_score for r in fallback_evals) / len(fallback_evals)
        if fallback_evals else 0.0
    )
    combined_brier = result.mean_brier

    print("[2/4] Scoring method split:")
    print("  Kalshi-scored: {:>5d} evals, Brier {:.4f}".format(
        len(kalshi_evals), kalshi_brier,
    ))
    print("  1F fallback:   {:>5d} evals, Brier {:.4f}".format(
        len(fallback_evals), fallback_brier,
    ))
    print("  Combined:      {:>5d} evals, Brier {:.4f}".format(
        len(kalshi_evals) + len(fallback_evals), combined_brier,
    ))
    # Sanity check
    total = len(kalshi_evals) + len(fallback_evals)
    assert total == result.total_evaluations, (
        "Split count mismatch: {} + {} != {}".format(
            len(kalshi_evals), len(fallback_evals), result.total_evaluations,
        )
    )
    print("  (sanity check: {} + {} = {} total — OK)".format(
        len(kalshi_evals), len(fallback_evals), total,
    ))
    print()

    # ── Step 3: Build market Brier for matching pairs ─────────────────
    print("[3/4] Computing market Brier from candlesticks...")
    t0 = time.time()

    con = duckdb.connect(DB_PATH, read_only=True)
    settlements_df = load_settlements(con)
    candles_df = load_last_candle_per_hour(con)
    con.close()

    print("  Loaded {} settlement rows, {} candle snapshots".format(
        len(settlements_df), len(candles_df),
    ))

    # Convert pandas Timestamps to date for comparison
    settlements_df["event_date"] = pd.to_datetime(
        settlements_df["event_date"]
    ).dt.date
    candles_df["candle_date"] = pd.to_datetime(
        candles_df["candle_date"]
    ).dt.date

    # Group Kalshi-scored evals by (date, update_hour_et)
    # For each group, compute both model mean Brier and market Brier
    # type: Dict[int, List[dict]]
    by_hour = defaultdict(list)  # hour -> list of {model_brier, market_brier}

    matched_count = 0
    skipped_no_candle = 0

    for r in kalshi_evals:
        market_brier = compute_market_brier(
            settlements_df, candles_df, r.settlement_date, r.update_hour_et,
        )
        if market_brier is None:
            skipped_no_candle += 1
            continue

        matched_count += 1
        by_hour[r.update_hour_et].append({
            "model_brier": r.brier_score,
            "market_brier": market_brier,
            "date": r.settlement_date,
        })

    elapsed = time.time() - t0
    print("  Done in {:.0f}s — {} matched, {} skipped (missing candles)".format(
        elapsed, matched_count, skipped_no_candle,
    ))
    print()

    # ── Step 4: Report ────────────────────────────────────────────────
    print("=" * 70)
    print("SCORING METHOD SPLIT")
    print("=" * 70)
    print("  Kalshi-scored:   {:>5d} evals, Brier {:.4f}".format(
        len(kalshi_evals), kalshi_brier,
    ))
    print("  1F fallback:     {:>5d} evals, Brier {:.4f}".format(
        len(fallback_evals), fallback_brier,
    ))
    print("  Combined:        {:>5d} evals, Brier {:.4f}".format(
        total, combined_brier,
    ))
    print()

    print("=" * 70)
    print("APPLES-TO-APPLES (Kalshi brackets only, full candle coverage)")
    print("=" * 70)
    print()
    print("  {:<8s} {:>12s} {:>12s} {:>8s} {:>8s} {:>5s}".format(
        "ET Hour", "Model Brier", "Mkt Brier", "Gap", "Edge", "N",
    ))
    print("  " + "-" * 57)

    all_model = []
    all_market = []

    for hour in sorted(by_hour.keys()):
        entries = by_hour[hour]
        model_avg = sum(e["model_brier"] for e in entries) / len(entries)
        market_avg = sum(e["market_brier"] for e in entries) / len(entries)
        gap = market_avg - model_avg
        edge_pct = gap / market_avg * 100 if market_avg else 0.0

        all_model.extend(e["model_brier"] for e in entries)
        all_market.extend(e["market_brier"] for e in entries)

        print("  {:<8d} {:>12.4f} {:>12.4f} {:>+8.4f} {:>+7.1f}% {:>5d}".format(
            hour, model_avg, market_avg, gap, edge_pct, len(entries),
        ))

    if all_model:
        overall_model = sum(all_model) / len(all_model)
        overall_market = sum(all_market) / len(all_market)
        overall_gap = overall_market - overall_model
        overall_edge = overall_gap / overall_market * 100 if overall_market else 0.0

        print("  " + "-" * 57)
        print("  {:<8s} {:>12.4f} {:>12.4f} {:>+8.4f} {:>+7.1f}% {:>5d}".format(
            "ALL", overall_model, overall_market, overall_gap, overall_edge,
            len(all_model),
        ))
    print()
    print("  Edge = market_brier - model_brier (positive = model wins)")
    print()

    if not all_model:
        print("  WARNING: No matched (date, hour) pairs with full candle coverage.")
        print("  Cannot compute apples-to-apples comparison.")
        return

    if overall_gap > 0:
        print("  VERDICT: Model beats market by {:.4f} Brier ({:+.1f}%)".format(
            overall_gap, overall_edge,
        ))
    else:
        print("  VERDICT: Market beats model by {:.4f} Brier ({:+.1f}%)".format(
            -overall_gap, -overall_edge,
        ))


if __name__ == "__main__":
    main()
