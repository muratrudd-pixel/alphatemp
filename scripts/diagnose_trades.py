"""Diagnose where P&L is lost at trade level.

Runs the strategy backtester with OLS and breaks down trades by:
- Entry price, model probability, displacement
- Win rate by bucket
- Gross P&L vs fees
- Bracket type (tail vs interior)
- Trades per day
"""

import sys
from collections import defaultdict
from datetime import date

from loguru import logger

from core.db import DEFAULT_DB_PATH
from services.backtester import wf_regression_full
from services.strategy_backtester import BacktestConfig, StrategyBacktester


def bucket(val, edges):
    # type: (float, list) -> str
    """Assign val to a bucket label based on edges."""
    for i in range(len(edges) - 1):
        if val < edges[i + 1]:
            return "{:.0f}-{:.0f}".format(edges[i] * 100, edges[i + 1] * 100)
    return "{}+".format(int(edges[-1] * 100))


def analyze(trades):
    """Print detailed trade diagnostics."""
    if not trades:
        print("No trades to analyze.")
        return

    # Separate settlement trades from model-shift exits
    settlements = [t for t in trades if t.exit_type == "settlement"]
    model_shifts = [t for t in trades if t.exit_type == "model_shift"]

    print("\n" + "=" * 65)
    print("TRADE-LEVEL DIAGNOSTIC ({} trades)".format(len(trades)))
    print("=" * 65)

    print("\n--- Trade Types ---")
    print("  Settlement trades: {}".format(len(settlements)))
    print("  Model-shift exits: {}".format(len(model_shifts)))

    # ── 1. Entry price distribution ──
    print("\n--- Entry Price Distribution (ask cents) ---")
    price_edges = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    price_buckets = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0, "fees": 0.0})
    for t in settlements:
        price = t.entry_price / 100.0  # cents to probability
        b = bucket(price, price_edges)
        price_buckets[b]["n"] += 1
        price_buckets[b]["wins"] += 1 if t.settlement_result == 1 else 0
        price_buckets[b]["pnl"] += t.pnl
        price_buckets[b]["fees"] += t.fees_paid

    print("  {:>10s}  {:>5s}  {:>6s}  {:>10s}  {:>8s}  {:>10s}".format(
        "Price¢", "N", "Win%", "PnL¢", "Fees¢", "Net¢/trade"))
    for b_label in sorted(price_buckets.keys()):
        d = price_buckets[b_label]
        wr = d["wins"] / d["n"] * 100 if d["n"] else 0
        avg = d["pnl"] / d["n"] if d["n"] else 0
        print("  {:>10s}  {:>5d}  {:>5.1f}%  {:>10.1f}  {:>8.1f}  {:>10.2f}".format(
            b_label, d["n"], wr, d["pnl"], d["fees"], avg))

    # ── 2. Model probability at entry ──
    print("\n--- Model Probability at Entry ---")
    prob_edges = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    prob_buckets = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    for t in settlements:
        mp = t.model_prob_at_entry
        b = bucket(mp, prob_edges)
        prob_buckets[b]["n"] += 1
        prob_buckets[b]["wins"] += 1 if t.settlement_result == 1 else 0
        prob_buckets[b]["pnl"] += t.pnl

    print("  {:>10s}  {:>5s}  {:>6s}  {:>10s}".format("ModelP%", "N", "Win%", "PnL¢"))
    for b_label in sorted(prob_buckets.keys()):
        d = prob_buckets[b_label]
        wr = d["wins"] / d["n"] * 100 if d["n"] else 0
        print("  {:>10s}  {:>5d}  {:>5.1f}%  {:>10.1f}".format(
            b_label, d["n"], wr, d["pnl"]))

    # ── 3. Displacement at entry ──
    print("\n--- Displacement at Entry ---")
    disp_edges = [0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0]
    disp_buckets = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    for t in settlements:
        d_val = t.displacement_at_entry
        b = bucket(d_val, disp_edges)
        disp_buckets[b]["n"] += 1
        disp_buckets[b]["wins"] += 1 if t.settlement_result == 1 else 0
        disp_buckets[b]["pnl"] += t.pnl

    print("  {:>10s}  {:>5s}  {:>6s}  {:>10s}".format("Disp%", "N", "Win%", "PnL¢"))
    for b_label in sorted(disp_buckets.keys()):
        d = disp_buckets[b_label]
        wr = d["wins"] / d["n"] * 100 if d["n"] else 0
        print("  {:>10s}  {:>5d}  {:>5.1f}%  {:>10.1f}".format(
            b_label, d["n"], wr, d["pnl"]))

    # ── 4. Bracket type ──
    print("\n--- By Bracket Type ---")
    for label, filter_fn in [
        ("Lower tail", lambda t: t.bracket[0] is None),
        ("Interior", lambda t: t.bracket[0] is not None and t.bracket[1] is not None),
        ("Upper tail", lambda t: t.bracket[1] is None),
    ]:
        subset = [t for t in settlements if filter_fn(t)]
        if not subset:
            continue
        wins = sum(1 for t in subset if t.settlement_result == 1)
        total_pnl = sum(t.pnl for t in subset)
        print("  {}: {} trades, {:.1f}% win, PnL {:.1f}¢".format(
            label, len(subset), wins / len(subset) * 100, total_pnl))

    # ── 5. Gross P&L vs fees ──
    print("\n--- P&L Decomposition ---")
    total_pnl = sum(t.pnl for t in settlements)
    total_fees = sum(t.fees_paid for t in settlements)
    # Gross = what we'd have earned without fees
    # Settlement pnl already includes fee deduction at entry, so gross = pnl + fees
    gross_pnl = total_pnl + total_fees
    print("  Gross P&L (before fees): {:.1f}¢".format(gross_pnl))
    print("  Total fees paid:         {:.1f}¢".format(total_fees))
    print("  Net P&L:                 {:.1f}¢".format(total_pnl))
    print("  Fee drag:                {:.1f}%".format(
        total_fees / abs(gross_pnl) * 100 if gross_pnl != 0 else 0))

    # ── 6. Trades per day ──
    print("\n--- Trades Per Day ---")
    days = defaultdict(int)  # type: dict
    for t in settlements:
        days[t.event_date] += 1
    if days:
        counts = list(days.values())
        counts.sort()
        print("  Days with trades: {}".format(len(days)))
        print("  Avg trades/day:   {:.1f}".format(sum(counts) / len(counts)))
        print("  Max trades/day:   {}".format(max(counts)))
        print("  Median:           {}".format(counts[len(counts) // 2]))

    # ── 7. Win/loss amounts ──
    print("\n--- Win vs Loss Size ---")
    wins = [t for t in settlements if t.settlement_result == 1]
    losses = [t for t in settlements if t.settlement_result == 0]
    if wins:
        avg_win = sum(t.pnl for t in wins) / len(wins)
        print("  Avg win:  {:.1f}¢ (n={})".format(avg_win, len(wins)))
    if losses:
        avg_loss = sum(t.pnl for t in losses) / len(losses)
        print("  Avg loss: {:.1f}¢ (n={})".format(avg_loss, len(losses)))
        # What did losing trades buy at?
        avg_loss_price = sum(t.entry_price for t in losses) / len(losses)
        print("  Avg losing entry price: {:.1f}¢".format(avg_loss_price))

    # ── 8. Model-shift exits ──
    if model_shifts:
        print("\n--- Model-Shift Exits ---")
        shift_pnl = sum(t.pnl for t in model_shifts)
        shift_fees = sum(t.fees_paid for t in model_shifts)
        print("  Count: {}".format(len(model_shifts)))
        print("  Total P&L: {:.1f}¢".format(shift_pnl))
        print("  Total fees: {:.1f}¢".format(shift_fees))

    # ── 9. Example losing trades ──
    print("\n--- Sample Losing Trades (first 10) ---")
    losing = [t for t in settlements if t.settlement_result == 0][:10]
    for t in losing:
        print("  {} bracket={} entry={:.0f}¢ model_p={:.1%} disp={:.1%} pnl={:.1f}¢".format(
            t.event_date, t.bracket, t.entry_price,
            t.model_prob_at_entry, t.displacement_at_entry, t.pnl))

    print("\n" + "=" * 65)


def main():
    logger.info("Running strategy backtest for trade diagnosis...")

    config = BacktestConfig(
        starting_capital=100.0,
        burn_in_days=90,
        fixed_bet_size=1,
        start_date=date(2023, 6, 1),
        end_date=date(2026, 2, 1),
    )

    bt = StrategyBacktester(db_path=DEFAULT_DB_PATH, config=config)
    report = bt.run(wf_regression_full)

    trades = report.get("trades", [])
    analyze(trades)


if __name__ == "__main__":
    main()
