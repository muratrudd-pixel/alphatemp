# Handoff - 2026-03-22

## Current State
- Branch: `autoresearch/run-2026-03-10`
- All tests passing (46 dashboard + 24 feature builder + 32 strategy comparison)
- Dashboard running locally (PID varies, port 8050)
- Strategy engine live with optimized config

## What We Did (2026-03-22 — Full Day Session)

### 1. Strategy Backtest Comparison Runner
Built `scripts/strategy_comparison.py` from scratch — mirrors live engine exactly (FeatureBuilder + QRModel reuse). Compared 12 strategies × 4 threshold configs on 100 season-stratified days (Oct 2024 – Feb 2026).

**Winner: Baseline strategy + max 5 contracts/bracket.** P&L $+128, Sharpe 5.41, MaxDD -$19.25.

Key finding: `portfolio_ev` (mutually-exclusive EV optimizer) underperformed baseline when using real data. The simple "enter every signal that clears edge threshold" approach wins. Contract cap (5 vs 10) is the real risk lever.

### 2. Settlement Boundary Fix
Both `paper_trader.py` and `settlement.py` used cap-exclusive brackets `[floor, cap)`. Kalshi CFTC filing says both-inclusive `[floor, cap]`. Fixed in both files + dashboard review tab.

### 3. Late-Day Distribution Clamp
Model was assigning 30% to 64-65°F at 6 PM when running max was 61°F and temps falling. Added `clamp_late_day()` in `model.py` — post-model heuristic that collapses distribution when:
- `temp_drop > 3°F` (fallen well below peak)
- `forecast_upside < 2°F` (HRRR shows no remaining warming)

Backtested: clamp improves MaxDD from -$19.25 to -$17.49, P&L unchanged.

**Also tried adding temp_drop/forecast_upside as QR features (25-feat model) — made things WORSE.** P&L dropped $20, drawdown blew limit. Reverted. Linear QR can't learn the interaction; post-model heuristic is the right approach.

### 4. CLI/DSM Settlement High as Running Max
Market prices off CLI data (~62°F) which differs from METAR max (~61°F). Added `_get_cli_high()` to strategy engine — when CLI/DSM publishes, uses it as running_max for the clamp. Fixes the late-day distribution collapse to match what the market sees.

### 5. Edge Reversal Exits — Disabled
Tested three exit approaches:
1. **Old (vs current market):** Bad — exits profitable positions when market catches up to model
2. **Entry-price based (model_prob < entry/100):** P&L $107 vs $126 hold-to-settlement. Worse.
3. **High-conviction (model_prob < entry/200):** P&L $114 vs $126. Better but still worse than holding.

**Hold-to-settlement wins.** Model fluctuates too much intra-day for reliable exit decisions. Exit logic preserved but disabled (`if False`).

### 6. Config Changes
- `max_per_bracket`: was 2 in DB, updated to **5** (backtested optimal)
- `paper_positions`: wiped clean (old data from pre-strategy-change)
- `market_ticks`: wiped clean (old data feeding stale review tab)
- Tomorrow toggle removed from dashboard

### 7. Dashboard Fixes
- CLI settlement dot placed at time of peak METAR obs (was hardcoded noon)
- Review tab sorts newest-first (was severity-first)
- Settlement boundary fix in missed_edge detection

## Files Modified This Session
- `services/strategy_engine.py` — CLI high, clamp wiring, exit logic disabled
- `services/model.py` — `clamp_late_day()` function
- `services/feature_builder.py` — reverted to 23 features (25-feat attempt failed)
- `services/paper_trader.py` — settlement boundary fix
- `services/settlement.py` — settlement boundary fix + tail handling
- `ui/web_dashboard.py` — CLI dot timing, review sort, settlement boundary
- `templates/base.html` — removed Tomorrow toggle
- `scripts/strategy_comparison.py` — NEW: full backtest runner
- `scripts/backtest_25feat.py` — NEW: quick A/B backtest script
- `tests/test_strategy_comparison.py` — NEW: 32 tests
- `tests/test_paper_trader.py` — added cap-inclusive test
- `tests/test_settlement.py` — added cap-inclusive + tail tests
- `tests/test_feature_builder.py` — count assertions (reverted back to 23)
- `docs/superpowers/specs/2026-03-22-strategy-backtest-design.md` — NEW
- `docs/superpowers/plans/2026-03-22-strategy-comparison.md` — NEW

## Next Steps — Mac Mini Deployment
1. **Push code to remote** (git push to origin)
2. **SSH into Mini, git pull**
3. **Set up venv + dependencies on Mini**
4. **Configure launchd service** (plist already drafted locally: `com.alphatemp.dashboard.plist`)
5. **Auto-deploy on push** — GitHub webhook or git hook that auto-pulls and restarts
6. **Tailscale** already set up — dashboard accessible via Mini's Tailscale IP:8050
7. **Copy DuckDB** to Mini (or set up sync)

## Known Issues
- DuckDB write lock conflicts: FeatureBuilder opens read-write connections that conflict with dashboard. Systemic issue, not blocking.
- Synoptic API flaky ("no data: unknown") — AWC backup working
- NWS CLI fetcher can fail to upsert when dashboard holds DB lock (retries on next cycle)
- CLI dot timing uses METAR peak as proxy, not actual CLI-reported time of max
- Candlestick data only covers Oct 2024+ (limits backtest range)

## Backlog
1. **AI Model Analyst chatbot** — chat panel in dashboard to ask about trade rationale, P&L, data questions. See fairuse.bet screenshot for reference.
2. Full-resolution backtest on Mini (all 488 days, 1000 bootstrap iterations)
3. DuckDB connection pooling / shared connection refactor
4. Parse time-of-max from CLI text for accurate dot placement
5. Prior-day (tomorrow) market trading evaluation

## Verification Queries
```bash
cd ~/Projects/alphatemp/alphatemp

# Tests
python3 -m pytest tests/test_web_dashboard.py tests/test_strategy_comparison.py tests/test_feature_builder.py -q

# Check HRRR data is current
python3 -c "
import duckdb
con = duckdb.connect('data/alphatemp.duckdb', read_only=True)
print(con.execute('SELECT MAX(model_run) as latest_hrrr, MAX(ingested_at) as last_ingest FROM forecasts WHERE model_name = \'hrrr\'').fetchdf())
con.close()
"

# Check strategy config
python3 -c "
import duckdb
con = duckdb.connect('data/alphatemp.duckdb', read_only=True)
print(con.execute('SELECT key, value FROM paper_config ORDER BY key').fetchdf())
con.close()
"

# Confirm no stale processes
ps aux | grep "main.py" | grep -v grep

# Recent commits
git log --oneline -15
```
