# Dashboard V2 Design

**Date:** 2026-02-28
**Status:** Approved
**Scope:** KPI Header, System Health page, Settlement Countdown + Trade Blotter page, Polish existing pages, PaperTrader service shell

## Context

Current dashboard (FastAPI + Jinja2 + Plotly + Tailwind) has 4 pages: operations, performance, review, mobile. ~15 API endpoints, 60s polling, dark theme. Inspired by FairWeather Fund but designed for AlphaTemp's specific needs. Dashboard serves dual purpose: live trading cockpit during the day, research/analysis tool after hours.

## Design Decisions

- Keep multi-page structure, add new pages (don't consolidate)
- Automated paper trading — no manual trade entry UI
- Paper trader strategy logic is a placeholder (dedicated brainstorm session later)
- PaperTrader service follows existing async task pattern in main.py
- Supports YES and NO positions, early exit before settlement

---

## Section 1: KPI Header Bar

Persistent header strip across all pages. Replaces per-page nav with unified layout.

### Layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 🟢 LIVE  │  Model High: 47°F  │  Settlement: Pending (CLI)  │         │
│           │  Market Consensus: 46-48°F  │  Drift: +0.3°F    │  2:34p ET│
│           │  Open Positions: 3  │  Day P&L: +$2.40  │  Total: +$18.60  │
└──────────────────────────────────────────────────────────────────────────┘
Nav: [Operations] [Blotter] [Health] [Performance] [Review]
```

### Behavior

- Refreshes on 60s cycle with rest of dashboard
- System status dot: green (all fresh), amber (any source >30 min stale), red (DB unreachable)
- Settlement status escalates: "Pending" → "DSM: 47°F" → "CLI: 47°F (Final)"
- Today/Tomorrow toggle lives in header
- Nav bar integrated below KPI strip

### Implementation

- Add to Jinja2 base template (inherited by all pages)
- New endpoint: `GET /api/kpi-summary` — bundles key metrics in single call
- Data sources: `/api/health`, `/api/probability/nyc`, `/api/positions/nyc`
- Mobile: responsive version stacks vertically on narrow screens

---

## Section 2: System Health Page

Dedicated `/health` page for pipeline monitoring. Glanceable status board.

### Layout

4 sections stacked vertically:

**Data Freshness:** Per-source last-seen timestamp + status dot
- Observations (Synoptic), Observations (AWC/IEM), HRRR Forecasts, Market Ticks, Drift Signals, NWS Settlement (CLI), NWS Settlement (DSM)

**Pipeline Status:** Per-service running state, last cycle duration
- SynopticIngestor, IEMIngestor, HRRRFetcher, BiasEngine, MarketFetcher, NWSFetcher, PaperTrader

**Database:** Row counts per table, last write time, DB size

**Configuration:** Settlement station, neighbors, polling intervals, stale threshold, model, bias TTL

### Behavior

- Status dots: green <30 min, amber 30-60 min, red >60 min
- NWS sources use smarter thresholds (expected once/day)
- Refreshes on 60s cycle
- No charts — clean status tables only

### Implementation

- New endpoint: `GET /api/health/detailed` — pipeline heartbeats, row counts, config
- Each service logs heartbeat (last run time, duration, status) to a shared dict
- New template: `health.html`
- Config pulled from `constants.py` (static, rendered once)

---

## Section 3: Settlement Countdown + Trade Blotter

New `/blotter` page. Two sections: countdown at top, full bracket grid below.

### Settlement Countdown

```
┌─ Settlement Countdown ───────────────────────────────────────┐
│  KXHIGHNY  │  Feb 28  │  Settles in: 4h 12m                 │
│  Model High: 47.2°F → Bracket: 46-48°F (62%)                │
│  Market Mid: 58¢      │  Edge: +4.0%                         │
│  Running Obs Max: 45°F (as of 1:48p ET)                      │
│  Settlement Source: Pending (no CLI/DSM yet)                  │
├─ Tomorrow ───────────────────────────────────────────────────│
│  KXHIGHNY  │  Mar 1   │  Settles in: 28h 12m                │
│  Model High: 51.8°F → Bracket: 50-52°F (48%)                │
│  Market Mid: 41¢      │  Edge: +7.0%                         │
└──────────────────────────────────────────────────────────────┘
```

- Countdown ticks live (JS setInterval, independent of 60s data refresh)
- Settlement time from `kalshi_settlements.close_time`
- Shows today + tomorrow when both markets active
- Running obs max updates on 60s cycle
- Settlement source escalates: Pending → DSM → CLI (Final)

### Trade Blotter

```
Bracket │ Model │ Market │ Edge   │ Dir │ Pos │ Entry │ P&L
────────┼───────┼────────┼────────┼─────┼─────┼───────┼────
42-44   │  3.1% │   8¢   │ -4.9%  │  —  │  —  │   —   │  —
44-46   │ 18.2% │  15¢   │ +3.2%  │  —  │  —  │   —   │  —
46-48   │ 62.0% │  58¢   │ +4.0%  │ YES │  2  │  57¢  │ +$2
48-50   │ 14.5% │  16¢   │ -1.5%  │  —  │  —  │   —   │  —
50-52   │  2.2% │   5¢   │ -2.8%  │ NO  │  1  │  95¢  │ +$1

Summary: Open: 3 │ Exposure: $4.50 │ Day P&L: +$3 │ Liquidity: 342 │ Spread: 3¢
```

- Sorted by bracket floor ascending
- Active positions highlighted, non-positions dimmed
- P&L: unrealized during the day (vs market mid), realized after settlement
- Date toggle switches today/tomorrow

### Implementation

- New endpoint: `GET /api/blotter/{city}` — bundles brackets + positions + countdown
- New template: `blotter.html`, new JS: `blotter.js`

---

## Section 4: Polish Existing Pages

Targeted fixes, no redesigns.

### Operations

- **Stale data warning:** Amber banner when any source >30 min stale
- **Settlement source label:** Mark temp chart settlement marker with source (Running Max / DSM / CLI)
- **Error toasts:** Replace silent `catch(e) {}` with auto-dismissing notification bar

### Performance

- **Fix Brier comparison:** Wire stubbed endpoint to actual model vs market Brier from `kalshi_settlements`
- **Paper P&L tracking:** Show cumulative paper P&L alongside Brier scores once paper trader runs

### Review

- **Fix missed_edge filter:** Wire `missed_edge` categorization so the filter returns results
- **Settlement source on incidents:** Show CLI vs DSM vs running max on incident cards

### Mobile

- **Inherit KPI header:** Replace custom mini-header with responsive global KPI header
- **Blotter summary:** Compact positions + P&L below dashboard tab (no full bracket grid)

### Cross-cutting

- **Consistent error toasts:** Same notification pattern on every page
- **Loading states:** Skeleton placeholders while API calls in flight

---

## Section 5: PaperTrader Service (Shell)

Backend service feeding the blotter. Strategy logic is a placeholder — dedicated brainstorm session to follow.

### Service Interface

```python
class PaperTrader:
    """Async service in main.py task list."""

    async def run(self):
        """60s loop: check entries, monitor exits, settle positions."""

    async def _check_entries(self):
        """Evaluate brackets for edge. Strategy logic TBD."""

    async def _check_exits(self):
        """Monitor open positions for early exit. Strategy logic TBD."""

    async def _settle_positions(self):
        """On CLI/DSM arrival, resolve open positions → realized P&L."""

    async def _compute_fees(self, price, contracts):
        """Kalshi taker fee: max(ceil(0.07 * C * P * (1-P)), C * $0.01)"""
```

### Position Lifecycle

```
PENDING → OPEN → CLOSED (settled or early exit)
```

- **PENDING:** Edge detected, position recorded
- **OPEN:** Active, unrealized P&L tracked vs market mid
- **CLOSED:** Settled (CLI resolves YES/NO) or early exit (sold before settlement)

### Schema Additions to `paper_positions`

- `exit_reason` — 'settlement', 'early_exit', 'strategy_stop'
- `contracts` — number of contracts (currently implicit)
- `unrealized_pnl` — updated each cycle for open positions

### Placeholder Strategy

Ships with trivial rule: "enter when edge > 5%, exit on settlement only." Replaced with real strategy after dedicated brainstorm.

### Integration

- Added to `main.py` async task list
- Depends on: ProbabilityEngine, MarketFetcher, NWSFetcher
- No new infrastructure or dependencies
- Supports YES and NO positions
- Supports early exit before settlement

---

## Follow-up Sessions

1. **Paper trading strategy brainstorm** — entry/exit rules, edge thresholds, position sizing, NO-side logic, risk caps
2. **Portfolio & Risk page** — Kelly utilization, exposure breakdown (deferred from this design)
3. **Execution Monitor page** — filled orders, live P&L (deferred until live trading approved)
