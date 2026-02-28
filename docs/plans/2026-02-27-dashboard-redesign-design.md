# Dashboard Redesign — Design Document

**Date:** 2026-02-27
**Status:** Approved
**Replaces:** Current Flask+Plotly dashboard (`ui/web_dashboard.py` + `templates/index.html`)

## Overview

Full replacement of the existing operational dashboard with a trading-focused, three-tab dashboard. Same tech stack (FastAPI + Vanilla JS + Plotly + Tailwind), no build tools. Auto-refreshes every 30-60 seconds without page reload. Mobile-responsive with a simplified Trade view.

## Tech Stack

- **Backend:** FastAPI (existing) + Jinja2 templates
- **Frontend:** Vanilla JS + Plotly.js + Tailwind CSS (CDN)
- **Data:** DuckDB (read-only connections from dashboard)
- **Auto-refresh:** JavaScript `setInterval` polling (30s for trading data, 60s for charts)
- **No build step.** No npm. Deploys with existing `deploy.sh`.

## Architecture

### Routes

Desktop tabs are separate HTML templates served by FastAPI:
- `GET /` → redirect to `/operations`
- `GET /operations` → Operations tab (live trading view)
- `GET /performance` → Performance tab (historical analysis)
- `GET /review` → Review tab (model autopsy)
- `GET /mobile` → Mobile view (Dashboard + Trade tabs, client-side toggle)

API endpoints (JSON, consumed by frontend JS):
- `GET /api/observations/{city}?date=YYYY-MM-DD` — Observation feed (existing, enhanced)
- `GET /api/forecast-points/{city}?date=YYYY-MM-DD` — HRRR run summaries (existing)
- `GET /api/forecast-curve/{city}?date=YYYY-MM-DD` — Full forecast + obs + bands (existing)
- `GET /api/brackets/{city}?date=YYYY-MM-DD` — Model vs Kalshi bracket probabilities + edge
- `GET /api/positions/{city}?date=YYYY-MM-DD` — Active bets, near-misses, P&L
- `GET /api/market-swings/{city}?date=YYYY-MM-DD` — Material price movements + model-caught flag
- `GET /api/liquidity/{city}?date=YYYY-MM-DD` — Bid-ask width, volume by bracket
- `GET /api/performance?range=7d|30d|all` — Cumulative P&L, daily bars, win rate
- `GET /api/brier-comparison?range=7d|30d|all` — Model Brier vs Market Brier by hour
- `GET /api/edge-heatmap?range=7d|30d|all` — Edge concentration by hour x bracket
- `GET /api/review/incidents?range=7d|30d|all&filter=all|worst|lost|missed` — Incident cards
- `GET /api/review/patterns?range=30d` — Pattern summary + failure mode aggregation
- `GET /api/health` — DB connectivity + data freshness (existing)

### File Structure

```
ui/
  web_dashboard.py          # FastAPI app + all API routes
templates/
  base.html                 # Shared layout: nav, header, Tailwind/Plotly imports
  operations.html           # Tab 1: Live trading view
  performance.html          # Tab 2: Historical analysis
  review.html               # Tab 3: Model autopsy
  mobile.html               # Mobile: Dashboard + Trade tabs
static/
  js/
    operations.js            # Tab 1 chart rendering + polling
    performance.js           # Tab 2 chart rendering
    review.js                # Tab 3 incident rendering
    mobile.js                # Mobile view logic
    shared.js                # Common utilities (time formatting, color scales, fetch helpers)
  css/
    dashboard.css            # Custom styles beyond Tailwind (minimal)
```

## Tab 1: Operations (Live Trading View)

### Layout (Desktop)

```
┌────────────────────────────────────────────┬────────────────────────┐
│  TEMPERATURE CURVE (60%)                   │  OBS FEED (40%)        │
│  - Forecast line + obs dots                │  Scrolling METAR list  │
│  - 50%/90% confidence bands               │  Time, source, temp,   │
│  - Prior runs (ghosted, last 5)            │  wind/gusts/precip     │
│  - Settlement marker (diamond/circle)      │  Color by source       │
│  - Neighbor toggle (KLGA, KEWR)            │  New entries flash     │
│                                            ├────────────────────────┤
│                                            │  FORECAST RUNS         │
│                                            │  Run hour, high, delta │
│                                            │  Color: green/red      │
├──────────────────────────────┬─────────────┴────────────────────────┤
│  BRACKET SPREAD (50%)        │  POSITIONS & ALERTS (50%)           │
│  Horizontal bar chart        │  Active bets: bracket, direction,   │
│  Model prob vs Kalshi mid    │    entry, current, edge, P&L        │
│  Edge highlighted            │  Near-miss bets (close to threshold)│
│  Liquidity gauge below       │  Market swings + model-caught flag  │
│  Bid-ask average             │  Daily P&L                          │
└──────────────────────────────┴─────────────────────────────────────┘
```

### Data Sources
- Temperature curve: `/api/forecast-curve` (existing)
- Obs feed: `/api/observations` (existing, add wind/gust/precip fields)
- Forecast runs: `/api/forecast-points` (existing)
- Bracket spread: `/api/brackets` (new)
- Positions: `/api/positions` (new)
- Market swings: `/api/market-swings` (new)
- Liquidity: `/api/liquidity` (new)

### Refresh Intervals
- Obs feed + forecast runs: 30s
- Bracket spread + positions: 30s
- Temperature curve: 60s (heavier query)

## Tab 2: Performance (Historical Analysis)

### Layout (Desktop)

```
┌─────────────────────────────────┬──────────────────────────────────┐
│  CUMULATIVE P&L (50%)           │  DAILY P&L BARS (50%)            │
│  Line chart over time           │  Green/red bars per day          │
│  Net of fees                    │  Win rate, avg win/loss          │
├─────────────────────────────────┼──────────────────────────────────┤
│  MODEL vs MARKET BRIER (50%)   │  EDGE HEATMAP (50%)              │
│  By hour (ET), dual lines      │  Hour (ET) x Bracket grid        │
│  Shaded edge window            │  Dark = strong edge              │
├─────────────────────────────────┴──────────────────────────────────┤
│  BREAKDOWN STATS (100%)                                            │
│  By bracket, by hour, by edge threshold                            │
│  Gross, fees, net. Win rate per segment.                           │
└────────────────────────────────────────────────────────────────────┘
```

### Date Range Selector
- 7d, 30d, All — applied to all charts on the tab

## Tab 3: Review (Model Autopsy)

### Layout (Desktop)

```
┌────────────────────────────────────────────────────────────────────┐
│  FILTER: [All] [Worst Misses] [Lost Bets] [Missed Edge]   Sort ▼  │
├────────────────────────────────────────────────────────────────────┤
│  INCIDENT CARDS (scrolling list)                                   │
│  Each card:                                                        │
│  - Date, title, severity bar                                       │
│  - Settlement vs model peak bracket vs Kalshi peak bracket         │
│  - Narrative: what happened                                        │
│  - Category tag (slow drift, tail underweight, threshold, etc.)    │
│  - Dollar impact                                                   │
│  - Expandable: mini forecast-vs-obs chart + bracket timeline       │
├────────────────────────────────────────────────────────────────────┤
│  PATTERN SUMMARY                                                   │
│  Top failure modes ranked by frequency + cost                      │
│  Suggested actions (computed from category → known remedy mapping)  │
└────────────────────────────────────────────────────────────────────┘
```

### Incident Detection Logic
An incident is generated when any of these conditions hold:
1. **Worst miss:** Model peak bracket != settlement bracket by 2+ brackets
2. **Lost bet:** A placed bet settled against us
3. **Missed edge:** Model had >threshold edge, didn't bet, bracket settled YES (would have profited)
4. **Market swing:** Kalshi price moved >10¢ in <2 hours on any bracket

### Categories (auto-tagged)
- Slow drift response: obs diverged >2°F from forecast, model didn't adjust
- Tail bracket underweight: settlement in a bracket model assigned <5%
- Threshold too conservative: edge was present but below bet threshold
- Stale pricing window: Kalshi price didn't move for >4 hours, model had edge
- Regime miss: unusual weather pattern (precip, wind) disrupted forecast

## Mobile View

Two-tab client-side toggle (no page reload):

### Dashboard Tab (Mobile)
- Current state card: latest obs, model high, settlement, daily P&L
- Alerts: market swings + model-caught flag
- Recent obs list (last 5)

### Trade Tab (Mobile)
- Suggested bets: bracket, direction, model vs Kalshi, gross edge, net edge, ask price, volume, confidence, liquidity warning
- Near-threshold bets
- Active positions with current P&L
- Daily + all-time P&L

## Visual Design

- Dark theme: slate-900 background, slate-300 text (matching current)
- Font: Space Mono (monospace, matching current)
- Color palette:
  - Green (#22c55e): positive edge, profit, obs warmer
  - Red (#ef4444): negative edge, loss, obs cooler
  - Amber (#f59e0b): warnings, near-threshold, thin liquidity
  - Blue (#3b82f6): model data, forecasts
  - Teal (#14b8a6): bias-adjusted, ensemble
  - White (#f8fafc): obs data, settlement
- Accent: subtle glow on active bets, flash animation on new data

## Data Dependencies

### New tables needed: None
All data lives in existing tables. New API endpoints compute derived views:
- Bracket probabilities: computed from ensemble + P2B pipeline (services layer)
- Positions: tracked in a new `paper_positions` table (trades the model would make)
- P&L: computed from `paper_positions` + `kalshi_settlements`
- Incidents: computed by comparing model predictions vs settlements vs market prices

### New table: `paper_positions`
```sql
CREATE TABLE IF NOT EXISTS paper_positions (
    id INTEGER PRIMARY KEY,
    city VARCHAR,
    event_date DATE,
    bracket_floor INTEGER,
    bracket_cap INTEGER,
    direction VARCHAR,          -- 'YES' or 'NO'
    model_prob DOUBLE,
    market_price DOUBLE,
    edge DOUBLE,
    entry_price DOUBLE,
    entry_time TIMESTAMP,
    exit_price DOUBLE,          -- NULL until settled
    exit_time TIMESTAMP,        -- NULL until settled
    settled_yes BOOLEAN,        -- NULL until settled
    gross_pnl DOUBLE,           -- NULL until settled
    fees DOUBLE,                -- NULL until settled
    net_pnl DOUBLE,             -- NULL until settled
    status VARCHAR DEFAULT 'open'  -- 'open', 'settled', 'expired'
)
```

## Error Handling

- DuckDB read-only connections for all dashboard queries (no write contention)
- API endpoints return empty arrays (not errors) when no data available
- Frontend gracefully handles missing data (shows "--" or "No data" states)
- Health endpoint checks data freshness; dashboard shows stale-data warning if obs > 30min old

## Implementation Priority

1. Shared layout (base.html, nav, routing)
2. Operations tab (most complex, highest daily value)
3. API endpoints for brackets, positions, market swings
4. paper_positions table + position tracking logic
5. Performance tab
6. Review tab
7. Mobile view
