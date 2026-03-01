# Dashboard V2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Overhaul the AlphaTemp dashboard with a KPI header bar, system health page, settlement countdown + trade blotter, paper trading service shell, and polish to existing pages.

**Architecture:** Multi-page FastAPI + Jinja2 + Plotly + Tailwind dashboard. New pages added alongside existing ones. PaperTrader service runs as async task in main.py. All new API endpoints follow existing patterns (DuckDB queries → JSON responses). No new dependencies.

**Tech Stack:** Python 3.9, FastAPI, Jinja2, DuckDB, Plotly.js 2.35.0, Tailwind CSS (CDN), vanilla JS

**Design doc:** `docs/plans/2026-02-28-dashboard-v2-design.md`

---

### Task 1: Schema Changes + Service Heartbeat Infrastructure

Extend `paper_positions` table and add a shared heartbeat registry so services can report their status.

**Files:**
- Modify: `core/db.py:207-228` (paper_positions schema)
- Create: `core/heartbeat.py`
- Test: `tests/test_heartbeat.py`

**Step 1: Write failing test for heartbeat registry**

```python
# tests/test_heartbeat.py
from core.heartbeat import heartbeat_registry, record_heartbeat, get_all_heartbeats

def test_record_and_retrieve_heartbeat():
    record_heartbeat("TestService", duration_ms=150, status="ok")
    beats = get_all_heartbeats()
    assert "TestService" in beats
    assert beats["TestService"]["status"] == "ok"
    assert beats["TestService"]["duration_ms"] == 150
    assert beats["TestService"]["last_run"] is not None
```

**Step 2: Run test to verify it fails**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_heartbeat.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.heartbeat'`

**Step 3: Implement heartbeat registry**

```python
# core/heartbeat.py
"""Lightweight in-memory heartbeat registry for service health monitoring."""
from datetime import datetime, timezone
from typing import Dict, Optional

_heartbeats: Dict[str, dict] = {}

def record_heartbeat(
    service_name: str,
    duration_ms: float,
    status: str = "ok",
    error: Optional[str] = None,
) -> None:
    """Record a service heartbeat. Called at end of each polling cycle."""
    _heartbeats[service_name] = {
        "last_run": datetime.now(timezone.utc).isoformat(),
        "duration_ms": round(duration_ms, 1),
        "status": status,
        "error": error,
    }

def get_all_heartbeats() -> Dict[str, dict]:
    """Return all heartbeats. Used by /api/health/detailed."""
    return dict(_heartbeats)

def clear_heartbeats() -> None:
    """Reset registry. Used in tests."""
    _heartbeats.clear()
```

**Step 4: Run test to verify it passes**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_heartbeat.py -v`
Expected: PASS

**Step 5: Add schema columns to paper_positions**

In `core/db.py`, update the `paper_positions` CREATE TABLE to add three columns:

```sql
-- Add after existing columns, before the closing paren:
exit_reason VARCHAR,        -- 'settlement', 'early_exit', 'strategy_stop'
contracts INTEGER DEFAULT 1,
unrealized_pnl DOUBLE DEFAULT 0.0,
```

No test needed — schema is applied via `CREATE TABLE IF NOT EXISTS` on startup. Verify manually:

Run: `cd ~/Projects/alphatemp/alphatemp && python -c "from core.db import init_db; init_db(); print('Schema OK')"`

**Step 6: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add core/heartbeat.py core/db.py tests/test_heartbeat.py
git commit -m "feat: add heartbeat registry and extend paper_positions schema"
```

---

### Task 2: KPI Header Bar

Replace per-page nav in `base.html` with unified KPI header. Add `/api/kpi-summary` endpoint.

**Files:**
- Modify: `templates/base.html:13-52` (header + nav)
- Modify: `ui/web_dashboard.py` (add KPI endpoint)
- Modify: `static/js/shared.js` (add KPI refresh function)
- Test: `tests/test_web_dashboard.py` (if exists, add KPI test)

**Step 1: Write failing test for KPI endpoint**

```python
# tests/test_kpi_endpoint.py
import pytest
from fastapi.testclient import TestClient
from ui.web_dashboard import app

client = TestClient(app)

def test_kpi_summary_returns_expected_fields():
    resp = client.get("/api/kpi-summary?city=nyc")
    assert resp.status_code == 200
    data = resp.json()
    assert "system_status" in data
    assert "model_high" in data
    assert "settlement" in data
    assert "open_positions" in data
    assert "day_pnl" in data
    assert "total_pnl" in data
    assert "drift" in data
```

**Step 2: Run test to verify it fails**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_kpi_endpoint.py -v`
Expected: FAIL — 404 (endpoint doesn't exist)

**Step 3: Implement `/api/kpi-summary` endpoint**

Add to `ui/web_dashboard.py` after the health endpoint (~line 134):

```python
@app.get("/api/kpi-summary")
async def kpi_summary(city: str = "nyc"):
    """Bundled KPI metrics for the persistent header bar."""
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        # System status from health check
        health = await api_health()
        system_status = "red" if health.get("error") else (
            "amber" if health.get("obs_stale") or health.get("fcst_stale") else "green"
        )

        # Model high from probability engine
        forecast = prob_engine.calculate_city(city, None)
        model_high = forecast.center if forecast else None

        # Settlement status from nws_daily
        today_et = get_today_et()
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
        """, [city]).fetchone()
        drift = round(drift_row[0], 1) if drift_row else 0.0

        # Positions P&L
        positions = con.execute("""
            SELECT
                COUNT(*) FILTER (WHERE status = 'open') as open_count,
                COALESCE(SUM(net_pnl) FILTER (WHERE event_date = ?), 0) as day_pnl,
                COALESCE(SUM(net_pnl), 0) as total_pnl
            FROM paper_positions WHERE city = ?
        """, [today_et, city]).fetchone()

        # Market consensus (highest-prob bracket from market)
        bracket_row = con.execute("""
            SELECT floor_strike, cap_strike FROM market_ticks
            WHERE city = ? AND captured_at >= NOW() - INTERVAL '10 minutes'
            ORDER BY yes_bid DESC LIMIT 1
        """, [city]).fetchone()
        market_consensus = f"{bracket_row[0]}-{bracket_row[1]}°F" if bracket_row else None

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
```

**Step 4: Run test to verify it passes**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_kpi_endpoint.py -v`
Expected: PASS

**Step 5: Update base.html with KPI header**

Replace the header section in `templates/base.html` (lines ~13-52) with:

```html
<!-- KPI Header Bar -->
<header class="border-b border-slate-700 bg-slate-900/80 backdrop-blur">
  <div class="flex items-center justify-between px-4 py-2 text-xs">
    <!-- Left: System status + title -->
    <div class="flex items-center gap-3">
      <span id="kpi-status-dot" class="glow-dot"></span>
      <span class="text-slate-300 font-bold tracking-wide">ALPHATEMP</span>
    </div>

    <!-- Center: Key metrics -->
    <div class="flex items-center gap-6 text-slate-400">
      <span>Model High: <strong id="kpi-model-high" class="text-white">--</strong></span>
      <span>Settlement: <strong id="kpi-settlement" class="text-white">--</strong></span>
      <span>Consensus: <strong id="kpi-consensus" class="text-slate-300">--</strong></span>
      <span>Drift: <strong id="kpi-drift" class="text-slate-300">--</strong></span>
    </div>

    <!-- Right: Positions + P&L + Clock -->
    <div class="flex items-center gap-6 text-slate-400">
      <span>Open: <strong id="kpi-open-pos" class="text-white">--</strong></span>
      <span>Day P&L: <strong id="kpi-day-pnl" class="text-white">--</strong></span>
      <span>Total: <strong id="kpi-total-pnl" class="text-white">--</strong></span>
      <span id="et-clock" class="text-slate-500">--:-- ET</span>
    </div>
  </div>

  <!-- Nav + Date toggle -->
  <div class="flex items-center justify-between px-4 py-1 border-t border-slate-800">
    <nav class="flex gap-1">
      {% for tab, label in [('operations', 'Operations'), ('blotter', 'Blotter'), ('health', 'Health'), ('performance', 'Performance'), ('review', 'Review')] %}
      <a href="/{{ tab }}"
         class="px-3 py-1 rounded text-xs {% if active_tab == tab %}bg-slate-700 text-white{% else %}text-slate-500 hover:text-slate-300{% endif %}">
        {{ label }}
      </a>
      {% endfor %}
    </nav>
    <div class="flex items-center gap-2">
      <button onclick="setDateToggle('today')" id="btn-today"
              class="px-2 py-0.5 rounded text-xs bg-slate-700 text-white">Today</button>
      <button onclick="setDateToggle('tomorrow')" id="btn-tomorrow"
              class="px-2 py-0.5 rounded text-xs text-slate-500 hover:text-slate-300">Tomorrow</button>
      <span class="text-slate-600 text-xs ml-2" id="countdown-display">60s</span>
    </div>
  </div>
</header>
```

**Step 6: Add KPI refresh to shared.js**

Append to `static/js/shared.js`:

```javascript
async function refreshKPI() {
    var data = await fetchAPI('/api/kpi-summary?city=' + (window.selectedCity || 'nyc'));
    if (!data) return;

    // Status dot
    var dot = document.getElementById('kpi-status-dot');
    if (dot) {
        dot.style.background = data.system_status === 'green' ? '#10b981' :
                               data.system_status === 'amber' ? '#f59e0b' : '#ef4444';
    }

    setText('kpi-model-high', data.model_high ? data.model_high + '°F' : '--');

    // Settlement
    var stl = data.settlement || {};
    if (stl.source === 'NWS_CLI') {
        setText('kpi-settlement', stl.temp + '°F (CLI)');
    } else if (stl.source === 'DSM') {
        setText('kpi-settlement', stl.temp + '°F (DSM)');
    } else {
        setText('kpi-settlement', 'Pending');
    }

    setText('kpi-consensus', data.market_consensus || '--');
    setText('kpi-drift', data.drift !== null ? (data.drift > 0 ? '+' : '') + data.drift + '°F' : '--');
    setText('kpi-open-pos', data.open_positions);

    var dayPnl = formatPnL(data.day_pnl || 0);
    var totalPnl = formatPnL(data.total_pnl || 0);
    setColoredText('kpi-day-pnl', dayPnl.text, dayPnl.colorClass);
    setColoredText('kpi-total-pnl', totalPnl.text, totalPnl.colorClass);
}

function setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
}

function setColoredText(id, text, colorClass) {
    var el = document.getElementById(id);
    if (el) {
        el.textContent = text;
        el.className = colorClass;
    }
}
```

Add `refreshKPI()` call to the base.html inline script's refresh cycle (alongside clock update).

**Step 7: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add templates/base.html ui/web_dashboard.py static/js/shared.js tests/test_kpi_endpoint.py
git commit -m "feat: add KPI header bar with persistent metrics across all pages"
```

---

### Task 3: System Health Page

New `/health` page with pipeline status, data freshness, DB stats, and config display.

**Files:**
- Create: `templates/health.html`
- Modify: `ui/web_dashboard.py` (add route + API endpoint)
- Create: `static/js/health.js`
- Test: `tests/test_health_detailed.py`

**Step 1: Write failing test for detailed health endpoint**

```python
# tests/test_health_detailed.py
import pytest
from fastapi.testclient import TestClient
from ui.web_dashboard import app

client = TestClient(app)

def test_health_detailed_returns_sections():
    resp = client.get("/api/health/detailed")
    assert resp.status_code == 200
    data = resp.json()
    assert "freshness" in data
    assert "pipelines" in data
    assert "database" in data
    assert "config" in data
```

**Step 2: Run test to verify it fails**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_health_detailed.py -v`
Expected: FAIL — 404

**Step 3: Implement `/api/health/detailed` endpoint**

Add to `ui/web_dashboard.py`:

```python
@app.get("/api/health/detailed")
async def health_detailed():
    """Comprehensive system health for the /health page."""
    from core.heartbeat import get_all_heartbeats
    from core.constants import (
        OBS_POLL_INTERVAL, FORECAST_POLL_INTERVAL,
        MARKET_POLL_INTERVAL, DRIFT_POLL_INTERVAL,
        NWS_CLI_POLL_INTERVAL, NWS_DAILY_HIGH_POLL_INTERVAL,
    )

    con = duckdb.connect(DB_PATH, read_only=True)
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
        for key, query in freshness_queries.items():
            row = con.execute(query).fetchone()
            ts = row[0] if row else None
            freshness[key] = {
                "last_seen": ts.isoformat() if ts else None,
                "age_minutes": round((datetime.now(timezone.utc) - ts).total_seconds() / 60, 1) if ts else None,
            }

        # Pipeline heartbeats
        pipelines = get_all_heartbeats()

        # Database stats
        table_stats = {}
        for table in ["observations", "forecasts", "market_ticks", "drift_signals",
                       "nws_daily", "paper_positions", "kalshi_settlements",
                       "kalshi_candlesticks", "kalshi_trades"]:
            row = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            table_stats[table] = {"rows": row[0]}

        db_size_row = con.execute("CALL pragma_database_size()").fetchone()
        db_size = db_size_row[4] if db_size_row else "unknown"  # human-readable size

        # Config (static)
        config = {
            "settlement_station": "KNYC",
            "neighbor_stations": ["KLGA", "KEWR", "KJFK"],
            "polling_intervals": {
                "observations": OBS_POLL_INTERVAL,
                "forecasts": FORECAST_POLL_INTERVAL,
                "market": MARKET_POLL_INTERVAL,
                "drift": DRIFT_POLL_INTERVAL,
                "nws_cli": NWS_CLI_POLL_INTERVAL,
                "nws_daily": NWS_DAILY_HIGH_POLL_INTERVAL,
            },
            "stale_threshold_min": 30,
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
```

Add the page route:

```python
@app.get("/health")
async def health_page(request: Request):
    return templates.TemplateResponse("health.html", {
        "request": request, "active_tab": "health"
    })
```

**Step 4: Run test to verify it passes**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_health_detailed.py -v`
Expected: PASS

**Step 5: Create health.html template**

```html
<!-- templates/health.html -->
{% extends "base.html" %}
{% block title %}Health — AlphaTemp{% endblock %}
{% block content %}
<div class="p-4 space-y-6 max-w-5xl mx-auto">

  <!-- Data Freshness -->
  <div class="bg-slate-800/50 rounded-lg p-4">
    <h2 class="text-sm font-bold text-slate-300 mb-3 tracking-wide">DATA FRESHNESS</h2>
    <div id="freshness-table" class="space-y-1 text-xs font-mono"></div>
  </div>

  <!-- Pipeline Status -->
  <div class="bg-slate-800/50 rounded-lg p-4">
    <h2 class="text-sm font-bold text-slate-300 mb-3 tracking-wide">PIPELINE STATUS</h2>
    <div id="pipeline-table" class="space-y-1 text-xs font-mono"></div>
  </div>

  <!-- Database -->
  <div class="bg-slate-800/50 rounded-lg p-4">
    <h2 class="text-sm font-bold text-slate-300 mb-3 tracking-wide">DATABASE</h2>
    <div id="db-table" class="space-y-1 text-xs font-mono"></div>
    <div id="db-size" class="text-xs text-slate-500 mt-2"></div>
  </div>

  <!-- Configuration -->
  <div class="bg-slate-800/50 rounded-lg p-4">
    <h2 class="text-sm font-bold text-slate-300 mb-3 tracking-wide">CONFIGURATION</h2>
    <div id="config-table" class="space-y-1 text-xs font-mono"></div>
  </div>

</div>
{% endblock %}
{% block scripts %}
<script src="/static/js/health.js"></script>
{% endblock %}
```

**Step 6: Create health.js**

```javascript
// static/js/health.js

var STALE_THRESHOLD = 30; // minutes
var NWS_STALE_THRESHOLD = 1440; // 24 hours for NWS sources

function statusDot(ageMin, threshold) {
    if (ageMin === null) return '<span class="text-slate-600">●</span>';
    if (ageMin <= threshold) return '<span class="text-emerald-400">●</span>';
    if (ageMin <= threshold * 2) return '<span class="text-amber-400">●</span>';
    return '<span class="text-red-400">●</span>';
}

function formatAge(ageMin) {
    if (ageMin === null) return 'never';
    if (ageMin < 1) return '<1 min ago';
    if (ageMin < 60) return Math.round(ageMin) + ' min ago';
    if (ageMin < 1440) return Math.round(ageMin / 60) + 'h ago';
    return Math.round(ageMin / 1440) + 'd ago';
}

var FRESHNESS_LABELS = {
    observations_synoptic: 'Observations (Synoptic)',
    observations_awc: 'Observations (AWC/IEM)',
    forecasts_hrrr: 'HRRR Forecasts',
    market_ticks: 'Market Ticks (Kalshi)',
    drift_signals: 'Drift Signals',
    nws_cli: 'NWS Settlement (CLI)',
    nws_dsm: 'NWS Settlement (DSM)',
};

async function refreshHealth() {
    var data = await fetchAPI('/api/health/detailed');
    if (!data) return;

    // Freshness
    var html = '';
    for (var key in FRESHNESS_LABELS) {
        var f = data.freshness[key] || {};
        var threshold = key.startsWith('nws_') ? NWS_STALE_THRESHOLD : STALE_THRESHOLD;
        html += '<div class="flex justify-between items-center py-1 border-b border-slate-700/50">'
            + '<span class="text-slate-400">' + FRESHNESS_LABELS[key] + '</span>'
            + '<span>' + formatAge(f.age_minutes) + ' ' + statusDot(f.age_minutes, threshold) + '</span>'
            + '</div>';
    }
    document.getElementById('freshness-table').innerHTML = html;

    // Pipelines
    html = '';
    var pipelines = data.pipelines || {};
    for (var name in pipelines) {
        var p = pipelines[name];
        var pDot = p.status === 'ok'
            ? '<span class="text-emerald-400">●</span>'
            : '<span class="text-red-400">●</span>';
        html += '<div class="flex justify-between items-center py-1 border-b border-slate-700/50">'
            + '<span class="text-slate-400">' + name + '</span>'
            + '<span class="text-slate-500">last cycle: ' + p.duration_ms + 'ms</span>'
            + '<span>' + (p.status === 'ok' ? 'Running' : p.error || 'Error') + ' ' + pDot + '</span>'
            + '</div>';
    }
    if (!Object.keys(pipelines).length) {
        html = '<div class="text-slate-600">No heartbeats received yet</div>';
    }
    document.getElementById('pipeline-table').innerHTML = html;

    // Database
    html = '';
    var tables = data.database.tables || {};
    for (var tbl in tables) {
        html += '<div class="flex justify-between items-center py-1 border-b border-slate-700/50">'
            + '<span class="text-slate-400">' + tbl + '</span>'
            + '<span class="text-slate-300">' + tables[tbl].rows.toLocaleString() + ' rows</span>'
            + '</div>';
    }
    document.getElementById('db-table').innerHTML = html;
    document.getElementById('db-size').textContent = 'DB size: ' + (data.database.size || 'unknown');

    // Config
    html = '';
    var cfg = data.config || {};
    var configRows = [
        ['Settlement station', cfg.settlement_station],
        ['Neighbor stations', (cfg.neighbor_stations || []).join(', ')],
        ['Stale threshold', cfg.stale_threshold_min + ' min'],
        ['Model', cfg.model],
        ['Bias TTL', cfg.bias_ttl_min + ' min'],
    ];
    var intervals = cfg.polling_intervals || {};
    for (var svc in intervals) {
        configRows.push(['Poll: ' + svc, intervals[svc] + 's']);
    }
    for (var i = 0; i < configRows.length; i++) {
        html += '<div class="flex justify-between items-center py-1 border-b border-slate-700/50">'
            + '<span class="text-slate-400">' + configRows[i][0] + '</span>'
            + '<span class="text-slate-300">' + configRows[i][1] + '</span>'
            + '</div>';
    }
    document.getElementById('config-table').innerHTML = html;
}

refreshHealth();
setInterval(refreshHealth, 60000);
```

**Step 7: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add templates/health.html static/js/health.js ui/web_dashboard.py tests/test_health_detailed.py
git commit -m "feat: add system health page with pipeline status and data freshness"
```

---

### Task 4: Integrate Heartbeats into Existing Services

Wire `record_heartbeat()` calls into every async service so the health page shows live pipeline status.

**Files:**
- Modify: `services/ingestor.py` (SynopticIngestor)
- Modify: `services/iem_ingestor.py` (IEMIngestor)
- Modify: `services/forecast.py` (HRRRFetcher)
- Modify: `services/bias.py` (BiasEngine)
- Modify: `services/market_fetcher.py` (MarketFetcher)
- Modify: `services/nws_fetcher.py` (NWSFetcher)

**Step 1: Identify the pattern**

Each service has an async `run()` method with a `while True` loop containing a `try/except` block and an `await asyncio.sleep(interval)`. The heartbeat call goes at the end of each successful cycle.

Pattern to add in each service:

```python
import time
from core.heartbeat import record_heartbeat

# Inside the try block of the main loop, wrap the cycle:
cycle_start = time.monotonic()
# ... existing cycle code ...
record_heartbeat(
    "ServiceName",
    duration_ms=(time.monotonic() - cycle_start) * 1000,
)
```

And in the `except` block:

```python
record_heartbeat("ServiceName", duration_ms=0, status="error", error=str(e))
```

**Step 2: Apply pattern to all 6 services**

Apply the pattern above to each service's `run()` method. Use the service class name as the heartbeat label:
- `SynopticIngestor` → `"SynopticIngestor"`
- `IEMIngestor` → `"IEMIngestor"`
- `HRRRFetcher` → `"HRRRFetcher"`
- `BiasEngine` → `"BiasEngine"`
- `MarketFetcher` → `"MarketFetcher"`
- `NWSFetcher` → `"NWSFetcher"`

**Step 3: Verify by running the dashboard locally**

Run: `cd ~/Projects/alphatemp/alphatemp && python main.py --dashboard &`
Wait 60 seconds, then: `curl localhost:8050/api/health/detailed | python -m json.tool | head -30`
Expected: `pipelines` dict should have entries for each service

**Step 4: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add services/ingestor.py services/iem_ingestor.py services/forecast.py services/bias.py services/market_fetcher.py services/nws_fetcher.py
git commit -m "feat: add heartbeat reporting to all pipeline services"
```

---

### Task 5: PaperTrader Service Shell

Create the PaperTrader async service with placeholder strategy. Registers in main.py.

**Files:**
- Create: `services/paper_trader.py`
- Modify: `main.py:29-43` (add PaperTrader to task list)
- Test: `tests/test_paper_trader.py`

**Step 1: Write failing tests**

```python
# tests/test_paper_trader.py
import pytest
import duckdb
from unittest.mock import MagicMock, AsyncMock
from services.paper_trader import PaperTrader

@pytest.fixture
def trader():
    return PaperTrader(db_path=":memory:")

def test_compute_fees_basic(trader):
    """Kalshi taker fee: max(ceil(0.07 * C * P * (1-P)), C * 0.01)"""
    fee = trader._compute_fee(price_cents=50, contracts=1)
    # 0.07 * 1 * 0.50 * 0.50 = 0.0175 → ceil → 0.02
    assert fee == 0.02

def test_compute_fees_min_floor(trader):
    """Fee should never be less than $0.01 per contract."""
    fee = trader._compute_fee(price_cents=99, contracts=1)
    # 0.07 * 1 * 0.99 * 0.01 = 0.000693 → ceil → 0.01
    # min = 1 * 0.01 = 0.01
    assert fee == 0.01

def test_position_lifecycle():
    """Position goes OPEN → CLOSED on settlement."""
    trader = PaperTrader(db_path=":memory:")
    trader._init_db()

    # Simulate an entry
    trader._record_entry(
        city="nyc", event_date="2026-02-28",
        bracket_floor=46, bracket_cap=48,
        direction="YES", model_prob=0.62,
        market_price=58, entry_price=57, contracts=2,
    )

    con = duckdb.connect(":memory:")
    # Verify position exists and is open
    # (Implementation will use the trader's own connection)
```

**Step 2: Run test to verify it fails**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_paper_trader.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement PaperTrader**

```python
# services/paper_trader.py
"""Automated paper trading service. Runs as async task in main.py.

Strategy logic is a PLACEHOLDER — entry/exit rules will be designed
in a dedicated brainstorming session. This shell provides:
- Position lifecycle (OPEN → CLOSED)
- Fee computation (Kalshi taker fee model)
- Settlement resolution
- Heartbeat reporting
"""
import asyncio
import math
import time
from datetime import datetime, timezone
from typing import Optional

import duckdb
from loguru import logger

from core.heartbeat import record_heartbeat
from core.timezone import get_today_et


PAPER_TRADE_INTERVAL = 60  # seconds


class PaperTrader:
    def __init__(self, db_path: str = "data/alphatemp.duckdb"):
        self.db_path = db_path

    def _compute_fee(self, price_cents: int, contracts: int) -> float:
        """Kalshi taker fee: max(ceil(0.07 * C * P * (1-P)), C * $0.01).
        Returns fee in dollars.
        """
        p = price_cents / 100.0
        raw = 0.07 * contracts * p * (1.0 - p)
        fee = max(math.ceil(raw * 100) / 100, contracts * 0.01)
        return round(fee, 2)

    def _init_db(self):
        """Ensure paper_positions table exists (idempotent)."""
        con = duckdb.connect(self.db_path)
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS paper_positions (
                    id INTEGER PRIMARY KEY,
                    city VARCHAR,
                    event_date DATE,
                    bracket_floor INTEGER,
                    bracket_cap INTEGER,
                    direction VARCHAR,
                    model_prob DOUBLE,
                    market_price DOUBLE,
                    edge DOUBLE,
                    entry_price DOUBLE,
                    entry_time TIMESTAMP,
                    exit_price DOUBLE,
                    exit_time TIMESTAMP,
                    settled_yes BOOLEAN,
                    gross_pnl DOUBLE,
                    fees DOUBLE,
                    net_pnl DOUBLE,
                    status VARCHAR DEFAULT 'open',
                    exit_reason VARCHAR,
                    contracts INTEGER DEFAULT 1,
                    unrealized_pnl DOUBLE DEFAULT 0.0
                )
            """)
        finally:
            con.close()

    def _record_entry(
        self, city: str, event_date: str,
        bracket_floor: int, bracket_cap: int,
        direction: str, model_prob: float,
        market_price: float, entry_price: float,
        contracts: int = 1,
    ) -> None:
        """Write a new open position to paper_positions."""
        edge = model_prob - (market_price / 100.0)
        fee = self._compute_fee(int(entry_price), contracts)
        con = duckdb.connect(self.db_path)
        try:
            con.execute("""
                INSERT INTO paper_positions
                (city, event_date, bracket_floor, bracket_cap, direction,
                 model_prob, market_price, edge, entry_price, entry_time,
                 fees, status, contracts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
            """, [
                city, event_date, bracket_floor, bracket_cap, direction,
                round(model_prob, 4), market_price, round(edge, 4),
                entry_price, datetime.now(timezone.utc),
                fee, contracts,
            ])
        finally:
            con.close()

    async def _check_entries(self):
        """Evaluate brackets for tradeable edge. PLACEHOLDER STRATEGY.

        Current rule: enter when |edge| > 5%. Replace with real strategy
        after dedicated brainstorm session.
        """
        # TODO: Real strategy logic
        pass

    async def _check_exits(self):
        """Monitor open positions for early exit signals. PLACEHOLDER.

        Current rule: hold to settlement. Replace with real exit logic.
        """
        # TODO: Real exit logic (edge flip, stop-loss, etc.)
        pass

    async def _settle_positions(self):
        """Resolve open positions when settlement data arrives."""
        con = duckdb.connect(self.db_path)
        try:
            # Find open positions with settlement data available
            rows = con.execute("""
                SELECT p.id, p.direction, p.entry_price, p.contracts,
                       p.bracket_floor, p.bracket_cap, p.fees,
                       n.max_temp_f
                FROM paper_positions p
                JOIN nws_daily n ON n.obs_date = p.event_date
                    AND n.station_id = 'KNYC'
                    AND n.source = 'NWS_CLI'
                WHERE p.status = 'open'
                  AND p.city = 'nyc'
            """).fetchall()

            for row in rows:
                pos_id, direction, entry_price, contracts, floor_s, cap_s, entry_fee, actual_high = row
                settled_yes = floor_s <= actual_high < cap_s

                if direction == "YES":
                    won = settled_yes
                else:  # NO
                    won = not settled_yes

                if won:
                    gross = (100 - entry_price) * contracts / 100.0
                else:
                    gross = -(entry_price * contracts / 100.0)

                net = round(gross - entry_fee, 2)

                con.execute("""
                    UPDATE paper_positions
                    SET status = 'closed', exit_reason = 'settlement',
                        settled_yes = ?, gross_pnl = ?, net_pnl = ?,
                        exit_time = ?, exit_price = ?
                    WHERE id = ?
                """, [
                    settled_yes, round(gross, 2), net,
                    datetime.now(timezone.utc),
                    100 if won else 0,
                    pos_id,
                ])
        finally:
            con.close()

    async def _update_unrealized(self):
        """Update unrealized P&L for open positions using current market mid."""
        con = duckdb.connect(self.db_path)
        try:
            rows = con.execute("""
                SELECT p.id, p.direction, p.entry_price, p.contracts,
                       p.bracket_floor, p.bracket_cap, p.city
                FROM paper_positions p
                WHERE p.status = 'open'
            """).fetchall()

            for row in rows:
                pos_id, direction, entry_price, contracts, floor_s, cap_s, city = row
                # Get latest market mid for this bracket
                tick = con.execute("""
                    SELECT (yes_bid + yes_ask) / 2.0 as mid
                    FROM market_ticks
                    WHERE city = ? AND floor_strike = ? AND cap_strike = ?
                    ORDER BY captured_at DESC LIMIT 1
                """, [city, floor_s, cap_s]).fetchone()

                if tick:
                    market_mid = tick[0]
                    if direction == "YES":
                        unrealized = (market_mid - entry_price) * contracts / 100.0
                    else:
                        unrealized = (entry_price - market_mid) * contracts / 100.0
                    con.execute(
                        "UPDATE paper_positions SET unrealized_pnl = ? WHERE id = ?",
                        [round(unrealized, 2), pos_id],
                    )
        finally:
            con.close()

    async def run(self):
        """Main loop — runs every 60s alongside other services."""
        logger.info("PaperTrader started (placeholder strategy)")
        while True:
            try:
                cycle_start = time.monotonic()
                await self._check_entries()
                await self._check_exits()
                await self._settle_positions()
                await self._update_unrealized()
                record_heartbeat(
                    "PaperTrader",
                    duration_ms=(time.monotonic() - cycle_start) * 1000,
                )
            except Exception as e:
                logger.error(f"PaperTrader error: {e}")
                record_heartbeat("PaperTrader", duration_ms=0, status="error", error=str(e))
            await asyncio.sleep(PAPER_TRADE_INTERVAL)
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_paper_trader.py -v`
Expected: PASS

**Step 5: Register PaperTrader in main.py**

Add to `main.py` imports and task list:

```python
from services.paper_trader import PaperTrader

# In startup, after existing service instances:
paper_trader = PaperTrader()

# Add to tasks list:
tasks.append(paper_trader.run())
```

**Step 6: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add services/paper_trader.py main.py tests/test_paper_trader.py
git commit -m "feat: add PaperTrader service shell with placeholder strategy"
```

---

### Task 6: Settlement Countdown + Trade Blotter Page

New `/blotter` page combining settlement countdown and bracket grid with positions.

**Files:**
- Create: `templates/blotter.html`
- Create: `static/js/blotter.js`
- Modify: `ui/web_dashboard.py` (add route + API endpoint)
- Test: `tests/test_blotter_endpoint.py`

**Step 1: Write failing test for blotter endpoint**

```python
# tests/test_blotter_endpoint.py
import pytest
from fastapi.testclient import TestClient
from ui.web_dashboard import app

client = TestClient(app)

def test_blotter_returns_expected_sections():
    resp = client.get("/api/blotter/nyc")
    assert resp.status_code == 200
    data = resp.json()
    assert "countdown" in data
    assert "brackets" in data
    assert "positions_summary" in data

def test_blotter_countdown_has_fields():
    resp = client.get("/api/blotter/nyc")
    data = resp.json()
    cd = data["countdown"]
    assert "event_date" in cd
    assert "model_high" in cd
    assert "model_bracket" in cd
    assert "running_obs_max" in cd
    assert "settlement" in cd
```

**Step 2: Run test to verify it fails**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_blotter_endpoint.py -v`
Expected: FAIL — 404

**Step 3: Implement `/api/blotter/{city}` endpoint**

Add to `ui/web_dashboard.py`:

```python
@app.get("/api/blotter/{city}")
async def blotter_data(city: str = "nyc", date: str = None):
    """Bundled data for the blotter page: countdown + brackets + positions."""
    from core.timezone import get_today_et, to_et

    target_date = date or get_today_et()
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        # --- Countdown ---
        forecast = prob_engine.calculate_city(city, None)
        model_high = forecast.center if forecast else None
        model_bracket = None
        model_bracket_prob = None
        if forecast and forecast.bracket_probs:
            # Find the 2°F bracket containing the model high
            floor_2f = int((model_high // 2) * 2)
            # Sum 1°F probs for the 2°F bracket
            model_bracket_prob = round(
                forecast.bracket_probs.get(floor_2f, 0)
                + forecast.bracket_probs.get(floor_2f + 1, 0), 3
            )
            model_bracket = f"{floor_2f}-{floor_2f + 2}"

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

        # Market close time (settlement time)
        close_row = con.execute("""
            SELECT close_time FROM kalshi_settlements
            WHERE city = ? AND event_date = ?
            LIMIT 1
        """, [city, target_date]).fetchone()

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

        # --- Brackets (reuse existing logic) ---
        brackets_resp = await api_brackets(city, target_date)

        # --- Positions ---
        positions = con.execute("""
            SELECT bracket_floor, bracket_cap, direction, contracts,
                   entry_price, unrealized_pnl, net_pnl, status, exit_reason
            FROM paper_positions
            WHERE city = ? AND event_date = ?
            ORDER BY bracket_floor
        """, [city, target_date]).fetchall()

        position_list = []
        for p in positions:
            position_list.append({
                "bracket_floor": p[0], "bracket_cap": p[1],
                "direction": p[2], "contracts": p[3],
                "entry_price": p[4],
                "pnl": p[5] if p[7] == "open" else p[6],
                "status": p[7], "exit_reason": p[8],
            })

        # Summary
        open_count = sum(1 for p in positions if p[7] == "open")
        day_exposure = sum((p[4] * p[3] / 100.0) for p in positions if p[7] == "open")
        day_pnl = sum(
            (p[5] if p[7] == "open" else (p[6] or 0))
            for p in positions
        )

        return {
            "countdown": countdown,
            "brackets": brackets_resp,
            "positions": position_list,
            "positions_summary": {
                "open_count": open_count,
                "day_exposure": round(day_exposure, 2),
                "day_pnl": round(day_pnl, 2),
            },
        }
    finally:
        con.close()
```

Add the page route:

```python
@app.get("/blotter")
async def blotter_page(request: Request):
    return templates.TemplateResponse("blotter.html", {
        "request": request, "active_tab": "blotter"
    })
```

**Step 4: Run test to verify it passes**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_blotter_endpoint.py -v`
Expected: PASS

**Step 5: Create blotter.html template**

```html
<!-- templates/blotter.html -->
{% extends "base.html" %}
{% block title %}Blotter — AlphaTemp{% endblock %}
{% block content %}
<div class="p-4 space-y-4 max-w-6xl mx-auto">

  <!-- Settlement Countdown -->
  <div id="countdown-section" class="bg-slate-800/50 rounded-lg p-4">
    <h2 class="text-sm font-bold text-slate-300 mb-3 tracking-wide">SETTLEMENT COUNTDOWN</h2>
    <div id="countdown-today" class="mb-4"></div>
    <div id="countdown-tomorrow"></div>
  </div>

  <!-- Trade Blotter -->
  <div class="bg-slate-800/50 rounded-lg p-4">
    <h2 class="text-sm font-bold text-slate-300 mb-3 tracking-wide">TRADE BLOTTER</h2>
    <table class="w-full text-xs font-mono">
      <thead>
        <tr class="text-slate-500 border-b border-slate-700">
          <th class="py-2 text-left">Bracket</th>
          <th class="py-2 text-right">Model</th>
          <th class="py-2 text-right">Market</th>
          <th class="py-2 text-right">Edge</th>
          <th class="py-2 text-center">Dir</th>
          <th class="py-2 text-right">Pos</th>
          <th class="py-2 text-right">Entry</th>
          <th class="py-2 text-right">P&L</th>
        </tr>
      </thead>
      <tbody id="blotter-body"></tbody>
    </table>
    <div id="blotter-summary" class="mt-3 pt-3 border-t border-slate-700 text-xs text-slate-400"></div>
  </div>

</div>
{% endblock %}
{% block scripts %}
<script src="/static/js/blotter.js"></script>
{% endblock %}
```

**Step 6: Create blotter.js**

```javascript
// static/js/blotter.js

var countdownInterval = null;
var closeTimeUTC = null;

function startCountdownTicker() {
    if (countdownInterval) clearInterval(countdownInterval);
    countdownInterval = setInterval(function() {
        if (!closeTimeUTC) return;
        var now = Date.now();
        var diff = new Date(closeTimeUTC).getTime() - now;
        if (diff <= 0) {
            document.getElementById('countdown-timer').textContent = 'Settled';
            return;
        }
        var h = Math.floor(diff / 3600000);
        var m = Math.floor((diff % 3600000) / 60000);
        var s = Math.floor((diff % 60000) / 1000);
        document.getElementById('countdown-timer').textContent =
            h + 'h ' + m + 'm ' + s + 's';
    }, 1000);
}

function renderCountdown(cd, elId) {
    var el = document.getElementById(elId);
    if (!el || !cd) { if (el) el.innerHTML = ''; return; }

    closeTimeUTC = cd.close_time;

    var settlementText = 'Pending';
    if (cd.settlement.source === 'NWS_CLI') {
        settlementText = cd.settlement.temp + '°F (CLI — Final)';
    } else if (cd.settlement.source === 'DSM') {
        settlementText = cd.settlement.temp + '°F (DSM)';
    }

    var obsMaxText = cd.running_obs_max !== null
        ? cd.running_obs_max + '°F'
        : '--';
    if (cd.obs_max_time) {
        obsMaxText += ' (as of ' + toET(cd.obs_max_time, {timeOnly: true}) + ')';
    }

    el.innerHTML =
        '<div class="flex items-baseline gap-4 mb-2">'
        + '<span class="text-slate-300 font-bold">KXHIGHNY</span>'
        + '<span class="text-slate-500">' + cd.event_date + '</span>'
        + '<span class="text-white">Settles in: <strong id="countdown-timer">--</strong></span>'
        + '</div>'
        + '<div class="grid grid-cols-2 gap-x-8 gap-y-1 text-slate-400">'
        + '<span>Model High: <strong class="text-white">'
            + (cd.model_high ? cd.model_high + '°F' : '--') + '</strong>'
            + (cd.model_bracket ? ' → ' + cd.model_bracket + '°F ('
                + Math.round((cd.model_bracket_prob || 0) * 100) + '%)' : '')
        + '</span>'
        + '<span>Running Obs Max: <strong class="text-white">' + obsMaxText + '</strong></span>'
        + '<span>Settlement: <strong class="text-white">' + settlementText + '</strong></span>'
        + '</div>';

    startCountdownTicker();
}

function renderBlotter(brackets, positions) {
    var body = document.getElementById('blotter-body');
    if (!body) return;

    // Index positions by bracket
    var posMap = {};
    (positions || []).forEach(function(p) {
        posMap[p.bracket_floor + '-' + p.bracket_cap] = p;
    });

    var html = '';
    var bracketList = (brackets && brackets.brackets) ? brackets.brackets : [];
    bracketList.forEach(function(b) {
        var key = b.floor + '-' + b.cap;
        var pos = posMap[key];
        var hasPos = pos && pos.status === 'open';
        var rowClass = hasPos ? 'text-white' : 'text-slate-500';
        var indicator = hasPos ? '<span class="text-emerald-400 mr-1">●</span>' : '';

        var edge = b.model_prob - (b.market_mid || 0) / 100;
        var edgeFmt = formatEdge(edge * 100);

        html += '<tr class="' + rowClass + ' border-b border-slate-800/50">'
            + '<td class="py-1.5">' + indicator + b.floor + '-' + b.cap + '°F</td>'
            + '<td class="text-right">' + (b.model_prob * 100).toFixed(1) + '%</td>'
            + '<td class="text-right">' + (b.market_mid || '--') + '¢</td>'
            + '<td class="text-right ' + edgeFmt.colorClass + '">' + edgeFmt.text + '</td>'
            + '<td class="text-center">' + (pos ? pos.direction : '—') + '</td>'
            + '<td class="text-right">' + (pos ? pos.contracts : '—') + '</td>'
            + '<td class="text-right">' + (pos ? pos.entry_price + '¢' : '—') + '</td>'
            + '<td class="text-right">'
                + (pos ? formatPnL(pos.pnl || 0).text : '—')
            + '</td>'
            + '</tr>';
    });
    body.innerHTML = html;
}

function renderSummary(summary) {
    var el = document.getElementById('blotter-summary');
    if (!el || !summary) return;
    var pnl = formatPnL(summary.day_pnl || 0);
    el.innerHTML =
        'Open: <strong class="text-white">' + summary.open_count + '</strong>'
        + ' · Exposure: <strong class="text-white">$' + (summary.day_exposure || 0).toFixed(2) + '</strong>'
        + ' · Day P&L: <strong class="' + pnl.colorClass + '">' + pnl.text + '</strong>';
}

async function refreshBlotter() {
    var date = getTargetDate(window.selectedDate || 'today');
    var data = await fetchAPI('/api/blotter/nyc?date=' + date);
    if (!data) return;

    renderCountdown(data.countdown, 'countdown-today');
    renderBlotter(data.brackets, data.positions);
    renderSummary(data.positions_summary);

    // Also fetch tomorrow for the second countdown
    var tmrw = getTargetDate('tomorrow');
    var tmrwData = await fetchAPI('/api/blotter/nyc?date=' + tmrw);
    if (tmrwData && tmrwData.countdown) {
        renderCountdown(tmrwData.countdown, 'countdown-tomorrow');
    }
}

function onDateChange() {
    refreshBlotter();
}

refreshBlotter();
setInterval(refreshBlotter, 60000);
```

**Step 7: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add templates/blotter.html static/js/blotter.js ui/web_dashboard.py tests/test_blotter_endpoint.py
git commit -m "feat: add settlement countdown and trade blotter page"
```

---

### Task 7: Polish Operations Page

Add stale data warning, settlement source label, and error toasts.

**Files:**
- Modify: `templates/base.html` (add toast container)
- Modify: `static/js/shared.js` (add toast function)
- Modify: `static/js/operations.js` (stale warning, settlement label)
- Modify: `static/css/dashboard.css` (toast styles)

**Step 1: Add toast notification system to shared.js**

Append to `static/js/shared.js`:

```javascript
function showToast(message, type) {
    type = type || 'error';
    var container = document.getElementById('toast-container');
    if (!container) return;
    var colors = {
        error: 'bg-red-900/80 border-red-700 text-red-200',
        warning: 'bg-amber-900/80 border-amber-700 text-amber-200',
        info: 'bg-slate-800/80 border-slate-600 text-slate-300',
    };
    var toast = document.createElement('div');
    toast.className = 'px-4 py-2 rounded border text-xs font-mono mb-2 transition-opacity duration-500 ' + (colors[type] || colors.info);
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(function() { toast.style.opacity = '0'; }, 8000);
    setTimeout(function() { toast.remove(); }, 10000);
}
```

Update `fetchAPI` in `shared.js` to show toast on failure:

```javascript
// Replace the existing fetchAPI error handling:
// On fetch failure, call showToast with the error message
```

**Step 2: Add toast container to base.html**

Add before closing `</body>` tag:

```html
<div id="toast-container" class="fixed bottom-4 right-4 z-50 max-w-sm"></div>
```

**Step 3: Add stale data warning to operations.js**

In `refreshAll()`, after fetching health data:

```javascript
async function checkStaleness() {
    var health = await fetchAPI('/api/health');
    if (!health) return;
    var banner = document.getElementById('stale-banner');
    if (!banner) return;
    var warnings = [];
    if (health.obs_stale) warnings.push('Observations (' + Math.round(health.obs_age_minutes) + ' min)');
    if (health.fcst_stale) warnings.push('Forecasts (' + Math.round(health.fcst_age_minutes) + ' min)');
    if (warnings.length > 0) {
        banner.textContent = '⚠ Stale data: ' + warnings.join(', ');
        banner.classList.remove('hidden');
    } else {
        banner.classList.add('hidden');
    }
}
```

Add to `operations.html` at top of content block:

```html
<div id="stale-banner" class="hidden bg-amber-900/50 border border-amber-700 text-amber-200 text-xs px-4 py-2 rounded mb-2"></div>
```

**Step 4: Update settlement marker label in operations.js**

In `refreshTempChart()`, update the settlement marker trace name (~line 210-226) to include source:

```javascript
// Change the settlement trace name to show source:
name: data.settlement_source === 'nws_cli' ? 'NWS Settlement (CLI)'
    : data.settlement_source === 'dsm' ? 'Settlement (DSM)'
    : 'Running High (est)',
```

**Step 5: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add static/js/shared.js static/js/operations.js templates/base.html templates/operations.html static/css/dashboard.css
git commit -m "fix: add stale data warnings, settlement source labels, error toasts"
```

---

### Task 8: Polish Performance Page

Wire up Brier comparison stub and add paper P&L tracking.

**Files:**
- Modify: `ui/web_dashboard.py:1127-1134` (Brier endpoint)
- Modify: `static/js/performance.js` (render Brier chart, add P&L section)

**Step 1: Implement Brier comparison endpoint**

Replace the stub at `web_dashboard.py:1127-1134`:

```python
@app.get("/api/brier-comparison")
async def brier_comparison(time_range: str = Query("30d", alias="range")):
    """Model vs market Brier score comparison over time."""
    days = {"7d": 7, "30d": 30, "all": 9999}.get(time_range, 30)
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        rows = con.execute("""
            WITH settled AS (
                SELECT
                    s.event_date,
                    s.floor_strike,
                    s.cap_strike,
                    s.settled_yes,
                    s.close_time
                FROM kalshi_settlements s
                WHERE s.city = 'nyc'
                  AND s.settled_yes IS NOT NULL
                  AND s.event_date >= CURRENT_DATE - INTERVAL ? DAY
                ORDER BY s.event_date
            )
            SELECT
                s.event_date,
                -- Market Brier: (market_prob - outcome)^2
                AVG(POWER(
                    COALESCE(c.price_close, 50) / 100.0
                    - CASE WHEN s.settled_yes THEN 1.0 ELSE 0.0 END,
                2)) as market_brier,
                COUNT(*) as n_brackets
            FROM settled s
            LEFT JOIN kalshi_candlesticks c ON c.market_ticker = (
                SELECT market_ticker FROM kalshi_settlements s2
                WHERE s2.event_date = s.event_date
                  AND s2.floor_strike = s.floor_strike
                  AND s2.cap_strike = s.cap_strike
                LIMIT 1
            ) AND c.period_minutes = 60
              AND c.end_period_ts = (
                  SELECT MAX(end_period_ts) FROM kalshi_candlesticks c2
                  WHERE c2.market_ticker = c.market_ticker
                    AND c2.period_minutes = 60
              )
            GROUP BY s.event_date
            ORDER BY s.event_date
        """, [days]).fetchall()

        return {
            "dates": [r[0].isoformat() for r in rows],
            "market_brier": [round(r[1], 4) if r[1] else None for r in rows],
            "model_brier": [],  # TODO: requires storing model predictions at settlement time
            "note": "Model Brier requires logging predictions at market close — not yet implemented",
        }
    finally:
        con.close()
```

**Step 2: Update performance.js to render Brier chart**

Find the Brier chart render function in `performance.js` and update it to use the actual data instead of showing "requires backtester integration."

**Step 3: Add paper P&L section to performance page**

Add a new section to `performance.html` and `performance.js` that queries paper_positions for cumulative P&L:

```javascript
async function refreshPaperPnL(range) {
    var data = await fetchAPI('/api/performance?range=' + range);
    if (!data || !data.daily) return;
    // Render alongside existing charts
}
```

**Step 4: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add ui/web_dashboard.py static/js/performance.js templates/performance.html
git commit -m "fix: wire up Brier comparison endpoint and add paper P&L to performance"
```

---

### Task 9: Polish Review + Mobile Pages

Fix missed_edge filter, add settlement source to incidents, update mobile layout.

**Files:**
- Modify: `ui/web_dashboard.py` (review incidents endpoint — missed_edge categorization)
- Modify: `static/js/review.js` (settlement source on cards)
- Modify: `templates/mobile.html` (inherit KPI header, add blotter summary)
- Modify: `static/js/mobile.js` (fetch blotter summary)

**Step 1: Fix missed_edge categorization**

In `web_dashboard.py`, find `_categorize_incident` and ensure the `missed_edge` category is returned when the model had edge > 5% but no position was taken.

**Step 2: Add settlement source to incident cards**

In `review.js`, update the card rendering to include the settlement source label:

```javascript
// Add after the existing card content:
+ '<div class="text-slate-600 text-[10px]">Settlement: ' + (incident.settlement_source || 'unknown') + '</div>'
```

The API endpoint needs to join `nws_daily` to include the settlement source in incident data.

**Step 3: Update mobile to use KPI header**

`mobile.html` already extends `base.html`, so it inherits the KPI header automatically after Task 2. Remove any duplicate header/nav elements from `mobile.html`.

**Step 4: Add compact blotter summary to mobile**

Add to `mobile.js`:

```javascript
async function refreshMobileBlotter() {
    var date = getTargetDate(window.selectedDate || 'today');
    var data = await fetchAPI('/api/blotter/nyc?date=' + date);
    if (!data) return;
    var el = document.getElementById('mobile-blotter');
    if (!el) return;
    var positions = data.positions || [];
    var open = positions.filter(function(p) { return p.status === 'open'; });
    if (open.length === 0) {
        el.innerHTML = '<div class="text-slate-600 text-xs">No open positions</div>';
        return;
    }
    var html = '';
    open.forEach(function(p) {
        var pnl = formatPnL(p.pnl || 0);
        html += '<div class="flex justify-between py-1 border-b border-slate-800/50 text-xs">'
            + '<span>' + p.bracket_floor + '-' + p.bracket_cap + '°F ' + p.direction + '</span>'
            + '<span class="' + pnl.colorClass + '">' + pnl.text + '</span>'
            + '</div>';
    });
    el.innerHTML = html;
}
```

Add `<div id="mobile-blotter">` section to `mobile.html`.

**Step 5: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add ui/web_dashboard.py static/js/review.js static/js/mobile.js templates/mobile.html templates/review.html
git commit -m "fix: missed_edge filter, settlement source on incidents, mobile blotter summary"
```

---

### Task 10: Cross-Cutting Polish + Loading States

Add skeleton loading placeholders and verify everything works end-to-end.

**Files:**
- Modify: `static/css/dashboard.css` (skeleton animation)
- Modify: `static/js/shared.js` (skeleton helper)
- Modify: all page JS files (add loading states)

**Step 1: Add skeleton CSS**

Append to `static/css/dashboard.css`:

```css
.skeleton {
    background: linear-gradient(90deg, #1e293b 25%, #334155 50%, #1e293b 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s ease-in-out infinite;
    border-radius: 4px;
    height: 1em;
}
@keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
```

**Step 2: Add skeleton helper to shared.js**

```javascript
function showSkeleton(containerId, rows) {
    rows = rows || 5;
    var el = document.getElementById(containerId);
    if (!el) return;
    var html = '';
    for (var i = 0; i < rows; i++) {
        html += '<div class="skeleton mb-2" style="width:' + (60 + Math.random() * 30) + '%; height: 14px;"></div>';
    }
    el.innerHTML = html;
}
```

**Step 3: Add loading states to each page**

In each page's JS, call `showSkeleton()` for the relevant containers before the first data fetch. The skeleton is replaced when actual content renders.

**Step 4: End-to-end verification**

Run: `cd ~/Projects/alphatemp/alphatemp && python main.py --dashboard`
Verify manually:
- [ ] KPI header appears on all pages with live data
- [ ] `/health` page shows freshness, pipelines, DB stats, config
- [ ] `/blotter` page shows countdown timer ticking, bracket grid
- [ ] Operations page shows stale warning when applicable
- [ ] Error toasts appear on API failures
- [ ] Mobile inherits KPI header and shows blotter summary
- [ ] Loading skeletons appear briefly before data loads

**Step 5: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add static/css/dashboard.css static/js/shared.js static/js/operations.js static/js/performance.js static/js/review.js static/js/mobile.js static/js/health.js static/js/blotter.js
git commit -m "feat: add loading skeletons and cross-cutting polish"
```

---

## Summary

| Task | What | New Files | Commits |
|------|------|-----------|---------|
| 1 | Schema + heartbeat infra | `core/heartbeat.py`, `tests/test_heartbeat.py` | 1 |
| 2 | KPI Header Bar | `tests/test_kpi_endpoint.py` | 1 |
| 3 | System Health Page | `templates/health.html`, `static/js/health.js`, `tests/test_health_detailed.py` | 1 |
| 4 | Heartbeat integration | — | 1 |
| 5 | PaperTrader shell | `services/paper_trader.py`, `tests/test_paper_trader.py` | 1 |
| 6 | Blotter page | `templates/blotter.html`, `static/js/blotter.js`, `tests/test_blotter_endpoint.py` | 1 |
| 7 | Polish operations | — | 1 |
| 8 | Polish performance | — | 1 |
| 9 | Polish review + mobile | — | 1 |
| 10 | Loading states + verify | — | 1 |

**Total: 10 tasks, 10 commits, 8 new files**

## Follow-up (separate sessions)
- Paper trading strategy brainstorm (entry/exit rules, thresholds, sizing)
- Portfolio & Risk page
- Execution Monitor page
