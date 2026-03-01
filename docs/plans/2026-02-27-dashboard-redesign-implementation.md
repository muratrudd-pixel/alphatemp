# Dashboard Redesign Implementation Plan

> **Fee Notice (2026-02-28):** The hardcoded `0.11` fee hurdle in JS code below is WRONG. Actual Kalshi fee: taker fee = max(ceil(0.07*C*P*(1-P)), C*$0.01). No settlement fee. Real hurdle ~1-2%. When implementing, use dynamic fee calculation.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the existing single-page dashboard with a three-tab trading terminal (Operations, Performance, Review) plus a mobile Trade view.

**Architecture:** FastAPI serves separate HTML templates per tab. Each tab has its own JS file for rendering + polling. Shared utilities in `shared.js`. New `paper_positions` table tracks model-suggested trades. All new API endpoints read from existing DuckDB tables + `paper_positions`.

**Tech Stack:** FastAPI, Jinja2, Vanilla JS, Plotly.js, Tailwind CSS (CDN), DuckDB

**Design doc:** `docs/plans/2026-02-27-dashboard-redesign-design.md`

---

## Phase 1: Foundation

### Task 1: Create `paper_positions` table

**Files:**
- Modify: `core/db.py` (add table creation after existing tables)
- Test: `tests/test_db.py` (add table existence test)

**Step 1: Write the failing test**

Add to `tests/test_db.py`:
```python
def test_paper_positions_table_exists(tmp_db):
    """paper_positions table should be created by init_db."""
    con = duckdb.connect(str(tmp_db))
    tables = con.execute("SELECT table_name FROM information_schema.tables WHERE table_name = 'paper_positions'").fetchall()
    con.close()
    assert len(tables) == 1
```

**Step 2: Run test to verify it fails**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_db.py::test_paper_positions_table_exists -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `core/db.py` `init_db()` function, after the last `CREATE TABLE`:
```python
    con.execute("""
        CREATE TABLE IF NOT EXISTS paper_positions (
            id INTEGER,
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
            status VARCHAR DEFAULT 'open'
        )
    """)
    con.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_positions_id
        ON paper_positions(id)
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_paper_positions_city_date
        ON paper_positions(city, event_date)
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_paper_positions_status
        ON paper_positions(status)
    """)
```

**Step 4: Run test to verify it passes**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_db.py::test_paper_positions_table_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add core/db.py tests/test_db.py
git commit -m "feat: add paper_positions table for tracking model-suggested trades"
```

---

### Task 2: Create shared base template and static file structure

**Files:**
- Create: `templates/base.html`
- Create: `static/js/shared.js`
- Create: `static/css/dashboard.css`
- Modify: `ui/web_dashboard.py` (mount static files, add tab routes)

**Step 1: Create directory structure**

Run: `mkdir -p ~/Projects/alphatemp/alphatemp/static/js ~/Projects/alphatemp/alphatemp/static/css`

**Step 2: Create `templates/base.html`**

This is the shared layout all tabs extend. Extract the head, nav, and footer from the existing `index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlphaTemp — {% block title %}Dashboard{% endblock %}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/static/css/dashboard.css">
    <script src="/static/js/shared.js"></script>
    {% block head %}{% endblock %}
</head>
<body class="bg-slate-900 text-slate-300 font-mono min-h-screen">
    <!-- Header -->
    <header class="border-b border-slate-700 px-6 py-3 flex items-center justify-between">
        <div class="flex items-center gap-4">
            <div class="flex items-center gap-2">
                <span id="status-dot" class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                <span class="text-lg font-bold text-slate-100">AlphaTemp</span>
            </div>
            <span class="text-slate-500">NYC</span>
        </div>
        <!-- Tab Navigation -->
        <nav class="hidden md:flex items-center gap-1">
            <a href="/operations" class="px-4 py-2 rounded text-sm {% if active_tab == 'operations' %}bg-slate-700 text-slate-100{% else %}text-slate-400 hover:text-slate-200 hover:bg-slate-800{% endif %} transition">Operations</a>
            <a href="/performance" class="px-4 py-2 rounded text-sm {% if active_tab == 'performance' %}bg-slate-700 text-slate-100{% else %}text-slate-400 hover:text-slate-200 hover:bg-slate-800{% endif %} transition">Performance</a>
            <a href="/review" class="px-4 py-2 rounded text-sm {% if active_tab == 'review' %}bg-slate-700 text-slate-100{% else %}text-slate-400 hover:text-slate-200 hover:bg-slate-800{% endif %} transition">Review</a>
        </nav>
        <div class="flex items-center gap-4">
            <!-- Date Toggle -->
            <div class="flex items-center gap-2 text-sm">
                <button id="btn-today" onclick="setDateToggle('today')" class="px-3 py-1 rounded bg-slate-700 text-slate-100">Today</button>
                <button id="btn-tomorrow" onclick="setDateToggle('tomorrow')" class="px-3 py-1 rounded text-slate-400 hover:bg-slate-800">Tomorrow</button>
            </div>
            <div class="text-sm text-slate-400">
                <span id="et-clock"></span>
            </div>
            <div class="text-xs text-slate-500">
                <span id="refresh-countdown"></span>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="p-4 md:p-6">
        {% block content %}{% endblock %}
    </main>

    <!-- Shared state + refresh logic -->
    <script>
        const selectedCity = 'NYC';
        let selectedDate = 'today';
        let refreshTimer = null;
        let countdown = 60;

        function setDateToggle(which) {
            selectedDate = which;
            document.getElementById('btn-today').className = which === 'today'
                ? 'px-3 py-1 rounded bg-slate-700 text-slate-100'
                : 'px-3 py-1 rounded text-slate-400 hover:bg-slate-800';
            document.getElementById('btn-tomorrow').className = which === 'tomorrow'
                ? 'px-3 py-1 rounded bg-slate-700 text-slate-100'
                : 'px-3 py-1 rounded text-slate-400 hover:bg-slate-800';
            if (typeof refreshAll === 'function') refreshAll();
        }

        function updateClock() {
            const now = new Date();
            document.getElementById('et-clock').textContent =
                now.toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', second: '2-digit' }) + ' ET';
        }

        function updateCountdown() {
            countdown--;
            if (countdown <= 0) {
                countdown = 60;
                if (typeof refreshAll === 'function') refreshAll();
            }
            const el = document.getElementById('refresh-countdown');
            if (el) el.textContent = countdown + 's';
        }

        setInterval(updateClock, 1000);
        setInterval(updateCountdown, 1000);
        updateClock();
    </script>
    {% block scripts %}{% endblock %}
</body>
</html>
```

**Step 3: Create `static/js/shared.js`**

```javascript
// AlphaTemp shared utilities

/**
 * Convert UTC ISO string to ET display string.
 * @param {string} utcIso - UTC ISO timestamp
 * @param {object} opts - Intl.DateTimeFormat options override
 * @returns {string} Formatted ET time string
 */
function toET(utcIso, opts) {
    const defaults = { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit' };
    return new Date(utcIso).toLocaleTimeString('en-US', opts || defaults);
}

/**
 * Get the target date string (YYYY-MM-DD) based on selected toggle.
 * @param {string} which - 'today' or 'tomorrow'
 * @returns {string} ISO date string in ET
 */
function getTargetDate(which) {
    const now = new Date();
    const etNow = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
    if (which === 'tomorrow') {
        etNow.setDate(etNow.getDate() + 1);
    }
    return etNow.toISOString().slice(0, 10);
}

/**
 * Fetch JSON from API with error handling.
 * @param {string} url - API endpoint URL
 * @returns {Promise<object|null>} Parsed JSON or null on error
 */
async function fetchAPI(url) {
    try {
        const resp = await fetch(url);
        if (!resp.ok) return null;
        return await resp.json();
    } catch (e) {
        console.error('API fetch failed:', url, e);
        return null;
    }
}

/**
 * Format a dollar amount with sign and color class.
 * @param {number} amount - Dollar amount
 * @returns {object} { text, colorClass }
 */
function formatPnL(amount) {
    if (amount == null) return { text: '--', colorClass: 'text-slate-400' };
    const sign = amount >= 0 ? '+' : '';
    return {
        text: sign + '$' + Math.abs(amount).toFixed(2),
        colorClass: amount >= 0 ? 'text-emerald-400' : 'text-red-400'
    };
}

/**
 * Format edge percentage with color.
 * @param {number} edge - Edge as decimal (0.12 = 12%)
 * @returns {object} { text, colorClass }
 */
function formatEdge(edge) {
    if (edge == null) return { text: '--', colorClass: 'text-slate-400' };
    const pct = (edge * 100).toFixed(1);
    const sign = edge >= 0 ? '+' : '';
    return {
        text: sign + pct + '%',
        colorClass: edge >= 0.10 ? 'text-emerald-400' : edge >= 0 ? 'text-amber-400' : 'text-red-400'
    };
}

/**
 * Plotly dark theme layout defaults.
 */
const PLOTLY_LAYOUT = {
    paper_bgcolor: '#0f172a',
    plot_bgcolor: '#0f172a',
    font: { family: 'Space Mono, monospace', color: '#94a3b8', size: 11 },
    margin: { l: 50, r: 20, t: 30, b: 40 },
    xaxis: { gridcolor: '#1e293b', zerolinecolor: '#334155' },
    yaxis: { gridcolor: '#1e293b', zerolinecolor: '#334155' },
};

const PLOTLY_CONFIG = { displayModeBar: false, responsive: true };

/**
 * Color constants matching Tailwind.
 */
const COLORS = {
    green: '#22c55e',
    red: '#ef4444',
    amber: '#f59e0b',
    blue: '#3b82f6',
    teal: '#14b8a6',
    white: '#f8fafc',
    slate400: '#94a3b8',
    slate500: '#64748b',
    slate700: '#334155',
};
```

**Step 4: Create `static/css/dashboard.css`**

```css
/* AlphaTemp dashboard custom styles */
body { font-family: 'Space Mono', monospace; }

/* Flash animation for new data */
@keyframes flash-new {
    0% { background-color: rgba(59, 130, 246, 0.3); }
    100% { background-color: transparent; }
}
.flash-new { animation: flash-new 2s ease-out; }

/* Active bet glow */
.bet-active { box-shadow: 0 0 8px rgba(34, 197, 94, 0.3); }

/* Severity bar */
.severity-bar {
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(to right, #22c55e, #f59e0b, #ef4444);
}

/* Scrollable feed panel */
.feed-scroll {
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: #334155 transparent;
}
.feed-scroll::-webkit-scrollbar { width: 4px; }
.feed-scroll::-webkit-scrollbar-thumb { background: #334155; border-radius: 2px; }
```

**Step 5: Update `ui/web_dashboard.py` — mount static files and add tab routes**

Add static file mounting after FastAPI app creation (near line 16):
```python
from fastapi.staticfiles import StaticFiles
import os

# After app = FastAPI(...)
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
```

Replace the existing `GET /` route (lines 40-43) with:
```python
@app.get("/")
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/operations")

@app.get("/operations")
async def operations_page(request: Request):
    return templates.TemplateResponse("operations.html", {"request": request, "active_tab": "operations"})

@app.get("/performance")
async def performance_page(request: Request):
    return templates.TemplateResponse("performance.html", {"request": request, "active_tab": "performance"})

@app.get("/review")
async def review_page(request: Request):
    return templates.TemplateResponse("review.html", {"request": request, "active_tab": "review"})

@app.get("/mobile")
async def mobile_page(request: Request):
    return templates.TemplateResponse("mobile.html", {"request": request, "active_tab": "mobile"})
```

**Step 6: Create stub templates for each tab**

Create `templates/operations.html`:
```html
{% extends "base.html" %}
{% block title %}Operations{% endblock %}
{% block content %}
<div class="text-center text-slate-500 py-20">Operations tab — coming soon</div>
{% endblock %}
```

Create `templates/performance.html`:
```html
{% extends "base.html" %}
{% block title %}Performance{% endblock %}
{% block content %}
<div class="text-center text-slate-500 py-20">Performance tab — coming soon</div>
{% endblock %}
```

Create `templates/review.html`:
```html
{% extends "base.html" %}
{% block title %}Review{% endblock %}
{% block content %}
<div class="text-center text-slate-500 py-20">Review tab — coming soon</div>
{% endblock %}
```

Create `templates/mobile.html`:
```html
{% extends "base.html" %}
{% block title %}Trade{% endblock %}
{% block content %}
<div class="text-center text-slate-500 py-20">Mobile view — coming soon</div>
{% endblock %}
```

**Step 7: Write test for new routes**

Add to `tests/test_web_dashboard.py`:
```python
def test_root_redirects_to_operations(client):
    """GET / should redirect to /operations."""
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 307
    assert "/operations" in resp.headers["location"]

def test_operations_page(client):
    """GET /operations should return 200 with HTML."""
    resp = client.get("/operations")
    assert resp.status_code == 200
    assert b"Operations" in resp.content

def test_performance_page(client):
    """GET /performance should return 200 with HTML."""
    resp = client.get("/performance")
    assert resp.status_code == 200
    assert b"Performance" in resp.content

def test_review_page(client):
    """GET /review should return 200 with HTML."""
    resp = client.get("/review")
    assert resp.status_code == 200
    assert b"Review" in resp.content
```

**Step 8: Run tests**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_web_dashboard.py -v -k "redirect or operations_page or performance_page or review_page"`
Expected: PASS

**Step 9: Commit**

```bash
git add templates/base.html templates/operations.html templates/performance.html templates/review.html templates/mobile.html static/ ui/web_dashboard.py tests/test_web_dashboard.py
git commit -m "feat: add multi-tab layout with base template, shared JS/CSS, tab routes"
```

---

## Phase 2: Operations Tab

### Task 3: Build the temperature curve panel (port from existing)

**Files:**
- Modify: `templates/operations.html`
- Create: `static/js/operations.js`

**Step 1: Build the operations.html layout grid**

Replace the stub content in `templates/operations.html`:
```html
{% extends "base.html" %}
{% block title %}Operations{% endblock %}
{% block content %}
<div class="grid grid-cols-1 lg:grid-cols-5 gap-4">
    <!-- Temperature Curve: 3/5 width -->
    <div class="lg:col-span-3 bg-slate-800 rounded-lg p-4 border border-slate-700">
        <div class="flex items-center justify-between mb-2">
            <h2 class="text-sm font-bold text-slate-100">Temperature Curve</h2>
            <div class="flex gap-2">
                <button id="btn-neighbor-klga" onclick="toggleNeighbor('KLGA')" class="text-xs px-2 py-1 rounded border border-slate-600 text-slate-400 hover:text-slate-200">KLGA</button>
                <button id="btn-neighbor-kewr" onclick="toggleNeighbor('KEWR')" class="text-xs px-2 py-1 rounded border border-slate-600 text-slate-400 hover:text-slate-200">KEWR</button>
            </div>
        </div>
        <div id="temp-chart" style="height: 350px;"></div>
    </div>

    <!-- Right column: Obs Feed + Forecast Runs: 2/5 width -->
    <div class="lg:col-span-2 flex flex-col gap-4">
        <!-- Observation Feed -->
        <div class="bg-slate-800 rounded-lg p-4 border border-slate-700 flex-1">
            <h2 class="text-sm font-bold text-slate-100 mb-2">Observations</h2>
            <div id="obs-feed" class="feed-scroll" style="max-height: 200px;"></div>
        </div>
        <!-- Forecast Runs -->
        <div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
            <h2 class="text-sm font-bold text-slate-100 mb-2">Forecast Runs</h2>
            <div id="fcst-feed"></div>
        </div>
    </div>
</div>

<!-- Bottom row: Bracket Spread + Positions -->
<div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
    <!-- Bracket Spread -->
    <div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
        <h2 class="text-sm font-bold text-slate-100 mb-2">Bracket Spread</h2>
        <div id="bracket-chart" style="height: 280px;"></div>
        <div id="liquidity-bar" class="mt-2"></div>
    </div>
    <!-- Positions & Alerts -->
    <div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
        <h2 class="text-sm font-bold text-slate-100 mb-2">Positions & Alerts</h2>
        <div id="positions-panel"></div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script src="/static/js/operations.js"></script>
{% endblock %}
```

**Step 2: Create `static/js/operations.js` — temperature curve renderer**

Port the `renderRibbon()` function from existing `index.html` (lines 307-527), adapting to use shared utilities:

```javascript
// operations.js — Operations tab rendering + polling

let neighborsActive = {};

function toggleNeighbor(station) {
    neighborsActive[station] = !neighborsActive[station];
    const btn = document.getElementById('btn-neighbor-' + station.toLowerCase());
    if (neighborsActive[station]) {
        btn.classList.add('bg-slate-600', 'text-slate-100');
        btn.classList.remove('text-slate-400');
    } else {
        btn.classList.remove('bg-slate-600', 'text-slate-100');
        btn.classList.add('text-slate-400');
    }
    refreshTempChart();
}

async function refreshTempChart() {
    const date = getTargetDate(selectedDate);
    const data = await fetchAPI('/api/forecast-curve/' + selectedCity + '?date=' + date);
    if (!data) return;

    const traces = [];

    // Confidence ribbon (90%)
    if (data.ribbon && data.ribbon.length > 0) {
        const rTimes = data.ribbon.map(r => r.valid_at);
        traces.push({
            x: rTimes.concat(rTimes.slice().reverse()),
            y: data.ribbon.map(r => r.upper_90).concat(data.ribbon.slice().reverse().map(r => r.lower_90)),
            fill: 'toself', fillcolor: 'rgba(59,130,246,0.08)',
            line: { color: 'transparent' }, showlegend: false, hoverinfo: 'skip',
        });
        // 50% ribbon
        traces.push({
            x: rTimes.concat(rTimes.slice().reverse()),
            y: data.ribbon.map(r => r.upper_50).concat(data.ribbon.slice().reverse().map(r => r.lower_50)),
            fill: 'toself', fillcolor: 'rgba(59,130,246,0.15)',
            line: { color: 'transparent' }, showlegend: false, hoverinfo: 'skip',
        });
    }

    // Forecast center line
    if (data.forecasts && data.forecasts.length > 0) {
        traces.push({
            x: data.forecasts.map(f => f.valid_at),
            y: data.forecasts.map(f => f.temp_f),
            mode: 'lines', line: { color: COLORS.blue, width: 2, dash: 'dot' },
            name: 'HRRR Forecast',
        });
    }

    // Bias-adjusted line
    if (data.bias_adjusted_forecasts && data.bias_adjusted_forecasts.length > 0) {
        traces.push({
            x: data.bias_adjusted_forecasts.map(f => f.valid_at),
            y: data.bias_adjusted_forecasts.map(f => f.temp_f),
            mode: 'lines', line: { color: COLORS.teal, width: 2 },
            name: 'Bias-Adjusted',
        });
    }

    // Observations
    if (data.observations && data.observations.length > 0) {
        traces.push({
            x: data.observations.map(o => o.observed_at),
            y: data.observations.map(o => o.temp_f),
            mode: 'lines+markers', line: { color: COLORS.white, width: 1.5 },
            marker: { size: 4, color: COLORS.white },
            name: 'Observed',
        });
    }

    // 6-hour synoptic max markers
    if (data.six_hr_maxes && data.six_hr_maxes.length > 0) {
        traces.push({
            x: data.six_hr_maxes.map(m => m.observed_at),
            y: data.six_hr_maxes.map(m => m.six_hr_max_f),
            mode: 'markers', marker: { symbol: 'triangle-up', size: 10, color: COLORS.amber },
            name: '6hr Max',
        });
    }

    // Settlement marker
    if (data.observed_high != null) {
        const symbol = data.settlement_source === 'NWS_CLI' ? 'diamond' : 'circle';
        traces.push({
            x: [data.observed_high_at],
            y: [data.observed_high],
            mode: 'markers',
            marker: { symbol: symbol, size: 14, color: COLORS.amber, line: { width: 2, color: '#fff' } },
            name: 'Settlement High',
        });
    }

    // Prior runs (ghosted)
    if (data.prior_runs) {
        data.prior_runs.forEach(function(run) {
            traces.push({
                x: run.forecasts.map(f => f.valid_at),
                y: run.forecasts.map(f => f.temp_f),
                mode: 'lines', line: { color: 'rgba(59,130,246,0.2)', width: 1 },
                showlegend: false, hoverinfo: 'skip',
            });
        });
    }

    // Neighbor obs
    const neighborColors = { KLGA: '#a78bfa', KEWR: '#fb923c' };
    if (data.neighbor_obs) {
        Object.keys(data.neighbor_obs).forEach(function(station) {
            if (!neighborsActive[station]) return;
            const obs = data.neighbor_obs[station];
            if (obs && obs.length > 0) {
                traces.push({
                    x: obs.map(o => o.observed_at),
                    y: obs.map(o => o.temp_f),
                    mode: 'lines', line: { color: neighborColors[station] || COLORS.slate400, width: 1, dash: 'dash' },
                    name: station,
                });
            }
        });
    }

    const layout = Object.assign({}, PLOTLY_LAYOUT, {
        showlegend: true,
        legend: { x: 0, y: 1.12, orientation: 'h', font: { size: 10 } },
        yaxis: Object.assign({}, PLOTLY_LAYOUT.yaxis, { title: '°F' }),
    });

    Plotly.react('temp-chart', traces, layout, PLOTLY_CONFIG);
}

// Placeholder functions for other panels — implemented in later tasks
async function refreshObsFeed() {
    const date = getTargetDate(selectedDate);
    const data = await fetchAPI('/api/observations/' + selectedCity + '?date=' + date);
    if (!data || !data.observations) {
        document.getElementById('obs-feed').innerHTML = '<div class="text-slate-500 text-xs">No observations</div>';
        return;
    }

    const rows = data.observations.slice(0, 20).map(function(obs) {
        const time = toET(obs.observed_at || obs.time);
        const source = obs.source || obs.type || '';
        const temp = obs.temp_f != null ? obs.temp_f + '°F' : '--';
        const sourceColor = source.includes('SPECI') ? 'text-amber-400' :
                           source.includes('CLI') ? 'text-emerald-400' :
                           source.includes('Synoptic') ? 'text-teal-400' : 'text-slate-300';
        return '<div class="flex justify-between text-xs py-1 border-b border-slate-700/50">' +
            '<span class="text-slate-400">' + time + '</span>' +
            '<span class="' + sourceColor + '">' + source + '</span>' +
            '<span class="text-slate-200 font-bold">' + temp + '</span>' +
            '</div>';
    }).join('');

    document.getElementById('obs-feed').innerHTML = rows || '<div class="text-slate-500 text-xs">No observations</div>';
}

async function refreshFcstFeed() {
    const date = getTargetDate(selectedDate);
    const data = await fetchAPI('/api/forecast-points/' + selectedCity + '?date=' + date);
    if (!data || !data.runs) {
        document.getElementById('fcst-feed').innerHTML = '<div class="text-slate-500 text-xs">No forecast runs</div>';
        return;
    }

    const rows = data.runs.map(function(run) {
        const runHr = run.model_run_hour != null ? run.model_run_hour + 'z' : '--';
        const high = run.forecast_high != null ? run.forecast_high + '°F' : '--';
        const delta = run.delta_temp;
        let deltaStr = '';
        let deltaColor = 'text-slate-400';
        if (delta != null && delta !== 0) {
            deltaStr = (delta > 0 ? '+' : '') + delta.toFixed(1);
            deltaColor = delta > 0 ? 'text-emerald-400' : 'text-red-400';
        }
        return '<div class="flex justify-between text-xs py-1 border-b border-slate-700/50">' +
            '<span class="text-slate-400">' + runHr + '</span>' +
            '<span class="text-slate-200 font-bold">' + high + '</span>' +
            '<span class="' + deltaColor + '">' + deltaStr + '</span>' +
            '</div>';
    }).join('');

    document.getElementById('fcst-feed').innerHTML = rows;
}

async function refreshBracketChart() {
    // Implemented in Task 4
    document.getElementById('bracket-chart').innerHTML = '<div class="text-slate-500 text-xs text-center py-10">Bracket spread — wiring in progress</div>';
}

async function refreshPositions() {
    // Implemented in Task 5
    document.getElementById('positions-panel').innerHTML = '<div class="text-slate-500 text-xs text-center py-10">Positions — wiring in progress</div>';
}

function refreshAll() {
    refreshTempChart();
    refreshObsFeed();
    refreshFcstFeed();
    refreshBracketChart();
    refreshPositions();
}

// Initial load
refreshAll();
```

**Step 3: Verify manually**

Run: `cd ~/Projects/alphatemp/alphatemp && python -c "from ui.web_dashboard import app; print('App loaded OK')"`
Expected: prints "App loaded OK" without import errors.

**Step 4: Commit**

```bash
git add templates/operations.html static/js/operations.js
git commit -m "feat: build operations tab with temp curve, obs feed, forecast runs"
```

---

### Task 4: Build the bracket spread API endpoint + chart

**Files:**
- Modify: `ui/web_dashboard.py` (add `/api/brackets/{city}` route)
- Modify: `static/js/operations.js` (implement `refreshBracketChart`)
- Test: `tests/test_web_dashboard.py`

**Step 1: Write the failing test**

```python
def test_brackets_endpoint(client):
    """GET /api/brackets/NYC should return bracket comparison data."""
    resp = client.get("/api/brackets/NYC")
    assert resp.status_code == 200
    data = resp.json()
    assert "brackets" in data
    assert "city" in data
```

**Step 2: Run test to verify it fails**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_web_dashboard.py::test_brackets_endpoint -v`
Expected: FAIL (404, route doesn't exist)

**Step 3: Implement the `/api/brackets/{city}` endpoint**

Add to `ui/web_dashboard.py` after the existing `/api/market/{city}` route:

```python
@app.get("/api/brackets/{city}")
async def get_brackets(city: str, date: str = None):
    """Model bracket probabilities vs Kalshi market prices with edge calculation."""
    city = city.upper()
    if city not in CITIES:
        return {"city": city, "brackets": [], "error": "Unknown city"}

    station_id = CITIES[city]["settlement"]

    # Get model probabilities from ProbabilityEngine
    try:
        forecast = engine.calculate_city(city)
    except Exception:
        forecast = None

    model_probs = {}
    if forecast and forecast.bracket_probs:
        # Map 1°F probs to 2°F Kalshi brackets
        for temp_f, prob in forecast.bracket_probs.items():
            floor = (temp_f // 2) * 2
            cap = floor + 2
            key = (floor, cap)
            model_probs[key] = model_probs.get(key, 0) + prob

    # Get latest Kalshi market prices
    if date:
        target_date = date
    else:
        now_et = datetime.now(ZoneInfo("America/New_York"))
        target_date = now_et.strftime("%Y-%m-%d")

    con = get_connection(DB_PATH)
    try:
        market_rows = con.execute("""
            SELECT floor_strike, cap_strike, yes_bid, yes_ask, volume, open_interest
            FROM market_ticks
            WHERE city = ? AND captured_at::DATE = ?
            ORDER BY captured_at DESC
        """, [city, target_date]).fetchall()
    except Exception:
        market_rows = []
    finally:
        con.close()

    # Deduplicate — keep latest tick per bracket
    market_by_bracket = {}
    for row in market_rows:
        key = (int(row[0]), int(row[1]))
        if key not in market_by_bracket:
            market_by_bracket[key] = {
                "yes_bid": row[2], "yes_ask": row[3],
                "volume": row[4], "open_interest": row[5],
            }

    # Merge model + market into bracket list
    all_brackets = sorted(set(list(model_probs.keys()) + list(market_by_bracket.keys())))
    brackets = []
    for floor, cap in all_brackets:
        m_prob = model_probs.get((floor, cap), 0)
        mkt = market_by_bracket.get((floor, cap), {})
        mid = None
        if mkt.get("yes_bid") is not None and mkt.get("yes_ask") is not None:
            mid = (mkt["yes_bid"] + mkt["yes_ask"]) / 2
        edge = (m_prob - mid) if mid is not None else None
        brackets.append({
            "floor": floor, "cap": cap,
            "model_prob": round(m_prob, 4),
            "market_mid": round(mid, 4) if mid else None,
            "edge": round(edge, 4) if edge is not None else None,
            "yes_bid": mkt.get("yes_bid"),
            "yes_ask": mkt.get("yes_ask"),
            "volume": mkt.get("volume", 0),
            "open_interest": mkt.get("open_interest", 0),
        })

    # Liquidity summary
    total_volume = sum(b.get("volume", 0) or 0 for b in brackets)
    avg_spread = None
    spreads = [b["yes_ask"] - b["yes_bid"] for b in brackets if b["yes_bid"] is not None and b["yes_ask"] is not None]
    if spreads:
        avg_spread = round(sum(spreads) / len(spreads), 4)

    return {
        "city": city,
        "date": target_date,
        "brackets": brackets,
        "liquidity": {"total_volume": total_volume, "avg_spread": avg_spread},
        "model_center": forecast.center if forecast else None,
        "model_std": forecast.std if forecast else None,
    }
```

**Step 4: Run test**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_web_dashboard.py::test_brackets_endpoint -v`
Expected: PASS

**Step 5: Implement `refreshBracketChart()` in `operations.js`**

Replace the placeholder:
```javascript
async function refreshBracketChart() {
    const date = getTargetDate(selectedDate);
    const data = await fetchAPI('/api/brackets/' + selectedCity + '?date=' + date);
    if (!data || !data.brackets || data.brackets.length === 0) {
        document.getElementById('bracket-chart').innerHTML = '<div class="text-slate-500 text-xs text-center py-10">No bracket data</div>';
        document.getElementById('liquidity-bar').innerHTML = '';
        return;
    }

    const brackets = data.brackets.filter(function(b) { return b.model_prob > 0.01 || b.market_mid > 0.01; });
    const labels = brackets.map(function(b) { return b.floor + '-' + b.cap + '°F'; });

    const traces = [
        {
            y: labels, x: brackets.map(function(b) { return b.model_prob * 100; }),
            type: 'bar', orientation: 'h', name: 'Model',
            marker: { color: COLORS.blue }, opacity: 0.8,
        },
        {
            y: labels, x: brackets.map(function(b) { return (b.market_mid || 0) * 100; }),
            type: 'bar', orientation: 'h', name: 'Kalshi',
            marker: { color: COLORS.amber }, opacity: 0.6,
        },
    ];

    const layout = Object.assign({}, PLOTLY_LAYOUT, {
        barmode: 'group',
        showlegend: true,
        legend: { x: 0.7, y: 1.05, orientation: 'h', font: { size: 10 } },
        xaxis: Object.assign({}, PLOTLY_LAYOUT.xaxis, { title: 'Probability %', range: [0, 50] }),
        yaxis: Object.assign({}, PLOTLY_LAYOUT.yaxis, { autorange: 'reversed' }),
        margin: { l: 80, r: 20, t: 20, b: 40 },
    });

    Plotly.react('bracket-chart', traces, layout, PLOTLY_CONFIG);

    // Liquidity bar
    const liq = data.liquidity;
    const volText = liq.total_volume != null ? 'Vol: ' + liq.total_volume : '';
    const spreadText = liq.avg_spread != null ? 'Avg Spread: ' + (liq.avg_spread * 100).toFixed(1) + '¢' : '';
    document.getElementById('liquidity-bar').innerHTML =
        '<div class="flex justify-between text-xs text-slate-400">' +
        '<span>' + volText + '</span><span>' + spreadText + '</span></div>';
}
```

**Step 6: Commit**

```bash
git add ui/web_dashboard.py static/js/operations.js tests/test_web_dashboard.py
git commit -m "feat: add bracket spread API endpoint and chart"
```

---

### Task 5: Build the positions & alerts API endpoint + panel

**Files:**
- Modify: `ui/web_dashboard.py` (add `/api/positions/{city}` and `/api/market-swings/{city}`)
- Modify: `static/js/operations.js` (implement `refreshPositions`)
- Test: `tests/test_web_dashboard.py`

**Step 1: Write failing tests**

```python
def test_positions_endpoint(client):
    """GET /api/positions/NYC should return positions data."""
    resp = client.get("/api/positions/NYC")
    assert resp.status_code == 200
    data = resp.json()
    assert "active" in data
    assert "near_misses" in data
    assert "daily_pnl" in data

def test_market_swings_endpoint(client):
    """GET /api/market-swings/NYC should return market swing data."""
    resp = client.get("/api/market-swings/NYC")
    assert resp.status_code == 200
    data = resp.json()
    assert "swings" in data
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_web_dashboard.py -k "positions_endpoint or market_swings" -v`
Expected: FAIL

**Step 3: Implement `/api/positions/{city}`**

```python
@app.get("/api/positions/{city}")
async def get_positions(city: str, date: str = None):
    """Active paper positions, near-misses, and P&L."""
    city = city.upper()
    if city not in CITIES:
        return {"city": city, "active": [], "near_misses": [], "daily_pnl": 0, "total_pnl": 0}

    if not date:
        now_et = datetime.now(ZoneInfo("America/New_York"))
        date = now_et.strftime("%Y-%m-%d")

    con = get_connection(DB_PATH)
    try:
        # Active positions for this date
        active = con.execute("""
            SELECT id, bracket_floor, bracket_cap, direction, model_prob,
                   market_price, edge, entry_price, entry_time, status
            FROM paper_positions
            WHERE city = ? AND event_date = ? AND status = 'open'
            ORDER BY entry_time DESC
        """, [city, date]).fetchall()

        active_list = [{
            "id": r[0], "bracket": str(r[1]) + "-" + str(r[2]) + "°F",
            "direction": r[3], "model_prob": r[4], "market_price": r[5],
            "edge": r[6], "entry_price": r[7],
            "entry_time": r[8].isoformat() if r[8] else None,
        } for r in active]

        # Daily P&L (settled positions today)
        daily_pnl_row = con.execute("""
            SELECT COALESCE(SUM(net_pnl), 0)
            FROM paper_positions
            WHERE city = ? AND event_date = ? AND status = 'settled'
        """, [city, date]).fetchone()
        daily_pnl = daily_pnl_row[0] if daily_pnl_row else 0

        # Total P&L (all time)
        total_pnl_row = con.execute("""
            SELECT COALESCE(SUM(net_pnl), 0)
            FROM paper_positions
            WHERE city = ? AND status = 'settled'
        """, [city]).fetchone()
        total_pnl = total_pnl_row[0] if total_pnl_row else 0

    except Exception:
        active_list = []
        daily_pnl = 0
        total_pnl = 0
    finally:
        con.close()

    # Near-misses: brackets where model edge is 5-10% (close to threshold but not triggered)
    # Computed from current bracket data
    near_misses = []
    try:
        bracket_data = await get_brackets(city, date)
        for b in bracket_data.get("brackets", []):
            edge = b.get("edge")
            if edge is not None and 0.05 <= edge < 0.10:
                near_misses.append({
                    "bracket": str(b["floor"]) + "-" + str(b["cap"]) + "°F",
                    "edge": round(edge, 4),
                    "threshold": 0.10,
                })
    except Exception:
        pass

    return {
        "city": city, "date": date,
        "active": active_list,
        "near_misses": near_misses,
        "daily_pnl": round(daily_pnl, 2),
        "total_pnl": round(total_pnl, 2),
    }
```

**Step 4: Implement `/api/market-swings/{city}`**

```python
@app.get("/api/market-swings/{city}")
async def get_market_swings(city: str, date: str = None):
    """Detect material market price movements (>10¢ in <2 hours)."""
    city = city.upper()
    if city not in CITIES:
        return {"city": city, "swings": []}

    if not date:
        now_et = datetime.now(ZoneInfo("America/New_York"))
        date = now_et.strftime("%Y-%m-%d")

    con = get_connection(DB_PATH)
    try:
        # Get all ticks for today, grouped by bracket
        ticks = con.execute("""
            SELECT floor_strike, cap_strike, captured_at,
                   (yes_bid + yes_ask) / 2.0 as mid
            FROM market_ticks
            WHERE city = ? AND captured_at::DATE = ?
            ORDER BY floor_strike, cap_strike, captured_at
        """, [city, date]).fetchall()
    except Exception:
        ticks = []
    finally:
        con.close()

    # Detect swings: >10¢ move within any 2-hour window
    swings = []
    bracket_ticks = {}
    for row in ticks:
        key = (int(row[0]), int(row[1]))
        if key not in bracket_ticks:
            bracket_ticks[key] = []
        bracket_ticks[key].append({"time": row[2], "mid": row[3]})

    for (floor, cap), tick_list in bracket_ticks.items():
        for i in range(1, len(tick_list)):
            for j in range(i):
                t_diff = (tick_list[i]["time"] - tick_list[j]["time"]).total_seconds()
                if t_diff > 7200:  # >2 hours apart, skip
                    continue
                price_diff = tick_list[i]["mid"] - tick_list[j]["mid"]
                if abs(price_diff) >= 0.10:  # 10¢ swing
                    swings.append({
                        "bracket": str(floor) + "-" + str(cap) + "°F",
                        "from_price": round(tick_list[j]["mid"], 2),
                        "to_price": round(tick_list[i]["mid"], 2),
                        "change": round(price_diff, 2),
                        "from_time": tick_list[j]["time"].isoformat(),
                        "to_time": tick_list[i]["time"].isoformat(),
                    })
                    break  # One swing per bracket per start point

    # Deduplicate — keep largest swing per bracket
    seen = {}
    for s in swings:
        key = s["bracket"]
        if key not in seen or abs(s["change"]) > abs(seen[key]["change"]):
            seen[key] = s
    swings = sorted(seen.values(), key=lambda s: abs(s["change"]), reverse=True)

    return {"city": city, "date": date, "swings": swings[:10]}
```

**Step 5: Implement `refreshPositions()` in `operations.js`**

Replace placeholder:
```javascript
async function refreshPositions() {
    const date = getTargetDate(selectedDate);
    const [posData, swingData] = await Promise.all([
        fetchAPI('/api/positions/' + selectedCity + '?date=' + date),
        fetchAPI('/api/market-swings/' + selectedCity + '?date=' + date),
    ]);

    let html = '';

    // Active Bets
    html += '<div class="mb-4">';
    html += '<h3 class="text-xs text-slate-400 uppercase tracking-wide mb-2">Active Bets</h3>';
    if (posData && posData.active && posData.active.length > 0) {
        posData.active.forEach(function(pos) {
            const edge = formatEdge(pos.edge);
            html += '<div class="bet-active bg-slate-700/50 rounded p-2 mb-2 border border-slate-600">' +
                '<div class="flex justify-between text-sm">' +
                '<span class="text-slate-100 font-bold">' + pos.bracket + '</span>' +
                '<span class="text-xs">' + pos.direction + ' @ ' + (pos.entry_price * 100).toFixed(0) + '¢</span>' +
                '</div>' +
                '<div class="flex justify-between text-xs mt-1">' +
                '<span class="' + edge.colorClass + '">Edge: ' + edge.text + '</span>' +
                '<span class="text-slate-400">Model: ' + (pos.model_prob * 100).toFixed(0) + '%</span>' +
                '</div></div>';
        });
    } else {
        html += '<div class="text-slate-500 text-xs">No active bets</div>';
    }
    html += '</div>';

    // Near Misses
    html += '<div class="mb-4">';
    html += '<h3 class="text-xs text-slate-400 uppercase tracking-wide mb-2">Near Threshold</h3>';
    if (posData && posData.near_misses && posData.near_misses.length > 0) {
        posData.near_misses.forEach(function(nm) {
            html += '<div class="text-xs py-1 flex justify-between">' +
                '<span class="text-amber-400">' + nm.bracket + '</span>' +
                '<span class="text-slate-400">edge ' + (nm.edge * 100).toFixed(0) + '% (threshold ' + (nm.threshold * 100).toFixed(0) + '%)</span>' +
                '</div>';
        });
    } else {
        html += '<div class="text-slate-500 text-xs">No near misses</div>';
    }
    html += '</div>';

    // Market Swings
    html += '<div class="mb-4">';
    html += '<h3 class="text-xs text-slate-400 uppercase tracking-wide mb-2">Market Swings</h3>';
    if (swingData && swingData.swings && swingData.swings.length > 0) {
        swingData.swings.forEach(function(s) {
            const dir = s.change > 0 ? '↑' : '↓';
            const color = s.change > 0 ? 'text-emerald-400' : 'text-red-400';
            html += '<div class="text-xs py-1 flex justify-between">' +
                '<span class="' + color + '">⚡ ' + s.bracket + ' ' + dir + (Math.abs(s.change) * 100).toFixed(0) + '¢</span>' +
                '<span class="text-slate-400">' + (s.from_price * 100).toFixed(0) + '→' + (s.to_price * 100).toFixed(0) + '¢</span>' +
                '</div>';
        });
    } else {
        html += '<div class="text-slate-500 text-xs">No swings today</div>';
    }
    html += '</div>';

    // P&L
    if (posData) {
        const daily = formatPnL(posData.daily_pnl);
        const total = formatPnL(posData.total_pnl);
        html += '<div class="border-t border-slate-700 pt-2 flex justify-between text-sm">' +
            '<span>Daily: <span class="' + daily.colorClass + ' font-bold">' + daily.text + '</span></span>' +
            '<span>Total: <span class="' + total.colorClass + ' font-bold">' + total.text + '</span></span>' +
            '</div>';
    }

    document.getElementById('positions-panel').innerHTML = html;
}
```

**Step 6: Run tests**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_web_dashboard.py -k "positions or market_swings" -v`
Expected: PASS

**Step 7: Commit**

```bash
git add ui/web_dashboard.py static/js/operations.js tests/test_web_dashboard.py
git commit -m "feat: add positions, near-misses, market swings API + panel"
```

---

## Phase 3: Performance Tab

### Task 6: Build performance API endpoints

**Files:**
- Modify: `ui/web_dashboard.py` (add `/api/performance`, `/api/brier-comparison`, `/api/edge-heatmap`)
- Test: `tests/test_web_dashboard.py`

**Step 1: Write failing tests**

```python
def test_performance_endpoint(client):
    resp = client.get("/api/performance?range=30d")
    assert resp.status_code == 200
    data = resp.json()
    assert "cumulative_pnl" in data
    assert "daily_bars" in data
    assert "win_rate" in data

def test_brier_comparison_endpoint(client):
    resp = client.get("/api/brier-comparison?range=30d")
    assert resp.status_code == 200
    data = resp.json()
    assert "by_hour" in data

def test_edge_heatmap_endpoint(client):
    resp = client.get("/api/edge-heatmap?range=30d")
    assert resp.status_code == 200
    data = resp.json()
    assert "cells" in data
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement all three endpoints**

`/api/performance`:
```python
@app.get("/api/performance")
async def get_performance(range: str = "30d"):
    """Cumulative P&L, daily bars, win rate, breakdown stats."""
    days = {"7d": 7, "30d": 30, "all": 9999}.get(range, 30)
    con = get_connection(DB_PATH)
    try:
        rows = con.execute("""
            SELECT event_date, SUM(net_pnl) as daily_pnl,
                   SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN net_pnl <= 0 THEN 1 ELSE 0 END) as losses,
                   SUM(fees) as daily_fees,
                   SUM(gross_pnl) as daily_gross
            FROM paper_positions
            WHERE status = 'settled'
              AND event_date >= CURRENT_DATE - INTERVAL ? DAY
            GROUP BY event_date
            ORDER BY event_date
        """, [days]).fetchall()

        daily_bars = [{"date": str(r[0]), "pnl": round(r[1], 2), "wins": r[2], "losses": r[3]} for r in rows]

        # Cumulative P&L series
        cumulative = []
        running = 0
        for bar in daily_bars:
            running += bar["pnl"]
            cumulative.append({"date": bar["date"], "pnl": round(running, 2)})

        total_wins = sum(b["wins"] for b in daily_bars)
        total_losses = sum(b["losses"] for b in daily_bars)
        total_trades = total_wins + total_losses
        win_rate = round(total_wins / total_trades, 4) if total_trades > 0 else 0

        # Breakdown by bracket
        bracket_rows = con.execute("""
            SELECT bracket_floor, bracket_cap,
                   SUM(net_pnl) as pnl, COUNT(*) as trades,
                   SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins
            FROM paper_positions
            WHERE status = 'settled'
              AND event_date >= CURRENT_DATE - INTERVAL ? DAY
            GROUP BY bracket_floor, bracket_cap
            ORDER BY bracket_floor
        """, [days]).fetchall()

        by_bracket = [{"bracket": str(r[0]) + "-" + str(r[1]), "pnl": round(r[2], 2),
                       "trades": r[3], "win_rate": round(r[4] / r[3], 4) if r[3] > 0 else 0} for r in bracket_rows]

        # Breakdown by edge bucket
        edge_rows = con.execute("""
            SELECT CASE
                WHEN edge >= 0.15 THEN '>15%'
                WHEN edge >= 0.10 THEN '10-15%'
                WHEN edge >= 0.05 THEN '5-10%'
                ELSE '<5%'
            END as bucket,
            SUM(net_pnl) as pnl, COUNT(*) as trades,
            SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins
            FROM paper_positions
            WHERE status = 'settled'
              AND event_date >= CURRENT_DATE - INTERVAL ? DAY
            GROUP BY bucket
            ORDER BY bucket DESC
        """, [days]).fetchall()

        by_edge = [{"bucket": r[0], "pnl": round(r[1], 2), "trades": r[2],
                    "win_rate": round(r[3] / r[2], 4) if r[2] > 0 else 0} for r in edge_rows]

        total_fees = sum(r[4] or 0 for r in rows)
        total_gross = sum(r[5] or 0 for r in rows)
        total_net = sum(r[1] for r in rows)

    except Exception:
        daily_bars, cumulative, by_bracket, by_edge = [], [], [], []
        win_rate, total_fees, total_gross, total_net, total_trades = 0, 0, 0, 0, 0
    finally:
        con.close()

    return {
        "range": range,
        "cumulative_pnl": cumulative,
        "daily_bars": daily_bars,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "total_gross": round(total_gross, 2),
        "total_fees": round(total_fees, 2),
        "total_net": round(total_net, 2),
        "by_bracket": by_bracket,
        "by_edge": by_edge,
    }
```

`/api/brier-comparison`:
```python
@app.get("/api/brier-comparison")
async def get_brier_comparison(range: str = "30d"):
    """Model Brier vs Market Brier by hour (ET). Requires settlement data to compute."""
    days = {"7d": 7, "30d": 30, "all": 9999}.get(range, 30)
    # This endpoint requires precomputed Brier scores from the backtester.
    # For now, return structure with empty data — will be populated when
    # backtester results are wired in from the other worktree.
    return {
        "range": range,
        "by_hour": [],
        "note": "Brier comparison requires backtester integration (in progress on separate worktree)"
    }
```

`/api/edge-heatmap`:
```python
@app.get("/api/edge-heatmap")
async def get_edge_heatmap(range: str = "30d"):
    """Edge concentration by hour (ET) x bracket."""
    days = {"7d": 7, "30d": 30, "all": 9999}.get(range, 30)

    con = get_connection(DB_PATH)
    try:
        rows = con.execute("""
            SELECT bracket_floor, bracket_cap,
                   EXTRACT(HOUR FROM entry_time AT TIME ZONE 'America/New_York') as hour_et,
                   AVG(edge) as avg_edge,
                   COUNT(*) as trades,
                   SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins
            FROM paper_positions
            WHERE status = 'settled'
              AND event_date >= CURRENT_DATE - INTERVAL ? DAY
            GROUP BY bracket_floor, bracket_cap, hour_et
            ORDER BY hour_et, bracket_floor
        """, [days]).fetchall()

        cells = [{"bracket": str(r[0]) + "-" + str(r[1]), "hour_et": int(r[2]),
                  "avg_edge": round(r[3], 4), "trades": r[4],
                  "win_rate": round(r[5] / r[4], 4) if r[4] > 0 else 0} for r in rows]

    except Exception:
        cells = []
    finally:
        con.close()

    return {"range": range, "cells": cells}
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add ui/web_dashboard.py tests/test_web_dashboard.py
git commit -m "feat: add performance, brier-comparison, edge-heatmap API endpoints"
```

---

### Task 7: Build the performance tab frontend

**Files:**
- Modify: `templates/performance.html`
- Create: `static/js/performance.js`

**Step 1: Build `templates/performance.html`**

```html
{% extends "base.html" %}
{% block title %}Performance{% endblock %}
{% block content %}
<!-- Date Range Selector -->
<div class="flex gap-2 mb-4">
    <button onclick="setRange('7d')" id="range-7d" class="px-3 py-1 rounded text-sm text-slate-400 hover:bg-slate-800">7d</button>
    <button onclick="setRange('30d')" id="range-30d" class="px-3 py-1 rounded text-sm bg-slate-700 text-slate-100">30d</button>
    <button onclick="setRange('all')" id="range-all" class="px-3 py-1 rounded text-sm text-slate-400 hover:bg-slate-800">All</button>
</div>

<!-- Top Row: P&L Charts -->
<div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
    <div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
        <h2 class="text-sm font-bold text-slate-100 mb-2">Cumulative P&L</h2>
        <div id="cumulative-chart" style="height: 250px;"></div>
    </div>
    <div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
        <h2 class="text-sm font-bold text-slate-100 mb-2">Daily P&L</h2>
        <div id="daily-chart" style="height: 250px;"></div>
    </div>
</div>

<!-- Middle Row: Brier + Edge Heatmap -->
<div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
    <div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
        <h2 class="text-sm font-bold text-slate-100 mb-2">Model vs Market Brier Score</h2>
        <div id="brier-chart" style="height: 250px;"></div>
    </div>
    <div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
        <h2 class="text-sm font-bold text-slate-100 mb-2">Edge Heatmap</h2>
        <div id="edge-heatmap" style="height: 250px;"></div>
    </div>
</div>

<!-- Bottom: Stats -->
<div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
    <h2 class="text-sm font-bold text-slate-100 mb-3">Breakdown</h2>
    <div id="breakdown-stats" class="grid grid-cols-1 md:grid-cols-3 gap-4"></div>
</div>
{% endblock %}

{% block scripts %}
<script src="/static/js/performance.js"></script>
{% endblock %}
```

**Step 2: Create `static/js/performance.js`**

```javascript
// performance.js — Performance tab charts + polling

let selectedRange = '30d';

function setRange(r) {
    selectedRange = r;
    ['7d', '30d', 'all'].forEach(function(opt) {
        const el = document.getElementById('range-' + opt);
        el.className = opt === r
            ? 'px-3 py-1 rounded text-sm bg-slate-700 text-slate-100'
            : 'px-3 py-1 rounded text-sm text-slate-400 hover:bg-slate-800';
    });
    refreshPerformance();
}

async function refreshPerformance() {
    const [perfData, brierData, heatData] = await Promise.all([
        fetchAPI('/api/performance?range=' + selectedRange),
        fetchAPI('/api/brier-comparison?range=' + selectedRange),
        fetchAPI('/api/edge-heatmap?range=' + selectedRange),
    ]);

    renderCumulativePnL(perfData);
    renderDailyBars(perfData);
    renderBrierComparison(brierData);
    renderEdgeHeatmap(heatData);
    renderBreakdown(perfData);
}

function renderCumulativePnL(data) {
    if (!data || !data.cumulative_pnl || data.cumulative_pnl.length === 0) {
        document.getElementById('cumulative-chart').innerHTML = '<div class="text-slate-500 text-xs text-center py-10">No P&L data yet</div>';
        return;
    }
    const trace = {
        x: data.cumulative_pnl.map(function(p) { return p.date; }),
        y: data.cumulative_pnl.map(function(p) { return p.pnl; }),
        type: 'scatter', mode: 'lines',
        line: { color: data.total_net >= 0 ? COLORS.green : COLORS.red, width: 2 },
        fill: 'tozeroy',
        fillcolor: data.total_net >= 0 ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)',
    };
    const layout = Object.assign({}, PLOTLY_LAYOUT, {
        showlegend: false,
        yaxis: Object.assign({}, PLOTLY_LAYOUT.yaxis, { title: 'Net P&L ($)' }),
    });
    Plotly.react('cumulative-chart', [trace], layout, PLOTLY_CONFIG);
}

function renderDailyBars(data) {
    if (!data || !data.daily_bars || data.daily_bars.length === 0) {
        document.getElementById('daily-chart').innerHTML = '<div class="text-slate-500 text-xs text-center py-10">No daily data yet</div>';
        return;
    }
    const trace = {
        x: data.daily_bars.map(function(d) { return d.date; }),
        y: data.daily_bars.map(function(d) { return d.pnl; }),
        type: 'bar',
        marker: {
            color: data.daily_bars.map(function(d) { return d.pnl >= 0 ? COLORS.green : COLORS.red; }),
        },
    };
    const layout = Object.assign({}, PLOTLY_LAYOUT, {
        showlegend: false,
        yaxis: Object.assign({}, PLOTLY_LAYOUT.yaxis, { title: 'Daily P&L ($)' }),
    });
    Plotly.react('daily-chart', [trace], layout, PLOTLY_CONFIG);
}

function renderBrierComparison(data) {
    if (!data || !data.by_hour || data.by_hour.length === 0) {
        const note = (data && data.note) ? data.note : 'No Brier data yet';
        document.getElementById('brier-chart').innerHTML = '<div class="text-slate-500 text-xs text-center py-10">' + note + '</div>';
        return;
    }
    const traces = [
        {
            x: data.by_hour.map(function(h) { return h.hour_et; }),
            y: data.by_hour.map(function(h) { return h.model_brier; }),
            name: 'Model', mode: 'lines+markers',
            line: { color: COLORS.blue, width: 2 },
        },
        {
            x: data.by_hour.map(function(h) { return h.hour_et; }),
            y: data.by_hour.map(function(h) { return h.market_brier; }),
            name: 'Market', mode: 'lines+markers',
            line: { color: COLORS.amber, width: 2 },
        },
    ];
    const layout = Object.assign({}, PLOTLY_LAYOUT, {
        xaxis: Object.assign({}, PLOTLY_LAYOUT.xaxis, { title: 'Hour (ET)' }),
        yaxis: Object.assign({}, PLOTLY_LAYOUT.yaxis, { title: 'Brier Score', autorange: 'reversed' }),
        legend: { x: 0, y: 1.1, orientation: 'h' },
    });
    Plotly.react('brier-chart', traces, layout, PLOTLY_CONFIG);
}

function renderEdgeHeatmap(data) {
    if (!data || !data.cells || data.cells.length === 0) {
        document.getElementById('edge-heatmap').innerHTML = '<div class="text-slate-500 text-xs text-center py-10">No edge data yet</div>';
        return;
    }
    // Build heatmap matrix
    var hours = [...new Set(data.cells.map(function(c) { return c.hour_et; }))].sort(function(a,b){return a-b;});
    var brackets = [...new Set(data.cells.map(function(c) { return c.bracket; }))].sort();
    var z = [];
    brackets.forEach(function(b) {
        var row = [];
        hours.forEach(function(h) {
            var cell = data.cells.find(function(c) { return c.bracket === b && c.hour_et === h; });
            row.push(cell ? cell.avg_edge * 100 : 0);
        });
        z.push(row);
    });
    var trace = {
        x: hours.map(function(h) { return h + ' ET'; }),
        y: brackets, z: z,
        type: 'heatmap',
        colorscale: [[0, '#1e293b'], [0.5, '#f59e0b'], [1, '#22c55e']],
        showscale: true,
        colorbar: { title: 'Edge %', titleside: 'right', ticksuffix: '%' },
    };
    var layout = Object.assign({}, PLOTLY_LAYOUT, { margin: { l: 80, r: 60, t: 20, b: 40 } });
    Plotly.react('edge-heatmap', [trace], layout, PLOTLY_CONFIG);
}

function renderBreakdown(data) {
    if (!data) { document.getElementById('breakdown-stats').innerHTML = ''; return; }

    var html = '';

    // Summary stats
    var pnl = formatPnL(data.total_net);
    html += '<div>' +
        '<h3 class="text-xs text-slate-400 uppercase tracking-wide mb-2">Summary</h3>' +
        '<div class="text-xs space-y-1">' +
        '<div class="flex justify-between"><span>Trades:</span><span>' + data.total_trades + '</span></div>' +
        '<div class="flex justify-between"><span>Win Rate:</span><span>' + (data.win_rate * 100).toFixed(1) + '%</span></div>' +
        '<div class="flex justify-between"><span>Gross:</span><span>' + formatPnL(data.total_gross).text + '</span></div>' +
        '<div class="flex justify-between"><span>Fees:</span><span class="text-red-400">-$' + Math.abs(data.total_fees).toFixed(2) + '</span></div>' +
        '<div class="flex justify-between font-bold"><span>Net:</span><span class="' + pnl.colorClass + '">' + pnl.text + '</span></div>' +
        '</div></div>';

    // By bracket
    html += '<div>' +
        '<h3 class="text-xs text-slate-400 uppercase tracking-wide mb-2">By Bracket</h3>' +
        '<div class="text-xs space-y-1">';
    (data.by_bracket || []).forEach(function(b) {
        var bp = formatPnL(b.pnl);
        html += '<div class="flex justify-between">' +
            '<span>' + b.bracket + '°F</span>' +
            '<span class="' + bp.colorClass + '">' + bp.text + ' (' + b.trades + ')</span></div>';
    });
    html += '</div></div>';

    // By edge bucket
    html += '<div>' +
        '<h3 class="text-xs text-slate-400 uppercase tracking-wide mb-2">By Edge</h3>' +
        '<div class="text-xs space-y-1">';
    (data.by_edge || []).forEach(function(e) {
        var ep = formatPnL(e.pnl);
        html += '<div class="flex justify-between">' +
            '<span>' + e.bucket + '</span>' +
            '<span class="' + ep.colorClass + '">' + ep.text + ' (' + (e.win_rate * 100).toFixed(0) + '% win)</span></div>';
    });
    html += '</div></div>';

    document.getElementById('breakdown-stats').innerHTML = html;
}

function refreshAll() { refreshPerformance(); }
refreshPerformance();
```

**Step 3: Commit**

```bash
git add templates/performance.html static/js/performance.js
git commit -m "feat: build performance tab with P&L charts, edge heatmap, breakdown stats"
```

---

## Phase 4: Review Tab

### Task 8: Build review/incident API endpoints

**Files:**
- Modify: `ui/web_dashboard.py` (add `/api/review/incidents`, `/api/review/patterns`)
- Test: `tests/test_web_dashboard.py`

**Step 1: Write failing tests**

```python
def test_review_incidents_endpoint(client):
    resp = client.get("/api/review/incidents?range=30d")
    assert resp.status_code == 200
    data = resp.json()
    assert "incidents" in data

def test_review_patterns_endpoint(client):
    resp = client.get("/api/review/patterns?range=30d")
    assert resp.status_code == 200
    data = resp.json()
    assert "patterns" in data
```

**Step 2: Run — expect FAIL**

**Step 3: Implement endpoints**

`/api/review/incidents`:
```python
@app.get("/api/review/incidents")
async def get_review_incidents(range: str = "30d", filter: str = "all"):
    """Generate incident cards from model vs settlement vs market comparison."""
    days = {"7d": 7, "30d": 30, "all": 9999}.get(range, 30)

    con = get_connection(DB_PATH)
    incidents = []
    try:
        # Get all settled positions with their outcomes
        positions = con.execute("""
            SELECT event_date, bracket_floor, bracket_cap, direction,
                   model_prob, market_price, edge, entry_price,
                   net_pnl, settled_yes
            FROM paper_positions
            WHERE status = 'settled'
              AND event_date >= CURRENT_DATE - INTERVAL ? DAY
            ORDER BY event_date DESC
        """, [days]).fetchall()

        # Get settlement data for context
        settlements = con.execute("""
            SELECT obs_date, max_temp_f
            FROM nws_daily
            WHERE station_id = 'KNYC'
              AND obs_date >= CURRENT_DATE - INTERVAL ? DAY
            ORDER BY obs_date DESC
        """, [days]).fetchall()
        settlement_map = {str(r[0]): r[1] for r in settlements}

        # Lost bets
        for pos in positions:
            if pos[8] is not None and pos[8] < 0:  # net_pnl < 0
                settlement_temp = settlement_map.get(str(pos[0]))
                incidents.append({
                    "date": str(pos[0]),
                    "type": "lost_bet",
                    "severity": min(abs(pos[8]) / 5.0, 1.0),  # Normalize to 0-1
                    "bracket": str(pos[1]) + "-" + str(pos[2]) + "°F",
                    "direction": pos[3],
                    "model_prob": pos[4],
                    "market_price": pos[5],
                    "edge": pos[6],
                    "net_pnl": round(pos[8], 2),
                    "settlement_temp": settlement_temp,
                    "category": _categorize_incident(pos, settlement_temp),
                    "narrative": _narrate_incident("lost_bet", pos, settlement_temp),
                })

        # Filter
        if filter == "worst":
            incidents = [i for i in incidents if i["severity"] >= 0.5]
        elif filter == "lost":
            incidents = [i for i in incidents if i["type"] == "lost_bet"]
        elif filter == "missed":
            incidents = [i for i in incidents if i["type"] == "missed_edge"]

        incidents.sort(key=lambda x: x["severity"], reverse=True)

    except Exception as e:
        logger.error(f"Error generating incidents: {e}")
    finally:
        con.close()

    return {"range": range, "filter": filter, "incidents": incidents[:50]}


def _categorize_incident(pos, settlement_temp):
    """Auto-tag incident category."""
    if settlement_temp is None:
        return "unknown"
    bracket_floor, bracket_cap = pos[1], pos[2]
    if settlement_temp < bracket_floor - 4 or settlement_temp > bracket_cap + 4:
        return "tail_bracket_underweight"
    if abs(pos[4] - pos[5]) < 0.05:
        return "threshold_too_conservative"
    return "model_miss"


def _narrate_incident(incident_type, pos, settlement_temp):
    """Generate a short narrative for the incident."""
    bracket = str(pos[1]) + "-" + str(pos[2]) + "°F"
    if incident_type == "lost_bet":
        settled = "YES" if pos[9] else "NO"
        return (f"Bet {pos[3]} on {bracket} at {pos[7]*100:.0f}¢. "
                f"Bracket settled {settled}. "
                f"Settlement temp: {settlement_temp}°F." if settlement_temp else
                f"Bet {pos[3]} on {bracket} at {pos[7]*100:.0f}¢. Lost.")
    return ""
```

`/api/review/patterns`:
```python
@app.get("/api/review/patterns")
async def get_review_patterns(range: str = "30d"):
    """Aggregate failure patterns from incidents."""
    incident_data = await get_review_incidents(range=range, filter="all")
    incidents = incident_data.get("incidents", [])

    # Group by category
    category_stats = {}
    for inc in incidents:
        cat = inc.get("category", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {"count": 0, "total_pnl": 0}
        category_stats[cat]["count"] += 1
        category_stats[cat]["total_pnl"] += inc.get("net_pnl", 0)

    # Known remedies
    remedies = {
        "slow_drift_response": "Consider dynamic std that widens when obs drift > 2°F",
        "tail_bracket_underweight": "Review tail bracket calibration (model underweights >2σ)",
        "threshold_too_conservative": "Backtest threshold at lower values (e.g., 8% vs 10%)",
        "stale_pricing": "Add stale-price detection to trigger model re-evaluation",
        "model_miss": "Review model accuracy for these conditions in backtester",
    }

    patterns = []
    for cat, stats in sorted(category_stats.items(), key=lambda x: x[1]["total_pnl"]):
        patterns.append({
            "category": cat,
            "count": stats["count"],
            "total_pnl": round(stats["total_pnl"], 2),
            "suggested_action": remedies.get(cat, "Investigate further"),
        })

    return {"range": range, "patterns": patterns}
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add ui/web_dashboard.py tests/test_web_dashboard.py
git commit -m "feat: add review incident and pattern API endpoints"
```

---

### Task 9: Build the review tab frontend

**Files:**
- Modify: `templates/review.html`
- Create: `static/js/review.js`

**Step 1: Build `templates/review.html`**

```html
{% extends "base.html" %}
{% block title %}Review{% endblock %}
{% block content %}
<!-- Filters -->
<div class="flex flex-wrap gap-2 mb-4">
    <div class="flex gap-2">
        <button onclick="setReviewRange('7d')" id="rr-7d" class="px-3 py-1 rounded text-sm text-slate-400 hover:bg-slate-800">7d</button>
        <button onclick="setReviewRange('30d')" id="rr-30d" class="px-3 py-1 rounded text-sm bg-slate-700 text-slate-100">30d</button>
        <button onclick="setReviewRange('all')" id="rr-all" class="px-3 py-1 rounded text-sm text-slate-400 hover:bg-slate-800">All</button>
    </div>
    <div class="border-l border-slate-600 mx-2"></div>
    <div class="flex gap-2">
        <button onclick="setFilter('all')" id="filter-all" class="px-3 py-1 rounded text-sm bg-slate-700 text-slate-100">All</button>
        <button onclick="setFilter('worst')" id="filter-worst" class="px-3 py-1 rounded text-sm text-slate-400 hover:bg-slate-800">Worst Misses</button>
        <button onclick="setFilter('lost')" id="filter-lost" class="px-3 py-1 rounded text-sm text-slate-400 hover:bg-slate-800">Lost Bets</button>
        <button onclick="setFilter('missed')" id="filter-missed" class="px-3 py-1 rounded text-sm text-slate-400 hover:bg-slate-800">Missed Edge</button>
    </div>
</div>

<!-- Incident Cards -->
<div id="incident-list" class="space-y-3 mb-6"></div>

<!-- Pattern Summary -->
<div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
    <h2 class="text-sm font-bold text-slate-100 mb-3">Pattern Summary</h2>
    <div id="pattern-summary"></div>
</div>
{% endblock %}

{% block scripts %}
<script src="/static/js/review.js"></script>
{% endblock %}
```

**Step 2: Create `static/js/review.js`**

```javascript
// review.js — Review tab incident rendering

let reviewRange = '30d';
let reviewFilter = 'all';

function setReviewRange(r) {
    reviewRange = r;
    ['7d', '30d', 'all'].forEach(function(opt) {
        var el = document.getElementById('rr-' + opt);
        el.className = opt === r
            ? 'px-3 py-1 rounded text-sm bg-slate-700 text-slate-100'
            : 'px-3 py-1 rounded text-sm text-slate-400 hover:bg-slate-800';
    });
    refreshReview();
}

function setFilter(f) {
    reviewFilter = f;
    ['all', 'worst', 'lost', 'missed'].forEach(function(opt) {
        var el = document.getElementById('filter-' + opt);
        el.className = opt === f
            ? 'px-3 py-1 rounded text-sm bg-slate-700 text-slate-100'
            : 'px-3 py-1 rounded text-sm text-slate-400 hover:bg-slate-800';
    });
    refreshReview();
}

async function refreshReview() {
    var [incData, patData] = await Promise.all([
        fetchAPI('/api/review/incidents?range=' + reviewRange + '&filter=' + reviewFilter),
        fetchAPI('/api/review/patterns?range=' + reviewRange),
    ]);

    renderIncidents(incData);
    renderPatterns(patData);
}

function renderIncidents(data) {
    var el = document.getElementById('incident-list');
    if (!data || !data.incidents || data.incidents.length === 0) {
        el.innerHTML = '<div class="text-slate-500 text-center py-10">No incidents in this range. Either the model is perfect or there\'s no trading data yet.</div>';
        return;
    }

    var html = '';
    data.incidents.forEach(function(inc) {
        var pnl = formatPnL(inc.net_pnl);
        var severityPct = (inc.severity * 100).toFixed(0);
        var severityColor = inc.severity >= 0.7 ? 'bg-red-500' :
                           inc.severity >= 0.4 ? 'bg-amber-500' : 'bg-emerald-500';
        var categoryLabel = inc.category.replace(/_/g, ' ');

        html += '<div class="bg-slate-800 rounded-lg p-4 border border-slate-700">' +
            '<div class="flex items-center justify-between mb-2">' +
            '<div class="flex items-center gap-3">' +
            '<span class="text-sm text-slate-100 font-bold">' + inc.date + '</span>' +
            '<span class="px-2 py-0.5 rounded text-xs bg-slate-700 text-slate-300">' + categoryLabel + '</span>' +
            '</div>' +
            '<div class="flex items-center gap-2">' +
            '<span class="' + pnl.colorClass + ' text-sm font-bold">' + pnl.text + '</span>' +
            '<div class="w-16 h-1 bg-slate-700 rounded overflow-hidden">' +
            '<div class="h-full ' + severityColor + ' rounded" style="width:' + severityPct + '%"></div>' +
            '</div></div></div>' +
            '<div class="text-xs space-y-1">' +
            '<div class="flex gap-4 text-slate-400">' +
            '<span>Bracket: <span class="text-slate-200">' + inc.bracket + '</span></span>' +
            '<span>Direction: <span class="text-slate-200">' + inc.direction + '</span></span>' +
            (inc.settlement_temp ? '<span>Settled: <span class="text-slate-200">' + inc.settlement_temp + '°F</span></span>' : '') +
            '</div>' +
            '<div class="flex gap-4 text-slate-400">' +
            '<span>Model: <span class="text-blue-400">' + (inc.model_prob * 100).toFixed(0) + '%</span></span>' +
            '<span>Market: <span class="text-amber-400">' + (inc.market_price * 100).toFixed(0) + '¢</span></span>' +
            '<span>Edge: ' + formatEdge(inc.edge).text + '</span>' +
            '</div>' +
            (inc.narrative ? '<div class="text-slate-300 mt-1">' + inc.narrative + '</div>' : '') +
            '</div></div>';
    });

    el.innerHTML = html;
}

function renderPatterns(data) {
    var el = document.getElementById('pattern-summary');
    if (!data || !data.patterns || data.patterns.length === 0) {
        el.innerHTML = '<div class="text-slate-500 text-xs">No patterns detected yet</div>';
        return;
    }

    var html = '<div class="space-y-3">';
    data.patterns.forEach(function(p, i) {
        var pnl = formatPnL(p.total_pnl);
        var label = p.category.replace(/_/g, ' ');
        html += '<div class="flex items-start gap-4 text-sm">' +
            '<span class="text-slate-500 w-4">' + (i + 1) + '.</span>' +
            '<div class="flex-1">' +
            '<div class="flex justify-between">' +
            '<span class="text-slate-200 capitalize">' + label + '</span>' +
            '<span class="' + pnl.colorClass + '">' + pnl.text + ' (' + p.count + ' incidents)</span>' +
            '</div>' +
            '<div class="text-xs text-slate-400 mt-1">→ ' + p.suggested_action + '</div>' +
            '</div></div>';
    });
    html += '</div>';

    el.innerHTML = html;
}

function refreshAll() { refreshReview(); }
refreshReview();
```

**Step 3: Commit**

```bash
git add templates/review.html static/js/review.js
git commit -m "feat: build review tab with incident cards, filter controls, pattern summary"
```

---

## Phase 5: Mobile View

### Task 10: Build the mobile template

**Files:**
- Modify: `templates/mobile.html`
- Create: `static/js/mobile.js`

**Step 1: Build `templates/mobile.html`**

```html
{% extends "base.html" %}
{% block title %}AlphaTemp Mobile{% endblock %}
{% block content %}
<!-- Mobile tab toggle -->
<div class="flex gap-1 mb-4 md:hidden">
    <button onclick="setMobileTab('dashboard')" id="mob-dashboard" class="flex-1 px-3 py-2 rounded text-sm bg-slate-700 text-slate-100">Dashboard</button>
    <button onclick="setMobileTab('trade')" id="mob-trade" class="flex-1 px-3 py-2 rounded text-sm text-slate-400">Trade</button>
</div>

<!-- Dashboard view -->
<div id="mobile-dashboard">
    <!-- Current State Card -->
    <div class="bg-slate-800 rounded-lg p-4 border border-slate-700 mb-3">
        <div class="grid grid-cols-2 gap-3 text-center">
            <div>
                <div class="text-xs text-slate-400">Latest Obs</div>
                <div id="mob-latest-obs" class="text-xl font-bold text-slate-100">--</div>
            </div>
            <div>
                <div class="text-xs text-slate-400">Model High</div>
                <div id="mob-model-high" class="text-xl font-bold text-blue-400">--</div>
            </div>
            <div>
                <div class="text-xs text-slate-400">Settlement</div>
                <div id="mob-settlement" class="text-xl font-bold text-amber-400">--</div>
            </div>
            <div>
                <div class="text-xs text-slate-400">Daily P&L</div>
                <div id="mob-daily-pnl" class="text-xl font-bold">--</div>
            </div>
        </div>
    </div>

    <!-- Alerts -->
    <div class="bg-slate-800 rounded-lg p-4 border border-slate-700 mb-3">
        <h2 class="text-xs text-slate-400 uppercase tracking-wide mb-2">Alerts</h2>
        <div id="mob-alerts"></div>
    </div>

    <!-- Recent Obs -->
    <div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
        <h2 class="text-xs text-slate-400 uppercase tracking-wide mb-2">Recent Observations</h2>
        <div id="mob-obs-feed"></div>
    </div>
</div>

<!-- Trade view -->
<div id="mobile-trade" class="hidden">
    <!-- Suggested Bets -->
    <div class="bg-slate-800 rounded-lg p-4 border border-slate-700 mb-3">
        <h2 class="text-xs text-slate-400 uppercase tracking-wide mb-2">Suggested Bets</h2>
        <div id="mob-suggested-bets"></div>
    </div>

    <!-- Near Threshold -->
    <div class="bg-slate-800 rounded-lg p-4 border border-slate-700 mb-3">
        <h2 class="text-xs text-slate-400 uppercase tracking-wide mb-2">Near Threshold</h2>
        <div id="mob-near-threshold"></div>
    </div>

    <!-- Active Positions -->
    <div class="bg-slate-800 rounded-lg p-4 border border-slate-700 mb-3">
        <h2 class="text-xs text-slate-400 uppercase tracking-wide mb-2">Active Positions</h2>
        <div id="mob-active-positions"></div>
    </div>

    <!-- P&L -->
    <div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
        <div class="flex justify-between text-sm">
            <span>Daily: <span id="mob-pnl-daily" class="font-bold">--</span></span>
            <span>All-time: <span id="mob-pnl-total" class="font-bold">--</span></span>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script src="/static/js/mobile.js"></script>
{% endblock %}
```

**Step 2: Create `static/js/mobile.js`**

```javascript
// mobile.js — Mobile view logic

function setMobileTab(tab) {
    document.getElementById('mobile-dashboard').className = tab === 'dashboard' ? '' : 'hidden';
    document.getElementById('mobile-trade').className = tab === 'trade' ? '' : 'hidden';
    document.getElementById('mob-dashboard').className = tab === 'dashboard'
        ? 'flex-1 px-3 py-2 rounded text-sm bg-slate-700 text-slate-100'
        : 'flex-1 px-3 py-2 rounded text-sm text-slate-400';
    document.getElementById('mob-trade').className = tab === 'trade'
        ? 'flex-1 px-3 py-2 rounded text-sm bg-slate-700 text-slate-100'
        : 'flex-1 px-3 py-2 rounded text-sm text-slate-400';
}

async function refreshMobile() {
    var date = getTargetDate(selectedDate);

    var [obsData, curveData, posData, swingData, bracketData] = await Promise.all([
        fetchAPI('/api/observations/' + selectedCity + '?date=' + date),
        fetchAPI('/api/forecast-curve/' + selectedCity + '?date=' + date),
        fetchAPI('/api/positions/' + selectedCity + '?date=' + date),
        fetchAPI('/api/market-swings/' + selectedCity + '?date=' + date),
        fetchAPI('/api/brackets/' + selectedCity + '?date=' + date),
    ]);

    // Current state card
    if (obsData && obsData.observations && obsData.observations.length > 0) {
        var latest = obsData.observations[0];
        document.getElementById('mob-latest-obs').textContent =
            (latest.temp_f != null ? latest.temp_f + '°F' : '--');
    }
    if (obsData) {
        document.getElementById('mob-latest-obs').textContent =
            obsData.running_high != null ? obsData.running_high + '°F' : '--';
    }
    if (curveData) {
        document.getElementById('mob-model-high').textContent =
            curveData.forecast_high != null ? curveData.forecast_high + '°F' : '--';
        document.getElementById('mob-settlement').textContent =
            curveData.observed_high != null ? curveData.observed_high + '°F' : '--';
    }
    if (posData) {
        var dp = formatPnL(posData.daily_pnl);
        document.getElementById('mob-daily-pnl').textContent = dp.text;
        document.getElementById('mob-daily-pnl').className = 'text-xl font-bold ' + dp.colorClass;
    }

    // Alerts (market swings)
    var alertHtml = '';
    if (swingData && swingData.swings && swingData.swings.length > 0) {
        swingData.swings.slice(0, 5).forEach(function(s) {
            var dir = s.change > 0 ? '↑' : '↓';
            var color = s.change > 0 ? 'text-emerald-400' : 'text-red-400';
            alertHtml += '<div class="text-xs py-1 ' + color + '">⚡ ' + s.bracket + ' ' +
                dir + (Math.abs(s.change) * 100).toFixed(0) + '¢</div>';
        });
    } else {
        alertHtml = '<div class="text-slate-500 text-xs">No alerts</div>';
    }
    document.getElementById('mob-alerts').innerHTML = alertHtml;

    // Recent obs
    var obsHtml = '';
    if (obsData && obsData.observations) {
        obsData.observations.slice(0, 5).forEach(function(obs) {
            var time = toET(obs.observed_at || obs.time);
            var src = obs.source || obs.type || '';
            obsHtml += '<div class="flex justify-between text-xs py-1 border-b border-slate-700/50">' +
                '<span class="text-slate-400">' + time + '</span>' +
                '<span class="text-slate-400">' + src + '</span>' +
                '<span class="text-slate-200">' + (obs.temp_f || '--') + '°F</span></div>';
        });
    }
    document.getElementById('mob-obs-feed').innerHTML = obsHtml || '<div class="text-slate-500 text-xs">No obs</div>';

    // Suggested bets (brackets with edge > threshold)
    var betHtml = '';
    if (bracketData && bracketData.brackets) {
        var suggestions = bracketData.brackets.filter(function(b) { return b.edge && b.edge >= 0.10; });
        if (suggestions.length > 0) {
            suggestions.sort(function(a, b) { return b.edge - a.edge; });
            suggestions.forEach(function(b) {
                var netEdge = b.edge - 0.11; // Fee hurdle
                var confidence = b.edge >= 0.15 ? 'HIGH' : 'MED';
                var confColor = confidence === 'HIGH' ? 'text-emerald-400' : 'text-amber-400';
                betHtml += '<div class="bg-slate-700/50 rounded p-3 mb-2 border border-slate-600">' +
                    '<div class="flex justify-between text-sm">' +
                    '<span class="text-slate-100 font-bold">✦ ' + b.floor + '-' + b.cap + '°F  BUY YES</span>' +
                    '<span class="' + confColor + ' text-xs">' + confidence + '</span>' +
                    '</div>' +
                    '<div class="grid grid-cols-2 gap-2 text-xs mt-2 text-slate-400">' +
                    '<div>Model: <span class="text-blue-400">' + (b.model_prob * 100).toFixed(0) + '%</span></div>' +
                    '<div>Kalshi: <span class="text-amber-400">' + (b.market_mid ? (b.market_mid * 100).toFixed(0) + '%' : '--') + '</span></div>' +
                    '<div>Edge: <span class="text-emerald-400">+' + (b.edge * 100).toFixed(0) + '%</span></div>' +
                    '<div>Net: <span class="' + (netEdge > 0 ? 'text-emerald-400' : 'text-red-400') + '">' + (netEdge > 0 ? '+' : '') + (netEdge * 100).toFixed(0) + '%</span></div>' +
                    '<div>Ask: ' + (b.yes_ask ? (b.yes_ask * 100).toFixed(0) + '¢' : '--') + '</div>' +
                    '<div>Vol: ' + (b.volume || 0) + '</div>' +
                    '</div>' +
                    (b.volume < 100 ? '<div class="text-amber-400 text-xs mt-1">⚠ Thin liquidity</div>' : '') +
                    '</div>';
            });
        } else {
            betHtml = '<div class="text-slate-500 text-xs">No bets meeting threshold</div>';
        }
    }
    document.getElementById('mob-suggested-bets').innerHTML = betHtml;

    // Near threshold
    var nearHtml = '';
    if (posData && posData.near_misses && posData.near_misses.length > 0) {
        posData.near_misses.forEach(function(nm) {
            nearHtml += '<div class="text-xs py-1 flex justify-between text-amber-400">' +
                '<span>' + nm.bracket + '</span>' +
                '<span>edge ' + (nm.edge * 100).toFixed(0) + '% (threshold ' + (nm.threshold * 100).toFixed(0) + '%)</span></div>';
        });
    } else {
        nearHtml = '<div class="text-slate-500 text-xs">None near threshold</div>';
    }
    document.getElementById('mob-near-threshold').innerHTML = nearHtml;

    // Active positions
    var posHtml = '';
    if (posData && posData.active && posData.active.length > 0) {
        posData.active.forEach(function(p) {
            posHtml += '<div class="text-xs py-1 flex justify-between">' +
                '<span class="text-slate-200">' + p.bracket + ' ' + p.direction + '</span>' +
                '<span class="text-slate-400">@ ' + (p.entry_price * 100).toFixed(0) + '¢</span></div>';
        });
    } else {
        posHtml = '<div class="text-slate-500 text-xs">No active positions</div>';
    }
    document.getElementById('mob-active-positions').innerHTML = posHtml;

    // P&L footer
    if (posData) {
        var daily = formatPnL(posData.daily_pnl);
        var total = formatPnL(posData.total_pnl);
        document.getElementById('mob-pnl-daily').textContent = daily.text;
        document.getElementById('mob-pnl-daily').className = 'font-bold ' + daily.colorClass;
        document.getElementById('mob-pnl-total').textContent = total.text;
        document.getElementById('mob-pnl-total').className = 'font-bold ' + total.colorClass;
    }
}

function refreshAll() { refreshMobile(); }
refreshMobile();
```

**Step 3: Commit**

```bash
git add templates/mobile.html static/js/mobile.js
git commit -m "feat: build mobile view with dashboard + trade tabs"
```

---

## Phase 6: Cleanup & Integration

### Task 11: Update existing dashboard tests

**Files:**
- Modify: `tests/test_web_dashboard.py` (update any tests that rely on old `GET /` returning HTML directly)

**Step 1: Check which tests break with the redirect**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_web_dashboard.py -v`

**Step 2: Fix any broken tests**

The old test for `GET /` probably expects 200 with HTML. Update it to expect a redirect:
```python
def test_root_redirect(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (301, 302, 307, 308)
```

**Step 3: Run full test suite**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/ -v`

**Step 4: Commit**

```bash
git add tests/
git commit -m "fix: update dashboard tests for new multi-tab routing"
```

---

### Task 12: Verify end-to-end and clean up old template

**Step 1: Start the dashboard locally**

Run: `cd ~/Projects/alphatemp/alphatemp && python main.py --dashboard`

**Step 2: Manually verify all tabs load**

Open in browser:
- `http://localhost:8050/operations` — temperature curve, obs feed, forecast runs, bracket chart, positions panel
- `http://localhost:8050/performance` — P&L charts (may show "no data" if paper_positions is empty)
- `http://localhost:8050/review` — incident list (may show "no incidents" if no trades)
- `http://localhost:8050/mobile` — mobile dashboard + trade tabs

**Step 3: Archive old template**

```bash
mv ~/Projects/alphatemp/alphatemp/templates/index.html ~/Projects/alphatemp/alphatemp/templates/index.html.bak
```

**Step 4: Commit**

```bash
git add templates/
git commit -m "chore: archive old single-page dashboard template"
```

---

## Summary

| Task | Phase | Description | New Files | Modified Files |
|------|-------|-------------|-----------|----------------|
| 1 | Foundation | paper_positions table | — | core/db.py, tests/test_db.py |
| 2 | Foundation | Base template + static files + routing | base.html, shared.js, dashboard.css, 4 stub templates | web_dashboard.py |
| 3 | Operations | Temperature curve + obs/fcst feeds | operations.js | operations.html |
| 4 | Operations | Bracket spread API + chart | — | web_dashboard.py, operations.js |
| 5 | Operations | Positions + market swings API + panel | — | web_dashboard.py, operations.js |
| 6 | Performance | Performance API endpoints | — | web_dashboard.py |
| 7 | Performance | Performance tab frontend | performance.js | performance.html |
| 8 | Review | Review API endpoints | — | web_dashboard.py |
| 9 | Review | Review tab frontend | review.js | review.html |
| 10 | Mobile | Mobile template + JS | mobile.js | mobile.html |
| 11 | Cleanup | Fix broken tests | — | tests/test_web_dashboard.py |
| 12 | Cleanup | E2E verify + archive old template | — | templates/ |
