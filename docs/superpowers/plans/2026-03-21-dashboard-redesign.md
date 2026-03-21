# Dashboard Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Polish and restructure the AlphaTemp dashboard with a CSS design system, restructure 3 tabs, and migrate hosting to Mac Mini.

**Architecture:** Create a shared `design-system.css` with CSS variables and component classes. Apply it across all tabs — polish Operations, restructure Health/Review/Blotter. Finally, set up launchd on Mac Mini with Tailscale access.

**Tech Stack:** CSS (custom properties), Jinja2 templates, vanilla JS, Flask, launchd, Tailscale

**Spec:** `docs/superpowers/specs/2026-03-21-dashboard-redesign-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `static/css/design-system.css` | Create | CSS variables + component classes |
| `static/css/dashboard.css` | Modify | Update existing classes, deprecate old ones |
| `templates/base.html` | Modify | Font swap, link design-system.css |
| `templates/operations.html` | Modify | Apply design system classes to card headers |
| `static/js/operations.js` | Modify | Inter→Space Mono, pill migration, border migration |
| `templates/health.html` | Modify | Restructure to 2×2 grid |
| `static/js/health.js` | Modify | Row striping, design system classes |
| `templates/review.html` | Modify | Restructure to sidebar layout |
| `static/js/review.js` | Modify | Sidebar pattern summary, filter pills, incident cards |
| `templates/blotter.html` | Modify | Apply design system to tables, risk metrics |
| `static/js/blotter.js` | Modify | Pill migration, calendar styling, filter pills |
| `scripts/run-dashboard.sh` | Create | Wrapper script for launchd |
| `com.alphatemp.dashboard.plist` | Create | launchd service definition (deployed to ~/Library/LaunchAgents/) |

---

### Task 1: Create Design System CSS

**Files:**
- Create: `static/css/design-system.css`

- [ ] **Step 1: Create design-system.css with CSS variables**

```css
/* AlphaTemp Design System */

:root {
  /* Page */
  --page-bg: #f0f1f3;
  --card-bg: #ffffff;
  --card-bg-alt: #f7f8fa;
  --header-bg: #1a1b2e;

  /* Text */
  --text-primary: #374151;
  --text-secondary: #6b7280;
  --text-muted: #9ca3af;

  /* Status */
  --profit: #10b981;
  --loss: #ef4444;
  --warning: #f59e0b;
  --info: #3b82f6;
  --purple: #8b5cf6;

  /* Status backgrounds (tinted) */
  --profit-bg: #dcfce7;
  --profit-text: #16a34a;
  --loss-bg: #fee2e2;
  --loss-text: #dc2626;
  --warning-bg: #fef3c7;
  --warning-text: #92400e;
  --info-bg: #dbeafe;
  --info-text: #2563eb;
  --orange-bg: #ffedd5;
  --orange-text: #ea580c;
  --neutral-bg: #f3f4f6;
  --neutral-text: #6b7280;

  /* Fonts */
  --font-ui: 'Space Mono', monospace;
  --font-data: 'JetBrains Mono', 'Roboto Mono', monospace;

  /* Surfaces */
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
  --shadow-md: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
  --radius: 10px;
  --radius-sm: 6px;
  --radius-pill: 9999px;
}
```

- [ ] **Step 2: Add component classes**

Append to `design-system.css`:

```css
/* Card header labels */
.at-card-header {
  font-family: var(--font-ui);
  font-size: 10px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 1px;
}

/* Stat cells (KPI grids) */
.at-stat {
  background: var(--card-bg);
  border-radius: var(--radius-sm);
  padding: 14px;
  box-shadow: var(--shadow-sm);
}
.at-stat .at-card-header {
  margin-bottom: 4px;
}
.at-stat-value {
  font-family: var(--font-data);
  font-size: 20px;
  font-weight: 700;
  color: var(--text-primary);
}

/* Tables */
.at-table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--font-data);
  font-size: 11px;
}
.at-table thead tr {
  background: var(--card-bg-alt);
}
.at-table th {
  padding: 8px 14px;
  font-family: var(--font-ui);
  font-size: 10px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-weight: 500;
}
.at-table td {
  padding: 10px 14px;
  color: var(--text-primary);
}
.at-table tbody tr:nth-child(even) {
  background: var(--card-bg-alt);
}

/* Pill badges */
.at-pill {
  display: inline-block;
  padding: 2px 8px;
  border-radius: var(--radius-pill);
  font-size: 10px;
  font-weight: 600;
  font-family: var(--font-ui);
  line-height: 1.4;
}
.at-pill-profit { background: var(--profit-bg); color: var(--profit-text); }
.at-pill-loss { background: var(--loss-bg); color: var(--loss-text); }
.at-pill-warning { background: var(--warning-bg); color: var(--warning-text); }
.at-pill-info { background: var(--info-bg); color: var(--info-text); }
.at-pill-orange { background: var(--orange-bg); color: var(--orange-text); }
.at-pill-neutral { background: var(--neutral-bg); color: var(--neutral-text); }

/* Filter pills */
.at-filter-pill {
  padding: 5px 12px;
  border-radius: var(--radius-sm);
  font-size: 10px;
  font-family: var(--font-ui);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  cursor: pointer;
  border: none;
  transition: all 0.15s;
}
.at-filter-pill-active {
  background: #1f2937;
  color: #ffffff;
}
.at-filter-pill-inactive {
  background: var(--neutral-bg);
  color: var(--text-secondary);
}

/* Key-value rows (health, config) */
.at-kv-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 10px;
  border-radius: 4px;
  font-family: var(--font-data);
  font-size: 11px;
}
.at-kv-row:nth-child(odd) {
  background: var(--card-bg-alt);
}
.at-kv-label {
  color: var(--text-primary);
}
.at-kv-value {
  color: var(--text-primary);
}
```

- [ ] **Step 3: Verify file serves correctly**

Run: `curl -s http://localhost:8050/static/css/design-system.css | head -5`
Expected: CSS content visible (`:root {`)

- [ ] **Step 4: Commit**

```bash
git add static/css/design-system.css
git commit -m "feat: add CSS design system with variables and component classes"
```

---

### Task 2: Update dashboard.css and base.html

**Files:**
- Modify: `static/css/dashboard.css` (lines 4, 10-15, 18-20, 101-133)
- Modify: `templates/base.html` (lines 11, 12, 14, 27, 53, 54)

- [ ] **Step 1: Update dashboard.css**

Changes to make:
- Line 4: `font-family: 'Inter'` → `font-family: var(--font-ui)`
- Lines 10-15 `.at-card`: Remove `border: 1px solid rgba(0,0,0,0.08)`, change `border-radius: 12px` → `border-radius: var(--radius)`, change `box-shadow` → `box-shadow: var(--shadow-md)`
- Lines 18-20 `.at-mono`: Change `font-family` value → `font-family: var(--font-data)`
- Lines 101-133: Add `/* DEPRECATED — mobile only */` comment before `.pill`, `.pill-positive`, `.pill-negative`, `.pill-neutral`, `.at-border-subtle`, `.at-border-faint` blocks

- [ ] **Step 2: Update base.html fonts and CSS link**

Changes to make:
- Line 11: Replace Google Fonts URL with `https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=JetBrains+Mono:wght@400;500;700&display=swap`
- After line 11, before the `dashboard.css` link: Add `<link rel="stylesheet" href="/static/css/design-system.css">`
- Line 14: Replace `font-family: 'Inter', -apple-system, sans-serif` → `font-family: 'Space Mono', monospace`
- Lines 28-34: KPI scorecard label `<span>` elements — add `style="font-family: var(--font-ui);"` to each label span (the ones with `text-gray-400 text-[10px] uppercase tracking-wider`)
- Line 27: `style="font-family: 'JetBrains Mono', monospace;"` is fine — it's already the data font. No change.
- Lines 53-54: Same — JetBrains Mono references stay for countdown/clock.

- [ ] **Step 3: Run existing tests to verify nothing breaks**

Run: `cd ~/Projects/alphatemp/alphatemp && python3 -m pytest tests/test_web_dashboard.py -q`
Expected: All tests pass

- [ ] **Step 4: Verify in browser — fonts and card styling changed**

Run: `curl -s http://localhost:8050/operations | grep -c "Space Mono"`
Expected: At least 1 match (from the Google Fonts link)

- [ ] **Step 5: Commit**

```bash
git add static/css/dashboard.css templates/base.html
git commit -m "feat: swap Inter for Space Mono, update card styles, link design system"
```

---

### Task 3: Polish Operations Tab

**Files:**
- Modify: `templates/operations.html` (card header `<h2>` elements)
- Modify: `static/js/operations.js` (lines 188, 274, 427, 388, 417, 481, 515, 598, 651-673, 812)

- [ ] **Step 1: Update operations.html card headers**

Add `at-card-header` class to all `<h2>` section headers in the template. Remove any remaining inline Inter font references or border styles on `.at-card` elements.

- [ ] **Step 2: Migrate Inter font references in operations.js**

Three changes:
- Line 188: `"Inter, sans-serif"` → `"Space Mono, monospace"`
- Line 274: `"Inter, sans-serif"` → `"Space Mono, monospace"`
- Line 427: `style="font-family: Inter, sans-serif;"` → `style="font-family: 'Space Mono', monospace;"`

- [ ] **Step 3: Migrate pill classes in operations.js**

Seven changes:
- Line 651: `"pill pill-positive"` → `"at-pill at-pill-profit"`
- Line 653: `"pill pill-negative"` → `"at-pill at-pill-loss"`
- Line 655: `"pill pill-neutral"` → `"at-pill at-pill-neutral"`
- Line 669: `"pill pill-positive"` → `"at-pill at-pill-profit"`
- Line 671: `"pill pill-negative"` → `"at-pill at-pill-loss"`
- Line 673: `"pill pill-neutral"` → `"at-pill at-pill-neutral"`
- Line 812: `'pill pill-positive'` → `'at-pill at-pill-profit'`

- [ ] **Step 4: Migrate border classes in operations.js**

Five changes:
- Line 388: Remove `at-border-subtle` from obs feed header row class string
- Line 417: Remove `at-border-faint` from obs feed data row class string
- Line 481: Remove `at-border-subtle` from fcst feed header row class string
- Line 515: Remove `at-border-faint` from fcst feed data row class string
- Line 598: Remove `at-border-subtle` from bracket ladder `<thead>` row class string

For header rows (388, 481, 598): add `background: var(--card-bg-alt)` inline style or add the `.at-table` parent class.

Also remove `at-border-faint` from bracket ladder body rows (~line 687 in `refreshBracketLadder()`).

- [ ] **Step 5: Run tests**

Run: `cd ~/Projects/alphatemp/alphatemp && python3 -m pytest tests/test_web_dashboard.py -k operations -q`
Expected: All operations tests pass

- [ ] **Step 6: Verify visually — check Operations tab in browser**

Open `http://localhost:8050/operations` and confirm: Space Mono on labels, no visible card borders, row striping on tables, new pill styles on order book.

- [ ] **Step 7: Commit**

```bash
git add templates/operations.html static/js/operations.js
git commit -m "feat: apply design system to Operations tab — fonts, pills, borders"
```

---

### Task 4: Restructure Health Tab

**Files:**
- Modify: `templates/health.html` (full restructure — 32 lines)
- Modify: `static/js/health.js` (lines 46, 74, 96, 125 + DOM structure)

- [ ] **Step 1: Restructure health.html to 2×2 grid**

Replace the `max-w-5xl mx-auto` wrapper with a 2-column grid:

```html
{% extends "base.html" %}
{% block title %}Health{% endblock %}
{% block content %}
<div class="grid grid-cols-1 lg:grid-cols-2 gap-3">
    <!-- Data Freshness — TOP LEFT -->
    <div class="at-card p-4">
        <h2 class="at-card-header mb-3">Data Freshness</h2>
        <div id="freshness-table" class="at-mono"></div>
    </div>
    <!-- Pipeline Status — TOP RIGHT -->
    <div class="at-card p-4">
        <h2 class="at-card-header mb-3">Pipeline Status</h2>
        <div id="pipeline-table" class="at-mono"></div>
    </div>
    <!-- Database — BOTTOM LEFT -->
    <div class="at-card p-4">
        <h2 class="at-card-header mb-3">Database</h2>
        <div id="db-table" class="at-mono"></div>
        <div id="db-size" class="at-mono mt-2"></div>
    </div>
    <!-- Configuration — BOTTOM RIGHT -->
    <div class="at-card p-4">
        <h2 class="at-card-header mb-3">Configuration</h2>
        <div id="config-table" class="at-mono"></div>
    </div>
</div>
{% endblock %}
{% block scripts %}
<script src="/static/js/health.js?v=3"></script>
{% endblock %}
```

- [ ] **Step 2: Update health.js — replace at-border-faint with at-kv-row**

For all four render functions (`renderFreshness`, `renderPipeline`, `renderDB`, `renderConfig`), replace the row div generation:

Old pattern (~lines 46, 74, 96, 125):
```js
'<div class="flex justify-between items-center py-1 at-border-faint">'
```

New pattern:
```js
'<div class="at-kv-row">'
```

For the Database card, convert to an `.at-table` `<table>` element instead of div rows. Use `<thead>` with Table/Rows headers and `<tbody>` with row data.

For status dots in the freshness card: update dot HTML to include `box-shadow: 0 0 4px <color>` glow for fresh sources (green dot = glow, amber/red = no glow).

For pipeline status badges: replace any existing badge markup with `.at-pill .at-pill-profit` (Running), `.at-pill .at-pill-loss` (Error), `.at-pill .at-pill-neutral` (Idle).

- [ ] **Step 3: Run tests**

Run: `cd ~/Projects/alphatemp/alphatemp && python3 -m pytest tests/test_web_dashboard.py -q`
Expected: All tests pass

- [ ] **Step 4: Verify visually**

Open `http://localhost:8050/health` and confirm: 2×2 grid layout, row striping in all cards, status dots with glow, pipeline pills, full-width utilization.

- [ ] **Step 5: Commit**

```bash
git add templates/health.html static/js/health.js
git commit -m "feat: restructure Health tab to 2x2 grid with design system"
```

---

### Task 5: Restructure Review Tab

**Files:**
- Modify: `templates/review.html` (full restructure — 56 lines)
- Modify: `static/js/review.js` (lines 33, 35, 57, 59, 112, 97-154, 162-193)

- [ ] **Step 1: Restructure review.html to sidebar layout**

Replace the linear stack with a two-column layout:

```html
{% extends "base.html" %}
{% block title %}Review{% endblock %}
{% block content %}
<!-- Filter Bar -->
<div class="flex items-center justify-between mb-4">
    <div id="range-buttons" class="flex gap-1"></div>
    <div id="filter-buttons" class="flex gap-1"></div>
</div>

<!-- Two-column: Incidents + Sidebar -->
<div class="flex gap-3" style="align-items: flex-start;">
    <!-- Incident Cards (scrollable) -->
    <div id="incident-list" class="flex-1 space-y-3"></div>

    <!-- Pattern Summary Sidebar (sticky) -->
    <div class="at-card p-4" style="width: 280px; position: sticky; top: 16px; flex-shrink: 0;">
        <h2 class="at-card-header mb-3">Patterns</h2>
        <div id="pattern-summary"></div>
    </div>
</div>
{% endblock %}
{% block scripts %}
<script src="/static/js/review.js?v=4"></script>
{% endblock %}
```

- [ ] **Step 2: Update review.js — filter buttons**

Update `setReviewRange()` (~lines 33-35) and `setFilter()` (~lines 57-59):

Replace Tailwind button class logic with:
- Active: `"at-filter-pill at-filter-pill-active"`
- Inactive: `"at-filter-pill at-filter-pill-inactive"`

For category filter chips, add count badges and color tinting:
- Worst: when active, use `background: var(--loss-bg); color: var(--loss-text)` via inline style
- Lost: `var(--warning-bg)` / `var(--warning-text)`
- Missed: `var(--info-bg)` / `var(--info-text)`

- [ ] **Step 3: Update review.js — incident card markup**

Update the card generation (~lines 97-154). Key changes:
- Remove `border` and `at-border-subtle` from card class string (line 112)
- Use `.at-card` class only
- Top row: date in Space Mono + category `.at-pill` badge (left), P&L value large and colored (right)
- Middle row: bracket · direction (YES = `.at-pill .at-pill-info`, NO = `.at-pill .at-pill-orange`) · settled temp
- Metric boxes: 3 side-by-side divs with tinted backgrounds (blue for model, amber for market, green/red for edge), label in `.at-card-header` style, value in large `var(--font-data)`
- Narrative text: `color: var(--text-secondary); font-size: 11px; line-height: 1.4`

- [ ] **Step 4: Update review.js — pattern summary as sidebar content**

Update pattern summary generation (~lines 162-193). Replace the `<ol>` list with:
- Each pattern: div with `border-left: 3px solid <category-color>` and `padding-left: 10px`
- Category name (Space Mono, bold) + aggregate P&L (large, colored) on same line using flex
- Incident count below in muted text
- Suggested action text in `var(--text-muted)` smaller font
- At bottom: Key Insight callout box with `background: var(--card-bg-alt); border-radius: var(--radius-sm); padding: 10px`

The pattern summary now renders into `#pattern-summary` div in the sidebar, not at the bottom of the page.

- [ ] **Step 5: Run tests**

Run: `cd ~/Projects/alphatemp/alphatemp && python3 -m pytest tests/test_web_dashboard.py -q`
Expected: All tests pass

- [ ] **Step 6: Verify visually**

Open `http://localhost:8050/review` and confirm: filter bar with styled pills, two-column layout, incident cards with metric boxes, sticky pattern summary sidebar.

- [ ] **Step 7: Commit**

```bash
git add templates/review.html static/js/review.js
git commit -m "feat: restructure Review tab — sidebar pattern summary, styled incident cards"
```

---

### Task 6: Polish Blotter Tab

**Files:**
- Modify: `templates/blotter.html` (lines 7-21, 48, 70, 72, 92, 94)
- Modify: `static/js/blotter.js` (lines 51-55, 164-168, 230-288, 346-352, 514, 616-617)

- [ ] **Step 1: Update blotter.html — design system classes**

Changes:
- Summary stats bar (lines 7-21): Add `at-card-header` to stat labels, apply `.at-stat` sizing pattern
- Risk metrics grid (line 48): Add `.at-stat` class to each grid cell
- Live positions table (line 70): Add `at-table` class to `<table>` element
- Live positions `<thead>` (line 72): Remove `at-border-subtle`, add Space Mono header styling
- Closed positions table (line 92): Add `at-table` class
- Closed positions `<thead>` (line 94): Remove `at-border-subtle`

- [ ] **Step 2: Update blotter.js — pill migration in renderSettleBanner()**

Three changes:
- Line 346: `"pill pill-neutral"` → `"at-pill at-pill-neutral"`
- Line 349: `"pill pill-positive"` → `"at-pill at-pill-profit"`
- Line 352: `"pill pill-positive"` → `"at-pill at-pill-profit"`

- [ ] **Step 3: Update blotter.js — sideClass() for YES/NO pills**

Update `sideClass()` function (~lines 51-55) to return `.at-pill` classes:
- YES → return class string that applies `.at-pill .at-pill-info`
- NO → return class string that applies `.at-pill .at-pill-orange`

Update call sites to wrap the side text in a `<span>` with the pill classes:
- `renderLivePositions()` (~lines 523-526)
- `renderClosedPositions()` (~lines 629-632)

- [ ] **Step 4: Update blotter.js — border class migration**

Two changes:
- Line 514: Remove `"at-border-faint"` from live positions row class — `.at-table` striping handles it
- Line 616: Remove `"at-border-faint"` from closed positions row class
- Line 617: Change `"border-l-2 border-l-amber-400"` (Tailwind) to `style="border-left: 3px solid var(--warning)"` for settlement-held rows

- [ ] **Step 5: Update blotter.js — range button classes**

Update `setBlotterRange()` (~lines 164-168):
- Active: `"at-filter-pill at-filter-pill-active"`
- Inactive: `"at-filter-pill at-filter-pill-inactive"`

- [ ] **Step 6: Update blotter.js — calendar day styling**

Update calendar cell generation (~lines 230-288):
- Profit days: Replace `bg-emerald-50` / `bg-emerald-200` with `background: var(--profit-bg)`
- Loss days: Replace `bg-red-50` / `bg-red-200` with `background: var(--loss-bg)`
- All day cells: Add `border-radius: var(--radius-sm)` (6px)
- Today cell: Replace `ring-1 ring-gray-400` with `background: #1f2937; color: #fff; box-shadow: 0 0 0 2px var(--info)`

- [ ] **Step 7: Update blotter.html — settlement banner styling**

Change the settlement banner card from white `.at-card` to amber-tinted:
- Replace `class="at-card ..."` with `style="background: var(--warning-bg); border-radius: var(--radius);"` and add `box-shadow: var(--shadow-md)`
- Update labels inside to use `.at-card-header` with `color: var(--warning-text)`

- [ ] **Step 8: Run tests**

Run: `cd ~/Projects/alphatemp/alphatemp && python3 -m pytest tests/test_web_dashboard.py -q`
Expected: All tests pass

- [ ] **Step 9: Verify visually**

Open `http://localhost:8050/blotter` and confirm: styled stats bar, amber settlement banner, calendar with rounded tinted cells, pill-styled side indicators, row-striped tables, settlement-held amber bar.

- [ ] **Step 10: Commit**

```bash
git add templates/blotter.html static/js/blotter.js
git commit -m "feat: apply design system to Blotter — pills, tables, calendar, settlement banner"
```

---

### Task 7: Cache Bust and Final Polish

**Files:**
- Modify: `templates/base.html` (cache bust CSS links)
- Modify: `templates/operations.html` (cache bust JS link)
- Modify: `templates/blotter.html` (cache bust JS link)
- Modify: `templates/health.html` (cache bust JS link)
- Modify: `templates/review.html` (cache bust JS link)

- [ ] **Step 1: Bump all ?v= query strings**

In every template, find the `<script src=` and `<link href=` tags and bump the version number. Add `?v=1` to `design-system.css` and `dashboard.css` if they don't already have one.

- [ ] **Step 2: Run full test suite**

Run: `cd ~/Projects/alphatemp/alphatemp && python3 -m pytest tests/test_web_dashboard.py -q`
Expected: All tests pass (7 passed, 0 failed)

- [ ] **Step 3: Visual sweep of all tabs**

Open each tab in the browser and verify:
- `http://localhost:8050/operations` — Space Mono labels, no card borders, new pills
- `http://localhost:8050/blotter` — Amber banner, styled calendar, pill side indicators
- `http://localhost:8050/health` — 2×2 grid, full width, status dot glows
- `http://localhost:8050/review` — Sidebar pattern summary, metric boxes, filter pills

- [ ] **Step 4: Commit**

```bash
git add templates/ static/
git commit -m "chore: bump cache bust versions across all tabs"
```

---

### Task 8: Mac Mini Hosting Setup

**Files:**
- Create: `scripts/run-dashboard.sh`
- Create: `com.alphatemp.dashboard.plist` (template — deployed to `~/Library/LaunchAgents/`)

- [ ] **Step 1: Create wrapper script**

```bash
#!/bin/bash
cd ~/Projects/alphatemp/alphatemp
source .env
exec ~/Projects/alphatemp/alphatemp/venv/bin/python main.py
```

```bash
mkdir -p scripts
```

Save to `scripts/run-dashboard.sh` and `chmod +x scripts/run-dashboard.sh`.

- [ ] **Step 2: Create launchd plist**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.alphatemp.dashboard</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>/Users/russellrudd/Projects/alphatemp/alphatemp/scripts/run-dashboard.sh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/russellrudd/Projects/alphatemp/alphatemp</string>
    <key>KeepAlive</key>
    <true/>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/russellrudd/Projects/alphatemp/alphatemp/logs/dashboard.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/russellrudd/Projects/alphatemp/alphatemp/logs/dashboard.log</string>
</dict>
</plist>
```

Save to `com.alphatemp.dashboard.plist` in the project root (template for deployment).

- [ ] **Step 3: Deploy and load on Mini**

```bash
# Copy plist to LaunchAgents
cp com.alphatemp.dashboard.plist ~/Library/LaunchAgents/

# Create logs directory
mkdir -p ~/Projects/alphatemp/alphatemp/logs

# Load the service (bootstrap is the modern macOS approach)
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.alphatemp.dashboard.plist

# Verify it's running
launchctl list | grep alphatemp
curl -s http://localhost:8050/api/health | python3 -m json.tool
```

Expected: Service listed as running, health endpoint returns JSON.

- [ ] **Step 4: Verify Tailscale access**

From another device on Tailscale:
```bash
curl -s http://<mini-tailscale-ip>:8050/api/health
```
Expected: Health JSON returned.

- [ ] **Step 5: Verify auto-restart**

```bash
launchctl kill SIGTERM gui/$(id -u)/com.alphatemp.dashboard
sleep 3
curl -s http://localhost:8050/api/health | python3 -m json.tool
```
Expected: Service restarts automatically, health endpoint responds.

- [ ] **Step 6: Commit**

```bash
git add scripts/run-dashboard.sh com.alphatemp.dashboard.plist
git commit -m "feat: add launchd service for Mac Mini hosting"
```

---

### Task 9: Archive Docker/DO Deployment

**Files:**
- Move: `deploy.sh` → `deploy/archive/deploy.sh`
- Move: `Dockerfile` → `deploy/archive/Dockerfile`
- Move: `docker-compose.yml` → `deploy/archive/docker-compose.yml`

- [ ] **Step 1: Move deployment files to archive**

```bash
mkdir -p deploy/archive
git mv deploy.sh deploy/archive/deploy.sh
git mv Dockerfile deploy/archive/Dockerfile
git mv docker-compose.yml deploy/archive/docker-compose.yml
```

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: archive Docker/DigitalOcean deployment files"
```
