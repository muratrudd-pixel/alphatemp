# AlphaTemp Dashboard Redesign

## Overview

Polish and restructure the AlphaTemp dashboard across all tabs, establish a CSS design system for consistency, and migrate hosting from DigitalOcean to a Mac Mini with Tailscale access.

## Decisions

- **Theme:** Polished Light — dark header (#1a1b2e), light body (#f0f1f3), white cards
- **Scope:** Restructure Health, Review, and Blotter tabs. Polish Operations tab (no structural changes)
- **Fonts:** Space Mono for all UI text (labels, headers, nav). JetBrains Mono for all numerical/tabular data
- **Approach:** CSS Design System first — centralize variables and component classes, then apply across all tabs
- **Hosting:** Move from DigitalOcean Docker to Mac Mini with launchd + Tailscale
- **Mobile:** Skip for now

---

## 1. Design System (`static/css/design-system.css`)

### CSS Variables

```css
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
  --radius: 10px;  /* intentional tightening from 12px — more technical feel */
  --radius-sm: 6px;
  --radius-pill: 9999px;  /* fully rounded pill ends */
}
```

### Font Loading

Replace Inter with Space Mono in `base.html`:

```html
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
```

### Component Classes

**`.at-card`** — Card panels (no border, shadow only):
- `background: var(--card-bg)`
- `border-radius: var(--radius)`
- `box-shadow: var(--shadow-md)`
- No `border` property

**`.at-card-header`** — Card section header:
- `font-family: var(--font-ui)`
- `font-size: 10px`
- `color: var(--text-muted)`
- `text-transform: uppercase`
- `letter-spacing: 1px`

**`.at-stat`** — Metric cell (KPI grids):
- `background: var(--card-bg)`
- `border-radius: var(--radius-sm)`
- `padding: 14px`
- `box-shadow: var(--shadow-sm)`
- Label: `.at-card-header` style
- Value: `var(--font-data)`, 20px, bold

**`.at-table`** — Data tables:
- No border-collapse dividers
- Header row: `background: var(--card-bg-alt)`, Space Mono 10px uppercase
- Body rows: alternating white / `var(--card-bg-alt)` (striping)
- Cell padding: `10px 14px`
- All cell text: `var(--font-data)`, 11px

**`.at-pill`** — Badge/pill:
- `font-family: var(--font-ui)`
- `font-size: 10px`, `font-weight: 600`
- `padding: 2px 8px`
- `border-radius: var(--radius-pill)`
- Variants:
  - `.at-pill-profit` — `background: var(--profit-bg); color: var(--profit-text)`
  - `.at-pill-loss` — `background: var(--loss-bg); color: var(--loss-text)`
  - `.at-pill-warning` — `background: var(--warning-bg); color: var(--warning-text)`
  - `.at-pill-info` — `background: var(--info-bg); color: var(--info-text)`
  - `.at-pill-orange` — `background: var(--orange-bg); color: var(--orange-text)` (Blotter NO side)
  - `.at-pill-neutral` — `background: var(--neutral-bg); color: var(--neutral-text)`

**`.at-filter-pill`** — Filter/range selector:
- `font-family: var(--font-ui)`
- `font-size: 10px`
- `padding: 5px 12px`
- `border-radius: var(--radius-sm)`
- Active: `background: #1f2937; color: #fff`
- Inactive: `background: #f3f4f6; color: var(--text-secondary)`

### Existing Classes to Update

- `.at-mono` — **keep**, update definition to `font-family: var(--font-data)` for consistency. Used 34+ times across templates and JS — no rename needed, just redefine to use the variable.
- `.glow-dot` — unchanged (already good)
- `.skeleton` — unchanged
- `.feed-scroll` — unchanged
- `.toggle-pill` — unchanged
- `.severity-bar` — unchanged
- `.stale-banner` — unchanged
- `.sortable` — unchanged

### Deprecated Classes — Migration Plan

**Remove from `dashboard.css`:** `.pill`, `.pill-positive`, `.pill-negative`, `.pill-neutral`
- **Migration:** Find all usages in JS files and replace with `.at-pill .at-pill-{variant}`. Specific locations:
  - `operations.js` — bracket ladder edge/EV pills (7 usages)
  - `blotter.js` — `renderSettleBanner()` settlement badges (3 usages)

**Remove from `dashboard.css`:** `.at-border-subtle`, `.at-border-faint`
- **Migration by usage type:**
  - Table body row dividers → replaced by `.at-table` row striping (remove class)
  - Table `<thead>` row underlines (blotter.html lines 72, 94) → handled by `.at-table` header background (`var(--card-bg-alt)`) providing visual separation (remove class)
  - Non-table dividers in feed/grid layouts (operations.js line 388, etc.) → replace with `border-bottom: 1px solid rgba(0,0,0,0.05)` inline where still needed, or remove if striping handles it
  - `mobile.js` (5 usages) → leave as-is since mobile is out of scope. Keep the CSS class definitions behind a `/* DEPRECATED — mobile only */` comment until mobile is redesigned

### Shared JS Updates (`shared.js`)

- `PLOTLY_LAYOUT.font.family` → `"JetBrains Mono, Roboto Mono, monospace"` (already correct)
- `PLOTLY_LAYOUT.paper_bgcolor` / `plot_bgcolor` → keep `#ffffff`
- No other changes needed — color palette and config are fine

---

## 2. Base Template (`base.html`)

- Swap Google Fonts link: Inter → Space Mono
- Update `<body>` font-family to `'Space Mono', monospace`
- Link new `design-system.css` (load before `dashboard.css`)
- Header: swap any remaining Inter references to Space Mono
- KPI scorecard labels: already use inline styles, update to `font-family: var(--font-ui)`

---

## 3. Operations Tab (polish only)

No layout changes. Apply design system:

- Card headers: swap to `.at-card-header` class
- Order book ladder table: apply `.at-table` styling (row striping, Space Mono headers)
- Observation feed table: apply `.at-table` styling
- Forecast runs table: apply `.at-table` styling
- Remove any `border: 1px solid` on `.at-card` elements
- Plotly chart container: no changes (white background stays)
- Legend text: update to Space Mono
- **Inter font references in `operations.js`:** Update 3 hardcoded `"Inter, sans-serif"` references to `'Space Mono', monospace`:
  - Settlement marker textfont (~line 188)
  - Chart "Now" annotation font (~line 274)
  - Obs feed badge container inline style (~line 427)
- **Pill migration:** Replace all `.pill .pill-positive` / `.pill-negative` / `.pill-neutral` usages in bracket ladder with `.at-pill .at-pill-{variant}`

---

## 4. Health Tab (restructure)

**Current:** Single column (`max-w-5xl`), four stacked cards.

**New:** 2×2 grid filling full width.

```
┌─────────────────────┬─────────────────────┐
│   Data Freshness    │   Pipeline Status   │
│   (7 sources)       │   (5 pipelines)     │
├─────────────────────┼─────────────────────┤
│   Database          │   Configuration     │
│   (table stats)     │   (key-value)       │
└─────────────────────┴─────────────────────┘
```

**Grid:** `grid-template-columns: 1fr 1fr; gap: 12px`

**Data Freshness card:**
- Each source row: name (left), age value in `var(--font-data)` (center-right), status dot (right)
- Status dots: 8px, green with glow = fresh, amber static = marginal, red static = stale
- Row striping via alternating `var(--card-bg-alt)` backgrounds
- Age text color: `var(--text-primary)` when fresh, `var(--warning-text)` when marginal, `var(--loss-text)` when stale

**Pipeline Status card:**
- Each row: pipeline name, cycle duration in ms, status pill
- Running = `.at-pill-profit`, Error = `.at-pill-loss`, Idle = `.at-pill-neutral`
- Row striping

**Database card:**
- `.at-table` with columns: Table, Rows
- Row count formatted with commas
- Footer row: DB file size

**Configuration card:**
- Key-value rows: key in `var(--text-muted)`, value in `var(--text-primary)` with `var(--font-data)`
- Row striping

---

## 5. Review Tab (restructure)

**Current:** Linear stack — filter bar → incident cards → pattern summary at bottom.

**New:** Filter bar on top, then two-column layout — incident cards left (~65%), pattern summary sidebar right (280px).

**Note:** Existing incident card markup in `review.js` adds explicit Tailwind `border` class alongside `at-border-subtle`. Both must be removed when applying `.at-card` (shadow-only, no border).

```
┌─────────────────────────────────────────────┐
│  [7D] [30D] [All]    [All] [Worst] [Lost]  │
├────────────────────────────┬────────────────┤
│                            │   PATTERNS     │
│   Incident Card 1          │                │
│   Incident Card 2          │   ▌ Worst Miss │
│   Incident Card 3          │   ▌ Lost Bets  │
│   ...scrollable...         │   ▌ Missed     │
│                            │                │
│                            │   Key Insight  │
└────────────────────────────┴────────────────┘
```

**Filter bar:**
- Time range pills (left): `.at-filter-pill` style
- Category chips (right): color-tinted backgrounds with count badges
  - Worst: `var(--loss-bg)` / `var(--loss-text)`
  - Lost: `var(--warning-bg)` / `var(--warning-text)`
  - Missed: `var(--info-bg)` / `var(--info-text)`

**Incident cards:**
- Top row: date (Space Mono, muted) + category pill | P&L value (large, colored)
- Middle row: bracket · direction (YES blue / NO orange pill) · settled temp
- Metric boxes: 3-up grid — Model% (blue tint), Market% (amber tint), Edge% (green/red tint)
  - Each box: Space Mono label on top, large JetBrains Mono value below
- Narrative text: 11px, `var(--text-secondary)`, 1.4 line-height

**Pattern Summary sidebar (280px, sticky):**
- Each pattern entry: colored left border (3px) matching category
  - Name (Space Mono, bold) + aggregate P&L (large, colored) on same line
  - Incident count below
  - Suggested action text in muted smaller font
- Key Insight callout at bottom: `var(--card-bg-alt)` background, compact text

---

## 6. Blotter Tab (polish)

Layout structure preserved: summary stats bar on top, calendar sidebar left (260px), content right.

**Summary stats bar:**
- Apply `.at-stat` sizing inline (larger values, Space Mono labels)
- Range pills: `.at-filter-pill` style
- Horizontal layout with `justify-content: space-between`

**Calendar sidebar:**
- Month header: Space Mono, month name + P&L value
- Day headers: Space Mono 9px uppercase
- Day cells: `border-radius: var(--radius-sm)`
- Profit days: `var(--profit-bg)` tint
- Loss days: `var(--loss-bg)` tint
- Today: dark fill (`#1f2937`) + blue ring (`box-shadow: 0 0 0 2px var(--info)`)
- No-trade days: plain

**Settlement banner:**
- `background: var(--warning-bg)` instead of white card with border
- Three sections: date + source pill | running high | countdown timer
- All labels: Space Mono uppercase

**Risk metrics:**
- 4-column `.at-stat` grid: Capital Locked, Portfolio EV, Max Loss, Day P&L

**Live Positions table:**
- `.at-table` styling applied to existing column set (Bracket, Side, Model, Qty, Entry, Mark, P&L, ROC, Opened — column set unchanged)
- Side column: YES = `.at-pill-info`, NO = `.at-pill-orange`
- P&L column: colored by sign

**Settled table:**
- `.at-table` styling applied to existing column set (Bracket, Side, Model, Qty, Entry, Exit, Gross, Fees, Net, Opened, Settled — column set unchanged)
- Settlement-held rows: `border-left: 3px solid var(--warning)` + STL label

---

## 7. Mac Mini Hosting

### launchd Service

Create a wrapper script `scripts/run-dashboard.sh` that sources `.env` then execs python:

```bash
#!/bin/bash
cd ~/Projects/alphatemp/alphatemp
source .env
exec ~/Projects/alphatemp/alphatemp/venv/bin/python main.py
```

Create `~/Library/LaunchAgents/com.alphatemp.dashboard.plist`:
- `ProgramArguments`: `/bin/bash`, `~/Projects/alphatemp/alphatemp/scripts/run-dashboard.sh`
- `WorkingDirectory`: `~/Projects/alphatemp/alphatemp`
- `KeepAlive`: true (auto-restart on crash)
- `RunAtLoad`: true (start on login)
- `StandardOutPath` / `StandardErrorPath`: `~/Projects/alphatemp/alphatemp/logs/dashboard.log`

### Network

- Bind to `0.0.0.0:8050` (already the case if using Flask's default)
- Tailscale handles access — no port forwarding, no firewall changes needed
- Access via `http://<mini-tailscale-ip>:8050` from any Tailscale device

### Migration Steps

1. Verify dashboard runs on Mini: `cd ~/Projects/alphatemp/alphatemp && python3 main.py`
2. Create and load launchd plist
3. Verify auto-restart: `launchctl kill SIGTERM gui/$(id -u)/com.alphatemp.dashboard`
4. Verify Tailscale access from another device
5. Archive `deploy.sh`, `Dockerfile`, `docker-compose.yml` (move to `deploy/archive/`)

---

## Files Modified

| File | Change |
|------|--------|
| `static/css/design-system.css` | **New** — CSS variables + component classes |
| `static/css/dashboard.css` | Update existing classes, remove deprecated ones |
| `templates/base.html` | Font swap, link design-system.css |
| `templates/operations.html` | Apply design system classes |
| `static/js/operations.js` | Font references in dynamic HTML |
| `templates/health.html` | Restructure to 2×2 grid |
| `static/js/health.js` | Update DOM generation to use new classes |
| `templates/review.html` | Restructure to sidebar layout |
| `static/js/review.js` | Update DOM generation, sidebar logic |
| `templates/blotter.html` | Apply design system, polish calendar |
| `static/js/blotter.js` | Update DOM generation to use new classes |
| `static/js/shared.js` | Minor — no structural changes needed |
| `com.alphatemp.dashboard.plist` | **New** — launchd service definition |

## Files Not Modified

- `templates/mobile.html`, `static/js/mobile.js` — skipped per decision
- `main.py` — no backend changes
- `ui/web_dashboard.py` — no API changes
- Any `services/` or `core/` files

---

## Implementation Notes

- **Cache busting:** Bump all `?v=N` query strings on JS/CSS includes in templates after changes are complete.
- **Toast colors:** `showToast()` in shared.js uses dark backgrounds (`bg-red-900/80`, etc.) — these are intentionally dark for high contrast on the light theme. No change needed.
- **Mobile CSS preservation:** Keep `.at-border-subtle` and `.at-border-faint` definitions in `dashboard.css` behind a `/* DEPRECATED — mobile only */` comment until mobile tab is redesigned.
