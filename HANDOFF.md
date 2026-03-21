# Handoff - 2026-03-21

## Current State
- Branch: `autoresearch/run-2026-03-10`
- Dashboard running on Mac Mini via launchd (`com.alphatemp.dashboard`) at `100.81.62.78:8050`
- All 46 tests passing

## What We Did (2026-03-21 Session)

### Dashboard Redesign — Full Implementation
1. **CSS Design System** — Created `static/css/design-system.css` with CSS variables, component classes (`.at-card-header`, `.at-stat`, `.at-table`, `.at-pill`, `.at-filter-pill`, `.at-kv-row`)
2. **Font swap** — Inter → Space Mono for UI text, JetBrains Mono stays for data
3. **Operations tab** — Applied design system (fonts, pills, borders). No layout changes.
4. **Health tab** — Restructured from single narrow column to 2×2 grid (Data Freshness + Pipeline | Database + Config)
5. **Review tab** — Restructured from linear stack to sidebar layout (pattern summary as sticky right sidebar, not buried at bottom)
6. **Blotter tab** — Applied design system (pills, calendar styling, amber settlement banner, filter pills, table striping)
7. **Cache busting** — Bumped all `?v=` strings
8. **Mac Mini hosting** — Created `scripts/run-dashboard.sh` wrapper + `com.alphatemp.dashboard.plist`, deployed via launchd with KeepAlive
9. **Docker archived** — `deploy.sh`, `Dockerfile`, `docker-compose.yml` moved to `deploy/archive/`

### Files Modified
- `static/css/design-system.css` (new)
- `static/css/dashboard.css`
- `templates/base.html`, `operations.html`, `health.html`, `review.html`, `blotter.html`
- `static/js/operations.js`, `health.js`, `review.js`, `blotter.js`
- `scripts/run-dashboard.sh` (new)
- `com.alphatemp.dashboard.plist` (new)
- `docs/superpowers/specs/2026-03-21-dashboard-redesign-design.md` (new)
- `docs/superpowers/plans/2026-03-21-dashboard-redesign.md` (new)

## Known Issues
- Mini DB has 1.2M forecast rows from a prior setup but chart shows "no run data" — likely a date filter issue (forecasts may be old). Didn't investigate.
- Synoptic API returning "no data: unknown" warnings — known flaky API, AWC/IEM backup is working
- Forecasts 5146 min stale — the HRRR/GFS fetcher needs time to populate or the existing data is too old for today's date filter

## Next Steps
1. Investigate "no run data" on Operations chart — check if forecast dates match today's filter
2. Optionally rsync historical DB from MBP: `rsync -av --progress ~/Projects/alphatemp/alphatemp/data/ mini:~/Projects/alphatemp/alphatemp/data/`
3. Items from prior handoff still pending:
   - Herbie timeout (system stability)
   - Running high as QR feature (model accuracy)
   - Time-decay GFS/ECMWF influence
   - Execute tail bracket + dedup + EV filter plan

## Verification Queries
```bash
cd ~/Projects/alphatemp/alphatemp

# Tests
python3 -m pytest tests/test_web_dashboard.py -q

# Dashboard health (on Mini)
curl -s http://100.81.62.78:8050/api/health | python3 -m json.tool

# Confirm commits
git log --oneline -12

# Check launchd service
ssh mini "launchctl list | grep alphatemp"
```
