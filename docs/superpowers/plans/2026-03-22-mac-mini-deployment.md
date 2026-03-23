# Mac Mini Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy alphatemp to Mac Mini as always-on production box with one-command deploys and Cloudflare Tunnel for public access.

**Architecture:** Git clone on Mini, Python 3.12 venv, launchd for process management, deploy.sh for push-button deploys from MacBook, Cloudflare Tunnel for public HTTPS.

**Tech Stack:** Python 3.12, launchd, rsync, Cloudflare Tunnel, Tailscale

**Spec:** `docs/superpowers/specs/2026-03-22-mac-mini-deployment-design.md`

**SSH target:** `russellrudd@100.120.114.84` (Mac Mini via Tailscale)

---

### Task 1: Fix `run-dashboard.sh` — Remove `source .env`

**Why:** `source .env` is fragile with special characters in API keys. `main.py` already calls `load_dotenv()` which handles this safely. Remove the shell-level sourcing.

**Files:**
- Modify: `scripts/run-dashboard.sh`

- [ ] **Step 1: Edit the script**

Replace the entire file with:
```bash
#!/bin/bash
cd ~/Projects/alphatemp/alphatemp
exec ~/Projects/alphatemp/alphatemp/venv/bin/python main.py --dashboard
```

- [ ] **Step 2: Verify the script is executable**

Run: `chmod +x scripts/run-dashboard.sh`

- [ ] **Step 3: Commit**

```bash
git add scripts/run-dashboard.sh
git commit -m "fix: remove source .env from run-dashboard.sh — python-dotenv handles it"
```

---

### Task 2: Update launchd plist — Fix `launchctl` compatibility

**Why:** `launchctl load` is deprecated on modern macOS. Update plist and add comments documenting correct commands.

**Files:**
- Modify: `com.alphatemp.dashboard.plist`

- [ ] **Step 1: Verify plist paths**

The plist already uses `/Users/russellrudd/` paths which match the Mini. No path changes needed. Confirm by reading the file.

- [ ] **Step 2: Add a comment file with launchctl commands**

Create `scripts/launchd-commands.md`:
```markdown
# Launchd Commands (macOS 26+)

## Install (one-time)
cp com.alphatemp.dashboard.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.alphatemp.dashboard.plist

## Stop
launchctl bootout gui/$(id -u)/com.alphatemp.dashboard

## Restart
launchctl kickstart -k gui/$(id -u)/com.alphatemp.dashboard

## Check status
launchctl print gui/$(id -u)/com.alphatemp.dashboard

## View logs
tail -f ~/Projects/alphatemp/alphatemp/logs/dashboard.log
```

- [ ] **Step 3: Commit**

```bash
git add scripts/launchd-commands.md
git commit -m "docs: add launchd command reference for macOS 26+"
```

---

### Task 3: Create `.env.example`

**Why:** The Mini needs env vars but there's no template documenting which ones are required. Create one.

**Files:**
- Create: `.env.example`

- [ ] **Step 1: Create the template**

```bash
# Required
KALSHI_API_KEY=
KALSHI_API_SECRET=

# Optional (disables Synoptic ingestor if missing)
SYNOPTIC_TOKEN=
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: add .env.example with required env vars"
```

---

### Task 4: Create `scripts/deploy.sh`

**Why:** One-command deploy from MacBook to Mini. SSH in, pull, reinstall deps if changed, restart service.

**Files:**
- Create: `scripts/deploy.sh`

- [ ] **Step 1: Write the deploy script**

```bash
#!/bin/bash
set -euo pipefail

MINI="russellrudd@100.120.114.84"

echo "==> Deploying alphatemp to Mac Mini..."

# Pull latest code and conditionally update deps
ssh "$MINI" bash -l <<'REMOTE'
set -euo pipefail
cd ~/Projects/alphatemp/alphatemp

# Capture current requirements hash
OLD_HASH=$(md5 -q requirements.txt 2>/dev/null || echo "none")

echo "==> Pulling latest..."
git pull

# Only pip install if requirements changed
NEW_HASH=$(md5 -q requirements.txt 2>/dev/null || echo "none")
if [ "$OLD_HASH" != "$NEW_HASH" ]; then
    echo "==> Requirements changed, installing..."
    venv/bin/pip install -r requirements.txt -q
else
    echo "==> Requirements unchanged, skipping pip install"
fi

echo "==> Restarting service..."
launchctl kickstart -k gui/$(id -u)/com.alphatemp.dashboard

echo "==> Waiting for startup..."
sleep 3

if curl -sf http://localhost:8050 > /dev/null 2>&1; then
    echo "==> Health check PASSED — dashboard is up"
else
    echo "==> Health check FAILED — check logs/dashboard.log"
    tail -20 ~/Projects/alphatemp/alphatemp/logs/dashboard.log
    exit 1
fi
REMOTE

echo "==> Deploy complete!"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x scripts/deploy.sh`

- [ ] **Step 3: Test locally that the script parses**

Run: `bash -n scripts/deploy.sh`
Expected: no output (no syntax errors)

- [ ] **Step 4: Commit**

```bash
git add scripts/deploy.sh
git commit -m "feat: add one-command deploy script for Mac Mini"
```

---

### Task 5: Push code to GitHub

**Why:** Mini will clone from GitHub. All changes from Tasks 1-4 need to be on the remote.

- [ ] **Step 1: Push current branch**

```bash
git push origin autoresearch/run-2026-03-10
```

---

### Task 6: Bootstrap Mini — Clone, venv, deps

**Why:** One-time setup on the Mini. Clone repo, create Python environment, install deps.

All commands run via SSH from MacBook.

- [ ] **Step 1: Clone the repo**

```bash
ssh russellrudd@100.120.114.84 bash -l <<'REMOTE'
set -euo pipefail
mkdir -p ~/Projects/alphatemp
git clone https://github.com/muratrudd-pixel/alphatemp.git ~/Projects/alphatemp/alphatemp
cd ~/Projects/alphatemp/alphatemp
git checkout autoresearch/run-2026-03-10
echo "==> Clone complete on branch $(git branch --show-current)"
REMOTE
```

- [ ] **Step 2: Create venv and install deps**

```bash
ssh russellrudd@100.120.114.84 bash -l <<'REMOTE'
set -euo pipefail
cd ~/Projects/alphatemp/alphatemp
/opt/homebrew/bin/python3.12 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
echo "==> Pip install complete"
REMOTE
```

Expected: all packages install. If any fail (cfgrib, pygrib most likely), note the error — may need `brew install eccodes` flags.

- [ ] **Step 3: Smoke test imports**

```bash
ssh russellrudd@100.120.114.84 bash -l <<'REMOTE'
cd ~/Projects/alphatemp/alphatemp
venv/bin/python -c "
import duckdb, scipy, xgboost, cfgrib, numpy, pandas, polars
print('All critical imports OK')
"
REMOTE
```

Expected: `All critical imports OK`

- [ ] **Step 4: Create directories**

```bash
ssh russellrudd@100.120.114.84 bash -l <<'REMOTE'
mkdir -p ~/Projects/alphatemp/alphatemp/logs
mkdir -p ~/Projects/alphatemp/alphatemp/data
REMOTE
```

---

### Task 7: Transfer `.env` and DuckDB

**Why:** These files aren't in git. SCP the env vars, rsync the 3.1 GB database.

- [ ] **Step 1: Transfer .env**

Run from MacBook:
```bash
scp ~/Projects/alphatemp/alphatemp/.env russellrudd@100.120.114.84:~/Projects/alphatemp/alphatemp/.env
```

- [ ] **Step 2: Transfer DuckDB (3.1 GB)**

Run from MacBook:
```bash
rsync -avP ~/Projects/alphatemp/alphatemp/data/alphatemp.duckdb \
  russellrudd@100.120.114.84:~/Projects/alphatemp/alphatemp/data/
```

Expected: ~3.1 GB transfer. Resumable if interrupted.

- [ ] **Step 3: Verify DuckDB on Mini**

```bash
ssh russellrudd@100.120.114.84 bash -l <<'REMOTE'
cd ~/Projects/alphatemp/alphatemp
venv/bin/python -c "
import duckdb
con = duckdb.connect('data/alphatemp.duckdb', read_only=True)
row_count = con.execute('SELECT COUNT(*) FROM forecasts').fetchone()[0]
latest = con.execute('SELECT MAX(model_run) FROM forecasts').fetchone()[0]
con.close()
print(f'Forecasts: {row_count:,} rows, latest: {latest}')
print('DuckDB OK')
"
REMOTE
```

Expected: row count and latest model run matching MacBook values.

---

### Task 8: Install and start launchd service

**Why:** launchd keeps the dashboard running permanently — auto-starts on boot, restarts on crash.

- [ ] **Step 1: Install the plist**

```bash
ssh russellrudd@100.120.114.84 bash -l <<'REMOTE'
cd ~/Projects/alphatemp/alphatemp
cp com.alphatemp.dashboard.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.alphatemp.dashboard.plist
echo "==> Service installed"
REMOTE
```

- [ ] **Step 2: Verify it's running**

```bash
ssh russellrudd@100.120.114.84 bash -l <<'REMOTE'
sleep 3
launchctl print gui/$(id -u)/com.alphatemp.dashboard 2>&1 | head -5
curl -sf http://localhost:8050 > /dev/null && echo "Dashboard UP" || echo "Dashboard DOWN"
REMOTE
```

Expected: service state = running, "Dashboard UP"

- [ ] **Step 3: Check from MacBook via Tailscale**

Run from MacBook:
```bash
curl -sf http://100.120.114.84:8050 > /dev/null && echo "Tailscale access OK" || echo "Tailscale access FAILED"
```

Expected: `Tailscale access OK`

---

### Task 9: Test deploy script end-to-end

**Why:** Verify the deploy script works before relying on it.

- [ ] **Step 1: Run deploy.sh from MacBook**

```bash
./scripts/deploy.sh
```

Expected output:
```
==> Deploying alphatemp to Mac Mini...
==> Pulling latest...
Already up to date.
==> Requirements unchanged, skipping pip install
==> Restarting service...
==> Waiting for startup...
==> Health check PASSED — dashboard is up
==> Deploy complete!
```

- [ ] **Step 2: Verify dashboard is accessible**

Open `http://100.120.114.84:8050` in browser from MacBook.

---

### Task 10: Set up Cloudflare Tunnel

**Why:** Public HTTPS access to the dashboard. Domain TBD — set up the tunnel infrastructure now.

- [ ] **Step 1: Install cloudflared on Mini**

```bash
ssh russellrudd@100.120.114.84 bash -l <<'REMOTE'
/opt/homebrew/bin/brew install cloudflared
cloudflared --version
REMOTE
```

- [ ] **Step 2: Authenticate with Cloudflare**

This opens a browser on the Mini. Russell needs to be at the Mini or use screen sharing.

```bash
ssh russellrudd@100.120.114.84 bash -l -c "cloudflared tunnel login"
```

Russell: complete the browser auth flow. This saves credentials to `~/.cloudflared/`.

- [ ] **Step 3: Create the tunnel**

```bash
ssh russellrudd@100.120.114.84 bash -l <<'REMOTE'
cloudflared tunnel create alphatemp
echo "==> Save the tunnel ID printed above"
REMOTE
```

- [ ] **Step 4: Write config file**

Russell provides tunnel ID from Step 3. Paste it into the command below:

```bash
# Replace <TUNNEL_ID> with the actual ID from Step 3 before running
TUNNEL_ID="<TUNNEL_ID>"

ssh russellrudd@100.120.114.84 bash -l <<REMOTE
cat > ~/.cloudflared/config.yml <<EOF
tunnel: $TUNNEL_ID
credentials-file: /Users/russellrudd/.cloudflared/$TUNNEL_ID.json

ingress:
  - service: http://localhost:8050
  - service: http_status:404
EOF
echo "==> Config written"
cat ~/.cloudflared/config.yml
REMOTE
```

Note: no hostname in ingress yet — the catch-all routes all traffic to the dashboard. Hostname will be added when domain is purchased. The `http_status:404` rule is required by cloudflared as a terminal catch-all.

- [ ] **Step 5: Test tunnel manually**

```bash
ssh russellrudd@100.120.114.84 bash -l -c "cloudflared tunnel run alphatemp"
```

Expected: tunnel connects, shows "Registered tunnel connection" messages. Ctrl+C to stop.

- [ ] **Step 6: Install as launchd service**

```bash
ssh russellrudd@100.120.114.84 bash -l -c "cloudflared service install"
```

- [ ] **Step 7: Verify tunnel service is running**

```bash
ssh russellrudd@100.120.114.84 bash -l <<'REMOTE'
launchctl list | grep cloudflared
cloudflared tunnel info alphatemp
REMOTE
```

Expected: cloudflared process listed, tunnel info shows active connections.

---

### Task 11: Final verification and commit

- [ ] **Step 1: Verify everything from MacBook**

```bash
# Tailscale access
curl -sf http://100.120.114.84:8050 > /dev/null && echo "Tailscale: OK" || echo "Tailscale: FAIL"

# Deploy script
./scripts/deploy.sh

# Services on Mini
ssh russellrudd@100.120.114.84 bash -l <<'REMOTE'
launchctl print gui/$(id -u)/com.alphatemp.dashboard 2>&1 | head -3
launchctl list | grep cloudflared
echo "==> All services checked"
REMOTE
```

- [ ] **Step 2: Commit any remaining changes**

```bash
git add -A
git commit -m "feat: complete Mac Mini deployment setup"
git push origin autoresearch/run-2026-03-10
```

- [ ] **Step 3: Update CLAUDE.md deployment section**

In `CLAUDE.md`, replace the `## Current Architecture` deployment line:
```
- **Deployment:** Docker on DigitalOcean droplet via `deploy.sh`
```
with:
```
- **Deployment:** Mac Mini (M4 Pro) via launchd + `scripts/deploy.sh`. Cloudflare Tunnel for public HTTPS.
```

- [ ] **Step 4: Update handoff**

Update `HANDOFF.md` with:
- Mini is the production box
- Deploy via `./scripts/deploy.sh`
- Cloudflare Tunnel set up (domain TBD)
- Tailscale access at `100.120.114.84:8050`
