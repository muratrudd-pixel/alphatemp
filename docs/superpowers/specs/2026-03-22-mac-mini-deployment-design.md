# Mac Mini Deployment Design

**Date:** 2026-03-22
**Status:** Approved

## Goal

Move alphatemp from Russell's MacBook to the Mac Mini as a permanent, always-on production box. Full system: dashboard, data ingestion, strategy engine. Accessible privately via Tailscale and publicly via Cloudflare Tunnel.

## Target Environment

- **Machine:** Mac Mini M4 Pro, 24GB RAM, 811GB free
- **OS:** macOS 26.2
- **Python:** 3.12.13 (Homebrew)
- **Tailscale IP:** 100.120.114.84
- **System deps installed:** eccodes, proj, python@3.12

## 1. Mini Bootstrap (One-Time)

### 1.1 Clone Repository

```bash
mkdir -p ~/Projects/alphatemp
git clone https://github.com/muratrudd-pixel/alphatemp.git ~/Projects/alphatemp/alphatemp
cd ~/Projects/alphatemp/alphatemp
git checkout autoresearch/run-2026-03-10
```

**Note:** `autoresearch/run-2026-03-10` is the current working branch with all strategy engine + dashboard changes. Merge to `main` when stabilized.

### 1.2 Python Environment

The codebase targets Python 3.9 compatibility but runs on 3.12. Verify deps install cleanly.

```bash
/opt/homebrew/bin/python3.12 -m venv venv
venv/bin/pip install -r requirements.txt
```

**Smoke test after install:**
```bash
venv/bin/python -c "import duckdb, scipy, xgboost, cfgrib; print('All critical imports OK')"
```

If any package fails to build wheels on 3.12, install `python@3.9` via Homebrew and rebuild the venv with that instead.

### 1.3 Environment Variables

SCP `.env` from MacBook — never committed to git.

```bash
# From MacBook:
scp ~/Projects/alphatemp/alphatemp/.env russellrudd@100.120.114.84:~/Projects/alphatemp/alphatemp/.env
```

Set up direnv:
```bash
# On Mini:
brew install direnv
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
echo 'dotenv' > .envrc
direnv allow
```

### 1.4 DuckDB Transfer

3.1 GB file — use rsync for resume support. Create `data/` directory first.

```bash
# On Mini:
mkdir -p ~/Projects/alphatemp/alphatemp/data

# From MacBook:
rsync -avP ~/Projects/alphatemp/alphatemp/data/alphatemp.duckdb \
  russellrudd@100.120.114.84:~/Projects/alphatemp/alphatemp/data/
```

**Post-transfer verification** (on Mini):
```bash
venv/bin/python -c "
import duckdb
con = duckdb.connect('data/alphatemp.duckdb', read_only=True)
print(con.execute('SELECT COUNT(*) FROM forecasts').fetchone())
print(con.execute('SELECT MAX(model_run) FROM forecasts').fetchone())
con.close()
print('DuckDB OK')
"
```

### 1.5 Logs Directory

```bash
mkdir -p ~/Projects/alphatemp/alphatemp/logs
```

## 2. Launchd Service

Install `com.alphatemp.dashboard.plist` to `~/Library/LaunchAgents/`.

The plist runs `scripts/run-dashboard.sh` which:
1. `cd` to project directory
2. Executes `venv/bin/python main.py --dashboard`

The shell script does NOT `source .env` — env vars are loaded by `python-dotenv` inside the application via `load_dotenv()`. This avoids shell escaping issues with API keys.

Configuration:
- `KeepAlive: true` — auto-restarts on crash
- `RunAtLoad: true` — starts on login
- Logs to `logs/dashboard.log`

Path in plist must match Mini's home directory (`/Users/russellrudd`).

```bash
cp com.alphatemp.dashboard.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.alphatemp.dashboard.plist
```

To stop:
```bash
launchctl bootout gui/$(id -u)/com.alphatemp.dashboard
```

To restart:
```bash
launchctl kickstart -k gui/$(id -u)/com.alphatemp.dashboard
```

## 3. Deploy Script

`scripts/deploy.sh` — to be created. Runs from MacBook, one command deploys to Mini.

```
./scripts/deploy.sh
```

Steps:
1. SSH into Mini via Tailscale IP (`100.120.114.84`)
2. `git pull` in project directory
3. `venv/bin/pip install -r requirements.txt` (only if requirements.txt changed)
4. Restart launchd service via `launchctl kickstart -k gui/$(id -u)/com.alphatemp.dashboard`
5. Wait 3 seconds, health check `curl localhost:8050`
6. Print success/failure status

The script uses a single SSH connection with multiplexed commands to minimize latency.

## 4. Cloudflare Tunnel

### 4.1 Install

```bash
brew install cloudflared
```

### 4.2 Authenticate and Create Tunnel

```bash
cloudflared tunnel login          # Opens browser for Cloudflare auth
cloudflared tunnel create alphatemp
```

### 4.3 Configuration

`~/.cloudflared/config.yml` on Mini:

```yaml
tunnel: <TUNNEL_ID>
credentials-file: /Users/russellrudd/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: <DOMAIN_TBD>
    service: http://localhost:8050
  - service: http_status:404
```

### 4.4 Launchd Service for Tunnel

```bash
cloudflared service install
```

This creates a launchd plist that auto-starts the tunnel on boot and reconnects on failure.

**Verify tunnel is running:**
```bash
launchctl list | grep cloudflared
cloudflared tunnel info alphatemp
```

### 4.5 DNS (Deferred)

Once Russell purchases a domain and adds it to Cloudflare:

```bash
cloudflared tunnel route dns alphatemp <domain>
```

Update `config.yml` hostname and restart tunnel.

## 5. Access

| Method | URL | Auth |
|--------|-----|------|
| Private (Tailscale) | `http://100.120.114.84:8050` | Tailscale network only |
| Public (Cloudflare) | `https://<domain>` | Open (add Cloudflare Access later if needed) |

## 6. File Inventory

New files to create:
- `scripts/deploy.sh` — deploy script (runs from MacBook)

Files to modify:
- `com.alphatemp.dashboard.plist` — verify paths match Mini
- `scripts/run-dashboard.sh` — remove `source .env`, rely on python-dotenv

Files to transfer (not in git):
- `.env` — environment variables
- `data/alphatemp.duckdb` — database (3.1 GB)

## 7. Out of Scope

- Auto-deploy on git push (manual deploy script is sufficient)
- Database syncing between MacBook and Mini (one-time transfer)
- Authentication/access control on dashboard (can add Cloudflare Access later)
- Custom domain purchase (deferred)
