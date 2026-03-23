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
