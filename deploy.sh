#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== AlphaTemp Deployment ==="

# Validate .env
if [ ! -f .env ]; then
    echo "ERROR: .env file not found"; exit 1
fi
source .env
if [ -z "${SYNOPTIC_TOKEN:-}" ]; then
    echo "ERROR: SYNOPTIC_TOKEN not set in .env"; exit 1
fi

# Ensure data dir exists
mkdir -p data

# Build and launch
echo "Building image..."
docker compose build

echo "Starting in detached mode..."
docker compose up -d

echo ""
echo "=== Deployment complete ==="
echo "Dashboard: http://localhost:8050"
echo "Health:    http://localhost:8050/api/health"
echo "Logs:      docker compose logs -f"
