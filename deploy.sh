#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

usage() {
    echo "Usage: $0 [--remote USER@HOST]"
    echo ""
    echo "  No flags     Local deployment (docker compose up)"
    echo "  --remote     Deploy to a remote VPS via rsync + SSH"
    echo ""
    echo "Examples:"
    echo "  $0                              # local deploy"
    echo "  $0 --remote root@167.99.42.10   # push to VPS"
    exit 1
}

REMOTE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --remote) REMOTE="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Validate .env
if [ ! -f .env ]; then
    echo "ERROR: .env file not found"; exit 1
fi
source .env
if [ -z "${SYNOPTIC_TOKEN:-}" ]; then
    echo "ERROR: SYNOPTIC_TOKEN not set in .env"; exit 1
fi

if [ -n "$REMOTE" ]; then
    # --- Remote deployment ---
    echo "=== AlphaTemp Remote Deployment ==="
    echo "Target: $REMOTE"

    REMOTE_DIR="/opt/alphatemp"

    echo "Syncing project files..."
    rsync -avz --delete \
        --exclude '.git' \
        --exclude '.env' \
        --exclude '.pytest_cache' \
        --exclude '__pycache__' \
        --exclude 'venv/' \
        --exclude 'data/' \
        --exclude '.claude/' \
        --exclude '*.pyc' \
        --exclude '*.duckdb' \
        --exclude '*.duckdb.wal' \
        ./ "$REMOTE:$REMOTE_DIR/"

    echo "Copying .env to remote..."
    scp .env "$REMOTE:$REMOTE_DIR/.env"

    echo "Building and starting on remote..."
    ssh "$REMOTE" "cd $REMOTE_DIR && mkdir -p data && docker compose build && docker compose up -d"

    echo ""
    echo "=== Remote deployment complete ==="
    echo "Dashboard: http://${REMOTE#*@}:8050"
    echo "Health:    http://${REMOTE#*@}:8050/api/health"
    echo "Logs:      ssh $REMOTE 'cd $REMOTE_DIR && docker compose logs -f'"
else
    # --- Local deployment ---
    echo "=== AlphaTemp Local Deployment ==="

    mkdir -p data

    echo "Building image..."
    docker compose build

    echo "Starting in detached mode..."
    docker compose up -d

    echo ""
    echo "=== Deployment complete ==="
    echo "Dashboard: http://localhost:8050"
    echo "Health:    http://localhost:8050/api/health"
    echo "Logs:      docker compose logs -f"
fi
