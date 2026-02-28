#!/bin/bash
# Parallel HRRR backfill — launches all 24 run hours simultaneously.
# Each run hour writes to its own DuckDB file.
#
# Usage:
#   ./scripts/hrrr_parallel_backfill.sh              # All 24 hours
#   ./scripts/hrrr_parallel_backfill.sh --start 2022-01-01  # Custom start date
#
# Monitor:
#   tail -f logs/hrrr_*z.log
#   ./scripts/hrrr_parallel_backfill.sh --status      # Check progress

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Delay between requests — reduce on EC2 (co-located with S3)
DELAY="${HRRR_DELAY:-0.1}"
START_DATE="${1:-}"
EXTRA_ARGS=""

# Handle --status flag
if [ "${1:-}" = "--status" ]; then
    echo "=== HRRR Backfill Status ==="
    for hour in $(seq 0 23); do
        hz=$(printf "%02dz" "$hour")
        log="logs/hrrr_${hz}.log"
        if [ -f "$log" ]; then
            # Get last progress line
            last=$(grep -E "Day |complete" "$log" 2>/dev/null | tail -1)
            if [ -n "$last" ]; then
                echo "  ${hz}: $last"
            else
                echo "  ${hz}: running (no progress yet)"
            fi
        else
            echo "  ${hz}: not started"
        fi
    done
    # Count running processes
    running=$(pgrep -f "backfill_hrrr_full.py --run-hour" | wc -l | tr -d ' ')
    echo ""
    echo "Running processes: $running / 24"
    exit 0
fi

# Parse start date if provided
if [ -n "$START_DATE" ] && [ "$START_DATE" != "--start" ]; then
    EXTRA_ARGS="--start $START_DATE"
elif [ "${1:-}" = "--start" ] && [ -n "${2:-}" ]; then
    EXTRA_ARGS="--start $2"
fi

# Create logs directory
mkdir -p logs data

echo "=== HRRR Parallel Backfill ==="
echo "Delay: ${DELAY}s between requests"
echo "Extra args: ${EXTRA_ARGS:-none}"
echo ""

# Launch all 24 run hours in parallel
PIDS=()
for hour in $(seq 0 23); do
    hz=$(printf "%02dz" "$hour")
    log="logs/hrrr_${hz}.log"

    echo "Starting HRRR ${hz}..."
    PYTHONPATH="$PROJECT_DIR" nohup python3 scripts/backfill_hrrr_full.py \
        --run-hour "$hour" \
        --delay "$DELAY" \
        $EXTRA_ARGS \
        > "$log" 2>&1 &

    PIDS+=($!)
    # Stagger launches by 2 seconds to avoid thundering herd on S3
    sleep 2
done

echo ""
echo "All 24 processes launched."
echo "PIDs: ${PIDS[*]}"
echo ""
echo "Monitor with:"
echo "  ./scripts/hrrr_parallel_backfill.sh --status"
echo "  tail -f logs/hrrr_12z.log"
echo ""
echo "To stop all:"
echo "  pkill -f 'backfill_hrrr_full.py --run-hour'"
