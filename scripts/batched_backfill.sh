#!/bin/bash
# Batched HRRR + GFS backfill — memory-safe for 8 GB EC2 instances.
# Runs HRRR in batches of N concurrent processes, then GFS one at a time.
# All Python scripts have resume support, so safe to restart.
#
# Usage:
#   ./scripts/batched_backfill.sh --start 2021-06-01 --end 2021-12-31
#   ./scripts/batched_backfill.sh --start 2021-06-01 --end 2021-12-31 --hrrr-only
#   ./scripts/batched_backfill.sh --start 2021-06-01 --end 2021-12-31 --gfs-only
#   ./scripts/batched_backfill.sh --status
#
# Concurrency: HRRR_BATCH_SIZE (default 6), GFS runs 1 at a time.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

HRRR_BATCH_SIZE="${HRRR_BATCH_SIZE:-4}"
DELAY="${HRRR_DELAY:-0.1}"
START_DATE=""
END_DATE=""
HRRR_ONLY=false
GFS_ONLY=false

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start) START_DATE="$2"; shift 2 ;;
        --end) END_DATE="$2"; shift 2 ;;
        --hrrr-only) HRRR_ONLY=true; shift ;;
        --gfs-only) GFS_ONLY=true; shift ;;
        --status)
            echo "=== Backfill Status ==="
            echo ""
            echo "--- HRRR ---"
            for hour in $(seq 0 23); do
                hz=$(printf "%02dz" "$hour")
                log="logs/hrrr_${hz}.log"
                if [ -f "$log" ]; then
                    last=$(grep -oP 'Day \d+/\d+ \(\s*\d+%\).*' "$log" 2>/dev/null | tail -1)
                    if [ -n "$last" ]; then
                        echo "  ${hz}: $last"
                    else
                        echo "  ${hz}: log exists (no progress line)"
                    fi
                else
                    echo "  ${hz}: not started"
                fi
            done
            echo ""
            echo "--- GFS ---"
            for rh in 6 12 18; do
                log="logs/gfs_$(printf '%02d' $rh)z.log"
                if [ -f "$log" ]; then
                    last=$(grep -oP 'Day \d+/\d+ \(\s*\d+%\).*' "$log" 2>/dev/null | tail -1)
                    if [ -n "$last" ]; then
                        echo "  $(printf '%02d' $rh)z: $last"
                    else
                        echo "  $(printf '%02d' $rh)z: log exists (no progress line)"
                    fi
                else
                    echo "  $(printf '%02d' $rh)z: not started"
                fi
            done
            echo ""
            hrrr_running=$(pgrep -f "backfill_hrrr_full" 2>/dev/null | wc -l | tr -d ' ')
            gfs_running=$(pgrep -f "backfill_gfs_ucar" 2>/dev/null | wc -l | tr -d ' ')
            echo "Running: HRRR=$hrrr_running GFS=$gfs_running"
            free -h | head -2
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$START_DATE" ] || [ -z "$END_DATE" ]; then
    echo "Usage: $0 --start YYYY-MM-DD --end YYYY-MM-DD [--hrrr-only|--gfs-only]"
    exit 1
fi

mkdir -p logs data

# Kill any existing backfill processes
echo "Stopping existing backfill processes..."
pkill -f "backfill_hrrr_full" 2>/dev/null || true
pkill -f "backfill_gfs_ucar" 2>/dev/null || true
sleep 2

echo "=== Batched Backfill ==="
echo "Date range: $START_DATE to $END_DATE"
echo "HRRR batch size: $HRRR_BATCH_SIZE"
echo "Delay: ${DELAY}s"
echo ""

run_hrrr_batch() {
    local batch_hours=("$@")
    local pids=()

    echo "[$(date '+%H:%M:%S')] Launching HRRR batch: ${batch_hours[*]}"

    for hour in "${batch_hours[@]}"; do
        hz=$(printf "%02dz" "$hour")
        log="logs/hrrr_${hz}.log"

        PYTHONPATH="$PROJECT_DIR" nohup python3 scripts/backfill_hrrr_full.py \
            --run-hour "$hour" \
            --start "$START_DATE" \
            --end "$END_DATE" \
            --delay "$DELAY" \
            >> "$log" 2>&1 &

        pids+=($!)
        sleep 1  # Stagger slightly
    done

    echo "[$(date '+%H:%M:%S')] Waiting for batch (PIDs: ${pids[*]})..."

    # Wait for all processes in this batch
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid" 2>/dev/null; then
            failed=$((failed + 1))
        fi
    done

    if [ "$failed" -gt 0 ]; then
        echo "[$(date '+%H:%M:%S')] WARNING: $failed process(es) exited non-zero in this batch"
    else
        echo "[$(date '+%H:%M:%S')] Batch complete (all OK)"
    fi
}

run_gfs_sequential() {
    for rh in 6 12 18; do
        local rhz=$(printf "%02dz" "$rh")
        local log="logs/gfs_${rhz}.log"

        echo "[$(date '+%H:%M:%S')] Starting GFS ${rhz}..."

        PYTHONPATH="$PROJECT_DIR" python3 scripts/backfill_gfs_ucar.py \
            --run-hour "$rh" \
            --start "$START_DATE" \
            --end "$END_DATE" \
            --delay "$DELAY" \
            >> "$log" 2>&1

        echo "[$(date '+%H:%M:%S')] GFS ${rhz} complete"
    done
}

# --- HRRR ---
if [ "$GFS_ONLY" = false ]; then
    echo "=== HRRR Backfill (batches of $HRRR_BATCH_SIZE) ==="

    # Build array of all 24 hours
    ALL_HOURS=($(seq 0 23))

    # Process in batches
    for ((i=0; i<${#ALL_HOURS[@]}; i+=HRRR_BATCH_SIZE)); do
        batch=("${ALL_HOURS[@]:i:HRRR_BATCH_SIZE}")
        echo ""
        echo "--- Batch $((i/HRRR_BATCH_SIZE + 1)) of $(( (${#ALL_HOURS[@]} + HRRR_BATCH_SIZE - 1) / HRRR_BATCH_SIZE )) ---"
        run_hrrr_batch "${batch[@]}"
    done

    echo ""
    echo "=== All HRRR batches complete ==="
    echo ""
fi

# --- GFS ---
if [ "$HRRR_ONLY" = false ]; then
    echo "=== GFS Backfill (sequential, 1 at a time) ==="
    run_gfs_sequential
    echo ""
    echo "=== All GFS complete ==="
fi

echo ""
echo "=== All backfills complete ==="
echo "Run --status to verify, then download DuckDB files."
