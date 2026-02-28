#!/bin/bash
# Fleet Harvest — download, merge, gate-check, and optionally terminate EC2 fleet.
#
# Steps:
#   1. Check all instances for completion (no python3 processes)
#   2. Download all backfill_*.duckdb files from each instance
#   3. Merge each into the main DB
#   4. Run phase1 gate check
#   5. Optionally terminate instances and clean up AWS
#
# Usage:
#   ./scripts/fleet_harvest.sh                    # Full pipeline
#   ./scripts/fleet_harvest.sh --check-only       # Just check if fleet is done
#   ./scripts/fleet_harvest.sh --download-only    # Download without merging
#   ./scripts/fleet_harvest.sh --skip-terminate   # Don't terminate instances

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

KEY="$HOME/.ssh/alphatemp-hrrr.pem"
DOWNLOAD_DIR="data/ec2_harvest"
MAIN_DB="data/alphatemp.duckdb"

# Fleet config
IPS=(
    44.204.17.140
    3.80.188.43
    3.83.140.131
    3.83.173.71
    3.84.4.112
    32.192.64.214
    13.218.60.69
    54.144.18.212
)

INSTANCE_IDS=(
    i-0cb60b4e2b3297fa8
    i-0e6a0e80fd68a0112
    i-09e2a3d495e9f7e37
    i-0a510a018c58df2cf
    i-05349669e69290a75
    i-0c59a6e8826d05e43
    i-017516e1962d87bce
    i-0ba9ac13ea6b5837b
)

CHECK_ONLY=false
DOWNLOAD_ONLY=false
SKIP_TERMINATE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check-only) CHECK_ONLY=true; shift ;;
        --download-only) DOWNLOAD_ONLY=true; shift ;;
        --skip-terminate) SKIP_TERMINATE=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ============================================================
# Step 1: Check fleet completion
# ============================================================
check_fleet() {
    echo "=== Step 1: Checking Fleet Status ==="
    local all_done=true

    for ip in "${IPS[@]}"; do
        procs=$(ssh -i "$KEY" -o ConnectTimeout=5 -o StrictHostKeyChecking=no ec2-user@"$ip" \
            "pgrep -c python3 2>/dev/null || echo 0" 2>/dev/null || echo "UNREACHABLE")

        if [ "$procs" = "UNREACHABLE" ]; then
            echo "  $ip: UNREACHABLE"
            all_done=false
        elif [ "$procs" -gt 0 ]; then
            echo "  $ip: STILL RUNNING ($procs processes)"
            all_done=false
        else
            echo "  $ip: DONE"
        fi
    done

    if [ "$all_done" = true ]; then
        echo ""
        echo "All instances complete."
        return 0
    else
        echo ""
        echo "Some instances still running. Run with --check-only to monitor."
        return 1
    fi
}

if ! check_fleet; then
    if [ "$CHECK_ONLY" = true ]; then
        exit 0
    fi
    echo ""
    read -p "Some instances still running. Continue downloading anyway? [y/N] " answer
    if [[ ! "$answer" =~ ^[Yy] ]]; then
        echo "Aborting. Re-run when all instances are done."
        exit 0
    fi
fi

if [ "$CHECK_ONLY" = true ]; then
    exit 0
fi

# ============================================================
# Step 2: Download DuckDB files from all instances
# ============================================================
echo ""
echo "=== Step 2: Downloading DuckDB Files ==="
mkdir -p "$DOWNLOAD_DIR"

total_files=0
for i in "${!IPS[@]}"; do
    ip="${IPS[$i]}"
    instance_dir="$DOWNLOAD_DIR/instance_$((i+1))_${ip//./_}"
    mkdir -p "$instance_dir"

    echo ""
    echo "--- $ip ---"

    # List available DuckDB files on the instance
    files=$(ssh -i "$KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no ec2-user@"$ip" \
        "ls ~/alphatemp/data/backfill_*.duckdb 2>/dev/null" 2>/dev/null || echo "")

    if [ -z "$files" ]; then
        echo "  No backfill files found — skipping"
        continue
    fi

    file_count=$(echo "$files" | wc -l | tr -d ' ')
    echo "  Found $file_count DuckDB files"

    # Download each file
    for remote_file in $files; do
        basename=$(basename "$remote_file")
        local_path="$instance_dir/$basename"

        if [ -f "$local_path" ]; then
            echo "  $basename: already downloaded, skipping"
            continue
        fi

        echo -n "  $basename: downloading... "
        if scp -i "$KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
            ec2-user@"$ip":"$remote_file" "$local_path" 2>/dev/null; then
            size=$(ls -lh "$local_path" | awk '{print $5}')
            echo "OK ($size)"
            total_files=$((total_files + 1))
        else
            echo "FAILED"
        fi
    done
done

echo ""
echo "Downloaded $total_files files to $DOWNLOAD_DIR/"

if [ "$DOWNLOAD_ONLY" = true ]; then
    echo "Download complete. Run without --download-only to merge."
    exit 0
fi

# ============================================================
# Step 3: Merge all temp DBs into main
# ============================================================
echo ""
echo "=== Step 3: Merging Into Main DB ==="

# Activate venv for Python scripts
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

merge_count=0
merge_errors=0

# Find all downloaded DuckDB files
for db_file in $(find "$DOWNLOAD_DIR" -name "backfill_*.duckdb" -type f | sort); do
    basename=$(basename "$db_file")
    echo -n "  Merging $basename... "

    if PYTHONPATH="$PROJECT_DIR" python3 scripts/backfill_merge.py "$db_file" --main-db "$MAIN_DB" 2>/dev/null; then
        echo "OK"
        merge_count=$((merge_count + 1))
    else
        echo "FAILED"
        merge_errors=$((merge_errors + 1))
    fi
done

echo ""
echo "Merged $merge_count files ($merge_errors errors)"

# ============================================================
# Step 4: Run gate check
# ============================================================
echo ""
echo "=== Step 4: Phase 1 Gate Check ==="
PYTHONPATH="$PROJECT_DIR" python3 scripts/phase1_gate_check.py --db "$MAIN_DB"
gate_result=$?

# ============================================================
# Step 5: Terminate instances (optional)
# ============================================================
if [ "$SKIP_TERMINATE" = true ]; then
    echo ""
    echo "Skipping termination (--skip-terminate)."
    echo "To terminate manually:"
    echo "  aws ec2 terminate-instances --instance-ids ${INSTANCE_IDS[*]}"
    exit $gate_result
fi

echo ""
echo "=== Step 5: Terminate EC2 Fleet ==="
echo "Instance IDs: ${INSTANCE_IDS[*]}"
read -p "Terminate all 8 instances? This cannot be undone. [y/N] " answer
if [[ ! "$answer" =~ ^[Yy] ]]; then
    echo "Skipped termination."
    echo "To terminate manually:"
    echo "  aws ec2 terminate-instances --instance-ids ${INSTANCE_IDS[*]}"
    exit $gate_result
fi

echo "Terminating instances..."
if aws ec2 terminate-instances \
    --instance-ids "${INSTANCE_IDS[@]}" \
    --query 'TerminatingInstances[].{ID:InstanceId,State:CurrentState.Name}' \
    --output table 2>/dev/null; then
    echo "All instances terminated."
else
    echo "WARNING: Termination command failed. Check AWS console."
fi

# Clean up security group (wait for instances to fully terminate)
echo ""
echo "Waiting 60s for instances to terminate before cleaning up security group..."
sleep 60
echo "Deleting security group sg-0729bc1135388b966..."
aws ec2 delete-security-group --group-id sg-0729bc1135388b966 2>/dev/null && echo "Deleted." || echo "Could not delete (may still have dependencies)."

echo ""
echo "=== Fleet Harvest Complete ==="
echo "Gate check result: $([ $gate_result -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
exit $gate_result
