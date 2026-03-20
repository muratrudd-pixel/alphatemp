#!/bin/bash
# EC2 Full-Resolution QR Run
# 1. Merge local backfill files into main DB
# 2. Run qr_multimodel at full resolution (24 run hours, 24 update hours)
#
# Usage: nohup bash scripts/ec2_run_qr_full.sh > logs/qr_full_run.log 2>&1 &

set -euo pipefail

cd /home/ec2-user/alphatemp
source .venv/bin/activate
export PYTHONPATH=/home/ec2-user/alphatemp

MAIN_DB="data/alphatemp.duckdb"
mkdir -p logs

echo "$(date) === Step 1: Merge local backfill files ==="

merge_count=0
for db_file in data/backfill_*.duckdb; do
    [ -f "$db_file" ] || continue
    basename=$(basename "$db_file")
    echo -n "  Merging $basename... "
    if python3 scripts/backfill_merge.py "$db_file" --main-db "$MAIN_DB" 2>/dev/null; then
        echo "OK"
        merge_count=$((merge_count + 1))
    else
        echo "FAILED"
    fi
done
echo "$(date) Merged $merge_count files"

echo ""
echo "$(date) === Step 2: Verify GFS coverage ==="
python3 -c "
import duckdb
con = duckdb.connect('$MAIN_DB', read_only=True)
r = con.execute('''
    SELECT model_name, EXTRACT(HOUR FROM model_run) as rh,
           COUNT(DISTINCT CAST(model_run AS DATE)) as days
    FROM forecasts
    WHERE model_name IN ('gfs','ecmwf')
    GROUP BY 1, 2 ORDER BY 1, 2
''').fetchall()
for row in r:
    print(f'  {row[0]} {int(row[1]):02d}z: {row[2]} days')
con.close()
"

echo ""
echo "$(date) === Step 3: Run full-resolution QR analysis ==="
echo "  Config: qr_multimodel + multimodel_full baseline"
echo "  Run hours: all 24"
echo "  Update hours: all 24"
echo "  Dates: all available"
echo ""

python3 scripts/phase3_qr_analysis.py 2>&1

echo ""
echo "$(date) === COMPLETE ==="
