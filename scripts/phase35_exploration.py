"""Phase 3.5: Signal exploration for extended weather variables.

Computes Pearson correlation and F-test significance of each extended
variable against forecast error, per model (GFS, ECMWF).

Gate: |r| > 0.05 AND p < 0.05 to survive.

Usage:
    PYTHONPATH=. python scripts/phase35_exploration.py
"""

import duckdb
import numpy as np
from scipy import stats
from loguru import logger

DB_PATH = "data/alphatemp.duckdb"
STATION_ID = "KNYC"

# Extended variable daily summaries to test
VARIABLE_SPECS = [
    ("mean_dewpoint", "AVG(fe.dewpoint_2m_f)"),
    ("mean_humidity", "AVG(fe.humidity_2m)"),
    ("max_wind", "MAX(fe.wind_speed_10m)"),
    ("mean_wind_dir", "AVG(fe.wind_dir_10m)"),
    ("max_gusts", "MAX(fe.wind_gusts_10m)"),
    ("mean_pressure", "AVG(fe.pressure_msl)"),
    ("mean_cloud", "AVG(fe.cloud_cover)"),
    ("total_precip", "COALESCE(SUM(fe.precipitation), 0)"),
    ("mean_radiation", "AVG(fe.shortwave_rad)"),
    ("mean_cape", "AVG(fe.cape)"),
    ("dewpoint_depression", "AVG(f.temp_f) - AVG(fe.dewpoint_2m_f)"),
]


def get_signal_data(con, model_name):
    # type: (duckdb.DuckDBPyConnection, str) -> list
    """Pull forecast error + daily summary stats for one model.

    Returns list of dicts with 'error' and each variable name as keys.
    """
    # Build SELECT columns for extended variables
    select_cols = ",\n        ".join(
        "{} AS {}".format(sql_expr, var_name)
        for var_name, sql_expr in VARIABLE_SPECS
    )

    # For GFS/ECMWF: run_hour=0 (midnight model_run)
    query = """
        WITH daily_data AS (
            SELECT
                n.obs_date,
                MAX(f.temp_f) - n.max_temp_f AS error,
                {select_cols}
            FROM nws_daily n
            JOIN forecasts f
                ON f.station_id = n.station_id
                AND f.model_run::DATE = n.obs_date
                AND EXTRACT(HOUR FROM f.model_run) = 0
                AND f.model_name = ?
            LEFT JOIN forecast_extended fe
                ON fe.station_id = f.station_id
                AND fe.model_run = f.model_run
                AND fe.valid_at = f.valid_at
                AND fe.model_name = f.model_name
            WHERE n.station_id = ?
                AND n.max_temp_f IS NOT NULL
            GROUP BY n.obs_date, n.max_temp_f
            HAVING COUNT(fe.dewpoint_2m_f) > 0
        )
        SELECT * FROM daily_data ORDER BY obs_date
    """.format(select_cols=select_cols)

    rows = con.execute(query, [model_name, STATION_ID]).fetchall()
    col_names = ["obs_date", "error"] + [v[0] for v in VARIABLE_SPECS]

    results = []
    for row in rows:
        d = {}
        for i, name in enumerate(col_names):
            d[name] = row[i]
        results.append(d)

    return results


def analyze_signals(data, model_name):
    # type: (list, str) -> list
    """Compute correlation and significance for each variable.

    Returns list of (var_name, correlation, p_value, pass_fail).
    """
    if not data:
        logger.warning("No data for model {}", model_name)
        return []

    errors = np.array([d["error"] for d in data], dtype=np.float64)

    results = []
    for var_name, _ in VARIABLE_SPECS:
        values = []
        valid_errors = []
        for d in data:
            v = d[var_name]
            if v is not None and not np.isnan(v):
                values.append(v)
                valid_errors.append(d["error"])

        if len(values) < 30:
            results.append((var_name, float("nan"), float("nan"), "SKIP (n<30)"))
            continue

        arr = np.array(values, dtype=np.float64)
        err = np.array(valid_errors, dtype=np.float64)

        # Check for zero variance
        if np.std(arr) < 1e-10:
            results.append((var_name, 0.0, 1.0, "FAIL (zero var)"))
            continue

        r, p = stats.pearsonr(arr, err)
        passed = abs(r) > 0.05 and p < 0.05
        results.append((var_name, r, p, "PASS" if passed else "FAIL"))

    return results


def main():
    con = duckdb.connect(DB_PATH, read_only=True)

    print("=" * 80)
    print("Phase 3.5: Extended Variable Signal Exploration")
    print("=" * 80)
    print()

    all_survivors = {}

    for model_name in ["gfs", "ecmwf"]:
        print("-" * 60)
        print("Model: {}".format(model_name.upper()))
        print("-" * 60)

        data = get_signal_data(con, model_name)
        print("  Days with extended data: {}".format(len(data)))
        print()

        results = analyze_signals(data, model_name)

        print("  {:<22s} {:>8s} {:>12s}  {:s}".format(
            "Variable", "r", "p-value", "Gate"))
        print("  " + "-" * 56)

        survivors = []
        for var_name, r, p, gate in results:
            if np.isnan(r):
                print("  {:<22s} {:>8s} {:>12s}  {:s}".format(
                    var_name, "n/a", "n/a", gate))
            else:
                print("  {:<22s} {:>8.4f} {:>12.6f}  {:s}".format(
                    var_name, r, p, gate))
            if gate == "PASS":
                survivors.append(var_name)

        all_survivors[model_name] = survivors
        print()
        print("  Survivors: {}".format(len(survivors)))
        if survivors:
            print("  -> {}".format(", ".join(survivors)))
        print()

    con.close()

    # Final gate
    print("=" * 80)
    total_survivors = sum(len(v) for v in all_survivors.values())
    if total_survivors == 0:
        print("GATE: FAIL -- No variables passed signal threshold for any model.")
        print("RECOMMENDATION: Skip Tasks 6-8. Extended variables have no signal.")
    else:
        print("GATE: PASS -- {} variables passed across models.".format(total_survivors))
        for m, surv in all_survivors.items():
            if surv:
                print("  {}: {}".format(m.upper(), ", ".join(surv)))
    print("=" * 80)


if __name__ == "__main__":
    main()
