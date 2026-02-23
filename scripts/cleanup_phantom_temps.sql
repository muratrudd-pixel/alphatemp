-- One-time cleanup: null out phantom temperatures from METAR stubs.
-- Stubs like "METAR KMIA AUTO" lack a Zulu timestamp (e.g. 230505Z)
-- and carry garbage air_temp_set_1 values from the Synoptic API.
--
-- This preserves observation rows (for gap analysis) but removes the bad temps.
--
-- Usage:
--   duckdb data/alphatemp.duckdb < scripts/cleanup_phantom_temps.sql

-- Preview affected rows first
SELECT station_id, observed_at, temp_f, raw_metar
FROM observations
WHERE raw_metar IS NOT NULL
  AND NOT regexp_matches(raw_metar, '\d{6}Z')
  AND temp_f IS NOT NULL
ORDER BY observed_at DESC
LIMIT 20;

-- Apply the fix
UPDATE observations
SET temp_f = NULL, temp_c_tenth = NULL
WHERE raw_metar IS NOT NULL
  AND NOT regexp_matches(raw_metar, '\d{6}Z')
  AND temp_f IS NOT NULL;
