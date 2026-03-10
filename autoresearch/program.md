# AlphaTemp Autoresearch

## Objective
Lower the composite score by improving the probability model in experiment.py.
Lower score = better. You're optimizing a weather prediction model that
predicts NYC daily high temperature brackets for Kalshi event markets.

## The Loop
1. Read experiment.py and understand the current model
2. Form a hypothesis (e.g., "adding wind speed as a feature should help")
3. Modify ONLY experiment.py — keep the model_fn signature unchanged
4. Run: PYTHONPATH=. ../venv/bin/python autoresearch/run_experiment.py
5. If exit code 0 → improvement! Run:
   git add autoresearch/experiment.py && git commit -m "experiment: [DESCRIPTION] — score X.XXXX (was Y.YYYY)"
6. If exit code 1 → discard. Run: git checkout -- autoresearch/experiment.py
7. Repeat. Try a different hypothesis.

## What You Can Change (ONLY in experiment.py)
- Feature engineering (add/remove/transform features)
- Model architecture (QR, XGBoost, ridge, ensemble, neural net, anything)
- Hyperparameters (window size, quantile positions, regularization)
- CDF construction (tail treatment, interpolation method)
- Training strategy (rolling window, expanding window, seasonal weighting)

## What You CANNOT Change
- The model_fn(provider, ref_time) → Optional[Dict[int, float]] signature
- Any file outside autoresearch/experiment.py
- Walk-forward validation — training data must be BEFORE eval date (no future leakage)
- Kalshi fee math — the fee formula is fixed reality
- Settlement source — NWS CLI from KNYC, non-negotiable
- Database schema — no adding/dropping tables
- Must remain Python 3.9 compatible (no match/case, no dict | dict, use typing.Dict not dict[])

## Available Data (via BacktestDataProvider)
The provider gives you access to a DuckDB connection (`provider._shared_con`) and metadata:
- `provider.station_id` — "KNYC"
- `provider.model_run` — datetime of the HRRR run being evaluated
- `provider.get_forecast_high(station_id)` — MAX(temp_f) for this run
- `provider._shared_con` — DuckDB connection for direct queries

### Database Tables You Can Query
- `forecasts` — hourly temps (station_id, model_run, valid_at, temp_f, model_name, fxx)
  - model_name: 'hrrr' (all hours), 'gfs' (00z ONLY — 06z/12z/18z contaminated), 'ecmwf'
  - Settlement day filter: (EXTRACT(HOUR FROM model_run) + fxx) >= 5 AND < 29
- `observations` — METAR obs (station_id, observed_at, temp_f) for KNYC, KLGA, KEWR
- `nws_daily` — settlement truth (station_id, obs_date, max_temp_f, source)
- `forecast_extended` — dewpoint, humidity, wind, pressure, cloud, precip, radiation
- `mesonet_obs` — 5-min NYC stations (station_id, observed_at, temp_f) for BKNYRD, QNASTO
- `kalshi_settlements` — bracket definitions (event_date, floor_strike, cap_strike, settled_yes)

### Key Domain Facts
- All timestamps are UTC unless explicitly stated
- KNYC reports hourly at ~:51 past the hour
- Settlement uses NWS CLI Local Standard Time (EST = UTC-5, NO daylight saving)
- HRRR is 3km resolution, runs every hour, 18h forecast horizon
- GFS 06z/12z/18z data is POISONED (Mediterranean weather). Only use 00z.
- Temperature brackets on Kalshi are 2°F wide

## Strategy Tips
- Small changes first. One variable at a time.
- If 3 consecutive experiments in one direction fail, pivot completely.
- The current model uses quantile regression — but you can try anything.
- Seasonal patterns matter — summer and winter behave very differently.
- More features isn't always better. Regularization helps.
- The forecast_extended table has rich data (dewpoint, wind, cloud cover) that's currently unused.
- Consider time-of-day effects: model accuracy varies by hour.
- Don't forget about the physical floor: daily high can't go below running_max.

## Scoring
Composite (lower = better):
- 45% Net P&L (fee-adjusted, inverted + normalized)
- 35% Brier score (full day, all update hours)
- 20% Hit rate (top-1 bracket accuracy, inverted + normalized)

Run time budget: 90 seconds max per experiment.
