# AlphaTemp Autoresearch

## ⛔ MANDATORY DIRECTIVE — READ THIS FIRST ⛔
The LP quantile regression model has been fully exhausted (22+ features, lambda tuning, hyperparameter sweeps). DO NOT make any more changes to the LP model. No more features, no more lambda tuning, no more MIN_SAMPLES tweaks, no more radius changes. The LP ceiling has been found.

YOUR NEXT EXPERIMENT MUST replace the LP solver with XGBoost quantile regression. See section 1 in the Priority Queue below. If you are about to commit a change that still uses `linprog` or `fit_quantile_regression`, STOP and rewrite it to use XGBoost instead.

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

## Priority Experiment Queue

Try these in order. Each is a major direction — run 3-5 variants before moving on.

**⚠️ STOP ADDING FEATURES TO THE LP MODEL. The LP quantile regression ceiling has been reached with 22 features. Move to XGBoost NOW. Do not add more features, tune lambdas, or tweak the LP solver. The next experiment MUST replace the model architecture.**

### 1. XGBoost / Gradient Boosted Trees (DO THIS NOW)
Replace the LP-based quantile regression with XGBoost quantile regression.
- `pip install xgboost` is available in the venv
- Use `objective='reg:quantile'` with `quantile_alpha` parameter
- Train one model per quantile (same as current QR approach)
- XGBoost handles nonlinear feature interactions automatically
- Try both with current features AND with raw features (let XGBoost find interactions)
- Careful: XGBoost can overfit on 180 days of data. Use early stopping or regularization.

### 2. Random Forest Quantile Regression
Use sklearn's RandomForestRegressor to predict quantiles.
- For each quantile, train on the same (X, y) data
- At prediction time, use individual tree predictions to build empirical CDF
- `from sklearn.ensemble import RandomForestRegressor`
- Key advantage: naturally handles nonlinear relationships
- Key risk: slow if n_estimators is too high (stay under 100)

### 3. Hour-Specific Models
Train separate coefficient sets for different update hours instead of pooling.
- Early hours (0-8 ET): less obs data, rely more on forecast
- Late hours (12-20 ET): more obs data, divergence signals matter more
- Could do 2-3 groups instead of 6 individual models (more training data per group)

### 4. Bracket-Aware Optimization
Current model optimizes for 1°F accuracy but Kalshi brackets are 2°F wide.
- Adjust CDF integration to match 2°F bracket boundaries
- Or: directly optimize for Kalshi bracket Brier instead of 1°F Brier
- Small change but could meaningfully improve Brier and P&L

### 5. Seasonal Stratification
Train separate models for warm season (May-Sep) vs cold season (Oct-Apr).
- Temperature dynamics differ: summer has higher variance, stronger diurnal cycle
- Bias patterns are seasonal (HRRR warm bias stronger in summer)
- Could also try 4 seasons or month-based weighting

### 6. Ensemble: QR + XGBoost + Climatology
Blend predictions from multiple model architectures.
- Weight by recent performance (exponential decay)
- Ensembles almost always beat individual models
- Try after implementing XGBoost separately first

## Strategy Tips
- Small changes first. One variable at a time.
- If 3 consecutive experiments in one direction fail, pivot completely.
- The current model uses quantile regression — but you can try anything.
- Seasonal patterns matter — summer and winter behave very differently.
- More features isn't always better. Regularization helps.
- The forecast_extended table has rich data (dewpoint, wind, cloud cover) — much of it now used.
- Consider time-of-day effects: model accuracy varies by hour.
- Don't forget about the physical floor: daily high can't go below running_max.
- P&L now uses displacement-based betting: only bets when model Brier < 80% of uniform.
  Sharper, better-calibrated distributions will place more winning bets.

## Scoring
Composite (lower = better):
- 45% Net P&L (displacement-based, fee-adjusted, inverted + normalized)
- 35% Brier score (full day, all update hours)
- 20% Hit rate (top-1 bracket accuracy, inverted + normalized)

P&L only counts events where model shows genuine edge (Brier < 80% of uniform).
This rewards models that produce sharp, confident, and accurate distributions.

Run time budget: 90 seconds max per experiment.
