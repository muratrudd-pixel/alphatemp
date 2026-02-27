---
name: backtesting-analysis
description: Guide for AUTHORING new analysis phases, ablation studies, and gate evaluations in alphatemp. Use PROACTIVELY when writing a new phase analysis script, adding model variants, designing gate criteria, implementing walk-forward validation, adding features to the regression pipeline, or structuring analysis output. NOT for running existing backtests (use backtest-runner). Triggers on new phase, analysis script, ablation, gate criteria, walk-forward, feature engineering, divergence features, Phase 2B, Phase 3, ensemble analysis, regression variant, improvement threshold, or writing a new model function.
tools: Read, Write, Edit, Bash, Grep
model: opus
---

# Backtesting Analysis — Authoring Guide

How to write new analysis phases, model variants, and gate evaluations. For running existing backtests, see `backtest-runner`.

## Analysis Script Structure

Every analysis script follows a consistent multi-part structure. See `scripts/phase3_analysis.py` and `scripts/phase3b_analysis.py` as templates.

```python
#!/usr/bin/env python3
"""Phase N: [Description] — AlphaTemp Analysis."""

import duckdb
from datetime import date
from services.backtester import Backtester

DB = "data/alphatemp.duckdb"
STATION = "KNYC"
CITY = "NYC"

def main():
    bt = Backtester(db_path=DB, city=CITY)

    # ═══════════════════════════════════════════
    # Part 1: [Section Name]
    # ═══════════════════════════════════════════
    print("=" * 60)
    print("Part 1: [Description]")
    print("=" * 60)

    # Run backtests, collect results
    results = {}
    for variant_name, model_fn in variants.items():
        result = bt.run(model_fn, run_hours=[0, 6, 12, 18])
        results[variant_name] = result

    # Print results table
    print(f"\n{'Variant':<20} | {'Brier':>7} | {'vs Baseline':>10} | {'Evals':>6}")
    print("-" * 60)
    for name, r in results.items():
        pct = (1.0 - r.mean_brier / baseline_brier) * 100
        print(f"{name:<20} | {r.mean_brier:7.4f} | {pct:+9.1f}% | {r.total_evaluations:6d}")

    # Gate check
    champion = min(results, key=lambda k: results[k].mean_brier)
    gate = results[champion].mean_brier < baseline_brier
    print(f"\nChampion: {champion} (Brier {results[champion].mean_brier:.4f})")
    print(f"GATE: {'PASS' if gate else 'FAIL'} — {champion} {'beats' if gate else 'does NOT beat'} baseline")

    # ═══════════════════════════════════════════
    # Part 2: [Next Section]
    # ═══════════════════════════════════════════
    # ...

if __name__ == "__main__":
    main()
```

## The ModelFn Interface

All model functions share this signature:

```python
def model_fn(provider: BacktestDataProvider, ref_time: datetime) -> Optional[Dict[int, float]]:
    """Return 1°F integer bracket probabilities, or None if insufficient data."""
```

**For ensemble access**, model functions expose `.raw`:
```python
def model_fn(provider, ref_time):
    raw = model_fn.raw(provider, ref_time)  # Returns Optional[(center, std)]
    if raw is None:
        return None
    center, std = raw
    return generate_bracket_probs(center, std)
```

The `.raw` interface is critical for ensemble combination — it provides the Gaussian parameters before bracket discretization.

## Walk-Forward Validation

Strict temporal ordering. No future peeking.

```python
WALK_FORWARD_MIN_DAYS = 90

for obs_date in settlement_dates:
    # 1. Training data: ONLY dates strictly BEFORE obs_date
    training = [row for row in all_data if row.date < obs_date]

    if len(training) < WALK_FORWARD_MIN_DAYS:
        continue  # Skip — insufficient training data

    # 2. Fit model on training data
    coeffs = fit_regression(training)

    # 3. Predict for obs_date
    prediction = predict(coeffs, obs_date_features)

    # 4. THEN add obs_date to training pool (for next iteration)
    all_data.append(obs_date_row)
```

**Critical ordering:** Predict FIRST, then append to training. Swapping this order causes data leakage (this bug has happened before — see error log).

## Feature Engineering Conventions

### Phase 2 Features (Regression)
```python
_FEATURE_SETS = {
    "full":  [0, 1, 2, 3],  # All features
    "fcst":  [0],            # Forecast high only
    "month": [1, 2],         # Seasonal encoding only
    "delta": [3],            # Day-to-day momentum only
}

# Feature indices:
# 0 = fcst_high (raw forecast maximum temperature)
# 1 = sin(2π × month / 12)  — seasonal encoding
# 2 = cos(2π × month / 12)  — prevents Jan/Dec discontinuity
# 3 = delta_temp (actual[D-1] - actual[D-2]) — momentum
# 4..11 = extended variables (dewpoint, humidity, wind, pressure, cloud, precip, radiation, dewpoint_depression)
```

### Phase 2B Features (Divergence — Real-Time Obs)
```python
_PHASE2B_FEATURE_SETS = {
    "full":    [0, 1, 2, 3],  # All divergence features
    "instant": [0],           # temp_divergence only
    "cumul":   [1],           # cumulative_divergence only
    "runmax":  [2],           # running_max_divergence only (usually dominant)
    "slope":   [3],           # slope_divergence only
}

# Divergence features (from services/divergence.py):
# 0 = temp_divergence       — latest obs - latest interpolated forecast
# 1 = cumulative_divergence — mean(obs - fcst) across all pairs
# 2 = running_max_divergence — max(obs) - max(forecast curve up to T)
# 3 = slope_divergence      — trend of (obs - fcst) gap over time (needs ≥3 pairs)
```

### Ablation Study Pattern

Test each feature subset independently to find which features contribute:

```python
variants = {}
for name, indices in _FEATURE_SETS.items():
    def make_model(idx=indices):  # Capture idx in closure
        def model(provider, ref_time):
            return regression_model(provider, ref_time, feature_indices=idx)
        return model
    variants[f"regression_{name}"] = make_model()

# Run all variants
for name, model in variants.items():
    result = bt.run(model)
    # ...compare against baseline
```

**Important:** Use `idx=indices` default argument to capture the loop variable in the closure. Without this, all closures share the last value.

## Two-Level Cache (Phase 2B)

The Phase 2B backtester uses a two-level cache to avoid redundant computation:

**Level 1** — Keyed by `(con_id, run_hour, station_id, model_name)`:
- Forecast curves per date
- Observation series per date
- Phase 2 predictions per date (walk-forward fitted)
- Computed once per run_hour, reused across all update_hours

**Level 2** — Keyed by `(con_id, run_hour, station_id, update_hour_et, model_name)`:
- Training rows with divergence features truncated at that update_hour
- Per update_hour OLS regression fit
- Depends on Level 1 data

When adding new Phase 2B features:
- If the feature depends only on forecast/obs data → add to Level 1
- If it depends on the update_hour cutoff → add to Level 2
- Always `_cache.clear()` between tests in fixtures

## Gate Criteria

### Standard Gate (Phase 1 → Phase 2)
```python
gate = champion_brier < baseline_brier
# Binary PASS/FAIL
```

### Improvement Gate (Phase 2B)
```python
improvement = (1.0 - new_brier / baseline_brier) * 100

if improvement > 2.0:    gate = "KEEP"      # Clear win
elif improvement > 1.0:  gate = "DISCUSS"   # Marginal, worth discussing
else:                    gate = "MARGINAL"  # Not worth the complexity
```

### Correlation Gate (Ensemble)
```python
# At least one model pair with error correlation < 0.7
pairs = [("hrrr", "gfs"), ("hrrr", "ecmwf"), ("gfs", "ecmwf")]
gate = any(correlations[pair] < 0.7 for pair in pairs)
```

### Weight Learning Gate
```python
if learned_brier < equal_brier:
    print(f"Learned weights win by {pct:+.1f}% — use them")
else:
    print(f"Equal weights win — keep equal weights")
```

## Ensemble Combination

Gaussian mixture of bias-corrected models:

```python
from services.ensemble import combine_mixture_brackets

# Get raw (center, std) from each model
preds = []
for model_fn in [hrrr_model, gfs_model, ecmwf_model]:
    raw = model_fn.raw(provider, ref_time)
    if raw:
        preds.append(raw)

weights = [1/3, 1/3, 1/3]  # Equal weights (or learned)

# Combine
brackets = combine_mixture_brackets(preds, weights, radius=15)
# Returns Dict[int, float] — 1°F bracket probabilities
```

**Weighting schemes tested:**
- Equal weights: `[1/3, 1/3, 1/3]` — default, simple, robust
- Inverse-Brier: weight ∝ 1/brier — data-driven but barely improved over equal
- Walk-forward adaptive: 90-day expanding window — Phase 3.5 result (+1.31%)

## Multi-Model Backtesting

GFS and ECMWF only have midnight UTC runs (Open-Meteo convention). HRRR runs every hour.

```python
# HRRR: evaluate at all run hours
hrrr_result = bt.run(hrrr_model, run_hours=[0, 6, 12, 18])

# GFS/ECMWF: only midnight
gfs_result = bt.run(gfs_model, run_hours=[0], model_name='gfs')
ecmwf_result = bt.run(ecmwf_model, run_hours=[0], model_name='ecmwf')

# Ensemble: HRRR at current run_hour, GFS/ECMWF always at midnight
```

## Phase 2B Update Hours

Phase 2B evaluates at specific Eastern Time hours when real observations can correct the forecast:

```python
update_hours_et = [14, 15, 16, 17, 18]  # 2 PM to 6 PM ET
```

**Crossover point:** ~14 ET — before this, Phase 2 (no obs) is better. After this, Phase 2B (with obs) adds value. This has been validated for HRRR; GFS/ECMWF crossover still needs verification.

## Reporting Conventions

```python
# Table header
print(f"\n{'Variant':<20} | {'Brier':>7} | {'vs Base':>7} | {'Evals':>6} | {'00z':>7}")
print("-" * 70)

# Data rows
for name, r in results.items():
    pct = (1.0 - r.mean_brier / baseline) * 100
    hour_0 = r.by_run_hour.get(0, {}).get("mean_brier", float("nan"))
    print(f"{name:<20} | {r.mean_brier:7.4f} | {pct:+6.1f}% | {r.total_evaluations:6d} | {hour_0:7.4f}")

# Section footer
print(f"\nChampion: {champion} (Brier {best:.4f})")
print(f"GATE: {'PASS' if gate else 'FAIL'}")
print("=" * 60)
```

## Checklist for New Analysis Phases

- [ ] Script follows Part N structure with clear section headers
- [ ] Baseline comparison included (what are we comparing against?)
- [ ] Walk-forward validation (no data leakage)
- [ ] Ablation study (which features/variants contribute?)
- [ ] Gate criteria defined before running (what counts as PASS?)
- [ ] Per-run-hour breakdown reported
- [ ] Champion selected by Brier score
- [ ] Results logged to stdout in consistent table format
- [ ] Phase results summarized in HANDOFF.md after completion
