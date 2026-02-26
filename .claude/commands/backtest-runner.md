---
name: backtest-runner
description: Standardized workflow for running backtests, interpreting results, and comparing models in the alphatemp system. Use PROACTIVELY whenever running a backtest, evaluating a new model function, comparing model performance, analyzing Brier scores, interpreting hit rates, or debugging backtest results. Triggers on words like backtest, brier, hit rate, model comparison, historical performance, scoring, uniform model, bias-corrected model, or any discussion about evaluating models against historical data.
tools: Read, Bash, Edit, Grep
model: opus
---

# AlphaTemp Backtest Runner

Standardized workflow for running and interpreting backtests against historical data.

## Quick Start

### Run a backtest against real data:
```python
from services.backtester import Backtester, uniform_model, bias_corrected_model

bt = Backtester(db_path="data/alphatemp.duckdb", city="NYC")

# Phase 0: Uniform dummy model (baseline)
result = bt.run(uniform_model)
bt.print_summary(result)

# Phase 1: Bias-corrected Gaussian model
result = bt.run(bias_corrected_model)
bt.print_summary(result)
```

### Run for specific date range or run hours:
```python
from datetime import date

result = bt.run(
    bias_corrected_model,
    start_date=date(2025, 6, 1),
    end_date=date(2025, 8, 31),
    run_hours=[12],  # 12z only
)
```

## How the Backtester Works

For each settlement date in `nws_daily`:
  For each HRRR run hour (00z, 06z, 12z, 18z):
    1. Construct a `BacktestDataProvider` anchored to that model_run
    2. Set `ref_time = model_run + 2h` (simulates HRRR data availability lag)
    3. Call `model_fn(provider, ref_time)` → 1°F integer bracket probabilities
    4. Map to Kalshi 2°F brackets if available for that date
    5. Compute Brier score against settlement outcome

### Scoring Methods
- **Kalshi 2°F scoring** — Used when `kalshi_settlements` has bracket data for that date. Maps model's 1°F probabilities into Kalshi's 2°F bracket structure (inclusive interior, open-ended tails).
- **1°F fallback scoring** — Used when no Kalshi bracket data exists. Scores against the actual NWS high rounded to nearest integer.

## Interpreting Results

### Brier Score
- **0.0** = perfect prediction (all probability on the correct bracket)
- **~0.06** = uniform distribution across ~31 brackets (uninformative baseline)
- **~1.0** = uniform model scored against real data (baseline from Phase 0)
- **Lower is better.** A model must beat the uniform baseline to be useful.

Multi-category Brier: `BS = Σ(p_k - o_k)²` where `o_k = 1` for the settled bracket.

### Hit Rate
- **Top-1 hit rate:** How often the argmax of model probabilities matches the actual high temperature bracket
- A good model should achieve >20% top-1 hit rate (vs ~3% for uniform over 31 brackets)

### Per-Run-Hour Breakdown
The `by_run_hour` dict shows performance at each HRRR run time:
- **00z/06z** — Longer lead time, higher uncertainty, expect worse scores
- **12z/18z** — Shorter lead time, better forecast data, expect better scores
- Large performance gaps between run hours suggest the model's time_factor needs tuning

## Writing a New Model Function

Model functions have this signature:
```python
def my_model(provider: BacktestDataProvider, ref_time: datetime) -> Optional[Dict[int, float]]:
    """Return 1°F integer bracket probabilities, or None if can't forecast."""
```

Requirements:
- Return `Dict[int, float]` mapping integer temperatures to probabilities
- Probabilities should sum to ~1.0
- Return `None` if insufficient data (will be counted as skipped)
- Use `provider` for all data access (not direct DB queries)

Available provider methods:
- `provider.get_forecast_high(station_id)` — MAX(temp_f) from the specific model_run
- `provider.get_bias_stats(station_id)` — StationBias with mean_bias, std_error, sample_days
- `provider.get_drift_score(city)` — Always 0.0 in backtest (no historical drift signals)
- `provider.get_recent_drift_scores(city, limit)` — Always empty in backtest
- `provider.get_recent_forecast_highs(station_id, limit)` — Last N model run highs (time-fenced)

## Comparing Models

When comparing two models, run both on the same date range and run hours:

```python
result_a = bt.run(model_a, start_date=start, end_date=end, run_hours=[12])
result_b = bt.run(model_b, start_date=start, end_date=end, run_hours=[12])

print(f"Model A: Brier={result_a.mean_brier:.4f}, Hit={result_a.top1_hit_rate:.1%}")
print(f"Model B: Brier={result_b.mean_brier:.4f}, Hit={result_b.top1_hit_rate:.1%}")
```

Key comparisons:
- **Brier score** — Primary metric. Lower wins.
- **Hit rate** — Secondary. Higher wins.
- **Per-run-hour** — Check if one model dominates at specific run hours
- **Kalshi vs fallback scoring** — Note the mix. Kalshi-scored results are more meaningful for real trading.

## Common Issues

1. **High skip count** — Missing forecast data for many model_runs. Check HRRR backfill coverage.
2. **All results use fallback scoring** — No Kalshi settlement data in date range. Use a range that overlaps with Kalshi history.
3. **Model returns None frequently** — The model function can't find data. Check provider queries.
4. **Brier score > 2.0** — Model is actively harmful. Probabilities may not sum to 1.0 or bracket mapping is wrong.

## Data Coverage (as of Phase 0)

- NWS daily highs: ~1,902 settlement days
- Kalshi bracket data: ~1,659 dates
- HRRR forecasts: ~144K rows across 4 run hours
- Overlap (12z): ~1,900 days with both forecast and settlement data
