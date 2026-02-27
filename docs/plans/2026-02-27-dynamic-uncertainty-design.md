# Phase 3.7: Dynamic Uncertainty — Design Document

**Created:** 2026-02-27
**Status:** DESIGNED
**Depends on:** Phase 3.6 results (determines baseline)

---

## Problem Statement

The model outputs a Gaussian with roughly fixed std (2.4-3.0°F) regardless of how much evidence is available. At midnight with no observations, std ≈ 2.5°F. At 3 PM with 15 hours of obs clearly tracking, std ≈ 2.4°F. At 5 PM with the daily high obviously locked in, std ≈ 2.3°F.

The std barely moves. Meanwhile, the Kalshi market's Brier drops from 0.63 (midnight) to 0.14 (6 PM) because traders collapse uncertainty as evidence arrives. The model can't do this.

**Consequences:**
1. **Brier score suffers** — the model can't express high confidence even when it should
2. **Calibration is poor** — when the model says 40% for a bracket at 5 PM, it might really be 85%
3. **Kelly sizing is broken** — can't size bets properly without calibrated probabilities

---

## Architecture

Add a parallel variance regression at the Phase 2B layer. Each model gets two regressions:

```
Phase 2B currently:
  mean_regression: residual ~ divergence_features → (predicted_residual, FIXED residual_std)

Phase 2B proposed:
  mean_regression: residual ~ divergence_features → predicted_residual
  variance_regression: log(residual² + 1e-6) ~ variance_features → predicted_std
  return (predicted_residual, predicted_std)
```

The mean regression is untouched. The variance regression trains on `log(squared_residuals)` from the mean regression, learning *how wrong the mean regression tends to be* given today's conditions.

Each NWP model (HRRR, GFS, ECMWF) gets its own variance regression because their error profiles differ.

### Why log(residual²)?

- Modeling heteroscedasticity via log-squared-residuals is a well-established econometric technique (Harvey's multiplicative heteroscedasticity model)
- Log transform guarantees `exp(prediction)` is always positive — no negative std
- Symmetric in OLS, maps naturally to the existing walk-forward architecture

### Output

```python
predicted_std = sqrt(exp(variance_regression_prediction))
# Floored at 0.3°F (minimum certainty)
# Capped at 5.0°F (prevent runaway uncertainty from degenerate early-window fits)
```

---

## Winsorization (Critical Guardrail)

Squared residuals are extremely sensitive to outliers. A single freak 10°F miss creates a residual² of 100 that permanently inflates predicted variance in an expanding window.

**Fix:** Before squaring, clip residuals to the [2nd, 98th] percentile of the training window. This prevents outlier explosions while preserving the signal from genuinely uncertain conditions.

---

## Variance Feature Candidates (Full Ablation)

### Tier 1 — High-confidence candidates
| Feature | Source | Hypothesis |
|---------|--------|-----------|
| `update_hour` | 0-18 ET | Primary driver of uncertainty collapse |
| `divergence_slope` | Phase 2B | Near zero = trajectory stabilized = high confidence |
| `neighbor_peak_signal` | Phase 3.6 | Neighbors cooling = Central Park peak likely in |
| `forecast_spread` | Multi-model | Model disagreement = uncertainty |

### Tier 2 — Promising but unproven
| Feature | Source | Hypothesis |
|---------|--------|-----------|
| `run_to_run_convergence` | HRRR 00z/06z/12z/18z | Tight spread across runs = model locked in |
| `hours_until_sunset` | Astronomical calc | Season×time interaction: 3 PM Dec vs 3 PM Jul |
| `cumulative_divergence` | Phase 2B (killed +0.6% as mean predictor) | Accumulated drift magnitude predicts instability |
| `running_max_divergence` | Phase 2B | Worst divergence seen = how bad it got |

### Tier 3 — Re-evaluated killed features
| Feature | Source | Hypothesis |
|---------|--------|-----------|
| `fcst_high` | Phase 2 | Extreme temperatures harder to predict |
| `dewpoint_depression` | Extended weather vars | Dry air = more volatile temps |
| `mean_cloud_cover` | Extended weather vars | Cloud cover stabilizes temperature swings |
| `max_wind` | Extended weather vars | High wind = well-mixed = more predictable |
| `mean_pressure` | Extended weather vars | Synoptic pattern proxy |

### Dropped
- `obs_count` — nearly perfectly collinear with `update_hour` (hourly KNYC reports). Would destabilize OLS coefficients.

### Ablation approach
Test each feature individually, then test winning combinations. Same methodology as Phase 3.6.

---

## Evaluation Framework

### Primary metrics (both must pass to clear gate)
1. **Brier score** — >2% improvement vs current best (Phase 3.5/3.6 baseline)
2. **Expected Calibration Error (ECE)** — bin predictions into deciles, compare predicted vs observed frequency. Must decrease.

### Secondary metrics (tracked, not gated)
- **Log-loss (cross-entropy)** — heavily punishes extreme overconfidence. Critical for Kelly sizing safety. Catches the failure mode where overconfident bets blow up the bankroll.
- **Brier decomposition — Reliability term** — isolates calibration from discrimination. Should approach 0.
- **Std trajectory by hour** — plot predicted std across 0-18 ET averaged over all dates. Must show clear downward slope (wide at midnight, narrow by afternoon). Flat = variance regression isn't learning.

### Evaluation window
Full 0-18 ET. Per-hour breakdowns to see where dynamic std helps most.

### Baseline comparison
Static-std variant (current best) vs dynamic-std variant, same mean regression, same ensemble weights. Isolates the effect of dynamic std.

---

## Integration with Existing Pipeline

### What changes
- `_fit_and_predict_phase2b()` gains a second regression (variance) alongside the existing mean regression
- New module: `services/variance_model.py` — winsorization, log-variance OLS, hours_until_sunset
- New analysis script for ablation matrix

### What doesn't change
- Mean regression — untouched, same center predictions
- Ensemble — already loops over per-model `(center, std)`, no changes needed
- Bracket probability computation — already uses the std it's given
- Phase 3.6 neighbor features — feed into variance regression as candidates, pipeline unchanged

### Dependency
Phase 3.6 results determine the baseline. If Phase 3.6 passes, baseline includes neighbor features. If killed, baseline is Phase 3.5 ensemble.

---

## Future: Quantile Regression (Escape Hatch)

The Gaussian assumption breaks late in the day — if it's 72°F at 2 PM, the high *can't* be below 72, but the Gaussian assigns probability there. If OLS variance regression passes the gate but calibration is still poor in the tails (assigning probability to physically impossible temperatures), quantile regression is the documented next step.

**How it would work:** Instead of predicting mean+std, predict the 10th, 25th, 50th, 75th, and 90th percentiles of the error directly. This naturally handles asymmetric risk and heavy tails without the Gaussian assumption.

**Logged for:** Potential Phase 3.8 if needed.

---

## Key Design Decisions

1. **Variance regression, not regime lookup** — OLS captures continuous relationships; lookup tables fragment the data
2. **Per-model variance regression** — each NWP model's error profile differs
3. **Winsorization at [2nd, 98th] percentile** — prevent outlier explosion in expanding window
4. **Drop obs_count, keep update_hour** — collinearity trap identified by external review
5. **Full ablation** — test all candidate features including killed mean predictors
6. **Dual gate** — Brier >2% AND lower ECE
7. **Log-loss as secondary metric** — catches extreme overconfidence that Brier forgives
8. **Quantile regression documented as escape hatch** — if Gaussian tails are problematic
