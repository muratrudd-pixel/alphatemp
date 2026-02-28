# Phase 2B Design Review — Intra-Day Observation Updates for Temperature Prediction

## Prompt for Gemini Deep Research

I'm building a system to predict daily high temperature bracket probabilities for Kalshi prediction markets (KXHIGHNY series — NYC / Central Park). I need your critical evaluation of the approach described below. Specifically:

1. **Is the residual regression approach sound?** We train Phase 2B on the residual from Phase 2 (what Phase 2's model didn't predict) rather than retraining on raw error with all features combined. Is this the right decomposition, or would a single unified model be better?

2. **Are the right divergence features selected?** We chose instantaneous divergence, cumulative divergence, running max divergence, and 6-hour synoptic max divergence. Are there better features from the intra-day obs-vs-forecast comparison that we should consider?

3. **Should we compare observations against the raw or bias-adjusted HRRR curve?** We chose the raw curve because Phase 2's bias correction applies to the daily MAX, not individual hourly points. Spreading a daily-max bias across the hourly curve would be physically wrong. The regression target already nets out Phase 2. Is this reasoning correct?

4. **Is simple per-update-hour regression the right starting point, or should we jump directly to RAFT (Rapid Adjustment of Forecast Trajectories) or Kalman filtering?** We're following a "prove signal exists with simple methods first" philosophy.

5. **Any methodological red flags?** Walk-forward bias, data leakage risks, overfitting concerns, missing signals?

---

## System Overview

### What we're predicting
Daily high temperature for NYC (Central Park / KNYC) in 2°F brackets for Kalshi markets. Evaluated using multi-category Brier score (lower = better, 0 = perfect).

### Current model performance
| Model | Brier Score | Description |
|-------|------------|-------------|
| Uniform (clueless) | 1.02 | Equal probability across all brackets |
| Phase 1 (flat bias) | 0.8356 | Walk-forward per-run-hour mean bias correction |
| **Phase 2 (regression)** | **0.7979** | Walk-forward OLS on fcst_high + sin/cos(month) + delta_temp |
| Kalshi market (overnight) | 0.63 | What traders produce at 00z-06z |
| Kalshi market (evening) | 0.14 | What traders produce by 18z |

Gap to close: our model at 0.80 vs market overnight at 0.63. The market narrows to 0.14 by evening because traders watch real-time observations. We don't watch any observations yet.

### Data available
- **HRRR forecasts:** 4 runs/day (00z, 06z, 12z, 18z), each producing 18 hourly temperature forecasts. Stored in DB with (station_id, model_run, valid_at, temp_f).
- **KNYC observations:** Hourly METARs (~12-20/day) with temp_f. Additionally, 6-hour synoptic max/min reported at 00z, 06z, 12z, 18z in METAR remark groups (already parsed and stored).
- **NWS daily settlement:** Authoritative daily high from NWS Climate Report.
- **Historical data:** ~1,900 days of aligned HRRR forecasts + NWS daily highs + Kalshi settlements.

### What Phase 2 already does
Walk-forward OLS regression predicting HRRR daily high error from features available at model-run time:
```
error = β₀ + β₁·fcst_high + β₂·sin(2π·month/12) + β₃·cos(2π·month/12) + β₄·delta_temp
```
Key finding: HRRR has a temperature-dependent warm bias (~+0.1°F per 1°F of forecast). Monthly seasonal effect significant (summer overprediction +1-2.3°F, winter underprediction -1.5-2.5°F). Day-over-day temperature change (delta_temp) has zero predictive signal.

Trained with expanding window (all prior dates, strict obs_date < current_date). Minimum 90-day training window.

---

## Phase 2B Proposed Design

### Core idea
As observations arrive throughout the day, compare them against the HRRR forecast curve to detect same-day divergence. Use this divergence to adjust Phase 2's prediction.

### Feature set (obs vs RAW HRRR curve)

We compare observations against the **raw (unbiased) HRRR forecast curve**, not a bias-adjusted curve. Rationale: Phase 2's bias prediction applies to the daily MAX, not to individual hourly forecast points. There's no principled way to distribute a daily-max bias across the 18-hour curve. The regression target (residual from Phase 2) already accounts for Phase 2's prediction.

| Feature | Formula | Intuition |
|---------|---------|-----------|
| Instantaneous divergence | `obs(T) - interpolated_fcst(T)` | Current gap between reality and HRRR at time T |
| Cumulative divergence | `mean(obs_i - fcst_i)` for all obs midnight→T | Persistent bias today, robust to transients |
| Running max divergence | `max(obs) - max(fcst curve up to T)` | Observed peak vs forecast peak so far. Converges to actual answer by afternoon. |
| Synoptic 6-hr max divergence | `six_hr_max - max(fcst over same 6h window)` | Authoritative station max from METAR synoptic group. Available at 12z and 18z only. |

### Model structure: residual regression

```
Phase 2:   predicted_bias = f(fcst_high, month, delta_temp)
Phase 2B:  residual = actual_error - Phase2_predicted_bias
           predicted_residual = g(temp_div, cumul_div, runmax_div, synoptic_div)
Final:     adjusted_bias = Phase2_predicted_bias + Phase2B_predicted_residual
           center = fcst_high - adjusted_bias
           bracket_probs = Gaussian CDF over 1°F brackets
```

Phase 2B is trained on the **residual** from Phase 2, not the raw error. This means:
- Phase 2 handles the structural bias (temperature regime + season)
- Phase 2B handles the day-specific surprise (obs diverging from expectations)
- The two compose cleanly: Phase 2B only needs to explain what Phase 2 missed

Walk-forward expanding window, separate regression per (run_hour, update_hour), same 90-day minimum as Phase 2.

### Update schedule

Hourly from 08 ET through 18 ET (11 update points per day). Aligned with METAR hourly reports (~:56 past each hour) and 6-hour synoptic max releases at 12z and 18z.

### Ablation design

| Variant | Features | Tests whether... |
|---------|----------|-----------------|
| `wf_phase2b_full` | all 4 features | Full model |
| `wf_phase2b_temp` | 3 temp features (no synoptic) | Synoptic max adds value? |
| `wf_phase2b_instant` | instantaneous only | Single-point divergence sufficient? |
| `wf_phase2b_cumul` | cumulative only | Persistence is the key signal? |
| `wf_phase2b_runmax` | running max only | Peak tracking is what matters? |

### Decision gate
- <1% Brier improvement → kill Phase 2B, not worth the complexity
- 1-2% → marginal, discuss tradeoffs
- \>2% → clear win, keep it

---

## Alternative Approaches Considered

### RAFT (Rapid Adjustment of Forecast Trajectories, Schuhen et al. 2019)
- Estimates the temporal error correlation matrix across all forecast hours from historical data
- Uses conditional Gaussian updating: given observed errors at verified hours, predict errors at unverified future hours
- Mathematically equivalent to Gaussian Process regression on the error curve
- **Pro:** Optimal use of all available information — uses the full structure of how errors at different forecast hours correlate
- **Con:** Requires estimating and inverting a covariance matrix. More complex to implement and debug. Hard to add non-temperature features.
- **Our reasoning:** Start with simple regression to prove the signal exists. RAFT is a natural upgrade path — same signal, more elegant processing. If regression shows >2% improvement, RAFT could squeeze out more.

### Kalman Filter (Homleid 1995, Galanis 2002)
- State vector: diurnal bias correction terms (one per time-of-day bin)
- Observation model: obs-fcst error at current hour
- Transition model: near-identity (biases evolve slowly)
- **Pro:** Handles non-stationary bias that changes over weeks/months
- **Con:** Primarily tracks slow-evolving systematic error. Our expanding-window regression already adapts. The day-specific signal (surprise events) is what matters for Phase 2B.
- **Our reasoning:** Walk-forward regression with an expanding window already adapts to changing bias patterns. Kalman adds value mainly for rapid regime changes, which Phase 2's month encoding captures.

### Single unified model (not residual decomposition)
- One big regression: `error ~ fcst_high + month + delta_temp + divergence_features`
- **Pro:** Single model, no composition
- **Con:** The divergence features are only available when observations exist. Training sample shrinks (only dates with enough obs at each update_hour). Phase 2 features would be re-learned from a subset of the data, potentially degrading. Harder to diagnose which layer is contributing.

---

## Key Properties of Our Dataset

- **~1,900 settlement dates** (2020-2026), all with HRRR + NWS + observations
- **KNYC observations:** ~12-20 per day, hourly METARs at ~:56 past each hour
- **6-hour synoptic max/min:** Available at 00z, 06z, 12z, 18z in select METARs
- **HRRR forecast curve:** 18 hourly points per model run
- **HRRR bias pattern:** Temperature-dependent (+0.1°F per 1°F of forecast), seasonal (summer hot, winter cold), per-run-hour
- **Walk-forward evaluation:** No future data leakage. Expanding training window. 90-day minimum.
- **Evaluation metric:** Multi-category Brier score on Kalshi's 2°F bracket structure (12-15 brackets per day)

## Questions for Your Assessment

1. Is there a better decomposition than Phase 2 (structural) + Phase 2B (same-day obs)?
2. Are we missing any high-value features from the METAR/HRRR comparison that don't require heavy parsing?
3. Is per-update-hour regression the right level of granularity, or should we pool across update hours with update_hour as a feature?
4. For the running_max_divergence feature — is there a more principled way to estimate the daily max from partial-day observations than just taking max(obs so far)?
5. Any concerns about the walk-forward training methodology for observation-based features?
