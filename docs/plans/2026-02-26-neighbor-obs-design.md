# Neighbor Station Observations — Design Doc

**Created:** 2026-02-26
**Status:** Approved
**Phase:** 3.6 (between Phase 3.5 adaptive weights and Phase 4 strategy)

---

## Problem Statement

Phase 2B obs divergence features only help after 14 ET because KNYC (Central Park) reports hourly — too sparse to establish a reliable temperature trajectory in the morning. Meanwhile, KLGA (LaGuardia) and KEWR (Newark) have 1-minute ASOS data already in the DB (583K obs each, back to Dec 2020) and are completely unused.

The Kalshi market is weakest from midnight through early morning (Brier ~0.63-0.87). That's where our model has the best shot at edge — but it's exactly where obs features currently hurt.

## Success Criteria

- **Primary:** Push the obs crossover point earlier than 14 ET. If neighbor data makes divergence features net-positive at 10 ET or earlier, that opens hours where the market is thinnest.
- **Secondary:** >2% Brier improvement across any sustained block of hours in the 0-18 ET window.
- **Kill condition:** No improvement anywhere in 0-18 ET.

---

## Ablation Matrix

6 variants testing 3 station-mapping strategies x 2 integration methods, across all 3 models (HRRR, GFS, ECMWF):

### Station-Mapping Strategies

**A. Raw nearby temps** — Use KLGA/KEWR observations directly, no offset correction. Simplest approach. If KLGA reads 73F, treat it as roughly 73F. Divergence features are relative to the forecast, so systematic offsets may wash out.

**B. Learned station-to-KNYC offset** — Walk-forward expanding-window mean difference between each neighbor and KNYC at matching timestamps. Apply offset before computing divergence. More accurate absolute temps, but adds a calibration layer needing sufficient overlapping obs.

**C. Trend only (rate of change)** — Don't use neighbor absolute temps. Extract slope/acceleration of KLGA/KEWR between KNYC reports to infer what KNYC is doing between its hourly readings. Sidesteps the offset problem, throws away magnitude.

### Integration Methods

**1. New features alongside KNYC** — Keep all existing KNYC divergence features. Add neighbor-derived features as additional Phase 2B regression inputs. Regression decides weights.

**2. Blended curve replacing KNYC** — Build a single enhanced temperature curve (KNYC-anchored, neighbor-interpolated between reports). Compute same divergence features from blended curve instead of raw KNYC.

### Variant Summary

| Variant | Station Mapping | Integration | Description |
|---------|----------------|-------------|-------------|
| A1 | Raw nearby | New features | Add raw KLGA/KEWR divergence alongside KNYC |
| A2 | Raw nearby | Blended curve | Blended curve from raw neighbor + KNYC temps |
| B1 | Learned offset | New features | Offset-corrected neighbors as new features |
| B2 | Learned offset | Blended curve | Offset-corrected neighbors blended with KNYC |
| C1 | Trend only | New features | Neighbor slope/acceleration as new features |
| C2 | Trend only | Blended curve | Neighbor trends interpolate KNYC between reports |

---

## Neighbor Peak Signal

Additional feature for all variants: **neighbor_peak_signal** — detects when KLGA/KEWR temperatures start declining, indicating the daily high has likely passed.

Possible implementations:
- Minutes since neighbor running_max last increased
- Binary: neighbor slope turned negative
- Continuous: rate of decline after peak

This is a leading indicator for Central Park's peak. If both airports peaked at 2:15 PM, the model can confidently say "the high is in" 45 minutes before KNYC's 3 PM report. Especially powerful in the 13-16 ET window.

---

## Architecture & Data Flow

### What changes

Only the obs divergence feature computation in the Phase 2B layer. Everything upstream (Phase 2 regression) and downstream (Phase 3.5 ensemble + adaptive weights) stays untouched.

### Current flow (KNYC only)

```
KNYC hourly obs -> compute_divergence_features() -> [running_max, instant, cumul, slope] -> Phase 2B regression
```

### New flow — "new features" variants (A1, B1, C1)

```
KNYC hourly obs  -> compute_divergence_features()  -> [knyc_running_max, ...]        -+
KLGA 1-min obs   -> compute_neighbor_features()    -> [neighbor_running_max, ...]     -+-> Phase 2B regression
KEWR 1-min obs   -> compute_neighbor_features()    -> [neighbor_running_max, ...]     -+
                                                      [neighbor_peak_signal]          -+
```

### New flow — "blended curve" variants (A2, B2, C2)

```
KNYC + KLGA + KEWR -> build_blended_curve() -> single enhanced temp series -> compute_divergence_features() -> Phase 2B regression
```

### New code

- `services/neighbor_obs.py` — offset learning, trend extraction, curve blending, peak signal
- Extensions to `services/divergence.py` — neighbor feature computation
- `scripts/phase36_neighbor_analysis.py` — ablation runner + improvement curves

### No database changes

`BacktestDataProvider.get_observations_in_range()` already supports any station_id. No schema migrations needed.

---

## Evaluation Plan

**Baseline:** Phase 3.5 ensemble + adaptive weights (current best)

**Window:** 0-18 ET, every hour. Full picture, no artificial cutoffs.

**Ablation order:** Run all 6 variants on HRRR first. Extend winners to GFS/ECMWF and through the ensemble.

### Key outputs

1. **Crossover chart** — Hour where obs features flip from harmful to helpful for each variant vs baseline. Current crossover: 14 ET.
2. **Improvement curve** — Brier delta per hour (0-18 ET) for each variant vs baseline.
3. **Feature importance** — Which neighbor-derived features carry the most weight in the winning variant.
4. **Station contribution** — Does KLGA help more than KEWR? (KLGA is closer to Central Park.)

### Gate

- **PASS:** >2% Brier improvement across any sustained block of hours
- **DISCUSS:** 1-2% improvement, or improvement at isolated hours only
- **KILL:** No improvement anywhere in 0-18 ET

---

## Future Phase: Dynamic Uncertainty (Phase 3.7)

**Not being built now.** Documented here because it was identified during this design as a known architectural limitation.

### Problem

The model outputs a Gaussian with roughly fixed standard deviation regardless of how much evidence has accumulated. At midnight, a 2-3F spread is reasonable. By late afternoon with the high clearly observed, the model should be near-certain but can't collapse its distribution. The market does this — Kalshi Brier drops from 0.63 overnight to 0.14 by close.

### Why it matters

Edge calculation requires accurate confidence. If the model says "30% on the right bracket" when reality is 90%, we're leaving money on the table. If it says 30% when reality is 10%, we're burning capital. Position sizing (Kelly criterion) needs calibrated probabilities.

### Likely approach (to be designed properly later)

- Residual std becomes a function of update hour and obs quality
- Could range from simple (per-update-hour residual std from Phase 2B regression) to sophisticated (Bayesian updating that narrows with each observation)
- Neighbor station infrastructure from Phase 3.6 feeds directly into this

### Critical: Re-evaluate killed features as variance predictors

All prior go/no-go gates evaluated features as MEAN predictors under a fixed std. Features that predict VARIANCE (how confident to be) are invisible under fixed std. When building dynamic uncertainty, re-test:

- **cumulative_divergence** (killed at +0.6%) — may predict "how settled is the temperature trajectory"
- **slope_divergence** (killed at +0.6%) — slope near zero may signal "peak is locked in"
- **Extended weather vars** (all killed in Phase 3.5) — cloud cover, wind, humidity may not predict bias direction but could predict forecast certainty
- **neighbor_peak_signal** (new in Phase 3.6) — direct confidence indicator

These features may have been incorrectly killed as bias predictors when their real value is as confidence predictors. No data or code was lost — the experiment is clean.

### Gate

Model-implied confidence correlates with actual bracket hit rate (calibration curve). When the model says 80%, it should be right ~80% of the time.

### Depends on

Phase 3.6 (neighbor obs) completing first. More observation data = better confidence estimation.
