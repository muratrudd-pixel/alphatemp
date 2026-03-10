# AlphaTemp Autoresearch — Design Document

**Date:** 2026-03-09
**Inspired by:** Karpathy's autoresearch (autonomous AI experiment loop)
**Approach:** Hybrid — isolated experiment module + shared infrastructure

## Objective

Build an autonomous experiment loop where Claude Code iteratively modifies
a probability model, runs a backtest, measures a composite score, commits
improvements, discards failures — and repeats ~60-120 times overnight.

## Architecture

### File Structure

```
alphatemp/
├── autoresearch/                    # Experiment sandbox
│   ├── program.md                   # Agent instructions (mission briefing)
│   ├── experiment.py                # THE FILE the agent modifies (~400 lines)
│   ├── run_experiment.py            # Harness: runs, scores, logs (LOCKED)
│   ├── results.tsv                  # Experiment log (auto-generated)
│   └── best_score.txt               # Current best composite score
│
├── services/                        # LOCKED — agent cannot modify
│   ├── backtester.py                # Brier scoring, bracket mapping
│   ├── data_provider.py             # BacktestDataProvider
│   ├── strategy_backtester.py       # Fee math, P&L calculation
│   └── quantile_model.py            # Starting reference for experiment.py
│
└── core/                            # LOCKED — agent cannot modify
    ├── db.py                        # DuckDB connection
    └── constants.py                 # Station coords, cities
```

### Boundary Rule

`experiment.py` imports from `services/` and `core/` but the agent only
modifies `experiment.py`. Everything else is locked.

## experiment.py — The Agent-Owned File

Exports one function with a fixed signature:

```python
def model_fn(provider: BacktestDataProvider, ref_time: datetime) -> Optional[Dict[int, float]]:
    """Return 1°F integer bracket probabilities, or None if can't forecast."""
```

Optional `DESCRIPTION` string variable for self-documentation.

Starting version: copy of current best QR model (wf_qr_multimodel_v4),
extracted into this single file. Contains feature engineering, model training,
CDF construction, bracket probability output, all hyperparameters.

Agent can freely restructure — add classes, swap libraries, try entirely
different architectures. Only contract: export `model_fn` with that signature.

## run_experiment.py — The Locked Harness

```
1. Import model_fn from experiment.py
2. Run backtester in quick mode:
   - 6 update hours (0, 4, 8, 12, 16, 20 ET)
   - 4 run hours (0, 6, 12, 18 UTC)
   - All available settlement dates with Kalshi bracket data
3. Compute composite score (lower = better):
   - 45% Net P&L (fee-adjusted, inverted + normalized)
   - 35% Brier score (full day)
   - 20% Hit rate (inverted + normalized)
4. Print: "SCORE: X.XXXX"
5. Append to results.tsv
6. Exit code 0 if improvement, exit code 1 if not
```

### Time Cap

90 seconds hard limit per experiment. If exceeded, killed and counted as
failure. Rationale: if it can't score in 90 seconds, it's too expensive for
production.

### Normalization

- P&L: invert sign, scale to [0, 1] using baseline min/max (computed on first run)
- Hit rate: `1 - hit_rate` (so lower = better, matching Brier direction)
- Baseline scores cached from the starting experiment.py

## Composite Score

| Component | Weight | Direction |
|---|---|---|
| Net P&L (fee-adjusted) | 45% | Inverted + normalized (lower = better) |
| Brier score (full day) | 35% | Already lower = better |
| Top-1 hit rate | 20% | Inverted + normalized (lower = better) |

## Constraints (Hard Rules)

1. **Walk-forward validation** — no future data leakage
2. **Kalshi fee math** — fixed formula, not tunable
3. **Settlement source** — NWS CLI from KNYC, non-negotiable
4. **Database schema** — agent cannot add/drop tables
5. **Python 3.9 compatibility** — no 3.10+ syntax

## What the Agent Can Modify

- Feature engineering (add/remove/transform features)
- Model architecture (QR, XGBoost, ridge, ensemble, neural net)
- Hyperparameters (window size, quantile positions, regularization)
- CDF construction (tail treatment, interpolation method)
- Training strategy (rolling window, expanding window, seasonal weighting)

## Git Workflow

```
1. Agent modifies experiment.py
2. Runs: python run_experiment.py
3. Exit 0 → git add + git commit with score in message
4. Exit 1 → git checkout -- experiment.py
5. Repeat
```

All experiments on dedicated branch: `autoresearch/run-YYYY-MM-DD`.
Linear commits, each is a verified improvement.

## Launch Procedure

```bash
cd ~/Projects/alphatemp/alphatemp
git checkout -b autoresearch/run-2026-03-09
claude
# > Read autoresearch/program.md and start running experiments.
# > Run fully autonomously. Don't ask for confirmation.
```

## Available Data

- **HRRR:** hourly temps, 18h horizon, 3km grid (all run hours)
- **GFS:** 00z only (06z/12z/18z contaminated — DO NOT USE)
- **ECMWF:** via Open-Meteo composite
- **Observations:** KNYC, KLGA, KEWR METAR (hourly at ~:51)
- **NWS settlement:** daily high from CLI report
- **Kalshi brackets:** 2°F wide, historical settlements

## Expected Performance

- ~30-60 seconds per experiment (quick-mode backtest)
- ~60-120 experiments per hour
- ~500-1000 experiments overnight (8 hours)
- ~10-20% success rate (normal for research)
- Net: 50-200 verified improvements per overnight run
