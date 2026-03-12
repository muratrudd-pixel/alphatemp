# Paper Trading System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a fully autonomous paper trading system that continuously evaluates edge in Kalshi KXHIGHNY temperature markets, places simulated trades, tracks P&L net of fees, with circuit breakers and a dashboard kill switch.

**Architecture:** A Strategy Engine service runs every 5 minutes, reads latest forecast/obs/market data from DuckDB, runs the QR model forward, computes bracket probabilities, compares to Kalshi market prices, and sends trade decisions to the existing PaperTrader. A Settlement service auto-settles positions when NWS CLI data arrives. A new Dashboard tab shows live positions, unrealized P&L, and circuit breaker status.

**Tech Stack:** Python 3.9, DuckDB, FastAPI, scipy (linprog), numpy, asyncio, Plotly, Tailwind CSS

**Design Doc:** `docs/plans/2026-03-11-paper-trading-design.md`

---

## Task 1: Extract QR Model (`services/model.py`)

Extract the quantile regression model from `autoresearch/experiment.py` into a reusable service module with two public methods: `fit()` and `predict_bracket_probs()`.

**Files:**
- Create: `services/model.py`
- Create: `tests/test_model.py`
- Reference: `autoresearch/experiment.py` (source of truth for math)

**Step 1: Write the failing test**

```python
# tests/test_model.py
import numpy as np
import pytest
from services.model import QRModel

def test_fit_returns_coefficients():
    """fit() should return 7 coefficient arrays (one per quantile)."""
    model = QRModel()
    rng = np.random.RandomState(42)
    X = rng.randn(200, 23)
    y = X[:, 1] * 0.5 + rng.randn(200) * 2  # fcst_high-ish signal
    coeffs = model.fit(X, y, run_hour=0, date_key="2026-03-11")
    assert coeffs is not None
    assert len(coeffs) == 7
    # Each coeff has 24 elements (intercept + 23 features)
    for c in coeffs:
        assert len(c) == 24

def test_fit_caches_scale_params():
    """fit() should cache mean/std for later prediction."""
    model = QRModel()
    rng = np.random.RandomState(42)
    X = rng.randn(200, 23)
    y = rng.randn(200)
    model.fit(X, y, run_hour=0, date_key="2026-03-11")
    assert (0, "2026-03-11") in model._scale_cache
    mean, std = model._scale_cache[(0, "2026-03-11")]
    assert len(mean) == 23
    assert len(std) == 23

def test_predict_bracket_probs_returns_dict():
    """predict_bracket_probs() should return {bracket_floor: probability}."""
    model = QRModel()
    rng = np.random.RandomState(42)
    X = rng.randn(200, 23)
    y = rng.randn(200) * 3
    model.fit(X, y, run_hour=0, date_key="2026-03-11")
    features = rng.randn(23)
    probs = model.predict_bracket_probs(features, fcst_high=72.0,
                                         run_hour=0, date_key="2026-03-11")
    assert probs is not None
    assert isinstance(probs, dict)
    # Probabilities should sum to ~1.0
    assert abs(sum(probs.values()) - 1.0) < 0.01
    # All probabilities should be non-negative
    assert all(p >= 0 for p in probs.values())

def test_predict_without_fit_returns_none():
    """predict_bracket_probs() should return None if not fitted."""
    model = QRModel()
    features = np.zeros(23)
    probs = model.predict_bracket_probs(features, fcst_high=72.0,
                                         run_hour=0, date_key="2026-03-11")
    assert probs is None

def test_fit_rejects_insufficient_data():
    """fit() should return None if fewer than MIN_SAMPLES rows."""
    model = QRModel()
    X = np.random.randn(30, 23)  # < 60 MIN_SAMPLES
    y = np.random.randn(30)
    coeffs = model.fit(X, y, run_hour=0, date_key="2026-03-11")
    assert coeffs is None
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/russellrudd/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.model'`

**Step 3: Implement `services/model.py`**

Extract from `autoresearch/experiment.py`:
- `fit_quantile_regression()` → `QRModel._fit_single_quantile()`
- Feature standardization logic → `QRModel.fit()`
- `build_bracket_probs()` logic → `QRModel._build_cdf()` and `QRModel.predict_bracket_probs()`
- Constants: QUANTILES, WINDOW, MIN_SAMPLES, LAMBDA factors, RADIUS

```python
# services/model.py
"""Quantile Regression model for temperature bracket prediction.

Extracted from autoresearch/experiment.py. Same math, packaged for live use.
"""
import numpy as np
from scipy.optimize import linprog
from typing import Dict, List, Optional, Tuple

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
WINDOW = 180
MIN_SAMPLES = 60
LAMBDA_UPPER_FACTOR = 0.29
LAMBDA_LOWER_FACTOR = 0.36
RADIUS = 18
MIN_BRACKET_PROB = 0.0001


class QRModel:
    def __init__(self):
        self._coeff_cache = {}   # (run_hour, date_key) → list of 7 coeff arrays
        self._scale_cache = {}   # (run_hour, date_key) → (feat_mean, feat_std)

    def fit(self, X_train, y_train, run_hour, date_key):
        # type: (np.ndarray, np.ndarray, int, str) -> Optional[List[np.ndarray]]
        """Fit 7 quantile regressions. Returns coefficients or None."""
        if len(X_train) < MIN_SAMPLES:
            return None

        feat_mean = X_train.mean(axis=0)
        feat_std = X_train.std(axis=0)
        feat_std[feat_std < 1e-10] = 1.0
        X_scaled = (X_train - feat_mean) / feat_std

        coefficients = []
        for tau in QUANTILES:
            coeff = self._fit_single_quantile(X_scaled, y_train, tau)
            if coeff is None:
                return None
            coefficients.append(coeff)

        key = (run_hour, date_key)
        self._coeff_cache[key] = coefficients
        self._scale_cache[key] = (feat_mean, feat_std)
        return coefficients

    def predict_bracket_probs(self, features, fcst_high, run_hour, date_key):
        # type: (np.ndarray, float, int, str) -> Optional[Dict[int, float]]
        """Predict bracket probabilities for given features."""
        key = (run_hour, date_key)
        coefficients = self._coeff_cache.get(key)
        scale = self._scale_cache.get(key)
        if coefficients is None or scale is None:
            return None

        feat_mean, feat_std = scale
        features_scaled = (features - feat_mean) / feat_std
        x_row = np.concatenate([[1.0], features_scaled])

        error_quantiles = [float(x_row @ coeff) for coeff in coefficients]
        temp_quantiles = [fcst_high - eq for eq in error_quantiles]

        return self._build_bracket_probs(temp_quantiles, fcst_high)

    def _fit_single_quantile(self, X_scaled, y, tau):
        # type: (np.ndarray, np.ndarray, float) -> Optional[np.ndarray]
        """Solve single quantile regression via LP. Mirrors experiment.py."""
        n, p = X_scaled.shape
        X_aug = np.column_stack([np.ones(n), X_scaled])
        p_aug = p + 1

        # Decision vars: [beta (p_aug), u+ (n), u- (n)]
        c = np.concatenate([
            np.zeros(p_aug),
            np.full(n, tau),
            np.full(n, 1.0 - tau)
        ])

        # Equality: X_aug @ beta + u+ - u- = y
        from scipy.sparse import hstack as sp_hstack, eye as sp_eye, csc_matrix
        A_eq = sp_hstack([
            csc_matrix(X_aug),
            sp_eye(n, format='csc'),
            -sp_eye(n, format='csc')
        ], format='csc')
        b_eq = y

        bounds = [(None, None)] * p_aug + [(0, None)] * (2 * n)

        try:
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                           method='highs', options={'presolve': True, 'time_limit': 10.0})
            if result.success:
                return result.x[:p_aug]
            return None
        except Exception:
            return None

    def _build_bracket_probs(self, temp_quantiles, center):
        # type: (List[float], float) -> Dict[int, float]
        """Build piecewise-linear CDF with exp tails → bracket probabilities."""
        sorted_pairs = sorted(zip(temp_quantiles, [1.0 - t for t in QUANTILES]))

        if len(sorted_pairs) < 2:
            return {}

        temps = [p[0] for p in sorted_pairs]
        taus = [p[1] for p in sorted_pairs]

        spread_lower = max(temps[1] - temps[0], 0.5)
        spread_upper = max(temps[-1] - temps[-2], 0.5)
        lambda_lower = LAMBDA_LOWER_FACTOR / spread_lower
        lambda_upper = LAMBDA_UPPER_FACTOR / spread_upper

        def cdf_at(t):
            if t <= temps[0]:
                return taus[0] * np.exp(-lambda_lower * (temps[0] - t))
            if t >= temps[-1]:
                return 1.0 - (1.0 - taus[-1]) * np.exp(-lambda_upper * (t - temps[-1]))
            for i in range(len(temps) - 1):
                if temps[i] <= t <= temps[i + 1]:
                    frac = (t - temps[i]) / (temps[i + 1] - temps[i])
                    return taus[i] + frac * (taus[i + 1] - taus[i])
            return 0.5

        center_int = int(round(center))
        bracket_probs = {}
        for k in range(center_int - RADIUS, center_int + RADIUS + 1, 2):
            prob = cdf_at(k + 1.0) - cdf_at(float(k))
            if prob >= MIN_BRACKET_PROB:
                bracket_probs[k] = prob

        # Normalize
        total = sum(bracket_probs.values())
        if total > 0:
            bracket_probs = {k: v / total for k, v in bracket_probs.items()}

        return bracket_probs
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/russellrudd/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_model.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add services/model.py tests/test_model.py
git commit -m "feat: extract QR model from experiment.py into services/model.py"
```

---

## Task 2: Feature Builder (`services/feature_builder.py`)

Build the 23-feature vector from live DB state for a given target date and update hour.

**Files:**
- Create: `services/feature_builder.py`
- Create: `tests/test_feature_builder.py`
- Reference: `autoresearch/experiment.py:get_training_data()` for feature definitions

**Step 1: Write the failing test**

```python
# tests/test_feature_builder.py
import duckdb
import pytest
import numpy as np
from datetime import date, datetime
from services.feature_builder import FeatureBuilder

@pytest.fixture
def test_db(tmp_path):
    """Create a minimal test DB with required tables and sample data."""
    db_path = str(tmp_path / "test.duckdb")
    con = duckdb.connect(db_path)

    # Create tables matching production schema
    con.execute("""
        CREATE TABLE forecasts (
            station_id VARCHAR, model_run TIMESTAMP, valid_at TIMESTAMP,
            temp_f DOUBLE, model_name VARCHAR DEFAULT 'hrrr',
            fxx INTEGER, is_spinup BOOLEAN DEFAULT FALSE
        )
    """)
    con.execute("""
        CREATE TABLE forecast_extended (
            station_id VARCHAR, model_run TIMESTAMP, valid_at TIMESTAMP,
            model_name VARCHAR, dewpoint_2m_f DOUBLE, humidity_2m DOUBLE,
            wind_gusts_10m DOUBLE, cape DOUBLE, shortwave_rad DOUBLE,
            total_precip DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE nws_daily (
            station_id VARCHAR, obs_date DATE, max_temp_f DOUBLE,
            min_temp_f DOUBLE, source VARCHAR DEFAULT 'NWS_CLI',
            ingested_at TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE drift_signals (
            city VARCHAR, calculated_at TIMESTAMP, model_run TIMESTAMP,
            drift_score DOUBLE, slope_divergence DOUBLE,
            forecast_trend DOUBLE, confidence DOUBLE, projected_high DOUBLE
        )
    """)

    con.close()
    return db_path

def test_feature_vector_length(test_db):
    """Feature vector should have exactly 23 elements."""
    builder = FeatureBuilder(test_db)
    # Will return None if insufficient data, but interface should be correct
    result = builder.build_features(target_date=date(2026, 3, 12), update_hour=6)
    # With empty DB, should return None
    assert result is None

def test_feature_names():
    """Feature names should match the 23 expected features."""
    assert len(FeatureBuilder.FEATURE_NAMES) == 23
    assert "fcst_high" in FeatureBuilder.FEATURE_NAMES
    assert "lag_error" in FeatureBuilder.FEATURE_NAMES
    assert "cum_div" in FeatureBuilder.FEATURE_NAMES

def test_training_data_shape(test_db):
    """get_training_data should return (X, y) with correct dimensions."""
    builder = FeatureBuilder(test_db)
    result = builder.get_training_data(
        target_date=date(2026, 3, 12),
        update_hour=0,
        window_days=180
    )
    # Empty DB → None
    assert result is None
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/russellrudd/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_feature_builder.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement `services/feature_builder.py`**

Adapt `get_training_data()` from `autoresearch/experiment.py` for both historical training and live prediction. The SQL queries should match experiment.py's feature construction exactly.

Key implementation details:
- Uses `gold_multi_model_features` view if available, falls back to manual JOINs
- `build_features()` returns `(feature_vector, fcst_high)` tuple or None
- `get_training_data()` returns `(X, y, dates)` tuple or None — same as experiment.py but parameterized
- Both use the same 23 feature columns in the same order as experiment.py

Read `autoresearch/experiment.py:get_training_data()` for the exact SQL query and feature column order. Replicate that query, parameterized by target_date and window.

**Step 4: Run tests to verify they pass**

Run: `cd /Users/russellrudd/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_feature_builder.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add services/feature_builder.py tests/test_feature_builder.py
git commit -m "feat: add feature builder for live QR model prediction"
```

---

## Task 3: Paper Config Table & Circuit Breakers

Add a `paper_config` table for runtime-adjustable circuit breaker thresholds, and implement the breaker logic.

**Files:**
- Modify: `core/db.py` (add table creation in `init_db()`)
- Create: `services/circuit_breakers.py`
- Create: `tests/test_circuit_breakers.py`

**Step 1: Write the failing test**

```python
# tests/test_circuit_breakers.py
import duckdb
import pytest
from datetime import date, datetime
from services.circuit_breakers import CircuitBreakers

DEFAULTS = {
    "max_daily_loss_cents": -1000,   # -$10.00
    "max_open_positions": 5,
    "max_per_bracket": 2,
    "min_edge_pct": 5.0,
    "cooldown_minutes": 30,
    "kill_switch": False,
}

@pytest.fixture
def test_db(tmp_path):
    db_path = str(tmp_path / "test.duckdb")
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE paper_config (
            key VARCHAR PRIMARY KEY,
            value VARCHAR,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE paper_positions (
            id INTEGER, city VARCHAR, event_date DATE,
            bracket_floor INTEGER, bracket_cap INTEGER,
            direction VARCHAR, entry_price DOUBLE, entry_time TIMESTAMP,
            exit_time TIMESTAMP, net_pnl DOUBLE, status VARCHAR DEFAULT 'open',
            contracts INTEGER DEFAULT 1, exit_reason VARCHAR
        )
    """)
    for k, v in DEFAULTS.items():
        con.execute("INSERT INTO paper_config VALUES (?, ?, CURRENT_TIMESTAMP)", [k, str(v)])
    con.close()
    return db_path

def test_kill_switch_blocks_all(test_db):
    con = duckdb.connect(test_db)
    con.execute("UPDATE paper_config SET value='True' WHERE key='kill_switch'")
    con.close()
    cb = CircuitBreakers(test_db)
    allowed, reason = cb.check(
        bracket_floor=72, bracket_cap=74,
        edge_pct=10.0, market_date=date(2026, 3, 12)
    )
    assert not allowed
    assert "kill_switch" in reason

def test_min_edge_blocks_low_edge(test_db):
    cb = CircuitBreakers(test_db)
    allowed, reason = cb.check(
        bracket_floor=72, bracket_cap=74,
        edge_pct=3.0, market_date=date(2026, 3, 12)
    )
    assert not allowed
    assert "edge" in reason.lower()

def test_sufficient_edge_passes(test_db):
    cb = CircuitBreakers(test_db)
    allowed, reason = cb.check(
        bracket_floor=72, bracket_cap=74,
        edge_pct=8.0, market_date=date(2026, 3, 12)
    )
    assert allowed

def test_max_daily_loss_blocks(test_db):
    """After losing $10+, no new entries."""
    con = duckdb.connect(test_db)
    # Insert a closed losing position for today
    con.execute("""
        INSERT INTO paper_positions VALUES
        (1, 'nyc', '2026-03-12', 72, 74, 'YES', 50.0,
         '2026-03-12 10:00:00', '2026-03-12 17:00:00',
         -1100.0, 'closed', 1, 'settlement')
    """)
    con.close()
    cb = CircuitBreakers(test_db)
    allowed, reason = cb.check(
        bracket_floor=68, bracket_cap=70,
        edge_pct=10.0, market_date=date(2026, 3, 12)
    )
    assert not allowed
    assert "daily_loss" in reason.lower()

def test_max_open_positions_blocks(test_db):
    """With 5 open positions, no new entries."""
    con = duckdb.connect(test_db)
    for i in range(5):
        floor = 60 + i * 2
        con.execute("""
            INSERT INTO paper_positions VALUES
            (?, 'nyc', '2026-03-12', ?, ?, 'YES', 50.0,
             '2026-03-12 10:00:00', NULL, NULL, 'open', 1, NULL)
        """, [i + 1, floor, floor + 2])
    con.close()
    cb = CircuitBreakers(test_db)
    allowed, reason = cb.check(
        bracket_floor=80, bracket_cap=82,
        edge_pct=10.0, market_date=date(2026, 3, 12)
    )
    assert not allowed
    assert "max_open" in reason.lower()

def test_cooldown_blocks_recent_exit(test_db):
    """Can't re-enter a bracket within 30 min of exit."""
    con = duckdb.connect(test_db)
    con.execute("""
        INSERT INTO paper_positions VALUES
        (1, 'nyc', '2026-03-12', 72, 74, 'YES', 50.0,
         '2026-03-12 10:00:00', '2026-03-12 10:15:00',
         -50.0, 'closed', 1, 'edge_reversal')
    """)
    con.close()
    cb = CircuitBreakers(test_db)
    # Check within 30 min of exit (mock "now" via parameter)
    allowed, reason = cb.check(
        bracket_floor=72, bracket_cap=74,
        edge_pct=10.0, market_date=date(2026, 3, 12),
        now=datetime(2026, 3, 12, 10, 30)
    )
    assert not allowed
    assert "cooldown" in reason.lower()
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/russellrudd/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_circuit_breakers.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Add `paper_config` table to `core/db.py`**

Add to `init_db()`:

```python
con.execute("""
    CREATE TABLE IF NOT EXISTS paper_config (
        key VARCHAR PRIMARY KEY,
        value VARCHAR,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
# Seed defaults if empty
count = con.execute("SELECT COUNT(*) FROM paper_config").fetchone()[0]
if count == 0:
    defaults = [
        ("max_daily_loss_cents", "-1000"),
        ("max_open_positions", "5"),
        ("max_per_bracket", "2"),
        ("min_edge_pct", "5.0"),
        ("cooldown_minutes", "30"),
        ("kill_switch", "False"),
    ]
    con.executemany(
        "INSERT INTO paper_config VALUES (?, ?, CURRENT_TIMESTAMP)", defaults
    )
```

**Step 4: Implement `services/circuit_breakers.py`**

```python
# services/circuit_breakers.py
"""Circuit breakers for paper trading. All checks must pass before entry."""
import duckdb
from datetime import date, datetime, timedelta
from typing import Optional, Tuple


class CircuitBreakers:
    def __init__(self, db_path):
        # type: (str) -> None
        self.db_path = db_path

    def _get_config(self):
        # type: () -> dict
        con = duckdb.connect(self.db_path, read_only=True)
        rows = con.execute("SELECT key, value FROM paper_config").fetchall()
        con.close()
        return {k: v for k, v in rows}

    def check(self, bracket_floor, bracket_cap, edge_pct, market_date, now=None):
        # type: (int, int, float, date, Optional[datetime]) -> Tuple[bool, str]
        """Run all circuit breakers. Returns (allowed, reason)."""
        if now is None:
            now = datetime.utcnow()

        cfg = self._get_config()
        con = duckdb.connect(self.db_path, read_only=True)

        try:
            # 1. Kill switch
            if cfg.get("kill_switch", "False").lower() == "true":
                return False, "kill_switch: trading disabled"

            # 2. Min edge
            min_edge = float(cfg.get("min_edge_pct", "5.0"))
            if edge_pct < min_edge:
                return False, f"edge too low: {edge_pct:.1f}% < {min_edge:.1f}%"

            # 3. Max daily loss
            max_loss = float(cfg.get("max_daily_loss_cents", "-1000"))
            daily_pnl = con.execute("""
                SELECT COALESCE(SUM(net_pnl), 0)
                FROM paper_positions
                WHERE event_date = ? AND status = 'closed'
            """, [market_date]).fetchone()[0]
            if daily_pnl <= max_loss:
                return False, f"daily_loss breaker: {daily_pnl:.0f}c <= {max_loss:.0f}c"

            # 4. Max open positions
            max_open = int(cfg.get("max_open_positions", "5"))
            open_count = con.execute("""
                SELECT COUNT(*) FROM paper_positions WHERE status = 'open'
            """).fetchone()[0]
            if open_count >= max_open:
                return False, f"max_open breaker: {open_count} >= {max_open}"

            # 5. Max per bracket
            max_bracket = int(cfg.get("max_per_bracket", "2"))
            bracket_count = con.execute("""
                SELECT COALESCE(SUM(contracts), 0)
                FROM paper_positions
                WHERE status = 'open'
                  AND bracket_floor = ? AND bracket_cap = ?
            """, [bracket_floor, bracket_cap]).fetchone()[0]
            if bracket_count >= max_bracket:
                return False, f"max_per_bracket: {bracket_count} >= {max_bracket}"

            # 6. Cooldown
            cooldown_min = int(cfg.get("cooldown_minutes", "30"))
            last_exit = con.execute("""
                SELECT MAX(exit_time)
                FROM paper_positions
                WHERE bracket_floor = ? AND bracket_cap = ?
                  AND status = 'closed' AND exit_reason != 'settlement'
            """, [bracket_floor, bracket_cap]).fetchone()[0]
            if last_exit is not None:
                elapsed = (now - last_exit).total_seconds() / 60.0
                if elapsed < cooldown_min:
                    return False, f"cooldown: {elapsed:.0f}m < {cooldown_min}m"

            return True, "all_clear"
        finally:
            con.close()
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/russellrudd/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_circuit_breakers.py -v`
Expected: All 7 tests PASS

**Step 6: Commit**

```bash
git add core/db.py services/circuit_breakers.py tests/test_circuit_breakers.py
git commit -m "feat: add paper_config table and circuit breaker logic"
```

---

## Task 4: Strategy Engine (`services/strategy_engine.py`)

The core service: checks for new data, builds features, runs model, calculates edge, sends decisions.

**Files:**
- Create: `services/strategy_engine.py`
- Create: `tests/test_strategy_engine.py`
- Reference: `services/model.py` (Task 1), `services/feature_builder.py` (Task 2)

**Step 1: Write the failing test**

```python
# tests/test_strategy_engine.py
import pytest
from unittest.mock import MagicMock, patch
from datetime import date
from services.strategy_engine import StrategyEngine

def test_compute_edge():
    """Edge = model_prob - market_implied_prob."""
    engine = StrategyEngine.__new__(StrategyEngine)
    edge = engine._compute_edge(model_prob=0.20, market_ask_cents=12)
    # Model says 20%, market asks 12c (12% implied) → +8% edge on YES
    assert abs(edge - 8.0) < 0.1

def test_no_edge_no_trade():
    """Below 5% edge → no trade signal."""
    engine = StrategyEngine.__new__(StrategyEngine)
    engine.min_edge_pct = 5.0
    signals = engine._generate_signals(
        bracket_probs={72: 0.15, 74: 0.12},
        market_prices={72: {"yes_ask": 14, "yes_bid": 13},
                       74: {"yes_ask": 11, "yes_bid": 10}},
    )
    # 15% model vs 14c ask = +1% edge → below threshold
    # 12% model vs 11c ask = +1% edge → below threshold
    assert len(signals) == 0

def test_positive_edge_generates_signal():
    """Above 5% edge → trade signal."""
    engine = StrategyEngine.__new__(StrategyEngine)
    engine.min_edge_pct = 5.0
    signals = engine._generate_signals(
        bracket_probs={72: 0.25},
        market_prices={72: {"yes_ask": 15, "yes_bid": 13}},
    )
    # 25% model vs 15c ask = +10% edge → signal
    assert len(signals) == 1
    assert signals[0]["bracket_floor"] == 72
    assert signals[0]["direction"] == "YES"
    assert signals[0]["edge_pct"] > 5.0

def test_negative_edge_generates_no_signal():
    """Model says 10%, market asks 20c → sell NO (but we only buy YES/NO for now)."""
    engine = StrategyEngine.__new__(StrategyEngine)
    engine.min_edge_pct = 5.0
    signals = engine._generate_signals(
        bracket_probs={72: 0.10},
        market_prices={72: {"yes_ask": 20, "yes_bid": 18,
                           "no_ask": 82, "no_bid": 80}},
    )
    # Model 10% vs market 20% → model thinks YES overpriced
    # → buy NO at 82c? Only if edge is sufficient
    # NO edge = (1-0.10) - (1-0.20) = 0.90 - 0.80 = +10%
    assert len(signals) == 1
    assert signals[0]["direction"] == "NO"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/russellrudd/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_strategy_engine.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement `services/strategy_engine.py`**

```python
# services/strategy_engine.py
"""Strategy engine: data freshness → features → model → edge → trade signals."""
import asyncio
import duckdb
import numpy as np
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from loguru import logger

from services.model import QRModel
from services.feature_builder import FeatureBuilder
from services.circuit_breakers import CircuitBreakers

CYCLE_INTERVAL = 300  # 5 minutes
MIN_EDGE_PCT = 5.0


class StrategyEngine:
    def __init__(self, db_path, paper_trader):
        self.db_path = db_path
        self.paper_trader = paper_trader
        self.model = QRModel()
        self.feature_builder = FeatureBuilder(db_path)
        self.circuit_breakers = CircuitBreakers(db_path)
        self.min_edge_pct = MIN_EDGE_PCT
        self._last_data_hash = None

    async def run(self):
        """Main async loop — runs every CYCLE_INTERVAL seconds."""
        logger.info("StrategyEngine started")
        while True:
            try:
                await self._cycle()
            except Exception as e:
                logger.error(f"StrategyEngine cycle error: {e}")
            await asyncio.sleep(CYCLE_INTERVAL)

    async def _cycle(self):
        """Single evaluation cycle."""
        # 1. Determine target date (tomorrow)
        now_et = datetime.now()  # Assume server is ET or convert
        target_date = (now_et + timedelta(days=1)).date()
        update_hour = now_et.hour

        # 2. Check data freshness
        data_hash = self._get_data_hash()
        if data_hash == self._last_data_hash:
            logger.debug("No new data since last cycle, skipping")
            return
        self._last_data_hash = data_hash

        # 3. Get training data and fit model
        date_key = str(target_date)
        training = self.feature_builder.get_training_data(
            target_date=target_date,
            update_hour=update_hour,
            window_days=180
        )
        if training is None:
            logger.warning("Insufficient training data")
            return

        X_train, y_train, _ = training
        coeffs = self.model.fit(X_train, y_train, update_hour, date_key)
        if coeffs is None:
            logger.warning("Model fit failed")
            return

        # 4. Build today's features
        result = self.feature_builder.build_features(target_date, update_hour)
        if result is None:
            logger.warning("Could not build features for prediction")
            return
        features, fcst_high = result

        # 5. Predict bracket probabilities
        bracket_probs = self.model.predict_bracket_probs(
            features, fcst_high, update_hour, date_key
        )
        if not bracket_probs:
            logger.warning("No bracket probabilities produced")
            return

        # 6. Get market prices
        market_prices = self._get_market_prices(target_date)
        if not market_prices:
            logger.warning("No market prices available")
            return

        # 7. Generate trade signals
        signals = self._generate_signals(bracket_probs, market_prices)
        logger.info(f"Generated {len(signals)} trade signals for {target_date}")

        # 8. Check edge reversals for open positions
        await self._check_edge_reversals(bracket_probs, market_prices, target_date)

        # 9. Send signals to paper trader
        for signal in signals:
            allowed, reason = self.circuit_breakers.check(
                bracket_floor=signal["bracket_floor"],
                bracket_cap=signal["bracket_floor"] + 2,
                edge_pct=signal["edge_pct"],
                market_date=target_date
            )
            if allowed:
                await self.paper_trader.enter_position(
                    city="nyc",
                    event_date=target_date,
                    bracket_floor=signal["bracket_floor"],
                    bracket_cap=signal["bracket_floor"] + 2,
                    direction=signal["direction"],
                    model_prob=signal["model_prob"],
                    market_price=signal["market_price"],
                    edge=signal["edge_pct"] / 100.0,
                )
                logger.info(f"Entry: {signal['direction']} {signal['bracket_floor']}-"
                          f"{signal['bracket_floor']+2} edge={signal['edge_pct']:.1f}%")
            else:
                logger.debug(f"Blocked: {reason}")

    def _compute_edge(self, model_prob, market_ask_cents):
        # type: (float, int) -> float
        """Edge in percentage points: model_prob - market_implied."""
        return (model_prob - market_ask_cents / 100.0) * 100.0

    def _generate_signals(self, bracket_probs, market_prices):
        # type: (Dict[int, float], Dict[int, dict]) -> List[dict]
        """Generate trade signals where edge exceeds threshold."""
        signals = []
        for bracket_floor, model_prob in bracket_probs.items():
            if bracket_floor not in market_prices:
                continue
            mp = market_prices[bracket_floor]

            # YES edge: model_prob vs yes_ask
            if "yes_ask" in mp and mp["yes_ask"] is not None and mp["yes_ask"] > 0:
                yes_edge = self._compute_edge(model_prob, mp["yes_ask"])
                if yes_edge >= self.min_edge_pct:
                    signals.append({
                        "bracket_floor": bracket_floor,
                        "direction": "YES",
                        "model_prob": model_prob,
                        "market_price": mp["yes_ask"],
                        "edge_pct": yes_edge,
                    })
                    continue  # Don't also signal NO on same bracket

            # NO edge: (1 - model_prob) vs no_ask
            if "no_ask" in mp and mp["no_ask"] is not None and mp["no_ask"] > 0:
                no_model = 1.0 - model_prob
                no_edge = self._compute_edge(no_model, mp["no_ask"])
                if no_edge >= self.min_edge_pct:
                    signals.append({
                        "bracket_floor": bracket_floor,
                        "direction": "NO",
                        "model_prob": no_model,
                        "market_price": mp["no_ask"],
                        "edge_pct": no_edge,
                    })

        return signals

    def _get_market_prices(self, target_date):
        # type: (date) -> Dict[int, dict]
        """Get latest market prices per bracket from market_ticks."""
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            rows = con.execute("""
                SELECT floor_strike, cap_strike,
                       yes_bid * 100 as yes_bid, yes_ask * 100 as yes_ask,
                       no_bid * 100 as no_bid, no_ask * 100 as no_ask
                FROM market_ticks
                WHERE city = 'nyc'
                  AND captured_at = (SELECT MAX(captured_at) FROM market_ticks WHERE city = 'nyc')
            """).fetchall()
            prices = {}
            for row in rows:
                floor = int(row[0])
                prices[floor] = {
                    "yes_bid": row[2], "yes_ask": row[3],
                    "no_bid": row[4], "no_ask": row[5],
                }
            return prices
        finally:
            con.close()

    def _get_data_hash(self):
        # type: () -> Optional[str]
        """Hash of latest data timestamps to detect new data."""
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            parts = []
            for q in [
                "SELECT MAX(model_run) FROM forecasts WHERE model_name='hrrr'",
                "SELECT MAX(model_run) FROM forecasts WHERE model_name='gfs'",
                "SELECT MAX(model_run) FROM forecasts WHERE model_name='ecmwf'",
                "SELECT MAX(calculated_at) FROM drift_signals",
                "SELECT MAX(captured_at) FROM market_ticks",
            ]:
                try:
                    val = con.execute(q).fetchone()[0]
                    parts.append(str(val))
                except Exception:
                    parts.append("none")
            return "|".join(parts)
        finally:
            con.close()

    async def _check_edge_reversals(self, bracket_probs, market_prices, target_date):
        """Check open positions for edge reversal — signal exit if edge flipped."""
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            open_positions = con.execute("""
                SELECT id, bracket_floor, bracket_cap, direction, model_prob
                FROM paper_positions
                WHERE status = 'open' AND event_date = ?
            """, [target_date]).fetchall()
        finally:
            con.close()

        for pos_id, floor, cap, direction, orig_prob in open_positions:
            model_prob = bracket_probs.get(floor, 0.0)
            if floor in market_prices:
                mp = market_prices[floor]
                if direction == "YES":
                    current_edge = self._compute_edge(model_prob, mp.get("yes_bid", 50))
                else:
                    current_edge = self._compute_edge(1.0 - model_prob, mp.get("no_bid", 50))

                if current_edge < 0:
                    # Edge reversed — signal exit
                    exit_price = mp.get("yes_bid" if direction == "YES" else "no_bid", 50)
                    await self.paper_trader.exit_position(
                        position_id=pos_id,
                        exit_price=exit_price,
                        reason="edge_reversal"
                    )
                    logger.info(f"Edge reversal exit: position {pos_id}, "
                              f"current_edge={current_edge:.1f}%")
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/russellrudd/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_strategy_engine.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add services/strategy_engine.py tests/test_strategy_engine.py
git commit -m "feat: add strategy engine for continuous edge evaluation"
```

---

## Task 5: Enhance Paper Trader (`services/paper_trader.py`)

Add `enter_position()` and `exit_position()` public methods that the strategy engine calls, replacing the stub `_check_entries()` and `_check_exits()`.

**Files:**
- Modify: `services/paper_trader.py`
- Modify: `tests/test_paper_trader.py` (add new tests)

**Step 1: Write the failing test**

Add to `tests/test_paper_trader.py`:

```python
@pytest.mark.asyncio
async def test_enter_position_records_trade(paper_trader_with_db):
    """enter_position() should insert an open position."""
    pt = paper_trader_with_db
    await pt.enter_position(
        city="nyc", event_date=date(2026, 3, 12),
        bracket_floor=72, bracket_cap=74,
        direction="YES", model_prob=0.20,
        market_price=12.0, edge=0.08
    )
    con = duckdb.connect(pt.db_path, read_only=True)
    row = con.execute("SELECT * FROM paper_positions WHERE bracket_floor=72").fetchone()
    con.close()
    assert row is not None
    # status should be 'open'
    assert "open" in str(row)

@pytest.mark.asyncio
async def test_exit_position_closes_trade(paper_trader_with_db):
    """exit_position() should mark position as closed with reason."""
    pt = paper_trader_with_db
    await pt.enter_position(
        city="nyc", event_date=date(2026, 3, 12),
        bracket_floor=72, bracket_cap=74,
        direction="YES", model_prob=0.20,
        market_price=12.0, edge=0.08
    )
    # Get the position ID
    con = duckdb.connect(pt.db_path, read_only=True)
    pos_id = con.execute("SELECT id FROM paper_positions WHERE bracket_floor=72").fetchone()[0]
    con.close()

    await pt.exit_position(pos_id, exit_price=18.0, reason="edge_reversal")

    con = duckdb.connect(pt.db_path, read_only=True)
    row = con.execute("SELECT status, exit_reason, exit_price FROM paper_positions WHERE id=?",
                       [pos_id]).fetchone()
    con.close()
    assert row[0] == "closed"
    assert row[1] == "edge_reversal"
    assert row[2] == 18.0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/russellrudd/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_paper_trader.py -v -k "enter_position or exit_position"`
Expected: FAIL — `AttributeError: 'PaperTrader' object has no attribute 'enter_position'`

**Step 3: Add methods to `services/paper_trader.py`**

Add two public async methods:

```python
async def enter_position(self, city, event_date, bracket_floor, bracket_cap,
                          direction, model_prob, market_price, edge):
    """Public entry point called by StrategyEngine."""
    self._record_entry(
        city=city, event_date=event_date,
        bracket_floor=bracket_floor, bracket_cap=bracket_cap,
        direction=direction, model_prob=model_prob,
        market_price=market_price,
        entry_price=market_price,  # paper trade at ask
        contracts=1
    )

async def exit_position(self, position_id, exit_price, reason):
    """Close a position early (edge reversal, kill switch, etc.)."""
    con = duckdb.connect(self.db_path)
    try:
        entry = con.execute(
            "SELECT entry_price, contracts, direction FROM paper_positions WHERE id=?",
            [position_id]
        ).fetchone()
        if entry is None:
            return

        entry_price, contracts, direction = entry
        exit_fee = self._compute_fee(exit_price, contracts)

        if direction == "YES":
            gross = (exit_price - entry_price) * contracts
        else:
            gross = (entry_price - exit_price) * contracts

        entry_fee = self._compute_fee(entry_price, contracts)
        net = gross - entry_fee - exit_fee

        con.execute("""
            UPDATE paper_positions
            SET exit_price = ?, exit_time = CURRENT_TIMESTAMP,
                exit_reason = ?, status = 'closed',
                gross_pnl = ?, fees = ?, net_pnl = ?
            WHERE id = ?
        """, [exit_price, reason, gross, entry_fee + exit_fee, net, position_id])
    finally:
        con.close()
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/russellrudd/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_paper_trader.py -v`
Expected: All tests PASS (old + new)

**Step 5: Commit**

```bash
git add services/paper_trader.py tests/test_paper_trader.py
git commit -m "feat: add enter/exit position methods to paper trader"
```

---

## Task 6: Settlement Service (`services/settlement.py`)

Polls for NWS CLI data and triggers auto-settlement of open positions.

**Files:**
- Create: `services/settlement.py`
- Create: `tests/test_settlement.py`

**Step 1: Write the failing test**

```python
# tests/test_settlement.py
import duckdb
import pytest
from datetime import date, datetime
from services.settlement import SettlementService

@pytest.fixture
def test_db(tmp_path):
    db_path = str(tmp_path / "test.duckdb")
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE nws_daily (
            station_id VARCHAR, obs_date DATE, max_temp_f DOUBLE,
            min_temp_f DOUBLE, source VARCHAR, ingested_at TIMESTAMP,
            UNIQUE (station_id, obs_date)
        )
    """)
    con.execute("""
        CREATE TABLE paper_positions (
            id INTEGER, city VARCHAR, event_date DATE,
            bracket_floor INTEGER, bracket_cap INTEGER,
            direction VARCHAR, entry_price DOUBLE, entry_time TIMESTAMP,
            exit_price DOUBLE, exit_time TIMESTAMP,
            settled_yes BOOLEAN, gross_pnl DOUBLE, fees DOUBLE,
            net_pnl DOUBLE, status VARCHAR DEFAULT 'open',
            exit_reason VARCHAR, contracts INTEGER DEFAULT 1,
            model_prob DOUBLE, market_price DOUBLE, edge DOUBLE,
            unrealized_pnl DOUBLE DEFAULT 0.0
        )
    """)
    con.close()
    return db_path

def test_settlement_resolves_winning_yes(test_db):
    """YES position wins when actual high falls in bracket."""
    con = duckdb.connect(test_db)
    con.execute("""
        INSERT INTO nws_daily VALUES
        ('KNYC', '2026-03-11', 73.0, 55.0, 'NWS_CLI', CURRENT_TIMESTAMP)
    """)
    con.execute("""
        INSERT INTO paper_positions VALUES
        (1, 'nyc', '2026-03-11', 72, 74, 'YES', 30.0,
         '2026-03-11 10:00:00', NULL, NULL, NULL, NULL, NULL, NULL,
         'open', NULL, 1, 0.25, 15.0, 0.10, 0.0)
    """)
    con.close()

    svc = SettlementService(test_db)
    settled = svc.settle_date(date(2026, 3, 11))
    assert settled == 1

    con = duckdb.connect(test_db, read_only=True)
    row = con.execute("SELECT status, settled_yes, net_pnl FROM paper_positions WHERE id=1").fetchone()
    con.close()
    assert row[0] == "closed"
    assert row[1] == True
    assert row[2] > 0  # winner

def test_settlement_resolves_losing_yes(test_db):
    """YES position loses when actual high outside bracket."""
    con = duckdb.connect(test_db)
    con.execute("""
        INSERT INTO nws_daily VALUES
        ('KNYC', '2026-03-11', 68.0, 55.0, 'NWS_CLI', CURRENT_TIMESTAMP)
    """)
    con.execute("""
        INSERT INTO paper_positions VALUES
        (1, 'nyc', '2026-03-11', 72, 74, 'YES', 30.0,
         '2026-03-11 10:00:00', NULL, NULL, NULL, NULL, NULL, NULL,
         'open', NULL, 1, 0.25, 15.0, 0.10, 0.0)
    """)
    con.close()

    svc = SettlementService(test_db)
    settled = svc.settle_date(date(2026, 3, 11))
    assert settled == 1

    con = duckdb.connect(test_db, read_only=True)
    row = con.execute("SELECT status, settled_yes, net_pnl FROM paper_positions WHERE id=1").fetchone()
    con.close()
    assert row[0] == "closed"
    assert row[1] == False
    assert row[2] < 0  # loser

def test_no_settlement_without_cli(test_db):
    """No settlement if nws_daily has no NWS_CLI entry."""
    con = duckdb.connect(test_db)
    # ACIS source, not CLI
    con.execute("""
        INSERT INTO nws_daily VALUES
        ('KNYC', '2026-03-11', 73.0, 55.0, 'ACIS', CURRENT_TIMESTAMP)
    """)
    con.execute("""
        INSERT INTO paper_positions VALUES
        (1, 'nyc', '2026-03-11', 72, 74, 'YES', 30.0,
         '2026-03-11 10:00:00', NULL, NULL, NULL, NULL, NULL, NULL,
         'open', NULL, 1, 0.25, 15.0, 0.10, 0.0)
    """)
    con.close()

    svc = SettlementService(test_db)
    settled = svc.settle_date(date(2026, 3, 11))
    assert settled == 0  # No settlement from ACIS
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/russellrudd/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_settlement.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement `services/settlement.py`**

```python
# services/settlement.py
"""Auto-settlement service. Polls NWS CLI data and settles open paper positions."""
import asyncio
import duckdb
from datetime import date, datetime, timedelta
from loguru import logger
from typing import Optional

from services.paper_trader import PaperTrader

POLL_INTERVAL_PRELIMINARY = 600   # 10 min between 4-6 PM ET
POLL_INTERVAL_FINAL = 1800        # 30 min between 1-2 AM ET
POLL_INTERVAL_IDLE = 3600         # 1 hour otherwise


class SettlementService:
    def __init__(self, db_path, paper_trader=None):
        # type: (str, Optional[PaperTrader]) -> None
        self.db_path = db_path
        self.paper_trader = paper_trader

    async def run(self):
        """Main async loop — polls for settlement data."""
        logger.info("SettlementService started")
        while True:
            try:
                # Determine which dates need settlement
                unsettled = self._get_unsettled_dates()
                for d in unsettled:
                    count = self.settle_date(d)
                    if count > 0:
                        logger.info(f"Settled {count} positions for {d}")
            except Exception as e:
                logger.error(f"Settlement error: {e}")

            interval = self._get_poll_interval()
            await asyncio.sleep(interval)

    def settle_date(self, market_date):
        # type: (date) -> int
        """Settle all open positions for a given date. Returns count settled."""
        con = duckdb.connect(self.db_path)
        try:
            # Check for NWS CLI data (authoritative source only)
            cli = con.execute("""
                SELECT max_temp_f FROM nws_daily
                WHERE station_id = 'KNYC' AND obs_date = ?
                  AND source IN ('NWS_CLI', 'DSM')
            """, [market_date]).fetchone()

            if cli is None:
                return 0

            actual_high = cli[0]

            # Get open positions for this date
            positions = con.execute("""
                SELECT id, bracket_floor, bracket_cap, direction,
                       entry_price, contracts
                FROM paper_positions
                WHERE event_date = ? AND status = 'open'
            """, [market_date]).fetchall()

            settled_count = 0
            for pos_id, floor, cap, direction, entry_price, contracts in positions:
                # Bracket is [floor, cap) — cap exclusive
                in_bracket = floor <= actual_high < cap
                settled_yes = in_bracket

                if direction == "YES":
                    if settled_yes:
                        # Winner: receive 100c per contract
                        gross = (100.0 - entry_price) * contracts
                    else:
                        gross = -entry_price * contracts
                elif direction == "NO":
                    if not settled_yes:
                        gross = (100.0 - entry_price) * contracts
                    else:
                        gross = -entry_price * contracts

                # Fee at entry only (no settlement fee on Kalshi)
                fee = self._compute_fee(entry_price, contracts)
                net = gross - fee

                con.execute("""
                    UPDATE paper_positions
                    SET exit_price = ?, exit_time = CURRENT_TIMESTAMP,
                        settled_yes = ?, gross_pnl = ?, fees = ?,
                        net_pnl = ?, status = 'closed', exit_reason = 'settlement'
                    WHERE id = ?
                """, [100.0 if (direction == "YES" and settled_yes) or
                           (direction == "NO" and not settled_yes) else 0.0,
                      settled_yes, gross, fee, net, pos_id])
                settled_count += 1

            return settled_count
        finally:
            con.close()

    def _compute_fee(self, price_cents, contracts):
        # type: (float, int) -> float
        """Kalshi taker fee in cents."""
        import math
        p = price_cents / 100.0
        raw = 0.07 * contracts * p * (1.0 - p)
        return max(math.ceil(round(raw * 100, 10)) / 100.0 * 100, contracts * 1.0)

    def _get_unsettled_dates(self):
        # type: () -> list
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            rows = con.execute("""
                SELECT DISTINCT event_date FROM paper_positions
                WHERE status = 'open'
                ORDER BY event_date
            """).fetchall()
            return [r[0] for r in rows]
        finally:
            con.close()

    def _get_poll_interval(self):
        # type: () -> int
        """Adjust poll frequency based on time of day (ET)."""
        hour = datetime.now().hour  # Assuming ET
        if 16 <= hour <= 18:
            return POLL_INTERVAL_PRELIMINARY
        elif 1 <= hour <= 2:
            return POLL_INTERVAL_FINAL
        return POLL_INTERVAL_IDLE
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/russellrudd/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_settlement.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add services/settlement.py tests/test_settlement.py
git commit -m "feat: add settlement service for auto-settling paper positions"
```

---

## Task 7: Dashboard Trading Tab

Add a Trading tab to the existing FastAPI dashboard with live positions, P&L, and circuit breaker controls.

**Files:**
- Modify: `ui/web_dashboard.py` (add API endpoints + tab)
- Create: `tests/test_dashboard_trading.py`

**Step 1: Write the failing test**

```python
# tests/test_dashboard_trading.py
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

def test_trading_positions_endpoint():
    """GET /api/trading/positions should return open positions."""
    # This test verifies the endpoint exists and returns JSON
    from ui.web_dashboard import app
    client = TestClient(app)
    response = client.get("/api/trading/positions?city=nyc")
    assert response.status_code == 200
    data = response.json()
    assert "positions" in data

def test_trading_pnl_endpoint():
    """GET /api/trading/pnl should return P&L summary."""
    from ui.web_dashboard import app
    client = TestClient(app)
    response = client.get("/api/trading/pnl?city=nyc")
    assert response.status_code == 200
    data = response.json()
    assert "realized" in data
    assert "unrealized" in data
    assert "total" in data

def test_trading_breakers_endpoint():
    """GET /api/trading/breakers should return circuit breaker status."""
    from ui.web_dashboard import app
    client = TestClient(app)
    response = client.get("/api/trading/breakers")
    assert response.status_code == 200
    data = response.json()
    assert "breakers" in data

def test_kill_switch_toggle():
    """POST /api/trading/kill-switch should toggle the kill switch."""
    from ui.web_dashboard import app
    client = TestClient(app)
    response = client.post("/api/trading/kill-switch", json={"enabled": True})
    assert response.status_code == 200
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/russellrudd/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_dashboard_trading.py -v`
Expected: FAIL — endpoints don't exist yet

**Step 3: Add API endpoints to `ui/web_dashboard.py`**

Add these endpoints to the existing FastAPI app:

```python
@app.get("/api/trading/positions")
async def trading_positions(city: str = "nyc"):
    """Live positions with unrealized P&L."""
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        rows = con.execute("""
            SELECT p.id, p.bracket_floor, p.bracket_cap, p.direction,
                   p.contracts, p.entry_price, p.entry_time,
                   p.model_prob, p.edge, p.unrealized_pnl,
                   m.yes_bid * 100 as current_bid, m.yes_ask * 100 as current_ask
            FROM paper_positions p
            LEFT JOIN (
                SELECT floor_strike, cap_strike, yes_bid, yes_ask,
                       ROW_NUMBER() OVER (PARTITION BY floor_strike ORDER BY captured_at DESC) as rn
                FROM market_ticks WHERE city = ?
            ) m ON m.floor_strike = CAST(p.bracket_floor AS DOUBLE)
                AND m.cap_strike = CAST(p.bracket_cap AS DOUBLE)
                AND m.rn = 1
            WHERE p.status = 'open' AND p.city = ?
            ORDER BY p.entry_time DESC
        """, [city, city]).fetchall()

        positions = []
        for row in rows:
            entry_time = row[6]
            held = ""
            if entry_time:
                delta = datetime.utcnow() - entry_time
                hours, remainder = divmod(int(delta.total_seconds()), 3600)
                minutes = remainder // 60
                held = f"{hours}h {minutes}m"

            positions.append({
                "id": row[0], "bracket": f"{row[1]}-{row[2]}",
                "direction": row[3], "qty": row[4],
                "entry": row[5], "current_bid": row[10],
                "current_ask": row[11], "edge": row[8],
                "unrealized": row[9], "time_held": held,
            })
        return {"positions": positions}
    finally:
        con.close()

@app.get("/api/trading/pnl")
async def trading_pnl(city: str = "nyc"):
    """P&L summary: realized, unrealized, total."""
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        realized = con.execute("""
            SELECT COALESCE(SUM(net_pnl), 0) FROM paper_positions
            WHERE city = ? AND status = 'closed'
        """, [city]).fetchone()[0]

        unrealized = con.execute("""
            SELECT COALESCE(SUM(unrealized_pnl), 0) FROM paper_positions
            WHERE city = ? AND status = 'open'
        """, [city]).fetchone()[0]

        # Daily breakdown
        daily = con.execute("""
            SELECT event_date,
                   SUM(CASE WHEN status='closed' THEN net_pnl ELSE 0 END) as realized,
                   SUM(CASE WHEN status='open' THEN unrealized_pnl ELSE 0 END) as unrealized,
                   COUNT(*) as trades,
                   SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins
            FROM paper_positions WHERE city = ?
            GROUP BY event_date ORDER BY event_date
        """, [city]).fetchall()

        return {
            "realized": realized, "unrealized": unrealized,
            "total": realized + unrealized,
            "daily": [{"date": str(d[0]), "realized": d[1], "unrealized": d[2],
                       "trades": d[3], "wins": d[4]} for d in daily],
        }
    finally:
        con.close()

@app.get("/api/trading/breakers")
async def trading_breakers():
    """Current circuit breaker status."""
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        config = dict(con.execute("SELECT key, value FROM paper_config").fetchall())
        open_count = con.execute(
            "SELECT COUNT(*) FROM paper_positions WHERE status='open'"
        ).fetchone()[0]
        today_pnl = con.execute("""
            SELECT COALESCE(SUM(net_pnl), 0) FROM paper_positions
            WHERE status='closed' AND event_date = CURRENT_DATE
        """).fetchone()[0]

        return {"breakers": {
            "kill_switch": config.get("kill_switch", "False").lower() == "true",
            "max_daily_loss": {"threshold": float(config.get("max_daily_loss_cents", -1000)),
                              "current": today_pnl},
            "max_open": {"threshold": int(config.get("max_open_positions", 5)),
                        "current": open_count},
            "min_edge_pct": float(config.get("min_edge_pct", 5.0)),
            "cooldown_minutes": int(config.get("cooldown_minutes", 30)),
        }}
    finally:
        con.close()

@app.post("/api/trading/kill-switch")
async def toggle_kill_switch(body: dict):
    """Toggle the kill switch."""
    enabled = body.get("enabled", False)
    con = duckdb.connect(DB_PATH)
    try:
        con.execute("""
            UPDATE paper_config SET value = ?, updated_at = CURRENT_TIMESTAMP
            WHERE key = 'kill_switch'
        """, [str(enabled)])
        return {"kill_switch": enabled}
    finally:
        con.close()
```

Add Trading tab HTML template following existing tab patterns (Plotly chart for cumulative P&L, table for live positions, breaker status cards, kill switch button).

**Step 4: Run tests to verify they pass**

Run: `cd /Users/russellrudd/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_dashboard_trading.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add ui/web_dashboard.py tests/test_dashboard_trading.py
git commit -m "feat: add Trading tab to dashboard with positions, P&L, and circuit breakers"
```

---

## Task 8: Wire Into Main Orchestrator

Add StrategyEngine and SettlementService to the async service loop in `main.py`.

**Files:**
- Modify: `main.py`
- Create: `tests/test_integration.py`

**Step 1: Write the failing test**

```python
# tests/test_integration.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

def test_main_includes_strategy_engine():
    """main.py should import and start StrategyEngine."""
    import main
    assert hasattr(main, 'StrategyEngine') or 'strategy_engine' in dir(main)

def test_main_includes_settlement():
    """main.py should import and start SettlementService."""
    import main
    assert hasattr(main, 'SettlementService') or 'settlement' in dir(main)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/russellrudd/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_integration.py -v`
Expected: FAIL — imports not found

**Step 3: Modify `main.py`**

Add to imports:
```python
from services.strategy_engine import StrategyEngine
from services.settlement import SettlementService
```

Add to `main()` task list:
```python
strategy = StrategyEngine(db_path=DB_PATH, paper_trader=paper_trader)
settlement = SettlementService(db_path=DB_PATH, paper_trader=paper_trader)

tasks = [
    # ... existing services ...
    strategy.run(),
    settlement.run(),
]
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/russellrudd/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_integration.py -v`
Expected: All 2 tests PASS

**Step 5: Smoke test the full system**

Run: `cd /Users/russellrudd/Projects/alphatemp/alphatemp && PYTHONPATH=. timeout 30 venv/bin/python main.py --dashboard 2>&1 | head -50`
Expected: All services start without import errors. Strategy engine logs "started." Dashboard accessible at localhost:8050.

**Step 6: Commit**

```bash
git add main.py tests/test_integration.py
git commit -m "feat: wire strategy engine and settlement into main orchestrator"
```

---

## Task 9: End-to-End Verification

Run the full test suite and verify the system works with real data.

**Step 1: Run all tests**

Run: `cd /Users/russellrudd/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Verify model produces reasonable output with real DB**

```bash
cd /Users/russellrudd/Projects/alphatemp/alphatemp
PYTHONPATH=. venv/bin/python -c "
from services.feature_builder import FeatureBuilder
from services.model import QRModel
from datetime import date

fb = FeatureBuilder('data/alphatemp.duckdb')
model = QRModel()

# Get training data for tomorrow
training = fb.get_training_data(date(2026, 3, 12), update_hour=0, window_days=180)
if training:
    X, y, dates = training
    print(f'Training samples: {len(X)}')
    coeffs = model.fit(X, y, 0, '2026-03-12')
    print(f'Model fitted: {coeffs is not None}')

    features_result = fb.build_features(date(2026, 3, 12), update_hour=0)
    if features_result:
        features, fcst_high = features_result
        probs = model.predict_bracket_probs(features, fcst_high, 0, '2026-03-12')
        if probs:
            print(f'Brackets: {len(probs)}')
            for k in sorted(probs.keys()):
                if probs[k] > 0.01:
                    print(f'  {k}-{k+2}: {probs[k]*100:.1f}%')
        else:
            print('No bracket probs')
    else:
        print('No live features')
else:
    print('No training data')
"
```

**Step 3: Verify dashboard Trading tab loads**

Run: `cd /Users/russellrudd/Projects/alphatemp/alphatemp && PYTHONPATH=. timeout 10 venv/bin/python -c "
from fastapi.testclient import TestClient
from ui.web_dashboard import app
client = TestClient(app)
for endpoint in ['/api/trading/positions?city=nyc', '/api/trading/pnl?city=nyc', '/api/trading/breakers']:
    r = client.get(endpoint)
    print(f'{endpoint}: {r.status_code} - {list(r.json().keys())}')
"`

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: paper trading system complete — model, strategy engine, settlement, dashboard"
```
