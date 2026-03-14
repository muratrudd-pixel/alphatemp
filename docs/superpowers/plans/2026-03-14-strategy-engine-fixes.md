# Strategy Engine Fixes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable tail bracket trading, prevent duplicate positions, and add fee-adjusted EV filter to the strategy engine.

**Architecture:** Three targeted changes to the trading pipeline (strategy_engine → circuit_breakers → paper_trader). Market prices switch from int-keyed dicts to tuple-keyed `(Optional[int], Optional[int])` dicts. No DB schema changes.

**Tech Stack:** Python 3.9, DuckDB, pytest, pytest-asyncio, loguru

**Spec:** `docs/superpowers/specs/2026-03-14-strategy-engine-fixes-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `services/strategy_engine.py` | Modify | Tuple keys, tail aggregation, dedup guard, EV filter |
| `services/circuit_breakers.py` | Modify | Optional[int] params, NULL-safe SQL |
| `services/paper_trader.py` | Modify | Optional[int] params, tail settlement, tail unrealized |
| `tests/test_strategy_engine.py` | Modify | Update existing tests for tuple keys + add new tests |
| `tests/test_circuit_breakers.py` | Modify | Append tail bracket tests |
| `tests/test_paper_trader.py` | Modify | Append tail settlement tests |

### Existing Test Patterns

All three test files already exist. Follow these conventions:
- **circuit_breakers**: Uses `_create_test_db(tmp_path)` helper that creates `paper_config` + `paper_positions` tables. Fixture `test_db(tmp_path)` returns db path. Fixture `breakers(test_db)` returns `CircuitBreakers` instance.
- **paper_trader**: Uses `test_db(tmp_path)` fixture creating `paper_positions`, `nws_daily`, `market_ticks` tables. Fixture `trader(test_db)` returns `PaperTrader` instance. `nws_daily.ingested_at` is nullable in test schema.
- **strategy_engine**: Uses `engine()` fixture creating `StrategyEngine` via `__new__` with mocked deps. Sets `min_edge_pct = 5.0`. Does NOT test DB queries.

---

## Chunk 1: Circuit Breakers (Tail Bracket Support)

### Task 1: Update circuit breakers to handle Optional[int]

**Files:**
- Modify: `services/circuit_breakers.py`

- [ ] **Step 1: Add `_bracket_label` helper and update imports**

At the top of `services/circuit_breakers.py`, add `Optional` to imports and add the helper above the class:

```python
from typing import Dict, Optional, Tuple

# ...

def _bracket_label(floor, cap):
    # type: (Optional[int], Optional[int]) -> str
    """Human-readable bracket label."""
    if floor is None and cap is not None:
        return "<=%d" % cap
    if cap is None and floor is not None:
        return ">=%d" % floor
    if floor is not None and cap is not None:
        return "[%d, %d)" % (floor, cap)
    return "?"
```

- [ ] **Step 2: Update `check()` signature**

Change the type annotations for `bracket_floor` and `bracket_cap`:

```python
    def check(
        self,
        bracket_floor=None,   # type: Optional[int]
        bracket_cap=None,      # type: Optional[int]
        edge_pct=0.0,          # type: float
        market_date="",        # type: str
        now=None,              # type: Optional[datetime]
    ):
        # type: (...) -> Tuple[bool, str]
```

- [ ] **Step 3: Update max_per_bracket SQL (around line 76-80)**

Replace the existing SQL and format string:

```python
            # 5. Max per bracket — NULL-safe matching for tail brackets
            max_bracket = int(cfg.get("max_per_bracket", "2"))
            bracket_contracts = con.execute(
                "SELECT COALESCE(SUM(contracts), 0) FROM paper_positions "
                "WHERE status = 'open' "
                "AND (bracket_floor = ? OR (bracket_floor IS NULL AND ? IS NULL)) "
                "AND (bracket_cap = ? OR (bracket_cap IS NULL AND ? IS NULL))",
                [bracket_floor, bracket_floor, bracket_cap, bracket_cap],
            ).fetchone()[0]
            if bracket_contracts >= max_bracket:
                label = _bracket_label(bracket_floor, bracket_cap)
                return (
                    False,
                    "max_per_bracket: %d contracts on %s (>= %d limit)"
                    % (bracket_contracts, label, max_bracket),
                )
```

- [ ] **Step 4: Update cooldown SQL (around line 90-96)**

Replace the existing SQL and format string:

```python
            cooldown_min = int(cfg.get("cooldown_minutes", "30"))
            last_exit = con.execute(
                "SELECT MAX(exit_time) FROM paper_positions "
                "WHERE status = 'closed' "
                "AND exit_reason != 'settlement' "
                "AND (bracket_floor = ? OR (bracket_floor IS NULL AND ? IS NULL)) "
                "AND (bracket_cap = ? OR (bracket_cap IS NULL AND ? IS NULL))",
                [bracket_floor, bracket_floor, bracket_cap, bracket_cap],
            ).fetchone()[0]
            if last_exit is not None:
                if isinstance(last_exit, str):
                    last_exit = datetime.fromisoformat(last_exit)
                elapsed = now - last_exit
                if elapsed < timedelta(minutes=cooldown_min):
                    remaining = cooldown_min - int(elapsed.total_seconds() / 60)
                    label = _bracket_label(bracket_floor, bracket_cap)
                    return (
                        False,
                        "cooldown: %d min remaining on %s" % (remaining, label),
                    )
```

- [ ] **Step 5: Verify existing circuit breaker tests still pass**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_circuit_breakers.py -v`
Expected: All 8 existing tests PASS (they use int params, which still work).

### Task 2: Add tail bracket circuit breaker tests

**Files:**
- Modify: `tests/test_circuit_breakers.py`

- [ ] **Step 6: Append tail bracket test class**

Append to the end of `tests/test_circuit_breakers.py`:

```python
class TestTailBrackets:
    """Tail brackets (None floor or cap) pass through circuit breakers."""

    def test_lower_tail_passes(self, breakers):
        allowed, reason = breakers.check(
            bracket_floor=None, bracket_cap=49, edge_pct=10.0,
            market_date="2026-03-14",
        )
        assert allowed is True
        assert reason == "all_clear"

    def test_upper_tail_passes(self, breakers):
        allowed, reason = breakers.check(
            bracket_floor=56, bracket_cap=None, edge_pct=10.0,
            market_date="2026-03-14",
        )
        assert allowed is True
        assert reason == "all_clear"

    def test_lower_tail_max_per_bracket(self, test_db):
        """2 open positions on lower tail -> blocked by max_per_bracket."""
        con = duckdb.connect(test_db)
        for i in range(2):
            con.execute("""
                INSERT INTO paper_positions
                (id, city, event_date, bracket_floor, bracket_cap, direction,
                 entry_price, entry_time, status, contracts)
                VALUES (?, 'NYC', '2026-03-14', NULL, 49, 'YES',
                        20, '2026-03-14 10:00:00', 'open', 1)
            """, [i + 1])
        con.close()

        cb = CircuitBreakers(db_path=test_db)
        allowed, reason = cb.check(
            bracket_floor=None, bracket_cap=49, edge_pct=10.0,
            market_date="2026-03-14",
        )
        assert allowed is False
        assert "max_per_bracket" in reason

    def test_lower_tail_cooldown(self, test_db):
        """Recently exited lower tail -> blocked by cooldown."""
        now = datetime(2026, 3, 14, 14, 0, 0)
        exit_time = now - timedelta(minutes=10)
        con = duckdb.connect(test_db)
        con.execute("""
            INSERT INTO paper_positions
            (id, city, event_date, bracket_floor, bracket_cap, direction,
             entry_price, entry_time, exit_price, exit_time,
             net_pnl, status, exit_reason, contracts)
            VALUES
            (1, 'NYC', '2026-03-14', NULL, 49, 'YES',
             20, '2026-03-14 10:00:00', 25, ?, 0.05, 'closed', 'edge_reversal', 1)
        """, [exit_time])
        con.close()

        cb = CircuitBreakers(db_path=test_db)
        allowed, reason = cb.check(
            bracket_floor=None, bracket_cap=49, edge_pct=10.0,
            market_date="2026-03-14", now=now,
        )
        assert allowed is False
        assert "cooldown" in reason

    def test_tail_does_not_match_interior(self, test_db):
        """Open position on (None, 49) should NOT block (49, 51)."""
        con = duckdb.connect(test_db)
        con.execute("""
            INSERT INTO paper_positions
            (id, city, event_date, bracket_floor, bracket_cap, direction,
             entry_price, entry_time, status, contracts)
            VALUES (1, 'NYC', '2026-03-14', NULL, 49, 'YES',
                    20, '2026-03-14 10:00:00', 'open', 2)
        """)
        con.close()

        cb = CircuitBreakers(db_path=test_db)
        allowed, reason = cb.check(
            bracket_floor=49, bracket_cap=51, edge_pct=10.0,
            market_date="2026-03-14",
        )
        assert allowed is True
```

- [ ] **Step 7: Run all circuit breaker tests**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_circuit_breakers.py -v`
Expected: All 13 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add services/circuit_breakers.py tests/test_circuit_breakers.py
git commit -m "feat: circuit breakers accept Optional[int] for tail brackets"
```

---

## Chunk 2: Paper Trader (Tail Settlement + Unrealized)

### Task 3: Update paper trader for tail brackets

**Files:**
- Modify: `services/paper_trader.py`

- [ ] **Step 9: Add Optional import**

Add `Optional` to the typing import at top if not present:
```python
from typing import Optional
```

- [ ] **Step 10: Update type comments on `enter_position` and `_record_entry`**

In `_record_entry` (line 55-56), change:
```python
        bracket_floor,  # type: Optional[int]
        bracket_cap,    # type: Optional[int]
```

Same for `enter_position` (line 90-91).

- [ ] **Step 11: Fix `_settle_positions()` settlement logic**

Replace line 219 (`settled_yes = floor_val <= actual_high < cap_val`) with:
```python
                # Tail bracket settlement (matches KalshiBracket.contains)
                if floor_val is None:
                    settled_yes = actual_high < cap_val    # lower tail
                elif cap_val is None:
                    settled_yes = actual_high > floor_val  # upper tail
                else:
                    settled_yes = floor_val <= actual_high < cap_val  # interior
```

- [ ] **Step 12: Fix `_update_unrealized()` market tick lookup**

Replace the single query block (lines 267-274) with:
```python
                # Match market tick based on bracket type
                if floor_val is None:
                    # Lower tail
                    tick = con.execute("""
                        SELECT yes_bid, no_bid FROM market_ticks
                        WHERE city = ? AND floor_strike IS NULL
                          AND cap_strike = CAST(? AS DOUBLE)
                        ORDER BY captured_at DESC LIMIT 1
                    """, [city, cap_val]).fetchone()
                elif cap_val is None:
                    # Upper tail
                    tick = con.execute("""
                        SELECT yes_bid, no_bid FROM market_ticks
                        WHERE city = ? AND floor_strike = CAST(? AS DOUBLE)
                          AND cap_strike IS NULL
                        ORDER BY captured_at DESC LIMIT 1
                    """, [city, floor_val]).fetchone()
                else:
                    # Interior
                    tick = con.execute("""
                        SELECT yes_bid, no_bid FROM market_ticks
                        WHERE city = ? AND floor_strike = CAST(? AS DOUBLE)
                          AND cap_strike IS NOT NULL
                        ORDER BY captured_at DESC LIMIT 1
                    """, [city, floor_val]).fetchone()
```

- [ ] **Step 13: Verify existing paper trader tests still pass**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_paper_trader.py -v`
Expected: All 15 existing tests PASS.

### Task 4: Add tail bracket paper trader tests

**Files:**
- Modify: `tests/test_paper_trader.py`

- [ ] **Step 14: Append tail settlement and unrealized test classes**

Append to the end of `tests/test_paper_trader.py`:

```python
class TestSettlementTailBrackets:
    """Tail bracket settlement logic."""

    @pytest.mark.asyncio
    async def test_lower_tail_yes_wins(self, trader, test_db):
        """<=49 YES with actual high 42 -> YES wins (42 < 49)."""
        trader._record_entry(
            city="NYC", event_date="2026-03-14",
            bracket_floor=None, bracket_cap=49,
            direction="YES", model_prob=0.9,
            market_price=76, entry_price=76,
        )
        con = duckdb.connect(test_db)
        con.execute("""
            INSERT INTO nws_daily (station_id, obs_date, max_temp_f, source)
            VALUES ('KNYC', '2026-03-14', 42, 'NWS_CLI')
        """)
        con.close()

        await trader._settle_positions()

        con = duckdb.connect(test_db, read_only=True)
        row = con.execute(
            "SELECT status, settled_yes, gross_pnl FROM paper_positions"
        ).fetchone()
        con.close()
        assert row[0] == "closed"
        assert row[1] is True
        assert abs(row[2] - 0.24) < 0.001  # (100 - 76) / 100

    @pytest.mark.asyncio
    async def test_lower_tail_boundary_loses(self, trader, test_db):
        """<=49 YES with actual high exactly 49 -> loses (49 is NOT < 49)."""
        trader._record_entry(
            city="NYC", event_date="2026-03-14",
            bracket_floor=None, bracket_cap=49,
            direction="YES", model_prob=0.9,
            market_price=76, entry_price=76,
        )
        con = duckdb.connect(test_db)
        con.execute("""
            INSERT INTO nws_daily (station_id, obs_date, max_temp_f, source)
            VALUES ('KNYC', '2026-03-14', 49, 'NWS_CLI')
        """)
        con.close()

        await trader._settle_positions()

        con = duckdb.connect(test_db, read_only=True)
        row = con.execute("SELECT settled_yes FROM paper_positions").fetchone()
        con.close()
        assert row[0] is False

    @pytest.mark.asyncio
    async def test_upper_tail_yes_wins(self, trader, test_db):
        """>=56 YES with actual high 60 -> YES wins (60 > 56)."""
        trader._record_entry(
            city="NYC", event_date="2026-03-14",
            bracket_floor=56, bracket_cap=None,
            direction="YES", model_prob=0.9,
            market_price=10, entry_price=10,
        )
        con = duckdb.connect(test_db)
        con.execute("""
            INSERT INTO nws_daily (station_id, obs_date, max_temp_f, source)
            VALUES ('KNYC', '2026-03-14', 60, 'NWS_CLI')
        """)
        con.close()

        await trader._settle_positions()

        con = duckdb.connect(test_db, read_only=True)
        row = con.execute("SELECT status, settled_yes FROM paper_positions").fetchone()
        con.close()
        assert row[0] == "closed"
        assert row[1] is True

    @pytest.mark.asyncio
    async def test_upper_tail_boundary_loses(self, trader, test_db):
        """>=56 YES with actual high exactly 56 -> loses (56 is NOT > 56)."""
        trader._record_entry(
            city="NYC", event_date="2026-03-14",
            bracket_floor=56, bracket_cap=None,
            direction="YES", model_prob=0.9,
            market_price=10, entry_price=10,
        )
        con = duckdb.connect(test_db)
        con.execute("""
            INSERT INTO nws_daily (station_id, obs_date, max_temp_f, source)
            VALUES ('KNYC', '2026-03-14', 56, 'NWS_CLI')
        """)
        con.close()

        await trader._settle_positions()

        con = duckdb.connect(test_db, read_only=True)
        row = con.execute("SELECT settled_yes FROM paper_positions").fetchone()
        con.close()
        assert row[0] is False

    @pytest.mark.asyncio
    async def test_lower_tail_no_wins(self, trader, test_db):
        """<=49 NO with actual high 52 -> NO wins."""
        trader._record_entry(
            city="NYC", event_date="2026-03-14",
            bracket_floor=None, bracket_cap=49,
            direction="NO", model_prob=0.1,
            market_price=30, entry_price=30,
        )
        con = duckdb.connect(test_db)
        con.execute("""
            INSERT INTO nws_daily (station_id, obs_date, max_temp_f, source)
            VALUES ('KNYC', '2026-03-14', 52, 'NWS_CLI')
        """)
        con.close()

        await trader._settle_positions()

        con = duckdb.connect(test_db, read_only=True)
        row = con.execute(
            "SELECT status, settled_yes, gross_pnl FROM paper_positions"
        ).fetchone()
        con.close()
        assert row[0] == "closed"
        assert row[1] is False  # bracket settled NO
        assert row[2] > 0       # NO side wins


class TestUnrealizedTailBrackets:
    @pytest.mark.asyncio
    async def test_lower_tail_unrealized(self, trader, test_db):
        """Unrealized P&L for lower tail position using tail market tick."""
        trader._record_entry(
            city="NYC", event_date="2026-03-14",
            bracket_floor=None, bracket_cap=49,
            direction="YES", model_prob=0.9,
            market_price=76, entry_price=76,
        )
        # Market tick: floor_strike IS NULL, cap_strike = 49
        con = duckdb.connect(test_db)
        con.execute("""
            INSERT INTO market_ticks
            (market_id, city, captured_at, yes_bid, yes_ask, no_bid, no_ask,
             floor_strike, cap_strike)
            VALUES ('TAIL1', 'NYC', '2026-03-14 12:00:00',
                    0.80, 0.84, 0.16, 0.20, NULL, 49)
        """)
        con.close()

        await trader._update_unrealized()

        con = duckdb.connect(test_db, read_only=True)
        row = con.execute("SELECT unrealized_pnl FROM paper_positions").fetchone()
        con.close()
        # yes_bid = 0.80 -> 80c, entry = 76c, unrealized = (80-76)/100 = $0.04
        assert abs(row[0] - 0.04) < 0.001
```

- [ ] **Step 15: Run all paper trader tests**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_paper_trader.py -v`
Expected: All 22 tests PASS.

- [ ] **Step 16: Commit**

```bash
git add services/paper_trader.py tests/test_paper_trader.py
git commit -m "feat: paper trader handles tail bracket settlement and unrealized P&L"
```

---

## Chunk 3: Strategy Engine (Tuple Keys + Dedup + EV Filter)

### Task 5: Update strategy engine pure logic methods

**Files:**
- Modify: `services/strategy_engine.py`

- [ ] **Step 17: Add imports and `_bracket_label` helper**

Add `math` and `Tuple` to imports:
```python
import math
# ...
from typing import Any, Dict, List, Optional, Tuple
```

Add `_bracket_label` helper at module level (above the class, after constants):
```python
def _bracket_label(floor, cap):
    # type: (Optional[int], Optional[int]) -> str
    """Human-readable bracket label."""
    if floor is None and cap is not None:
        return "<=%d" % cap
    if cap is None and floor is not None:
        return ">=%d" % floor
    if floor is not None and cap is not None:
        return "[%d, %d)" % (floor, cap)
    return "?"
```

- [ ] **Step 18: Add `min_ev_cents` to `__init__` and `_load_config`**

In `__init__`, after `self.min_edge_pct = 5.0`:
```python
        self.min_ev_cents = 2.0  # will be read from paper_config
```

In `_load_config`, after the `min_edge_pct` read block, add:
```python
                row = con.execute(
                    "SELECT value FROM paper_config WHERE key = 'min_ev_cents'"
                ).fetchone()
                if row:
                    self.min_ev_cents = float(row[0])
```

- [ ] **Step 19: Add `_compute_ev` static method**

Add after `_compute_edge`:
```python
    @staticmethod
    def _compute_ev(edge_pct, market_ask_cents):
        # type: (float, int) -> float
        """Fee-adjusted expected value in cents.

        For a $1 contract: EV = edge_pct - fee_cents.
        Works for both YES and NO because edge_pct already encodes
        the probability-weighted advantage for the chosen side.
        """
        price_decimal = market_ask_cents / 100.0
        fee_cents = max(
            math.ceil(round(0.07 * price_decimal * (1 - price_decimal) * 100, 10)),
            1,
        )
        return edge_pct - fee_cents
```

- [ ] **Step 20: Rewrite `_get_market_prices` for tuple keys**

Replace the `prices` dict building logic (lines 439-451):
```python
        prices = {}  # type: Dict[Tuple[Optional[int], Optional[int]], Dict[str, int]]
        for row in rows:
            floor_strike, cap_strike, yes_bid, yes_ask, no_bid, no_ask = row
            # Skip degenerate (both None) or missing prices
            if (floor_strike is None and cap_strike is None) or any(
                v is None for v in [yes_bid, yes_ask, no_bid, no_ask]
            ):
                continue
            floor_key = int(floor_strike) if floor_strike is not None else None
            cap_key = int(cap_strike) if cap_strike is not None else None
            prices[(floor_key, cap_key)] = {
                "yes_bid": int(round(yes_bid * 100)),
                "yes_ask": int(round(yes_ask * 100)),
                "no_bid": int(round(no_bid * 100)),
                "no_ask": int(round(no_ask * 100)),
            }
        return prices
```

- [ ] **Step 21: Rewrite `_aggregate_to_kalshi_brackets`**

Replace the entire method:
```python
    @staticmethod
    def _aggregate_to_kalshi_brackets(bracket_probs, market_prices):
        # type: (Dict[int, float], Dict[Tuple[Optional[int], Optional[int]], Dict[str, int]]) -> Dict[Tuple[Optional[int], Optional[int]], float]
        """Aggregate 1-degree model probs into Kalshi brackets (including tails).

        Bracket types:
        - Interior (floor, cap): sum probs for floor <= k <= cap
        - Lower tail (None, cap): sum probs for k < cap
        - Upper tail (floor, None): sum probs for k > floor
        """
        aggregated = {}  # type: Dict[Tuple[Optional[int], Optional[int]], float]
        for key in market_prices:
            floor, cap = key
            if floor is None and cap is not None:
                total = sum(p for k, p in bracket_probs.items() if k < cap)
            elif cap is None and floor is not None:
                total = sum(p for k, p in bracket_probs.items() if k > floor)
            elif floor is not None and cap is not None:
                total = sum(
                    p for k, p in bracket_probs.items() if floor <= k <= cap
                )
            else:
                continue
            aggregated[key] = total
        return aggregated
```

- [ ] **Step 22: Rewrite `_generate_signals` with EV filter**

Replace the entire method:
```python
    def _generate_signals(self, bracket_probs, market_prices):
        # type: (Dict[int, float], Dict[Tuple[Optional[int], Optional[int]], Dict[str, int]]) -> List[Dict[str, Any]]
        """Generate trade signals comparing model probs to market prices.

        Includes EV filter: signals must have fee-adjusted EV >= min_ev_cents.
        """
        kalshi_probs = self._aggregate_to_kalshi_brackets(bracket_probs, market_prices)

        signals = []  # type: List[Dict[str, Any]]

        for bracket_key, model_prob in kalshi_probs.items():
            floor, cap = bracket_key
            prices = market_prices[bracket_key]

            yes_ask = prices["yes_ask"]
            no_ask = prices["no_ask"]

            # YES edge
            yes_edge = self._compute_edge(model_prob, yes_ask)
            # NO edge
            no_prob = 1.0 - model_prob
            no_edge = self._compute_edge(no_prob, no_ask)

            if yes_edge >= self.min_edge_pct:
                ev = self._compute_ev(yes_edge, yes_ask)
                if ev >= self.min_ev_cents:
                    signals.append({
                        "bracket_floor": floor,
                        "bracket_cap": cap,
                        "direction": "YES",
                        "model_prob": model_prob,
                        "market_price": yes_ask,
                        "edge_pct": yes_edge,
                    })
            elif no_edge >= self.min_edge_pct:
                ev = self._compute_ev(no_edge, no_ask)
                if ev >= self.min_ev_cents:
                    signals.append({
                        "bracket_floor": floor,
                        "bracket_cap": cap,
                        "direction": "NO",
                        "model_prob": model_prob,
                        "market_price": no_ask,
                        "edge_pct": no_edge,
                    })

        return signals
```

- [ ] **Step 23: Update `_check_edge_reversals` for tuple keys**

In `_check_edge_reversals`, replace the bracket lookup (around line 326-328):
```python
            bracket_key = (pos["bracket_floor"], pos["bracket_cap"])
            model_prob = kalshi_probs.get(bracket_key)
            prices = market_prices.get(bracket_key)
```

Update the log message (around line 347):
```python
                label = _bracket_label(pos["bracket_floor"], pos["bracket_cap"])
                logger.info(
                    "Edge reversal: pos {} {} {} edge={:.1f}%",
                    pos["id"], label, pos["direction"], edge,
                )
```

### Task 6: Update existing strategy engine tests for tuple keys

**Files:**
- Modify: `tests/test_strategy_engine.py`

- [ ] **Step 24: Update `engine()` fixture**

Add `min_ev_cents` to the fixture (after `eng.min_edge_pct = 5.0`):
```python
    eng.min_ev_cents = 2.0
```

- [ ] **Step 25: Update `TestAggregateToKalshiBrackets` to use tuple keys**

Replace the two existing tests:
```python
class TestAggregateToKalshiBrackets:
    def test_sums_adjacent_1f_probs(self, engine):
        """Two 1°F probs should sum to one 2°F Kalshi bracket prob."""
        bracket_probs = {72: 0.08, 73: 0.07, 74: 0.06, 75: 0.05}
        market_prices = {
            (72, 73): {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
            (74, 75): {"yes_bid": 8, "yes_ask": 10, "no_bid": 88, "no_ask": 90},
        }
        result = engine._aggregate_to_kalshi_brackets(bracket_probs, market_prices)
        # (72,73) = probs[72] + probs[73] = 0.08 + 0.07 = 0.15
        assert abs(result[(72, 73)] - 0.15) < 0.001
        # (74,75) = probs[74] + probs[75] = 0.06 + 0.05 = 0.11
        assert abs(result[(74, 75)] - 0.11) < 0.001

    def test_missing_adjacent_uses_zero(self, engine):
        """If model only has one of the two 1°F components, other defaults to 0."""
        bracket_probs = {72: 0.10}  # no key 73
        market_prices = {
            (72, 73): {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        result = engine._aggregate_to_kalshi_brackets(bracket_probs, market_prices)
        assert abs(result[(72, 73)] - 0.10) < 0.001
```

- [ ] **Step 26: Update `TestGenerateSignals` to use tuple keys**

Update all market_prices dicts and signal assertions. Replace the class:
```python
class TestGenerateSignals:
    def test_no_edge_no_trade(self, engine):
        """Below threshold -> no signal."""
        bracket_probs = {72: 0.15, 74: 0.10}
        market_prices = {
            (72, 73): {"yes_bid": 11, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
            (74, 75): {"yes_bid": 8, "yes_ask": 9, "no_bid": 89, "no_ask": 91},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert len(signals) == 0

    def test_positive_edge_generates_yes_signal(self, engine):
        """Model > market ask -> YES signal when edge >= min_edge_pct."""
        bracket_probs = {72: 0.25}
        market_prices = {
            (72, 73): {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert len(signals) == 1
        sig = signals[0]
        assert sig["bracket_floor"] == 72
        assert sig["bracket_cap"] == 73
        assert sig["direction"] == "YES"
        assert abs(sig["model_prob"] - 0.25) < 0.001
        assert sig["market_price"] == 12
        assert abs(sig["edge_pct"] - 13.0) < 0.01

    def test_no_signal_when_model_says_unlikely(self, engine):
        """Model says bracket unlikely -> NO signal."""
        bracket_probs = {72: 0.02}
        market_prices = {
            (72, 73): {"yes_bid": 1, "yes_ask": 3, "no_bid": 80, "no_ask": 85},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert len(signals) == 1
        sig = signals[0]
        assert sig["direction"] == "NO"
        assert sig["bracket_floor"] == 72
        assert sig["bracket_cap"] == 73

    def test_yes_preferred_over_no_on_same_bracket(self, engine):
        """If both YES and NO have edge, prefer YES."""
        bracket_probs = {72: 0.55}
        market_prices = {
            (72, 73): {"yes_bid": 40, "yes_ask": 45, "no_bid": 35, "no_ask": 40},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert len(signals) == 1
        assert signals[0]["direction"] == "YES"

    def test_multiple_brackets_multiple_signals(self, engine):
        """Multiple brackets can each generate signals."""
        bracket_probs = {72: 0.30, 74: 0.25}
        market_prices = {
            (72, 73): {"yes_bid": 15, "yes_ask": 18, "no_bid": 78, "no_ask": 82},
            (74, 75): {"yes_bid": 10, "yes_ask": 12, "no_bid": 85, "no_ask": 88},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert len(signals) == 2

    def test_bracket_not_in_market_skipped(self, engine):
        """Brackets with no market data are silently skipped."""
        bracket_probs = {72: 0.30, 99: 0.05}
        market_prices = {
            (72, 73): {"yes_bid": 15, "yes_ask": 18, "no_bid": 78, "no_ask": 82},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert all(s["bracket_floor"] == 72 for s in signals)
```

- [ ] **Step 27: Update `TestCheckEdgeReversals` to use tuple keys**

Update all market_prices and bracket_probs dicts. The bracket_probs dict stays as int keys (model outputs 1°F probs). The market_prices dict becomes tuple-keyed. Positions now need both floor and cap for the tuple lookup.

Replace the class:
```python
class TestCheckEdgeReversals:
    def test_edge_reversal_exits_position(self, engine):
        """When edge flips negative, call paper_trader.exit_position."""
        engine.paper_trader.exit_position = AsyncMock()
        open_positions = [
            {"id": 1, "bracket_floor": 72, "bracket_cap": 73,
             "direction": "YES", "entry_price": 12},
        ]
        bracket_probs = {72: 0.08}
        market_prices = {
            (72, 73): {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        asyncio.get_event_loop().run_until_complete(
            engine._check_edge_reversals(bracket_probs, market_prices, open_positions)
        )
        engine.paper_trader.exit_position.assert_called_once_with(1, 10, "edge_reversal")

    def test_no_exit_when_edge_still_positive(self, engine):
        """When edge is still positive, don't exit."""
        engine.paper_trader.exit_position = AsyncMock()
        open_positions = [
            {"id": 1, "bracket_floor": 72, "bracket_cap": 73,
             "direction": "YES", "entry_price": 12},
        ]
        bracket_probs = {72: 0.25}
        market_prices = {
            (72, 73): {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        asyncio.get_event_loop().run_until_complete(
            engine._check_edge_reversals(bracket_probs, market_prices, open_positions)
        )
        engine.paper_trader.exit_position.assert_not_called()

    def test_no_side_edge_reversal(self, engine):
        """NO position: edge reversal when (1 - model_prob) < no_ask/100."""
        engine.paper_trader.exit_position = AsyncMock()
        open_positions = [
            {"id": 2, "bracket_floor": 72, "bracket_cap": 73,
             "direction": "NO", "entry_price": 85},
        ]
        bracket_probs = {72: 0.25}
        market_prices = {
            (72, 73): {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        asyncio.get_event_loop().run_until_complete(
            engine._check_edge_reversals(bracket_probs, market_prices, open_positions)
        )
        engine.paper_trader.exit_position.assert_called_once_with(2, 86, "edge_reversal")

    def test_zero_edge_exits_position(self, engine):
        """When edge is exactly zero, should exit."""
        engine.paper_trader.exit_position = AsyncMock()
        open_positions = [
            {"id": 1, "bracket_floor": 72, "bracket_cap": 73,
             "direction": "YES", "entry_price": 12},
        ]
        bracket_probs = {72: 0.12}
        market_prices = {
            (72, 73): {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        asyncio.get_event_loop().run_until_complete(
            engine._check_edge_reversals(bracket_probs, market_prices, open_positions)
        )
        engine.paper_trader.exit_position.assert_called_once_with(1, 10, "edge_reversal")

    def test_bracket_missing_from_model_skips(self, engine):
        """Position on a bracket the model didn't predict -> skip."""
        engine.paper_trader.exit_position = AsyncMock()
        open_positions = [
            {"id": 1, "bracket_floor": 99, "bracket_cap": 100,
             "direction": "YES", "entry_price": 12},
        ]
        bracket_probs = {72: 0.20}
        market_prices = {
            (72, 73): {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        asyncio.get_event_loop().run_until_complete(
            engine._check_edge_reversals(bracket_probs, market_prices, open_positions)
        )
        engine.paper_trader.exit_position.assert_not_called()
```

### Task 7: Add new strategy engine tests (tail, dedup, EV)

**Files:**
- Modify: `tests/test_strategy_engine.py`

- [ ] **Step 28: Append new test classes for tail brackets, dedup, and EV**

Append to end of `tests/test_strategy_engine.py`:

```python
# ---------------------------------------------------------------------------
# Tail bracket aggregation
# ---------------------------------------------------------------------------

class TestAggregateTailBrackets:
    def test_lower_tail(self, engine):
        """Lower tail (None, 49): sums all probs for k < 49."""
        bracket_probs = {46: 0.05, 47: 0.10, 48: 0.15, 49: 0.20, 50: 0.25}
        market_prices = {(None, 49): {"yes_ask": 20}}
        result = engine._aggregate_to_kalshi_brackets(bracket_probs, market_prices)
        assert abs(result[(None, 49)] - 0.30) < 1e-9  # 0.05 + 0.10 + 0.15

    def test_upper_tail(self, engine):
        """Upper tail (56, None): sums all probs for k > 56."""
        bracket_probs = {55: 0.10, 56: 0.05, 57: 0.03, 58: 0.02}
        market_prices = {(56, None): {"yes_ask": 5}}
        result = engine._aggregate_to_kalshi_brackets(bracket_probs, market_prices)
        assert abs(result[(56, None)] - 0.05) < 1e-9  # 0.03 + 0.02

    def test_mixed_brackets(self, engine):
        """Full bracket set: lower tail + interior + upper tail."""
        bracket_probs = {
            44: 0.05, 45: 0.10, 46: 0.15, 47: 0.20,
            48: 0.20, 49: 0.15, 50: 0.10, 51: 0.05,
        }
        market_prices = {
            (None, 45): {"yes_ask": 5},
            (45, 46): {"yes_ask": 20},
            (47, 48): {"yes_ask": 30},
            (49, 50): {"yes_ask": 20},
            (51, None): {"yes_ask": 5},
        }
        result = engine._aggregate_to_kalshi_brackets(bracket_probs, market_prices)
        assert abs(result[(None, 45)] - 0.05) < 1e-9   # k < 45: only k=44
        assert abs(result[(45, 46)] - 0.25) < 1e-9     # k=45 + k=46
        assert abs(result[(47, 48)] - 0.40) < 1e-9     # k=47 + k=48
        assert abs(result[(49, 50)] - 0.25) < 1e-9     # k=49 + k=50
        assert abs(result[(51, None)] - 0.0) < 1e-9    # k > 51: none


# ---------------------------------------------------------------------------
# Tail bracket signal generation
# ---------------------------------------------------------------------------

class TestTailBracketSignals:
    def test_lower_tail_yes_signal(self, engine):
        """Lower tail with high model prob should generate YES signal."""
        bracket_probs = {k: 0.01 for k in range(40, 60)}
        bracket_probs[42] = 0.30
        bracket_probs[43] = 0.30
        market_prices = {
            (None, 45): {"yes_ask": 20, "yes_bid": 18, "no_ask": 82, "no_bid": 80},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert len(signals) == 1
        assert signals[0]["bracket_floor"] is None
        assert signals[0]["bracket_cap"] == 45
        assert signals[0]["direction"] == "YES"

    def test_upper_tail_no_signal(self, engine):
        """Upper tail: model says unlikely -> NO signal."""
        bracket_probs = {k: 0.01 for k in range(40, 60)}
        market_prices = {
            (56, None): {"yes_ask": 30, "yes_bid": 28, "no_ask": 72, "no_bid": 70},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert len(signals) == 1
        assert signals[0]["bracket_floor"] == 56
        assert signals[0]["bracket_cap"] is None
        assert signals[0]["direction"] == "NO"


# ---------------------------------------------------------------------------
# EV filter
# ---------------------------------------------------------------------------

class TestEVFilter:
    def test_compute_ev_high_edge(self, engine):
        """High edge, moderate price -> positive EV."""
        ev = engine._compute_ev(20.0, 50)
        # fee = max(ceil(0.07 * 0.5 * 0.5 * 100), 1) = max(2, 1) = 2
        assert abs(ev - 18.0) < 0.01

    def test_compute_ev_penny_bet(self, engine):
        """Low price penny bet -> EV barely positive."""
        ev = engine._compute_ev(3.0, 1)
        # fee = max(ceil(0.07 * 0.01 * 0.99 * 100), 1) = max(1, 1) = 1
        assert abs(ev - 2.0) < 0.01

    def test_ev_filter_blocks_signal(self, engine):
        """Signal with edge >= min but EV < min_ev should be filtered."""
        engine.min_ev_cents = 5.0
        engine.min_edge_pct = 1.0
        bracket_probs = {72: 0.04}  # model says 4%
        market_prices = {
            (72, 73): {"yes_ask": 1, "yes_bid": 0, "no_ask": 99, "no_bid": 98},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        # YES edge = (4 - 1) * 100 = wait, edge_pct = (0.04 - 0.01) * 100 = 3.0
        # EV = 3.0 - 1 = 2.0, which is < 5.0. Should be blocked.
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# Dedup guard
# ---------------------------------------------------------------------------

class TestDedupGuard:
    def test_dedup_filters_held_bracket(self):
        """Signal on bracket already held should be filtered."""
        open_positions = [
            {"bracket_floor": None, "bracket_cap": 49, "direction": "YES"},
            {"bracket_floor": 49, "bracket_cap": 51, "direction": "NO"},
        ]
        signals = [
            {"bracket_floor": None, "bracket_cap": 49, "direction": "YES", "edge_pct": 10},
            {"bracket_floor": 53, "bracket_cap": 55, "direction": "NO", "edge_pct": 8},
        ]
        held = {
            (p["bracket_floor"], p["bracket_cap"], p["direction"])
            for p in open_positions
        }
        filtered = [
            s for s in signals
            if (s["bracket_floor"], s["bracket_cap"], s["direction"]) not in held
        ]
        assert len(filtered) == 1
        assert filtered[0]["bracket_floor"] == 53

    def test_dedup_allows_different_direction(self):
        """Same bracket, different direction should pass."""
        open_positions = [
            {"bracket_floor": 49, "bracket_cap": 51, "direction": "YES"},
        ]
        signals = [
            {"bracket_floor": 49, "bracket_cap": 51, "direction": "NO", "edge_pct": 10},
        ]
        held = {
            (p["bracket_floor"], p["bracket_cap"], p["direction"])
            for p in open_positions
        }
        filtered = [
            s for s in signals
            if (s["bracket_floor"], s["bracket_cap"], s["direction"]) not in held
        ]
        assert len(filtered) == 1

    def test_none_in_tuple_equality(self):
        """Python None in tuples supports correct equality for set membership."""
        held = {(None, 49, "YES")}
        assert (None, 49, "YES") in held
        assert (None, 49, "NO") not in held
        assert (49, None, "YES") not in held
```

- [ ] **Step 29: Run all strategy engine tests**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_strategy_engine.py -v`
Expected: All tests PASS (existing updated + new).

- [ ] **Step 30: Commit**

```bash
git add services/strategy_engine.py tests/test_strategy_engine.py
git commit -m "feat: strategy engine tuple keys, tail aggregation, EV filter"
```

### Task 8: Update strategy engine main cycle (dedup + tail logging)

**Files:**
- Modify: `services/strategy_engine.py`

- [ ] **Step 31: Add dedup guard to `_cycle`**

After the edge reversal check (after line ~163) and before the signal loop, add:
```python
        # 8b. Filter out signals for brackets already held
        if open_positions:
            held = {
                (p["bracket_floor"], p["bracket_cap"], p["direction"])
                for p in open_positions
            }
            before = len(signals)
            signals = [
                s for s in signals
                if (s["bracket_floor"], s["bracket_cap"], s["direction"]) not in held
            ]
            if before != len(signals):
                logger.info(
                    "Dedup: filtered {} signals (already held)", before - len(signals)
                )
```

- [ ] **Step 32: Update signal processing loop for tail brackets**

Replace the signal loop (lines 166-199):
```python
        market_date_str = target_date.isoformat()
        for sig in signals:
            bracket_floor = sig["bracket_floor"]
            bracket_cap = sig["bracket_cap"]

            # For paper_positions: interior brackets use legacy cap convention
            pp_cap = bracket_cap
            if bracket_floor is not None and bracket_cap is not None:
                pp_cap = bracket_floor + BRACKET_WIDTH

            allowed, reason = self.circuit_breakers.check(
                bracket_floor=bracket_floor,
                bracket_cap=pp_cap,
                edge_pct=sig["edge_pct"],
                market_date=market_date_str,
            )

            if allowed:
                label = _bracket_label(bracket_floor, bracket_cap)
                logger.info(
                    "TRADE: {} {} edge={:.1f}% model={:.1f}% market={}c",
                    sig["direction"], label,
                    sig["edge_pct"], sig["model_prob"] * 100,
                    sig["market_price"],
                )
                await self.paper_trader.enter_position(
                    city="NYC",
                    event_date=market_date_str,
                    bracket_floor=bracket_floor,
                    bracket_cap=pp_cap,
                    direction=sig["direction"],
                    model_prob=sig["model_prob"],
                    market_price=sig["market_price"],
                    edge=sig["edge_pct"],
                )
            else:
                label = _bracket_label(bracket_floor, bracket_cap)
                logger.debug(
                    "Blocked: {} {} — {}",
                    label, sig["direction"], reason,
                )
```

- [ ] **Step 33: Run full test suite**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/ -v --ignore=venv`
Expected: All tests PASS.

- [ ] **Step 34: Commit**

```bash
git add services/strategy_engine.py
git commit -m "feat: dedup guard and tail bracket logging in main cycle"
```

---

## Chunk 4: Smoke Test Against Live DB

### Task 9: Verify against live data

- [ ] **Step 35: Smoke test market prices with tail brackets**

```bash
cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -c "
from services.strategy_engine import StrategyEngine
from datetime import date

engine = StrategyEngine.__new__(StrategyEngine)
engine.db_path = 'data/alphatemp.duckdb'
prices = engine._get_market_prices(date(2026, 3, 14))
for key, p in sorted(prices.items(), key=lambda x: (x[0][0] if x[0][0] is not None else -999, x[0][1] if x[0][1] is not None else 999)):
    print(f'{key}: bid={p[\"yes_bid\"]}c ask={p[\"yes_ask\"]}c')
"
```
Expected: Should show tail brackets `(None, X)` and `(X, None)` alongside interior brackets.

- [ ] **Step 36: Smoke test signal generation**

```bash
cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -c "
from services.strategy_engine import StrategyEngine

engine = StrategyEngine.__new__(StrategyEngine)
engine.min_edge_pct = 5.0
engine.min_ev_cents = 2.0

# Simulate: model says 95% chance high is <=48
bracket_probs = {}
for k in range(35, 65):
    bracket_probs[k] = 0.001
bracket_probs[42] = 0.30
bracket_probs[43] = 0.25
bracket_probs[44] = 0.20
bracket_probs[45] = 0.10
bracket_probs[46] = 0.05

market_prices = {
    (None, 45): {'yes_ask': 20, 'yes_bid': 18, 'no_ask': 82, 'no_bid': 80},
    (45, 46): {'yes_ask': 15, 'yes_bid': 13, 'no_ask': 87, 'no_bid': 85},
    (47, 48): {'yes_ask': 25, 'yes_bid': 23, 'no_ask': 77, 'no_bid': 75},
    (49, 50): {'yes_ask': 20, 'yes_bid': 18, 'no_ask': 82, 'no_bid': 80},
    (51, None): {'yes_ask': 5, 'yes_bid': 3, 'no_ask': 97, 'no_bid': 95},
}

signals = engine._generate_signals(bracket_probs, market_prices)
for s in signals:
    print(f'{s[\"direction\"]:>3} ({s[\"bracket_floor\"]}, {s[\"bracket_cap\"]}) edge={s[\"edge_pct\"]:.1f}% market={s[\"market_price\"]}c')
"
```
Expected: Should generate a YES signal on the lower tail bracket `(None, 45)`.
