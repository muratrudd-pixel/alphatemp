# Strategy Engine Fixes — Tail Brackets, Dedup, EV Filter

**Date:** 2026-03-14
**Status:** Approved
**Context:** March 13 paper trading revealed three issues: tail brackets skipped entirely (biggest edge source), 149 redundant trades per day, and penny YES bets with negative fee-adjusted EV.

---

## Problem Statement

### 1. Tail Brackets Excluded
`strategy_engine._get_market_prices()` skips rows where `floor_strike IS NULL` (line 442). This drops both lower and upper tail brackets. On March 13, the lower tail (<=45°F) had 79% edge and was the single most logical bet — settlement was 42°F. Zero tail trades have ever been placed.

### 2. Excessive Trade Cadence
The strategy engine cycles every 5 minutes and opens a new position each cycle on every bracket with edge. On March 13 this produced 149 trades expressing the same thesis repeatedly. The data hash check (line 92) doesn't prevent this because market ticks update every 60s.

### 3. Penny YES Bets Lose Money After Fees
The engine bought 63 YES contracts on 45-47 and 47-49 at 1-5c each (8% edge). All settled at $0, netting -$2.64. The minimum Kalshi fee is $0.01/contract, which exceeds the expected value on sub-5c contracts. Edge % alone doesn't capture this.

---

## Design

### Fix 1: Tail Bracket Trading

**Key type change:** Market prices keyed by `(Optional[int], Optional[int])` tuples instead of `int`.
- Lower tail: `(None, 49)` — no floor, cap at 49
- Upper tail: `(56, None)` — floor at 56, no cap
- Interior: `(49, 50)` — both defined (Kalshi cap = floor + 1)

#### Bracket Convention Note

Two bracket_cap conventions exist in the codebase:
- **Kalshi/market_ticks:** cap = floor + 1 (e.g., floor=49, cap=50 for the "49-50" bracket)
- **paper_positions (legacy):** cap = floor + BRACKET_WIDTH (2) (e.g., floor=49, cap=51)

This fix does NOT resolve the convention mismatch for interior brackets. Interior brackets continue to use `bracket_cap = bracket_floor + BRACKET_WIDTH` when stored in paper_positions, preserving backward compatibility with existing positions and the `floor <= actual_high < cap` settlement logic. Tail brackets use None for the missing bound.

The convention mismatch is tracked as tech debt in the HANDOFF.md and can be resolved separately.

#### strategy_engine.py

**`_get_market_prices()`** — Change the guard on line 442 from:
```python
if floor_strike is None or any(v is None for v in [yes_bid, yes_ask, no_bid, no_ask]):
    continue
```
To:
```python
# Skip only if BOTH floor and cap are None (degenerate) or any price is None
if (floor_strike is None and cap_strike is None) or any(v is None for v in [yes_bid, yes_ask, no_bid, no_ask]):
    continue
```
Build keys as `(Optional[int], Optional[int])` tuples from `(floor_strike, cap_strike)`. Return type becomes `Dict[Tuple[Optional[int], Optional[int]], Dict[str, int]]`.

**`_aggregate_to_kalshi_brackets()`** — Replace the current `range(BRACKET_WIDTH)` loop with tuple-key-aware logic:

```python
@staticmethod
def _aggregate_to_kalshi_brackets(bracket_probs, market_prices):
    # type: (Dict[int, float], Dict[Tuple[Optional[int], Optional[int]], Dict[str, int]]) -> Dict[Tuple[Optional[int], Optional[int]], float]
    aggregated = {}
    for key in market_prices:
        floor, cap = key
        if floor is None and cap is not None:
            # Lower tail: sum all probs for k < cap
            total = sum(p for k, p in bracket_probs.items() if k < cap)
        elif cap is None and floor is not None:
            # Upper tail: sum all probs for k > floor
            total = sum(p for k, p in bracket_probs.items() if k > floor)
        elif floor is not None and cap is not None:
            # Interior: sum probs for floor <= k <= cap (both inclusive)
            total = sum(p for k, p in bracket_probs.items() if floor <= k <= cap)
        else:
            continue  # degenerate (None, None) — skip
        aggregated[key] = total
    return aggregated
```

Note: The model outputs probs for center +/- ~18°F. For tail brackets far from center, the sum may undercount (the model's CDF has nonzero mass beyond its output range). This is a pre-existing limitation of the integer-bracket aggregation approach and is acceptable — tails near center (the common case for tradeable edge) will aggregate correctly.

**`_generate_signals()`** — Signals carry both `bracket_floor: Optional[int]` and `bracket_cap: Optional[int]`. For interior brackets, continue to derive `bracket_cap = bracket_floor + BRACKET_WIDTH` for paper_positions compatibility. For tail brackets, use None.

**`_check_edge_reversals()`** — Update to use tuple keys. Currently does `kalshi_probs.get(bracket_floor)` which will break when keys become tuples. Change to `kalshi_probs.get((pos["bracket_floor"], pos["bracket_cap"]))`. Also update `market_prices.get(bracket_floor)` to use the same tuple key.

**Main cycle (lines 167-199)** — Pass floor/cap from signal dict to circuit breakers and paper trader. For interior brackets, paper_trader receives `bracket_cap = floor + BRACKET_WIDTH` (legacy convention). For tail brackets, receives None.

**Log messages (line 180)** — Format tail brackets as `[<=49)` for lower tail, `[56>=)` for upper tail instead of printing None.

#### circuit_breakers.py

**`check()`** — Change signature to `bracket_floor: Optional[int], bracket_cap: Optional[int]`.

SQL queries for per-bracket checks (max_per_bracket line 78, cooldown line 94) need NULL-safe matching:
```sql
WHERE (bracket_floor = ? OR (bracket_floor IS NULL AND ? IS NULL))
  AND (bracket_cap = ? OR (bracket_cap IS NULL AND ? IS NULL))
```
Pass each value twice (e.g., `[bracket_floor, bracket_floor, bracket_cap, bracket_cap]`).

Format strings for log messages (lines 84-85, 105): use helper to format bracket labels:
- Lower tail (floor=None): `"<=X°F"`
- Upper tail (cap=None): `">=X°F"`
- Interior: `"[X, Y)"`

#### paper_trader.py

**`enter_position()` / `_record_entry()`** — Accept `Optional[int]` for floor/cap. No other logic change — DuckDB INTEGER columns accept NULL by default.

**`_settle_positions()`** — Replace `floor_val <= actual_high < cap_val` (line 219) with:
```python
if floor_val is None:
    settled_yes = actual_high < cap_val    # lower tail: strictly less than
elif cap_val is None:
    settled_yes = actual_high > floor_val  # upper tail: strictly greater than
else:
    settled_yes = floor_val <= actual_high < cap_val  # interior (legacy cap convention)
```
The interior case preserves the existing `< cap_val` logic because paper_positions stores `cap = floor + BRACKET_WIDTH` (not Kalshi cap). With cap = floor + 2, `floor <= temp < floor + 2` correctly covers 2 integer temperatures.

The tail cases match `KalshiBracket.contains()` in `services/backtester.py:67-82` (verified against KXHIGHNY settlement data).

**`_update_unrealized()`** — Replace the single market_ticks query with branching:
```python
if floor_val is None:
    # Lower tail: match on cap only
    tick = con.execute("""
        SELECT yes_bid, no_bid FROM market_ticks
        WHERE city = ? AND floor_strike IS NULL AND cap_strike = CAST(? AS DOUBLE)
        ORDER BY captured_at DESC LIMIT 1
    """, [city, cap_val]).fetchone()
elif cap_val is None:
    # Upper tail: match on floor only
    tick = con.execute("""
        SELECT yes_bid, no_bid FROM market_ticks
        WHERE city = ? AND floor_strike = CAST(? AS DOUBLE) AND cap_strike IS NULL
        ORDER BY captured_at DESC LIMIT 1
    """, [city, floor_val]).fetchone()
else:
    # Interior: match on floor, require non-null cap
    tick = con.execute("""
        SELECT yes_bid, no_bid FROM market_ticks
        WHERE city = ? AND floor_strike = CAST(? AS DOUBLE) AND cap_strike IS NOT NULL
        ORDER BY captured_at DESC LIMIT 1
    """, [city, floor_val]).fetchone()
```

#### DB schema

No change. `paper_positions.bracket_floor` and `bracket_cap` are `INTEGER`, which accepts NULL in DuckDB. No migration needed.

---

### Fix 2: Duplicate Position Guard

**Location:** `strategy_engine._cycle()`, between signal generation (step 7) and circuit breaker checks (step 9).

**Logic:** After fetching `open_positions` (already done on line 159), build a set of `(bracket_floor, bracket_cap, direction)` from open positions. Filter signals to exclude any matching an existing open position.

```python
held = {(p["bracket_floor"], p["bracket_cap"], p["direction"]) for p in open_positions}
signals = [s for s in signals if (s["bracket_floor"], s["bracket_cap"], s["direction"]) not in held]
```

**None handling:** DuckDB returns Python `None` for SQL NULL columns. Python tuples containing `None` are valid set members and support equality comparison: `(None, 49, "YES") == (None, 49, "YES")` is True. The dedup check works correctly for tail brackets.

This is a signal-level filter, not a circuit breaker — it prevents redundant entries before they reach risk controls.

---

### Fix 3: Minimum EV Filter

**Location:** `strategy_engine._generate_signals()`, after computing edge.

**Config:** `min_ev_cents` loaded in `_load_config()` from `paper_config` table. Default to 2 if key doesn't exist (no DB migration needed — just a Python fallback).

**EV derivation (for both YES and NO):**

For a YES signal with `edge_pct = (model_prob - ask/100) * 100`:
```
EV = model_prob * (100 - ask) - (1 - model_prob) * ask - fee
   = model_prob * 100 - ask - fee
   = edge_pct - fee_cents
```

For a NO signal with `edge_pct = ((1 - model_prob) - no_ask/100) * 100`:
```
EV = (1 - model_prob) * (100 - no_ask) - model_prob * no_ask - fee
   = (1 - model_prob) * 100 - no_ask - fee
   = edge_pct - fee_cents
```

The formula `ev_cents = edge_pct - fee_cents` works for both sides because `edge_pct` already encodes the probability-weighted advantage.

**Fee estimate (matches paper_trader._compute_fee for 1 contract):**
```python
price_decimal = market_ask_cents / 100.0
fee_cents = max(math.ceil(round(0.07 * price_decimal * (1 - price_decimal) * 100, 10)), 1)
```
The `round(..., 10)` before `ceil()` prevents floating-point artifacts (copied from existing `paper_trader._compute_fee` pattern).

**Only emit signal if `ev_cents >= min_ev_cents`.**

**Example: 3c YES, 8% edge:**
- fee = max(ceil(0.07 * 0.03 * 0.97 * 100), 1) = max(1, 1) = 1c
- EV = 8 - 1 = 7c. Passes the 2c threshold.

This means the EV filter alone won't block all penny bets — the dedup guard (Fix 2) is the primary defense. It prevents the March 13 pattern where 46 copies of the same marginally-positive trade amplified model miscalibration into a -$1.69 loss. The EV filter catches the truly worthless signals (e.g., 1c contract with 3% edge = EV of 2c, barely passes; same contract with 2% edge = EV of 1c, blocked).

---

## Files Changed

| File | Changes |
|------|---------|
| `services/strategy_engine.py` | Tuple keys, tail aggregation, dedup guard, EV filter, edge reversal tuple keys, log formatting |
| `services/circuit_breakers.py` | Optional[int] params, NULL-safe SQL, tail bracket log formatting |
| `services/paper_trader.py` | Optional[int] params, tail settlement, tail unrealized lookup |
| `tests/test_strategy_engine.py` | Tail bracket signals, dedup, EV filter tests |
| `tests/test_circuit_breakers.py` | Tail bracket circuit breaker tests |
| `tests/test_paper_trader.py` | Tail bracket settlement tests |

## What Stays the Same

- Model (`services/model.py`) — still outputs 1°F integer probs
- Dashboard (`ui/web_dashboard.py`) — already handles tail brackets
- Fee calculation — unchanged
- Market fetcher — already captures tail bracket ticks
- DB schema — no migration needed
- Interior bracket cap convention in paper_positions (cap = floor + BRACKET_WIDTH)
