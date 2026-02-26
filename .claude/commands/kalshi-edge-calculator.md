---
name: kalshi-edge-calculator
description: Workflow for computing and evaluating trading edge in Kalshi weather markets, comparing model probabilities to market prices, and calculating net-of-fees expected value. Use PROACTIVELY whenever Russell asks about edge, expected value, whether a trade is worth taking, fee impact, position sizing, or profit potential on a specific contract. Triggers on terms like edge, expected value, EV, worth trading, fee impact, net profit, position size, how much to bet, or market vs model.
tools: Read, Bash, Grep
model: opus
---

# Kalshi Edge Calculator

Step-by-step workflow for evaluating trading opportunities in Kalshi weather markets.

## The Edge Calculation Workflow

### Step 1: Get Model Probability

Run the probability engine for the target city and date:
```python
from services.probability import ProbabilityEngine

engine = ProbabilityEngine()
forecast = engine.calculate_city("NYC")

# Model's probability for each 1°F bracket
for temp, prob in sorted(forecast.bracket_probs.items(), key=lambda x: -x[1])[:5]:
    print(f"  {temp}°F: {prob:.1%}")
```

### Step 2: Map to Kalshi Brackets

Kalshi uses 2°F brackets. Map the model's 1°F probabilities:

```python
# Example: Kalshi bracket "76-77°F"
# Model probability = P(76°F) + P(77°F)
model_prob = forecast.bracket_probs.get(76, 0) + forecast.bracket_probs.get(77, 0)
```

For tail brackets:
- Lower tail "Below X°F": sum all P(temp) where temp < X
- Upper tail "Above X°F": sum all P(temp) where temp > X

### Step 3: Get Market Price

The market-implied probability is simply the contract price:
```
market_prob = contract_price  (e.g., $0.35 → 35% implied probability)
```

For YES contracts: `market_prob = yes_ask` (cost to buy YES)
For NO contracts: `market_prob = 1 - yes_bid` (cost to buy NO)

### Step 4: Calculate Raw Edge

```
raw_edge = model_prob - market_prob
```

- **Positive edge** → model thinks the event is MORE likely than the market
- **Negative edge** → model thinks it's LESS likely (potential NO trade)

### Step 5: Calculate Net-of-Fees Expected Value

**Fee layers (all must be applied):**

1. **Trading fee (1%):** Applied to the cost of entering the position
2. **Settlement fee (10%):** Applied to winnings only (if the contract settles in your favor)
3. **Withdrawal fee (2%):** Applied when withdrawing profits

**For a YES trade at price P with C contracts:**
```
entry_cost = C × P
trading_fee = entry_cost × 0.01
total_cost = entry_cost + trading_fee

# If contract settles YES:
gross_payout = C × $1.00
winnings = gross_payout - entry_cost
settlement_fee = winnings × 0.10
net_payout = gross_payout - settlement_fee

# Expected value:
EV = (model_prob × net_payout) - ((1 - model_prob) × total_cost) - trading_fee
```

**Simplified:**
```
net_EV_per_contract = (model_prob × (1 - P) × 0.90) - ((1 - model_prob) × P) - (P × 0.01)
```

### Step 6: Determine If Trade Is Justified

A trade is justified ONLY when:
1. `net_EV_per_contract > 0` (positive expected value after ALL fees)
2. The edge exceeds the minimum threshold (TBD — establish through backtesting)
3. Model confidence is sufficient (check forecast.std — high uncertainty = less reliable edge)

### Step 7: Position Sizing (Future)

When Kelly sizing is implemented:
```
kelly_fraction = edge / odds
position_size = bankroll × kelly_fraction × fractional_kelly_multiplier
```

Current constraint: $100 starting bankroll, paper trading only.

## Fee Impact Reference Table

| Contract Price | Trading Fee | Settlement Fee (if win) | Breakeven Edge |
|---------------|-------------|------------------------|----------------|
| $0.10 | $0.001 | $0.09 | ~2% |
| $0.25 | $0.0025 | $0.068 | ~4% |
| $0.50 | $0.005 | $0.045 | ~6% |
| $0.75 | $0.0075 | $0.023 | ~4% |
| $0.90 | $0.009 | $0.009 | ~2% |

Fees hurt most at mid-range prices (around $0.50) where both trading and settlement fees are significant.

## Red Flags — When NOT to Trade

- Model std is high (>3.0°F) — distribution is too uncertain
- Edge is below breakeven after fees — no profit potential
- Market is illiquid (wide bid-ask spread) — execution cost erodes edge
- Data is stale — check that forecasts and observations are current
- Settlement window is imminent — verify you're trading the right contract day

## Common Mistakes

1. **Ignoring settlement fees** — The 10% settlement fee on winnings is the biggest fee drag
2. **Using bid instead of ask** — You pay the ask to enter, not the bid
3. **Not accounting for withdrawal fees** — The 2% withdrawal fee applies to all profits taken out
4. **Comparing against wrong bracket** — Verify the Kalshi bracket maps correctly to your model's temperature range
5. **Trading stale model output** — Always check when the model last ran before acting on its probabilities
