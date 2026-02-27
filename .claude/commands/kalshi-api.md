---
name: kalshi-api
description: Kalshi API integration patterns, gotchas, and conventions for alphatemp. Use PROACTIVELY when writing or modifying code that calls the Kalshi API, parsing tickers, handling market data, building new endpoints, or debugging API failures. NOT for trading strategy (use alphatemp-trading or kalshi-edge-calculator). Triggers on Kalshi API, ticker parsing, rate limit, pagination, cursor, candlestick, trade history, settlement fetch, exchange.py, market_fetcher, backfill_kalshi, RSA signing, API auth, 429 error, or ticker format.
tools: Read, Write, Edit, Bash, Grep
model: opus
---

# Kalshi API Integration Patterns

How to correctly interact with the Kalshi API in alphatemp. This skill covers the API layer — for trading strategy, see `alphatemp-trading` and `kalshi-edge-calculator`.

## Ticker Format — CRITICAL

Kalshi tickers encode dates as **YYMMMDD**, NOT DDMMMYY.

```
KXHIGHNY-26FEB25 → February 25, 2026
         ^^   ^^
         YY   DD

# Parsing regex
r"-(\d{2})([A-Z]{3})(\d{2})"
# group(1) = year (26 → 2026)
# group(2) = month abbreviation (FEB)
# group(3) = day (25)
```

This has caused a production bug before. The year comes FIRST, the day comes LAST. Always verify parsed dates fall in the expected range (2021–2026).

**Month map:** `{"JAN":1, "FEB":2, "MAR":3, "APR":4, "MAY":5, "JUN":6, "JUL":7, "AUG":8, "SEP":9, "OCT":10, "NOV":11, "DEC":12}`

## Authentication

RSA-PSS signing with SHA-256. The signing payload is `timestamp_ms + method + path` (path excludes query params).

```python
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# Sign: timestamp_str + method + path
signature = private_key.sign(
    message.encode(),
    padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.MAX_LENGTH,
    ),
    hashes.SHA256(),
)

headers = {
    "KALSHI-ACCESS-KEY": api_key,
    "KALSHI-ACCESS-TIMESTAMP": str(int(time.time() * 1000)),
    "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
}
```

**Private key in .env:** Stored as single line with `\\n` literal. Must replace before loading:
```python
api_secret = os.getenv("KALSHI_API_SECRET").replace("\\n", "\n")
```

**Validate on startup:** Always call `get_exchange_status()` immediately after client init to verify credentials work.

## Base URLs

```python
PROD = "https://api.elections.kalshi.com/trade-api/v2"
DEMO = "https://demo-api.kalshi.co/trade-api/v2"
```

## API Endpoints Reference

| Method | Path | Purpose | Key Params |
|--------|------|---------|------------|
| `search_markets` | `/markets` | Find open markets | `series_ticker`, `status`, `limit` |
| `get_market` | `/markets/{ticker}` | Single market details | — |
| `get_orderbook` | `/markets/{ticker}/orderbook` | Bid/ask depth | `depth` (default 5) |
| `get_settled_markets` | `/markets` | Settled markets (paginated) | `series_ticker`, `status="settled"`, `cursor`, `limit` |
| `get_candlesticks` | `/series/{series}/markets/{ticker}/candlesticks` | OHLCV data | `start_ts`, `end_ts`, `period_interval` |
| `get_trades` | `/markets/trades` | Trade history (paginated) | `ticker`, `cursor`, `limit` (max 1000) |
| `place_order` | `/portfolio/orders` | Place trade | **BLOCKED** by `TRADING_ENABLED = False` |

## Price Units

Prices come from the API in **cents (integer 0–99)**. Convert to probability:

```python
yes_bid_prob = yes_bid_cents / 100.0  # e.g., 35 → 0.35
```

- `market_ticks` table stores prices as **decimals (0.0–1.0)**
- `kalshi_trades` table stores prices as **integer cents**
- Always check which convention you're reading from

## Pagination Pattern

All paginated endpoints return `{"data": [...], "cursor": "..."}`.

```python
def fetch_all(client, endpoint_fn, **kwargs):
    all_items = []
    cursor = None

    while True:
        data = api_call_with_retry(endpoint_fn, cursor=cursor, **kwargs)
        items = data.get("markets", []) or data.get("trades", [])
        all_items.extend(items)

        cursor = data.get("cursor")
        if not cursor or not items:
            break

        time.sleep(REQUEST_DELAY)

    return all_items
```

**Rules:**
- Empty cursor OR empty items list = end of data
- Always `time.sleep(REQUEST_DELAY)` between pages
- Kalshi's `limit` max is 1000 per page for trades

## Rate Limiting

Kalshi returns HTTP 429 when rate limited.

```python
REQUEST_DELAY = 0.5   # Between successful requests
BACKOFF_BASE = 2.0    # Exponential: 4s, 8s, 16s, 32s, 64s
MAX_RETRIES = 5

def api_call_with_retry(fn, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code == 429:
                wait = BACKOFF_BASE ** (attempt + 1)
                logger.warning(f"Rate limited, backing off {wait:.0f}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Max retries exceeded on rate limit")
```

## Timestamp Handling

API returns ISO 8601 with Z suffix. Parse with:

```python
created_time = datetime.fromisoformat(
    created_time_str.replace("Z", "+00:00")
)
```

All timestamps are UTC. Store as naive UTC in DuckDB (strip tzinfo before insert).

## Candlestick Response Format

The response format has varied — handle both nested and flat structures:

```python
# Nested format
yes_bid_close = candle.get("yes_bid", {}).get("close")
price_open = candle.get("price", {}).get("open")

# Flat format (fallback)
if yes_bid_close is None:
    yes_bid_close = candle.get("yes_bid_close")
if price_open is None:
    price_open = candle.get("price_open")
```

**Time window for candlestick fetch:** 2 days before event_date through close_time.
```python
start_ts = int(datetime.combine(event_date, datetime.min.time(),
    tzinfo=timezone.utc).timestamp()) - 172800  # 2 days = 172800 seconds
```

## Settlement Result Mapping

```python
result_str = market.get("result", "")
settled_yes = 1 if result_str == "yes" else (0 if result_str == "no" else None)
```

Always handle the `None` case — some markets may have unexpected result values.

## Series Map

```python
SERIES_MAP = {
    "NYC": "KXHIGHNY",
    # Future expansion: "NYC_LOW": "KXLOWTNYC", etc.
}
```

## Table Schemas (Quick Reference)

| Table | UNIQUE Key | Price Unit |
|-------|-----------|------------|
| `kalshi_settlements` | `market_ticker` | N/A |
| `kalshi_candlesticks` | `(market_ticker, end_period_ts)` | Decimal (0–100) |
| `kalshi_trades` | `trade_id` | Integer cents |
| `market_ticks` | None (append-only) | Decimal (0.0–1.0) |

## The Three-Phase Backfill

`scripts/backfill_kalshi_history.py` follows a strict A → B → C order:

**Phase A: Settlements** — Fetch all settled markets, parse tickers, store bracket outcomes. Must run first (B and C depend on the settlement list).

**Phase B: Candlesticks** — For each settled market, fetch 1-minute OHLCV data. Supports `--refetch` to re-pull with wider windows. Supports `source_db` for temp-DB pattern.

**Phase C: Trades** — For each settled market, paginate through all trade history. Supports `source_db` for temp-DB pattern.

```bash
python scripts/backfill_kalshi_history.py                # All phases
python scripts/backfill_kalshi_history.py --settlements   # Phase A only
python scripts/backfill_kalshi_history.py --candlesticks  # Phase B only
python scripts/backfill_kalshi_history.py --trades        # Phase C only
```

## Kill Switch

```python
TRADING_ENABLED = False  # Top of exchange.py
```

Any `place_order()` call is logged and returns `{"blocked": True}` when disabled. Do NOT change this without explicit approval from Russell.

## Common Mistakes

1. **YYMMMDD not DDMMMYY** — Parse ticker dates year-first
2. **Cents vs decimals** — Check which table you're reading from
3. **Missing close_time** — Fall back to `event_date + 3 days` for candlestick window
4. **Cursor check** — Must check BOTH `not cursor` AND `not items` to detect end of pagination
5. **Z suffix** — Always `.replace("Z", "+00:00")` before `fromisoformat()`
6. **Private key newlines** — `.replace("\\n", "\n")` when loading from .env
