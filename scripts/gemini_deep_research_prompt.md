# Deep Research: Public Data Sources for Kalshi NYC Temperature Trading

## Context

I'm building a model to predict daily high temperature settlement brackets on Kalshi (KXHIGHNY markets) for New York City. The settlement station is KNYC (Central Park). Brackets are 2°F wide (e.g., 30-31°F, 32-33°F). Markets trade from midnight to settlement (~midnight ET the following day).

My model currently uses:
- **HRRR** (High-Resolution Rapid Refresh): 3km resolution NWP model, runs hourly, ~2h publication lag. This is my primary forecast input.
- **GFS** (Global Forecast System): 0.25° resolution, runs 4x/day (00z/06z/12z/18z), ~3.5-4h publication lag.
- **ECMWF IFS**: 0.25° resolution, runs 4x/day, ~7-8h publication lag. Only 00z data currently ingested.
- **KNYC hourly observations**: METAR reports at :51 past each hour. No sub-hourly temperature data available from this station.
- **KLGA/KEWR observations**: 5-minute rows exist but temperature is only populated at :51 hourly (same as KNYC).

## The Problem

When I cross-reference minute-level Kalshi market price movements against public data releases (HRRR availability + KNYC observations), I find that **only 31% of material market price moves (>5 cent jumps on any bracket) occur within 5 minutes of a data release**. The other 69% of moves happen with no new HRRR run or KNYC observation.

### Specific Example — January 9, 2025

- Actual high: 33°F (winning bracket: 32-33°F)
- Every HRRR run all day forecast a max of 28-30°F (3°F cold bias)
- At 09:19 ET, the market jumped the 32-33 bracket to 56 cents (56% probability), even though KNYC was only 24°F and the latest HRRR (12z, available ~09:00) said 30.1°F max
- Our model at noon was at 30% on the winning bracket while the market was at 65%
- The market was clearly incorporating information our model doesn't have

### Minute-Level Timelines — Market Moves vs Data Releases

For each day below, I've tagged every material market move (>5c on any bracket) with the most recent public data release and how many minutes ago it occurred. Moves marked `**` happened within 5 minutes of a data release. Moves without `**` had no recent data trigger — these are the ones I need to explain.

**HRRR availability assumption**: run_hour + 2 hours (e.g., HRRR 12z available ~14:00 UTC = 09:00 ET).
**KNYC observations**: arrive at :51 past each hour.

#### January 9, 2025 — Actual High: 33°F, Winning Bracket: 32-33°F

HRRR forecast high all day: 28-30°F (consistently 3°F too cold).

```
Time ET   Bracket  Move    Mid   Vol   Last data release
08:01     32-33   +12c    39c   304   HRRR 11z — 30.2F (1m ago) **
08:01     >33     -14c    46c   425   HRRR 11z — 30.2F (1m ago) **
08:07     32-33    -7c    32c    70   HRRR 11z — 30.2F (7m ago)
08:49     32-33    +8c    42c     0   HRRR 11z — 30.2F (49m ago)
09:19     32-33   +14c    56c   178   HRRR 12z — 30.1F (19m ago)       ← KEY MYSTERY
09:19     >33     -22c    28c   275   HRRR 12z — 30.1F (19m ago)
09:55     32-33   -13c    42c     0   KNYC 26F (4m ago) **
10:39     32-33    +7c    59c   219   HRRR 13z — 30.1F (39m ago)
11:56     30-31    +5c    26c   686   KNYC 29F (5m ago) **
12:08     30-31   +10c    43c   222   HRRR 15z — 30.3F (8m ago)
12:51     32-33    +6c    57c   101   KNYC 31F (0m ago) **
12:57     30-31   -11c    32c  1212   KNYC 31F (6m ago)
13:14     32-33   +14c    82c    40   HRRR 16z — 30.0F (14m ago)
13:23     32-33   +20c    88c    57   HRRR 16z — 30.0F (23m ago)
13:29     32-33   -16c    70c    45   HRRR 16z — 30.0F (29m ago)
15:26     >33      -9c    11c   354   HRRR 18z — 30.2F (26m ago)
```

Key mystery: At 09:19 ET, KNYC was only 24°F and the latest HRRR said 30.1°F max. Yet the market jumped 32-33 to 56c. What public information was available at 09:19 ET on Jan 9, 2025 that could explain this?

#### January 16, 2025 — Actual High: 30°F, Winning Bracket: 30-31°F (B30.5)

HRRR forecast high: 29-32°F (varied through day).

```
Time ET   Bracket  Move    Mid   Vol   Last data release
09:10     B30.5    -5c    40c   378   HRRR 12z — 32.0F (10m ago)
10:28     B30.5   -16c    28c   216   HRRR 13z — 31.1F (28m ago)
10:28     B32.5    +5c    60c    16   HRRR 13z — 31.1F (28m ago)
10:55     B30.5    +6c    38c     0   KNYC 28F (4m ago) **
11:56     B30.5    +8c    48c   109   KNYC 27F (5m ago) **
12:19     B30.5    +5c    52c   104   HRRR 15z — 30.6F (19m ago)
13:04     B30.5    +6c    55c     0   HRRR 16z — 30.1F (4m ago) **
13:43     B30.5    +8c    64c   137   HRRR 16z — 30.1F (43m ago)
13:53     B30.5    +8c    72c    42   KNYC 29F (2m ago) **
13:58     B30.5   +14c    88c   452   KNYC 29F (7m ago)
14:03     B30.5   -10c    74c    49   HRRR 17z — 29.4F (3m ago) **
14:41     B30.5   +10c    82c   141   HRRR 17z — 29.4F (41m ago)
```

Note: At 10:28 ET, the market shifted AWAY from the winning bracket (B30.5 dropped 16c) toward B32.5, despite HRRR 13z forecasting 31.1°F. Something caused the market to temporarily favor a higher bracket — then corrected later.

#### January 24, 2025 — Actual High: 33°F, Winning Bracket: 33-34°F (B33.5)

HRRR forecast high: 32.5-33.2°F (quite accurate this day).

```
Time ET   Bracket  Move    Mid   Vol   Last data release
08:35     B31.5    +9c    64c  1624   HRRR 11z — 33.2F (35m ago)
08:39     B31.5   +13c    70c   594   HRRR 11z — 33.2F (39m ago)
08:40     B31.5   -14c    57c   250   HRRR 11z — 33.2F (40m ago)
10:51     B31.5   +13c    70c    71   KNYC 29F (0m ago) **
11:07     B31.5    +6c    66c  1254   HRRR 14z — 32.5F (7m ago)
11:47     B31.5    +8c    64c   422   HRRR 14z — 32.5F (47m ago)
12:50     B33.5   +23c    60c   458   HRRR 15z — 32.5F (50m ago)       ← KEY MYSTERY
13:04     B33.5   +12c    60c   282   HRRR 16z — 33.2F (4m ago) **
14:56     B31.5   +34c    64c   944   KNYC 32F (5m ago) **
15:20     B33.5   -21c    20c   441   HRRR 18z — 33.2F (20m ago)
15:20     B31.5   +20c    76c   718   HRRR 18z — 33.2F (20m ago)
15:30     B33.5   +16c    51c   126   HRRR 18z — 33.2F (30m ago)
16:26     B33.5   +68c    92c  1087   HRRR 19z — 32.8F (26m ago)       ← MASSIVE MOVE
16:26     B31.5   -70c     6c  2065   HRRR 19z — 32.8F (26m ago)
```

Key mysteries:
- 12:50 ET: B33.5 jumps +23c on 458 contracts. Last HRRR was 50 minutes ago, last obs 59 minutes ago. What triggered this?
- 16:26 ET: +68c / -70c swing on massive volume. KNYC last reported 32°F at 15:51, temperature dropping. Market suddenly locked in 33-34 as the winner. Did the NWS CLI preliminary report or some other settlement signal come out?

### What I've Validated

1. **14-day trailing HRRR bias correction** improves MAE by 13% vs raw HRRR. HRRR errors have autocorrelation r=0.34 at lag 1 day, persisting through 2 weeks. The market may be doing this intuitively.
2. **Neighbor stations (KLGA, KEWR)** do NOT lead KNYC — they're 0.6-1.1°F warmer on average but don't predict the KNYC daily high any better than KNYC's own observations.
3. The pattern holds across multiple days (Jan 9, Jan 16, Jan 24) — consistently ~30% of moves near data, ~70% with no obvious public trigger.

## Research Questions

1. **What publicly available NWP products could weather-savvy Kalshi traders be using that I'm not?** Specifically interested in:
   - **NBM (National Blend of Models)**: How is it accessed? What's the publication schedule and latency? How does it compare to raw HRRR for daily high temperature forecasts at a single point? Is historical data available for backtesting?
   - **HRRR-based MOS or statistical post-processing products**: Are there bias-corrected HRRR products published by NOAA or available on AWS/Google Cloud?
   - **NAM (North American Mesoscale)**: Does it provide meaningfully different signal from HRRR for NYC daily highs?
   - **RAP (Rapid Refresh)**: HRRR's parent model — does it offer anything HRRR doesn't for this use case?
   - **HREF/SREF/GEFS ensemble products**: Are ensemble spread or probability products available that would give uncertainty estimates useful for bracket probability?

2. **What observation data sources provide sub-hourly NYC temperature data?**
   - **NYC Micronet / NYS Mesonet**: 29 stations, reportedly 5-minute frequency. Is this data publicly accessible in real-time? How do I access it? Does it include Central Park or nearby stations?
   - **ASOS 1-minute or 5-minute data**: Do KJFK, KLGA, or KEWR publish 1-minute or 5-minute temperature data that's accessible in real-time (not just archived)? My database shows 5-minute rows for KLGA/KEWR but temperature is NULL except at :51.
   - **MADIS (Meteorological Assimilation Data Ingest System)**: Does this provide denser observation coverage for NYC?
   - **Central Park automated sensors**: Is there any public feed with sub-hourly Central Park temperature beyond the hourly METAR?

3. **What NWS text products might traders be reading?**
   - **Area Forecast Discussion (AFD)**: How often is this updated for NYC (OKX office)? Could traders be extracting forecast high temperatures from the AFD text before the NBM updates?
   - **NWS point forecast for Central Park**: Is this updated more frequently than 2x/day? What's the data source for the point forecast?
   - **Spot forecasts or SPC mesoscale discussions**: Anything that would update intra-day?

4. **What commercial or semi-public weather data could be in play?**
   - **Weather Underground PWS (Personal Weather Stations)**: Are there stations near Central Park reporting every few minutes?
   - **Tomorrow.io, ClimaCell, or other commercial APIs**: Do they offer real-time bias-corrected NWP that could give traders an edge?
   - **AccuWeather MinuteCast or similar**: Any product that gives minute-level or 15-minute temperature nowcasts?

5. **For the specific pattern where the market moves without any new NWP or observation data**, what are the most likely explanations?
   - Is there a public data product I'm not considering that updates between HRRR runs?
   - Are traders likely using ensemble-derived probability distributions rather than deterministic forecasts?
   - Could traders be using their own statistical post-processing (similar to our trailing bias correction) to adjust raw NWP in real-time?

## What I Need From This Research

For each data source identified:
- **Access method**: API URL, AWS bucket, or download mechanism
- **Update frequency and latency**: How often does it publish and how quickly after the model run?
- **Historical availability**: Can I backtest against it? How far back does it go?
- **Relevance to NYC daily high temperature**: Is this actually useful for my specific use case?
- **Real-time accessibility**: Can I ingest this programmatically for live trading?

I'm specifically NOT interested in:
- Paid proprietary weather data that costs >$100/month
- Academic research datasets not available in real-time
- Global/hemispheric products that don't resolve NYC
- Precipitation or wind-only products (I need temperature)
