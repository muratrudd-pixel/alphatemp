"""AlphaTemp core constants — magnet numbers, FLB thresholds, station config."""

from typing import Set


def generate_magnets(f_min: int = 0, f_max: int = 130) -> Set[int]:
    """Generate magnet Fahrenheit integers.

    ASOS sensors report in 0.1°C. When converting each tenth to °F and rounding,
    certain integers capture 6 tenths instead of 5. These repeat every 9°F.
    """
    counts: dict = {}
    c_min_raw = (f_min - 32) * 5.0 / 9.0
    c_max_raw = (f_max - 32) * 5.0 / 9.0
    c_start = int(c_min_raw * 10) - 10
    c_end = int(c_max_raw * 10) + 10

    for c_tenth in range(c_start, c_end + 1):
        c_val = c_tenth / 10.0
        f_rounded = round(c_val * 9.0 / 5.0 + 32.0)
        if f_min <= f_rounded <= f_max:
            counts[f_rounded] = counts.get(f_rounded, 0) + 1

    return {f_int for f_int, count in counts.items() if count >= 6}


# Pre-computed magnets for the typical settlement range
MAGNETS: Set[int] = generate_magnets(0, 130)

# Favorite-Longshot Bias thresholds
FLB_FAVORITE_FLOOR: float = 0.50
FLB_LONGSHOT_CEILING: float = 0.15

# Station configuration: settlement station + neighbors
CITIES: dict = {
    "NYC": {"settlement": "KNYC", "neighbors": ["KLGA", "KEWR"]},
    "PHL": {"settlement": "KPHL", "neighbors": ["KPNE"]},
    "CHI": {"settlement": "KMDW", "neighbors": ["KORD"]},
    "MIA": {"settlement": "KMIA", "neighbors": ["KOPF"]},
    "LA":  {"settlement": "KLAX", "neighbors": ["KHHR"]},
}

# Polling interval in seconds
POLL_INTERVAL_SECONDS: int = 60

# IEM ASOS polling interval (matches Synoptic — 60s poll with 90-min lookback)
IEM_POLL_INTERVAL_SECONDS: int = 60

# Forecast polling interval (15 minutes — HRRR updates hourly, this catches trickle-in)
FORECAST_POLL_INTERVAL_SECONDS: int = 900

# NWS daily high poll interval (4 hours — CLI reports publish once daily)
NWS_DAILY_POLL_INTERVAL_SECONDS: int = 14400

# NWS CLI direct poll interval (30 min — same-day settlement data)
NWS_CLI_POLL_INTERVAL_SECONDS: int = 1800

# Settlement station coordinates for HRRR grid point extraction
STATION_COORDS: dict = {
    "KNYC": (40.7789, -73.9692),
    "KPHL": (39.8721, -75.2411),
    "KMDW": (41.7868, -87.7522),
    "KMIA": (25.7959, -80.2870),
    "KLAX": (33.9425, -118.4081),
}


def get_all_station_ids() -> list:
    """Return flat list of all 11 station ICAO codes."""
    stations = []
    for city in CITIES.values():
        stations.append(city["settlement"])
        stations.extend(city["neighbors"])
    return stations
