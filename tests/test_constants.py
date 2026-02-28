# tests/test_constants.py
from core.constants import (
    generate_magnets,
    get_all_station_ids,
    MAGNETS,
    FLB_FAVORITE_FLOOR,
    FLB_LONGSHOT_CEILING,
    CITIES,
    POLL_INTERVAL_SECONDS,
    STATION_COORDS,
    FORECAST_POLL_INTERVAL_SECONDS,
)


def test_generate_magnets_returns_set():
    result = generate_magnets(30, 110)
    assert isinstance(result, set)
    assert len(result) > 0


def test_known_magnets_present():
    """40, 49, 51, 60 are known magnet numbers in the 9°F repeating pattern."""
    result = generate_magnets(30, 110)
    for known in [40, 49, 51, 60]:
        assert known in result, f"{known} should be a magnet"


def test_magnet_captures_six_tenths():
    """Each magnet integer should map to 6+ Celsius tenths when rounded."""
    result = generate_magnets(30, 110)
    for mag in result:
        count = 0
        for c_int in range(max(-400, int((mag - 35) / 1.8 * 10)), int((mag - 28) / 1.8 * 10)):
            c_val = c_int / 10.0
            f_val = round(c_val * 9.0 / 5.0 + 32.0)
            if f_val == mag:
                count += 1
        assert count >= 6, f"Magnet {mag} only captures {count} tenths"


def test_non_magnets_capture_five_or_fewer():
    """Non-magnet integers should capture 5 or fewer Celsius tenths."""
    magnets = generate_magnets(30, 110)
    for f_int in range(30, 111):
        if f_int not in magnets:
            count = 0
            for c_int in range(int((f_int - 35) / 1.8 * 10), int((f_int - 28) / 1.8 * 10)):
                c_val = c_int / 10.0
                f_val = round(c_val * 9.0 / 5.0 + 32.0)
                if f_val == f_int:
                    count += 1
            assert count <= 5, f"Non-magnet {f_int} captures {count} tenths"


def test_flb_thresholds():
    assert FLB_FAVORITE_FLOOR == 0.50
    assert FLB_LONGSHOT_CEILING == 0.15


def test_cities_config():
    assert "NYC" in CITIES
    assert CITIES["NYC"]["settlement"] == "KNYC"
    assert "KLGA" in CITIES["NYC"]["neighbors"]
    assert len(CITIES) == 1


def test_all_stations_list():
    """All NYC stations should be extractable from CITIES config."""
    all_stations = get_all_station_ids()
    assert isinstance(all_stations, list)
    assert len(all_stations) == 4  # KNYC + KLGA + KEWR + KJFK


def test_poll_interval():
    assert POLL_INTERVAL_SECONDS == 60


def test_station_coords_exist_for_all_settlements():
    """Every settlement station must have coordinates."""
    for city_cfg in CITIES.values():
        stid = city_cfg["settlement"]
        assert stid in STATION_COORDS, f"Missing coords for {stid}"
        lat, lon = STATION_COORDS[stid]
        assert -90 <= lat <= 90, f"Invalid lat for {stid}"
        assert -180 <= lon <= 180, f"Invalid lon for {stid}"


def test_station_coords_settlement_and_kjfk():
    """Coords should exist for settlement station and KJFK."""
    assert len(STATION_COORDS) == 2
    assert "KNYC" in STATION_COORDS
    assert "KJFK" in STATION_COORDS


def test_forecast_poll_interval():
    assert FORECAST_POLL_INTERVAL_SECONDS == 900
