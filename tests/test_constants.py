# tests/test_constants.py
from core.constants import (
    generate_magnets,
    get_all_station_ids,
    MAGNETS,
    FLB_FAVORITE_FLOOR,
    FLB_LONGSHOT_CEILING,
    CITIES,
    POLL_INTERVAL_SECONDS,
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
    assert "CHI" in CITIES
    assert CITIES["CHI"]["settlement"] == "KMDW"
    assert "KORD" in CITIES["CHI"]["neighbors"]
    assert len(CITIES) == 5


def test_all_stations_list():
    """All 11 stations should be extractable from CITIES config."""
    all_stations = get_all_station_ids()
    assert isinstance(all_stations, list)
    assert len(all_stations) == 11


def test_poll_interval():
    assert POLL_INTERVAL_SECONDS == 60
