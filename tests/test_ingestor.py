from services.ingestor import parse_t_group


def test_parse_positive_temp():
    assert parse_t_group("T02280167") == 22.8


def test_parse_negative_temp():
    assert parse_t_group("T10051012") == -0.5


def test_parse_zero():
    assert parse_t_group("T00000000") == 0.0


def test_parse_no_t_group():
    assert parse_t_group("RMK AO2 SLP135") is None


def test_parse_embedded_in_metar():
    metar = "RMK AO2 SLP135 T02280167 10272 20228 53012"
    assert parse_t_group(metar) == 22.8


def test_parse_negative_freezing():
    assert parse_t_group("T10171028") == -1.7


def test_parse_hot_day():
    assert parse_t_group("T03890350") == 38.9
