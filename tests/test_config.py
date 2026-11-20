import pytest

from mower_pov.config import load_config


def test_default_config_frame_count():
    config = load_config("configs/backyard.toml")
    assert config.frame_count == 1200
    assert config.audio_frames == 1_920_000


@pytest.mark.parametrize(("field", "value"), [
    ("grass.min_height_m", -1), ("render.fps", 0), ("mower.blade_radius_m", -1),
    ("render.width", 641), ("camera.pitch_deg", float("nan")),
    ("world.width_m", float("inf")), ("render.fps", True),
])
def test_invalid_dimensions_or_rates_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        load_config("configs/backyard.toml", {field: value})
