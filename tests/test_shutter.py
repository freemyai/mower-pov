import pytest

from mower_pov.rendering.shutter import Exposure


def test_box_exposure_samples_are_centered_in_absolute_frames():
    exposure = Exposure(8.0, 3)
    assert exposure.sample_frames(607, 30, 40) == pytest.approx((606.92, 607, 607.08))
    assert exposure.sample_frames(120, 30, 40) == pytest.approx((119.92, 120, 120.08))
    assert exposure.sample_frames(607, 30, 40) == pytest.approx((606.92, 607, 607.08))


def test_box_exposure_samples_are_clipped_to_the_recorded_timeline():
    exposure = Exposure(20.0, 3)
    assert exposure.sample_frames(0, 30, 40) == pytest.approx((.05, .15, .25))
    assert exposure.sample_frames(1199, 30, 39.97) == pytest.approx(
        (1198.7666666667, 1198.9, 1199.0333333333)
    )


def test_instant_exposure_replays_the_exact_absolute_frame():
    assert Exposure(0.0, 3).sample_frames(607, 30, 40) == (607,)
    assert Exposure(0.0, 1).sample_frames(607, 30, 40) == (607,)


@pytest.mark.parametrize("milliseconds,samples", [
    (-1, 3), (float("nan"), 3), (float("inf"), 3), (8.0, 1), (8.0, 4), (8.0, 3.0),
])
def test_invalid_exposures_are_rejected(milliseconds, samples):
    with pytest.raises(ValueError):
        Exposure(milliseconds, samples)


def test_exposure_cannot_exceed_the_frame_period():
    with pytest.raises(ValueError, match="frame period"):
        Exposure(40.0, 3).sample_frames(100, 30, 40)
