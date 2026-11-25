import pytest

from mower_pov.cases.render_failure import failure_window, window_frames


def test_failure_at_start_and_end_uses_absolute_time():
    assert failure_window(0., 10.) == (0., 2.)
    assert failure_window(10., 10.) == (8., 10.)
    assert failure_window(3.5, 10.) == (1.5, 5.5)
    assert window_frames((1.5, 5.5), 30, 300)[0] == 45
    assert window_frames((1.5, 5.5), 30, 300)[-1] == 164


def test_subframe_failure_at_zero_can_be_rendered():
    assert window_frames(failure_window(0., .01), 30, 1) == (0,)


def test_reject_negative_or_outside_failure_time():
    with pytest.raises(ValueError, match="outside"):
        failure_window(11., 10.)
