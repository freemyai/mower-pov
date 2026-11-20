import numpy as np

from mower_pov.rendering.showcase_motion import SHOTS, jump, look


def test_showcase_covers_eighteen_seconds():
    assert sum(shot.end_frame - shot.first_frame for shot in SHOTS) == 18 * 30
    assert len({shot.name for shot in SHOTS}) == 4


def test_child_bounces_and_recovers_without_state_history():
    values = np.array([jump(t) for t in np.linspace(26., 31., 301)])
    assert values[0, 0] == values[-1, 0] == 0
    assert .15 < max(values[:, 0]) < .35
    assert (values[:, 0] >= 0).all()
    reference = jump(27.234)
    jump(30.456)
    assert jump(27.234) == reference


def test_showcase_cameras_use_distinct_grounded_compositions():
    aims = [look((shot.first_frame + shot.end_frame) / 60) for shot in SHOTS]
    assert len({pose[0] for pose in aims}) == 4
    assert all(pose[0][2] > .17 and 33 <= pose[2] <= 70 for pose in aims)
