import numpy as np
import pytest

from mower_pov.actors.behavior import DOG_TRACK, dog_pose
from mower_pov.actors.gait import DOG_LEGS, evaluate_dog
from mower_pov.config import load_config
from mower_pov.terrain import build_terrain, sample_height


@pytest.fixture(scope="module")
def terrain():
    return build_terrain(load_config("configs/backyard.toml"))


@pytest.mark.parametrize("name,start", [("front_right", 18.0), ("front_right", 20.2), ("front_right", 21.39)])
def test_world_paw_contact_is_fixed_for_point_one_second(terrain, name, start):
    poses = [evaluate_dog(t, terrain) for t in np.linspace(start, start + .10, 21)]
    legs = [pose.legs[name] for pose in poses]
    assert all(leg.planted for leg in legs)
    points = np.asarray([leg.foot_world for leg in legs])
    assert np.linalg.norm(points - points[0], axis=1).max() < .002
    np.testing.assert_allclose(points[:, 2], sample_height(terrain, points[:, :2]), atol=.005)


def test_dog_joint_lengths_and_time_order(terrain):
    times = np.linspace(17.4, 23.2, 151)
    poses = [evaluate_dog(t, terrain) for t in times]
    assert poses == [evaluate_dog(t, terrain) for t in times[::-1]][::-1]
    for pose in poses:
        for spec in DOG_LEGS:
            leg = pose.legs[spec.name]
            a, b, c = map(np.asarray, (leg.hip, leg.knee, leg.ankle))
            assert np.linalg.norm(b - a) == pytest.approx(spec.upper, abs=1e-10)
            assert np.linalg.norm(c - b) == pytest.approx(spec.lower, abs=1e-10)
    assert all(leg.planted for leg in poses[-1].legs.values())
    assert dog_pose(25).distance == pytest.approx(DOG_TRACK.length)
    assert any(not leg.planted for pose in poses for leg in pose.legs.values())
