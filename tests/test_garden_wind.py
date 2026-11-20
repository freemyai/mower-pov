import numpy as np

from mower_pov.rendering.blender_wind import breeze, foliage_positions, leaf_tilt


def test_gust_moves_through_the_lawn_at_absolute_time():
    x = np.array([0., .05, 1.5])
    y = np.zeros_like(x)
    first = breeze(18.5, x, y)
    later = breeze(19.0, x, y)
    assert np.linalg.norm(first[0] - first[1]) < np.linalg.norm(first[0] - first[2])
    assert not np.allclose(first, later)
    np.testing.assert_array_equal(first, breeze(18.5, x, y))


def test_world_wind_has_consistent_direction_across_leaf_azimuths():
    yaw = np.array([0., np.pi / 2])
    tilt = leaf_tilt(np.array([[.12, .02], [.12, .02]]), yaw)
    c, s = np.cos(yaw), np.sin(yaw)
    np.testing.assert_allclose(c * tilt[:, 1] + s * tilt[:, 0], .12)
    np.testing.assert_allclose(s * tilt[:, 1] - c * tilt[:, 0], .02)


def test_foliage_moves_above_its_anchor_and_replays_out_of_order():
    points = np.array([[.2, -.8, 0.], [.2, -.8, .4], [.2, -.8, .8]])
    flexibility = np.array([0., .5, 1.])
    baseline = foliage_positions(points, flexibility, 20.7)
    next_pose = foliage_positions(points, flexibility, 21.5)
    np.testing.assert_array_equal(baseline[0], points[0])
    assert np.linalg.norm(baseline[2] - points[2]) > np.linalg.norm(baseline[1] - points[1])
    assert not np.allclose(next_pose, baseline)
    np.testing.assert_array_equal(foliage_positions(points, flexibility, 20.7), baseline)
