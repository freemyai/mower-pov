import numpy as np

from mower_pov.cases.contracts import ObstacleSpec
from mower_pov.cases.geometry import swept_obstacle_clearance


def test_corner_hits_between_safe_endpoint_poses():
    post = ObstacleSpec("corner-post", "disc", ((.32, 0.),), .012)
    bound = swept_obstacle_clearance(np.zeros((2, 2)), np.array([-.72, .72]),
                                     (.50, .42), post)
    assert bound[0] < 0


def test_thin_hose_intersects_translation_between_safe_samples():
    slit = ObstacleSpec("thin-hose", "polyline_sweep", ((-.05, -.5), (.05, .5)), .007)
    bound = swept_obstacle_clearance(np.array([[-.6, 0.], [.6, 0.]]), np.zeros(2),
                                     (.50, .42), slit)
    assert bound[0] < 0


def test_stationary_disc_distance_is_analytic():
    post = ObstacleSpec("post", "disc", ((.45, 0.),), .05)
    bound = swept_obstacle_clearance(np.zeros((2, 2)), np.zeros(2), (.50, .42), post)
    np.testing.assert_allclose(bound, [.15], rtol=0, atol=1e-14)
