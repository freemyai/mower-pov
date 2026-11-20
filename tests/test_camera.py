import numpy as np

from mower_pov.camera import camera_mount, camera_pose
from mower_pov.config import load_config
from mower_pov.motion import integrate_motion, make_demo_schedule, sample_pose
from mower_pov.terrain import build_terrain


def test_camera_mount_is_constant():
    cfg = load_config("configs/backyard.toml")
    terrain = build_terrain(cfg)
    trajectory = integrate_motion(cfg, terrain, make_demo_schedule(cfg, terrain))
    for t in np.random.default_rng(1).uniform(0, 40, 20):
        body = sample_pose(trajectory, t)
        np.testing.assert_allclose(np.linalg.inv(body) @ camera_pose(cfg, body),
                                   camera_mount(cfg), atol=1e-6)
    np.testing.assert_allclose(camera_mount(cfg)[:3, 3], [.18, 0, .22])
    assert camera_mount(cfg)[2, 2] < 0
