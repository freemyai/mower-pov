from dataclasses import replace

import numpy as np
import pytest

from mower_pov.config import load_config
from mower_pov.contracts import ControlSchedule
from mower_pov.motion import (integrate_motion, make_demo_schedule, sample_pose,
                              validate_swept_body, wheel_speeds)
from mower_pov.terrain import build_terrain


def test_straight_drive_and_differential_turn():
    cfg = load_config("configs/smoke.toml")
    terrain = build_terrain(cfg)
    controls = ControlSchedule(np.array([0.]), np.array([.2]), np.array([0.]), np.array([True]))
    traj = integrate_motion(cfg, terrain, controls)
    np.testing.assert_allclose(traj.position_m[:, 1], 0)
    np.testing.assert_allclose(wheel_speeds(.20, .50, .32), [.12, .28])


def test_limits_and_heading_unwrap():
    cfg = load_config("configs/smoke.toml")
    terrain = build_terrain(cfg)
    controls = ControlSchedule(np.array([0.]), np.array([2.]), np.array([2.]), np.array([True]))
    traj = integrate_motion(cfg, terrain, controls)
    dt = 1 / cfg.world.step_hz
    assert max(abs(traj.v_mps)) <= cfg.mower.max_speed_mps
    assert max(abs(traj.omega_radps)) <= cfg.mower.max_omega_radps
    assert max(abs(np.diff(traj.v_mps))) <= dt * cfg.mower.max_accel_mps2 + 1e-12
    assert max(abs(np.diff(traj.omega_radps))) <= dt * cfg.mower.max_alpha_radps2 + 1e-12
    assert traj.unwrapped_yaw[-1] > np.pi
    idx = np.searchsorted(traj.unwrapped_yaw, np.pi)
    np.testing.assert_allclose(sample_pose(traj, traj.time_s[idx - 1])[:3, :3],
                               sample_pose(traj, traj.time_s[idx])[:3, :3], atol=.004)


def test_swept_body_rejects_thin_obstacle():
    cfg = load_config("configs/smoke.toml")
    terrain = build_terrain(cfg)
    demo = integrate_motion(cfg, terrain, make_demo_schedule(cfg, terrain))
    traj = replace(demo, time_s=np.array([0., 1.]),
                   position_m=np.array([[0., 0., 0.], [2., 0., 0.]]))
    obstacle = replace(terrain, obstacle_xyxy_m=(.999, .1, 1.001, .11))
    assert validate_swept_body(cfg, obstacle, traj)


def test_demo_returns_along_neighboring_row():
    cfg = load_config("configs/backyard.toml")
    terrain = build_terrain(cfg)
    assert cfg.world.duration_s == 40
    traj = integrate_motion(cfg, terrain, make_demo_schedule(cfg, terrain))
    assert validate_swept_body(cfg, terrain, traj) == []
    assert sample_pose(traj, 24)[1, 3] == pytest.approx(cfg.mower.row_spacing_m, abs=.001)
    assert sample_pose(traj, 24)[0, 0] < -.99
