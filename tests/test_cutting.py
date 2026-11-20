from dataclasses import replace

import numpy as np
import pytest

from mower_pov.config import load_config
from mower_pov.contracts import BladeRoots, Trajectory
from mower_pov.cutting import compute_cut_state, measure_sweep_error, sample_grass_heights
from mower_pov.terrain import build_terrain


def fixture(xy=((.5, 0), (.5, .16), (.6, 0)), h=(.10, .10, .03), cells=.01):
    cfg = load_config("configs/backyard.toml", {"world.cell_m": cells})
    terreno = build_terrain(cfg)
    h = np.asarray(h)
    zeros = np.zeros(len(h))
    roots = BladeRoots(np.arange(len(h)), np.column_stack([xy, zeros]), h,
                      *[zeros] * 6)
    return cfg, terreno, roots


def trajectory(xy, times, enabled=None, yaw=None):
    n = len(times)
    zeros = np.zeros(n)
    return Trajectory(np.asarray(times), np.column_stack([xy, zeros]),
                      np.repeat(np.eye(3)[None, ...], n, axis=0),
                      zeros if yaw is None else np.asarray(yaw),
                      zeros, zeros, zeros, zeros, np.zeros((n, 2)),
                      np.ones(n, dtype=bool) if enabled is None else np.asarray(enabled))


def test_first_cut_time_matches_straight_sweep():
    cfg, terreno, roots = fixture()
    state = compute_cut_state(cfg, terreno, trajectory([(0, 0), (1, 0)], [0., 1.]), roots)
    assert state.root_first_cut_s[0] == pytest.approx(.36, abs=1 / 240)
    assert measure_sweep_error(terreno, state, lambda x: np.c_[x, np.zeros_like(x)],
                               cfg.mower.blade_radius_m) <= .02


def test_outside_or_blade_off_never_cuts():
    cfg, terreno, roots = fixture()
    traj = trajectory([(0, 0), (1, 0)], [0., 1.])
    assert np.isposinf(compute_cut_state(cfg, terreno, traj, roots).root_first_cut_s[1])
    off = compute_cut_state(cfg, terreno, replace(traj, blade_enabled=np.array([False, False])), roots)
    assert np.isposinf(off.root_first_cut_s).all()
    np.testing.assert_array_equal(sample_grass_heights(roots, off, 1), roots.h0_m)
    assert off.wheel_contact_count == 2


def test_camera_offset_is_not_cutter_offset():
    cfg, terreno, roots = fixture()
    traj = trajectory([(0, 0), (1, 0)], [0., 1.])
    moved = replace(cfg, camera=replace(cfg.camera, mount_xyz_m=(.36, 0., .22)))
    np.testing.assert_array_equal(compute_cut_state(cfg, terreno, traj, roots).root_first_cut_s,
                                  compute_cut_state(moved, terreno, traj, roots).root_first_cut_s)


def test_no_future_cut_leak():
    cfg, terreno, roots = fixture()
    traj = trajectory([(0, 0), (1, 0)], [4.64, 5.64])
    state = compute_cut_state(cfg, terreno, traj, roots)
    assert sample_grass_heights(roots, state, 4.9)[0] == .10
    assert sample_grass_heights(roots, state, 5.001)[0] == .045


def test_repeat_pass_preserves_short_grass():
    cfg, terreno, roots = fixture()
    traj = trajectory([(0, 0), (1, 0), (0, 0)], [0., 1., 2.])
    state = compute_cut_state(cfg, terreno, traj, roots)
    np.testing.assert_allclose(sample_grass_heights(roots, state, 2), [.045, .10, .03])
    np.testing.assert_array_equal(sample_grass_heights(roots, state, 2),
                                  sample_grass_heights(roots, state, 1))


def test_curved_sweep_converges():
    errors = []
    for n, cell in [(40, .01), (80, .005)]:
        cfg, terreno, roots = fixture(cells=cell)
        f = lambda time: np.c_[np.sin(time * np.pi / 2), 1 - np.cos(time * np.pi / 2)]
        times = np.linspace(0, 1, n)
        state = compute_cut_state(cfg, terreno, trajectory(f(times), times), roots)
        errors.append(measure_sweep_error(terreno, state, f, .14))
    assert errors[1] <= errors[0] <= .02


def test_offset_cut_sweep_accounts_for_pitch():
    from mower_pov.motion import sample_pose

    cfg, terrain, roots = fixture(xy=((.60, 0),), h=(.10,))
    cfg = replace(cfg, mower=replace(cfg.mower, blade_xy_m=(.5, 0.)))
    traj = trajectory([(0, 0), (0, 0)], [0., 1.])
    traj = replace(traj, pitch_rad=np.deg2rad([-60., 60.]))
    orientation = np.stack([sample_pose(traj, t)[:3, :3] for t in traj.time_s])
    traj = replace(traj, orientation=orientation)
    state = compute_cut_state(cfg, terrain, traj, roots)
    assert state.root_first_cut_s[0] == pytest.approx(
        .5 - np.arccos((.60 - .14) / .5) / np.deg2rad(120), abs=.001
    )
