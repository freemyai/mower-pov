"""Check the archived demo's analytic cutting fixture with the public state API."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from mower_pov.camera import camera_pose
from mower_pov.config import load_config
from mower_pov.contracts import BladeRoots, Trajectory
from mower_pov.cutting import compute_cut_state, sample_grass_heights
from mower_pov.terrain import build_terrain


def test_demo_straight_cut_and_return():
    repo = Path(__file__).resolve().parents[1]
    fixture = json.loads((repo / "evidence/v0.1/golden-straight-cut.json").read_text())
    config = load_config(repo / fixture["config"])
    assert config.mower.blade_radius_m == fixture["blade_radius_m"]
    assert config.grass.cut_height_m == fixture["cut_height_m"]
    terrain = build_terrain(config)

    times = np.array(fixture["time_s"])
    xy = np.array(fixture["centre_xy_m"])
    zeros = np.zeros(len(times))
    orientations = np.repeat(np.eye(3)[None, ...], len(times), axis=0)
    trajectory = Trajectory(
        times, np.column_stack((xy, zeros)), orientations, np.array(fixture["yaw_rad"]),
        zeros, zeros, zeros, zeros, np.zeros((len(times), 2)), np.ones(len(times), dtype=bool),
    )
    heights = np.array(fixture["initial_height_m"])
    root_zeros = np.zeros(len(heights))
    roots = BladeRoots(
        np.arange(len(heights)), np.column_stack((fixture["root_xy_m"], root_zeros)),
        heights, *([root_zeros] * 6),
    )
    state = compute_cut_state(config, terrain, trajectory, roots)
    first_cut = [np.inf if time is None else time for time in fixture["expected_first_cut_s"]]
    np.testing.assert_allclose(state.root_first_cut_s, first_cut, rtol=0, atol=1e-12)

    shifted = replace(config, camera=replace(
        config.camera, mount_xyz_m=tuple(fixture["alternate_camera_mount_xyz_m"]),
    ))
    assert not np.array_equal(camera_pose(config, np.eye(4)), camera_pose(shifted, np.eye(4)))
    shifted_state = compute_cut_state(shifted, terrain, trajectory, roots)
    np.testing.assert_array_equal(shifted_state.root_first_cut_s, state.root_first_cut_s)
    np.testing.assert_array_equal(shifted_state.t_first_cut_s, state.t_first_cut_s)

    for sample in fixture["height_samples"]:
        actual = sample_grass_heights(roots, state, sample["time_s"])
        np.testing.assert_allclose(actual, sample["height_m"], rtol=0, atol=1e-12)
