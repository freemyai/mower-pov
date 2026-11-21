from dataclasses import replace

import numpy as np
import pytest

from mower_pov.cache import load_bundle, prepare_run, save_bundle
from mower_pov.config import load_config
from mower_pov.course_layout import COURSE_OBSTACLES, COURSE_ROW_Y, COURSE_X_LIMITS, obstacle_clearance
from mower_pov.course import plan_course
from mower_pov.obstacles import distance_to_segment


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    config = load_config("configs/mowing-course.toml", {"world.cell_m": .035, "grass.roots_per_m2": 350})
    return prepare_run(config, tmp_path_factory.mktemp("course"))


def test_obstacles_actually_block_nominal_mowing(work):
    x = np.linspace(*COURSE_X_LIMITS, 160)
    obstructed = set()
    for y in COURSE_ROW_Y:
        xy = np.column_stack((x, np.full_like(x, y)))
        for obstacle in COURSE_OBSTACLES:
            if obstacle_clearance(xy, np.zeros_like(x), work.resolved_config, obstacle).min() < 0:
                obstructed.add(obstacle.name)
    assert obstructed == {obstacle.name for obstacle in COURSE_OBSTACLES}


def test_course_bypasses_each_obstacle_and_preserves_clearance(work):
    fsm = work.story
    bypassed = set(fsm.obstacle_name[fsm.mode == "BYPASS"])
    for obstacle in COURSE_OBSTACLES:
        assert obstacle.name in bypassed
        assert ((fsm.mode == "REPLAN") & (fsm.obstacle_name == obstacle.name)).any()
        assert ((fsm.mode == "REJOIN") & (fsm.obstacle_name == obstacle.name)).any()
    assert np.min(fsm.clearance_m) > .027


def test_course_turns_four_times_and_finishes_at_rest(work):
    traj = work.trajectory
    turning = work.story.mode == "TURN"
    starts = np.flatnonzero(turning & ~np.r_[False, turning[:-1]])
    stops = np.flatnonzero(~turning & np.r_[False, turning[:-1]])
    assert len(starts) == len(stops) == 4
    assert np.allclose(abs(traj.unwrapped_yaw[stops] - traj.unwrapped_yaw[starts]), np.pi, atol=.012)
    assert traj.time_s[-1] > 60
    assert traj.v_mps[-1] == work.story.rotor_fraction[-1] == 0
    for i in stops:
        xy = traj.position_m[i, :2]
        assert np.min(abs(xy[1] - COURSE_ROW_Y)) < .005
        assert np.all(np.isfinite(traj.position_m[i]))


def test_kinematics_and_coverage_across_turns(work):
    traj = work.trajectory
    m = work.resolved_config.mower
    dt = np.diff(traj.time_s)
    assert np.max(abs(traj.v_mps)) <= m.max_speed_mps + 1e-7
    assert np.max(abs(traj.omega_radps)) <= m.max_omega_radps + 1e-7
    assert np.max(abs(np.diff(traj.v_mps)) / dt) <= m.max_accel_mps2 + .012
    assert np.max(abs(np.diff(traj.omega_radps)) / dt) <= m.max_alpha_radps2 + 1e-5
    roots = work.roots.root_xyz_m[:, :2]
    first = work.cut_state.root_first_cut_s
    assert np.isfinite(first).sum() > 900
    # Each row receives cuts in its own time interval, with all earlier cuts retained.
    turning = work.story.mode == "TURN"
    stops = np.flatnonzero(~turning & np.r_[False, turning[:-1]])
    assert all(np.sum(first < traj.time_s[i]) > 100 for i in stops)
    for obstacle in COURSE_OBSTACLES:
        points = obstacle.points
        spans = list(zip(points, points[1:])) or [(points[0], points[0])]
        distances = np.minimum.reduce([distance_to_segment(roots, a, b) for a, b in spans])
        assert not np.isfinite(first[distances < obstacle.radius_m]).any()


def test_saved_course_replays_without_replanning(work):
    replay = load_bundle(save_bundle(work))
    assert replay.run_id == work.run_id
    for name in ("mode", "obstacle_name", "clearance_m", "rotor_fraction"):
        np.testing.assert_array_equal(getattr(replay.story, name), getattr(work.story, name))
    assert replay.manifest["identity"]["course_layout"]


def test_rejects_unreachable_time_budget(work):
    world = replace(work.resolved_config.world, duration_s=25.)
    config = replace(work.resolved_config, world=world)
    with pytest.raises(ValueError, match="needs .* seconds"):
        plan_course(config)
