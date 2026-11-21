from dataclasses import replace

import numpy as np
import pytest

from mower_pov.actors.behavior import actor_pose, story_track
from mower_pov.cache import load_bundle, prepare_run, save_bundle
from mower_pov.config import load_config
from mower_pov.cutting import compute_cut_state
from mower_pov.obstacles import validate_story_layout
from mower_pov.terrain import build_terrain


def dynamic_config():
    return load_config("configs/backyard-dynamic.toml", {"world.cell_m": .04, "grass.roots_per_m2": 100})


@pytest.fixture(scope="module")
def story(tmp_path_factory):
    return prepare_run(dynamic_config(), tmp_path_factory.mktemp("story"))


def test_dog_crossing_causes_brake_and_wait(story):
    times = story.trajectory.time_s
    window = (times > 4) & (times < 20)
    modes = story.story.mode[window]
    assert {"MOW", "BRAKE", "WAIT", "CLEAR_CONFIRM", "RESUME"} <= set(modes)
    assert np.all(story.trajectory.v_mps[window & (story.story.mode == "WAIT")] == 0)
    assert max(story.story.clearance_m[window]) > .15
    assert min(story.story.clearance_m) > .04


def test_stationary_hedgehog_keeps_mower_waiting(tmp_path):
    config = replace(dynamic_config(), scenario="hedgehog-stays")
    bundle = prepare_run(config, tmp_path)
    wait = bundle.trajectory.time_s > 32
    assert np.all(bundle.story.mode[wait] == "WAIT")
    assert np.all(bundle.trajectory.v_mps[wait] == 0)
    assert actor_pose(bundle, "hedgehog", 39).xy == actor_pose(bundle, "hedgehog", 32).xy


def test_clearance_confirmation_is_not_timer_only(story):
    indices = np.flatnonzero(story.story.mode == "RESUME")
    entries = [i for i in indices if story.story.mode[i - 1] == "CLEAR_CONFIRM"]
    assert len(entries) == 2
    for i in entries:
        assert not story.story.blocked[i]
        window = story.trajectory.time_s >= story.trajectory.time_s[i] - .63
        window[i:] = False
        assert window.any()
        assert np.all(story.story.mode[window] == "CLEAR_CONFIRM")
        assert not story.story.blocked[window].any()


def test_scene_conflict_is_reported_not_hidden(story):
    with pytest.raises(ValueError, match="trampoline"):
        validate_story_layout(story.resolved_config, story.terrain,
                              np.array([[3.5, 3.0], [3.5, 3.7]]))


def test_cut_history_matches_replanned_trajectory(tmp_path):
    config = replace(dynamic_config(), scenario="toy-detour")
    bundle = prepare_run(config, tmp_path)
    assert "REPLAN" in bundle.story.mode
    assert np.max(bundle.trajectory.position_m[:, 1]) > 1.15
    reproduced = compute_cut_state(config, bundle.terrain, bundle.trajectory, bundle.roots)
    np.testing.assert_array_equal(bundle.cut_state.root_first_cut_s, reproduced.root_first_cut_s)
    assert not np.any(np.isfinite(bundle.cut_state.root_first_cut_s)
                      & (np.linalg.norm(bundle.roots.root_xyz_m[:, :2] - (.9, .60), axis=1) < .12))


def test_blade_spin_down_is_recorded(story):
    command = story.story.blade_command
    stops, = np.where(command[:-1] & ~command[1:])
    assert len(stops) >= 2
    i = stops[0] + 1
    assert story.story.rotor_fraction[i] > .9
    assert story.trajectory.blade_enabled[i]
    assert 0 < story.story.rotor_fraction[i + 20] < story.story.rotor_fraction[i]
    assert story.story.rotor_fraction[i + 205] == 0


def test_trampoline_lower_bar_collision(story):
    with pytest.raises(ValueError, match="trampoline lower"):
        validate_story_layout(story.resolved_config, story.terrain,
                              np.array([[3.55, 3.2], [3.55, 3.66]]))


def test_actor_and_controller_are_immutable_random_access(story):
    bundle = load_bundle(save_bundle(story))
    assert bundle.run_id == story.run_id
    np.testing.assert_array_equal(bundle.story.rotor_fraction, story.story.rotor_fraction)
    assert not bundle.story.mode.flags.writeable
    for time in (29.013, 12.173, 25.2, 0.0):
        assert actor_pose(bundle, "dog", time) == story_track(story, "dog").evaluate(time)


def test_kinematic_limits(story):
    trajectory = story.trajectory
    dt = np.diff(trajectory.time_s)
    mower = story.resolved_config.mower
    assert np.max(np.abs(np.diff(trajectory.v_mps)) / dt) <= mower.max_accel_mps2 + 1e-8
    assert np.max(np.abs(np.diff(trajectory.omega_radps)) / dt) <= mower.max_alpha_radps2 + 1e-5
    assert np.max(np.abs(trajectory.omega_radps)) <= mower.max_omega_radps


def test_return_camera_faces_the_previously_cut_swath(story):
    motion = story.trajectory
    turn = story.story.mode == "TURN"
    assert np.sum(turn) / story.resolved_config.world.step_hz > 4.
    assert np.all(motion.v_mps[turn] == 0)
    assert motion.unwrapped_yaw[-1] == pytest.approx(-np.pi, abs=.01)
    assert motion.position_m[-1, 0] < motion.position_m[turn][0, 0] - .40
    roots = story.roots.root_xyz_m[:, :2]
    prior = (roots[:, 0] < motion.position_m[-1, 0] - .25)
    prior &= (roots[:, 0] > motion.position_m[-1, 0] - 1.4) & (abs(roots[:, 1] - 2.86) < .10)
    assert np.count_nonzero(prior) > 10
    assert np.all(story.cut_state.root_first_cut_s[prior] < motion.time_s[turn][0])
