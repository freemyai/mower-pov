from mower_pov.course_coverage import SweepOption, select_coverage_sequence


def test_sequence_values_area_already_available_on_later_row():
    rows = (
        (SweepOption(0b0011, 4, 1), SweepOption(0b1100, 4.7, 1)),
        (SweepOption(0b0011, 4, 1),),
    )
    assert select_coverage_sequence(rows, target=0b1111, cell_area=.05, seconds=9) == (1, 0)
    assert select_coverage_sequence(rows, target=0b1111, cell_area=.05, seconds=8) == (0, 0)


def test_idle_command_uses_time_to_stop_and_restart_before_real_cut():
    import numpy as np
    from mower_pov.course_coverage import idle_blade_commands

    times = np.arange(0., 6.01, .01)
    command = times < 5.5
    planned, windows = idle_blade_commands(times, command, np.array([.02, .82, 1., 4.45, 4.6, 5.4]))
    assert len(windows) == 1
    start, stop = windows[0]
    assert times[start] > 1.
    assert times[stop] + .65 <= 4.45
    assert not planned[start:stop].any()
    assert planned[start - 1] and planned[stop]
    assert (planned[times >= 5.5] == command[times >= 5.5]).all()


def test_actual_coast_replays_every_first_cut_and_preserves_the_trajectory():
    import numpy as np

    from mower_pov.config import load_config
    from mower_pov.contracts import StoryState
    from mower_pov.course import coast_over_cut_grass
    from mower_pov.cutting import compute_cut_state
    from mower_pov.motion import trajectory_from_path
    from mower_pov.terrain import build_terrain, generate_blade_roots

    config = load_config("configs/smoke.toml", {"world.duration_s": 7.})
    terrain = build_terrain(config)
    times = np.arange(7 * config.world.step_hz + 1) / config.world.step_hz
    progress = .15 * (np.clip(times, 0., 2.) + np.clip(times - 5., 0., 2.))
    still = np.zeros_like(times)
    rotor = np.ones_like(times)
    command = rotor.astype(bool)
    path = trajectory_from_path(config, terrain, times, np.column_stack((progress, still)),
                                still, np.gradient(progress, times), still, command, rotor)
    roots = generate_blade_roots(config, terrain)
    before = compute_cut_state(config, terrain, path, roots)
    story = StoryState(np.empty((0, 6)), np.full(len(times), "MOW"), ~command, np.full(len(times), ""),
                       np.full(len(times), np.inf), command, rotor)
    replay, updated, after, windows = coast_over_cut_grass(config, terrain, path, story, roots, before)
    assert len(windows) == 1 and windows[0]["stopped_s"] > 1.
    assert updated.rotor_fraction.min() == 0.
    assert not replay.blade_enabled[np.argmin(abs(times - 3.5))]
    np.testing.assert_array_equal(replay.position_m, path.position_m)
    np.testing.assert_array_equal(after.t_first_cut_s, before.t_first_cut_s)
    np.testing.assert_array_equal(after.root_first_cut_s, before.root_first_cut_s)
