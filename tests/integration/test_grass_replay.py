import numpy as np

from mower_pov.cache import prepare_run
from mower_pov.camera import camera_pose
from mower_pov.config import load_config
from mower_pov.motion import sample_pose
from mower_pov.rendering.blender_camera import blender_camera_matrix
from mower_pov.rendering.blender_grass import blade_length, grass_time_s, wind_angle


def test_camera_axis_conversion():
    cfg = load_config("configs/smoke.toml")
    optical = camera_pose(cfg, np.eye(4))
    blender = blender_camera_matrix(optical)
    projected = np.linalg.inv(blender) @ optical @ np.array(
        [[0., .4, 0.], [0., 0., -.4], [1., 1., 1.], [1., 1., 1.]]
    )
    np.testing.assert_allclose(projected[2], -1.)
    assert projected[0, 1] > 0 and projected[1, 2] > 0


def test_grass_height_and_identity_replay(tmp_path):
    cfg = load_config("configs/smoke.toml", {"world.duration_s": 4.})
    bundle = prepare_run(cfg, tmp_path)
    roots, state = bundle.roots, bundle.cut_state
    ids, xyz = roots.root_id.copy(), roots.root_xyz_m.copy()
    i = np.flatnonzero(np.isfinite(state.root_first_cut_s))[0]
    t = state.root_first_cut_s[i]
    assert blade_length(roots.h0_m[i], t, t - .001, state.cut_height_m) == roots.h0_m[i]
    assert blade_length(roots.h0_m[i], t, t + .001, state.cut_height_m) == state.cut_height_m
    np.testing.assert_array_equal(roots.root_xyz_m, xyz)
    np.testing.assert_array_equal(roots.root_id, ids)


def test_subframe_grass_and_wind_evaluation():
    a, b = grass_time_s(120.25, 30), grass_time_s(120.75, 30)
    t_cut = 120.5 / 30
    assert blade_length(.1, t_cut, a, .045) == .1
    assert blade_length(.1, t_cut, b, .045) == .045
    assert wind_angle(a, .3, .8) != wind_angle(b, .3, .8)
    assert wind_angle(a, .3, .8) == wind_angle(grass_time_s(120.25, 30), .3, .8)


def test_clip_uses_absolute_timeline(tmp_path):
    cfg = load_config("configs/backyard.toml", {"grass.roots_per_m2": 140})
    bundle = prepare_run(cfg, tmp_path)
    t = grass_time_s(735, cfg.render.fps)
    i = np.flatnonzero(bundle.cut_state.root_first_cut_s < 10)
    assert i.size > 0 and t == 24.5
    assert (blade_length(bundle.roots.h0_m[i], bundle.cut_state.root_first_cut_s[i],
                         t, bundle.cut_state.cut_height_m) == .045).all()
    assert sample_pose(bundle.trajectory, t)[0, 0] < -.99
