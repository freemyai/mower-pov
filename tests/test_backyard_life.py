from pathlib import Path

import numpy as np

from mower_pov.cache import prepare_run
from mower_pov.config import load_config
from mower_pov.rendering.life_motion import ball_pose, dog_pose, hedgehog_pose, layout_clearance, trot_step


def test_garden_life_is_repeatable_out_of_order():
    queries = [0.77, 1.89, 4.8, 17.8, 19.12, 21.2, 30.9]
    for actor in (hedgehog_pose, dog_pose, ball_pose):
        forward = [actor(time) for time in queries]
        assert [actor(time) for time in reversed(queries)] == forward[::-1]
        assert all(np.isfinite([*pose.xy, pose.heading, pose.distance]).all() for pose in forward)


def test_all_set_dressing_stays_outside_the_mower_sweep(tmp_path: Path):
    cfg = load_config("configs/smoke.toml", {"grass.roots_per_m2": 15})
    bundle = prepare_run(cfg, tmp_path)
    results = layout_clearance(bundle)
    assert results.keys() >= {"hedgehog", "dog", "ball", "trampoline", "fence", "toys"}
    assert min(results.values()) > .09


def test_dog_follows_ball_and_comes_to_a_stop():
    for t in (18.55, 19.1, 19.75, 20.3, 21.0):
        assert ball_pose(t).xy[0] < dog_pose(t).xy[0]
        assert dog_pose(t).xy[1] >= 2.8
    assert dog_pose(23.0) == dog_pose(39.0)


def test_trot_foot_has_ground_contact_and_lifts_during_recovery():
    phase = np.linspace(0., 1., 120, endpoint=False)
    foot_x, foot_z = trot_step(phase)
    assert (foot_z[phase < .60] == 0).all()
    assert max(foot_z) > .045
    assert np.ptp(foot_x) > .20
    assert trot_step(0.) == trot_step(1.)


def test_ball_passes_clear_of_the_toys():
    from mower_pov.rendering.life_motion import BALL_RADIUS, TOY_XY, TOY_RADIUS

    ball_lane_y = ball_pose(19).xy[1]
    assert abs(ball_lane_y - TOY_XY[1]) > BALL_RADIUS + TOY_RADIUS


def test_custom_bed_does_not_intersect_the_dog_lane(tmp_path):
    from mower_pov.rendering.life_motion import layout_available

    config = load_config("configs/smoke.toml", {
        "world.flower_bed_xyxy_m": [2.5, 2.8, 3.1, 3.5], "grass.roots_per_m2": 15,
    })
    assert not layout_available(prepare_run(config, tmp_path))


def test_scene_conflict_is_reported_not_hidden(tmp_path):
    import pytest

    from mower_pov.rendering.life_motion import require_life_layout

    config = load_config("configs/smoke.toml", {
        "world.flower_bed_xyxy_m": [2.5, 2.8, 3.1, 3.5], "grass.roots_per_m2": 15,
    })
    with pytest.raises(ValueError, match="dog.*flower bed"):
        require_life_layout(prepare_run(config, tmp_path))


def test_trampoline_padding_clears_perimeter_in_the_stock_yard():
    from mower_pov.rendering.life_motion import TRAMPOLINE_RADIUS, TRAMPOLINE_XY

    cfg = load_config("configs/backyard.toml")
    inner_fence_y = cfg.world.depth_m / 2 - .04 - .041 / 2
    assert TRAMPOLINE_XY[1] + TRAMPOLINE_RADIUS + .054 < inner_fence_y
