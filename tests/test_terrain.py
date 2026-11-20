from dataclasses import replace

import numpy as np

from mower_pov.config import load_config
from mower_pov.terrain import build_terrain, generate_blade_roots, sample_height, grid_to_xy, xy_to_grid


def test_terrain_seed_is_deterministic():
    cfg = load_config("configs/smoke.toml")
    a, b = build_terrain(cfg), build_terrain(cfg)
    np.testing.assert_array_equal(a.height_m, b.height_m)
    np.testing.assert_array_equal(generate_blade_roots(cfg, a).root_xyz_m,
                                  generate_blade_roots(cfg, b).root_xyz_m)
    assert not np.array_equal(a.height_m, build_terrain(replace(cfg, seed=1)).height_m)


def test_blade_roots_stay_on_terrain():
    cfg = load_config("configs/smoke.toml")
    terrain = build_terrain(cfg)
    roots = generate_blade_roots(cfg, terrain)
    xy = roots.root_xyz_m[:, :2]
    ij = xy_to_grid(terrain, xy)
    assert terrain.mowable_mask[ij[:, 0], ij[:, 1]].all()
    np.testing.assert_allclose(roots.root_xyz_m[:, 2], sample_height(terrain, xy))
    assert np.all((roots.h0_m >= .08) & (roots.h0_m <= .12))
    assert len(np.unique(roots.root_id)) == len(xy)


def test_world_grid_coordinate_roundtrip():
    teren = build_terrain(load_config("configs/smoke.toml"))
    row_col = np.array([[4, 10], [715, 1168]])
    np.testing.assert_array_equal(xy_to_grid(teren, grid_to_xy(teren, row_col)), row_col)
