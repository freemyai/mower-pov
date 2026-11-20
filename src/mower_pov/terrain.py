"""Seeded terrain and persistent, independently seeded grass geometry attributes."""

import numpy as np

from .contracts import BladeRoots, RunConfig, Terrain


def grid_to_xy(terrain, row_col):
    return terrain.origin_xy_m + (np.asarray(row_col)[..., ::-1] + .5) * terrain.cell_size_m


def xy_to_grid(terrain, xy_m):
    return np.floor((np.asarray(xy_m) - terrain.origin_xy_m) / terrain.cell_size_m
                    ).astype(np.int64)[..., ::-1]


def build_terrain(config: RunConfig) -> Terrain:
    world = config.world
    origin = np.array([-world.width_m / 2, -world.depth_m / 2])
    nx, ny = np.ceil(np.array([world.width_m, world.depth_m]) / world.cell_m).astype(int)
    x = origin[0] + (np.arange(nx)[None, :] + .5) * world.cell_m
    y = origin[1] + (np.arange(ny)[:, None] + .5) * world.cell_m
    phases = np.random.default_rng([config.seed, 1]).uniform(0, 2 * np.pi, 3)
    height = (.032 * np.sin(1.6 * x + .55 * y + phases[0])
              + .017 * np.sin(2.1 * y - .47 * x + phases[1])
              + .006 * np.sin(5.3 * x + 2.4 * y + phases[2]))
    x0, y0, x1, y1 = world.flower_bed_xyxy_m
    mask = ~((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1))
    return Terrain(height, origin, world.cell_m, mask, world.flower_bed_xyxy_m)


def sample_height(terrain: Terrain, xy_m: np.ndarray) -> np.ndarray:
    """Bilinear interpolation with constant extension at the outermost centers."""
    grid = terrain.height_m
    coords = (np.asarray(xy_m) - terrain.origin_xy_m) / terrain.cell_size_m - .5
    x, y = np.moveaxis(np.clip(coords, [0, 0], np.array(grid.shape[::-1]) - 1), -1, 0)
    i, j = np.floor(x).astype(int), np.floor(y).astype(int)
    i1, j1 = np.minimum(i + 1, grid.shape[1] - 1), np.minimum(j + 1, grid.shape[0] - 1)
    u, v = x - i, y - j
    return (1 - v) * ((1 - u) * grid[j, i] + u * grid[j, i1]) + v * (
        (1 - u) * grid[j1, i] + u * grid[j1, i1])


def generate_blade_roots(config: RunConfig, terrain: Terrain) -> BladeRoots:
    rng = np.random.default_rng([config.seed, 2])
    w, g = config.world, config.grass
    count = round(w.width_m * w.depth_m * g.roots_per_m2)
    # Candidate indices precede masking so the flower-bed boundary preserves IDs.
    xy = terrain.origin_xy_m + rng.random((count, 2)) * [w.width_m, w.depth_m]
    row, col = xy_to_grid(terrain, xy).T
    x0, y0, x1, y1 = terrain.obstacle_xyxy_m
    x, y = xy.T
    in_bed = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    ids = np.flatnonzero(terrain.mowable_mask[row, col] & ~in_bed)
    params = rng.random((count, 7))[ids]
    height, width, bend, phase, azimuth, tint, dry_tip = params.T
    xy = xy[ids]
    return BladeRoots(ids, np.column_stack([xy, sample_height(terrain, xy)]),
                      g.min_height_m + height * (g.max_height_m - g.min_height_m),
                      .0018 + width * .0032, .12 + bend * .63, phase * 2 * np.pi,
                      azimuth * 2 * np.pi, tint, dry_tip)
