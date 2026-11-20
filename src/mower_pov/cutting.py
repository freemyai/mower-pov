"""Continuous disc sweeps over piecewise linear cutter centers."""

import numpy as np

from .contracts import BladeRoots, CutState, RunConfig, Terrain, Trajectory
from .motion import sample_pose
from .terrain import grid_to_xy, xy_to_grid


def _first_fraction(xy, start, delta, radius):
    """First intersection of moving circle and points, in normalized segment time."""
    rel = xy - start
    radius_sq = radius ** 2
    inside = np.einsum("ij,ij->i", rel, rel) <= radius_sq
    length_sq = delta @ delta
    if length_sq == 0:
        return np.where(inside, 0., np.inf)
    # Use the perpendicular distance to avoid cancellation at grazing roots.
    perpendicular_sq = (rel[:, 0] * delta[1] - rel[:, 1] * delta[0]) ** 2 / length_sq
    spread = (radius_sq - perpendicular_sq) / length_sq
    center = (rel @ delta) / length_sq
    entry = center - np.sqrt(np.maximum(spread, 0))
    return np.where(inside, 0., np.where((spread >= 0) & (entry >= 0) & (entry <= 1),
                                         entry, np.inf))


def _segments(config, trajectory):
    offset = np.array([*config.mower.blade_xy_m, 0.])
    centers = (trajectory.position_m + trajectory.orientation @ offset)[:, :2]
    t = trajectory.time_s
    blade = trajectory.blade_enabled
    for i in np.flatnonzero(blade[:-1]):
        # Arc sagitta from a noncentral mount stays below 0.1 mm per subsegment.
        n = max(1, int(np.ceil(abs(trajectory.unwrapped_yaw[i + 1] -
                                   trajectory.unwrapped_yaw[i]) *
                               np.sqrt(np.linalg.norm(offset) / .0008))))
        if n == 1:
            yield t[i], t[i + 1], centers[i], centers[i + 1], bool(blade[i + 1])
            continue
        sub_times = np.linspace(t[i], t[i + 1], n + 1)
        start = centers[i]
        for j, stop_time in enumerate(sub_times[1:], 1):
            end = (sample_pose(trajectory, stop_time) @ [*offset, 1])[:2]
            yield sub_times[j - 1], stop_time, start, end, j < n or bool(blade[i + 1])
            start = end
    if blade[-1]:
        yield t[-1], t[-1], centers[-1], centers[-1], True


def compute_cut_state(config: RunConfig, terrain: Terrain, trajectory: Trajectory,
                      roots: BladeRoots) -> CutState:
    radius = config.mower.blade_radius_m
    grid = np.full(terrain.height_m.shape, np.inf)
    root_times = np.full(len(roots.root_id), np.inf)
    xy = roots.root_xyz_m[:, :2]
    sorted_ids = np.argsort(xy[:, 0])
    sorted_x = xy[sorted_ids, 0]
    for t0, t1, start, end, include_end in _segments(config, trajectory):
        low, high = np.minimum(start, end) - radius, np.maximum(start, end) + radius
        ij0, ij1 = xy_to_grid(terrain, [low, high])
        ij0, ij1 = np.maximum(ij0, 0), np.minimum(ij1 + 1, grid.shape)
        row0, col0 = ij0
        row1, col1 = ij1
        local = grid[row0:row1, col0:col1] if np.all(ij1 > ij0) else grid[:0, :0]
        valid = terrain.mowable_mask[row0:row1, col0:col1] if local.size else local.astype(bool)
        rr, cc = np.nonzero(np.isinf(local) & valid)
        cell_xy = grid_to_xy(terrain, np.column_stack((rr + row0, cc + col0)))
        first = _first_fraction(cell_xy, start, end - start, radius)
        hit = first <= 1 if include_end else first < 1
        local[rr[hit], cc[hit]] = t0 + first[hit] * (t1 - t0)

        a, b = np.searchsorted(sorted_x, low[0]), np.searchsorted(sorted_x, high[0], side="right")
        ids = sorted_ids[a:b]
        ids = ids[(xy[ids, 1] >= low[1]) & (xy[ids, 1] <= high[1]) & np.isinf(root_times[ids])]
        first = _first_fraction(xy[ids], start, end - start, radius)
        hit = first <= 1 if include_end else first < 1
        root_times[ids[hit]] = t0 + first[hit] * (t1 - t0)

    # Independent absolute-time wheel contact paths. Keep both passes available for replay.
    track = config.mower.track_m
    wheel_xy = (trajectory.position_m[:, None, :2]
                + trajectory.orientation[:, None, :2, 1] * np.array([track / 2, -track / 2])[None, :, None])
    contact_xy = np.stack((wheel_xy[:-1], wheel_xy[1:]), axis=2)
    return CutState(grid, config.grass.cut_height_m, root_times, xy_to_grid(terrain, xy),
                    np.column_stack((trajectory.time_s[:-1], trajectory.time_s[1:])),
                    contact_xy, .035)


def sample_grass_heights(roots: BladeRoots, state: CutState, time_s: float) -> np.ndarray:
    if not np.isfinite(time_s):
        raise ValueError("grass query time must be finite")
    return np.where(time_s >= state.root_first_cut_s, np.minimum(roots.h0_m, state.cut_height_m),
                    roots.h0_m)


def measure_sweep_error(terrain: Terrain, state: CutState, center_path, radius: float) -> float:
    """Max signed-distance error of observed grid edges against a dense reference path.

    Intended for synthetic sweep fixtures wholly inside the mowable area.
    """
    covered = np.isfinite(state.t_first_cut_s)
    across_x = np.argwhere(covered[:, 1:] != covered[:, :-1]) + [0, .5]
    across_y = np.argwhere(covered[1:, :] != covered[:-1, :]) + [.5, 0]
    points = grid_to_xy(terrain, np.vstack((across_x, across_y)))
    if not len(points):
        return np.inf
    curve = center_path(np.linspace(0, 1, 2049))
    error = 0.
    for group in np.array_split(points, max(1, (len(points) + 127) // 128)):
        distances = np.linalg.norm(group[:, None, :] - curve, axis=2)
        error = max(error, float(np.max(np.abs(np.min(distances, axis=1) - radius))))
    return error
