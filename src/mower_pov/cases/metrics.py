"""Independent centred-disc sweep metrics, using only actual sampled actions."""

from dataclasses import dataclass

import numpy as np

EVALUATOR_VERSION = "1"


@dataclass(frozen=True)
class Metrics:
    evaluator_version: str
    cell_m: float
    target_area_m2: float
    target_cut_m2: float
    target_fraction_cut: float
    unique_cut_m2: float
    repeated_area_m2: float
    extra_visit_area_m2: float
    distance_m: float


def target_grid(world, radius, resolution=None):
    cell = world.cell_m if resolution is None else resolution
    origin = np.array([-world.width_m / 2, -world.depth_m / 2])
    width, depth = np.ceil(np.asarray([world.width_m, world.depth_m]) / cell).astype(int)
    x = origin[0] + (np.arange(width) + .5) * cell
    y = origin[1] + (np.arange(depth) + .5) * cell
    x0, x1 = world.x_limits_m
    dx = x - np.clip(x, x0, x1)
    dy = np.min(abs(y[:, None] - np.asarray(world.row_y_m)), axis=1)
    target = dx[None, :] ** 2 + dy[:, None] ** 2 <= radius ** 2
    mask = np.ones(target.shape, bool)
    if world.flower_bed_xyxy_m is not None:
        left, bottom, right, top = world.flower_bed_xyxy_m
        mask &= ~((x >= left) & (x <= right))[None, :] | ~((y >= bottom) & (y <= top))[:, None]
    return origin, mask, target & mask


def _disc_segment(points, start, end, radius):
    """Return contact entry fraction and end occupancy for each point."""
    delta = end - start
    rel = points - start
    length2 = np.dot(delta, delta)
    radius2 = radius ** 2
    initial = np.sum(rel * rel, axis=1) <= radius2
    if length2 == 0:
        return np.where(initial, 0., np.inf), initial
    cross = rel[:, 0] * delta[1] - rel[:, 1] * delta[0]
    advance = rel @ delta / length2
    width2 = (radius2 - cross ** 2 / length2) / length2
    entry = advance - np.sqrt(np.maximum(width2, 0))
    hits = (width2 >= 0) & (entry >= 0) & (entry <= 1)
    tail = points - end
    return np.where(initial, 0., np.where(hits, entry, np.inf)), np.sum(tail * tail, axis=1) <= radius2


def trace_sweep(case, trace, resolution=None, *, check_roots=False):
    cell = case.world.cell_m if resolution is None else resolution
    radius = case.robot.blade_radius_m
    origin, mask, target = target_grid(case.world, radius, cell)
    first = np.full(mask.shape, np.inf)
    visits = np.zeros(mask.shape, np.int32)
    distance = 0.
    root_first = None
    if trace.trajectory is not None:
        tr = trace.trajectory
        xy, t, enabled = tr.position_m[:, :2], tr.time_s, tr.blade_enabled
        distance = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        last_touch = np.full(mask.shape, -2, np.int32)
        counted = np.zeros(mask.shape, bool)
        if check_roots:
            points = trace.roots.root_xyz_m[:, :2]
            order = np.argsort(points[:, 0])
            sorted_x = points[order, 0]
            root_first = np.full(len(points), np.inf)

        for i, (start, end) in enumerate(zip(xy, np.vstack((xy[1:], xy[-1:])))):
            next_i = min(i + 1, len(t) - 1)
            if i == next_i and not enabled[i]:
                break
            low, high = np.minimum(start, end) - radius, np.maximum(start, end) + radius
            low_ij = np.floor((low - origin) / cell).astype(int)[::-1]
            high_ij = np.floor((high - origin) / cell).astype(int)[::-1] + 1
            r0, c0 = np.clip(low_ij, 0, mask.shape)
            r1, c1 = np.clip(high_ij, 0, mask.shape)
            rr, cc = np.nonzero(mask[r0:r1, c0:c1])
            rr, cc = rr + r0, cc + c0
            centres = origin + (np.column_stack((cc, rr)) + .5) * cell
            entry, tail = _disc_segment(centres, start, end, radius)
            hit = np.isfinite(entry)
            dt = t[next_i] - t[i]
            arrival = t[i] + entry[hit] * dt
            rr_hit, cc_hit = rr[hit], cc[hit]
            prior = (last_touch[rr_hit, cc_hit] == i - 1) & counted[rr_hit, cc_hit]
            active = enabled[i] & ((arrival < t[next_i]) | enabled[next_i])
            visits[rr_hit, cc_hit] += active & ~prior
            r, c = rr_hit[active], cc_hit[active]
            first[r, c] = np.minimum(first[r, c], arrival[active])
            last_touch[rr[tail], cc[tail]] = i
            counted[rr[tail], cc[tail]] = False
            counted[rr_hit[tail[hit]], cc_hit[tail[hit]]] = (prior | active)[tail[hit]]

            if root_first is not None and enabled[i]:
                a, b = np.searchsorted(sorted_x, low[0]), np.searchsorted(sorted_x, high[0], side="right")
                indices = order[a:b]
                indices = indices[(points[indices, 1] >= low[1]) & (points[indices, 1] <= high[1])
                                  & np.isposinf(root_first[indices])]
                entries, _ = _disc_segment(points[indices], start, end, radius)
                valid = np.isfinite(entries)
                times = t[i] + entries[valid] * dt
                eligible = (times < t[next_i]) | enabled[next_i]
                root_first[indices[valid][eligible]] = times[eligible]

    squares = cell ** 2
    actual = np.isfinite(first)
    cut_target = np.count_nonzero(target & actual)
    all_target = np.count_nonzero(target)
    metrics = Metrics(EVALUATOR_VERSION, cell, float(all_target * squares), float(cut_target * squares),
                      float(cut_target / all_target) if all_target else 0.,
                      float(np.count_nonzero(actual) * squares),
                      float(np.count_nonzero(visits > 1) * squares),
                      float(np.maximum(visits - 1, 0).sum() * squares), distance)
    return metrics, first, root_first


def measure_episode(case, trace):
    return trace_sweep(case, trace)[0]


def convergence_report(case, trace, resolutions_m, *, area_tolerance_m2=.01):
    """Compare rasterized areas, independently of the continuous chassis checks."""
    from dataclasses import asdict

    levels = tuple(resolutions_m)
    if (len(levels) < 2 or any(not np.isfinite(x) or not .002 <= x <= .1 for x in levels)
            or any(a <= b for a, b in zip(levels, levels[1:]))):
        raise ValueError("resolutions must decrease strictly from coarse to fine within v1 limits")
    if not np.isfinite(area_tolerance_m2) or area_tolerance_m2 <= 0:
        raise ValueError("area tolerance must be finite and positive")
    finest = levels[-1]
    if case.world.width_m * case.world.depth_m / finest**2 > 8_000_000:
        raise ValueError("fine raster exceeds eight million cells; use a local scenario")
    measured = [asdict(trace_sweep(case, trace, cell)[0]) for cell in levels]
    last, before = measured[-1], measured[-2]
    fields = ("target_area_m2", "target_cut_m2", "unique_cut_m2", "repeated_area_m2", "extra_visit_area_m2")
    change = {field:abs(last[field] - before[field]) for field in fields}
    stable = max(change.values()) <= area_tolerance_m2
    status = "STABLE" if stable else "UNSTABLE"
    if trace.trajectory is None:
        status = "NOT_EVALUATED"
    return {"status":status, "tolerance_m2":area_tolerance_m2, "levels":measured,
            "last_step_area_change_m2":change, "applies_to":"coverage raster"}
