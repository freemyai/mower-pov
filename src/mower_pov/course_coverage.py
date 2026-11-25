"""Quantized blade footprints and whole-route selection for the backyard course."""

from dataclasses import dataclass
from functools import partial
import heapq
from itertools import combinations

import numpy as np

from .controller import SPIN_DOWN_S, SPIN_UP_S
from .course_layout import COURSE_ROW_Y, COURSE_X_LIMITS
from .obstacles import distance_to_segment
from .terrain import grid_to_xy


@dataclass(frozen=True)
class SweepOption:
    mask: int
    duration: float
    distance: float


class SweepGrid:
    """Small local grid for planning. Playback still uses the exact cutter sweep."""

    def __init__(self, config, layout):
        self.cell_m = .025
        self.radius = config.mower.blade_radius_m
        self.blade_offset = config.mower.blade_xy_m
        margin = .87 + self.radius + np.linalg.norm(self.blade_offset)
        self.origin = np.array([layout.x_limits_m[0] - margin, min(layout.row_y_m) - margin])
        self.width = int(np.ceil((layout.x_limits_m[1] + margin - self.origin[0]) / self.cell_m))
        self.height = int(np.ceil((max(layout.row_y_m) + margin - self.origin[1]) / self.cell_m))
        columns = self.origin[0] + (np.arange(self.width) + .5) * self.cell_m
        turn_band = .4 * config.mower.length_m
        interior = (columns > layout.x_limits_m[0] + turn_band) & (columns < layout.x_limits_m[1] - turn_band)
        self.interior = int.from_bytes(np.packbits(np.tile(interior, self.height), bitorder="little").tobytes(),
                                      "little")
        radius_cells = int(np.ceil(self.radius / self.cell_m)) + 1
        offsets = np.arange(-radius_cells, radius_cells + 1)
        self.dx, self.dy = np.meshgrid(offsets, offsets)
        self.dx, self.dy = self.dx.ravel(), self.dy.ravel()
        x = np.linspace(*layout.x_limits_m, 256)
        self.target = 0
        for y in layout.row_y_m:
            self.target |= self.footprint(np.column_stack((x, x * 0 + y)), np.zeros(len(x)), blade_offset=False)

    def footprint(self, xy, yaw, *, blade_offset=True):
        # The planning point spacing stays below half a cell, including in bends.
        if blade_offset and self.blade_offset != (0., 0.):
            x0, y0 = self.blade_offset
            c, s = np.cos(yaw), np.sin(yaw)
            xy = xy + np.column_stack((c * x0 - s * y0, s * x0 + c * y0))
        if len(xy) > 1:
            chord = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            samples = np.linspace(0., chord[-1], 1 + int(np.ceil(chord[-1] / (.48 * self.cell_m))))
            xy = np.column_stack([np.interp(samples, chord, axis) for axis in xy.T])
        pos = (xy - self.origin) / self.cell_m - .5
        ix = np.floor(pos[:, :1]).astype(int) + self.dx
        iy = np.floor(pos[:, 1:]).astype(int) + self.dy
        in_disc = (pos[:, :1] - ix) ** 2 + (pos[:, 1:] - iy) ** 2 <= (self.radius / self.cell_m) ** 2
        valid = in_disc & (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)
        result = np.zeros(self.width * self.height, dtype=bool)
        result[iy[valid] * self.width + ix[valid]] = True
        return int.from_bytes(np.packbits(result, bitorder="little").tobytes(), "little")


def _sequence_score(mask, repeat, count, duration, distance, *, target, cell_area, repeat_zone):
    unique = mask.bit_count()
    target_cells = (mask & target).bit_count()
    # A continuous approach and turn can overlap at the shared endpoint. The
    # inner area better estimates a full revisit from another pass.
    repeated_inside = (repeat & repeat_zone).bit_count()
    area_score = .40 * unique + 1.15 * target_cells - .20 * (count - unique) - 1.05 * repeated_inside
    return cell_area * area_score - .004 * duration - .005 * distance


def _combine(options, fixed_sweeps):
    mask = repeat = count = 0
    duration = distance = 0.
    for sweep in fixed_sweeps:
        repeat |= mask & sweep
        mask |= sweep
        count += sweep.bit_count()
    for option in options:
        repeat |= mask & option.mask
        mask |= option.mask
        count += option.mask.bit_count()
        duration += option.duration
        distance += option.distance
    return mask, repeat, count, duration, distance


def _choice_portfolio(row, context, score, cap=110):
    if len(row) <= cap:
        return range(len(row))
    mask, repeat, count, duration, distance = context

    def merit(index):
        c = row[index]
        return score(mask | c.mask, repeat | (mask & c.mask), count + c.mask.bit_count(),
                     duration + c.duration, distance + c.distance)

    ranked = heapq.nlargest(cap - 24, range(len(row)), key=merit)
    fast = heapq.nsmallest(24, range(len(row)), key=lambda i: row[i].duration)
    return tuple(dict.fromkeys((*ranked, *fast)))


def _polish_pairs(rows, choice, fixed, score, budget):
    """Revisit coordinated detours that were pruned while scoring a partial route."""
    best = list(choice)
    pairs = sorted(combinations(range(len(rows)), 2), key=lambda pair: (pair[1] - pair[0], pair))
    for left, right in pairs:
        context = _combine((row[j] for i, (row, j) in enumerate(zip(rows, best))
                            if i not in (left, right)), fixed)
        mask, repeat, count, time, distance = context
        current = _combine((row[j] for row, j in zip(rows, best)), fixed)
        best_value = score(*current)
        left_choices = _choice_portfolio(rows[left], context, score)
        right_choices = _choice_portfolio(rows[right], context, score)
        for i in left_choices:
            a = rows[left][i]
            covered = mask | a.mask
            visited_again = repeat | (mask & a.mask)
            copies = count + a.mask.bit_count()
            for j in right_choices:
                b = rows[right][j]
                duration = time + a.duration + b.duration
                if duration > budget + 1e-9:
                    continue
                value = score(covered | b.mask, visited_again | (covered & b.mask),
                              copies + b.mask.bit_count(), duration, distance + a.distance + b.distance)
                if value > best_value + 1e-12:
                    best_value, best[left], best[right] = value, i, j
    return tuple(best)


def select_coverage_sequence(rows, target, cell_area, seconds, width=96, fixed_sweeps=(), repeat_zone=-1):
    """Choose all rows together, favoring new target cells over repeated work."""
    fastest = [min(option.duration for option in row) for row in rows]
    score = partial(_sequence_score, target=target, cell_area=cell_area, repeat_zone=repeat_zone)
    possible = [0] * (len(rows) + 1)
    for i in range(len(rows) - 1, -1, -1):
        possible[i] = possible[i + 1]
        for option in rows[i]:
            possible[i] |= option.mask

    # Score units are square metres of new target grass.
    def alternatives(states, row, step):
        unfillable = target & ~possible[step + 1]
        budget = seconds - sum(fastest[step + 1:])
        for _, indices, seen, repeat, count, duration, distance in states:
            for j, candidate in enumerate(row):
                total_time = duration + candidate.duration
                if total_time > budget + 1e-9:
                    continue
                covered = seen | candidate.mask
                visited_again = repeat | (seen & candidate.mask)
                copies = count + candidate.mask.bit_count()
                total_distance = distance + candidate.distance
                committed = (covered & unfillable).bit_count()
                # The lookahead gives priority to cells that no remaining row can supply.
                value = score(covered, visited_again, copies, total_time, total_distance)
                value += .12 * cell_area * committed
                yield -value, indices + (j,), covered, visited_again, copies, total_time, total_distance

    beam = [(0., (), *_combine((), fixed_sweeps))]
    for step, row in enumerate(rows):
        beam = heapq.nsmallest(width, alternatives(beam, row, step))
        if not beam:
            return None
    finalists = [_polish_pairs(rows, state[1], fixed_sweeps, score, seconds) for state in beam[:3]]
    return max(finalists, key=lambda indices: score(*_combine(
        (row[j] for row, j in zip(rows, indices)), fixed_sweeps)))


def idle_blade_commands(times, blade_command, first_cut):
    """Allow a stopped rotor only in a proven gap between first cuts."""
    cut_times = np.unique(first_cut[np.isfinite(first_cut)])
    command = blade_command.copy()
    windows = []
    dt = times[1] - times[0]
    minimum = SPIN_DOWN_S + SPIN_UP_S + .25 + 4 * dt
    for i in np.flatnonzero(np.diff(cut_times) >= minimum):
        first = int(np.searchsorted(times, cut_times[i], side="right"))
        restart = int(np.searchsorted(times, cut_times[i + 1] - SPIN_UP_S - 3 * dt)) - 1
        if command[first:restart + 1].all():
            command[first:restart] = False
            windows.append((first, restart))
    return command, windows


def nominal_cut_fraction(bundle):
    """Measure on the actual terrain grid using the fixed, unobstructed target."""
    terrain = bundle.terrain
    radius = bundle.resolved_config.mower.blade_radius_m
    rc = np.argwhere(terrain.mowable_mask)
    xy = grid_to_xy(terrain, rc)
    x0, x1 = COURSE_X_LIMITS
    in_course = (xy[:, 0] >= x0 - radius) & (xy[:, 0] <= x1 + radius)
    in_course &= (xy[:, 1] >= min(COURSE_ROW_Y) - radius) & (xy[:, 1] <= max(COURSE_ROW_Y) + radius)
    rc, xy = rc[in_course], xy[in_course]
    target = np.zeros(len(xy), dtype=bool)
    for y in COURSE_ROW_Y:
        target |= distance_to_segment(xy, (x0, y), (x1, y)) <= radius
    return float(np.isfinite(bundle.cut_state.t_first_cut_s[tuple(rc[target].T)]).mean())
